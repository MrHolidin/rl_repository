"""Distributed PPO trainer — workers collect rollouts (CPU), host runs PPO (GPU).

Worker protocol
---------------
Host → Worker:
  ("load", sd_bytes)                    update agent weights; frozen pool unchanged
  ("add_to_pool", slot_id, sd_bytes)    add frozen snapshot with given slot identifier
  ("set_weights", {slot_id: win_rate})  update PFSP opponent sampling weights (raw EMA rates)
  ("play", round_idx)                   collect games; reply with rollout + outcomes
  ("stop",)                             terminate

Worker → Host (after "play"):
  ("rollout", n_games, n_steps, n_payload_bytes, play_s, upload_s, payload, outcomes)
  outcomes = List[Tuple[int, int]]: (slot_id, agent_result) per game
  agent_result: +1 learner won, −1 learner lost, 0 draw
  slot_id == _SLOT_CURRENT (-1) → current (always-updated) agent
  slot_id == _SLOT_SCRIPTED (-2) → scripted/random opponent (not in frozen pool)

Opponent sampling per game (within each worker):
  1. With prob current_fraction     → current self (latest weights)
  2. With prob frozen_fraction      → frozen checkpoint, PFSP-weighted by host-reported win rates
     (only if frozen_pool non-empty; otherwise this mass is current self-play)
  3. With prob scripted_fraction    → scripted bot sampled from scripted_distribution
     only when frozen_pool is non-empty (matches ``OpponentPool.sample_opponent``: no
     scripted opponents until at least one frozen agent exists; empty pool → pure self-play).

Host maintains DistributedPoolStats: central EMA win-rate per pool slot,
updated each round from all workers' game outcomes.  PFSP weights are sent
to workers so every worker uses the same sampling distribution.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import pickle
import random as _random
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.agents.ppo_structured_minibg_agent import (
    INFO_STRUCT_LEGAL,
    INFO_STRUCT_NEXT_LEGAL,
    MiniBGPPOStructuredAgent,
    StructuredMiniBGRolloutBuffer,
)
from src.envs import RewardConfig
from src.registry import make_game
from src.training.agent_perspective_env import AgentPerspectiveEnv, make_minibg_shaping_fn
from src.training.opponent_sampler import OpponentSampler
from src.training.trainer import BaseTrainer, TrainerCallback, Transition

_SLOT_CURRENT: int = -1   # game against the always-updated current agent
_SLOT_SCRIPTED: int = -2  # game against a scripted/random opponent


# ---------------------------------------------------------------------------
# Pool statistics (host-side, centralized)
# ---------------------------------------------------------------------------


@dataclass
class _SlotStats:
    slot_id: int
    games: int = 0
    wins: int = 0    # opponent won (learner lost)
    losses: int = 0  # opponent lost (learner won)
    draws: int = 0
    ema_win_rate: float = 0.5

    @property
    def cumulative_win_rate(self) -> float:
        if self.games <= 0:
            return 0.5
        return self.wins / self.games


class DistributedPoolStats:
    """Central EMA win-rate tracker for all opponent pool slots (host-side).

    Workers report (slot_id, agent_result) per game.  Host calls apply_outcomes()
    each round so stats never drift across workers.
    """

    def __init__(self, ema_beta: float = 0.05) -> None:
        self._beta = ema_beta
        self._slots: Dict[int, _SlotStats] = {}

    def register_slot(self, slot_id: int) -> None:
        self._slots[slot_id] = _SlotStats(slot_id=slot_id)

    def apply_outcomes(self, outcomes: List[Tuple[int, int]]) -> None:
        for slot_id, agent_result in outcomes:
            stats = self._slots.get(slot_id)
            if stats is None:
                continue
            stats.games += 1
            if agent_result == 1:
                stats.losses += 1
                y = 0.0
            elif agent_result == -1:
                stats.wins += 1
                y = 1.0
            else:
                stats.draws += 1
                y = 0.5
            stats.ema_win_rate = (1.0 - self._beta) * stats.ema_win_rate + self._beta * y

    def win_rates(self) -> Dict[int, float]:
        """Raw EMA win rates for real frozen slots only (sent to workers for PFSP)."""
        return {sid: s.ema_win_rate for sid, s in self._slots.items() if sid >= 0}

    def get_stats(self, *, pfsp_eps: float = 1e-2) -> List[Dict[str, Any]]:
        """Frozen-slot table for status JSON; includes PFSP selection_probability (same as ``_pfsp_choice``)."""
        frozen = [s for s in self._slots.values() if s.slot_id >= 0]
        frozen.sort(key=lambda s: s.slot_id)
        weights = [max(s.ema_win_rate, pfsp_eps) ** 2 for s in frozen]
        total_w = sum(weights)
        if total_w > 0:
            probs = [w / total_w for w in weights]
        elif frozen:
            probs = [1.0 / len(frozen)] * len(frozen)
        else:
            probs = []

        out: List[Dict[str, Any]] = []
        for s, p in zip(frozen, probs):
            out.append(
                {
                    "slot_id": s.slot_id,
                    "games": s.games,
                    "wins": s.wins,
                    "losses": s.losses,
                    "draws": s.draws,
                    "ema_win_rate": round(s.ema_win_rate, 4),
                    "cumulative_win_rate": round(s.cumulative_win_rate, 4),
                    "selection_probability": round(p, 4),
                }
            )
        return out


class _PoolStatusAdapter:
    """Thin shim so StatusFileCallback._write_self_play_frozen works unchanged."""

    def __init__(self, pool_stats: DistributedPoolStats) -> None:
        self._pool_stats = pool_stats

    @property
    def frozen_agents(self) -> List[Dict[str, Any]]:
        return self._pool_stats.get_stats()

    def get_frozen_stats_for_status(self) -> List[Dict[str, Any]]:
        return self._pool_stats.get_stats()


class _DistributedOpponentSamplerAdapter:
    """Makes DistributedTrainer.opponent_sampler compatible with CheckpointCallback / StatusFileCallback."""

    def __init__(self, pool_stats: DistributedPoolStats) -> None:
        self.opponent_pool = _PoolStatusAdapter(pool_stats)

    def on_checkpoint(self, path: "str | Path", episode_index: int) -> None:
        pass  # DistributedTrainer.train() drives add_to_pool via checkpoint_saved metric

    def on_episode_end(self, episode_index: int, info: Optional[dict] = None) -> None:
        pass


# ---------------------------------------------------------------------------
# Worker-side helpers (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------


class _MutableOpponentSampler(OpponentSampler):
    """Opponent sampler where the active opponent object can be swapped before each game."""

    def __init__(self) -> None:
        self._opponent: Optional[Any] = None

    def set_opponent(self, opponent: Any) -> None:
        self._opponent = opponent

    def sample(self) -> Any:
        return self._opponent

    def prepare(self, episode_index: int) -> None:
        pass

    def on_episode_end(self, episode_index: int, info: Optional[dict] = None) -> None:
        pass


# Keep for external use / backward compat
class FixedOpponentSampler(OpponentSampler):
    """Always returns the same opponent agent object."""

    def __init__(self, opponent: Any) -> None:
        self._opponent = opponent

    def prepare(self, episode_index: int) -> None:
        pass

    def sample(self) -> Any:
        return self._opponent

    def on_episode_end(self, episode_index: int, info: Optional[dict] = None) -> None:
        pass


def _apply_ppo_hparams(
    agent: MiniBGPPOStructuredAgent,
    *,
    rollout_steps: int,
    ppo_epochs: int,
    minibatch_size: int,
) -> None:
    agent.rollout_steps = int(rollout_steps)
    agent.ppo_epochs = int(ppo_epochs)
    agent.minibatch_size = int(minibatch_size)


def _buffer_to_payload(buf: StructuredMiniBGRolloutBuffer) -> bytes:
    return pickle.dumps(buf, protocol=pickle.HIGHEST_PROTOCOL)


def _payload_to_buffer(data: bytes) -> StructuredMiniBGRolloutBuffer:
    return pickle.loads(data)


def _merge_buffers(buffers: List[StructuredMiniBGRolloutBuffer]) -> StructuredMiniBGRolloutBuffer:
    out = StructuredMiniBGRolloutBuffer()
    for buf in buffers:
        out.obs.extend(buf.obs)
        out.legal_lists.extend(buf.legal_lists)
        out.action_indices.extend(buf.action_indices)
        out.complete_turn.extend(buf.complete_turn)
        out.occupied_masks.extend(buf.occupied_masks)
        out.order_picks.extend(buf.order_picks)
        out.rewards.extend(buf.rewards)
        out.dones.extend(buf.dones)
        out.values.extend(buf.values)
        out.log_probs.extend(buf.log_probs)
        out.next_obs.extend(buf.next_obs)
        out.next_legal_lists.extend(buf.next_legal_lists)
    return out


def _state_dict_bytes(agent: MiniBGPPOStructuredAgent, *, map_cpu: bool = False) -> bytes:
    import torch
    sd = agent.policy_net.state_dict()
    if map_cpu:
        sd = {k: v.detach().cpu() for k, v in sd.items()}
    bio = BytesIO()
    torch.save(sd, bio)
    return bio.getvalue()


def _load_state_dict_bytes(agent: MiniBGPPOStructuredAgent, payload: bytes) -> None:
    import torch
    bio = BytesIO(payload)
    sd = torch.load(bio, map_location=agent.device, weights_only=True)
    agent.policy_net.load_state_dict(sd)


def _pfsp_choice(
    frozen_pool: Dict[int, bytes],
    slot_win_rates: Dict[int, float],
    rng: _random.Random,
    eps: float = 1e-2,
) -> int:
    """PFSP-weighted choice: prefer opponents with higher win rate (stronger = more training signal)."""
    slots = list(frozen_pool.keys())
    weights = [max(slot_win_rates.get(s, 0.5), eps) ** 2 for s in slots]
    total = sum(weights)
    if total <= 0:
        return rng.choice(slots)
    r = rng.random() * total
    acc = 0.0
    for s, w in zip(slots, weights):
        acc += w
        if r <= acc:
            return s
    return slots[-1]


def _sample_scripted_key(dist: Dict[str, float], rng: _random.Random) -> str:
    keys = list(dist.keys())
    weights = [dist[k] for k in keys]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for k, w in zip(keys, weights):
        acc += w
        if r <= acc:
            return k
    return keys[-1]


def _create_scripted_opponent(key: str, *, seed: int) -> Any:
    from src.agents import RandomAgent
    if key == "random":
        return RandomAgent(seed=seed)
    from src.envs.minibg.heuristic_bots.agent_adapter import MiniBGHeuristicAgent
    from src.envs.minibg.heuristic_bots.tournament import make_bot
    return MiniBGHeuristicAgent(make_bot(key, seed))


def _collect_until_steps(
    agent: MiniBGPPOStructuredAgent,
    ppo_opponent: MiniBGPPOStructuredAgent,
    *,
    min_steps: int,
    mg: dict,
    current_sd: bytes,
    frozen_pool: Dict[int, bytes],
    slot_win_rates: Dict[int, float],
    current_fraction: float,
    scripted_fraction: float,
    scripted_distribution: Dict[str, float],
    game_rng: _random.Random,
    seed: int,
) -> Tuple[int, int, StructuredMiniBGRolloutBuffer, List[Tuple[int, int]]]:
    """Collect games until rollout buffer has >= min_steps transitions.

    Returns (n_games, n_steps, buffer, outcomes).
    outcomes: list of (slot_id, agent_result) per game.
    """
    import torch
    torch.set_num_threads(1)
    agent.train()
    ppo_opponent.eval()
    if hasattr(ppo_opponent, "epsilon"):
        ppo_opponent.epsilon = 0.0

    opp_sampler = _MutableOpponentSampler()
    base = make_game("minibg", reward_config=RewardConfig(), **mg)
    shaping = make_minibg_shaping_fn(float(mg.get("battle_damage_shaping", 0.0)))
    env = AgentPerspectiveEnv(
        base,
        opp_sampler,
        agent_first_probability=0.5,
        shaping_fn=shaping,
    )

    game_outcomes: List[Tuple[int, int]] = []
    n_games = 0
    # Fraction that goes to frozen pool (what's left after current + scripted).
    frozen_fraction = max(0.0, 1.0 - current_fraction - scripted_fraction)

    while len(agent.rollout_buffer) < min_steps:
        r = game_rng.random()

        if (
            frozen_pool
            and r >= 1.0 - scripted_fraction
            and scripted_fraction > 0
            and scripted_distribution
        ):
            # Scripted only once frozen pool exists (parity with OpponentPool.sample_opponent).
            key = _sample_scripted_key(scripted_distribution, game_rng)
            scripted_opp = _create_scripted_opponent(key, seed=seed + n_games * 997)
            opp_sampler.set_opponent(scripted_opp)
            opp_slot_id = _SLOT_SCRIPTED
        elif frozen_pool and r >= current_fraction:
            # Frozen checkpoint — PFSP-weighted by host-reported win rates
            opp_slot_id = _pfsp_choice(frozen_pool, slot_win_rates, game_rng)
            _load_state_dict_bytes(ppo_opponent, frozen_pool[opp_slot_id])
            opp_sampler.set_opponent(ppo_opponent)
        else:
            # Current self-play
            _load_state_dict_bytes(ppo_opponent, current_sd)
            opp_sampler.set_opponent(ppo_opponent)
            opp_slot_id = _SLOT_CURRENT

        obs = env.reset()
        last_info: dict = {}
        while not env.done:
            legal_list = env.legal_structured_actions()
            struct_act, board_perm, idx = agent.act_structured(
                obs, legal_list, env, deterministic=False
            )
            step = env.step_structured(struct_act, board_perm=board_perm)
            next_sl: List[Any] = [] if step.done else list(env.legal_structured_actions())
            transition = Transition(
                obs=obs,
                action=idx,
                reward=float(step.reward),
                next_obs=step.obs,
                terminated=step.terminated,
                truncated=step.truncated,
                info={
                    **(step.info if isinstance(step.info, dict) else {}),
                    INFO_STRUCT_LEGAL: legal_list,
                    INFO_STRUCT_NEXT_LEGAL: next_sl,
                },
                legal_mask=None,
                next_legal_mask=None,
            )
            agent.observe(transition)
            obs = step.obs
            if step.done:
                last_info = step.info if isinstance(step.info, dict) else {}

        winner = last_info.get("winner")
        agent_token = int(getattr(env, "agent_token", 1))
        if winner is None:
            raise RuntimeError(
                "DistributedTrainer: terminal step missing info['winner']; "
                "ensure AgentPerspectiveEnv propagates opponent-drain terminal info."
            )
        elif winner == agent_token:
            result = 1
        elif winner == -agent_token:
            result = -1
        else:
            result = 0
        game_outcomes.append((opp_slot_id, result))
        env.notify_episode_end(last_info)
        n_games += 1

    buf = agent.rollout_buffer
    n_steps = len(buf)
    agent.rollout_buffer = StructuredMiniBGRolloutBuffer()
    return n_games, n_steps, buf, game_outcomes


def _worker_main(
    worker_id: int,
    ck_path: str,
    min_steps: int,
    rollout_steps: int,
    ppo_epochs: int,
    minibatch_size: int,
    seed: int,
    mg: dict,
    device: str,
    current_fraction: float,
    scripted_fraction: float,
    scripted_distribution: Dict[str, float],
    max_pool_size: int,
    cmd_conn: Any,
) -> None:
    import torch
    torch.set_num_threads(1)

    agent = MiniBGPPOStructuredAgent.load(ck_path, device=device, seed=seed + worker_id)
    ppo_opponent = MiniBGPPOStructuredAgent.load(
        ck_path, device=device, seed=seed + worker_id + 913_123
    )
    _apply_ppo_hparams(
        agent,
        rollout_steps=rollout_steps,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
    )
    agent.train()

    current_sd: bytes = _state_dict_bytes(agent, map_cpu=False)
    # frozen_pool: slot_id → sd_bytes; insertion-ordered for FIFO eviction
    frozen_pool: Dict[int, bytes] = {}
    # PFSP win rates sent by host (raw EMA rates, not normalized)
    slot_win_rates: Dict[int, float] = {}

    game_rng = _random.Random(seed + worker_id * 777_777)
    round_seed_base = seed + worker_id * 10_000

    while True:
        msg = cmd_conn.recv()
        tag = msg[0]

        if tag == "load":
            _load_state_dict_bytes(agent, msg[1])
            current_sd = msg[1]

        elif tag == "add_to_pool":
            slot_id: int = msg[1]
            slot_sd: bytes = msg[2]
            if len(frozen_pool) >= max_pool_size:
                oldest = next(iter(frozen_pool))
                del frozen_pool[oldest]
            frozen_pool[slot_id] = slot_sd

        elif tag == "set_weights":
            slot_win_rates = dict(msg[1])

        elif tag == "play":
            round_idx: int = msg[1]
            t0 = time.perf_counter()
            ng, nsteps, buf, outcomes = _collect_until_steps(
                agent,
                ppo_opponent,
                min_steps=min_steps,
                mg=mg,
                current_sd=current_sd,
                frozen_pool=frozen_pool,
                slot_win_rates=slot_win_rates,
                current_fraction=current_fraction,
                scripted_fraction=scripted_fraction,
                scripted_distribution=scripted_distribution,
                game_rng=game_rng,
                seed=round_seed_base + round_idx * 1_000_003,
            )
            play_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            payload = _buffer_to_payload(buf)
            upload_s = time.perf_counter() - t1
            cmd_conn.send(("rollout", ng, nsteps, len(payload), play_s, upload_s, payload, outcomes))

        elif tag == "stop":
            break

        else:
            raise RuntimeError(f"unknown worker command: {tag!r}")


# ---------------------------------------------------------------------------
# DistributedCheckpointCallback
# ---------------------------------------------------------------------------


class DistributedCheckpointCallback(TrainerCallback):
    """Checkpoint callback that handles bulk step increments (one per round).

    Uses step // interval thresholding instead of step % interval == 0 so a
    checkpoint is always saved when the cumulative step count crosses any
    multiple of interval, even if the crossing happens mid-round.
    """

    def __init__(self, output_dir: "str | Path", interval: int, prefix: str = "checkpoint"):
        self.output_dir = Path(output_dir)
        self.interval = max(1, interval)
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_n: int = 0

    def on_step_end(
        self,
        trainer: "DistributedTrainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, Any],
    ) -> None:
        n = step // self.interval
        if n > self._last_n:
            self._last_n = n
            path = self.output_dir / f"{self.prefix}_{step}.pt"
            trainer.agent.save(str(path))
            metrics["checkpoint_saved"] = step
            if trainer.opponent_sampler is not None:
                trainer.opponent_sampler.on_checkpoint(path, trainer.episode_index)

    def on_train_end(self, trainer: "DistributedTrainer") -> None:
        path = self.output_dir / f"{self.prefix}_final.pt"
        trainer.agent.save(str(path))
        if trainer.opponent_sampler is not None:
            trainer.opponent_sampler.on_checkpoint(path, trainer.episode_index)


# ---------------------------------------------------------------------------
# DistributedRoundResult
# ---------------------------------------------------------------------------


@dataclass
class DistributedRoundResult:
    round_idx: int
    n_steps: int
    n_games: int
    metrics: Dict[str, Any]
    play_s: float
    host_s: float
    upload_bytes: int
    sd_bytes: bytes                      # updated state dict (already broadcast to workers)
    outcomes: List[Tuple[int, int]]      # (slot_id, agent_result) for every game this round


# ---------------------------------------------------------------------------
# DistributedTrainer
# ---------------------------------------------------------------------------


_DUMMY_TRANSITION: Transition = Transition(
    obs=np.zeros(1, dtype=np.float32),
    action=0,
    reward=0.0,
    next_obs=np.zeros(1, dtype=np.float32),
    terminated=False,
    truncated=False,
    info={},
)


class DistributedTrainer(BaseTrainer):
    """Distributed PPO training loop.

    Workers (CPU processes) collect game rollouts in parallel; the host (GPU)
    runs the PPO gradient update each round.

    Opponent sampling per game:
      - current_fraction   → current self (latest weights)
      - frozen_fraction    → frozen past checkpoints, PFSP-weighted
      - scripted_fraction  → scripted/heuristic bots

    Win-rate statistics are tracked centrally on the host (DistributedPoolStats)
    and PFSP weights are broadcast to workers after every round.

    on_step_end callbacks are fired once per round (not per transition).
    Use DistributedCheckpointCallback (not the standard CheckpointCallback)
    to handle bulk step increments correctly.
    on_episode_end is NOT called.
    """

    def __init__(
        self,
        agent: MiniBGPPOStructuredAgent,
        worker_ckpt_path: "str | Path",
        *,
        workers: int = 4,
        rollout_steps: int = 8096,
        ppo_epochs: int = 4,
        minibatch_size: int = 512,
        worker_device: str = "cpu",
        seed: int = 42,
        game_kwargs: Optional[Dict[str, Any]] = None,
        callbacks: Optional[Iterable[TrainerCallback]] = None,
        current_fraction: float = 0.4,
        scripted_fraction: float = 0.2,
        scripted_distribution: Optional[Dict[str, float]] = None,
        max_pool_size: int = 20,
        ema_beta: float = 0.05,
    ) -> None:
        self._pool_stats = DistributedPoolStats(ema_beta=ema_beta)
        self._pool_stats.register_slot(_SLOT_SCRIPTED)
        self._pool_stats.register_slot(_SLOT_CURRENT)
        super().__init__(
            callbacks=callbacks,
            opponent_sampler=_DistributedOpponentSamplerAdapter(self._pool_stats),
        )
        self.agent = agent
        self._worker_ckpt_path = Path(worker_ckpt_path)
        self._workers = workers
        self._rollout_steps = rollout_steps
        self._ppo_epochs = ppo_epochs
        self._minibatch_size = minibatch_size
        self._worker_device = worker_device
        self._seed = seed
        self._game_kwargs: Dict[str, Any] = dict(game_kwargs or {})
        self._current_fraction = current_fraction
        self._scripted_fraction = scripted_fraction
        self._scripted_distribution: Dict[str, float] = dict(
            scripted_distribution or {"random": 1.0}
        )
        self._max_pool_size = max_pool_size
        self._next_slot_id = 0
        self._conns: List[Any] = []
        self._procs: List[mp.Process] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, total_steps: int) -> None:
        self._target_total_steps = total_steps
        self._spawn_workers()

        for cb in self.callbacks:
            cb.on_train_begin(self)

        round_idx = 0
        while not self.stop_training and self.global_step < total_steps:
            result = self._run_round(round_idx)
            self.global_step += result.n_steps
            self.episode_index += result.n_games

            for cb in self.callbacks:
                cb.on_step_end(self, self.global_step, _DUMMY_TRANSITION, result.metrics)

            # If a checkpoint was saved this round, freeze current weights as new pool slot.
            if result.metrics.get("checkpoint_saved"):
                self._add_to_pool(result.sd_bytes)

            round_idx += 1

        for cb in self.callbacks:
            cb.on_train_end(self)
        self._shutdown_workers()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_round(self, round_idx: int) -> DistributedRoundResult:
        for conn in self._conns:
            conn.send(("play", round_idx))

        play_data: List[Tuple] = []
        for conn in self._conns:
            tag, ng, nsteps, nbytes, play_s, upload_s, payload, outcomes = conn.recv()
            assert tag == "rollout"
            play_data.append((play_s, upload_s, payload, ng, nsteps, nbytes, outcomes))

        merged = _merge_buffers([_payload_to_buffer(p[2]) for p in play_data])
        all_outcomes: List[Tuple[int, int]] = [o for p in play_data for o in p[6]]

        n_steps = len(merged)  # capture BEFORE update() clears the buffer
        if n_steps < self._rollout_steps:
            raise RuntimeError(
                f"Merged rollout has {n_steps} steps < rollout_steps={self._rollout_steps}. "
                "Increase per-worker target or worker count."
            )

        # Sync step_count: observe() runs on workers so host agent never increments it.
        self.agent.step_count += n_steps
        self.agent.rollout_buffer = merged
        t0 = perf_counter()
        metrics: Dict[str, Any] = self.agent.update() or {}
        host_s = perf_counter() - t0

        sd_bytes = _state_dict_bytes(self.agent, map_cpu=True)

        # Update pool stats, compute PFSP weights, broadcast to workers.
        self._pool_stats.apply_outcomes(all_outcomes)
        win_rates = self._pool_stats.win_rates()
        for conn in self._conns:
            conn.send(("load", sd_bytes))
            conn.send(("set_weights", win_rates))

        return DistributedRoundResult(
            round_idx=round_idx,
            n_steps=n_steps,
            n_games=sum(p[3] for p in play_data),
            metrics=metrics,
            play_s=max(p[0] for p in play_data),
            host_s=host_s,
            upload_bytes=sum(p[5] for p in play_data),
            sd_bytes=sd_bytes,
            outcomes=all_outcomes,
        )

    def _add_to_pool(self, sd_bytes: bytes) -> None:
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        self._pool_stats.register_slot(slot_id)
        for conn in self._conns:
            conn.send(("add_to_pool", slot_id, sd_bytes))

    def _spawn_workers(self) -> None:
        per_worker = math.ceil(self._rollout_steps / self._workers)
        ctx = mp.get_context("spawn")
        for w in range(self._workers):
            parent, child = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_worker_main,
                args=(
                    w,
                    str(self._worker_ckpt_path),
                    per_worker,
                    self._rollout_steps,
                    self._ppo_epochs,
                    self._minibatch_size,
                    self._seed,
                    self._game_kwargs,
                    self._worker_device,
                    self._current_fraction,
                    self._scripted_fraction,
                    self._scripted_distribution,
                    self._max_pool_size,
                    child,
                ),
                daemon=True,
            )
            p.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(p)

    def _shutdown_workers(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("stop",))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()


__all__ = [
    "DistributedCheckpointCallback",
    "DistributedPoolStats",
    "DistributedRoundResult",
    "DistributedTrainer",
    "FixedOpponentSampler",
]

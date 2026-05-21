"""Distributed PPO trainer — workers collect rollouts (CPU), host runs PPO (GPU).

Worker protocol
---------------
Host → Worker:
  ("load", sd_bytes)                         update learner weights
  ("sync_league", {slot_id: sd_bytes}, rates) authoritative frozen pool + PFSP EMA rates
  ("play", round_idx)                        collect games; reply with rollout + outcomes
  ("stop",)                                  terminate

Worker → Host (after "play"):
  ("rollout", n_games, n_steps, n_payload_bytes, play_s, upload_s, payload, outcomes)
  outcomes = List[Tuple[int, AgentOutcome]]: (slot_id, agent_result) per game
  agent_result: legacy ±1/0 or placement score in [0, 1] for BGLike
  slot_id == SLOT_CURRENT (-1) → current (always-updated) agent
  slot_id == SLOT_SCRIPTED (-2) → scripted/random opponent (not in frozen pool)

Opponent sampling (within each worker):
  MiniBG: one opponent per 2p game (same fractions as below).
  BGLike: independent sample per non-current lobby seat (8 − num_current_seats).

  Per sample:
  1. With prob current_fraction     → current self (latest weights)
  2. With prob frozen_fraction      → frozen checkpoint, PFSP-weighted (if pool non-empty)
  3. Remaining mass                 → scripted (includes past_fraction when pool is empty)

Host maintains ``LeagueController``: central EMA win-rate per pool slot,
updated each round **after** rollout from all workers' outcomes. Workers only
apply ``sync_league`` snapshots from the host (no local stat updates).
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import pickle
import random as _random
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.agents.ppo_agent import PPOAgent, RolloutBuffer
from src.agents.ppo_structured_minibg_agent import (
    INFO_STRUCT_LEGAL,
    INFO_STRUCT_NEXT_LEGAL,
    MiniBGPPOStructuredAgent,
    StructuredMiniBGRolloutBuffer,
)
from src.envs import RewardConfig
from src.registry import make_game
from src.training.agent_perspective_env import AgentPerspectiveEnv, make_minibg_shaping_fn
from src.training.bglike_perspective import (
    apply_bglike_segment_closures_after_observe,
    make_bglike_agent_perspective_env,
    make_bglike_shaping_fn,
)
from src.training.selfplay.league_state import AgentOutcome, normalize_agent_score
from src.training.opponent_sampler import OpponentSampler
from src.training.selfplay.league_policy import (
    OpponentKind,
    decide_opponent_kind,
    pfsp_sample,
    sample_scripted_key,
    self_play_enabled,
)
from src.training.selfplay.league_state import (
    LeagueController,
    SLOT_CURRENT,
    SLOT_SCRIPTED,
)
from src.training.stack_trace_diag import enable_stack_dump_on_signal
from src.training.trainer import BaseTrainer, TrainerCallback, Transition

# Backward-compatible alias
DistributedPoolStats = LeagueController


class _PoolStatusAdapter:
    """Thin shim so StatusFileCallback._write_self_play_frozen works unchanged."""

    def __init__(self, league: LeagueController) -> None:
        self._league = league

    @property
    def frozen_agents(self) -> List[Dict[str, Any]]:
        return self._league.get_frozen_stats_for_status()

    def get_frozen_stats_for_status(self) -> List[Dict[str, Any]]:
        return self._league.get_frozen_stats_for_status()


class _DistributedOpponentSamplerAdapter:
    """Makes DistributedTrainer.opponent_sampler compatible with CheckpointCallback / StatusFileCallback."""

    def __init__(self, league: LeagueController) -> None:
        self.opponent_pool = _PoolStatusAdapter(league)

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


class _DistributedBglikeOpponentSampler(OpponentSampler):
    """Independent per-seat opponent sampling for 8p BGLike (distributed workers)."""

    def __init__(self, **sample_kwargs: Any) -> None:
        self._sample_kwargs = sample_kwargs
        self._slot_by_seat: Dict[int, int] = {}

    def prepare(self, episode_index: int) -> None:
        self._sample_kwargs["game_index"] = episode_index
        self._slot_by_seat = {}

    def sample(self) -> Any:
        raise RuntimeError("BGLike distributed training uses sample_for_seats()")

    def sample_for_seats(self, seats: Sequence[int]) -> Dict[int, Any]:
        agents: Dict[int, Any] = {}
        slots: Dict[int, int] = {}
        for seat in seats:
            s = int(seat)
            opp, slot_id = _sample_distributed_opponent(
                seat_index=s,
                **self._sample_kwargs,
            )
            agents[s] = opp
            slots[s] = slot_id
        self._slot_by_seat = slots
        return agents

    @property
    def _episode_slot_by_seat(self) -> Dict[int, int]:
        return self._slot_by_seat


def _sample_distributed_opponent(
    *,
    game_rng: _random.Random,
    use_self_play: bool,
    frozen_pool: Dict[int, bytes],
    slot_win_rates: Dict[int, float],
    current_fraction: float,
    past_fraction: float,
    scripted_distribution: Dict[str, float],
    seed: int,
    game_index: int,
    seat_index: int,
    gid: str,
    current_sd: bytes,
    ppo_opponent: Any,
) -> Tuple[Any, int]:
    """Sample one opponent agent and league slot_id (per non-current lobby seat)."""
    if not use_self_play:
        key = sample_scripted_key(scripted_distribution, game_rng)
        opp = _create_scripted_opponent(
            key,
            seed=seed + game_index * 997 + seat_index * 17,
            game_id=gid,
        )
        return opp, SLOT_SCRIPTED

    kind = decide_opponent_kind(
        game_rng.random(),
        current_fraction=current_fraction,
        past_fraction=past_fraction,
        frozen_nonempty=bool(frozen_pool),
        has_current_agent=True,
    )
    if kind == OpponentKind.SCRIPTED:
        key = sample_scripted_key(scripted_distribution, game_rng)
        opp = _create_scripted_opponent(
            key,
            seed=seed + game_index * 997 + seat_index * 17,
            game_id=gid,
        )
        return opp, SLOT_SCRIPTED
    opp = copy.deepcopy(ppo_opponent)
    if kind == OpponentKind.FROZEN:
        slot_ids = list(frozen_pool.keys())
        rates = [slot_win_rates.get(s, 0.5) for s in slot_ids]
        slot_id = pfsp_sample(slot_ids, rates, game_rng)
        _load_state_dict_bytes(opp, frozen_pool[slot_id])
    else:
        slot_id = SLOT_CURRENT
        _load_state_dict_bytes(opp, current_sd)
    opp.eval()
    if hasattr(opp, "epsilon"):
        opp.epsilon = 0.0
    return opp, slot_id


def _assign_minibg_opponent(
    opp_sampler: _MutableOpponentSampler,
    ppo_opponent: Any,
    *,
    current_sd: bytes,
    frozen_pool: Dict[int, bytes],
    slot_win_rates: Dict[int, float],
    current_fraction: float,
    past_fraction: float,
    scripted_distribution: Dict[str, float],
    game_rng: _random.Random,
    seed: int,
    game_index: int,
    gid: str,
) -> int:
    """Pick opponent kind, wire ``opp_sampler`` / ``ppo_opponent``; return league slot_id."""
    kind = decide_opponent_kind(
        game_rng.random(),
        current_fraction=current_fraction,
        past_fraction=past_fraction,
        frozen_nonempty=bool(frozen_pool),
        has_current_agent=True,
    )
    if kind == OpponentKind.SCRIPTED:
        key = sample_scripted_key(scripted_distribution, game_rng)
        scripted_opp = _create_scripted_opponent(
            key, seed=seed + game_index * 997, game_id=gid
        )
        opp_sampler.set_opponent(scripted_opp)
        return SLOT_SCRIPTED
    opp = ppo_opponent
    if kind == OpponentKind.FROZEN:
        slot_ids = list(frozen_pool.keys())
        rates = [slot_win_rates.get(s, 0.5) for s in slot_ids]
        slot_id = pfsp_sample(slot_ids, rates, game_rng)
        _load_state_dict_bytes(opp, frozen_pool[slot_id])
    else:
        slot_id = SLOT_CURRENT
        _load_state_dict_bytes(opp, current_sd)
    opp_sampler.set_opponent(opp)
    return int(slot_id)


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


def _game_id(mg: dict) -> str:
    return str(mg.get("game_id", "minibg")).strip().lower()


def _minibg_game_kwargs(mg: dict) -> dict:
    skip = frozenset({"game_id", "use_structured", "num_current_seats"})
    return {k: v for k, v in mg.items() if k not in skip}


def _use_structured_collect(mg: dict) -> bool:
    return _game_id(mg) == "minibg" and bool(mg.get("use_structured", True))


def _apply_ppo_hparams(
    agent: Any,
    *,
    rollout_steps: int,
    ppo_epochs: int,
    minibatch_size: int,
) -> None:
    agent.rollout_steps = int(rollout_steps)
    agent.ppo_epochs = int(ppo_epochs)
    agent.minibatch_size = int(minibatch_size)


def _new_rollout_buffer(mg: dict) -> Any:
    if _use_structured_collect(mg):
        return StructuredMiniBGRolloutBuffer()
    return RolloutBuffer()


def _buffer_to_payload(buf: Any) -> bytes:
    return pickle.dumps(buf, protocol=pickle.HIGHEST_PROTOCOL)


def _payload_to_buffer(data: bytes, mg: dict) -> Any:
    buf = pickle.loads(data)
    if _use_structured_collect(mg) and not isinstance(buf, StructuredMiniBGRolloutBuffer):
        raise TypeError("expected StructuredMiniBGRolloutBuffer payload")
    if not _use_structured_collect(mg) and not isinstance(buf, RolloutBuffer):
        raise TypeError("expected RolloutBuffer payload")
    return buf


def _merge_buffers(buffers: List[Any], mg: dict) -> Any:
    if _use_structured_collect(mg):
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
    out = RolloutBuffer()
    for buf in buffers:
        out.obs.extend(buf.obs)
        out.actions.extend(buf.actions)
        out.rewards.extend(buf.rewards)
        out.dones.extend(buf.dones)
        out.values.extend(buf.values)
        out.log_probs.extend(buf.log_probs)
        out.legal_masks.extend(buf.legal_masks)
        out.next_obs.extend(buf.next_obs)
        out.next_legal_masks.extend(buf.next_legal_masks)
        out.seat_ids.extend(buf.seat_ids)
    return out


def _state_dict_bytes(agent: Any, *, map_cpu: bool = False) -> bytes:
    import torch
    sd = agent.policy_net.state_dict()
    if map_cpu:
        sd = {k: v.detach().cpu() for k, v in sd.items()}
    bio = BytesIO()
    torch.save(sd, bio)
    return bio.getvalue()


def _load_state_dict_bytes(agent: Any, payload: bytes) -> None:
    import torch
    bio = BytesIO(payload)
    sd = torch.load(bio, map_location=agent.device, weights_only=True)
    agent.policy_net.load_state_dict(sd)


def _load_dist_agent(
    ck_path: str,
    *,
    device: str,
    seed: int,
    mg: dict,
) -> Any:
    if _use_structured_collect(mg):
        return MiniBGPPOStructuredAgent.load(ck_path, device=device, seed=seed)
    return PPOAgent.load(ck_path, device=device, seed=seed)


def _create_scripted_opponent(key: str, *, seed: int, game_id: str) -> Any:
    from src.agents import RandomAgent
    if key == "random":
        return RandomAgent(seed=seed)
    gid = game_id.strip().lower()
    if gid == "bglike":
        from src.envs.bglike.heuristic_bots import make_heuristic_agent

        return make_heuristic_agent(key, seed=seed)
    from src.envs.minibg.heuristic_bots.agent_adapter import MiniBGHeuristicAgent
    from src.envs.minibg.heuristic_bots.tournament import make_bot

    return MiniBGHeuristicAgent(make_bot(key, seed))


def _make_collect_env(
    mg: dict,
    opp_sampler: OpponentSampler,
    *,
    seed: int,
) -> Any:
    gid = _game_id(mg)
    if gid == "bglike":
        lobby_kw = {
            k: v
            for k, v in mg.items()
            if k
            not in (
                "game_id",
                "use_structured",
                "num_current_seats",
                "battle_damage_shaping",
                "seed",
            )
        }
        return make_bglike_agent_perspective_env(
            opp_sampler,
            num_current_seats=int(mg.get("num_current_seats", 1)),
            seed=seed,
            shaping_fn=make_bglike_shaping_fn(float(mg.get("battle_damage_shaping", 0.0))),
            **lobby_kw,
        )
    base = make_game("minibg", reward_config=RewardConfig(), **_minibg_game_kwargs(mg))
    shaping = make_minibg_shaping_fn(float(mg.get("battle_damage_shaping", 0.0)))
    return AgentPerspectiveEnv(
        base,
        opp_sampler,
        agent_first_probability=0.5,
        shaping_fn=shaping,
    )


def _episode_agent_score(env: Any, last_info: dict, mg: dict) -> AgentOutcome:
    if _game_id(mg) == "bglike":
        from src.envs.bglike.placement import placement_score

        info = last_info if isinstance(last_info, dict) else {}
        placements = info.get("placements_current") or {}
        seat = info.get("acting_seat")
        if seat is None and hasattr(env, "_bg_base"):
            acting = getattr(env._bg_base, "acting_seat", None)
            if acting is not None:
                seat = acting
            elif getattr(env._bg_base, "current_seats", None):
                seat = env._bg_base.current_seats[0]
        if seat is not None and seat in placements:
            return placement_score(int(placements[seat]))
        place = info.get("placement")
        if place is not None:
            return placement_score(int(place))
        return 0.5
    winner = last_info.get("winner")
    agent_token = int(getattr(env, "agent_token", 1))
    if winner is None:
        raise RuntimeError(
            "DistributedTrainer: terminal step missing info['winner']; "
            "ensure AgentPerspectiveEnv propagates opponent-drain terminal info."
        )
    if winner == agent_token:
        return 1
    if winner == -agent_token:
        return -1
    return 0


def _collect_until_steps_flat(
    agent: PPOAgent,
    ppo_opponent: PPOAgent,
    *,
    min_steps: int,
    mg: dict,
    current_sd: bytes,
    frozen_pool: Dict[int, bytes],
    slot_win_rates: Dict[int, float],
    current_fraction: float,
    past_fraction: float,
    scripted_distribution: Dict[str, float],
    game_rng: _random.Random,
    seed: int,
    round_idx: int,
    start_episode: int,
) -> Tuple[int, int, RolloutBuffer, List[Tuple[int, AgentOutcome]]]:
    import torch

    torch.set_num_threads(1)
    agent.train()
    ppo_opponent.eval()

    gid = _game_id(mg)
    use_self_play = self_play_enabled(
        episode=round_idx,
        start_episode=start_episode,
        has_self_play_config=True,
    )
    if gid == "bglike":
        opp_sampler = _DistributedBglikeOpponentSampler(
            game_rng=game_rng,
            use_self_play=use_self_play,
            frozen_pool=frozen_pool,
            slot_win_rates=slot_win_rates,
            current_fraction=current_fraction,
            past_fraction=past_fraction,
            scripted_distribution=scripted_distribution,
            seed=seed,
            game_index=0,
            gid=gid,
            current_sd=current_sd,
            ppo_opponent=ppo_opponent,
        )
    else:
        opp_sampler = _MutableOpponentSampler()
    env = _make_collect_env(mg, opp_sampler, seed=seed)
    env.set_learner_agent(agent)

    game_outcomes: List[Tuple[int, AgentOutcome]] = []
    n_games = 0

    while len(agent.rollout_buffer) < min_steps:
        if gid != "bglike":
            if not use_self_play:
                key = sample_scripted_key(scripted_distribution, game_rng)
                scripted_opp = _create_scripted_opponent(
                    key, seed=seed + n_games * 997, game_id=gid
                )
                opp_sampler.set_opponent(scripted_opp)
                opp_slot_id = SLOT_SCRIPTED
            else:
                opp_slot_id = _assign_minibg_opponent(
                    opp_sampler,
                    ppo_opponent,
                    current_sd=current_sd,
                    frozen_pool=frozen_pool,
                    slot_win_rates=slot_win_rates,
                    current_fraction=current_fraction,
                    past_fraction=past_fraction,
                    scripted_distribution=scripted_distribution,
                    game_rng=game_rng,
                    seed=seed,
                    game_index=n_games,
                    gid=gid,
                )

        obs = env.reset()
        last_info: dict = {}
        while not env.done:
            legal = np.asarray(env.legal_actions_mask, dtype=bool)
            if not bool(legal.any()):
                break
            action = int(agent.act(obs, legal_mask=legal, deterministic=False))
            step = env.step(action)
            info = step.info if isinstance(step.info, dict) else {}
            step_done = bool(
                env.done
                or info.get("lobby_episode_done")
            )
            next_legal = (
                np.zeros_like(legal)
                if step_done
                else np.asarray(env.legal_actions_mask, dtype=bool)
            )
            transition = Transition(
                obs=obs,
                action=action,
                reward=float(step.reward),
                next_obs=step.obs,
                terminated=step.terminated,
                truncated=step.truncated,
                info=info,
                legal_mask=legal,
                next_legal_mask=next_legal,
            )
            agent.observe(transition)
            closure_outcomes = apply_bglike_segment_closures_after_observe(env, info)
            if gid == "bglike" and closure_outcomes:
                game_outcomes.extend(
                    (int(slot_id), normalize_agent_score(score))
                    for slot_id, score in closure_outcomes
                )
            obs = step.obs
            if step_done:
                last_info = info
                break

        if gid != "bglike":
            score = _episode_agent_score(env, last_info, mg)
            game_outcomes.append((opp_slot_id, normalize_agent_score(score)))
        env.notify_episode_end(last_info)
        n_games += 1

    buf = agent.rollout_buffer
    n_steps = len(buf)
    agent.rollout_buffer = RolloutBuffer()
    return n_games, n_steps, buf, game_outcomes


def _collect_until_steps_structured(
    agent: MiniBGPPOStructuredAgent,
    ppo_opponent: MiniBGPPOStructuredAgent,
    *,
    min_steps: int,
    mg: dict,
    current_sd: bytes,
    frozen_pool: Dict[int, bytes],
    slot_win_rates: Dict[int, float],
    current_fraction: float,
    past_fraction: float,
    scripted_distribution: Dict[str, float],
    game_rng: _random.Random,
    seed: int,
    round_idx: int,
    start_episode: int,
) -> Tuple[int, int, StructuredMiniBGRolloutBuffer, List[Tuple[int, AgentOutcome]]]:
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
    env = _make_collect_env(mg, opp_sampler, seed=seed)

    game_outcomes: List[Tuple[int, AgentOutcome]] = []
    n_games = 0
    use_self_play = self_play_enabled(
        episode=round_idx,
        start_episode=start_episode,
        has_self_play_config=True,
    )

    while len(agent.rollout_buffer) < min_steps:
        if not use_self_play:
            key = sample_scripted_key(scripted_distribution, game_rng)
            scripted_opp = _create_scripted_opponent(
                key, seed=seed + n_games * 997, game_id="minibg"
            )
            opp_sampler.set_opponent(scripted_opp)
            opp_slot_id = SLOT_SCRIPTED
        else:
            opp_slot_id = _assign_minibg_opponent(
                opp_sampler,
                ppo_opponent,
                current_sd=current_sd,
                frozen_pool=frozen_pool,
                slot_win_rates=slot_win_rates,
                current_fraction=current_fraction,
                past_fraction=past_fraction,
                scripted_distribution=scripted_distribution,
                game_rng=game_rng,
                seed=seed,
                game_index=n_games,
                gid="minibg",
            )

        obs = env.reset()
        last_info: dict = {}
        while not env.done:
            legal_list = env.legal_structured_actions()
            if not legal_list:
                from src.envs.minibg.invariants import assert_shop_has_legal_actions

                assert_shop_has_legal_actions(
                    env.state,
                    [],
                    where="distributed_trainer._collect_until_steps.agent_turn",
                )
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

        score = _episode_agent_score(env, last_info, mg)
        game_outcomes.append((opp_slot_id, normalize_agent_score(score)))
        env.notify_episode_end(last_info)
        n_games += 1

    buf = agent.rollout_buffer
    n_steps = len(buf)
    agent.rollout_buffer = StructuredMiniBGRolloutBuffer()
    return n_games, n_steps, buf, game_outcomes


def _collect_until_steps(
    agent: Any,
    ppo_opponent: Any,
    *,
    min_steps: int,
    mg: dict,
    current_sd: bytes,
    frozen_pool: Dict[int, bytes],
    slot_win_rates: Dict[int, float],
    current_fraction: float,
    past_fraction: float,
    scripted_distribution: Dict[str, float],
    game_rng: _random.Random,
    seed: int,
    round_idx: int,
    start_episode: int,
) -> Tuple[int, int, Any, List[Tuple[int, AgentOutcome]]]:
    if _use_structured_collect(mg):
        return _collect_until_steps_structured(
            agent,
            ppo_opponent,
            min_steps=min_steps,
            mg=mg,
            current_sd=current_sd,
            frozen_pool=frozen_pool,
            slot_win_rates=slot_win_rates,
            current_fraction=current_fraction,
            past_fraction=past_fraction,
            scripted_distribution=scripted_distribution,
            game_rng=game_rng,
            seed=seed,
            round_idx=round_idx,
            start_episode=start_episode,
        )
    return _collect_until_steps_flat(
        agent,
        ppo_opponent,
        min_steps=min_steps,
        mg=mg,
        current_sd=current_sd,
        frozen_pool=frozen_pool,
        slot_win_rates=slot_win_rates,
        current_fraction=current_fraction,
        past_fraction=past_fraction,
        scripted_distribution=scripted_distribution,
        game_rng=game_rng,
        seed=seed,
        round_idx=round_idx,
        start_episode=start_episode,
    )


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
    past_fraction: float,
    scripted_distribution: Dict[str, float],
    start_episode: int,
    cmd_conn: Any,
    run_dir: str,
) -> None:
    import os
    import torch

    os.environ["RL_RUN_DIR"] = run_dir
    enable_stack_dump_on_signal()
    torch.set_num_threads(1)

    agent = _load_dist_agent(
        ck_path, device=device, seed=seed + worker_id, mg=mg
    )
    ppo_opponent = _load_dist_agent(
        ck_path, device=device, seed=seed + worker_id + 913_123, mg=mg
    )
    _apply_ppo_hparams(
        agent,
        rollout_steps=rollout_steps,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
    )
    agent.train()

    current_sd: bytes = _state_dict_bytes(agent, map_cpu=False)
    frozen_pool: Dict[int, bytes] = {}
    slot_win_rates: Dict[int, float] = {}

    game_rng = _random.Random(seed + worker_id * 777_777)
    round_seed_base = seed + worker_id * 10_000

    while True:
        msg = cmd_conn.recv()
        tag = msg[0]

        if tag == "load":
            _load_state_dict_bytes(agent, msg[1])
            current_sd = msg[1]

        elif tag == "sync_league":
            frozen_pool = dict(msg[1])
            slot_win_rates = dict(msg[2])

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
                past_fraction=past_fraction,
                scripted_distribution=scripted_distribution,
                game_rng=game_rng,
                seed=round_seed_base + round_idx * 1_000_003,
                round_idx=round_idx,
                start_episode=start_episode,
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
    outcomes: List[Tuple[int, AgentOutcome]]


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

    Win-rate statistics are tracked centrally on the host (``LeagueController``)
    and the frozen pool + PFSP weights are broadcast after every round.

    on_step_end callbacks are fired once per round (not per transition).
    Use DistributedCheckpointCallback (not the standard CheckpointCallback)
    to handle bulk step increments correctly.
    on_episode_end is NOT called.
    """

    def __init__(
        self,
        agent: Any,
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
        past_fraction: float = 0.4,
        scripted_distribution: Optional[Dict[str, float]] = None,
        max_pool_size: int = 20,
        ema_beta: float = 0.05,
        start_episode: int = 0,
        run_dir: Optional[str] = None,
    ) -> None:
        self._league = LeagueController(ema_beta=ema_beta)
        self._league.register_meta_slot(SLOT_SCRIPTED)
        self._league.register_meta_slot(SLOT_CURRENT)
        self._frozen_pool: Dict[int, bytes] = {}
        super().__init__(
            callbacks=callbacks,
            opponent_sampler=_DistributedOpponentSamplerAdapter(self._league),
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
        self._past_fraction = past_fraction
        self._scripted_distribution: Dict[str, float] = dict(
            scripted_distribution or {"random": 1.0}
        )
        self._max_pool_size = max_pool_size
        self._start_episode = int(start_episode)
        self._run_dir = str(run_dir) if run_dir else ""
        self._conns: List[Any] = []
        self._procs: List[mp.Process] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, total_steps: int) -> None:
        import os

        if self._run_dir:
            os.environ["RL_RUN_DIR"] = self._run_dir
        enable_stack_dump_on_signal()
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

        merged = _merge_buffers(
            [_payload_to_buffer(p[2], self._game_kwargs) for p in play_data],
            self._game_kwargs,
        )
        all_outcomes: List[Tuple[int, AgentOutcome]] = [
            o for p in play_data for o in p[6]
        ]

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

        self._league.apply_outcomes(all_outcomes)
        for conn in self._conns:
            conn.send(("load", sd_bytes))
        self._broadcast_sync_league()

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
        slot_id = self._league.add_frozen_bytes(sd_bytes)
        self._frozen_pool[slot_id] = sd_bytes
        self._league.evict_worst_ema(self._max_pool_size)
        live = {s.slot_id for s in self._league.frozen_slots()}
        self._frozen_pool = {k: v for k, v in self._frozen_pool.items() if k in live}
        self._broadcast_sync_league()

    def _broadcast_sync_league(self) -> None:
        win_rates = self._league.win_rates()
        payload = dict(self._frozen_pool)
        for conn in self._conns:
            conn.send(("sync_league", payload, win_rates))

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
                    self._past_fraction,
                    self._scripted_distribution,
                    self._start_episode,
                    child,
                    self._run_dir,
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
    "LeagueController",
    "SLOT_CURRENT",
    "SLOT_SCRIPTED",
]

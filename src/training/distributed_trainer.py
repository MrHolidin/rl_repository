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
  outcomes = List[GameRecord]: one record per game or segment closure batch
  slot_id == SLOT_CURRENT (-1) → current (always-updated) agent
  slot_id < -1 → per-key scripted bot meta slots (-2, -3, ...)
  slot_id >= 0 → frozen checkpoint pool slot

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
    collect_bglike_lobby_league_outcome,
    make_bglike_agent_perspective_env,
    make_bglike_shaping_fn,
)
from src.training.selfplay.game_record import (
    GameRecord,
    SLOT_SCRIPTED,
    build_scripted_slot_map,
    invert_scripted_slot_map,
    minibg_record_from_learner_score,
)
from src.training.selfplay.league_config import LeagueSamplerConfig, LeagueSettings
from src.training.selfplay.league_sampler import LeagueSyncState, sample_league_opponent
from src.training.selfplay.league_state import (
    AgentOutcome,
    LeagueController,
    SLOT_CURRENT,
    normalize_agent_score,
)
from src.training.opponent_sampler import OpponentSampler
from src.training.selfplay.league_policy import self_play_enabled
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
        return self._league.get_pool_stats_for_status()

    def get_pool_stats_for_status(self) -> List[Dict[str, Any]]:
        return self._league.get_pool_stats_for_status()

    def get_status_file_data(self) -> Dict[str, Any]:
        return self._league.get_status_file_data()


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
        self._slot_id_to_scripted_key: Dict[int, str] = dict(
            sample_kwargs.get("slot_id_to_scripted_key") or {}
        )

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
    league_sync: LeagueSyncState,
    sampler: LeagueSamplerConfig,
    scripted_distribution: Dict[str, float],
    scripted_slot_ids: Dict[str, int],
    slot_id_to_scripted_key: Dict[int, str],
    seed: int,
    game_index: int,
    seat_index: int,
    gid: str,
    current_sd: bytes,
    ppo_opponent: Any,
    learner_agent: Any = None,
    dvd_num_identities: int = 0,
) -> Tuple[Any, int]:
    """Sample one opponent agent and league slot_id (per non-current lobby seat)."""
    frozen_pool = league_sync.frozen_pool
    sample = sample_league_opponent(
        game_rng=game_rng,
        use_self_play=use_self_play,
        sync=league_sync,
        sampler=sampler,
        scripted_distribution=scripted_distribution,
        scripted_slot_ids=scripted_slot_ids,
        slot_id_to_scripted_key=slot_id_to_scripted_key,
        frozen_pool=frozen_pool,
    )
    slot_id = int(sample.slot_id)
    if slot_id == SLOT_CURRENT:
        # DvD: a self-play seat is a *sibling* — the same live learner net under
        # a different fixed identity (true co-play, no deepcopy). The learner's
        # own current seats already span identities (per-seat mode); siblings
        # add more identity diversity at the table.
        if learner_agent is not None and dvd_num_identities > 0:
            from src.agents.ppo_dvd_agent import SiblingOpponent

            ident = game_rng.randrange(dvd_num_identities)
            return SiblingOpponent(learner_agent, identity=ident), slot_id
        opp = copy.deepcopy(ppo_opponent)
        _load_state_dict_bytes(opp, current_sd)
    elif slot_id in frozen_pool:
        opp = copy.deepcopy(ppo_opponent)
        _load_state_dict_bytes(opp, frozen_pool[slot_id])
        # A frozen DvD snapshot carries all identities; pin one explicitly so it
        # doesn't replay the random identity from its constructor.
        if dvd_num_identities > 0 and hasattr(opp, "set_episode_identity"):
            opp.set_episode_identity(game_rng.randrange(dvd_num_identities))
    else:
        key = sample.scripted_key or slot_id_to_scripted_key.get(slot_id, "random")
        opp = _create_scripted_opponent(
            key,
            seed=seed + game_index * 997 + seat_index * 17,
            game_id=gid,
        )
    opp.eval()
    if hasattr(opp, "epsilon"):
        opp.epsilon = 0.0
    return opp, slot_id


def _assign_minibg_opponent(
    opp_sampler: _MutableOpponentSampler,
    ppo_opponent: Any,
    *,
    league_sync: LeagueSyncState,
    sampler: LeagueSamplerConfig,
    current_sd: bytes,
    scripted_distribution: Dict[str, float],
    scripted_slot_ids: Dict[str, int],
    slot_id_to_scripted_key: Dict[int, str],
    game_rng: _random.Random,
    seed: int,
    game_index: int,
    gid: str,
    use_self_play: bool,
) -> Tuple[int, Optional[str]]:
    """Pick opponent kind, wire ``opp_sampler``; return league slot_id and scripted key."""
    sample = sample_league_opponent(
        game_rng=game_rng,
        use_self_play=use_self_play,
        sync=league_sync,
        sampler=sampler,
        scripted_distribution=scripted_distribution,
        scripted_slot_ids=scripted_slot_ids,
        slot_id_to_scripted_key=slot_id_to_scripted_key,
        frozen_pool=league_sync.frozen_pool,
    )
    slot_id = int(sample.slot_id)
    if slot_id == SLOT_CURRENT:
        opp = ppo_opponent
        _load_state_dict_bytes(opp, current_sd)
    elif slot_id in league_sync.frozen_pool:
        opp = ppo_opponent
        _load_state_dict_bytes(opp, league_sync.frozen_pool[slot_id])
    else:
        key = sample.scripted_key or slot_id_to_scripted_key.get(slot_id, "random")
        opp = _create_scripted_opponent(key, seed=seed + game_index * 997, game_id=gid)
    opp_sampler.set_opponent(opp)
    return slot_id, sample.scripted_key


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
    return {
        k: v for k, v in mg.items() if k not in skip and not k.startswith("dvd_")
    }


def _use_structured_collect(mg: dict) -> bool:
    gid = _game_id(mg)
    if gid not in ("minibg", "bglike"):
        return False
    return bool(mg.get("use_structured", True))


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
        # Per-worker episode_ids start from 0; offset by the running max so
        # merged sequences for BPTT (grouped by (episode_id, seat_id)) don't
        # collide across workers.
        next_ep_offset = 0
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
            # Only the global-last next_obs feeds the GAE bootstrap; buffers are
            # merged in order so the last non-empty worker's slot is the true
            # final transition. (next_legal_lists was never read — dropped.)
            if getattr(buf, "last_next_obs", None) is not None:
                out.last_next_obs = buf.last_next_obs
            out.seat_ids.extend(buf.seat_ids)
            if getattr(buf, "h_prev", None):
                out.h_prev.extend(buf.h_prev)
            ep_ids = getattr(buf, "episode_ids", None) or []
            if ep_ids:
                shifted = [int(e) + next_ep_offset for e in ep_ids]
                out.episode_ids.extend(shifted)
                next_ep_offset = max(shifted) + 1
            else:
                # Pad with a sentinel episode id so length matches `obs`. Non-recurrent
                # agents never look at episode_ids; recurrent ones populate them.
                out.episode_ids.extend([next_ep_offset] * len(buf.obs))
                next_ep_offset += 1
            # Battle-prediction-head fields. Workers either populate all 5 with
            # proper shapes (head enabled) or leave placeholder zeros (head off).
            # Either way length must match buf.obs.
            for field_name in (
                "own_board_obs",
                "opp_board_obs",
                "attack_first",
                "battle_target",
                "battle_target_valid",
            ):
                src_list = getattr(buf, field_name, None)
                if src_list is not None:
                    getattr(out, field_name).extend(src_list)
            # v8 distributional critic: per-row placement labels. An old-format
            # worker payload (no field) pads with -1 so lengths stay aligned;
            # those rows are simply masked out of the CE.
            pl = getattr(buf, "placement_label", None)
            if pl is not None and len(pl) == len(buf.obs):
                out.placement_label.extend(pl)
            else:
                out.placement_label.extend([-1] * len(buf.obs))
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


def _maybe_compile_encode_state(agent: Any) -> Any:
    """Opt-in torch.compile of the policy net forward on the collect path, to
    kill batch=1 dispatch overhead. torch.compile guards on shape/dtype and
    reads param *values* at runtime, so it stays correct across the per-round
    ``load_state_dict`` weight updates (unlike jit.freeze, which bakes weights).

    env RL_COMPILE_ENCODE (default "fwd"):
      "0"/"off"/"none" -> disabled (eager)
      "1"              -> compile encode_state only (static shapes, ~+30% collect)
      "fwd" (default)  -> compile the whole tensor forward incl. action head +
                          value (dynamic=True for variable Lmax; ~+48% collect)
    """
    import os

    mode = os.environ.get("RL_COMPILE_ENCODE", "fwd").lower()
    if mode in ("0", "off", "none", ""):
        return agent
    net = getattr(agent, "policy_net", None)
    if net is None or getattr(net, "_encode_compiled", False):
        return agent
    import torch

    if mode == "fwd" and hasattr(net, "policy_logits_value_from_tokens"):
        net.policy_logits_value_from_tokens = torch.compile(
            net.policy_logits_value_from_tokens, dynamic=True
        )
    elif hasattr(net, "encode_state"):
        net.encode_state = torch.compile(net.encode_state, dynamic=False)
    else:
        return agent
    net._encode_compiled = True
    return agent


def _maybe_compile_host_update(agent: Any) -> Any:
    """torch.compile the HOST policy forward (+backward) used by the PPO update.
    The update builds action tokens once per round with a global Lmax, so
    minibatch forward shapes are (minibatch, Lmax) and stable within a round
    (Lmax varies round-to-round -> dynamic=True). On GPU this is large-batch/
    GEMM-bound, so the win is smaller than the batch=1 collect path (measured
    host_s -25%). Default ON; set env RL_COMPILE_HOST=0/off to disable. One-time
    fwd+bwd compile cost is large (~round-0 only)."""
    import os

    if os.environ.get("RL_COMPILE_HOST", "fwd").lower() in ("0", "off", "none", ""):
        return agent
    net = getattr(agent, "policy_net", None)
    if net is None or getattr(net, "_host_compiled", False):
        return agent
    if not hasattr(net, "policy_logits_value_from_tokens"):
        return agent
    import torch

    net.policy_logits_value_from_tokens = torch.compile(
        net.policy_logits_value_from_tokens, dynamic=True
    )
    net._host_compiled = True
    return agent


def _load_dist_agent(
    ck_path: str,
    *,
    device: str,
    seed: int,
    mg: dict,
) -> Any:
    patch_build = mg.get("patch_build")
    if mg.get("dvd_network_type") in ("bglike_structured_v7", "bglike_structured_v8", "bglike_structured_v9", "bglike_structured_v10", "bglike_structured_v11"):
        from src.agents.ppo_dvd_agent import PPODvDAgent

        return _maybe_compile_encode_state(PPODvDAgent.load(
            ck_path,
            device=device,
            seed=seed,
            patch_build=patch_build,
            diversity_coef=float(mg.get("dvd_diversity_coef", 0.0)),
            diversity_ema=float(mg.get("dvd_diversity_ema", 0.1)),
            identity_tribes=mg.get("dvd_identity_tribes"),
            diversity_reward_mode=str(mg.get("dvd_reward_mode", "final")),
        ))
    if _use_structured_collect(mg):
        return _maybe_compile_encode_state(MiniBGPPOStructuredAgent.load(
            ck_path,
            device=device,
            seed=seed,
            patch_build=patch_build,
        ))
    return _maybe_compile_encode_state(PPOAgent.load(ck_path, device=device, seed=seed))


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
            and not k.startswith("dvd_")  # agent/sampler knobs, not env kwargs
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


def _make_bglike_opp_sampler(
    *,
    league_sync: LeagueSyncState,
    sampler: LeagueSamplerConfig,
    scripted_distribution: Dict[str, float],
    scripted_slot_ids: Dict[str, int],
    slot_id_to_scripted_key: Dict[int, str],
    game_rng: _random.Random,
    use_self_play: bool,
    seed: int,
    gid: str,
    current_sd: bytes,
    ppo_opponent: Any,
    learner_agent: Any = None,
    dvd_num_identities: int = 0,
) -> "_DistributedBglikeOpponentSampler":
    return _DistributedBglikeOpponentSampler(
        game_rng=game_rng,
        use_self_play=use_self_play,
        league_sync=league_sync,
        sampler=sampler,
        scripted_distribution=scripted_distribution,
        scripted_slot_ids=scripted_slot_ids,
        slot_id_to_scripted_key=slot_id_to_scripted_key,
        seed=seed,
        game_index=0,
        gid=gid,
        current_sd=current_sd,
        ppo_opponent=ppo_opponent,
        learner_agent=learner_agent,
        dvd_num_identities=int(dvd_num_identities),
    )


def _collect_until_steps_flat(
    agent: PPOAgent,
    ppo_opponent: PPOAgent,
    *,
    min_steps: int,
    mg: dict,
    current_sd: bytes,
    league_sync: LeagueSyncState,
    sampler: LeagueSamplerConfig,
    scripted_distribution: Dict[str, float],
    scripted_slot_ids: Dict[str, int],
    slot_id_to_scripted_key: Dict[int, str],
    game_rng: _random.Random,
    seed: int,
    round_idx: int,
    start_episode: int,
) -> Tuple[int, int, RolloutBuffer, List[GameRecord]]:
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
        opp_sampler = _make_bglike_opp_sampler(
            league_sync=league_sync,
            sampler=sampler,
            scripted_distribution=scripted_distribution,
            scripted_slot_ids=scripted_slot_ids,
            slot_id_to_scripted_key=slot_id_to_scripted_key,
            game_rng=game_rng,
            use_self_play=use_self_play,
            seed=seed,
            gid=gid,
            current_sd=current_sd,
            ppo_opponent=ppo_opponent,
            learner_agent=agent
            if mg.get("dvd_network_type")
            in ("bglike_structured_v7", "bglike_structured_v8", "bglike_structured_v9", "bglike_structured_v10", "bglike_structured_v11")
            else None,
            dvd_num_identities=int(mg.get("dvd_num_identities", 0)),
        )
    else:
        opp_sampler = _MutableOpponentSampler()
    env = _make_collect_env(mg, opp_sampler, seed=seed)
    env.set_learner_agent(agent)

    game_outcomes: List[GameRecord] = []
    n_games = 0
    opp_scripted_key: Optional[str] = None

    while len(agent.rollout_buffer) < min_steps:
        if gid != "bglike":
            opp_slot_id, opp_scripted_key = _assign_minibg_opponent(
                opp_sampler,
                ppo_opponent,
                league_sync=league_sync,
                sampler=sampler,
                current_sd=current_sd,
                scripted_distribution=scripted_distribution,
                scripted_slot_ids=scripted_slot_ids,
                slot_id_to_scripted_key=slot_id_to_scripted_key,
                game_rng=game_rng,
                seed=seed,
                game_index=n_games,
                gid=gid,
                use_self_play=use_self_play,
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
            step_done = bool(env.done)
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
            apply_bglike_segment_closures_after_observe(env, info)
            obs = step.obs
            if step_done:
                last_info = info
                break

        if gid == "bglike":
            last_info, lobby_record = collect_bglike_lobby_league_outcome(env, last_info)
            if lobby_record is not None:
                game_outcomes.append(lobby_record)
        else:
            score = _episode_agent_score(env, last_info, mg)
            game_outcomes.append(
                minibg_record_from_learner_score(
                    opp_slot_id,
                    normalize_agent_score(score),
                    scripted_key=opp_scripted_key,
                )
            )
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
    league_sync: LeagueSyncState,
    sampler: LeagueSamplerConfig,
    scripted_distribution: Dict[str, float],
    scripted_slot_ids: Dict[str, int],
    slot_id_to_scripted_key: Dict[int, str],
    game_rng: _random.Random,
    seed: int,
    round_idx: int,
    start_episode: int,
) -> Tuple[int, int, StructuredMiniBGRolloutBuffer, List[GameRecord]]:
    """Collect games until rollout buffer has >= min_steps transitions.

    Returns (n_games, n_steps, buffer, outcomes).
    outcomes: list of GameRecord per game or segment closure.
    """
    import torch
    torch.set_num_threads(1)
    agent.train()
    ppo_opponent.eval()
    if hasattr(ppo_opponent, "epsilon"):
        ppo_opponent.epsilon = 0.0

    gid = _game_id(mg)
    use_self_play = self_play_enabled(
        episode=round_idx,
        start_episode=start_episode,
        has_self_play_config=True,
    )
    if gid == "bglike":
        opp_sampler = _make_bglike_opp_sampler(
            league_sync=league_sync,
            sampler=sampler,
            scripted_distribution=scripted_distribution,
            scripted_slot_ids=scripted_slot_ids,
            slot_id_to_scripted_key=slot_id_to_scripted_key,
            game_rng=game_rng,
            use_self_play=use_self_play,
            seed=seed,
            gid=gid,
            current_sd=current_sd,
            ppo_opponent=ppo_opponent,
            learner_agent=agent
            if mg.get("dvd_network_type")
            in ("bglike_structured_v7", "bglike_structured_v8", "bglike_structured_v9", "bglike_structured_v10", "bglike_structured_v11")
            else None,
            dvd_num_identities=int(mg.get("dvd_num_identities", 0)),
        )
    else:
        opp_sampler = _MutableOpponentSampler()
    env = _make_collect_env(mg, opp_sampler, seed=seed)
    if gid == "bglike":
        env.set_learner_agent(agent)

    game_outcomes: List[GameRecord] = []
    n_games = 0
    opp_scripted_key: Optional[str] = None

    while len(agent.rollout_buffer) < min_steps:
        if gid != "bglike":
            opp_slot_id, opp_scripted_key = _assign_minibg_opponent(
                opp_sampler,
                ppo_opponent,
                league_sync=league_sync,
                sampler=sampler,
                current_sd=current_sd,
                scripted_distribution=scripted_distribution,
                scripted_slot_ids=scripted_slot_ids,
                slot_id_to_scripted_key=slot_id_to_scripted_key,
                game_rng=game_rng,
                seed=seed,
                game_index=n_games,
                gid=gid,
                use_self_play=use_self_play,
            )

        obs = env.reset()
        last_info: dict = {}
        while not env.done:
            legal_list = env.legal_structured_actions()
            if not legal_list:
                if gid == "minibg":
                    from src.envs.minibg.invariants import assert_shop_has_legal_actions

                    assert_shop_has_legal_actions(
                        env.state,
                        [],
                        where="distributed_trainer._collect_until_steps.agent_turn",
                    )
                break
            struct_act, board_perm, idx = agent.act_structured(
                obs, legal_list, env, deterministic=False
            )
            step = env.step_structured(struct_act, board_perm=board_perm)
            info = step.info if isinstance(step.info, dict) else {}
            step_done = bool(env.done)
            next_sl: List[Any] = (
                [] if step_done else list(env.legal_structured_actions())
            )
            transition = Transition(
                obs=obs,
                action=idx,
                reward=float(step.reward),
                next_obs=step.obs,
                terminated=step.terminated,
                truncated=step.truncated,
                info={
                    **info,
                    INFO_STRUCT_LEGAL: legal_list,
                    INFO_STRUCT_NEXT_LEGAL: next_sl,
                },
                legal_mask=None,
                next_legal_mask=None,
            )
            agent.observe(transition)
            apply_bglike_segment_closures_after_observe(env, info)
            obs = step.obs
            if step_done:
                last_info = info
                break

        if gid == "bglike":
            last_info, lobby_record = collect_bglike_lobby_league_outcome(env, last_info)
            if lobby_record is not None:
                game_outcomes.append(lobby_record)
        else:
            score = _episode_agent_score(env, last_info, mg)
            game_outcomes.append(
                minibg_record_from_learner_score(
                    opp_slot_id,
                    normalize_agent_score(score),
                    scripted_key=opp_scripted_key,
                )
            )
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
    league_sync: LeagueSyncState,
    sampler: LeagueSamplerConfig,
    scripted_distribution: Dict[str, float],
    scripted_slot_ids: Dict[str, int],
    slot_id_to_scripted_key: Dict[int, str],
    game_rng: _random.Random,
    seed: int,
    round_idx: int,
    start_episode: int,
) -> Tuple[int, int, Any, List[GameRecord]]:
    if _use_structured_collect(mg):
        return _collect_until_steps_structured(
            agent,
            ppo_opponent,
            min_steps=min_steps,
            mg=mg,
            current_sd=current_sd,
            league_sync=league_sync,
            sampler=sampler,
            scripted_distribution=scripted_distribution,
            scripted_slot_ids=scripted_slot_ids,
            slot_id_to_scripted_key=slot_id_to_scripted_key,
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
        league_sync=league_sync,
        sampler=sampler,
        scripted_distribution=scripted_distribution,
        scripted_slot_ids=scripted_slot_ids,
        slot_id_to_scripted_key=slot_id_to_scripted_key,
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
    sampler: LeagueSamplerConfig,
    scripted_distribution: Dict[str, float],
    scripted_slot_ids: Dict[str, int],
    slot_id_to_scripted_key: Dict[int, str],
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
    league_sync = LeagueSyncState(frozen_pool={}, win_rates={})

    game_rng = _random.Random(seed + worker_id * 777_777)
    round_seed_base = seed + worker_id * 10_000

    while True:
        msg = cmd_conn.recv()
        tag = msg[0]

        if tag == "load":
            _load_state_dict_bytes(agent, msg[1])
            current_sd = msg[1]

        elif tag == "sync_league":
            league_sync = msg[1]

        elif tag == "play":
            round_idx: int = msg[1]
            t0 = time.perf_counter()
            ng, nsteps, buf, outcomes = _collect_until_steps(
                agent,
                ppo_opponent,
                min_steps=min_steps,
                mg=mg,
                current_sd=current_sd,
                league_sync=league_sync,
                sampler=sampler,
                scripted_distribution=scripted_distribution,
                scripted_slot_ids=scripted_slot_ids,
                slot_id_to_scripted_key=slot_id_to_scripted_key,
                game_rng=game_rng,
                seed=round_seed_base + round_idx * 1_000_003,
                round_idx=round_idx,
                start_episode=start_episode,
            )
            play_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            payload = _buffer_to_payload(buf)
            upload_s = time.perf_counter() - t1
            dvd_snap = (
                agent.dvd_state_snapshot()
                if hasattr(agent, "dvd_state_snapshot")
                else None
            )
            cmd_conn.send(
                ("rollout", ng, nsteps, len(payload), play_s, upload_s, payload, outcomes, dvd_snap)
            )

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
    outcomes: List[GameRecord]


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
        rating: str = "ema",
        trueskill: Optional[Dict[str, Any]] = None,
        sampler_kind: str = "fractional",
        start_episode: int = 0,
        run_dir: Optional[str] = None,
        frozen_pool_checkpoints: Optional[Sequence[str]] = None,
    ) -> None:
        self._scripted_distribution: Dict[str, float] = dict(
            scripted_distribution or {"random": 1.0}
        )
        self._scripted_slot_ids = build_scripted_slot_map(self._scripted_distribution.keys())
        self._slot_id_to_scripted_key = invert_scripted_slot_map(self._scripted_slot_ids)
        self._sampler = LeagueSamplerConfig(
            kind=sampler_kind,
            current_self_fraction=current_fraction,
            past_self_fraction=past_fraction,
        )
        self._league = LeagueController(
            ema_beta=ema_beta,
            rating_kind=rating,
            trueskill=trueskill,
        )
        self._league.register_scripted_slots(self._scripted_slot_ids)
        self._league.register_meta_slot(SLOT_CURRENT)
        self._frozen_pool: Dict[int, bytes] = {}
        super().__init__(
            callbacks=callbacks,
            opponent_sampler=_DistributedOpponentSamplerAdapter(self._league),
        )
        self.agent = agent
        # Host runs the PPO update on GPU; allow TF32 for the fp32 matmuls
        # (measured ~1.3-1.5x on the large action-head GEMMs, free for RL).
        # Process-global but scoped to the host process (workers are separate).
        import torch as _torch

        if _torch.cuda.is_available():
            _torch.set_float32_matmul_precision("high")
        _maybe_compile_host_update(agent)
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
        self._max_pool_size = max_pool_size
        self._start_episode = int(start_episode)
        self._run_dir = str(run_dir) if run_dir else ""
        self._frozen_pool_checkpoints: List[str] = list(frozen_pool_checkpoints or [])
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
        if self._frozen_pool_checkpoints:
            self._seed_frozen_pool_from_checkpoints(self._frozen_pool_checkpoints)
        self._broadcast_sync_league()

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
        dvd_snaps: List[Any] = []
        for conn in self._conns:
            tag, ng, nsteps, nbytes, play_s, upload_s, payload, outcomes, dvd_snap = conn.recv()
            assert tag == "rollout"
            play_data.append((play_s, upload_s, payload, ng, nsteps, nbytes, outcomes))
            if dvd_snap is not None:
                dvd_snaps.append(dvd_snap)

        merged = _merge_buffers(
            [_payload_to_buffer(p[2], self._game_kwargs) for p in play_data],
            self._game_kwargs,
        )
        all_outcomes: List[GameRecord] = [
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
        # DvD: merge workers' φ / placement EMA / novelty accumulators into the
        # host agent so ``update()`` emits non-zero dvd_* metrics (φ lives on
        # the workers that collect rollouts; the host only runs the PPO update).
        if hasattr(self.agent, "merge_dvd_state") and dvd_snaps:
            self.agent.merge_dvd_state(dvd_snaps)
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
        slot_id = self._league.add_frozen_bytes(sd_bytes, copy_current_mu=True)
        self._frozen_pool[slot_id] = sd_bytes
        self._league.evict_worst(self._max_pool_size)
        live = {s.slot_id for s in self._league.frozen_slots()}
        self._frozen_pool = {k: v for k, v in self._frozen_pool.items() if k in live}
        self._broadcast_sync_league()

    def _seed_frozen_pool_from_checkpoints(self, paths: Sequence[str]) -> None:
        """Repopulate the frozen self-play pool from past-self checkpoints.

        Each periodic checkpoint corresponds to one historical freeze (the pool
        is filled from the same weights right after each checkpoint save), so a
        run's ``checkpoints/`` dir IS its past-self pool — just not persisted as
        one. We extract each file's ``policy_state_dict``, re-serialize it into
        the exact bytes format workers expect (``torch.save(state_dict)``), and
        register it as a frozen slot. Ratings start at the default prior
        (``copy_current_mu=False``): the restored agents earn real ratings as
        they play, protected from premature eviction by the grace period.
        Eviction (ordered by mu then episode) keeps the most-recent
        ``max_frozen_agents``.
        """
        import torch

        loaded = 0
        for p in paths:
            path = Path(p)
            if not path.is_file():
                print(f"[dist_ppo] seed frozen: skip missing {path}", flush=True)
                continue
            try:
                ck = torch.load(str(path), map_location="cpu", weights_only=False)
                sd = ck.get("policy_state_dict") if isinstance(ck, dict) else None
                if sd is None:
                    print(f"[dist_ppo] seed frozen: no policy_state_dict in {path.name}", flush=True)
                    continue
                bio = BytesIO()
                torch.save({k: v.detach().cpu() for k, v in sd.items()}, bio)
                sd_bytes = bio.getvalue()
                episode = int(ck.get("step_count", 0)) if isinstance(ck, dict) else 0
                slot_id = self._league.add_frozen_bytes(
                    sd_bytes, episode=episode, copy_current_mu=False
                )
                self._frozen_pool[slot_id] = sd_bytes
                loaded += 1
            except Exception as exc:  # keep startup resilient to a bad file
                print(f"[dist_ppo] seed frozen: failed {path.name}: {exc}", flush=True)

        # Cap to max_frozen_agents (grace fallback keeps the most-recent by episode).
        self._league.evict_worst(self._max_pool_size)
        live = {s.slot_id for s in self._league.frozen_slots()}
        self._frozen_pool = {k: v for k, v in self._frozen_pool.items() if k in live}
        print(
            f"[dist_ppo] seeded frozen pool: loaded {loaded}, kept {len(self._frozen_pool)} "
            f"(cap {self._max_pool_size})",
            flush=True,
        )

    def _broadcast_sync_league(self) -> None:
        snap = self._league.sync_snapshot()
        league_sync = LeagueSyncState(
            frozen_pool=dict(self._frozen_pool),
            win_rates=dict(snap.win_rates),
            rating_kind=snap.rating_kind,
            trueskill=dict(snap.trueskill),
        )
        for conn in self._conns:
            conn.send(("sync_league", league_sync))

    @staticmethod
    def _worker_mp_context():
        """Prefer ``forkserver`` so workers fork from a clean server process
        instead of re-importing torch in every worker.

        Under ``spawn`` each worker is a fresh interpreter that re-runs
        ``import torch`` — ~480 MB private RSS apiece (measured). ``forkserver``
        starts one minimal server process, preloads torch/numpy there once, and
        forks workers from it so those pages are shared copy-on-write. The
        server is its own process (not a fork of the CUDA-initialized main), and
        ``import torch`` doesn't create a CUDA context, so this stays
        CUDA-safe even though the learner holds a cuda model. Falls back to
        ``spawn`` where forkserver is unavailable (e.g. Windows)."""
        try:
            ctx = mp.get_context("forkserver")
            ctx.set_forkserver_preload(["torch", "numpy"])
            return ctx
        except (ValueError, AttributeError):
            return mp.get_context("spawn")

    def _spawn_workers(self) -> None:
        per_worker = math.ceil(self._rollout_steps / self._workers)
        ctx = self._worker_mp_context()
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
                    self._sampler,
                    self._scripted_distribution,
                    self._scripted_slot_ids,
                    self._slot_id_to_scripted_key,
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

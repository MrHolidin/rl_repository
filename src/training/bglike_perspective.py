"""AgentPerspectiveEnv factory for 8-player BGLike lobbies.

Each learner seat is an independent decision segment (shared weights, no merged
credit assignment). Segment boundaries: learner elimination or switch to another
learner seat. League outcomes are recorded per segment closure, not averaged.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agents.base_agent import BaseAgent
from src.envs.base import StepResult
from src.envs.bglike.action_map import NUM_ENV_ACTIONS
from src.envs.bglike.actions import NUM_PLAYERS
from src.envs.bglike.obs import OBS_DIM
from src.envs.bglike.lobby_env import BGLobbyMultiCurrentEnv, make_bglike_training_env
from src.envs.bglike.placement import placement_reward, placement_score
from src.envs.reward_config import RewardConfig
from src.training.agent_perspective_env import AgentPerspectiveEnv, ShapingFn
from src.training.opponent_sampler import OpponentSampler


def make_bglike_shaping_fn(scale: float) -> Optional[ShapingFn]:
    """Shaping from ``info['battle_signed_seat']`` (normalized health lost last combat)."""
    if not scale:
        return None

    def _fn(info: Dict[str, Any], _agent_token: int) -> float:
        v = info.get("battle_signed_seat")
        if v is None:
            return 0.0
        return float(scale) * float(v)

    return _fn


def read_opponent_slot_by_seat(opponent_sampler: Any) -> Dict[int, int]:
    """Return opponent league slot ids keyed by lobby seat (distributed + pool samplers)."""
    for attr in ("_slot_by_seat", "_episode_slot_by_seat"):
        mapping = getattr(opponent_sampler, attr, None)
        if isinstance(mapping, dict) and mapping:
            return {int(k): int(v) for k, v in mapping.items()}
    return {}


def league_outcomes_for_segment_closures(
    closures: Sequence[Dict[str, Any]],
    slot_by_seat: Dict[int, int],
) -> List[Tuple[int, float]]:
    """One league update per unique opponent slot per finished learner segment."""
    if not closures or not slot_by_seat:
        return []
    unique_slots = sorted(set(int(v) for v in slot_by_seat.values()))
    out: List[Tuple[int, float]] = []
    for item in closures:
        place = item.get("placement")
        if place is None:
            continue
        score = placement_score(int(place))
        for slot_id in unique_slots:
            out.append((slot_id, score))
    return out


def record_league_outcomes_to_sampler(
    opponent_sampler: Any,
    outcomes: Sequence[Tuple[int, float]],
) -> None:
    if not outcomes or opponent_sampler is None:
        return
    pool = getattr(opponent_sampler, "opponent_pool", None)
    if pool is not None and hasattr(pool, "record_outcome_for_slot"):
        for slot_id, score in outcomes:
            pool.record_outcome_for_slot(int(slot_id), score)  # type: ignore[operator]


class BGLikeAgentPerspectiveEnv(AgentPerspectiveEnv):
    """Reuses ``AgentPerspectiveEnv`` drain/placement; per-seat trajectory segments."""

    def __init__(
        self,
        base_env: BGLobbyMultiCurrentEnv,
        opponent_sampler: OpponentSampler,
        *,
        num_current_seats: Optional[int] = None,
        rng: Optional[random.Random] = None,
        shaping_fn: Optional[ShapingFn] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__(
            base_env,
            opponent_sampler,
            agent_first_probability=0.5,
            rng=rng,
            shaping_fn=shaping_fn,
            reward_config=reward_config,
        )
        self._bg_base = base_env
        self._num_current = num_current_seats
        self._learner: Optional[BaseAgent] = None
        self._opponent_slot_by_seat: Dict[int, int] = {}

    @property
    def supports_seat_segments(self) -> bool:
        return bool(getattr(self._bg_base, "uses_seat_segments", False))

    def set_learner_agent(self, agent: BaseAgent) -> None:
        self._learner = agent

    def reset(self):
        for _ in range(self.MAX_RESET_RETRIES):
            episode_seed = self._rng.randrange(self._SEED_SPACE)
            n = self._num_current or len(self._bg_base.current_seats)
            n = max(1, min(int(n), NUM_PLAYERS))
            self._bg_base.set_current_seats(
                tuple(sorted(self._rng.sample(range(NUM_PLAYERS), n)))
            )

            self.opponent_sampler.prepare(self._episode_index)
            if self._learner is None:
                raise RuntimeError("set_learner_agent() required before reset")
            current = set(self._bg_base.current_seats)
            opponent_seats = [s for s in range(NUM_PLAYERS) if s not in current]
            opponents_by_seat = self.opponent_sampler.sample_for_seats(opponent_seats)
            self._opponent_slot_by_seat = read_opponent_slot_by_seat(self.opponent_sampler)
            self._bg_base._opponent_slot_by_seat = dict(self._opponent_slot_by_seat)
            self._bg_base.set_agents(self._learner, opponents_by_seat)
            seen_env: set[int] = set()
            for opp in opponents_by_seat.values():
                oid = id(opp)
                if oid in seen_env:
                    continue
                seen_env.add(oid)
                if hasattr(opp, "set_env"):
                    opp.set_env(self._bg_base)
                if hasattr(opp, "epsilon"):
                    setattr(opp, "epsilon", 0.0)

            obs = self._bg_base.reset(seed=episode_seed)
            if self._bg_base.done:
                self._episode_index += 1
                continue

            self._agent_token = 1
            self._done = False
            return obs
        raise RuntimeError(
            "BGLikeAgentPerspectiveEnv: could not obtain a non-terminal initial state."
        )

    def apply_pending_segment_closures(self, info: Dict[str, Any]) -> None:
        """Close learner segments after ``observe()`` (all seats, including acting)."""
        closures = info.get("segment_closures") or ()
        if not closures:
            return
        learner = self._learner
        if learner is None:
            return
        close = getattr(learner, "close_segment", None)
        if close is None:
            return
        for item in closures:
            seat = int(item["seat"])
            rew = float(item["placement_reward"])
            if not close(seat, rew):
                raise AssertionError(
                    "segment_closures: seat "
                    f"{seat} has no prior rollout step to close "
                    f"(placement={item.get('placement')}, placement_reward={rew}). "
                    "Current-seat shop turns must go through the learner act/observe path."
                )

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")

        base_step = self._bg_base.step(action)
        info = dict(base_step.info) if isinstance(base_step.info, dict) else {}

        lobby_done = bool(self._bg_base.done or info.get("lobby_episode_done"))
        if lobby_done:
            reward = self._final_reward_for_agent(info)
        else:
            reward = self._reward_in_agent_perspective(base_step, agent_acted=True)
        if lobby_done:
            self._done = True

        return StepResult(
            obs=base_step.obs,
            reward=reward,
            terminated=lobby_done,
            truncated=False,
            info=info,
        )

    def step_structured(
        self,
        action,
        *,
        board_perm=None,
    ) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")

        base_step = self._bg_base.step_structured(action, board_perm=board_perm)
        info = dict(base_step.info) if isinstance(base_step.info, dict) else {}

        lobby_done = bool(self._bg_base.done or info.get("lobby_episode_done"))
        if lobby_done:
            reward = self._final_reward_for_agent(info)
        else:
            reward = self._reward_in_agent_perspective(base_step, agent_acted=True)
        if lobby_done:
            self._done = True

        return StepResult(
            obs=base_step.obs,
            reward=reward,
            terminated=lobby_done,
            truncated=False,
            info=info,
        )

    def _final_reward_for_agent(self, info: Dict[str, Any]) -> float:
        if isinstance(info, dict) and info.get("placement_reward") is not None:
            return float(info["placement_reward"])
        return super()._final_reward_for_agent(info)

    def _battle_shaping_for_acting_seat(self, info: Dict[str, Any]) -> float:
        if self.shaping_fn is None:
            return 0.0
        seat = info.get("acting_seat")
        if seat is None:
            return 0.0
        signed = -self._bg_base.lobby.last_battle_signed(int(seat))
        return float(
            self.shaping_fn({"battle_signed_seat": signed}, self._agent_token)
        )

    def _reward_in_agent_perspective(self, step, agent_acted: bool) -> float:
        info = step.info if isinstance(step.info, dict) else {}
        if not info.get("combat_advanced"):
            return 0.0
        return self._battle_shaping_for_acting_seat(info)

    def notify_episode_end(self, info: Dict[str, Any]) -> None:
        if self.opponent_sampler is None:
            self._episode_index += 1
            return

        # League outcomes are recorded per segment closure only (see
        # apply_bglike_segment_closures_after_observe); avoid duplicate/wrong-slot
        # updates via record_episode_outcome(_last_sample_slot_id).
        placements = info.get("placements_current") or {}
        self.opponent_sampler.on_episode_end(
            self._episode_index,
            {
                "agent_token": self._agent_token,
                "placements_current": placements,
                "info": info,
                "skip_league_record": True,
            },
        )
        self._episode_index += 1


def make_bglike_agent_perspective_env(
    opponent_sampler: OpponentSampler,
    *,
    current_seats: Optional[Sequence[int]] = None,
    num_current_seats: Optional[int] = None,
    seed: Optional[int] = None,
    shaping_fn: Optional[ShapingFn] = None,
    reward_config: Optional[RewardConfig] = None,
    rng: Optional[random.Random] = None,
    **lobby_kwargs: Any,
) -> BGLikeAgentPerspectiveEnv:
    base = make_bglike_training_env(
        current_seats=current_seats or (0,),
        seed=seed,
        reward_config=reward_config,
        **lobby_kwargs,
    )
    return BGLikeAgentPerspectiveEnv(
        base,
        opponent_sampler,
        num_current_seats=num_current_seats,
        rng=rng,
        shaping_fn=shaping_fn,
        reward_config=reward_config,
    )


__all__ = [
    "BGLikeAgentPerspectiveEnv",
    "apply_bglike_segment_closures_after_observe",
    "league_outcomes_for_segment_closures",
    "make_bglike_agent_perspective_env",
    "make_bglike_shaping_fn",
    "read_opponent_slot_by_seat",
    "record_league_outcomes_to_sampler",
]


def apply_bglike_segment_closures_after_observe(
    env: Any, info: Any
) -> List[Tuple[int, float]]:
    """After ``observe()``: close learner segments and record per-segment league outcomes."""
    if not isinstance(info, dict):
        return []
    apply = getattr(env, "apply_pending_segment_closures", None)
    if apply is not None:
        apply(info)
    closures = info.get("segment_closures") or ()
    slot_by_seat = getattr(env, "_opponent_slot_by_seat", None) or {}
    if not slot_by_seat and hasattr(env, "opponent_sampler"):
        slot_by_seat = read_opponent_slot_by_seat(env.opponent_sampler)
    outcomes = league_outcomes_for_segment_closures(closures, slot_by_seat)
    if outcomes and hasattr(env, "opponent_sampler"):
        record_league_outcomes_to_sampler(env.opponent_sampler, outcomes)
    return outcomes

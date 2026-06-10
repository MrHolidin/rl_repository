"""AgentPerspectiveEnv factory for 8-player BGLike lobbies.

Each learner seat is an independent decision segment (shared weights, no merged
credit assignment). Segment boundaries: learner elimination or switch to another
learner seat. League outcomes are recorded once per lobby at episode end.
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
from src.envs.bglike.placement import placement_reward
from src.training.selfplay.game_record import (
    GameRecord,
    game_record_for_lobby_end,
)
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


def submit_game_records_to_sampler(
    opponent_sampler: Any,
    records: Sequence[GameRecord],
) -> None:
    if not records or opponent_sampler is None:
        return
    pool = getattr(opponent_sampler, "opponent_pool", None)
    if pool is not None and hasattr(pool, "submit"):
        for record in records:
            pool.submit(record)  # type: ignore[operator]


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
        percent_high_game: float = 0.0,
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
        self._lobby_league_recorded: bool = False
        # Curriculum: fraction of games started in high mode. The decision lives
        # here in the training harness (with this env's RNG), not in the game.
        self._percent_high_game = max(0.0, min(1.0, float(percent_high_game)))

    @property
    def supports_seat_segments(self) -> bool:
        return bool(getattr(self._bg_base, "uses_seat_segments", False))

    def set_learner_agent(self, agent: BaseAgent) -> None:
        self._learner = agent

    def set_high_mode(self, flag: bool) -> None:
        """Forward an explicit per-game high-mode decision to the lobby."""
        self._bg_base.set_high_mode(flag)

    def reset(self):
        for _ in range(self.MAX_RESET_RETRIES):
            episode_seed = self._rng.randrange(self._SEED_SPACE)
            # Roll this game's high-mode flag (trainer-side decision, this env's
            # RNG) and push it to the lobby before it builds the initial state.
            if self._percent_high_game > 0.0:
                self._bg_base.set_high_mode(
                    self._rng.random() < self._percent_high_game
                )
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
            self._lobby_league_recorded = False
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

    def finish_lobby_to_end(self) -> Dict[str, Any]:
        """Auto-play opponents until lobby completes; apply pending segment closures."""
        info = self._bg_base.finish_lobby_to_end()
        self.apply_pending_segment_closures(info)
        self._done = bool(self._bg_base.done)
        return info

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")

        base_step = self._bg_base.step(action)
        info = dict(base_step.info) if isinstance(base_step.info, dict) else {}

        lobby_done = bool(self._bg_base.done)
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

        # On combat-resolution steps the env now carries per-seat snapshots of
        # the just-resolved combat (own/opp boards in their final pre-combat
        # order + signed uncapped damage + attack_first). The PPO agent reads
        # this in observe() and back-fills the last FINISH-row for each seat
        # with the battle-prediction head's target. Computed for *all* training
        # seats (alive at start of combat), so we don't miss any battles.
        if info.get("combat_advanced"):
            info["battle_data_per_seat"] = self._collect_battle_data_per_seat()

        lobby_done = bool(self._bg_base.done)
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

    def _collect_battle_data_per_seat(self) -> Dict[int, Dict[str, Any]]:
        """Snapshot per-seat battle data on a combat-resolution step.

        Returns a dict from training-seat to dict with keys
        ``own_board_obs``, ``opp_board_obs`` (np.ndarray (BOARD_SIZE, SLOT_DIM)),
        ``attack_first`` (float 0/1), ``damage_signed_uncapped`` (float).
        """
        from src.envs.bglike.obs import encode_board_minions

        out: Dict[int, Dict[str, Any]] = {}
        state = self._bg_base.state
        lobby = self._bg_base.lobby
        patch = lobby._game._patch
        card_id_to_dense = patch.card_id_to_dense
        for seat in self._bg_base.current_seats:
            player = state.players[seat]
            if not player.last_battle_snapshots:
                continue
            snap = player.last_battle_snapshots[0]
            out[int(seat)] = {
                "own_board_obs": encode_board_minions(
                    snap.own_board, card_id_to_dense=card_id_to_dense
                ),
                "opp_board_obs": encode_board_minions(
                    snap.opp_board, card_id_to_dense=card_id_to_dense
                ),
                "attack_first": float(player.last_attack_first),
                "damage_signed_uncapped": float(player.last_battle_raw_signed),
            }
        return out

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
        # Reliable per-lobby boundary for the learner: in bglike the learner
        # never sees a transition with terminated/truncated=True (it finishes
        # via segment closures / lobby end), so any per-episode agent state must
        # be reset here, not on a transition flag. DvD uses this to hand out a
        # fresh, collision-free seat→identity assignment each lobby.
        learner = self._learner
        if learner is not None:
            hook = getattr(learner, "on_episode_boundary", None)
            if hook is not None:
                hook()

        if self.opponent_sampler is None:
            self._episode_index += 1
            return

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
    percent_high_game: float = 0.0,
    **lobby_kwargs: Any,
) -> BGLikeAgentPerspectiveEnv:
    # ``percent_high_game`` is consumed by the perspective wrapper (trainer-side
    # curriculum), never forwarded to the inner lobby/game constructors.
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
        percent_high_game=percent_high_game,
    )


__all__ = [
    "BGLikeAgentPerspectiveEnv",
    "apply_bglike_segment_closures_after_observe",
    "collect_bglike_lobby_league_outcome",
    "finalize_bglike_lobby_league_record",
    "make_bglike_agent_perspective_env",
    "make_bglike_shaping_fn",
    "read_opponent_slot_by_seat",
    "submit_game_records_to_sampler",
]


def apply_bglike_segment_closures_after_observe(
    env: Any, info: Any
) -> None:
    """After ``observe()``: close learner segments (training only, no league update)."""
    if not isinstance(info, dict):
        return
    apply = getattr(env, "apply_pending_segment_closures", None)
    if apply is not None:
        apply(info)


def finalize_bglike_lobby_league_record(env: Any, info: Any) -> Optional[GameRecord]:
    """Build one lobby-end ``GameRecord`` after the lobby has finished."""
    if not isinstance(info, dict):
        return None
    placements_full = info.get("placements_full") or {}
    slot_by_seat = getattr(env, "_opponent_slot_by_seat", None) or {}
    if not slot_by_seat and hasattr(env, "opponent_sampler"):
        slot_by_seat = read_opponent_slot_by_seat(env.opponent_sampler)
    current_seats = info.get("current_seats")
    if current_seats is None:
        bg = getattr(env, "_bg_base", None)
        current_seats = getattr(bg, "current_seats", ()) if bg is not None else ()
    key_map = getattr(env, "_slot_id_to_scripted_key", None)
    if not key_map and hasattr(env, "opponent_sampler"):
        key_map = getattr(env.opponent_sampler, "_slot_id_to_scripted_key", None)
    record = game_record_for_lobby_end(
        current_seats=current_seats,
        slot_by_seat=slot_by_seat,
        placements_full=placements_full,
        slot_id_to_scripted_key=key_map or {},
    )
    if record is not None and hasattr(env, "opponent_sampler"):
        submit_game_records_to_sampler(env.opponent_sampler, records=[record])
    return record


def collect_bglike_lobby_league_outcome(
    env: Any, last_info: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[GameRecord]]:
    """Finish lobby if needed and return one league record for the full lobby."""
    if getattr(env, "_lobby_league_recorded", False):
        return dict(last_info or {}), None
    info = dict(last_info or {})
    if not getattr(env, "done", False):
        finish = getattr(env, "finish_lobby_to_end", None)
        if finish is not None:
            info = finish()
    if not getattr(env, "done", False):
        return info, None
    record = finalize_bglike_lobby_league_record(env, info)
    if record is not None:
        env._lobby_league_recorded = True
    return info, record

"""8-player lobby environment with multi-seat learned controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.base import SingleAgentEnv, StepResult
from src.envs.reward_config import RewardConfig

from .action_map import (
    A_APPLY_EFFECT_SKIP,
    A_PLACE_BASE,
    A_SWAP_BOARD_0,
    A_TARGET_BOARD_BASE,
    NUM_ENV_ACTIONS,
    NUM_SWAP_ADJ,
    is_apply_effect_skip,
    is_place,
    is_swap_board,
    is_target_board,
    place_slot,
    swap_adj_index_from_env_action,
    target_board_slot,
)
from .actions import NUM_PLAYERS, STARTING_HEALTH
from .game import BGLikeGame
from .rl_placement import (
    RlPlacePlan,
    commit_bglike_rl_place,
    commit_bglike_simple_place,
    open_rl_place_plan,
)
from .obs import OBS_DIM, build_observation
from .seat_config import build_training_lobby_configs
from .placement import (
    is_seat_eliminated,
    is_seat_finished,
    placement_for_seat,
    placement_reward,
    placement_reward_for_seat,
)
from .seat_config import SeatConfig, SeatKind, default_random_lobby, lobby_from_learned_seats
from .state import BGLikeState, PlayerPhase

INVALID_ACTION_REWARD = -1.0
MAX_DRAIN_STEPS = 1_000
MAX_DRAIN_TO_ACTING_STEPS = 1_000


@dataclass
class LobbyStepInfo:
    acting_seat: int
    eliminated_seats: Tuple[int, ...] = ()
    lobby_done: bool = False
    placements: Dict[int, int] = field(default_factory=dict)


@dataclass
class SeatTransition:
    seat: int
    obs: np.ndarray
    action: int
    reward: float
    terminated: bool
    info: Dict[str, Any]


@dataclass
class LobbyEpisodeResult:
    outcomes: Dict[int, LobbyStepInfo]
    transitions: Dict[int, List[SeatTransition]]


class BGLobbyEnv:
    """Full 8-player lobby simulator with per-seat controllers."""

    def __init__(
        self,
        seat_configs: Sequence[SeatConfig],
        *,
        learned_seats: Sequence[int],
        training_seats: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        shop_excluded_race=None,
        shop_full_tribes: bool = False,
    ) -> None:
        if len(seat_configs) != NUM_PLAYERS:
            raise ValueError(f"expected {NUM_PLAYERS} seat configs, got {len(seat_configs)}")
        self._seat_configs: List[SeatConfig] = list(seat_configs)
        self._learned_seats: Set[int] = frozenset(learned_seats)
        if not self._learned_seats:
            raise ValueError("learned_seats must be non-empty")
        for s in self._learned_seats:
            if self._seat_configs[s].kind != SeatKind.LEARNED:
                raise ValueError(f"seat {s} marked learned but config is {self._seat_configs[s].kind}")
        self._training_seats: Set[int] = frozenset(
            training_seats if training_seats is not None else learned_seats
        )
        if not self._training_seats <= self._learned_seats:
            raise ValueError("training_seats must be subset of learned_seats")
        self._game = BGLikeGame(
            seed=seed,
            shop_excluded_race=shop_excluded_race,
            shop_full_tribes=shop_full_tribes,
        )
        self._state: Optional[BGLikeState] = None
        self._finished_training: Set[int] = set()
        self._last_battle_signed: Dict[int, float] = {i: 0.0 for i in range(NUM_PLAYERS)}
        self._rl_pending: Dict[int, RlPlacePlan] = {}
        self._rl_place_budget_pending: Set[int] = set()
        self._heuristic_control_seat: Optional[int] = None
        self._replay_sink: Any = None
        self._replay_record_auto: bool = True
        self._replay_episode: int = -1
        self._replay_frame: int = 0

    @property
    def state(self) -> BGLikeState:
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state

    @property
    def game(self) -> BGLikeGame:
        return self._game

    @property
    def learned_seats(self) -> frozenset[int]:
        return self._learned_seats

    @property
    def training_seats(self) -> frozenset[int]:
        return self._training_seats

    def reset(self, seed: Optional[int] = None) -> BGLikeState:
        if seed is not None:
            self._game = BGLikeGame(
                seed=seed,
                shop_excluded_race=self._game._shop_excluded_race_fixed,
                shop_full_tribes=self._game._shop_full_tribes,
            )
        self._state = self._game.initial_state()
        self._finished_training = set()
        self._last_battle_signed = {i: 0.0 for i in range(NUM_PLAYERS)}
        self._rl_pending = {}
        self._rl_place_budget_pending = set()
        self._heuristic_control_seat = None
        if self._replay_sink is not None:
            self._replay_episode += 1
            self._replay_sink.episode_break(self._replay_episode)
            self._replay_frame = 0
        return self._state

    @property
    def lobby_done(self) -> bool:
        assert self._state is not None
        return self._state.done

    @property
    def episode_done(self) -> bool:
        """True when lobby ended or all training seats have finished."""
        assert self._state is not None
        return self.lobby_done or self._all_training_finished()

    def _all_training_finished(self) -> bool:
        assert self._state is not None
        return all(
            s in self._finished_training or is_seat_finished(self._state, s)
            for s in self._training_seats
        )

    def active_training_seats(self) -> Set[int]:
        assert self._state is not None
        return {
            s
            for s in self._training_seats
            if s not in self._finished_training and not is_seat_finished(self._state, s)
        }

    def current_seat(self) -> int:
        return self.state.current_player_index

    def _raise_drain_stall(self, *, where: str, cur: Optional[int] = None) -> None:
        s = self.state
        seat = s.current_player_index if cur is None else cur
        phase = s.players[seat].phase.name
        raise RuntimeError(
            f"{where}: drain iteration made no progress while lobby not done "
            f"(cur={seat}, phase={phase}, alive={s.alive}, "
            f"combat_round={s.combat_round}, round={s.round_number})"
        )

    def _raise_drain_exceeded_cap(self, *, where: str, steps: int, cap: int) -> None:
        s = self.state
        cur = s.current_player_index
        phase = s.players[cur].phase.name
        raise RuntimeError(
            f"{where}: drain exceeded {cap} steps without finishing "
            f"(steps={steps}, cur={cur}, phase={phase}, alive={s.alive}, "
            f"combat_round={s.combat_round}, round={s.round_number}, done={s.done})"
        )

    def _seat_can_act(self, seat: int) -> bool:
        s = self.state
        if is_seat_finished(s, seat) or seat not in s.alive:
            return False
        if s.current_player_index != seat:
            return False
        p = s.players[seat]
        return p.phase == PlayerPhase.SHOP or p.pending_choice is not None

    def rl_pending_for_seat(self, seat: int) -> Optional[RlPlacePlan]:
        return self._rl_pending.get(seat)

    def obs_for_seat(self, seat: int) -> np.ndarray:
        return build_observation(
            self.state,
            seat,
            self._last_battle_signed.get(seat, 0.0),
            is_my_turn=self._seat_can_act(seat),
            rl_pending=self._rl_pending.get(seat),
        )

    def last_battle_signed(self, seat: int) -> float:
        return float(self._last_battle_signed.get(seat, 0.0))

    def legal_mask_for_seat(self, seat: int) -> np.ndarray:
        mask = np.zeros(NUM_ENV_ACTIONS, dtype=bool)
        if not self._seat_can_act(seat):
            return mask
        plan = self._rl_pending.get(seat)
        if plan is not None:
            player = self.state.players[seat]
            for i in plan.eligible_on_board_live(player.board):
                mask[A_TARGET_BOARD_BASE + i] = True
            if plan.can_skip_second_adjacent():
                mask[A_APPLY_EFFECT_SKIP] = True
            return mask
        player = self.state.players[seat]
        if player.phase == PlayerPhase.SHOP and player.pending_choice is None:
            k = len(player.board)
            for i in range(NUM_SWAP_ADJ):
                if i + 1 < k:
                    mask[A_SWAP_BOARD_0 + i] = True
        for a in self._game.legal_actions(self.state):
            mask[int(a)] = True
        return mask

    def _mark_finished_training(self, seat: int) -> Optional[float]:
        if seat not in self._training_seats or seat in self._finished_training:
            return None
        if not is_seat_finished(self.state, seat):
            return None
        self._finished_training.add(seat)
        return placement_reward_for_seat(self.state, seat)

    def _finish_rl_place_after_effects(self, seat: int) -> None:
        assert self._state is not None
        player = self._state.players[seat]
        ref = player.placed_minion_pending_after
        if (
            ref is not None
            and ref in player.board
            and player.pending_choice is None
        ):
            self._game._shop_triggers.fire_after_friendly_minion_placed(player, ref)
        player.placed_minion_board_index = None
        player.placed_minion_pending_after = None
        from src.bg_recruitment.triples import flush_triple_reward_queue_if_idle

        flush_triple_reward_queue_if_idle(
            player, self._state.shop_excluded_race, rng=self._game._rng
        )

    def _try_begin_rl_place(self, seat: int, hand_slot: int) -> bool:
        assert self._state is not None
        if seat in self._rl_pending:
            return False
        player = self._state.players[seat]
        from .actions import Action as GameAction

        legal = {int(a) for a in self._game.legal_actions(self._state)}
        if int(GameAction.PLACE_HAND_0) + hand_slot not in legal:
            return False
        if player.hand[hand_slot] is None:
            return False
        plan = open_rl_place_plan(player, hand_slot)
        if plan is None:
            self._state = commit_bglike_simple_place(
                self._state, seat, hand_slot, self._game
            )
            self._finish_rl_place_after_effects(seat)
            player = self._state.players[seat]
            player.shop_actions_used += 1
            return True
        self._rl_pending[seat] = plan
        self._rl_place_budget_pending.add(seat)
        return True

    def _apply_rl_effect_pick(self, seat: int, board_idx: int) -> None:
        plan = self._rl_pending.get(seat)
        if plan is None:
            raise ValueError(f"seat {seat} has no rl_pending plan")
        player = self._state.players[seat]
        plan.record_pick_live(player.board, board_idx)
        if plan.is_complete():
            self._commit_rl_place(seat)

    def _apply_rl_effect_skip(self, seat: int) -> None:
        plan = self._rl_pending.get(seat)
        if plan is None:
            raise ValueError(f"seat {seat} has no rl_pending plan")
        plan.record_skip_second()
        if plan.is_complete():
            self._commit_rl_place(seat)

    def _commit_rl_place(self, seat: int) -> None:
        assert self._state is not None
        plan = self._rl_pending.pop(seat)
        self._state = commit_bglike_rl_place(self._state, seat, plan, self._game)
        self._finish_rl_place_after_effects(seat)
        if seat in self._rl_place_budget_pending:
            self._state.players[seat].shop_actions_used += 1
            self._rl_place_budget_pending.discard(seat)

    def close_replay(self) -> None:
        if self._replay_sink is not None:
            self._replay_sink.close()
            self._replay_sink = None

    def _apply_action(self, seat: int, action: int, *, auto: bool = False) -> LobbyStepInfo:
        s = self.state
        if not self._seat_can_act(seat):
            raise ValueError(f"seat {seat} cannot act now (current={s.current_player_index})")
        eliminated_before = {snap.seat for snap in s.eliminated}
        prev_combat_round = s.combat_round
        prev_hp = tuple(p.health for p in s.players)
        action_int = int(action)
        if seat in self._rl_pending:
            if is_apply_effect_skip(action_int):
                plan = self._rl_pending[seat]
                if not plan.can_skip_second_adjacent():
                    raise ValueError("APPLY_EFFECT_SKIP not allowed")
                self._apply_rl_effect_skip(seat)
            elif is_target_board(action_int):
                self._apply_rl_effect_pick(seat, target_board_slot(action_int))
            else:
                raise ValueError(
                    f"Expected TARGET_BOARD or APPLY_EFFECT_SKIP during rl_pending, got {action_int}"
                )
        elif is_swap_board(action_int):
            self._state = self._game.swap_board_adjacent(
                s, seat, swap_adj_index_from_env_action(action_int)
            )
        elif is_place(action_int) and seat == s.current_player_index:
            if not self._try_begin_rl_place(seat, place_slot(action_int)):
                raise ValueError(f"PLACE hand slot {place_slot(action_int)} failed")
        else:
            self._state = self._game.apply_action(s, action_int)
        if self.state.combat_round > prev_combat_round:
            for i in range(NUM_PLAYERS):
                delta = prev_hp[i] - self.state.players[i].health
                self._last_battle_signed[i] = float(delta) / float(STARTING_HEALTH)
        newly = tuple(
            snap.seat
            for snap in self.state.eliminated
            if snap.seat not in eliminated_before
        )
        placements: Dict[int, int] = {}
        for el in newly:
            if el in self._training_seats:
                placements[el] = placement_for_seat(self.state, el)
        info = LobbyStepInfo(
            acting_seat=seat,
            eliminated_seats=newly,
            lobby_done=self.state.done,
            placements=placements,
        )
        if self._replay_sink is not None and (self._replay_record_auto or not auto):
            from .replay import lobby_step_info_to_replay_info

            self._replay_frame += 1
            self._replay_sink.frame(
                episode=self._replay_episode,
                frame=self._replay_frame,
                seat=seat,
                action=action_int,
                auto=auto,
                illegal=False,
                state=self.state,
                info=lobby_step_info_to_replay_info(
                    info,
                    combat_advanced=self.state.combat_round > prev_combat_round,
                ),
            )
        return info

    def _choose_action(self, seat: int, *, deterministic: bool = False) -> int:
        cfg = self._seat_configs[seat]
        obs = self.obs_for_seat(seat)
        mask = self.legal_mask_for_seat(seat)
        if not bool(mask.any()):
            raise RuntimeError(f"no legal actions for seat {seat}")
        self._heuristic_control_seat = seat
        try:
            if cfg.kind == SeatKind.LEARNED:
                assert cfg.agent is not None
                return int(cfg.agent.act(obs, legal_mask=mask, deterministic=deterministic))
            assert cfg.agent is not None
            return int(cfg.agent.act(obs, legal_mask=mask, deterministic=deterministic))
        finally:
            self._heuristic_control_seat = None

    def step_auto(self, seat: Optional[int] = None, *, deterministic: bool = False) -> Tuple[int, int, LobbyStepInfo]:
        """Let the seat controller pick and apply an action. Returns (seat, action, info)."""
        s = seat if seat is not None else self.current_seat()
        action = self._choose_action(s, deterministic=deterministic)
        info = self._apply_action(s, action, auto=True)
        return s, action, info

    def step_action(self, seat: int, action: int) -> LobbyStepInfo:
        return self._apply_action(seat, int(action), auto=False)

    def drain_until_training_decision(
        self,
        *,
        deterministic: bool = False,
    ) -> List[Tuple[int, int, LobbyStepInfo]]:
        """Play non-blocking seats until a training seat must act or episode ends."""
        log: List[Tuple[int, int, LobbyStepInfo]] = []
        steps = 0
        while not self.episode_done and steps < MAX_DRAIN_STEPS:
            steps += 1
            active = self.active_training_seats()
            if not active:
                break
            cur = self.current_seat()
            if cur in active and self._seat_can_act(cur):
                break
            if not self._seat_can_act(cur):
                if self.state.done:
                    break
                self._raise_drain_stall(
                    where="BGLobbyEnv.drain_until_training_decision",
                    cur=cur,
                )
            seat, action, info = self.step_auto(cur, deterministic=deterministic)
            log.append((seat, action, info))
            for el in info.eliminated_seats:
                self._mark_finished_training(el)
            if self.state.done:
                for t in self._training_seats:
                    self._mark_finished_training(t)
                break
        if steps >= MAX_DRAIN_STEPS:
            self._raise_drain_exceeded_cap(
                where="BGLobbyEnv.drain_until_training_decision",
                steps=steps,
                cap=MAX_DRAIN_STEPS,
            )
        return log

    def drain_until_lobby_done(self, *, deterministic: bool = False) -> List[Tuple[int, int, LobbyStepInfo]]:
        log: List[Tuple[int, int, LobbyStepInfo]] = []
        steps = 0
        while not self.lobby_done and steps < MAX_DRAIN_STEPS:
            steps += 1
            cur = self.current_seat()
            if not self._seat_can_act(cur):
                if self.state.done:
                    break
                self._raise_drain_stall(
                    where="BGLobbyEnv.drain_until_lobby_done",
                    cur=cur,
                )
            seat, action, info = self.step_auto(cur, deterministic=deterministic)
            log.append((seat, action, info))
        if steps >= MAX_DRAIN_STEPS:
            self._raise_drain_exceeded_cap(
                where="BGLobbyEnv.drain_until_lobby_done",
                steps=steps,
                cap=MAX_DRAIN_STEPS,
            )
        if self.lobby_done:
            for s in range(NUM_PLAYERS):
                if is_seat_finished(self.state, s):
                    self._mark_finished_training(s)
        return log

    def finalize_placements(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for s in range(NUM_PLAYERS):
            if is_seat_finished(self.state, s):
                out[s] = placement_for_seat(self.state, s)
        return out


def run_lobby_episode(
    env: BGLobbyEnv,
    record_seats: Sequence[int],
    *,
    deterministic: bool = False,
) -> LobbyEpisodeResult:
    """Run one lobby, recording transitions for ``record_seats`` (learned policies)."""
    env.reset()
    transitions: Dict[int, List[SeatTransition]] = {s: [] for s in record_seats}
    record_set = frozenset(record_seats)
    steps = 0
    while not env.lobby_done and steps < MAX_DRAIN_STEPS:
        steps += 1
        cur = env.current_seat()
        if not env._seat_can_act(cur):
            if env.state.done:
                break
            env._raise_drain_stall(where="run_lobby_episode", cur=cur)
        if cur not in record_set:
            env.step_auto(cur, deterministic=deterministic)
            continue
        obs = env.obs_for_seat(cur)
        mask = env.legal_mask_for_seat(cur)
        action = env._choose_action(cur, deterministic=deterministic)
        step_info = env.step_action(cur, action)
        reward = 0.0
        terminated = False
        if cur in step_info.eliminated_seats or env.lobby_done:
            if is_seat_finished(env.state, cur):
                reward = placement_reward_for_seat(env.state, cur)
                terminated = True
        transitions[cur].append(
            SeatTransition(
                seat=cur,
                obs=obs,
                action=action,
                reward=reward,
                terminated=terminated,
                info={
                    "placement": step_info.placements.get(cur),
                    "lobby_done": step_info.lobby_done,
                },
            )
        )
        if env.lobby_done:
            break
    if steps >= MAX_DRAIN_STEPS and not env.lobby_done:
        env._raise_drain_exceeded_cap(
            where="run_lobby_episode",
            steps=steps,
            cap=MAX_DRAIN_STEPS,
        )
    placements = env.finalize_placements()
    outcomes = {
        s: LobbyStepInfo(acting_seat=s, placements={s: placements[s]})
        for s in record_seats
        if s in placements
    }
    return LobbyEpisodeResult(outcomes=outcomes, transitions=transitions)


class BGLobbySingleAgentEnv(SingleAgentEnv):
    """Single-agent API for one ``training_seat`` in a multi-seat lobby."""

    def __init__(
        self,
        training_seat: int,
        *,
        learned_seats: Optional[Sequence[int]] = None,
        seat_configs: Optional[Sequence[SeatConfig]] = None,
        agent_by_seat: Optional[Dict[int, BaseAgent]] = None,
        seed: Optional[int] = None,
        reward_config: Optional[RewardConfig] = None,
        shop_excluded_race=None,
        shop_full_tribes: bool = False,
    ) -> None:
        if not 0 <= training_seat < NUM_PLAYERS:
            raise ValueError(f"training_seat must be 0..{NUM_PLAYERS - 1}")
        ls = tuple(learned_seats if learned_seats is not None else (training_seat,))
        if training_seat not in ls:
            raise ValueError("training_seat must be in learned_seats")
        if seat_configs is None:
            seat_configs = lobby_from_learned_seats(ls, agent_by_seat=agent_by_seat, seed=seed)
        self._training_seat = training_seat
        self._lobby = BGLobbyEnv(
            seat_configs,
            learned_seats=ls,
            training_seats=(training_seat,),
            seed=seed,
            shop_excluded_race=shop_excluded_race,
            shop_full_tribes=shop_full_tribes,
        )
        self.reward_config = reward_config or RewardConfig(
            invalid_action=INVALID_ACTION_REWARD
        )
        self._done = False
        self._seed = seed

    @property
    def state(self) -> BGLikeState:
        return self._lobby.state

    @property
    def lobby(self) -> BGLobbyEnv:
        return self._lobby

    @property
    def done(self) -> bool:
        return self._done

    @property
    def training_seat(self) -> int:
        return self._training_seat

    def set_training_agent(self, agent: BaseAgent) -> None:
        cfg = self._lobby._seat_configs[self._training_seat]
        if cfg.kind != SeatKind.LEARNED:
            raise ValueError("training seat is not LEARNED")
        cfg.agent = agent

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._seed = seed
        self._lobby.reset(seed=self._seed)
        self._done = False
        self._drain_to_agent()
        return self._obs()

    def _obs(self) -> np.ndarray:
        return self._lobby.obs_for_seat(self._training_seat)

    @property
    def legal_actions_mask(self) -> np.ndarray:
        return self._lobby.legal_mask_for_seat(self._training_seat)

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done; call reset()")
        seat = self._training_seat
        if not self._lobby._seat_can_act(seat):
            raise RuntimeError(f"training seat {seat} cannot act")
        try:
            info = self._lobby.step_action(seat, int(action))
        except ValueError as exc:
            return StepResult(
                obs=self._obs(),
                reward=float(self.reward_config.invalid_action),
                terminated=False,
                truncated=False,
                info={"invalid_action": True, "error": str(exc)},
            )
        reward = 0.0
        terminated = False
        if seat in info.eliminated_seats or self._lobby.lobby_done:
            if is_seat_finished(self._lobby.state, seat):
                reward = placement_reward_for_seat(self._lobby.state, seat)
                terminated = True
                self._done = True
        if not terminated:
            self._drain_to_agent()
            if self._lobby.episode_done:
                if is_seat_finished(self._lobby.state, seat):
                    reward = placement_reward_for_seat(self._lobby.state, seat)
                terminated = True
                self._done = True
        place = None
        if is_seat_finished(self._lobby.state, seat):
            place = placement_for_seat(self._lobby.state, seat)
        return StepResult(
            obs=self._obs(),
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={
                "training_seat": seat,
                "placement": place,
                "placement_reward": placement_reward(place) if place else None,
                "lobby_done": self._lobby.lobby_done,
                "winner": self._lobby.state.winner,
            },
        )

    def _drain_to_agent(self) -> None:
        if self._done:
            return
        self._lobby.drain_until_training_decision()
        if self._lobby.episode_done:
            self._done = True

    def notify_episode_end(self, info: dict) -> None:
        return None


# Marker for AgentPerspectiveEnv: base env drains non-training seats internally.
DELEGATES_OPPONENT_PLAY = "delegates_opponent_play"


class BGLobbyMultiCurrentEnv(SingleAgentEnv):
    """Single-agent API when several lobby seats share the training policy."""

    def __init__(
        self,
        current_seats: Sequence[int],
        *,
        seed: Optional[int] = None,
        reward_config: Optional[RewardConfig] = None,
        shop_excluded_race=None,
        shop_full_tribes: bool = False,
    ) -> None:
        if not current_seats:
            raise ValueError("current_seats must be non-empty")
        self._current_seats: Tuple[int, ...] = tuple(sorted(set(current_seats)))
        for s in self._current_seats:
            if not 0 <= s < NUM_PLAYERS:
                raise ValueError(f"invalid seat {s}")
        self._current_agent: Optional[BaseAgent] = None
        self._opponents_by_seat: Dict[int, BaseAgent] = {}
        self._seed = seed
        self._shop_excluded_race = shop_excluded_race
        self._shop_full_tribes = shop_full_tribes
        self._lobby: Optional[BGLobbyEnv] = None
        self._acting_seat: Optional[int] = None
        self._rewarded_seats: Set[int] = set()
        self.reward_config = reward_config or RewardConfig(
            invalid_action=INVALID_ACTION_REWARD
        )
        self._done = False
        self._last_info: Dict[str, Any] = {}

    @property
    def delegates_opponent_play(self) -> bool:
        return True

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (OBS_DIM,)

    @property
    def num_actions(self) -> int:
        return int(NUM_ENV_ACTIONS)

    @property
    def current_seats(self) -> Tuple[int, ...]:
        return self._current_seats

    def set_current_seats(self, seats: Sequence[int]) -> None:
        if not seats:
            raise ValueError("seats must be non-empty")
        self._current_seats = tuple(sorted(set(seats)))
        for s in self._current_seats:
            if not 0 <= s < NUM_PLAYERS:
                raise ValueError(f"invalid seat {s}")

    @property
    def acting_seat(self) -> Optional[int]:
        return self._acting_seat

    @property
    def state(self) -> BGLikeState:
        assert self._lobby is not None
        return self._lobby.state

    @property
    def lobby(self) -> BGLobbyEnv:
        assert self._lobby is not None
        return self._lobby

    @property
    def done(self) -> bool:
        return self._done

    @property
    def uses_seat_segments(self) -> bool:
        return len(self._current_seats) >= 1

    def set_agents(
        self,
        current_agent: BaseAgent,
        opponents_by_seat: Dict[int, BaseAgent],
    ) -> None:
        self._current_agent = current_agent
        self._opponents_by_seat = dict(opponents_by_seat)
        if self._lobby is not None:
            self._rebuild_lobby()

    def _rebuild_lobby(self) -> None:
        assert self._current_agent is not None
        configs = build_training_lobby_configs(
            self._current_seats,
            self._current_agent,
            self._opponents_by_seat,
            seed=self._seed,
        )
        self._lobby = BGLobbyEnv(
            configs,
            learned_seats=self._current_seats,
            training_seats=self._current_seats,
            seed=self._seed,
            shop_excluded_race=self._shop_excluded_race,
            shop_full_tribes=self._shop_full_tribes,
        )

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._seed = seed
        if self._current_agent is None or not self._opponents_by_seat:
            raise RuntimeError("call set_agents(current, opponents_by_seat) before reset")
        self._rebuild_lobby()
        assert self._lobby is not None
        self._lobby.reset(seed=self._seed)
        self._done = False
        self._rewarded_seats = set()
        self._last_info = {}
        self._drain_to_acting()
        return self._obs()

    def _obs(self) -> np.ndarray:
        assert self._lobby is not None and self._acting_seat is not None
        return self._lobby.obs_for_seat(self._acting_seat)

    @property
    def legal_actions_mask(self) -> np.ndarray:
        if self._done or self._lobby is None or self._acting_seat is None:
            return np.zeros(NUM_ENV_ACTIONS, dtype=bool)
        return self._lobby.legal_mask_for_seat(self._acting_seat)

    def _active_current_seats(self) -> Set[int]:
        assert self._lobby is not None
        return {
            s
            for s in self._current_seats
            if s not in self._rewarded_seats
            and not is_seat_finished(self._lobby.state, s)
        }

    def _unrewarded_finished_seats(self) -> Dict[int, int]:
        assert self._lobby is not None
        out: Dict[int, int] = {}
        for s in self._current_seats:
            if s in self._rewarded_seats:
                continue
            if is_seat_finished(self._lobby.state, s):
                out[s] = placement_for_seat(self._lobby.state, s)
        return out

    def _all_current_rewarded(self) -> bool:
        return self._rewarded_seats >= set(self._current_seats)

    def _segment_closure_dict(self, seat: int, place: int) -> Dict[str, Any]:
        return {
            "seat": seat,
            "placement": place,
            "placement_reward": placement_reward(place),
        }

    def _finalize_lobby_if_needed(self) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
        """Close any finished current seats not yet rewarded; set ``_done`` when done."""
        assert self._lobby is not None
        closures: List[Dict[str, Any]] = []
        remaining = self._unrewarded_finished_seats()
        for seat, place in sorted(remaining.items()):
            closures.append(self._segment_closure_dict(seat, place))
            self._rewarded_seats.add(seat)
        placements = {
            s: placement_for_seat(self._lobby.state, s)
            for s in self._current_seats
            if s in self._rewarded_seats
        }
        if self._all_current_rewarded() or self._lobby.episode_done:
            self._done = True
        return closures, placements

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done; call reset()")
        assert self._lobby is not None and self._acting_seat is not None
        seat = self._acting_seat
        if not self._lobby._seat_can_act(seat):
            raise RuntimeError(f"acting seat {seat} cannot act")
        acting = seat
        combat_before = self._lobby.state.combat_round
        try:
            step_info = self._lobby.step_action(acting, int(action))
        except ValueError as exc:
            return StepResult(
                obs=self._obs(),
                reward=float(self.reward_config.invalid_action),
                terminated=False,
                truncated=False,
                info={"invalid_action": True, "error": str(exc)},
            )

        self._drain_to_acting()
        combat_after = self._lobby.state.combat_round
        combat_advanced = combat_after > combat_before

        finished = self._unrewarded_finished_seats()
        segment_closures: List[Dict[str, Any]] = []

        for s, place in sorted(finished.items()):
            segment_closures.append(self._segment_closure_dict(s, place))
            self._rewarded_seats.add(s)

        if not self._all_current_rewarded() and not self._lobby.episode_done:
            self._drain_to_acting()

        extra_closures, placements = self._finalize_lobby_if_needed()
        segment_closures.extend(extra_closures)

        info = self._build_step_info(
            acting,
            step_info,
            placements,
            combat_advanced=combat_advanced,
            segment_closures=segment_closures,
        )
        self._last_info = info
        if self._acting_seat is not None:
            obs = self._obs()
        else:
            obs = self._lobby.obs_for_seat(acting)
        return StepResult(
            obs=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

    def _build_step_info(
        self,
        acting_seat: int,
        step_info: LobbyStepInfo,
        placements: Dict[int, int],
        *,
        combat_advanced: bool = False,
        segment_closures: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        assert self._lobby is not None
        seat = acting_seat
        place = placements.get(seat)
        return {
            "acting_seat": seat,
            "next_acting_seat": self._acting_seat,
            "current_seats": self._current_seats,
            "placement": place,
            "placement_reward": placement_reward(place) if place is not None else None,
            "placements_current": placements,
            "segment_closures": list(segment_closures or ()),
            "lobby_episode_done": self._done,
            "combat_advanced": combat_advanced,
            "combat_round": self._lobby.state.combat_round,
            "lobby_done": self._lobby.lobby_done,
            "winner": self._lobby.state.winner,
            "eliminated_seats": step_info.eliminated_seats,
        }

    def _drain_to_acting(self) -> None:
        assert self._lobby is not None
        steps = 0
        cap = MAX_DRAIN_TO_ACTING_STEPS
        while not self._lobby.episode_done and steps < cap:
            steps += 1
            active = self._active_current_seats()
            if not active:
                break
            cur = self._lobby.current_seat()
            if cur in active and self._lobby._seat_can_act(cur):
                self._acting_seat = cur
                return
            for seat in sorted(active):
                if self._lobby._seat_can_act(seat):
                    self._acting_seat = seat
                    return
            if self._lobby._seat_can_act(cur):
                other_current = cur in self._current_seats
                _, _, drain_info = self._lobby.step_auto(
                    cur,
                    deterministic=other_current,
                )
                for el in drain_info.eliminated_seats:
                    self._lobby._mark_finished_training(el)
                continue
            advanced = False
            for seat in range(NUM_PLAYERS):
                if seat not in self._lobby.state.alive:
                    continue
                if not bool(self._lobby.legal_mask_for_seat(seat).any()):
                    continue
                other_current = seat in self._current_seats
                self._lobby.step_auto(seat, deterministic=other_current)
                advanced = True
                break
            if not advanced:
                if not self._lobby.state.done:
                    self._lobby._raise_drain_stall(
                        where="BGLobbyMultiCurrentEnv._drain_to_acting",
                        cur=cur,
                    )
                break
        if steps >= cap and not self._lobby.episode_done:
            self._lobby._raise_drain_exceeded_cap(
                where="BGLobbyMultiCurrentEnv._drain_to_acting",
                steps=steps,
                cap=cap,
            )
        self._acting_seat = None
        self._finalize_lobby_if_needed()
        if self._all_current_rewarded() or self._lobby.episode_done:
            self._done = True

    def notify_episode_end(self, info: dict) -> None:
        return None


def make_bglike_env(
    *,
    training_seat: int = 0,
    learned_seats: Optional[Sequence[int]] = None,
    agent_by_seat: Optional[Dict[int, BaseAgent]] = None,
    seed: Optional[int] = None,
    reward_config: Optional[RewardConfig] = None,
    **kwargs: Any,
) -> BGLobbySingleAgentEnv:
    """Factory used by ``register_game('bglike', ...)``."""
    return BGLobbySingleAgentEnv(
        training_seat=training_seat,
        learned_seats=learned_seats,
        agent_by_seat=agent_by_seat,
        seed=seed,
        reward_config=reward_config,
        **kwargs,
    )


def make_bglike_training_env(
    current_seats: Sequence[int],
    *,
    seed: Optional[int] = None,
    reward_config: Optional[RewardConfig] = None,
    **kwargs: Any,
) -> BGLobbyMultiCurrentEnv:
    """Multi-current lobby base for ``AgentPerspectiveEnv``."""
    return BGLobbyMultiCurrentEnv(
        current_seats,
        seed=seed,
        reward_config=reward_config,
        **kwargs,
    )


__all__ = [
    "BGLobbyEnv",
    "BGLobbyMultiCurrentEnv",
    "BGLobbySingleAgentEnv",
    "DELEGATES_OPPONENT_PLAY",
    "INVALID_ACTION_REWARD",
    "LobbyEpisodeResult",
    "LobbyStepInfo",
    "SeatTransition",
    "make_bglike_env",
    "make_bglike_training_env",
    "run_lobby_episode",
]

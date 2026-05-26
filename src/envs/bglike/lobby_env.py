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
    struct_action_to_game_action,
    struct_action_to_log_int,
    swap_adj_index_from_env_action,
    target_board_slot,
)
from .actions import (
    Action as GameAction,
    BOARD_SIZE,
    HAND_SIZE,
    MAX_SHOP_ACTIONS,
    MAX_SHOP_SLOTS,
    NUM_PLAYERS,
    STARTING_HEALTH,
    magnet_game_action,
)
from src.bg_lobby.player import BATTLE_HISTORY_LEN
from src.envs.minibg.structured_actions import (
    StructAction,
    StructActionType,
    structured_legal_set,
    validate_board_perm,
)
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
from .state import BGLikeState, PendingChoiceKind, PlayerPhase

INVALID_ACTION_REWARD = -1.0
MAX_DRAIN_STEPS = 1_000
MAX_DRAIN_TO_ACTING_STEPS = 1_000
MAX_DRAIN_TRACE_LINES = 48
MAX_DRAIN_REPEAT_WARN = 8


def _decode_drain_action(action: int) -> str:
    from .replay_render import decode_env_action_compact

    try:
        return decode_env_action_compact(int(action))
    except Exception:
        return f"action_{int(action)}"


def _board_signature(board: Sequence[Any]) -> Tuple[int, ...]:
    return tuple(id(m) for m in board)


def _format_drain_auto_line(
    step: int,
    *,
    cur: int,
    seat: int,
    action: int,
    elim: Sequence[int],
    board_sig: Tuple[int, ...] = (),
    controller: str = "",
    control_path: str = "",
    struct_action: str = "",
    deterministic: bool = False,
    prefix: str = "",
) -> str:
    ctrl = controller or "?"
    path = control_path or "?"
    struct_bit = f" struct={struct_action}" if struct_action else ""
    sig_bit = f" board_sig={board_sig}" if board_sig else ""
    head = f"{prefix}#{step} cur={cur} auto seat={seat}"
    return (
        f"{head} controller={ctrl} path={path} det={deterministic}"
        f"{struct_bit}{sig_bit} action={action} ({_decode_drain_action(action)}) "
        f"elim={tuple(elim)}"
    )


def _build_drain_debug_report(
    lobby: "BGLobbyEnv",
    *,
    controller: Any = None,
    drain_trace: Optional[Sequence[str]] = None,
) -> str:
    """Human-readable lobby snapshot when drain stalls or exceeds step cap."""
    s = lobby.state
    cur = s.current_player_index
    order_hint = ""
    try:
        oidx = s.shop_turn_order.index(cur)
        order_hint = f" (shop_turn index {oidx}/{len(s.shop_turn_order)})"
    except ValueError:
        order_hint = " (current seat NOT in shop_turn_order)"

    lines: List[str] = [
        f"current_player={cur} shop_turn_order={s.shop_turn_order}{order_hint}",
        (
            f"combat_round={s.combat_round} round={s.round_number} "
            f"lobby_done={lobby.lobby_done} episode_done={lobby.episode_done}"
        ),
    ]
    if controller is not None:
        active = controller._active_current_seats()
        lines.append(
            "controller: "
            f"current_seats={controller.current_seats} "
            f"active_current={sorted(active)} "
            f"rewarded={sorted(getattr(controller, '_rewarded_seats', set()))} "
            f"acting_seat={getattr(controller, '_acting_seat', None)} "
            f"controller_done={getattr(controller, '_done', None)}"
        )
    lines.append(
        f"alive={s.alive} eliminated="
        f"{[(snap.seat, snap.eliminated_combat_round) for snap in s.eliminated]}"
    )
    from src.training.controller_step import control_path_for_agent, describe_seat_controller

    lines.append("controllers:")
    for seat in range(NUM_PLAYERS):
        cfg = lobby._seat_configs[seat]
        agent = cfg.agent
        if agent is None:
            lines.append(f"  seat {seat}: (no agent)")
            continue
        ctrl = describe_seat_controller(
            agent,
            seat=seat,
            lobby=lobby,
            controller_env=controller,
        )
        path = control_path_for_agent(agent, lobby)
        learned = seat in lobby._learned_seats
        lines.append(
            f"  seat {seat}: controller={ctrl} path={path} learned={learned}"
        )
    lines.append("per-seat:")
    for seat in range(NUM_PLAYERS):
        p = s.players[seat]
        finished = is_seat_finished(s, seat)
        if seat not in s.alive and not finished:
            continue
        can = lobby._seat_can_act(seat)
        flat_n = int(lobby.legal_mask_for_seat(seat).sum()) if not finished else 0
        struct_n = (
            len(lobby.legal_structured_actions_for_seat(seat)) if not finished else 0
        )
        pending = p.pending_choice.kind.name if p.pending_choice is not None else None
        rl = seat in lobby._rl_pending
        cfg = lobby._seat_configs[seat]
        agent = cfg.agent
        ctrl_line = ""
        if agent is not None:
            ctrl = describe_seat_controller(
                agent,
                seat=seat,
                lobby=lobby,
                controller_env=controller,
            )
            path = control_path_for_agent(agent, lobby)
            ctrl_line = f" controller={ctrl} path={path}"
        lines.append(
            f"  seat {seat}: phase={p.phase.name} finished={finished} "
            f"is_current={seat == cur} can_act={can} "
            f"learned={seat in lobby._learned_seats} cfg={cfg.kind.name}{ctrl_line} "
            f"shop_used={p.shop_actions_used}/{MAX_SHOP_ACTIONS} "
            f"board={len(p.board)} hand_filled={sum(1 for x in p.hand if x is not None)} "
            f"pending={pending} rl_pending={rl} "
            f"legal_flat={flat_n} legal_struct={struct_n} hp={p.health} gold={p.gold}"
        )
    if drain_trace:
        lines.append(f"recent drain trace (last {MAX_DRAIN_TRACE_LINES}, oldest→newest):")
        for row in drain_trace:
            lines.append(f"  {row}")
    return "\n".join(lines)


def _append_drain_trace(trace: List[str], line: str) -> None:
    trace.append(line)
    if len(trace) > MAX_DRAIN_TRACE_LINES:
        del trace[0 : len(trace) - MAX_DRAIN_TRACE_LINES]


@dataclass
class LobbyStepInfo:
    acting_seat: int
    eliminated_seats: Tuple[int, ...] = ()
    lobby_done: bool = False
    placements: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AutoStepResult:
    seat: int
    action: int
    info: LobbyStepInfo
    controller: str = ""
    control_path: str = ""
    struct_action: str = ""


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
        shop_excluded_count: Optional[int] = None,
        shop_full_tribes: bool = False,
        replay: Optional[Any] = None,
        patch_dir: Optional[str] = None,
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
        self._patch_dir = patch_dir
        self._game = BGLikeGame(
            seed=seed,
            shop_excluded_race=shop_excluded_race,
            shop_excluded_count=shop_excluded_count,
            shop_full_tribes=shop_full_tribes,
            patch_dir=patch_dir,
        )
        self._state: Optional[BGLikeState] = None
        self._finished_training: Set[int] = set()
        self._last_battle_signed: Dict[int, float] = {i: 0.0 for i in range(NUM_PLAYERS)}
        self._rl_pending: Dict[int, RlPlacePlan] = {}
        self._rl_place_budget_pending: Set[int] = set()
        self._heuristic_control_seat: Optional[int] = None
        self._replay: Any = None
        if replay is not None:
            from .replay import attach_replay_config

            attach_replay_config(self, replay)

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
                shop_excluded_count=self._game._shop_excluded_count,
                shop_full_tribes=self._game._shop_full_tribes,
                patch_dir=self._patch_dir,
            )
        self._state = self._game.initial_state()
        self._finished_training = set()
        self._last_battle_signed = {i: 0.0 for i in range(NUM_PLAYERS)}
        self._rl_pending = {}
        self._rl_place_budget_pending = set()
        self._heuristic_control_seat = None
        if self._replay is not None:
            self._replay.on_reset()
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

    def _raise_drain_stall(
        self,
        *,
        where: str,
        cur: Optional[int] = None,
        drain_trace: Optional[Sequence[str]] = None,
    ) -> None:
        s = self.state
        seat = s.current_player_index if cur is None else cur
        phase = s.players[seat].phase.name
        debug = _build_drain_debug_report(
            self,
            controller=getattr(self, "_controller_env", None),
            drain_trace=drain_trace,
        )
        raise RuntimeError(
            f"{where}: drain iteration made no progress while lobby not done "
            f"(cur={seat}, phase={phase}, alive={s.alive}, "
            f"combat_round={s.combat_round}, round={s.round_number})\n\n"
            f"--- drain debug ---\n{debug}"
        )

    def _raise_drain_exceeded_cap(
        self,
        *,
        where: str,
        steps: int,
        cap: int,
        drain_trace: Optional[Sequence[str]] = None,
    ) -> None:
        s = self.state
        cur = s.current_player_index
        phase = s.players[cur].phase.name
        debug = _build_drain_debug_report(
            self,
            controller=getattr(self, "_controller_env", None),
            drain_trace=drain_trace,
        )
        raise RuntimeError(
            f"{where}: drain exceeded {cap} steps without finishing "
            f"(steps={steps}, cur={cur}, phase={phase}, alive={s.alive}, "
            f"combat_round={s.combat_round}, round={s.round_number}, done={s.done})\n\n"
            f"--- drain debug ---\n{debug}"
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
            patch=self._game._patch,
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

    def legal_structured_actions_for_seat(self, seat: int) -> List[StructAction]:
        """Legal structured tokens for ``seat`` (shop + COMPLETE_TURN; no mid-shop SWAP)."""
        if not self._seat_can_act(seat):
            return []
        if self.state.current_player_index != seat:
            return []

        player = self.state.players[seat]
        legal_game = {int(a) for a in self._game.legal_actions(self.state)}
        out: List[StructAction] = []

        plan = self._rl_pending.get(seat)
        if plan is not None:
            for tgt in plan.eligible_on_board_live(player.board):
                out.append(StructAction(StructActionType.APPLY_EFFECT, (tgt,)))
            if plan.can_skip_second_adjacent():
                out.append(StructAction(StructActionType.APPLY_EFFECT_SKIP))
            return out

        if player.pending_choice is not None:
            pc = player.pending_choice
            if pc.kind == PendingChoiceKind.TRANSFORM_SHOP_MINION:
                for slot in range(MAX_SHOP_SLOTS):
                    if (int(GameAction.BUY_SLOT_0) + slot) in legal_game:
                        out.append(StructAction(StructActionType.BUY, (slot,)))
                return out
            for i in range(3):
                if int(GameAction.DISCOVER_PICK_0) + i in legal_game:
                    out.append(StructAction(StructActionType.DISCOVER_PICK, (i,)))
            return out

        if int(GameAction.ROLL) in legal_game:
            out.append(StructAction(StructActionType.ROLL))
        if int(GameAction.LEVEL_UP) in legal_game:
            out.append(StructAction(StructActionType.LEVEL_UP))

        for slot in range(MAX_SHOP_SLOTS):
            if (int(GameAction.BUY_SLOT_0) + slot) in legal_game:
                out.append(StructAction(StructActionType.BUY, (slot,)))

        for pos in range(BOARD_SIZE):
            if (int(GameAction.SELL_BOARD_0) + pos) in legal_game:
                out.append(StructAction(StructActionType.SELL, (pos,)))

        for h in range(HAND_SIZE):
            if (int(GameAction.PLACE_HAND_0) + h) in legal_game:
                out.append(StructAction(StructActionType.PLACE, (h,)))

        for h in range(HAND_SIZE):
            for b in range(BOARD_SIZE):
                mg = int(magnet_game_action(h, b))
                if mg in legal_game:
                    out.append(StructAction(StructActionType.MAGNET, (h, b)))

        if (
            seat not in self._rl_pending
            and player.placed_minion_pending_after is None
        ):
            if int(GameAction.FINISH) in legal_game:
                out.append(StructAction(StructActionType.COMPLETE_TURN))
            if int(GameAction.FINISH_FREEZE_SHOP) in legal_game:
                out.append(StructAction(StructActionType.COMPLETE_TURN_FREEZE_SHOP))

        return out

    def _apply_complete_turn_for_seat(
        self,
        seat: int,
        perm: Tuple[int, ...],
        *,
        shop_phase_finish: int,
    ) -> None:
        assert self._state is not None
        player = self._state.players[seat]
        if player.phase != PlayerPhase.SHOP:
            raise ValueError(
                f"COMPLETE_TURN invalid phase {player.phase.name} for seat {seat}"
            )
        k = len(player.board)
        identity = tuple(range(BOARD_SIZE))
        if perm != identity and k > 0:
            self._state = self._game.reorder_board(self._state, seat, perm)
        self._state = self._game.apply_action(self._state, int(shop_phase_finish))

    def _mutate_flat_action(self, seat: int, action_int: int) -> None:
        s = self.state
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

    def _mutate_struct_action(
        self,
        seat: int,
        action: StructAction,
        board_perm: Optional[Tuple[int, ...]],
    ) -> None:
        if action.type in (
            StructActionType.COMPLETE_TURN,
            StructActionType.COMPLETE_TURN_FREEZE_SHOP,
        ):
            assert board_perm is not None
            shop_finish = (
                int(GameAction.FINISH_FREEZE_SHOP)
                if action.type == StructActionType.COMPLETE_TURN_FREEZE_SHOP
                else int(GameAction.FINISH)
            )
            self._apply_complete_turn_for_seat(
                seat, tuple(int(x) for x in board_perm), shop_phase_finish=shop_finish
            )
        elif action.type == StructActionType.APPLY_EFFECT:
            self._apply_rl_effect_pick(seat, action.args[0])
        elif action.type == StructActionType.APPLY_EFFECT_SKIP:
            self._apply_rl_effect_skip(seat)
        elif action.type == StructActionType.PLACE:
            if not self._try_begin_rl_place(seat, action.args[0]):
                raise ValueError(f"PLACE hand slot {action.args[0]} failed")
        else:
            ga = struct_action_to_game_action(action)
            self._state = self._game.apply_action(self.state, ga)

    def _apply_seat_step(
        self,
        seat: int,
        *,
        flat_action: int | None = None,
        struct_action: StructAction | None = None,
        board_perm: Optional[Tuple[int, ...]] = None,
        auto: bool = False,
    ) -> LobbyStepInfo:
        if (flat_action is None) == (struct_action is None):
            raise ValueError("exactly one of flat_action or struct_action must be provided")
        if not self._seat_can_act(seat):
            raise ValueError(
                f"seat {seat} cannot act now (current={self.state.current_player_index})"
            )

        eliminated_before = {snap.seat for snap in self.state.eliminated}
        prev_combat_round = self.state.combat_round
        prev_hp = tuple(p.health for p in self.state.players)

        if struct_action is not None:
            legal = structured_legal_set(
                tuple(self.legal_structured_actions_for_seat(seat))
            )
            if struct_action not in legal:
                raise ValueError(f"illegal structured action {struct_action!r} for seat {seat}")
            if struct_action.type in (
                StructActionType.COMPLETE_TURN,
                StructActionType.COMPLETE_TURN_FREEZE_SHOP,
            ):
                if board_perm is None:
                    raise ValueError("COMPLETE_TURN requires board_perm")
                validate_board_perm(tuple(board_perm), board_size=BOARD_SIZE)
            elif board_perm is not None:
                raise ValueError("board_perm only allowed for COMPLETE_TURN")
            self._mutate_struct_action(seat, struct_action, board_perm)
            action_int = struct_action_to_log_int(struct_action)
        else:
            action_int = int(flat_action)
            self._mutate_flat_action(seat, action_int)

        return self._finalize_seat_step(
            seat,
            action_int,
            eliminated_before=eliminated_before,
            prev_combat_round=prev_combat_round,
            prev_hp=prev_hp,
            auto=auto,
        )

    def step_structured_for_seat(
        self,
        seat: int,
        action: StructAction,
        *,
        board_perm: Optional[Tuple[int, ...]] = None,
    ) -> LobbyStepInfo:
        return self._apply_seat_step(
            seat,
            struct_action=action,
            board_perm=board_perm,
            auto=True,
        )

    def _update_battle_signed(self, prev_hp: Tuple[int, ...], prev_combat_round: int) -> None:
        if self.state.combat_round > prev_combat_round:
            for i in range(NUM_PLAYERS):
                delta = prev_hp[i] - self.state.players[i].health
                normalized = float(delta) / float(STARTING_HEALTH)
                self._last_battle_signed[i] = normalized
                p = self.state.players[i]
                p.battle_history = (p.battle_history + (normalized,))[-BATTLE_HISTORY_LEN:]

    def _lobby_step_info(
        self, seat: int, eliminated_before: Set[int]
    ) -> LobbyStepInfo:
        newly = tuple(
            snap.seat
            for snap in self.state.eliminated
            if snap.seat not in eliminated_before
        )
        placements: Dict[int, int] = {}
        for el in newly:
            if el in self._training_seats:
                placements[el] = placement_for_seat(self.state, el)
        return LobbyStepInfo(
            acting_seat=seat,
            eliminated_seats=newly,
            lobby_done=self.state.done,
            placements=placements,
        )

    def _finalize_seat_step(
        self,
        seat: int,
        action_int: int,
        *,
        eliminated_before: Set[int],
        prev_combat_round: int,
        prev_hp: Tuple[int, ...],
        auto: bool,
        illegal: bool = False,
    ) -> LobbyStepInfo:
        self._update_battle_signed(prev_hp, prev_combat_round)
        info = self._lobby_step_info(seat, eliminated_before)
        if self._replay is not None:
            self._replay.maybe_record(
                seat,
                action_int,
                info,
                state=self.state,
                prev_combat_round=prev_combat_round,
                auto=auto,
                illegal=illegal,
            )
        return info

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
            player,
            self._state.shop_excluded_race,
            rng=self._game._rng,
            patch=self._game._patch,
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
        if self._replay is not None:
            self._replay.close()
            self._replay = None

    def _apply_action(self, seat: int, action: int, *, auto: bool = False) -> LobbyStepInfo:
        return self._apply_seat_step(seat, flat_action=int(action), auto=auto)

    def step_auto(
        self, seat: Optional[int] = None, *, deterministic: bool = False
    ) -> AutoStepResult:
        """Let the seat controller pick and apply an action."""
        s = seat if seat is not None else self.current_seat()
        cfg = self._seat_configs[s]
        if cfg.agent is None:
            raise RuntimeError(f"seat {s} has no controller agent")
        from src.training.controller_step import lobby_seat_step

        result = lobby_seat_step(
            self,
            s,
            cfg.agent,
            deterministic=deterministic,
            controller_env=getattr(self, "_controller_env", None),
        )
        return AutoStepResult(
            seat=s,
            action=int(result.action),
            info=result.info,
            controller=result.controller,
            control_path=result.control_path,
            struct_action=result.struct_action,
        )

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
            auto = self.step_auto(cur, deterministic=deterministic)
            log.append((auto.seat, auto.action, auto.info))
            for el in auto.info.eliminated_seats:
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
        total_steps = 0
        round_steps = 0
        guard_combat_round = self.state.combat_round
        trace: List[str] = []
        while not self.lobby_done:
            if self.state.combat_round != guard_combat_round:
                guard_combat_round = self.state.combat_round
                round_steps = 0
                trace = []
            if round_steps >= MAX_DRAIN_STEPS:
                self._raise_drain_exceeded_cap(
                    where="BGLobbyEnv.drain_until_lobby_done",
                    steps=round_steps,
                    cap=MAX_DRAIN_STEPS,
                    drain_trace=trace,
                )
            total_steps += 1
            round_steps += 1
            cur = self.current_seat()
            if not self._seat_can_act(cur):
                if self.state.done:
                    break
                self._raise_drain_stall(
                    where="BGLobbyEnv.drain_until_lobby_done",
                    cur=cur,
                    drain_trace=trace,
                )
            auto = self.step_auto(cur, deterministic=deterministic)
            log.append((auto.seat, auto.action, auto.info))
            _append_drain_trace(
                trace,
                _format_drain_auto_line(
                    total_steps,
                    cur=cur,
                    seat=auto.seat,
                    action=auto.action,
                    elim=auto.info.eliminated_seats,
                    controller=auto.controller,
                    control_path=auto.control_path,
                    struct_action=auto.struct_action,
                    deterministic=deterministic,
                ),
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
    round_steps = 0
    guard_combat_round = env.state.combat_round
    while not env.lobby_done:
        if env.state.combat_round != guard_combat_round:
            guard_combat_round = env.state.combat_round
            round_steps = 0
        if round_steps >= MAX_DRAIN_STEPS:
            env._raise_drain_exceeded_cap(
                where="run_lobby_episode",
                steps=round_steps,
                cap=MAX_DRAIN_STEPS,
            )
        round_steps += 1
        cur = env.current_seat()
        if not env._seat_can_act(cur):
            if env.state.done:
                break
            env._raise_drain_stall(where="run_lobby_episode", cur=cur)
        if cur not in record_set:
            env.step_auto(cur, deterministic=deterministic)
            continue
        from src.training.controller_step import lobby_seat_step

        obs = env.obs_for_seat(cur)
        agent = env._seat_configs[cur].agent
        assert agent is not None
        step_result = lobby_seat_step(
            env,
            cur,
            agent,
            deterministic=deterministic,
            controller_env=getattr(env, "_controller_env", None),
        )
        action = int(step_result.action)
        step_info = step_result.info
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
        shop_excluded_count: Optional[int] = None,
        shop_full_tribes: bool = False,
        patch_dir: Optional[str] = None,
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
            shop_excluded_count=shop_excluded_count,
            shop_full_tribes=shop_full_tribes,
            patch_dir=patch_dir,
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
        shop_excluded_count: Optional[int] = None,
        shop_full_tribes: bool = False,
        patch_dir: Optional[str] = None,
    ) -> None:
        if not current_seats:
            raise ValueError("current_seats must be non-empty")
        self._current_seats: Tuple[int, ...] = tuple(sorted(set(current_seats)))
        for s in self._current_seats:
            if not 0 <= s < NUM_PLAYERS:
                raise ValueError(f"invalid seat {s}")
        self._current_agent: Optional[BaseAgent] = None
        self._opponents_by_seat: Dict[int, BaseAgent] = {}
        self._opponent_slot_by_seat: Dict[int, int] = {}
        self._seed = seed
        self._shop_excluded_race = shop_excluded_race
        self._shop_excluded_count = shop_excluded_count
        self._shop_full_tribes = shop_full_tribes
        self._patch_dir = patch_dir
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
            shop_excluded_count=self._shop_excluded_count,
            shop_full_tribes=self._shop_full_tribes,
            patch_dir=self._patch_dir,
        )
        self._lobby._controller_env = self

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

    def legal_structured_actions(self) -> List[StructAction]:
        if self._done or self._lobby is None or self._acting_seat is None:
            return []
        return self._lobby.legal_structured_actions_for_seat(self._acting_seat)

    def step_structured(
        self,
        action: StructAction,
        *,
        board_perm: Optional[Tuple[int, ...]] = None,
    ) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done; call reset()")
        assert self._lobby is not None and self._acting_seat is not None
        seat = self._acting_seat
        if not self._lobby._seat_can_act(seat):
            raise RuntimeError(f"acting seat {seat} cannot act")
        acting = seat
        combat_before = self._lobby.state.combat_round
        try:
            step_info = self._lobby.step_structured_for_seat(
                acting, action, board_perm=board_perm
            )
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

    def _placements_full(self) -> Dict[int, int]:
        """Final placement for every seat that has finished (1 = winner, 8 = first out)."""
        assert self._lobby is not None
        out: Dict[int, int] = {}
        for s in range(NUM_PLAYERS):
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
        if self._lobby.lobby_done:
            self._done = True
        return closures, placements

    def finish_lobby_to_end(self) -> Dict[str, Any]:
        """Auto-play remaining seats until the lobby winner is decided."""
        assert self._lobby is not None
        if not self._lobby.lobby_done:
            self._lobby.drain_until_lobby_done(deterministic=False)
        extra_closures, placements = self._finalize_lobby_if_needed()
        self._acting_seat = None
        self._done = bool(self._lobby.lobby_done)
        acting = self._current_seats[0] if self._current_seats else 0
        info = self._build_step_info(
            acting,
            LobbyStepInfo(
                acting_seat=acting,
                lobby_done=bool(self._lobby.lobby_done),
            ),
            placements,
            segment_closures=extra_closures,
        )
        self._last_info = info
        return info

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
            "placements_full": self._placements_full(),
            "segment_closures": list(segment_closures or ()),
            "lobby_episode_done": self._done,
            "combat_advanced": combat_advanced,
            "combat_round": self._lobby.state.combat_round,
            "lobby_done": self._lobby.lobby_done,
            "winner": self._lobby.state.winner,
            "eliminated_seats": step_info.eliminated_seats,
        }

    def _process_drain_auto_step(
        self,
        *,
        steps: int,
        cur: int,
        seat: int,
        deterministic: bool,
        trace: List[str],
        repeat_counts: Dict[Tuple[int, int, Tuple[int, ...]], int],
        prefix: str = "",
    ) -> LobbyStepInfo:
        assert self._lobby is not None
        board_sig = _board_signature(self._lobby.state.players[seat].board)
        auto = self._lobby.step_auto(seat, deterministic=deterministic)
        repeat_key = (auto.seat, auto.action, board_sig)
        repeat_counts[repeat_key] = repeat_counts.get(repeat_key, 0) + 1
        _append_drain_trace(
            trace,
            _format_drain_auto_line(
                steps,
                cur=cur,
                seat=auto.seat,
                action=auto.action,
                elim=auto.info.eliminated_seats,
                board_sig=board_sig,
                controller=auto.controller,
                control_path=auto.control_path,
                struct_action=auto.struct_action,
                deterministic=deterministic,
                prefix=prefix,
            ),
        )
        if repeat_counts[repeat_key] >= MAX_DRAIN_REPEAT_WARN:
            _append_drain_trace(
                trace,
                f"#{steps} REPEAT x{repeat_counts[repeat_key]} "
                f"seat={auto.seat} action={auto.action} "
                f"board_sig={board_sig} controller={auto.controller} "
                f"path={auto.control_path}",
            )
        return auto.info

    def _drain_to_acting(self) -> None:
        assert self._lobby is not None
        steps = 0
        cap = MAX_DRAIN_TO_ACTING_STEPS
        trace: List[str] = []
        repeat_counts: Dict[Tuple[int, int, Tuple[int, ...]], int] = {}
        while not self._lobby.episode_done and steps < cap:
            steps += 1
            active = self._active_current_seats()
            if not active:
                _append_drain_trace(trace, f"#{steps} no active current seats; stop drain")
                break
            cur = self._lobby.current_seat()
            if cur in active and self._lobby._seat_can_act(cur):
                self._acting_seat = cur
                _append_drain_trace(trace, f"#{steps} ready current seat {cur}")
                return
            for seat in sorted(active):
                if self._lobby._seat_can_act(seat):
                    self._acting_seat = seat
                    _append_drain_trace(trace, f"#{steps} ready active current seat {seat}")
                    return
            if self._lobby._seat_can_act(cur):
                other_current = cur in self._current_seats
                drain_info = self._process_drain_auto_step(
                    steps=steps,
                    cur=cur,
                    seat=cur,
                    deterministic=other_current,
                    trace=trace,
                    repeat_counts=repeat_counts,

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
                drain_info = self._process_drain_auto_step(
                    steps=steps,
                    cur=cur,
                    seat=seat,
                    deterministic=other_current,
                    trace=trace,
                    repeat_counts=repeat_counts,

                    prefix="scan ",
                )
                advanced = True
                break
            if not advanced:
                if not self._lobby.state.done:
                    self._lobby._raise_drain_stall(
                        where="BGLobbyMultiCurrentEnv._drain_to_acting",
                        cur=cur,
                        drain_trace=trace,
                    )
                break
        if steps >= cap and not self._lobby.episode_done:
            self._lobby._raise_drain_exceeded_cap(
                where="BGLobbyMultiCurrentEnv._drain_to_acting",
                steps=steps,
                cap=cap,
                drain_trace=trace,
            )
        self._acting_seat = None
        self._finalize_lobby_if_needed()
        if self._lobby.lobby_done:
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
    "AutoStepResult",
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

"""RL placement: PLACE → APPLY (geometry / target only) → commit via game APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

from src.bg_core.effects import (
    BuffAdjacentBattlecry,
    BuffTargetFriendlyBattlecry,
    ConsumeFriendlyBattlecry,
    Trigger,
)
from src.bg_core.minion import Minion, Race
from src.bg_recruitment.effect_modal import compute_eligible_buff_target
from src.bg_lobby.shared_pool import SharedCardPool
from src.bg_recruitment.place import place_from_hand
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.envs.minibg.state import CasterKind, CasterRef, MiniBGState, PlayerState

RlEffectParams = BuffTargetFriendlyBattlecry | BuffAdjacentBattlecry | ConsumeFriendlyBattlecry


class RlEffectKind(IntEnum):
    BUFF_TARGET_FRIENDLY = 0
    BUFF_ADJACENT_SLOT = 1


@dataclass
class RlPlacePlan:
    """Staged play from hand: minion stays in hand until commit."""

    hand_slot: int
    minion: Minion
    kind: RlEffectKind
    params: RlEffectParams
    picks: List[int] = field(default_factory=list)
    skipped_second: bool = False
    awaiting_second_adjacent: bool = False

    @property
    def apply_steps_total(self) -> int:
        """How many APPLY (or skip) steps this staged play needs (Brann does not add steps)."""
        if self.kind == RlEffectKind.BUFF_TARGET_FRIENDLY:
            return 1
        return 2

    @property
    def remaining(self) -> int:
        if self.kind == RlEffectKind.BUFF_TARGET_FRIENDLY:
            return 0 if self.picks else 1
        if self.awaiting_second_adjacent:
            return 1
        return max(0, 2 - len(self.picks))

    def can_skip_second_adjacent(self) -> bool:
        return (
            self.kind == RlEffectKind.BUFF_ADJACENT_SLOT
            and self.awaiting_second_adjacent
        )

    def eligible_on_board_live(
        self, board: Sequence[Minion]
    ) -> Tuple[int, ...]:
        if self.kind == RlEffectKind.BUFF_ADJACENT_SLOT:
            if not self.picks:
                return tuple(range(len(board)))
            if self.awaiting_second_adjacent:
                return tuple(i for i in range(len(board)) if i != self.picks[0])
            return ()
        assert isinstance(self.params, (BuffTargetFriendlyBattlecry, ConsumeFriendlyBattlecry))
        caster = CasterRef(CasterKind.HAND, hand_idx=self.hand_slot)
        if isinstance(self.params, ConsumeFriendlyBattlecry):
            return compute_eligible_buff_target(
                board,
                caster,
                BuffTargetFriendlyBattlecry(
                    filter_race=self.params.filter_race,
                    exclude_self=self.params.exclude_self,
                ),
            )
        return compute_eligible_buff_target(board, caster, self.params)

    def record_pick_live(self, board: Sequence[Minion], board_idx: int) -> None:
        if board_idx not in self.eligible_on_board_live(board):
            raise ValueError(
                f"RL pick {board_idx} not in eligible {self.eligible_on_board_live(board)}"
            )
        if self.kind == RlEffectKind.BUFF_ADJACENT_SLOT:
            if not self.picks:
                self.picks.append(board_idx)
                self.awaiting_second_adjacent = True
                return
            if self.awaiting_second_adjacent:
                self.picks.append(board_idx)
                self.awaiting_second_adjacent = False
                return
        self.picks.append(board_idx)

    def record_skip_second(self) -> None:
        if not self.can_skip_second_adjacent():
            raise ValueError("skip only valid on second adjacent pick")
        self.skipped_second = True
        self.awaiting_second_adjacent = False

    def is_complete(self) -> bool:
        if self.kind == RlEffectKind.BUFF_TARGET_FRIENDLY:
            return len(self.picks) >= 1
        if self.skipped_second:
            return True
        return len(self.picks) >= 2


RlPendingEffect = RlPlacePlan


def _hand_caster(hand_slot: int) -> CasterRef:
    return CasterRef(CasterKind.HAND, hand_idx=hand_slot)


def _targeted_effect_from_minion(
    minion: Minion,
) -> Optional[Tuple[RlEffectKind, RlEffectParams]]:
    for ab in minion.abilities:
        if ab.trigger != Trigger.ON_PLACE:
            continue
        if isinstance(ab.effect, BuffTargetFriendlyBattlecry):
            return RlEffectKind.BUFF_TARGET_FRIENDLY, ab.effect
        if isinstance(ab.effect, ConsumeFriendlyBattlecry):
            return RlEffectKind.BUFF_TARGET_FRIENDLY, ab.effect
        if isinstance(ab.effect, BuffAdjacentBattlecry):
            return RlEffectKind.BUFF_ADJACENT_SLOT, ab.effect
    return None


def open_rl_place_plan(
    player: PlayerState,
    hand_slot: int,
) -> Optional[RlPlacePlan]:
    minion = player.hand[hand_slot]
    if minion is None:
        return None
    spec = _targeted_effect_from_minion(minion)
    if spec is None:
        return None
    if len(player.board) == 0:
        return None
    kind, params = spec
    if kind == RlEffectKind.BUFF_TARGET_FRIENDLY:
        assert isinstance(params, (BuffTargetFriendlyBattlecry, ConsumeFriendlyBattlecry))
        eligible = compute_eligible_buff_target(
            player.board,
            _hand_caster(hand_slot),
            params
            if isinstance(params, BuffTargetFriendlyBattlecry)
            else BuffTargetFriendlyBattlecry(
                filter_race=params.filter_race,
                exclude_self=params.exclude_self,
            ),
        )
        if not eligible:
            return None
    return RlPlacePlan(
        hand_slot=hand_slot,
        minion=minion,
        kind=kind,
        params=params,
    )


def final_board_with_adjacent_insert(
    board: Sequence[Minion],
    minion: Minion,
    picks: Sequence[int],
    *,
    skipped_second: bool,
) -> List[Minion]:
    """Target layout after adjacent placement (used only to derive perm + insert_at)."""
    if not board:
        return [minion]
    if skipped_second or len(picks) < 2:
        t1 = picks[0]
        if t1 == 0:
            return [minion, *board]
        if t1 == len(board) - 1:
            return [*board, minion]
        anchor = board[t1]
        rest = [m for i, m in enumerate(board) if i != t1]
        return [*rest, anchor, minion]
    t1, t2 = int(picks[0]), int(picks[1])
    lo, hi = (t1, t2) if t1 < t2 else (t2, t1)
    m_lo, m_hi = board[lo], board[hi]
    before = list(board[:lo])
    middle = list(board[lo + 1 : hi])
    after = list(board[hi + 1 :])
    return before + [m_lo, minion, m_hi] + middle + after


def adjacent_placement_geometry(
    board: Sequence[Minion],
    played: Minion,
    picks: Sequence[int],
    *,
    skipped_second: bool,
    board_size: int,
) -> Tuple[Tuple[int, ...], int]:
    """Map RL adjacent picks → ``reorder_board`` perm and ``place_from_hand`` insert index."""
    final = final_board_with_adjacent_insert(
        board, played, picks, skipped_second=skipped_second
    )
    insert_at = final.index(played)
    reordered = [m for m in final if m is not played]
    k = len(board)
    head: list[int] = []
    for m in reordered:
        for i, b in enumerate(board):
            if b is m:
                head.append(i)
                break
        else:
            raise ValueError("reordered minion not on board")
    if len(set(head)) != k:
        raise ValueError(f"non-injective board reorder mapping: head={head!r}")
    tail = sorted(set(range(board_size)) - set(head))
    perm = tuple(head + tail)
    if len(perm) != board_size:
        raise ValueError(f"perm length {len(perm)} != board_size {board_size}")
    return perm, insert_at


def commit_rl_place_plan(
    state: MiniBGState,
    player_idx: int,
    plan: RlPlacePlan,
    *,
    board_size: int,
    shop_excluded_race: Optional[Race],
    triggers: ShopTriggers,
    rng,
    reorder_board,
    shared_pool: Optional[SharedCardPool] = None,
) -> MiniBGState:
    """Commit staged play: reorder (adjacent only) + ``place_from_hand`` + game battlecries."""
    player = state.players[player_idx]
    assert player.hand[plan.hand_slot] is plan.minion

    forced_buff_target: Optional[Minion] = None
    insert_at: Optional[int] = None

    if plan.kind == RlEffectKind.BUFF_TARGET_FRIENDLY:
        assert plan.picks
        forced_buff_target = player.board[plan.picks[0]]
        insert_at = len(player.board)
    else:
        perm, insert_at = adjacent_placement_geometry(
            player.board,
            plan.minion,
            plan.picks,
            skipped_second=plan.skipped_second,
            board_size=board_size,
        )
        state = reorder_board(state, player_idx, perm)
        player = state.players[player_idx]

    place_from_hand(
        player,
        plan.hand_slot,
        shop_excluded_race,
        board_size=board_size,
        triggers=triggers,
        rng=rng,
        insert_at=insert_at,
        apply_targeted_effects=True,
        forced_buff_target=forced_buff_target,
        shared_pool=shared_pool,
    )
    return state


def commit_simple_place_from_hand(
    player: PlayerState,
    hand_slot: int,
    shop_excluded_race: Optional[Race],
    *,
    board_size: int,
    triggers: ShopTriggers,
    rng,
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    place_from_hand(
        player,
        hand_slot,
        shop_excluded_race,
        board_size=board_size,
        triggers=triggers,
        rng=rng,
        apply_targeted_effects=True,
        shared_pool=shared_pool,
    )


__all__ = [
    "RlEffectKind",
    "RlEffectParams",
    "RlPendingEffect",
    "RlPlacePlan",
    "adjacent_placement_geometry",
    "commit_rl_place_plan",
    "commit_simple_place_from_hand",
    "final_board_with_adjacent_insert",
    "open_rl_place_plan",
]

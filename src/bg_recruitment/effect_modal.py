"""Pure helpers for applying targeted shop battlecries (no modal / pending state)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from src.bg_core.effects import BuffAdjacentBattlecry, BuffTargetFriendlyBattlecry, Keyword
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import CasterKind, CasterRef


def minion_matches_tribe(m: Minion, tribe) -> bool:
    if tribe is None:
        return True
    if tribe is Race.ALL:
        return True
    return m.race == tribe or m.race is Race.ALL


def caster_ref_from_board_minion(board: Sequence[Minion], minion: Minion) -> CasterRef:
    try:
        idx = board.index(minion)
    except ValueError:
        idx = max(0, len(board) - 1)
    return CasterRef(CasterKind.BOARD, board_idx=idx)


def adjacent_board_indices(
    board: Sequence[Minion], caster: CasterRef
) -> Tuple[int, ...]:
    if caster.kind != CasterKind.BOARD or caster.board_idx is None:
        return ()
    idx = caster.board_idx
    out: List[int] = []
    for j in (idx - 1, idx + 1):
        if 0 <= j < len(board):
            out.append(j)
    return tuple(out)


def compute_eligible_buff_target(
    board: Sequence[Minion],
    caster: CasterRef,
    effect: BuffTargetFriendlyBattlecry,
) -> Tuple[int, ...]:
    exclude_idx: Optional[int] = None
    if effect.exclude_self and caster.kind == CasterKind.BOARD:
        if caster.board_idx is not None and 0 <= caster.board_idx < len(board):
            exclude_idx = caster.board_idx
    out: List[int] = []
    for i, m in enumerate(board):
        if exclude_idx is not None and i == exclude_idx:
            continue
        if effect.filter_race is not None and not minion_matches_tribe(
            m, effect.filter_race
        ):
            continue
        out.append(i)
    return tuple(out)


def _apply_buff_target(
    board: Sequence[Minion], idx: int, effect: BuffTargetFriendlyBattlecry
) -> None:
    m = board[idx]
    m.bonus_attack += effect.attack
    m.bonus_health += effect.health


def _apply_adjacent_pick(
    board: Sequence[Minion], idx: int, effect: BuffAdjacentBattlecry
) -> None:
    m = board[idx]
    m.bonus_attack += effect.attack
    m.bonus_health += effect.health
    if effect.grant_taunt:
        m.keywords = frozenset(m.keywords | {Keyword.TAUNT})

"""Continuous stat auras as a board minion reads in the shop phase.

Combat re-derives auras on every step (``bg_combat/battle/auras.py``), so a
Mal'Ganis-buffed demon fights with the right numbers. Nothing derived them
outside combat: ``Minion.raw_attack``/``max_health`` are base + bonus only, and
a shop effect that copies another board minion's stats therefore read the
un-auraed body. The summing itself is ``stat_aura_bonus``, shared with combat.
"""

from __future__ import annotations

from typing import List, Tuple

from src.bg_core.board_helpers import (
    apply_attack_thresholds,
    has_attack_threshold_ability,
    stat_aura_bonus,
)
from src.bg_core.minion import Minion


def shop_stat_aura_bonus(board: List[Minion], minion: Minion) -> Tuple[int, int]:
    """Stats ``minion`` receives from the continuous auras of its boardmates."""
    return stat_aura_bonus(board, minion)


def shop_effective_stats(board: List[Minion], minion: Minion) -> Tuple[int, int]:
    """``minion``'s attack/health as the shop shows them, auras included."""
    atk, hp = stat_aura_bonus(board, minion)
    return minion.raw_attack + atk, minion.max_health + hp


def refresh_attack_thresholds(board: List[Minion]) -> None:
    """Grant the keywords the shop board's Attack values have just earned.

    The combat half of this rule lives in ``battle/auras.py``; this is the same
    latch on the shop board, so a minion buffed past its threshold between
    fights carries the keyword into the next one — and shows it while the seat
    is still shopping. Boards without a watcher pay one scan and leave.
    """
    if not any(has_attack_threshold_ability(m) for m in board):
        return
    for minion in board:
        attack, _ = shop_effective_stats(board, minion)
        apply_attack_thresholds(minion, attack)


__all__ = [
    "refresh_attack_thresholds",
    "shop_effective_stats",
    "shop_stat_aura_bonus",
]

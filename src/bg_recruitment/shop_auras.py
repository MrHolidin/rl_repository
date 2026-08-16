"""Continuous stat auras as a board minion reads in the shop phase.

Combat re-derives auras on every step (``bg_combat/battle/auras.py``), so a
Mal'Ganis-buffed demon fights with the right numbers. Nothing derived them
outside combat: ``Minion.raw_attack``/``max_health`` are base + bonus only, and
a shop effect that copies another board minion's stats therefore read the
un-auraed body. The summing itself is ``stat_aura_bonus``, shared with combat.
"""

from __future__ import annotations

from typing import List, Tuple

from src.bg_core.board_helpers import stat_aura_bonus
from src.bg_core.minion import Minion


def shop_stat_aura_bonus(board: List[Minion], minion: Minion) -> Tuple[int, int]:
    """Stats ``minion`` receives from the continuous auras of its boardmates."""
    return stat_aura_bonus(board, minion)


def shop_effective_stats(board: List[Minion], minion: Minion) -> Tuple[int, int]:
    """``minion``'s attack/health as the shop shows them, auras included."""
    atk, hp = stat_aura_bonus(board, minion)
    return minion.raw_attack + atk, minion.max_health + hp


__all__ = ["shop_effective_stats", "shop_stat_aura_bonus"]

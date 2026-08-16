"""Continuous stat auras as a board minion reads in the shop phase.

Combat re-derives auras on every step (``bg_combat/battle/auras.py``), so a
Mal'Ganis-buffed demon fights with the right numbers. Nothing derived them
outside combat: ``Minion.raw_attack``/``max_health`` are base + bonus only, and
a shop effect that copies another board minion's stats therefore read the
un-auraed body. Same aura kinds and same matching rules as combat, minus the
runtime — the shop board is a plain list, so indices come straight from it.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from src.bg_core.effects import (
    StatAura,
    Trigger,
)
from src.bg_core.board_helpers import buff_matching_hits
from src.bg_core.minion import Minion, Race


def _matches_tribe_for_aura(recipient: Minion, required: Any) -> bool:
    r = recipient.race
    if r is None:
        return False
    if required == Race.ALL or r == Race.ALL:
        return True
    return r == required


def _contribution(
    recipient: Minion, effect: object, *, idx_r: int, idx_s: int
) -> Tuple[int, int]:
    if not isinstance(effect, StatAura):
        return 0, 0
    if not buff_matching_hits(
        effect, recipient, idx_candidate=idx_r, idx_source=idx_s
    ):
        return 0, 0
    return effect.attack, effect.health


def shop_stat_aura_bonus(board: List[Minion], minion: Minion) -> Tuple[int, int]:
    """Stats ``minion`` receives from the continuous auras of its boardmates."""
    idx_r: Optional[int] = None
    for i, m in enumerate(board):
        if m is minion:
            idx_r = i
            break
    if idx_r is None:
        return 0, 0
    atk = 0
    hp = 0
    for idx_s, source in enumerate(board):
        if source is minion:
            continue
        for ab in source.abilities:
            if ab.trigger != Trigger.AURA:
                continue
            a, h = _contribution(minion, ab.effect, idx_r=idx_r, idx_s=idx_s)
            atk += a
            hp += h
    return atk, hp


def shop_effective_stats(board: List[Minion], minion: Minion) -> Tuple[int, int]:
    """``minion``'s attack/health as the shop shows them, auras included."""
    atk, hp = shop_stat_aura_bonus(board, minion)
    return minion.raw_attack + atk, minion.max_health + hp


__all__ = ["shop_effective_stats", "shop_stat_aura_bonus"]

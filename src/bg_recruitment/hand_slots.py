"""Hand slot helpers (variable hand size per ruleset)."""

from __future__ import annotations

from typing import Optional

from src.bg_lobby.player import PlayerState


def hand_size(player: PlayerState) -> int:
    return len(player.hand)


def hand_has_free_slot(player: PlayerState) -> bool:
    return any(s is None for s in player.hand)


def first_free_hand_slot(player: PlayerState) -> Optional[int]:
    for i, slot in enumerate(player.hand):
        if slot is None:
            return i
    return None


__all__ = ["first_free_hand_slot", "hand_has_free_slot", "hand_size"]

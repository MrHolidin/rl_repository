"""Hand slot helpers (variable hand size per ruleset)."""

from __future__ import annotations

from typing import Optional, Sequence

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
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


def apply_combat_hand_adds(
    player: PlayerState, card_ids: Sequence[str], patch: PatchContext
) -> None:
    for cid in card_ids:
        slot = first_free_hand_slot(player)
        if slot is None:
            break
        player.hand[slot] = make_minion(cid, patch=patch)


__all__ = ["apply_combat_hand_adds", "first_free_hand_slot", "hand_has_free_slot", "hand_size"]

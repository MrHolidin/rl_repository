"""Lockboxes — a Pirate card that pays out on a timer.

"Unplayable. In 5 turns, break this open and get a random Golden minion with a
type!" It sits in hand doing nothing, counts down one turn at a time, and on
zero replaces itself with a random Golden minion that has a tribe.

The rule worth writing down is what a *second* Lockbox does. Every card that
makes one says "Get a Lockbox. **If you already have one, it opens 1 turn
sooner**" — so they never stack: the seat holds one box, and further boxes are
spent accelerating it. Trinkets say the same with a bigger number. Modelling
them as ordinary cards that happen to be identical would quietly give a Pirate
seat five boxes and five payouts.
"""

from __future__ import annotations

from typing import Optional

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion, Race
from src.bg_core.spell_card import SpellCard
from src.bg_lobby.player import PlayerState

from .hand_slots import first_free_hand_slot

__all__ = [
    "LOCKBOX_CARD_ID",
    "LOCKBOX_TURNS",
    "find_lockbox",
    "give_lockbox",
    "tick_lockboxes",
    "is_lockbox",
]

LOCKBOX_CARD_ID = "BG36_520t"
#: Turns from being handed out to opening, per the card text.
LOCKBOX_TURNS = 5


def is_lockbox(card) -> bool:
    return isinstance(card, SpellCard) and card.card_id == LOCKBOX_CARD_ID


def _make_lockbox(turns_left: int) -> SpellCard:
    return SpellCard(
        card_id=LOCKBOX_CARD_ID,
        name="Lockbox",
        cost=0,
        # Unplayable by hand: nothing opens it early except the accelerators.
        turns_until_open=turns_left,
    )


def find_lockbox(player: PlayerState) -> Optional[int]:
    """Hand index of the seat's Lockbox, or None."""
    for i, card in enumerate(player.hand):
        if is_lockbox(card):
            return i
    return None


def give_lockbox(player: PlayerState, *, sooner: int = 1) -> bool:
    """Hand out a Lockbox, or bring the existing one ``sooner`` turns forward.

    Returns True when a box was created, False when an existing one was
    accelerated instead (or when there was no room in hand for a new one).
    """
    existing = find_lockbox(player)
    if existing is not None:
        card = player.hand[existing]
        player.hand[existing] = _make_lockbox(
            max(0, card.turns_until_open - max(0, int(sooner)))
        )
        return False
    slot = first_free_hand_slot(player)
    if slot is None:
        return False
    player.hand[slot] = _make_lockbox(LOCKBOX_TURNS)
    return True


def tick_lockboxes(
    player: PlayerState,
    *,
    rng,
    patch: PatchContext,
) -> Optional[Minion]:
    """Count the seat's Lockbox down one turn; open it at zero.

    Returns the minion it paid out, or None. Called at turn start, which is
    what makes "in 5 turns" mean five of the seat's own turns.
    """
    index = find_lockbox(player)
    if index is None:
        return None
    card = player.hand[index]
    remaining = card.turns_until_open - 1
    if remaining > 0:
        player.hand[index] = _make_lockbox(remaining)
        return None

    minion = _roll_golden_tribal_minion(rng=rng, patch=patch)
    player.hand[index] = minion
    return minion


def _roll_golden_tribal_minion(*, rng, patch: PatchContext) -> Minion:
    """A random Golden minion *with a type* — tribeless cards cannot come out."""
    from .triples import make_forged_golden_minion

    candidates = sorted(
        card_id
        for card_id in patch.pool_ids
        if getattr(patch.templates[card_id], "race", None) not in (None, Race.ALL)
    )
    if not candidates:
        raise RuntimeError("patch package has no tribal minion for a Lockbox to open")
    pick = candidates[int(rng.integers(0, len(candidates)))]
    return make_forged_golden_minion(pick, patch=patch)

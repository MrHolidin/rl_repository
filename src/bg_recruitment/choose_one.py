"""Choose One — two effects on one card, and the player takes one.

Eleven live cards print it, on minions and on Tavern spells alike, and four
more exist only to bend it: Thorned Trailblazer and Fandral's Fortune give a
card *both* halves, Bramble Tunneler hands you a random Choose One card, and
Turbo Hogrider watches for one being played.

The two halves are ordinary effects, so "both combined" is not a second code
path — it is applying the pair instead of one of them. That is the whole reason
:class:`ChooseOneEffect` holds effects rather than, say, a pair of card ids.

Opening the choice parks it on ``player.pending_choice`` like Adapt and
Discover do, so a seat can only ever owe one decision at a time. Unlike those,
it offers two options rather than three; the flat action space has no way to
express the pick yet, so the resolver is called directly (same arrangement as
Blood Gems and Spellcraft).
"""

from __future__ import annotations

from typing import Optional

from src.bg_core.effects import ChooseOneEffect, Trigger
from src.bg_core.minion import Minion
from src.bg_lobby.player import PendingChoice, PendingChoiceKind, PlayerState

__all__ = [
    "CHOOSE_ONE_OPTIONS",
    "open_choose_one",
    "resolve_choose_one",
    "grant_combined_choose_one",
    "fire_choose_one_played",
]

#: Every printing offers exactly two.
CHOOSE_ONE_OPTIONS = 2


def _label(effect: object, index: int) -> str:
    """A stable, human-readable option name for replays and the pending block."""
    return f"choose_one:{type(effect).__name__}:{index}"


def open_choose_one(
    player: PlayerState,
    effect: ChooseOneEffect,
    *,
    source: Optional[Minion] = None,
) -> None:
    """Park the two options on the seat, to be resolved by ``resolve_choose_one``."""
    options = (effect.first, effect.second)
    source_idx = None
    if source is not None:
        try:
            source_idx = player.board.index(source)
        except ValueError:
            source_idx = None
    player.pending_choice = PendingChoice(
        kind=PendingChoiceKind.CHOOSE_ONE,
        options=tuple(_label(e, i) for i, e in enumerate(options)),
        extra_modals_after=0,
        effects=options,
        source_board_idx=source_idx,
    )


def grant_combined_choose_one(player: PlayerState, count: int = 1) -> None:
    """Thorned Trailblazer: the next ``count`` Choose One cards take both halves."""
    player.choose_one_combined_charges += max(0, int(count))


def resolve_choose_one(
    player: PlayerState,
    option_index: int,
    *,
    apply_effect,
    fire_played_listeners=None,
) -> None:
    """Take option ``option_index`` — or both halves, if the seat has a charge.

    ``apply_effect(source, effect)`` runs one half; the caller supplies it so
    this module stays out of the shop dispatcher's business.
    """
    pc = player.pending_choice
    if pc is None or pc.kind is not PendingChoiceKind.CHOOSE_ONE:
        raise ValueError("no Choose One is pending")
    if not 0 <= option_index < len(pc.effects):
        raise ValueError(
            f"option {option_index} out of range for {len(pc.effects)} options"
        )

    source = None
    if pc.source_board_idx is not None and pc.source_board_idx < len(player.board):
        source = player.board[pc.source_board_idx]

    combined = player.choose_one_combined_charges > 0
    if combined:
        player.choose_one_combined_charges -= 1
        chosen = list(pc.effects)
    else:
        chosen = [pc.effects[option_index]]

    # Clear before applying: an option that opens its own modal (a Discover,
    # say) must be able to park itself where this one was.
    player.pending_choice = None
    for effect in chosen:
        apply_effect(source, effect)

    if fire_played_listeners is not None:
        fire_played_listeners(player)


def fire_choose_one_played(player: PlayerState, apply_effect) -> None:
    """Turbo Hogrider's trigger: a Choose One card was played this turn."""
    for listener in list(player.board):
        for ability in listener.abilities:
            if ability.trigger is Trigger.ON_CHOOSE_ONE_PLAYED:
                apply_effect(listener, ability.effect)

"""Spellcraft — the Naga keyword: a fresh spell every turn, spent that turn.

A Naga with Spellcraft hands its owner a spell when it is played and again at
the start of every turn. The spell is cast on one minion and its effect lasts
*until next turn*; unspent, the spell itself is discarded at end of turn. So
three separate lifetimes meet here, and each is a different rule:

* the **spell in hand** dies at end of turn (``discard_spellcraft_spells``);
* the **buff it applied** dies at the start of the next turn
  (``expire_temporary_buffs``), one recruit phase later — it has to survive the
  combat it was cast for;
* the **minion** keeps making new spells for as long as it is on the board.

Both expiries run at the seat's own turn boundary, so a spell cast in recruit N
is felt in combat N and gone in recruit N+1, which is what the cards mean.
"""

from __future__ import annotations

from typing import List, Optional

from src.bg_core.effects import (
    Ability,
    CreateSpellcraftSpellEffect,
    GrantTemporaryBuffEffect,
    Trigger,
)
from src.bg_core.board_helpers import fire_spell_cast_on
from src.bg_core.minion import Minion
from src.bg_core.spell_card import SpellCard
from src.bg_lobby.player import PlayerState

from .hand_slots import first_free_hand_slot

__all__ = [
    "can_play_spellcraft_spell",
    "is_spellcraft_spell",
    "make_spellcraft_spell",
    "give_spellcraft_spell",
    "apply_temporary_buff",
    "play_spellcraft_spell_from_hand",
    "expire_temporary_buffs",
    "discard_spellcraft_spells",
]


def is_spellcraft_spell(card) -> bool:
    return isinstance(card, SpellCard) and card.is_spellcraft


def make_spellcraft_spell(effect: CreateSpellcraftSpellEffect) -> SpellCard:
    """The card a Spellcraft minion hands you, built from its own text."""
    return SpellCard(
        card_id=effect.card_id or "SPELLCRAFT",
        name=effect.name,
        cost=0,
        is_spellcraft=True,
        abilities=(Ability(Trigger.ON_PLACE, effect.buff),),
    )


def give_spellcraft_spell(
    player: PlayerState, effect: CreateSpellcraftSpellEffect
) -> bool:
    """Put the spell in hand. Returns False when there was no room for it."""
    slot = first_free_hand_slot(player)
    if slot is None:
        return False
    player.hand[slot] = make_spellcraft_spell(effect)
    return True


def can_play_spellcraft_spell(player: PlayerState) -> bool:
    """Whether a Spellcraft spell in hand has anything to land on.

    With an empty board the spell is not discarded early and does not fizzle —
    it stays in hand, unplayable, until end of turn takes it like any other
    unspent Spellcraft spell. Same rule as a Blood Gem with nowhere to go.
    """
    return bool(player.board)


def apply_temporary_buff(target: Minion, buff: GrantTemporaryBuffEffect) -> None:
    """Stats and keyword that come off again at the owner's next turn."""
    target.temp_attack += buff.attack
    target.temp_health += buff.health
    if buff.keyword is not None:
        race_ok = buff.keyword_if_race is None or target.race == buff.keyword_if_race
        if race_ok:
            target.temp_keywords = frozenset(target.temp_keywords | {buff.keyword})
    fire_spell_cast_on(target)


def play_spellcraft_spell_from_hand(
    player: PlayerState, hand_index: int, board_index: int
) -> None:
    """Cast the spell in ``hand_index`` on the board minion at ``board_index``.

    The RL action space cannot express this yet (it has no "play a spell at a
    target" action), so the engine offers it directly — same arrangement as
    Blood Gems.
    """
    card = player.hand[hand_index]
    if not is_spellcraft_spell(card):
        raise ValueError(f"hand slot {hand_index} does not hold a Spellcraft spell")
    if not 0 <= board_index < len(player.board):
        raise ValueError(f"no minion at board index {board_index}")
    target = player.board[board_index]
    for ability in card.abilities:
        if isinstance(ability.effect, GrantTemporaryBuffEffect):
            apply_temporary_buff(target, ability.effect)
        else:
            raise NotImplementedError(
                f"Spellcraft spell effect {type(ability.effect).__name__} has no "
                f"handler ({card.card_id})"
            )
    player.hand[hand_index] = None


def expire_temporary_buffs(player: PlayerState) -> None:
    """Drop every "until next turn" buff. Called at the seat's turn start.

    Shop-phase minions are never damaged (combat runs on copies), so shedding
    temporary Health here cannot kill anything.
    """
    for minion in player.board:
        minion.temp_attack = 0
        minion.temp_health = 0
        if minion.temp_keywords:
            minion.temp_keywords = frozenset()


def discard_spellcraft_spells(player: PlayerState) -> int:
    """Unspent Spellcraft spells leave hand at end of turn. Returns how many."""
    dropped = 0
    for i, card in enumerate(player.hand):
        if is_spellcraft_spell(card):
            player.hand[i] = None
            dropped += 1
    return dropped

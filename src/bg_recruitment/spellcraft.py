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

import numpy as np

from typing import List, Optional

from src.bg_core.effects import (
    Ability,
    CreateSpellcraftSpellEffect,
    GrantTemporaryBuffEffect,
    Keyword,
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
    "flush_pending_spellcraft",
    "apply_temporary_buff",
    "play_spellcraft_spell_from_hand",
    "expire_temporary_buffs",
    "discard_spellcraft_spells",
]


def is_spellcraft_spell(card) -> bool:
    return isinstance(card, SpellCard) and card.is_spellcraft


def _improved_buff(
    player: PlayerState, effect: CreateSpellcraftSpellEffect
):
    """The spell's buff at the seat's current level.

    "Give a minion +1/+1 until next turn. (Improved by every 4 spells you've
    cast this game!)" — the level is read when the spell is *made*, which is
    what the seat sees on the card in hand.
    """
    from dataclasses import replace

    from .game_counts import improve_level

    level = improve_level(player, effect.counter, effect.per)
    if level == 1 or not isinstance(effect.buff, GrantTemporaryBuffEffect):
        # Only a buff has numbers to multiply; a spell that fetches a card is
        # improved by whatever its own effect says, not by this.
        return effect.buff
    return replace(
        effect.buff,
        attack=effect.buff.attack * level,
        health=effect.buff.health * level,
    )


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
    """Put the spell in hand, or set it aside until a slot opens.

    Blizzard, on the keyword: *"If your hand is full, Spellcraft spells will
    'wait' until there is a free space in your hand, instead of getting
    destroyed or being delayed a turn."* The exception is the keyword's own —
    an ordinary card handed to a full hand is destroyed, and a Blood Gem with
    it. Returns whether it reached the hand now.
    """
    from dataclasses import replace

    spell = make_spellcraft_spell(replace(effect, buff=_improved_buff(player, effect)))
    slot = first_free_hand_slot(player)
    if slot is None:
        player.pending_spellcraft = player.pending_spellcraft + (spell,)
        return False
    player.hand[slot] = spell
    return True


def flush_pending_spellcraft(player: PlayerState) -> int:
    """Hand over as many waiting Spellcraft spells as there is room for.

    Called wherever the hand may have shrunk — the same post-action moment the
    queued Triple reward is flushed at, and for the same reason: the card is
    owed, and the seat should not have to do anything to collect it.
    """
    if not player.pending_spellcraft:
        return 0
    handed = 0
    waiting = list(player.pending_spellcraft)
    while waiting:
        slot = first_free_hand_slot(player)
        if slot is None:
            break
        player.hand[slot] = waiting.pop(0)
        handed += 1
    player.pending_spellcraft = tuple(waiting)
    return handed


def can_play_spellcraft_spell(player: PlayerState) -> bool:
    """Whether a Spellcraft spell in hand has anything to land on.

    With an empty board the spell is not discarded early and does not fizzle —
    it stays in hand, unplayable, until end of turn takes it like any other
    unspent Spellcraft spell. Same rule as a Blood Gem with nowhere to go.
    """
    return bool(player.board)


def _keeps_first_spellcraft(target: Minion) -> bool:
    from src.bg_core.effects import FirstSpellcraftIsPermanentEffect, Trigger

    return any(
        ability.trigger is Trigger.AURA
        and isinstance(ability.effect, FirstSpellcraftIsPermanentEffect)
        for ability in target.abilities
    )


def apply_temporary_buff(
    target: Minion,
    buff: GrantTemporaryBuffEffect,
    *,
    player=None,
    patch=None,
    spell_card_id: str = "",
    from_spell: bool = True,
) -> None:
    """Stats and keyword that come off again at the owner's next turn.

    Unless the body keeps them: Lava Lurker makes the first Spellcraft spell
    cast on it each turn permanent, which is the same buff written to the
    lasting fields instead of the expiring ones.

    ``from_spell`` is false when a hero power hands out the same buff. Nothing
    was cast, so the cards that read a cast — Lava Lurker's permanence, "gain
    +1 Health whenever you cast a spell on this" — do not read this one.
    """
    permanent = (
        from_spell
        and _keeps_first_spellcraft(target)
        and not target.spellcraft_kept_this_turn
    )
    if permanent:
        target.spellcraft_kept_this_turn = True
        target.bonus_attack += buff.attack
        target.bonus_health += buff.health
    else:
        target.temp_attack += buff.attack
        target.temp_health += buff.health
    if buff.keyword is not None:
        race_ok = buff.keyword_if_race is None or target.race == buff.keyword_if_race
        if race_ok:
            if permanent:
                target.granted_keywords = target.granted_keywords | {buff.keyword}
            else:
                target.temp_keywords = frozenset(target.temp_keywords | {buff.keyword})
            # A Divine Shield is two things: the keyword and whether it is
            # still up. Combat asks for both, so granting the keyword alone
            # (Glowscale's whole printing) was worth exactly nothing.
            if buff.keyword is Keyword.SHIELD:
                target.has_shield = True
    if from_spell:
        fire_spell_cast_on(
            target,
            player=player,
            patch=patch,
            spell_card_id=spell_card_id,
            spellcraft=True,
        )


def _count_spell_cast(player: PlayerState, *, patch=None) -> None:
    from .game_counts import SPELLS_CAST, bump_seat_counter

    bump_seat_counter(player, SPELLS_CAST, patch=patch)


def play_spellcraft_spell_from_hand(
    player: PlayerState,
    hand_index: int,
    board_index: Optional[int] = None,
    *,
    shop_index: Optional[int] = None,
    patch=None,
) -> None:
    """Cast the spell in ``hand_index`` on a minion the seat names.

    Usually a friendly on the board. ``shop_index`` names one still on the
    counter instead — Sea Witch Zar'jira's spell is cast at the tavern, the
    same reach an ordinary targeted Tavern spell has.

    The RL action space cannot express this yet (it has no "play a spell at a
    target" action), so the engine offers it directly — same arrangement as
    Blood Gems.
    """
    card = player.hand[hand_index]
    if not is_spellcraft_spell(card):
        raise ValueError(f"hand slot {hand_index} does not hold a Spellcraft spell")
    if shop_index is not None:
        if not 0 <= shop_index < len(player.shop) or player.shop[shop_index] is None:
            raise ValueError(f"no minion in tavern slot {shop_index}")
        target = player.shop[shop_index]
    else:
        if board_index is None or not 0 <= board_index < len(player.board):
            raise ValueError(f"no minion at board index {board_index}")
        target = player.board[board_index]
    for ability in card.abilities:
        if isinstance(ability.effect, GrantTemporaryBuffEffect):
            apply_temporary_buff(
                target,
                ability.effect,
                player=player,
                patch=patch,
                spell_card_id=card.card_id,
            )
        elif patch is not None:
            # A Spellcraft spell that does something other than buff its target
            # — fetch a card, raise a seat bonus — is an ordinary effect, so it
            # goes to the dispatcher that knows them all and raises on the rest.
            from .shop_triggers import ShopTriggers

            from src.bg_core.board_helpers import seat_rng

            ShopTriggers(seat_rng(player), patch=patch).apply_shop_effect(
                player, target, ability.effect, placed=None
            )
        else:
            raise NotImplementedError(
                f"Spellcraft spell effect {type(ability.effect).__name__} needs a "
                f"patch to resolve ({card.card_id})"
            )
    player.hand[hand_index] = None
    _count_spell_cast(player, patch=patch)


def expire_temporary_buffs(player: PlayerState) -> None:
    """Drop every "until next turn" buff. Called at the seat's turn start.

    Every zone the seat owns, not just the board: a Spellcraft spell can be
    cast at a minion on the counter, and buying it would otherwise carry the
    "until next turn" stats past the turn boundary they were named for.

    Shop-phase minions are never damaged (combat runs on copies), so shedding
    temporary Health here cannot kill anything.
    """
    for minion in _owned_minions(player):
        _expire_one(minion)


def _owned_minions(player: PlayerState):
    for minion in player.board:
        yield minion
    for card in player.hand:
        if isinstance(card, Minion):
            yield card
    for card in player.shop:
        if isinstance(card, Minion):
            yield card


def _expire_one(minion: Minion) -> None:
    minion.temp_attack = 0
    minion.temp_health = 0
    if not minion.temp_keywords:
        return
    had_temp_shield = Keyword.SHIELD in minion.temp_keywords
    minion.temp_keywords = frozenset()
    # Only a shield the buff itself put up comes down with it: a minion whose
    # printing or a permanent grant carries Divine Shield keeps its own.
    if had_temp_shield and Keyword.SHIELD not in minion.all_keywords:
        minion.has_shield = False


def discard_spellcraft_spells(player: PlayerState) -> int:
    """Unspent Spellcraft spells leave hand at end of turn. Returns how many.

    A spell still waiting on a hand slot goes with them: waiting is within the
    turn it was made for, not a way of surviving it.
    """
    dropped = len(player.pending_spellcraft)
    player.pending_spellcraft = ()
    for i, card in enumerate(player.hand):
        if is_spellcraft_spell(card):
            player.hand[i] = None
            dropped += 1
    return dropped

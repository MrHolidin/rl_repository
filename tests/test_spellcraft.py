"""Spellcraft — three lifetimes that are easy to confuse for one.

The spell dies at end of turn, the buff it applied dies one turn later, and the
Naga making them stays. Each test below pins one of those boundaries; getting
any of them wrong shows up as a Naga board that either keeps stats it should
have shed or loses them a turn early.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import (
    Ability,
    CreateSpellcraftSpellEffect,
    GrantTemporaryBuffEffect,
    Keyword,
    Trigger,
)
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import spellcraft
from src.bg_recruitment.shop_triggers import ShopTriggers

PATCH_DIR = "data/bgcore/19_6_0_74257"

#: Mini-Myrmidon's shape: "Spellcraft: Give a minion +2 Attack until next turn."
MYRMIDON_SPELL = CreateSpellcraftSpellEffect(
    buff=GrantTemporaryBuffEffect(attack=2),
    card_id="BG23_000t",
    name="Myrmidon's Might",
)


@pytest.fixture(scope="module")
def patch():
    return PatchContext.load(Path(PATCH_DIR))


@pytest.fixture()
def triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


def _minion(card_id: str, atk: int = 1, hp: int = 1, **kw) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, **kw)


def _naga(effect: CreateSpellcraftSpellEffect = MYRMIDON_SPELL) -> Minion:
    return _minion(
        "myrmidon",
        2,
        2,
        race=Race.NAGA,
        abilities=(Ability(Trigger.ON_TURN_START, effect),),
    )


def _player(board=(), hand_size: int = 10, **kw) -> PlayerState:
    base = dict(
        health=40,
        gold=10,
        tavern_tier=1,
        board=list(board),
        shop=[None] * 6,
        hand=[None] * hand_size,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    base.update(kw)
    return PlayerState(**base)


def _spells_in_hand(player):
    return [c for c in player.hand if spellcraft.is_spellcraft_spell(c)]


# --------------------------------------------------------------------------- #
# The spell arrives each turn and is not a Tavern spell.
# --------------------------------------------------------------------------- #


def test_turn_start_hands_out_the_spell(triggers):
    player = _player([_naga()])
    triggers.fire_on_turn_start(player)
    assert len(_spells_in_hand(player)) == 1


def test_each_turn_brings_a_new_one(triggers):
    player = _player([_naga()])
    triggers.fire_on_turn_start(player)
    triggers.fire_on_turn_end(player)  # unused spell is discarded here
    triggers.fire_on_turn_start(player)
    assert len(_spells_in_hand(player)) == 1


def test_two_nagas_hand_out_two_spells(triggers):
    player = _player([_naga(), _naga()])
    triggers.fire_on_turn_start(player)
    assert len(_spells_in_hand(player)) == 2


def test_a_spellcraft_spell_is_not_a_tavern_spell(triggers):
    """The distinction "cast a spell" vs "cast a Tavern spell" rests on this."""
    player = _player([_naga()])
    triggers.fire_on_turn_start(player)
    spell = _spells_in_hand(player)[0]
    assert spell.is_spellcraft and not spell.is_tavern_spell


def test_a_full_hand_loses_the_spell(triggers):
    """No room, no spell — it is not queued for later."""
    player = _player([_naga()], hand_size=1)
    player.hand[0] = _minion("filler")
    triggers.fire_on_turn_start(player)
    assert _spells_in_hand(player) == []


# --------------------------------------------------------------------------- #
# Casting it.
# --------------------------------------------------------------------------- #


def test_casting_buffs_the_chosen_minion(triggers):
    target = _minion("target", 3, 3)
    player = _player([_naga(), target])
    triggers.fire_on_turn_start(player)
    hand_idx = player.hand.index(_spells_in_hand(player)[0])

    spellcraft.play_spellcraft_spell_from_hand(player, hand_idx, board_index=1)
    assert target.raw_attack == 5, "printed 3 plus the spell's +2"
    assert player.hand[hand_idx] is None, "the spell is spent"


def test_a_keyword_spell_lands_only_on_the_right_tribe():
    """Waverider gives everyone +2/+2 but Windfury only to a Naga."""
    buff = GrantTemporaryBuffEffect(
        attack=2, health=2, keyword=Keyword.WINDFURY, keyword_if_race=Race.NAGA
    )
    naga = _minion("naga", 1, 1, race=Race.NAGA)
    beast = _minion("beast", 1, 1, race=Race.BEAST)

    spellcraft.apply_temporary_buff(naga, buff)
    spellcraft.apply_temporary_buff(beast, buff)

    assert Keyword.WINDFURY in naga.all_keywords
    assert Keyword.WINDFURY not in beast.all_keywords
    assert beast.raw_attack == 3 and beast.max_health == 3, "stats land on both"


def test_casting_on_an_empty_slot_is_rejected(triggers):
    player = _player([_naga()])
    triggers.fire_on_turn_start(player)
    hand_idx = player.hand.index(_spells_in_hand(player)[0])
    with pytest.raises(ValueError, match="board index"):
        spellcraft.play_spellcraft_spell_from_hand(player, hand_idx, board_index=3)


# --------------------------------------------------------------------------- #
# The two expiries.
# --------------------------------------------------------------------------- #


def test_the_spell_is_discarded_unused_at_end_of_turn(triggers):
    player = _player([_naga()])
    triggers.fire_on_turn_start(player)
    assert _spells_in_hand(player)
    triggers.fire_on_turn_end(player)
    assert _spells_in_hand(player) == [], "an unspent Spellcraft spell is lost"


def test_the_buff_survives_the_turn_it_was_cast_on(triggers):
    """It is cast for the combat that follows, so end of turn must not clear it."""
    target = _minion("target", 3, 3)
    player = _player([_naga(), target])
    triggers.fire_on_turn_start(player)
    hand_idx = player.hand.index(_spells_in_hand(player)[0])
    spellcraft.play_spellcraft_spell_from_hand(player, hand_idx, board_index=1)

    triggers.fire_on_turn_end(player)
    assert target.raw_attack == 5, "the buff is still on for the combat"


def test_the_buff_is_gone_by_the_next_turn(triggers):
    target = _minion("target", 3, 3)
    player = _player([_naga(), target])
    triggers.fire_on_turn_start(player)
    hand_idx = player.hand.index(_spells_in_hand(player)[0])
    spellcraft.play_spellcraft_spell_from_hand(player, hand_idx, board_index=1)

    triggers.fire_on_turn_end(player)
    triggers.fire_on_turn_start(player)
    assert target.raw_attack == 3, "until next turn means exactly one turn"


def test_expiry_leaves_permanent_buffs_alone(triggers):
    """Only the temporary layer comes off; bought stats stay bought."""
    target = _minion("target", 3, 3)
    target.bonus_attack += 4
    player = _player([_naga(), target])
    spellcraft.apply_temporary_buff(target, GrantTemporaryBuffEffect(attack=2, health=2))
    assert (target.raw_attack, target.max_health) == (9, 5)

    triggers.fire_on_turn_start(player)
    assert (target.raw_attack, target.max_health) == (7, 3)


def test_a_temporary_keyword_expires_too(triggers):
    target = _minion("target", 3, 3, keywords=frozenset({Keyword.TAUNT}))
    player = _player([target])
    spellcraft.apply_temporary_buff(
        target, GrantTemporaryBuffEffect(keyword=Keyword.SHIELD)
    )
    assert Keyword.SHIELD in target.all_keywords

    triggers.fire_on_turn_start(player)
    assert Keyword.SHIELD not in target.all_keywords
    assert Keyword.TAUNT in target.all_keywords, "the printed keyword stays"

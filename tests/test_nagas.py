"""The Naga family: Spellcraft, and the cards that watch a spell land.

Most of this family turned on one constraint — a Spellcraft spell could only
carry a buff that expires. Some Nagas hand out a spell that fetches a card or
raises a seat bonus, and those are ordinary effects wearing a Spellcraft spell
as a wrapper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import spellcraft
from src.bg_recruitment.shop_triggers import ShopTriggers

PATCH_DIR = Path("data/bgcore/36_2_0_248348")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


@pytest.fixture()
def triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


def _card(patch, card_id):
    return make_minion(card_id, patch=patch)


def _player(patch, board=(), **kw) -> PlayerState:
    base = dict(
        health=30,
        gold=10,
        tavern_tier=5,
        board=list(board),
        shop=[None] * 7,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _naga(card_id="n", atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.NAGA)


def _cast_spellcraft(patch, triggers, player, target_index=0):
    """Hand out the seat's Spellcraft spells and cast the first one."""
    triggers.fire_on_turn_start(player)
    slot = next(
        i
        for i, c in enumerate(player.hand)
        if c is not None and spellcraft.is_spellcraft_spell(c)
    )
    spellcraft.play_spellcraft_spell_from_hand(player, slot, target_index, patch=patch)


# --------------------------------------------------------------------------- #
# A Spellcraft spell that is not a buff
# --------------------------------------------------------------------------- #


def test_rimescale_priestess_hands_out_a_spell_that_fetches(patch, triggers):
    priestess = _card(patch, "BG33_319")
    player = _player(patch, [priestess])
    _cast_spellcraft(patch, triggers, player)
    fetched = [
        c
        for c in player.hand
        if c is not None and getattr(c, "is_tavern_spell", False)
    ]
    assert len(fetched) == 1


def test_tranquil_meditative_raises_the_spell_bonus(patch, triggers):
    meditative = _card(patch, "BG32_835")
    player = _player(patch, [meditative])
    _cast_spellcraft(patch, triggers, player)
    assert (player.tavern_spell_bonus_attack, player.tavern_spell_bonus_health) == (1, 1)


def test_darkcrest_strategist_fetches_a_tier_one_naga(patch, triggers):
    strategist = _card(patch, "BG31_920")
    player = _player(patch, [strategist])
    _cast_spellcraft(patch, triggers, player)
    got = [c for c in player.hand if isinstance(c, Minion)]
    assert len(got) == 1
    assert got[0].race == Race.NAGA and got[0].tier == 1


def test_glowscale_still_hands_out_an_ordinary_buff(patch, triggers):
    glowscale = _card(patch, "BG23_008")
    target = _naga()
    player = _player(patch, [glowscale, target])
    _cast_spellcraft(patch, triggers, player, target_index=1)
    assert Keyword.SHIELD in target.all_keywords


# --------------------------------------------------------------------------- #
# Cards that watch a spell land
# --------------------------------------------------------------------------- #


def test_torrential_ruiner_pays_out_for_a_spell_on_any_naga(patch):
    from src.bg_recruitment.blood_gems import play_blood_gem_on

    ruiner = _card(patch, "BG36_622")
    naga = _naga()
    other = Minion(card_id="o", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [ruiner, naga, other])
    play_blood_gem_on(player, naga, patch=patch)
    # Everyone gets +3/+3; the Naga also has the Gem's own +1/+1.
    assert (other.raw_attack, other.max_health) == (4, 4)
    assert (naga.raw_attack, naga.max_health) == (5, 5)


def test_a_spell_on_something_else_does_not_wake_the_ruiner(patch):
    from src.bg_recruitment.blood_gems import play_blood_gem_on

    ruiner = _card(patch, "BG36_622")
    other = Minion(card_id="o", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [ruiner, other])
    play_blood_gem_on(player, other, patch=patch)
    assert (other.raw_attack, other.max_health) == (2, 2)  # just the Gem


def test_abyssal_bruiser_counts_tavern_spells_only(patch):
    from src.bg_recruitment.blood_gems import play_blood_gem_on
    from src.bg_recruitment.game_counts import refresh_count_bonuses
    from src.bg_recruitment.tavern_spells import play_tavern_spell_from_hand

    bruiser = _card(patch, "BG35_921")
    player = _player(patch, [bruiser])
    play_blood_gem_on(player, bruiser, patch=patch)  # a spell, not a Tavern spell
    refresh_count_bonuses(player)
    assert bruiser.raw_attack == bruiser.base_attack + 1  # only the Gem's own +1/+1

    player.hand[0] = patch.tavern_spells["BG28_810"]
    play_tavern_spell_from_hand(player, 0, rng=np.random.default_rng(0), patch=patch)
    refresh_count_bonuses(player)
    assert bruiser.raw_attack == bruiser.base_attack + 1 + 2


# --------------------------------------------------------------------------- #
# The improve tally, on three more cards
# --------------------------------------------------------------------------- #


def test_groundbreaker_grows_when_a_naga_is_played(patch, triggers):
    groundbreaker = _card(patch, "BG31_035")
    player = _player(patch, [groundbreaker])
    triggers.fire_after_friendly_minion_placed(player, _naga())
    assert (groundbreaker.raw_attack, groundbreaker.max_health) == (
        groundbreaker.base_attack + 1,
        groundbreaker.base_health + 1,
    )


def test_groundbreaker_grows_faster_after_four_spells(patch, triggers):
    from src.bg_recruitment.game_counts import SPELLS_CAST

    groundbreaker = _card(patch, "BG31_035")
    player = _player(patch, [groundbreaker])
    player.game_counts[SPELLS_CAST] = 4
    triggers.fire_after_friendly_minion_placed(player, _naga())
    assert groundbreaker.raw_attack == groundbreaker.base_attack + 2


def test_a_beast_played_does_not_feed_the_groundbreaker(patch, triggers):
    groundbreaker = _card(patch, "BG31_035")
    player = _player(patch, [groundbreaker])
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    triggers.fire_after_friendly_minion_placed(player, beast)
    assert groundbreaker.raw_attack == groundbreaker.base_attack


def test_showy_cyclist_pays_the_naga_it_leaves_behind(patch, triggers):
    cyclist = _card(patch, "BG31_925")
    naga = _naga(atk=2, hp=2)
    player = _player(patch, [cyclist, naga])
    triggers.apply_shop_effect(player, cyclist, cyclist.abilities[0].effect, None)
    assert (naga.raw_attack, naga.max_health) == (4, 4)


# --------------------------------------------------------------------------- #
# A spell the card aims itself
# --------------------------------------------------------------------------- #


def _beast(card_id):
    return Minion(card_id=card_id, base_attack=1, base_health=1, tier=1, race=Race.BEAST)


def test_fauna_whisperer_casts_on_both_neighbours(patch, triggers):
    """Natural Blessing pays everyone sharing the *target's* type, so the cast
    on each neighbour reaches every Beast on the board."""
    whisperer = _card(patch, "BG32_837")
    left, right, away = _beast("l"), _beast("r"), _beast("a")
    player = _player(patch, [left, whisperer, right, away])
    triggers.fire_on_turn_end(player)
    # Cast twice — once per neighbour — and every Beast is paid both times.
    assert (left.raw_attack, left.max_health) == (7, 7)
    assert (away.raw_attack, away.max_health) == (7, 7)


def test_a_tribeless_neighbour_shares_a_type_with_nobody(patch, triggers):
    whisperer = _card(patch, "BG32_837")
    plain = Minion(card_id="p", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [plain, whisperer])
    triggers.fire_on_turn_end(player)
    assert (plain.raw_attack, plain.max_health) == (1, 1)

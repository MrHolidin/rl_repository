"""Buying the Tavern spell off the counter.

Engine API only, the arrangement Blood Gems and Spellcraft have: the tavern
offers, the seat buys, the seat plays. `can_buy_tavern_spell` is the question
asked before the fact and `buy_tavern_spell` is the purchase — every refusal in
one is a raise in the other, so a driver that checks first never crashes and one
that does not gets told why.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.envs.minibg  # noqa: F401  (breaks a circular import at collection)
from src.bg_catalog.patch_context import PatchContext
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import tavern_spells as ts
from src.envs.bglike import actions as A

PATCH_DIR = Path("data/bgcore/36_2_0_248348")
HEALTH_SPELL = "BG28_571"  # Hasty Excavation — "costs Health to buy instead of Gold"
CHEAP_SPELL = "BG28_503"


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


def _seat(patch, *, gold=10, tier=4, hero_id=None) -> PlayerState:
    player = PlayerState(
        health=30,
        hero_damage_taken_total=0,
        gold=gold,
        tavern_tier=tier,
        ruleset=patch.meta.ruleset,
        board=[],
        shop=[None] * A.MAX_SHOP_SLOTS,
        hand=[None] * A.HAND_SIZE,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    player.round_number = 6
    if hero_id is not None:
        player.hero = patch.heroes[hero_id]
    return player


def _offer(patch, player, card_id=None):
    ts.offer_tavern_spells(
        player,
        rng=np.random.default_rng(0),
        patch=patch,
        card_ids=[card_id] if card_id else None,
    )
    return player.tavern_spell_offers[0]


# --------------------------------------------------------------------------- #
# The purchase
# --------------------------------------------------------------------------- #


def test_the_spell_moves_from_the_counter_to_the_hand(patch):
    player = _seat(patch)
    spell = _offer(patch, player)
    gold = player.gold

    assert ts.can_buy_tavern_spell(player, 0)
    assert ts.buy_tavern_spell(player, 0, patch=patch) is spell

    assert player.hand[0] is spell
    assert player.tavern_spell_offers == ()
    assert player.gold == gold - spell.cost
    # And it is gone from the counter, so it cannot be bought twice.
    assert not ts.can_buy_tavern_spell(player, 0)


def test_asking_first_and_being_refused_are_the_same_question(patch):
    """Every refusal `can_buy_tavern_spell` gives is one `buy_tavern_spell`
    raises, so the two can never disagree about what is allowed."""
    cases = []

    empty = _seat(patch)  # nothing on the counter
    cases.append(empty)

    broke = _seat(patch, gold=0)
    _offer(patch, broke, CHEAP_SPELL)
    cases.append(broke)

    full = _seat(patch)
    _offer(patch, full)
    full.hand = [patch.make_minion("BGS_119") for _ in full.hand]
    cases.append(full)

    finished = _seat(patch)
    _offer(patch, finished)
    finished.phase = PlayerPhase.DONE  # buying is a recruit-phase move
    cases.append(finished)

    for player in cases:
        assert not ts.can_buy_tavern_spell(player, 0)
        with pytest.raises(ts.TavernSpellNotAllowed):
            ts.buy_tavern_spell(player, 0, patch=patch)


def test_a_spell_bought_is_a_spell_playable(patch):
    """The point of the purchase: what it puts in hand is a card the seat can
    then cast, at a target when the card names one."""
    player = _seat(patch, gold=20)
    _offer(patch, player, CHEAP_SPELL)
    ts.buy_tavern_spell(player, 0, patch=patch)
    player.board.append(patch.make_minion("BGS_119"))

    ts.play_tavern_spell_from_hand(
        player, 0, target_board_index=0, rng=np.random.default_rng(0), patch=patch
    )
    assert player.hand[0] is None


# --------------------------------------------------------------------------- #
# What it costs
# --------------------------------------------------------------------------- #


def test_a_spell_bought_in_health_does_not_ask_for_gold(patch):
    player = _seat(patch, gold=0)
    spell = _offer(patch, player, HEALTH_SPELL)
    assert ts.spell_costs_health(spell)
    assert ts.can_buy_tavern_spell(player, 0)

    player.armor = 1
    health = player.health
    ts.buy_tavern_spell(player, 0, patch=patch)
    # Armor absorbs first, the way it does for any other hero damage.
    assert (player.health, player.armor) == (health - (spell.cost - 1), 0)
    assert player.gold == 0  # buying is not casting; the card pays out when played


def test_the_tavern_does_not_offer_a_purchase_that_would_kill(patch):
    player = _seat(patch, gold=0)
    spell = _offer(patch, player, HEALTH_SPELL)
    player.health, player.armor = spell.cost, 0
    assert not ts.can_buy_tavern_spell(player, 0)
    player.health = spell.cost + 1
    assert ts.can_buy_tavern_spell(player, 0)


def test_taethelans_every_third_spell_is_free(patch):
    """A passive that had never fired: nothing was calling the purchase it
    counts."""
    hero = next(
        hid for hid, h in patch.heroes.items() if h.name == "Tae'thelan Bloodwatcher"
    )
    player = _seat(patch, gold=99, hero_id=hero)
    paid = []
    for _ in range(6):
        _offer(patch, player)
        before = player.gold
        ts.buy_tavern_spell(player, 0, patch=patch)
        paid.append(before - player.gold)
        player.hand = [None] * len(player.hand)
    assert paid[2] == 0 and paid[5] == 0
    assert all(p > 0 for i, p in enumerate(paid) if i not in (2, 5))


def test_the_classic_packages_have_no_spell_to_buy(patch):
    """They carry no pool spells at all, so nothing there can be bought."""
    for package in ("data/bgcore/19_6_0_74257", "data/bgcore/15_6_2_36393"):
        ctx = PatchContext.load(Path(package))
        assert ts.tavern_spell_pool(6, patch=ctx) == [], package
        assert not ts.can_buy_tavern_spell(_seat(ctx), 0)

"""Tavern spells: offered, bought, cast.

Three separate moves, tested as three separate things — a bought spell sits in
hand until the seat plays it, and a spell can reach hand without ever having
been on the counter. The cards are the real tier-1 spells out of the 36.2.0
catalog, so a binding pointing at the wrong effect fails here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion
from src.bg_core.spell_card import SpellCard
from src.bg_lobby.player import PendingChoiceKind, PlayerPhase, PlayerState
from src.bg_recruitment.shop import effective_shop_offers_count, refresh_shop
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.tavern_spells import (
    TavernSpellNotAllowed,
    buy_tavern_spell,
    effective_tavern_spell_cost,
    offer_tavern_spells,
    play_tavern_spell_from_hand,
    tavern_spell_pool,
)

PATCH_DIR = Path("data/bgcore/36_2_0_248348")
CLASSIC_DIR = Path("data/bgcore/19_6_0_74257")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


def _player(patch: PatchContext, board=(), shop=None, **kw) -> PlayerState:
    base = dict(
        health=30,
        gold=10,
        tavern_tier=1,
        board=list(board),
        shop=list(shop) if shop is not None else [None] * 7,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _minion(card_id: str = "m", atk: int = 1, hp: int = 1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1)


def _rng(seed: int = 0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# The catalog
# --------------------------------------------------------------------------- #


def test_the_package_carries_its_tavern_spells(patch):
    pool = tavern_spell_pool(1, patch=patch)
    assert len(pool) == 8
    assert all(patch.tavern_spells[cid].is_tavern_spell for cid in pool)


def test_a_patch_older_than_tavern_spells_has_none():
    classic = PatchContext.load(CLASSIC_DIR)
    assert tavern_spell_pool(6, patch=classic) == []


def test_the_pool_grows_with_the_seats_tier(patch):
    assert len(tavern_spell_pool(1, patch=patch)) < len(tavern_spell_pool(3, patch=patch))


# --------------------------------------------------------------------------- #
# Offered, bought
# --------------------------------------------------------------------------- #


def test_a_rolled_tavern_shows_a_spell_beside_a_full_minion_row(patch):
    """One per roll, and it costs the seat nothing: a tier-1 tavern shows three
    minions *and* a spell."""
    player = _player(patch, tavern_tier=1)
    refresh_shop(player, None, rng=_rng(), patch=patch)
    minions = sum(1 for m in player.shop if m is not None)
    assert len(player.tavern_spell_offers) == 1
    assert minions == effective_shop_offers_count(player) == 3


def test_the_minion_row_is_the_same_size_it_was_without_spells(patch):
    for tier in (1, 2, 3, 4):
        player = _player(patch, tavern_tier=tier)
        refresh_shop(player, None, rng=_rng(), patch=patch)
        assert sum(1 for m in player.shop if m is not None) == (
            effective_shop_offers_count(player)
        )


def test_how_many_spells_a_roll_shows_is_a_ruleset_number(patch):
    from dataclasses import replace

    player = _player(patch, tavern_tier=3)
    player.ruleset = replace(patch.meta.ruleset, tavern_spells_per_roll=2)
    refresh_shop(player, None, rng=_rng(), patch=patch)
    assert len(player.tavern_spell_offers) == 2
    assert sum(1 for m in player.shop if m is not None) == (
        effective_shop_offers_count(player)
    )


def test_a_patch_with_no_spells_rolls_a_full_row_of_minions():
    """The 2021 packages predate Tavern spells; their tavern is unchanged."""
    classic = PatchContext.load(CLASSIC_DIR)
    player = _player(classic, tavern_tier=3)
    refresh_shop(player, None, rng=_rng(), patch=classic)
    assert player.tavern_spell_offers == ()
    assert sum(1 for m in player.shop if m is not None) == effective_shop_offers_count(
        player
    )


def test_a_frozen_tavern_keeps_the_spell_it_was_frozen_with(patch):
    player = _player(patch, tavern_tier=3)
    refresh_shop(player, None, rng=_rng(1), patch=patch)
    kept = player.tavern_spell_offers
    frozen = (True,) * len(player.shop)
    refresh_shop(player, None, rng=_rng(2), patch=patch, frozen_slots=frozen)
    assert player.tavern_spell_offers == kept


def test_rolling_again_replaces_the_spell_on_the_counter(patch):
    player = _player(patch, tavern_tier=3)
    refresh_shop(player, None, rng=_rng(1), patch=patch)
    refresh_shop(player, None, rng=_rng(2), patch=patch)
    assert len(player.tavern_spell_offers) == 1


def test_buying_pays_the_printed_cost_and_lands_in_hand(patch):
    player = _player(patch, gold=5)
    offer_tavern_spells(player, rng=_rng(), patch=patch, card_ids=["BG28_512"])  # cost 2
    spell = buy_tavern_spell(player, patch=patch)
    assert player.gold == 3
    assert player.hand[0] is spell and spell.card_id == "BG28_512"
    assert not player.tavern_spell_offers


def test_buying_without_the_gold_is_refused_and_spends_nothing(patch):
    player = _player(patch, gold=1)
    offer_tavern_spells(player, rng=_rng(), patch=patch, card_ids=["BG28_512"])  # cost 2
    with pytest.raises(TavernSpellNotAllowed):
        buy_tavern_spell(player, patch=patch)
    assert player.gold == 1 and player.tavern_spell_offers


def test_buying_an_empty_counter_is_refused(patch):
    with pytest.raises(TavernSpellNotAllowed):
        buy_tavern_spell(_player(patch), patch=patch)


# --------------------------------------------------------------------------- #
# Ominous Seer's discount
# --------------------------------------------------------------------------- #


def test_ominous_seer_discounts_the_next_spell(patch):
    seer = make_minion("BG31_330", patch=patch)
    player = _player(patch, [seer], gold=5)
    ShopTriggers(_rng(), patch=patch).fire_on_place(seer, player, None)
    assert player.tavern_spell_cost_delta == -1

    offer_tavern_spells(player, rng=_rng(), patch=patch, card_ids=["BG28_512"])  # cost 2
    assert effective_tavern_spell_cost(player, player.tavern_spell_offers[0]) == 1
    buy_tavern_spell(player, patch=patch)
    assert player.gold == 4


def test_the_discount_is_spent_by_the_purchase_that_used_it(patch):
    seer = make_minion("BG31_330", patch=patch)
    player = _player(patch, [seer], gold=10)
    ShopTriggers(_rng(), patch=patch).fire_on_place(seer, player, None)
    offer_tavern_spells(player, rng=_rng(), patch=patch, card_ids=["BG28_512"])
    buy_tavern_spell(player, patch=patch)
    assert player.tavern_spell_cost_delta == 0
    offer_tavern_spells(player, rng=_rng(), patch=patch, card_ids=["BG28_512"])
    assert effective_tavern_spell_cost(player, player.tavern_spell_offers[0]) == 2


def test_a_discount_never_makes_a_spell_pay_the_seat(patch):
    player = _player(patch, gold=5, tavern_spell_cost_delta=-5)
    offer_tavern_spells(player, rng=_rng(), patch=patch, card_ids=["BG28_810"])  # cost 1
    assert effective_tavern_spell_cost(player, player.tavern_spell_offers[0]) == 0
    buy_tavern_spell(player, patch=patch)
    assert player.gold == 5


def test_the_upgrade_discount_and_the_spell_discount_are_different_purses(patch):
    seer = make_minion("BG31_330", patch=patch)
    player = _player(patch, [seer], gold=10)
    before = player.upgrade_cost_delta
    ShopTriggers(_rng(), patch=patch).fire_on_place(seer, player, None)
    assert player.upgrade_cost_delta == before


# --------------------------------------------------------------------------- #
# Cast
# --------------------------------------------------------------------------- #


def _cast(patch, player, card_id: str, **kw):
    player.hand[0] = patch.tavern_spells[card_id]
    play_tavern_spell_from_hand(player, 0, rng=_rng(), patch=patch, **kw)


def test_tavern_coin_pays_a_gold(patch):
    player = _player(patch, gold=3)
    _cast(patch, player, "BG28_810")
    assert player.gold == 4


def test_fortify_gives_health_and_taunt_to_the_named_minion(patch):
    target, other = _minion("a"), _minion("b")
    player = _player(patch, [target, other])
    _cast(patch, player, "BG28_503", target_board_index=0)
    assert target.max_health == 4 and Keyword.TAUNT in target.all_keywords
    assert other.max_health == 1


def test_tavern_dish_banana_buffs_the_named_minion(patch):
    target = _minion("a")
    player = _player(patch, [target])
    _cast(patch, player, "BG28_897", target_board_index=0)
    assert (target.raw_attack, target.max_health) == (3, 3)


def test_them_apples_buffs_the_counter_not_the_board(patch):
    on_board = _minion("board")
    in_shop = _minion("shop")
    player = _player(patch, [on_board], shop=[in_shop] + [None] * 6)
    _cast(patch, player, "BG28_966")
    assert (in_shop.raw_attack, in_shop.max_health) == (2, 3)
    assert (on_board.raw_attack, on_board.max_health) == (1, 1)


def test_recruit_a_trainee_hands_over_a_tier_one_minion(patch):
    player = _player(patch, tavern_tier=4)
    _cast(patch, player, "BG28_504")
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1 and got[0].tier == 1


def test_enchanted_lasso_takes_a_minion_off_the_counter_for_free(patch):
    stolen = _minion("s0")
    player = _player(patch, shop=[stolen] + [None] * 6, gold=7)
    _cast(patch, player, "BG28_512")
    assert player.shop[0] is None
    assert any(c is stolen for c in player.hand)
    assert player.gold == 7  # a steal is not a purchase


def test_a_lasso_with_an_empty_tavern_does_nothing(patch):
    player = _player(patch)
    _cast(patch, player, "BG28_512")
    assert all(c is None for c in player.hand)


def test_a_new_sprout_opens_a_tier_one_discover(patch):
    player = _player(patch, tavern_tier=5)
    _cast(patch, player, "BG33_101")
    pc = player.pending_choice
    assert pc is not None and pc.kind == PendingChoiceKind.TAVERN_SPELL_DISCOVER
    assert len(pc.options) == 3
    assert all(patch.templates[cid].tier == 1 for cid in pc.options)


@pytest.mark.parametrize("option,expected", [(0, (4, 2)), (1, (2, 4))])
def test_alliance_flag_applies_the_half_the_seat_chose(patch, option, expected):
    target = _minion("a")
    player = _player(patch, [target])
    _cast(patch, player, "BG31_880", target_board_index=0, choose_one_option=option)
    assert (target.raw_attack, target.max_health) == expected


def test_a_cast_spell_leaves_the_hand(patch):
    player = _player(patch, [_minion("a")])
    _cast(patch, player, "BG28_897", target_board_index=0)
    assert player.hand[0] is None


def test_a_minion_in_hand_is_not_castable(patch):
    player = _player(patch)
    player.hand[0] = make_minion("BG31_803", patch=patch)
    with pytest.raises(TavernSpellNotAllowed):
        play_tavern_spell_from_hand(player, 0, rng=_rng(), patch=patch)


def test_a_blood_gem_is_not_a_tavern_spell(patch):
    """The distinction the SpellCard docstring rests on, kept honest."""
    from src.bg_recruitment.blood_gems import make_blood_gem

    player = _player(patch, [_minion("a")])
    player.hand[0] = make_blood_gem()
    with pytest.raises(TavernSpellNotAllowed):
        play_tavern_spell_from_hand(player, 0, rng=_rng(), patch=patch)

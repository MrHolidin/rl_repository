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


@pytest.fixture()
def triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


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


# --------------------------------------------------------------------------- #
# The verbs the pool asks for: get, Discover, cast, copy — and the cast trigger
# --------------------------------------------------------------------------- #


def test_getting_random_spells_fills_the_hand(patch):
    from src.bg_recruitment.tavern_spells import add_random_tavern_spells

    player = _player(patch, tavern_tier=3)
    assert add_random_tavern_spells(player, count=2, rng=_rng(), patch=patch) == 2
    got = [c for c in player.hand if c is not None]
    assert len(got) == 2 and all(s.is_tavern_spell for s in got)


def test_a_cost_filter_only_offers_what_it_can_afford(patch):
    """"Get two 1-Cost Tavern spells" — the filter is on the card, not the seat."""
    from src.bg_recruitment.tavern_spells import add_random_tavern_spells

    player = _player(patch, tavern_tier=6)
    add_random_tavern_spells(player, count=4, max_cost=1, rng=_rng(), patch=patch)
    assert all(c.cost <= 1 for c in player.hand if c is not None)


def test_the_gives_stats_filter_reads_the_bindings_not_the_text(patch):
    from src.bg_recruitment.tavern_spells import add_random_tavern_spells, spell_gives_stats

    assert spell_gives_stats(patch.tavern_spells["BG28_897"])  # Tavern Dish Banana
    assert not spell_gives_stats(patch.tavern_spells["BG28_810"])  # Tavern Coin
    player = _player(patch, tavern_tier=3)
    add_random_tavern_spells(player, count=4, gives_stats=True, rng=_rng(), patch=patch)
    assert all(spell_gives_stats(c) for c in player.hand if c is not None)


def test_a_full_hand_takes_what_fits(patch):
    from src.bg_recruitment.tavern_spells import add_random_tavern_spells

    player = _player(patch)
    for i in range(9):
        player.hand[i] = _minion(f"h{i}")
    assert add_random_tavern_spells(player, count=3, rng=_rng(), patch=patch) == 1


def test_discovering_a_spell_offers_three_spells(patch):
    from src.bg_recruitment.tavern_spells import open_tavern_spell_discover

    player = _player(patch, tavern_tier=3)
    assert open_tavern_spell_discover(player, rng=_rng(), patch=patch)
    pc = player.pending_choice
    assert pc.kind == PendingChoiceKind.SPELL_DISCOVER
    assert len(set(pc.options)) == 3
    assert all(patch.tavern_spells[o].is_tavern_spell for o in pc.options)


def test_the_discovered_spell_lands_in_hand_as_a_spell(patch):
    from src.bg_recruitment.discover import resolve_discover_pick
    from src.bg_recruitment.tavern_spells import open_tavern_spell_discover

    player = _player(patch, tavern_tier=3)
    open_tavern_spell_discover(player, rng=_rng(), patch=patch)
    wanted = player.pending_choice.options[1]
    resolve_discover_pick(
        player, 1, None, rng=_rng(), on_after_placed=lambda p, m: None, patch=patch
    )
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1
    assert isinstance(got[0], SpellCard) and got[0].card_id == wanted


def test_clever_castaway_discovers_one_for_two_gold(patch):
    from src.bg_recruitment.activate import activate_minion

    castaway = make_minion("BG36_342", patch=patch)
    player = _player(patch, [castaway], gold=5, tavern_tier=3)
    activate_minion(player, 0, rng=_rng(), patch=patch)
    assert player.pending_choice.kind == PendingChoiceKind.SPELL_DISCOVER
    assert player.gold == 3


def test_casting_remembers_which_spell_it_was(patch):
    player = _player(patch, gold=5)
    _cast(patch, player, "BG28_810")
    assert player.last_tavern_spell_cast == "BG28_810"


def test_a_copy_of_the_last_spell_is_a_card_not_a_cast(patch):
    from src.bg_recruitment.tavern_spells import apply_tavern_spell_effect
    from src.bg_core.effects import CopyLastTavernSpellEffect

    player = _player(patch, gold=3)
    _cast(patch, player, "BG28_810")  # Tavern Coin: +1 gold
    gold_after_cast = player.gold
    apply_tavern_spell_effect(
        player, CopyLastTavernSpellEffect(), rng=_rng(), patch=patch
    )
    assert player.gold == gold_after_cast  # the copy was not cast
    assert any(c is not None and c.card_id == "BG28_810" for c in player.hand)


def test_casting_a_random_spell_costs_nothing_and_keeps_no_card(patch):
    from src.bg_core.effects import CastRandomTavernSpellEffect
    from src.bg_recruitment.tavern_spells import apply_tavern_spell_effect

    target = _minion("t")
    player = _player(patch, [target], gold=4, tavern_tier=1)
    apply_tavern_spell_effect(
        player, CastRandomTavernSpellEffect(), rng=_rng(7), patch=patch, source=target
    )
    assert all(c is None for c in player.hand)  # nothing acquired
    assert player.last_tavern_spell_cast is not None


def test_a_cast_wakes_the_listeners_on_the_board(patch):
    from src.bg_core.effects import Ability, BuffSelf, Trigger

    listener = Minion(
        card_id="listener",
        base_attack=1,
        base_health=1,
        tier=1,
        abilities=(Ability(Trigger.ON_TAVERN_SPELL_CAST, BuffSelf(attack=1, health=1)),),
    )
    player = _player(patch, [listener])
    _cast(patch, player, "BG28_810")
    assert (listener.raw_attack, listener.max_health) == (2, 2)


def test_a_blood_gem_is_not_a_tavern_spell_for_the_listeners(patch):
    """The distinction SpellCard draws, kept honest on the new trigger too."""
    from src.bg_core.effects import Ability, BuffSelf, Trigger
    from src.bg_recruitment.blood_gems import play_blood_gem_on

    listener = Minion(
        card_id="listener",
        base_attack=1,
        base_health=1,
        tier=1,
        abilities=(Ability(Trigger.ON_TAVERN_SPELL_CAST, BuffSelf(attack=1, health=1)),),
    )
    player = _player(patch, [listener])
    play_blood_gem_on(player, listener)
    assert listener.raw_attack == 2  # the Gem's own +1/+1, and nothing else
    assert listener.max_health == 2


# --------------------------------------------------------------------------- #
# "Your Tavern spells give an extra +N"
# --------------------------------------------------------------------------- #


def _choose(patch, triggers, player, source, option: int) -> None:
    """Play a Choose One minion and take one half, the way the seat does."""
    from src.bg_recruitment.choose_one import resolve_choose_one

    triggers.fire_on_place(source, player, None)
    resolve_choose_one(
        player,
        option,
        apply_effect=lambda src, eff: triggers.apply_shop_effect(player, src, eff, None),
    )


def test_the_spell_bonus_is_added_to_what_a_spell_hands_out(patch, triggers):
    botanist = make_minion("BG32_237", patch=patch)  # Choose One; first half: +1 Attack
    target = _minion("t")
    player = _player(patch, [botanist, target])
    _choose(patch, triggers, player, botanist, 0)
    assert (player.tavern_spell_bonus_attack, player.tavern_spell_bonus_health) == (1, 0)

    _cast(patch, player, "BG28_897", target_board_index=1)  # Banana: +2/+2
    assert (target.raw_attack, target.max_health) == (4, 3)


def test_the_other_half_of_the_choice_gives_health(patch, triggers):
    botanist = make_minion("BG32_237", patch=patch)
    player = _player(patch, [botanist])
    _choose(patch, triggers, player, botanist, 1)
    assert (player.tavern_spell_bonus_attack, player.tavern_spell_bonus_health) == (0, 1)


def test_the_spell_bonus_does_not_touch_blood_gems(patch, triggers):
    """Different buffs: raising one says nothing about the other."""
    botanist = make_minion("BG32_237", patch=patch)
    target = _minion("t")
    player = _player(patch, [botanist, target])
    _choose(patch, triggers, player, botanist, 0)
    from src.bg_recruitment.blood_gems import play_blood_gem_on

    play_blood_gem_on(player, target)
    assert (target.raw_attack, target.max_health) == (2, 2)

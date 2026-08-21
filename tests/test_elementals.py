"""The Elemental family, and the field split it forced.

"Your Elementals give an extra +2 Health" modifies what *each* Elemental-played
grant hands out — a modifier on a modifier. The seat had one integer serving as
both the running total and the per-grant value, applied symmetrically, so a
Health-only bonus could not be said at all. It is a pair now, with the extra
kept beside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment.shop_triggers import ShopTriggers

PATCH_DIR = Path("data/bgcore/36_2_0_248348")
CLASSIC_DIR = Path("data/bgcore/19_6_0_74257")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


@pytest.fixture()
def triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


def _card(patch, card_id):
    return make_minion(card_id, patch=patch)


def _player(patch, board=(), shop=(), **kw) -> PlayerState:
    base = dict(
        health=30,
        gold=10,
        tavern_tier=5,
        board=list(board),
        shop=list(shop) + [None] * (7 - len(shop)),
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _elemental(card_id="e", atk=1, hp=1) -> Minion:
    return Minion(
        card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.ELEMENTAL
    )


# --------------------------------------------------------------------------- #
# The split
# --------------------------------------------------------------------------- #


def test_nomi_still_grants_symmetrically(patch, triggers):
    """The classic shape: what the granting card prints, both halves."""
    from src.bg_core.effects import IncrementShopTribeBonusEffect

    on_offer = _elemental("s")
    player = _player(patch, shop=[on_offer])
    triggers.apply_shop_effect(
        player,
        None,
        IncrementShopTribeBonusEffect(tribe=Race.ELEMENTAL, attack=4, health=4),
        None,
    )
    assert (on_offer.raw_attack, on_offer.max_health) == (5, 5)
    assert (player.shop_elemental_bonus, player.shop_elemental_bonus_health) == (4, 4)


def test_glowing_cinder_raises_what_every_elemental_gives(patch, triggers):
    """Not the tavern bonus — *whatever an Elemental gives*. Molten Rock's own
    +1 Health is an Elemental giving something, so it grows too."""
    cinder = _card(patch, "BG32_842")
    player = _player(patch)
    triggers.apply_shop_effect(player, cinder, cinder.abilities[0].effect, None)
    assert (player.elemental_gift_attack, player.elemental_gift_health) == (0, 2)

    rock = _card(patch, "BGS_127")  # Molten Rock: +1 Health per Elemental played
    player.board.append(rock)
    triggers.fire_after_friendly_minion_placed(player, _elemental())
    assert rock.max_health == rock.base_health + 3  # +1 printed, +2 extra


def test_the_gift_reaches_a_tavern_grant_too(patch, triggers):
    """Nomi is one of the things an Elemental gives, not the only one."""
    from src.bg_core.effects import IncrementShopTribeBonusEffect

    cinder = _card(patch, "BG32_842")
    giver = _elemental("giver")
    player = _player(patch, [giver])
    triggers.apply_shop_effect(player, cinder, cinder.abilities[0].effect, None)
    on_offer = _elemental("s")
    player.shop[0] = on_offer
    triggers.apply_shop_effect(
        player,
        giver,
        IncrementShopTribeBonusEffect(tribe=Race.ELEMENTAL, attack=1, health=1),
        None,
    )
    assert (on_offer.raw_attack, on_offer.max_health) == (2, 4)


def test_a_buff_from_something_else_is_not_amplified(patch, triggers):
    """The gift is about what *Elementals* give."""
    from src.bg_core.effects import BuffSelf

    cinder = _card(patch, "BG32_842")
    player = _player(patch)
    triggers.apply_shop_effect(player, cinder, cinder.abilities[0].effect, None)
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player.board.append(beast)
    triggers.apply_shop_effect(player, beast, BuffSelf(attack=0, health=1), None)
    assert beast.max_health == 2


def test_sand_swirler_raises_only_the_attack_half(patch, triggers):
    swirler = _card(patch, "BG32_841")
    player = _player(patch, [swirler])
    triggers.fire_on_place(swirler, player, None)
    assert (player.elemental_gift_attack, player.elemental_gift_health) == (1, 0)


def test_the_extras_stack(patch, triggers):
    swirler = _card(patch, "BG32_841")
    cinder = _card(patch, "BG32_842")
    player = _player(patch, [swirler])
    triggers.fire_on_place(swirler, player, None)
    triggers.apply_shop_effect(player, cinder, cinder.abilities[0].effect, None)
    assert (player.elemental_gift_attack, player.elemental_gift_health) == (1, 2)


def test_a_later_elemental_arrives_carrying_the_total(patch, triggers):
    from src.bg_core.effects import IncrementShopTribeBonusEffect
    from src.bg_recruitment.shop import apply_shop_tribe_bonus_to_minion

    player = _player(patch)
    triggers.apply_shop_effect(
        player,
        None,
        IncrementShopTribeBonusEffect(tribe=Race.ELEMENTAL, attack=2, health=3),
        None,
    )
    newcomer = _elemental("late")
    apply_shop_tribe_bonus_to_minion(newcomer, player)
    assert (newcomer.raw_attack, newcomer.max_health) == (3, 4)


def test_a_non_elemental_carries_nothing(patch, triggers):
    from src.bg_core.effects import IncrementShopTribeBonusEffect
    from src.bg_recruitment.shop import apply_shop_tribe_bonus_to_minion

    player = _player(patch)
    triggers.apply_shop_effect(
        player,
        None,
        IncrementShopTribeBonusEffect(tribe=Race.ELEMENTAL, attack=2, health=3),
        None,
    )
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    apply_shop_tribe_bonus_to_minion(beast, player)
    assert (beast.raw_attack, beast.max_health) == (1, 1)


def test_nomi_on_the_classic_package_is_unchanged():
    """The field the observation reads keeps its old meaning: on a patch with no
    "gives an extra" card, the two halves always move together."""
    import numpy as _np

    classic = PatchContext.load(CLASSIC_DIR)
    nomi = classic.make_minion("BGS_104")
    shop_elem = classic.make_minion("BGS_116")
    player = PlayerState(
        health=40, gold=10, tavern_tier=3, board=[nomi], shop=[shop_elem] + [None] * 5,
        hand=[None] * 10, phase=PlayerPhase.SHOP, shop_actions_used=0,
        ruleset=classic.meta.ruleset,
    )
    t = ShopTriggers(_np.random.default_rng(0), patch=classic)
    t.fire_after_friendly_minion_placed(player, classic.make_minion("BGS_115"))
    assert player.shop_elemental_bonus == player.shop_elemental_bonus_health == 1
    assert (shop_elem.bonus_attack, shop_elem.bonus_health) == (1, 1)


# --------------------------------------------------------------------------- #
# The rest of the family
# --------------------------------------------------------------------------- #


def test_dancing_barnstormer_pays_the_tavern_twice_over(patch, triggers):
    barnstormer = _card(patch, "BG26_162")
    elemental = _elemental("s")
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, [barnstormer], shop=[elemental, beast])
    triggers.fire_on_place(barnstormer, player, None)
    assert (elemental.raw_attack, elemental.max_health) == (9, 9)
    assert (beast.raw_attack, beast.max_health) == (1, 1)


def test_flourishing_frostling_counts_elementals_played(patch, triggers):
    from src.bg_recruitment.game_counts import refresh_count_bonuses

    frostling = _card(patch, "BG26_537")
    player = _player(patch, [frostling])
    # "Played", which is the tally fire_on_place keeps — not the after-placed
    # listener pass, which is a different question about the same moment.
    triggers.fire_on_place(_elemental(), player, None)
    refresh_count_bonuses(player)
    assert (frostling.raw_attack, frostling.max_health) == (
        frostling.base_attack + 2,
        frostling.base_health + 1,
    )


def test_unleashed_mana_surge_pays_the_elementals(patch, triggers):
    surge = _card(patch, "BG32_846")
    elemental = _elemental("e", 1, 1)
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, [surge, elemental, beast])
    triggers.fire_after_friendly_minion_placed(player, _elemental("played"))
    assert (elemental.raw_attack, elemental.max_health) == (5, 5)
    assert (beast.raw_attack, beast.max_health) == (1, 1)


def test_air_baller_shares_the_baller_tally(patch, triggers):
    friend = Minion(card_id="f", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [friend])
    triggers.fire_on_sell(_card(patch, "BG31_816"), player)  # Fire Baller: +1 Attack
    triggers.fire_on_sell(_card(patch, "BG36_181"), player)  # Air Baller, now level 2
    assert (friend.raw_attack, friend.max_health) == (1 + 1 + 4, 1 + 4)


# --------------------------------------------------------------------------- #
# Gold spent, and the card you buy
# --------------------------------------------------------------------------- #


def test_air_revenant_answers_every_seven_gold(patch, triggers):
    """Its spell is a standing tavern promise, so the answer is visible as one."""
    revenant = _card(patch, "BG34_858")
    player = _player(patch, [revenant])
    triggers.fire_gold_spent(player, 4)
    assert player.refresh_buffs == ()
    triggers.fire_gold_spent(player, 3)  # seven in total
    assert player.refresh_buffs == ((6, 6),)


def test_the_gold_countdown_is_each_bodys_own(patch, triggers):
    first = _card(patch, "BG34_858")
    second = _card(patch, "BG34_858")
    player = _player(patch, [first, second])
    triggers.fire_gold_spent(player, 3)
    assert first.gold_spent_seen == second.gold_spent_seen == 3


def test_stone_age_slab_pays_the_minion_you_bought(patch, triggers):
    slab = _card(patch, "BG34_950")
    bought = Minion(card_id="b", base_attack=2, base_health=3, tier=1)
    player = _player(patch, [slab])
    triggers.fire_on_bought(player, bought)
    # +10/+10 first, then doubled: 12/13 -> 24/26.
    assert (bought.raw_attack, bought.max_health) == (24, 26)


def test_the_slab_answers_once_a_turn(patch, triggers):
    slab = _card(patch, "BG34_950")
    player = _player(patch, [slab])
    first = Minion(card_id="b1", base_attack=1, base_health=1, tier=1)
    second = Minion(card_id="b2", base_attack=1, base_health=1, tier=1)
    triggers.fire_on_bought(player, first)
    triggers.fire_on_bought(player, second)
    assert (second.raw_attack, second.max_health) == (1, 1)
    triggers.fire_on_turn_start(player)
    third = Minion(card_id="b3", base_attack=1, base_health=1, tier=1)
    triggers.fire_on_bought(player, third)
    assert (third.raw_attack, third.max_health) > (1, 1)


def test_living_prison_takes_the_next_buys_stats(patch, triggers):
    from src.bg_recruitment.activate import activate_minion

    prison = _card(patch, "BG36_180")
    player = _player(patch, [prison], gold=5)
    activate_minion(player, 0, rng=np.random.default_rng(0), patch=patch)
    assert prison.wants_next_buy_stats

    bought = Minion(card_id="b", base_attack=4, base_health=6, tier=1)
    triggers.fire_on_bought(player, bought)
    assert (prison.raw_attack, prison.max_health) == (
        prison.base_attack + 4,
        prison.base_health + 6,
    )
    assert not prison.wants_next_buy_stats


def test_the_prison_waits_for_a_buy_and_takes_only_one(patch, triggers):
    prison = _card(patch, "BG36_180")
    player = _player(patch, [prison])
    prison.wants_next_buy_stats = 1
    triggers.fire_on_bought(player, Minion(card_id="b1", base_attack=2, base_health=2, tier=1))
    triggers.fire_on_bought(player, Minion(card_id="b2", base_attack=9, base_health=9, tier=1))
    assert (prison.raw_attack, prison.max_health) == (
        prison.base_attack + 2,
        prison.base_health + 2,
    )

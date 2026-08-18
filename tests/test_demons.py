"""The Demon family: hero damage, and eating out of the tavern.

Two seams carry most of them. Hero damage became an event — the listeners run
before the health is written, because a rewind has to be able to stop it — and
consuming a tavern minion grew the selectors the cards print: the biggest rather
than a random one, and every Demon rather than one the seat picks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState, apply_hero_damage
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


def _demon(card_id="d", atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.DEMON)


# --------------------------------------------------------------------------- #
# Hero damage
# --------------------------------------------------------------------------- #


def test_soul_rewinder_still_undoes_it(patch):
    rewinder = _card(patch, "BG26_174")
    player = _player(patch, [rewinder])
    apply_hero_damage(player, 7, patch=patch)
    assert player.health == 30
    assert rewinder.max_health == rewinder.base_health + 1


def test_tichondrius_watches_without_undoing(patch):
    """Not every answer is a rewind: this one lets the damage land."""
    tichondrius = _card(patch, "BG26_523")
    demon = _demon()
    other = Minion(card_id="o", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [tichondrius, demon, other])
    apply_hero_damage(player, 5, patch=patch)
    assert player.health == 25
    assert (demon.raw_attack, demon.max_health) == (4, 3)
    assert (other.raw_attack, other.max_health) == (1, 1)


def test_ashen_corruptor_undoes_it_and_feeds_the_tavern(patch):
    corruptor = _card(patch, "BG32_873")
    on_offer = Minion(card_id="s", base_attack=2, base_health=2, tier=1)
    player = _player(patch, [corruptor], shop=[on_offer])
    apply_hero_damage(player, 6, patch=patch)
    assert player.health == 30
    assert (on_offer.raw_attack, on_offer.max_health) == (3, 3)


def test_eredar_escapist_answers_every_four_points(patch):
    escapist = _card(patch, "BG36_733")
    player = _player(patch, [escapist])
    apply_hero_damage(player, 3, patch=patch)
    assert all(c is None for c in player.hand)  # 3 is not yet 4
    apply_hero_damage(player, 1, patch=patch)
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1 and got[0].is_tavern_spell


def test_the_countdown_refills(patch):
    escapist = _card(patch, "BG36_733")
    player = _player(patch, [escapist])
    apply_hero_damage(player, 8, patch=patch)
    assert sum(1 for c in player.hand if c is not None) == 2


def test_a_seat_with_no_watcher_just_takes_it(patch):
    player = _player(patch)
    apply_hero_damage(player, 7, patch=patch)
    assert player.health == 23


# --------------------------------------------------------------------------- #
# Eating out of the tavern
# --------------------------------------------------------------------------- #


def test_flaming_enforcer_eats_the_biggest(patch, triggers):
    enforcer = _card(patch, "BG34_500")
    small = Minion(card_id="small", base_attack=9, base_health=1, tier=1)
    big = Minion(card_id="big", base_attack=1, base_health=9, tier=1)
    player = _player(patch, [enforcer], shop=[small, big])
    triggers.fire_on_turn_end(player)
    assert (enforcer.raw_attack, enforcer.max_health) == (
        enforcer.base_attack + 1,
        enforcer.base_health + 9,
    )
    assert player.shop[0] is small and player.shop[1] is None


def test_soulkeeping_jailer_feeds_every_demon(patch):
    from src.bg_recruitment.activate import activate_minion

    jailer = _card(patch, "BG36_503")  # a Demon itself
    other_demon = _demon("d2")
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    meal = [Minion(card_id=f"m{i}", base_attack=2, base_health=2, tier=1) for i in range(3)]
    player = _player(patch, [jailer, other_demon, beast], shop=meal, gold=5)
    activate_minion(player, 0, rng=np.random.default_rng(0), patch=patch)
    assert (other_demon.raw_attack, other_demon.max_health) == (3, 3)
    assert (jailer.raw_attack, jailer.max_health) > (jailer.base_attack, jailer.base_health)
    assert (beast.raw_attack, beast.max_health) == (1, 1)
    assert sum(1 for m in player.shop if m is not None) == 1  # two eaten


def test_insatiable_urzul_eats_when_a_demon_arrives(patch, triggers):
    urzul = _card(patch, "BG21_004")
    meal = Minion(card_id="meal", base_attack=3, base_health=4, tier=1)
    player = _player(patch, [urzul], shop=[meal])
    triggers.fire_after_friendly_minion_placed(player, _demon())
    assert (urzul.raw_attack, urzul.max_health) == (
        urzul.base_attack + 3,
        urzul.base_health + 4,
    )
    assert player.shop[0] is None


def test_a_beast_arriving_leaves_the_urzul_hungry(patch, triggers):
    urzul = _card(patch, "BG21_004")
    meal = Minion(card_id="meal", base_attack=3, base_health=4, tier=1)
    player = _player(patch, [urzul], shop=[meal])
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    triggers.fire_after_friendly_minion_placed(player, beast)
    assert player.shop[0] is meal


def test_mind_muck_still_eats_exactly_once(patch):
    """The seat-picked shape reaches the dispatcher too and finds nobody to
    feed, which is what keeps it from eating twice."""
    from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries

    muck = _card(patch, "BG23_357")
    demon = _demon()
    meals = [Minion(card_id=f"m{i}", base_attack=1, base_health=1, tier=1) for i in range(2)]
    player = _player(patch, [muck, demon], shop=meals)
    apply_targeted_on_place_battlecries(
        ShopTriggers(np.random.default_rng(0), patch=patch),
        player,
        muck,
        rng=np.random.default_rng(0),
        forced_buff_target=demon,
    )
    assert sum(1 for m in player.shop if m is not None) == 1


# --------------------------------------------------------------------------- #
# The rest
# --------------------------------------------------------------------------- #


def test_void_pup_trainer_reaches_only_the_small_ones(patch, triggers):
    trainer = _card(patch, "BG35_152")
    low = Minion(card_id="low", base_attack=1, base_health=1, tier=3)
    high = Minion(card_id="high", base_attack=1, base_health=1, tier=4)
    player = _player(patch, [trainer], shop=[low, high])
    triggers.fire_on_place(trainer, player, None)
    assert (low.raw_attack, low.max_health) == (4, 4)
    assert (high.raw_attack, high.max_health) == (1, 1)


def test_champion_of_sargeras_pays_twice_over(patch, triggers):
    champion = _card(patch, "BG27_016")
    on_offer = Minion(card_id="s", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [champion], shop=[on_offer])
    triggers.fire_on_place(champion, player, None)
    assert (on_offer.raw_attack, on_offer.max_health) == (6, 6)
    triggers.apply_shop_effect(player, champion, champion.abilities[1].effect, None)
    assert (on_offer.raw_attack, on_offer.max_health) == (11, 11)


def test_twisted_wrathguard_answers_someone_elses_sale(patch, triggers):
    wrathguard = _card(patch, "BG35_155")
    sold = Minion(card_id="s", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [wrathguard, sold])
    triggers.fire_on_sell(sold, player)
    assert player.refresh_promises == {"BG35_150t": 1}


def test_imp_lusionist_leaves_two_copies(patch, triggers):
    imp = _card(patch, "BG36_731")
    player = _player(patch, [imp])
    triggers.apply_shop_effect(player, imp, imp.abilities[0].effect, None)
    got = [c for c in player.hand if c is not None]
    assert len(got) == 2 and all(c.card_id == "BG36_880" for c in got)

"""The modern package's heroes: passive powers only, and no new cards.

Patch 36.2.0 ships 121 heroes — 56 whose power is passive and 65 the seat
clicks. An active power needs somewhere in the frozen action space to be
pressed, so only the passive half is here, and only the part of it that the
descriptors and the cards this package already carries can express.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import src.envs.minibg  # noqa: F401  (breaks a circular import at collection)
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import hero_passives
from src.bg_recruitment.economy import (
    effective_buy_cost,
    effective_level_up_cost,
    effective_roll_cost,
)

PATCH_DIR = Path("data/bgcore/36_2_0_248348")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


@pytest.fixture(scope="module")
def catalog_heroes():
    with (PATCH_DIR / "catalog.json").open(encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)["heroes"]}


def _seat(patch, hero_id=None) -> PlayerState:
    player = PlayerState(
        health=patch.meta.ruleset.starting_health,
        gold=10,
        tavern_tier=1,
        board=[],
        shop=[None] * 7,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    if hero_id is not None:
        player.hero = patch.heroes[hero_id]
        hero_passives.apply_hero_on_game_start(
            player, round_number=1, patch=patch, rng=np.random.default_rng(0)
        )
    return player


# --------------------------------------------------------------------------- #
# The pool itself
# --------------------------------------------------------------------------- #


def test_the_package_ships_an_assignable_hero_pool(patch):
    assert patch.heroes
    assert patch.hero_pool_ids == frozenset(patch.heroes)


def test_every_hero_in_the_pool_has_a_passive_power(patch, catalog_heroes):
    """The active half waits on an action space that is frozen."""
    for hero_id in patch.hero_pool_ids:
        assert catalog_heroes[hero_id]["powerPassive"], hero_id


def test_no_hero_needs_a_card_the_package_does_not_carry(patch):
    """`HERO_TOKEN_IDS` is what a hero adds to the card index; the heroes that
    would need one are deliberately left out."""
    for hero_id, hero in patch.heroes.items():
        for passive in hero.passives:
            card_id = getattr(passive, "card_id", None)
            assert card_id is None or card_id in patch.templates, hero_id


def test_hero_health_and_armor_match_the_catalog(patch, catalog_heroes):
    for hero_id, hero in patch.heroes.items():
        row = catalog_heroes[hero_id]
        assert hero.start_armor == row["armor"], hero_id
        if hero.start_health is not None:
            assert hero.start_health == row["health"], hero_id


def test_absent_armor_reads_as_none_of_it(catalog_heroes):
    """Patchwerk is the only hero shipped without the field, and a null there
    would read as "unknown" where the truth is zero."""
    assert catalog_heroes["TB_BaconShop_HERO_34"]["armor"] == 0
    assert all(r["armor"] is not None for r in catalog_heroes.values())


# --------------------------------------------------------------------------- #
# Each of the eight, doing what its power prints
# --------------------------------------------------------------------------- #


def test_patchwerk_starts_with_thirty_extra_health(patch):
    """The catalog has already applied it: the row reads 60, not 30 plus."""
    assert _seat(patch, "TB_BaconShop_HERO_34").health == 60
    assert _seat(patch).health == 30


def test_armor_reaches_the_seat(patch):
    for hero_id, wanted in (
        ("TB_BaconShop_HERO_57", 13),
        ("TB_BaconShop_HERO_74", 6),
        ("TB_BaconShop_HERO_52", 18),
        ("TB_BaconShop_HERO_34", 0),
    ):
        assert _seat(patch, hero_id).armor == wanted, hero_id


def test_millhouse_moves_all_three_prices(patch):
    plain, millhouse = _seat(patch), _seat(patch, "TB_BaconShop_HERO_49")
    assert (effective_buy_cost(plain), effective_buy_cost(millhouse)) == (3, 2)
    assert (effective_roll_cost(plain), effective_roll_cost(millhouse)) == (1, 2)
    assert effective_level_up_cost(millhouse) == effective_level_up_cost(plain) + 1


def test_nozdormus_first_refresh_each_turn_is_free(patch):
    player = _seat(patch, "TB_BaconShop_HERO_57")
    hero_passives.apply_hero_on_turn_start(
        player, round_number=2, patch=patch, rng=np.random.default_rng(0)
    )
    assert effective_roll_cost(player) == 0


def test_omu_pays_two_gold_for_an_upgrade(patch):
    player = _seat(patch, "TB_BaconShop_HERO_74")
    gold = player.gold
    hero_passives.apply_hero_on_level_up(player)
    assert player.gold == gold + 2


def test_yseras_counter_always_shows_a_dragon(patch):
    from src.bg_recruitment.shop import refresh_shop

    for seed in range(6):
        player = _seat(patch, "TB_BaconShop_HERO_53")
        player.tavern_tier = 3
        refresh_shop(player, None, rng=np.random.default_rng(seed), patch=patch)
        offers = [m for m in player.shop if m is not None]
        assert any(m.race is Race.DRAGON for m in offers), seed


def test_chenvaala_discounts_the_upgrade_after_three_elementals(patch):
    player = _seat(patch, "TB_BaconShop_HERO_78")
    before = effective_level_up_cost(player)
    for _ in range(3):
        hero_passives.apply_hero_on_elemental_played(player)
    assert effective_level_up_cost(player) == max(0, before - 3)


def test_deathwing_and_alakir_reach_the_fight(patch):
    assert hero_passives.hero_combat_attack_aura(_seat(patch, "TB_BaconShop_HERO_52")) == 2
    granted = hero_passives.hero_start_combat_keywords(_seat(patch, "TB_BaconShop_HERO_76"))
    assert granted == frozenset({Keyword.WINDFURY, Keyword.SHIELD, Keyword.TAUNT})


def test_a_seat_with_no_hero_is_untouched(patch):
    """Every hero entry point is a no-op without one, which is what keeps the
    heroless path identical."""
    plain = _seat(patch)
    assert plain.hero is None
    assert plain.armor == 0
    assert hero_passives.hero_combat_attack_aura(plain) == 0
    assert hero_passives.hero_start_combat_keywords(plain) == frozenset()


# --------------------------------------------------------------------------- #
# The passives that brought a descriptor of their own
# --------------------------------------------------------------------------- #


def _minion(card_id="m", attack=1, health=1, race=None):
    from src.bg_core.minion import Minion

    return Minion(
        card_id=card_id, base_attack=attack, base_health=health, tier=1, race=race
    )


def test_hoggarr_pays_for_a_pirate_and_nothing_else(patch):
    player = _seat(patch, "BG26_HERO_101")
    gold = player.gold
    hero_passives.apply_hero_on_bought(_minion(race=Race.PIRATE), player)
    assert player.gold == gold + 1
    hero_passives.apply_hero_on_bought(_minion(race=Race.BEAST), player)
    assert player.gold == gold + 1


def test_gallywix_banks_the_gold_for_next_turn(patch):
    """Next turn, not this one, which is the whole shape of the card."""
    player = _seat(patch, "TB_BaconShop_HERO_10")
    gold = player.gold
    hero_passives.apply_hero_on_sell(
        _minion(), player, rng=np.random.default_rng(0), patch=patch
    )
    assert player.gold == gold
    assert player.gold_next_turn == 1


def test_flurgl_hands_over_a_murloc_every_fifth_sale(patch):
    player = _seat(patch, "TB_BaconShop_HERO_55")
    for i in range(1, 5):
        hero_passives.apply_hero_on_sell(
            _minion(), player, rng=np.random.default_rng(i), patch=patch
        )
    assert all(c is None for c in player.hand)

    hero_passives.apply_hero_on_sell(
        _minion(), player, rng=np.random.default_rng(5), patch=patch
    )
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1
    assert got[0].race in (Race.MURLOC, Race.ALL)  # an Amalgam is every type


def test_ini_pays_once_per_nine_deaths_and_does_not_double_pay(patch):
    """Read at the seat's own turn start: the deaths happen inside a fight, and
    a fight hands what it owes to the seat rather than into its hand."""
    from src.bg_recruitment.game_counts import DEATHS

    player = _seat(patch, "BG22_HERO_200")
    player.game_counts[DEATHS] = 8
    hero_passives.apply_hero_on_turn_start(
        player, round_number=3, patch=patch, rng=np.random.default_rng(0)
    )
    assert all(c is None for c in player.hand)

    player.game_counts[DEATHS] = 9
    hero_passives.apply_hero_on_turn_start(
        player, round_number=4, patch=patch, rng=np.random.default_rng(0)
    )
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1
    assert got[0].race in (Race.MECHANICAL, Race.ALL)

    # the same nine deaths are not paid twice
    hero_passives.apply_hero_on_turn_start(
        player, round_number=5, patch=patch, rng=np.random.default_rng(0)
    )
    assert len([c for c in player.hand if c is not None]) == 1


def test_taethelan_makes_every_third_tavern_spell_free(patch):
    from src.bg_recruitment.tavern_spells import buy_tavern_spell, offer_tavern_spells

    player = _seat(patch, "BG28_HERO_800")
    player.gold = 30
    paid = []
    for _ in range(6):
        offer_tavern_spells(
            player, rng=np.random.default_rng(0), patch=patch, card_ids=["BG28_897"]
        )
        before = player.gold
        buy_tavern_spell(player, 0, patch=patch)
        paid.append(before - player.gold)
        player.hand = [None] * len(player.hand)
    assert [p == 0 for p in paid] == [False, False, True, False, False, True]


def _fight(patch, player, mine, theirs, *, round_number=1, seed=0, initiative=True):
    from src.bg_combat.battle.seat import RecordingSeat
    from src.bg_combat.battle.simulate import simulate_battle
    from src.bg_recruitment.combat_seat import PlayerCombatSeat

    out = []
    simulate_battle(
        mine,
        theirs,
        p0_has_initiative=initiative,
        rng=np.random.default_rng(seed),
        combat_board_max=7,
        damage_cap=15,
        max_board_slots=7,
        p0_board_out=out,
        patch=patch,
        seats=(
            PlayerCombatSeat(player, patch=patch, round_number=round_number),
            RecordingSeat(),
        ),
    )
    return out


def test_greybough_grows_what_the_fight_summons(patch):
    """The Eternal Knight a Summoner's deathrattle puts down is 4/2 printed."""
    for hero_id, wanted in ((None, (4, 2, False)), ("TB_BaconShop_HERO_95", (5, 4, True))):
        player = _seat(patch, hero_id)
        out = _fight(
            patch,
            player,
            [patch.make_minion("BG25_009")],
            [_minion("foe", 40, 1)],
            initiative=False,
        )
        knight = next(m for m in out if m.card_id == "BG25_008")
        assert (
            knight.raw_attack,
            knight.max_health,
            Keyword.TAUNT in knight.all_keywords,
        ) == wanted, hero_id


def test_rokara_keeps_the_attack_it_earned(patch):
    """Permanently, so it goes back to the seat rather than dying with the copy
    that earned it."""
    body = patch.make_minion("BGS_119")
    body.base_attack, body.base_health = 5, 20
    player = _seat(patch, "BG20_HERO_100")
    player.board.append(body)
    _fight(patch, player, [body], [_minion("foe", 1, 1)])
    assert body.raw_attack == 6


def test_drekthar_copies_the_biggest_once_and_not_before_turn_seven(patch):
    for round_number, wanted in ((6, ["small", "big"]), (7, ["small", "big", "big"])):
        player = _seat(patch, "BG22_HERO_002")
        out = _fight(
            patch,
            player,
            [_minion("small", 1, 30), _minion("big", 9, 30)],
            [_minion("wall", 0, 300)],
            round_number=round_number,
        )
        assert [m.card_id for m in out] == wanted, round_number


def test_vanndar_reads_health_where_drekthar_reads_attack(patch):
    player = _seat(patch, "BG22_HERO_003")
    out = _fight(
        patch,
        player,
        [_minion("tough", 1, 40), _minion("sharp", 9, 10)],
        [_minion("wall", 0, 300)],
        round_number=7,
    )
    assert [m.card_id for m in out].count("tough") == 2

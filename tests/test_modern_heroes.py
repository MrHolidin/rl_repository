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
from src.bg_core.effects import Keyword, Trigger
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


def test_a_hero_is_either_passive_or_has_something_to_press(patch, catalog_heroes):
    """The catalog's own split, and the engine agreeing with it: a passive
    power has nothing to press, an active one does."""
    for hero_id, hero in patch.heroes.items():
        passive = catalog_heroes[hero_id]["powerPassive"]
        assert hero.has_power() != bool(passive), hero_id
        if not passive:
            assert hero.power_cost == catalog_heroes[hero_id]["powerCost"], hero_id


def test_no_hero_needs_a_card_the_package_does_not_carry(patch):
    """`HERO_TOKEN_IDS` is what a hero adds to the card index; the heroes that
    would need one are deliberately left out. A hero pays in minions *or*
    spells — a Brann, a Tavern Coin, a Triple Reward — so both catalogs count.
    """
    for hero_id, hero in patch.heroes.items():
        for passive in hero.passives:
            card_id = getattr(passive, "card_id", None)
            if card_id is None:
                continue
            assert (
                card_id in patch.templates or card_id in patch.tavern_spells
            ), f"{hero_id} wants {card_id}"


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


# --------------------------------------------------------------------------- #
# The countdown, refresh and start-of-combat passives
# --------------------------------------------------------------------------- #


def _turn(patch, player, round_number):
    """The seat's turn start, in the order the lobby runs it: the hero's
    per-turn levers, then the tavern is built, then the hero acts on it."""
    hero_passives.apply_hero_on_turn_start(
        player,
        round_number=round_number,
        patch=patch,
        rng=np.random.default_rng(round_number),
    )
    hero_passives.apply_hero_after_shop_refresh(
        player,
        round_number,
        patch=patch,
        rng=np.random.default_rng(round_number),
    )


def _hand(player):
    return [c.card_id for c in player.hand if c is not None]


def _buy(patch, player, minion):
    player.hero_buy_count += 1
    hero_passives.apply_hero_on_bought(
        minion, player, rng=np.random.default_rng(0), patch=patch
    )


def test_kaelthas_hands_over_a_tavern_coin_every_third_buy(patch):
    player = _seat(patch, "TB_BaconShop_HERO_60")
    for _ in range(6):
        _buy(patch, player, _minion())
    assert _hand(player) == ["BG28_810", "BG28_810"]


def test_dinotamer_brann_pays_once_and_only_for_battlecries(patch):
    player = _seat(patch, "TB_BaconShop_HERO_43")
    for _ in range(8):
        _buy(patch, player, _minion())  # no Battlecry
    assert _hand(player) == []

    player = _seat(patch, "TB_BaconShop_HERO_43")
    battlecry = patch.make_minion("BGS_020")
    for _ in range(8):
        _buy(patch, player, battlecry)
    assert _hand(player) == ["BG_LOE_077"]  # once per game


def test_guff_counts_tiers_rather_than_cards(patch):
    player = _seat(patch, "BG20_HERO_242")
    for _ in range(7):
        _buy(patch, player, _minion(attack=1, health=1))  # 7 Tiers' worth
    assert player.hero_tiers_bought == 7
    assert _hand(player) == []

    for _ in range(7):
        _buy(patch, player, patch.make_minion("BG25_354"))  # Tier 5 each
    assert player.hero_tiers_bought == 42
    assert _hand(player) == ["triple_reward_discover"] * (42 // 20)


def test_saurfangs_counter_grows_with_your_buys(patch):
    from src.bg_recruitment.shop import refresh_shop

    def _first_offer(buys):
        player = _seat(patch, "BG20_HERO_102")
        player.tavern_tier = 3
        player.hero_buy_count = buys
        refresh_shop(player, None, rng=np.random.default_rng(1), patch=patch)
        offer = next(m for m in player.shop if m is not None)
        printed = patch.templates[offer.card_id]
        return offer.raw_attack - printed.base_attack

    assert _first_offer(0) == 1
    assert _first_offer(6) == 3


def test_aranna_makes_the_first_buy_free_once_unlocked(patch):
    from src.bg_recruitment.economy import effective_buy_cost

    player = _seat(patch, "TB_BaconShop_HERO_59")
    _turn(patch, player, 5)
    assert effective_buy_cost(player) == 3  # 14 attacks have not happened

    player.hero_attacks = 14
    _turn(patch, player, 6)
    assert effective_buy_cost(player) == 0
    player.hero_free_buys -= 1  # spent by the purchase
    assert effective_buy_cost(player) == 3


def test_sindragosa_shows_one_fewer_and_freezes(patch):
    from src.bg_recruitment.economy import effective_buy_cost
    from src.bg_recruitment.shop import effective_shop_offers_count

    plain, sindragosa = _seat(patch), _seat(patch, "TB_BaconShop_HERO_27")
    plain.tavern_tier = sindragosa.tavern_tier = 4
    assert effective_shop_offers_count(sindragosa) == (
        effective_shop_offers_count(plain) - 1
    )
    assert effective_buy_cost(sindragosa) == 2
    hero_passives.apply_hero_on_turn_end(sindragosa)
    assert sindragosa.shop_freeze_next_round


def test_rakanishu_improves_every_third_turn(patch):
    """Game start *is* turn 1, so the base grant happens there."""
    player = _seat(patch, "TB_BaconShop_HERO_75")
    assert player.tavern_spell_bonus_attack == 1
    for round_number in (2, 3):
        _turn(patch, player, round_number)
    assert player.tavern_spell_bonus_attack == 1
    _turn(patch, player, 4)
    assert player.tavern_spell_bonus_attack == 2


def test_yogg_casts_from_turn_three(patch):
    player = _seat(patch, "TB_BaconShop_HERO_35")
    player.tavern_tier = 3
    _turn(patch, player, 2)
    assert player.last_tavern_spell_cast is None
    _turn(patch, player, 3)
    assert player.last_tavern_spell_cast is not None


def test_varden_copies_the_best_offer_and_freezes_both(patch):
    from src.bg_recruitment.shop import refresh_shop

    player = _seat(patch, "BG22_HERO_004")
    player.tavern_tier = 4
    refresh_shop(player, None, rng=np.random.default_rng(2), patch=patch)
    offers = [(i, m) for i, m in enumerate(player.shop) if m is not None]
    best_tier = max(m.tier for _, m in offers)
    doubled = [m.card_id for _, m in offers if m.tier == best_tier]
    assert len(doubled) == 2 and doubled[0] == doubled[1]
    frozen = [i for i, f in enumerate(player.shop_frozen) if f]
    assert len(frozen) == 2


def test_enhance_o_grants_two_bonus_keywords(patch):
    from src.bg_core.minion import BONUS_KEYWORDS
    from src.bg_recruitment.shop import refresh_shop

    player = _seat(patch, "BG24_HERO_204")
    player.tavern_tier = 4
    refresh_shop(player, None, rng=np.random.default_rng(3), patch=patch)
    granted = [k for m in player.shop if m is not None for k in m.granted_keywords]
    assert len(granted) == 2
    assert set(granted) <= BONUS_KEYWORDS


def test_afkay_and_faelin_discover_at_each_tier_they_name(patch):
    """A list, not a count: the modal chain re-rolls one kind, so these are
    opened one at a time."""
    from src.bg_recruitment.discover import resolve_discover_pick

    for hero_id, wanted in (
        ("TB_BaconShop_HERO_16", [3, 4]),
        ("BG22_HERO_201", [6, 4, 2]),
    ):
        player = _seat(patch, hero_id)
        for round_number in (2, 3):
            _turn(patch, player, round_number)
        seen = []
        while player.pending_choice is not None:
            tiers = {patch.templates[o].tier for o in player.pending_choice.options}
            seen.append(sorted(tiers))
            resolve_discover_pick(
                player,
                0,
                None,
                rng=np.random.default_rng(0),
                on_after_placed=lambda *_: None,
                patch=patch,
            )
            hero_passives.flush_hero_tier_discovers(
                player, rng=np.random.default_rng(1), patch=patch
            )
        assert seen == [[t] for t in wanted], hero_id


def test_thorims_pick_waits_on_sixty_gold(patch):
    player = _seat(patch, "BG27_HERO_801")
    promised = player.hero_promised_card
    assert patch.templates[promised].tier == 7

    player.hero_gold_spent_total = 59
    _turn(patch, player, 3)
    assert _hand(player) == []

    player.hero_gold_spent_total = 60
    _turn(patch, player, 4)
    assert _hand(player) == [promised]


def test_genn_swaps_its_power_on_turn_four(patch):
    from src.bg_lobby.player import PendingChoiceKind

    player = _seat(patch, "BG35_HERO_001")
    _turn(patch, player, 3)
    assert player.pending_choice is None
    _turn(patch, player, 4)
    assert player.pending_choice.kind is PendingChoiceKind.HERO_POWER_DISCOVER
    assert len(player.pending_choice.options) == 2
    assert "BG35_HERO_001" not in player.pending_choice.options


def test_illidan_buffs_both_ends_and_swings_at_once(patch):
    ends = [_minion("left", 3, 40), _minion("mid", 1, 40), _minion("right", 3, 40)]
    player = _seat(patch, "TB_BaconShop_HERO_08")
    out = _fight(patch, player, ends, [_minion("wall", 0, 300)])
    by_id = {m.card_id: m for m in out}
    assert (by_id["left"].raw_attack, by_id["right"].raw_attack) == (5, 5)
    assert by_id["mid"].raw_attack == 1


def test_wagtoggle_pays_one_of_each_type(patch):
    beast, murloc, other = (
        _minion("beast", 1, 40, race=Race.BEAST),
        _minion("murloc", 1, 40, race=Race.MURLOC),
        _minion("beast2", 1, 40, race=Race.BEAST),
    )
    player = _seat(patch, "TB_BaconShop_HERO_14")
    out = _fight(patch, player, [beast, murloc, other], [_minion("wall", 0, 300)])
    gained = sum(m.raw_attack - 1 for m in out)
    assert gained == 2  # one Beast and one Murloc, not both Beasts


# --------------------------------------------------------------------------- #
# Powers the seat presses
# --------------------------------------------------------------------------- #


def _armed(patch, hero_id, *, tier=6, gold=10, board=(), round_number=5):
    player = _seat(patch)
    player.hero = patch.heroes[hero_id]
    player.tavern_tier = tier
    player.gold = gold
    player.board.extend(board)
    hero_passives.apply_hero_on_game_start(
        player, round_number=1, patch=patch, rng=np.random.default_rng(0)
    )
    hero_passives.apply_hero_on_turn_start(
        player, round_number=round_number, patch=patch, rng=np.random.default_rng(0)
    )
    player.gold = gold
    return player


def _press(patch, player, seed=0):
    hero_passives.use_hero_power(
        player,
        rng=np.random.default_rng(seed),
        patch=patch,
        round_number=player.round_number,
    )


def test_the_action_is_offered_only_when_the_power_can_be_pressed(patch):
    from src.envs.bglike import actions as A

    player = _armed(patch, "BG32_HERO_001", gold=3)  # Cenarius, 3 Gold
    assert hero_passives.can_use_hero_power(player, 5)
    player.gold = 2
    assert not hero_passives.can_use_hero_power(player, 5)
    assert int(A.Action.HERO_POWER) == A.NUM_ACTIONS - 1


def test_a_seat_with_no_hero_never_offers_it(patch):
    assert not hero_passives.can_use_hero_power(_seat(patch), 5)


def test_pressing_pays_and_spends_the_use(patch):
    player = _armed(patch, "BG28_HERO_801", gold=10)  # Holli'dae, 1 Gold
    _press(patch, player)
    assert player.gold == 9
    assert len(_hand(player)) == 1
    assert not hero_passives.can_use_hero_power(player, 5)  # one use a turn


def test_blackthorn_may_be_pressed_twice_a_turn(patch):
    player = _armed(patch, "BG20_HERO_103", gold=10)
    _press(patch, player)
    assert hero_passives.can_use_hero_power(player, 5)
    _press(patch, player)
    assert not hero_passives.can_use_hero_power(player, 5)
    assert len(_hand(player)) == 4  # two Gems a press


def test_reno_is_once_per_game_not_once_per_turn(patch):
    body = patch.make_minion("BGS_119")
    player = _armed(patch, "TB_BaconShop_HERO_41", board=[body])
    _press(patch, player)
    assert body.is_golden
    hero_passives.apply_hero_on_turn_start(
        player, round_number=6, patch=patch, rng=np.random.default_rng(0)
    )
    assert not hero_passives.can_use_hero_power(player, 6)


def test_a_power_that_has_not_woken_up_is_not_offered(patch):
    assert not hero_passives.can_use_hero_power(
        _armed(patch, "TB_BaconShop_HERO_56", tier=3), 5
    )  # Alexstrasza unlocks at Tier 4
    assert hero_passives.can_use_hero_power(
        _armed(patch, "TB_BaconShop_HERO_56", tier=4), 5
    )
    # Shudderwock unlocks on Turn 3 — with a Battlecry on the board either
    # way, so the turn is the only thing being read.
    battlecry = next(
        c
        for c in sorted(patch.pool_ids)
        if any(ab.trigger is Trigger.ON_PLACE for ab in patch.templates[c].abilities)
    )
    assert not hero_passives.can_use_hero_power(
        _armed(
            patch,
            "TB_BaconShop_HERO_23",
            round_number=2,
            board=[patch.make_minion(battlecry)],
        ),
        2,
    )
    assert hero_passives.can_use_hero_power(
        _armed(
            patch,
            "TB_BaconShop_HERO_23",
            round_number=3,
            board=[patch.make_minion(battlecry)],
        ),
        3,
    )


def test_elises_price_climbs_with_each_use(patch):
    player = _armed(patch, "TB_BaconShop_HERO_42", tier=4, gold=10)
    assert hero_passives.hero_power_cost(player) == 1
    _press(patch, player)
    assert hero_passives.hero_power_cost(player) == 2
    assert {patch.templates[o].tier for o in player.pending_choice.options} == {4}


def test_the_rat_kings_discover_follows_the_turns_type(patch):
    player = _armed(patch, "TB_BaconShop_HERO_12")
    tribe = player.hero_rotating_tribe
    assert tribe is not None
    _press(patch, player)
    assert all(
        patch.templates[o].race is tribe for o in player.pending_choice.options
    )


def test_chromie_turns_the_counter_into_spells(patch):
    from src.bg_recruitment.shop import refresh_shop

    player = _armed(patch, "BG34_HERO_001")
    refresh_shop(player, None, rng=np.random.default_rng(1), patch=patch)
    _press(patch, player)
    assert all(m is None for m in player.shop)
    assert len(player.tavern_spell_offers) == 6


def test_mutanus_sells_and_moves_the_stats(patch):
    big, small = _minion("big", 5, 5), _minion("small", 1, 1)
    player = _armed(patch, "BG20_HERO_301", board=[big, small])
    gold = player.gold
    _press(patch, player)
    assert len(player.board) == 1
    assert player.gold == gold + 1  # sold, not destroyed
    assert player.board[0].raw_attack == 6


def test_the_jailer_trades_an_undead_for_an_undead(patch):
    undead = next(
        c for c in sorted(patch.pool_ids) if patch.templates[c].race is Race.UNDEAD
    )
    body = patch.make_minion(undead)
    player = _armed(patch, "TB_BaconShop_HERO_702", tier=2, board=[body])
    _press(patch, player)
    assert player.board == []
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1 and got[0].race in (Race.UNDEAD, Race.ALL)


def test_the_lich_king_and_george_hand_out_their_keyword(patch):
    body = _minion("mine")
    _press(patch, _armed(patch, "TB_BaconShop_HERO_22", board=[body]))
    assert Keyword.REBORN in body.all_keywords

    body = _minion("mine")
    player = _armed(patch, "TB_BaconShop_HERO_15", board=[body])
    _press(patch, player)
    assert Keyword.SHIELD in body.all_keywords and body.has_shield


def test_bazhial_steals_and_bleeds(patch):
    from src.bg_recruitment.shop import refresh_shop

    player = _armed(patch, "TB_BaconShop_HERO_25")
    refresh_shop(player, None, rng=np.random.default_rng(1), patch=patch)
    health, armor = player.health, player.armor
    assert armor == player.hero.start_armor > 0
    _press(patch, player)
    assert len(_hand(player)) == 1
    # "Take 2 damage" is damage like any other: armor absorbs it first, which
    # is the whole reason this hero starts with fifteen of it.
    assert (player.health, player.armor) == (health, armor - 2)
    player.armor = 1
    player.hero_power_uses_this_turn = 0
    _press(patch, player)
    assert (player.health, player.armor) == (health - 1, 0)


def test_cenarius_raises_the_gold_cap(patch):
    player = _armed(patch, "BG32_HERO_001")
    before = player.ruleset.gold_cap
    _press(patch, player)
    assert player.ruleset.gold_cap == before + 1


def test_shudderwock_fires_a_battlecry_and_only_on_a_body_that_has_one(patch):
    """A minion with no Battlecry is not a target this can take, so the
    fallback only looks at the ones carrying the trigger it means to fire."""
    from src.bg_core.minion import Race

    dragon = next(
        c
        for c in sorted(patch.pool_ids)
        if patch.templates[c].race is Race.DRAGON and c != "BG26_963"
    )
    for seed in range(5):
        synth = patch.make_minion("BG26_963")  # Battlecry: other Dragons +1/+1
        mate = patch.make_minion(dragon)
        printed = patch.templates[dragon].base_attack
        player = _armed(patch, "TB_BaconShop_HERO_23", board=[synth, mate], round_number=4)
        _press(patch, player, seed=seed)
        assert mate.raw_attack == printed + 1, seed


def test_pressing_never_reaches_the_dispatcher_with_nothing_to_do(patch):
    """Every bound power resolves through the Tavern-spell path, which raises
    on an effect nobody handles — so pressing each one is the check."""
    for hero_id, hero in patch.heroes.items():
        if not hero.has_power():
            continue
        board = [patch.make_minion("BG26_963"), patch.make_minion("BGS_119")]
        undead = next(
            c for c in sorted(patch.pool_ids) if patch.templates[c].race is Race.UNDEAD
        )
        board.append(patch.make_minion(undead))
        player = _armed(patch, hero_id, tier=6, gold=20, board=board, round_number=8)
        from src.bg_recruitment.shop import refresh_shop

        refresh_shop(player, None, rng=np.random.default_rng(0), patch=patch)
        # Tess refreshes with the last warband she saw; without one there is
        # nothing for the power to copy and the mask would rightly hide it.
        player.last_opponent_board = [patch.make_minion("BGS_119")]
        assert hero_passives.can_use_hero_power(player, 8), hero_id
        _press(patch, player)  # raises loudly if any effect is unhandled


# --------------------------------------------------------------------------- #
# How often a power may be pressed — four limits, and they compose
# --------------------------------------------------------------------------- #


def test_the_default_is_once_a_turn_and_it_comes_back(patch):
    player = _armed(patch, "BG28_HERO_801", gold=20)  # Holli'dae
    assert hero_passives.can_use_hero_power(player, 5)
    _press(patch, player)
    assert not hero_passives.can_use_hero_power(player, 5)

    hero_passives.apply_hero_on_turn_start(
        player, round_number=6, patch=patch, rng=np.random.default_rng(0)
    )
    player.gold = 20
    assert hero_passives.can_use_hero_power(player, 6)


def test_twice_per_turn_is_two_and_not_three(patch):
    player = _armed(patch, "BG20_HERO_103", gold=20)  # Blackthorn
    for _ in range(2):
        assert hero_passives.can_use_hero_power(player, 5)
        _press(patch, player)
    assert not hero_passives.can_use_hero_power(player, 5)


def test_charges_are_spent_for_the_whole_game_not_the_turn(patch):
    """"Once per game" is one charge; the pool also prints four, three and
    three. A new turn gives back the per-turn use and not the charge."""
    body = patch.make_minion("BGS_119")
    player = _armed(patch, "TB_BaconShop_HERO_41", board=[body])  # Reno, 1 charge
    assert patch.heroes["TB_BaconShop_HERO_41"].power_charges == 1
    _press(patch, player)
    for round_number in (6, 7, 8):
        hero_passives.apply_hero_on_turn_start(
            player, round_number=round_number, patch=patch, rng=np.random.default_rng(0)
        )
        player.gold = 20
        assert not hero_passives.can_use_hero_power(player, round_number)


def test_a_multi_charge_power_counts_down_across_turns(patch):
    """The shape Captain Eudora, Putricide and Zephrys print. Nothing binds it
    yet, so it is checked on the model rather than on a card."""
    from dataclasses import replace

    player = _armed(patch, "BG28_HERO_801", gold=40)
    player.hero = replace(player.hero, power_charges=3)
    for round_number in (5, 6, 7):
        assert hero_passives.can_use_hero_power(player, round_number), round_number
        _press(patch, player)
        hero_passives.apply_hero_on_turn_start(
            player, round_number=round_number + 1, patch=patch, rng=np.random.default_rng(0)
        )
        player.gold = 40
    assert player.hero_power_uses_game == 3
    assert not hero_passives.can_use_hero_power(player, 8)


def test_a_cooldown_keeps_it_asleep_for_the_turns_it_names(patch):
    """Snake Eyes' shape: pressed, then unusable for a number of turns."""
    from dataclasses import replace

    player = _armed(patch, "BG28_HERO_801", gold=40)
    player.hero = replace(player.hero, power_cooldown_turns=3)
    _press(patch, player)
    assert player.hero_power_ready_on_round == 8

    for round_number in (6, 7):
        hero_passives.apply_hero_on_turn_start(
            player, round_number=round_number, patch=patch, rng=np.random.default_rng(0)
        )
        player.gold = 40
        assert not hero_passives.can_use_hero_power(player, round_number), round_number

    hero_passives.apply_hero_on_turn_start(
        player, round_number=8, patch=patch, rng=np.random.default_rng(0)
    )
    player.gold = 40
    assert hero_passives.can_use_hero_power(player, 8)


def test_every_bound_powers_limit_matches_what_the_card_prints(patch, catalog_heroes):
    """The card says it or the default holds. Two say "Twice per turn", three
    of the 65 say "once per game"; the seven that print "(N left!)" as a
    countdown to a payout are not use limits and must not be read as charges.
    """
    import re

    for hero_id, hero in patch.heroes.items():
        if not hero.has_power():
            continue
        text = re.sub("<[^>]+>", "", catalog_heroes[hero_id]["powerText"] or "")
        wanted_uses = 2 if re.search(r"[Tt]wice per turn", text) else 1
        assert hero.power_uses == wanted_uses, hero_id
        if re.search(r"[Oo]nce per game", text):
            assert hero.power_charges == 1, hero_id
        else:
            assert hero.power_charges == 0, hero_id


# --------------------------------------------------------------------------- #
# The powers that needed an effect of their own
# --------------------------------------------------------------------------- #


def _stock(patch, player, seed=1):
    from src.bg_recruitment.shop import refresh_shop

    refresh_shop(player, None, rng=np.random.default_rng(seed), patch=patch)
    return player


def test_kraggs_gold_grows_with_the_round_and_is_spent_for_good(patch):
    for round_number, wanted in ((1, 2), (5, 6)):
        player = _armed(patch, "TB_BaconShop_HERO_68", gold=0, round_number=round_number)
        _press(patch, player)
        assert player.gold == wanted, round_number
        assert not hero_passives.can_use_hero_power(player, round_number)


def test_edwins_buff_improves_every_four_cards_bought(patch):
    for buys, wanted in ((0, 3), (4, 5), (8, 7)):
        body = _minion("mine")
        player = _armed(patch, "TB_BaconShop_HERO_01", board=[body])
        player.hero_buy_count = buys
        _press(patch, player)
        assert body.raw_attack == wanted, buys


def test_xyrella_takes_the_card_she_names_and_sets_it_to_two(patch):
    from src.bg_recruitment.tavern_spells import apply_tavern_spell_effect

    player = _stock(patch, _armed(patch, "BG20_HERO_101"))
    named = player.shop[2]
    apply_tavern_spell_effect(
        player,
        player.hero.power[0].effect,
        rng=np.random.default_rng(0),
        patch=patch,
        source=named,
    )
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1
    assert got[0].card_id == named.card_id
    assert (got[0].raw_attack, got[0].max_health) == (2, 2)


def test_pyramad_doubles_the_health_of_what_it_steals(patch):
    player = _stock(patch, _armed(patch, "TB_BaconShop_HERO_39"))
    _press(patch, player)
    got = next(c for c in player.hand if c is not None)
    printed = patch.templates[got.card_id]
    assert got.max_health == printed.base_health * 2
    assert got.raw_attack == printed.base_attack


def test_tess_fills_the_counter_with_the_last_warband(patch):
    player = _stock(patch, _armed(patch, "TB_BaconShop_HERO_50"))
    warband = [patch.make_minion(c) for c in sorted(patch.pool_ids)[:4]]
    for body in warband:
        body.bonus_attack += 10  # plain copies: the gains do not come along
    player.last_opponent_board = tuple(warband)
    _press(patch, player)
    offers = [m for m in player.shop if m is not None]
    assert [m.card_id for m in offers] == [m.card_id for m in warband]
    for offer in offers:
        assert offer.raw_attack == patch.templates[offer.card_id].base_attack


def test_toki_mixes_in_two_offers_from_a_tier_higher(patch):
    player = _stock(patch, _armed(patch, "TB_BaconShop_HERO_28", tier=3))
    _press(patch, player)
    tiers = [m.tier for m in player.shop if m is not None]
    assert sum(1 for t in tiers if t == 4) == 2
    assert all(t <= 4 for t in tiers)


def test_malygos_swaps_one_card_for_its_own_tier_twice_a_turn(patch):
    player = _stock(patch, _armed(patch, "TB_BaconShop_HERO_58"))
    before = [(m.card_id, m.tier) for m in player.shop if m is not None]
    _press(patch, player)
    after = [(m.card_id, m.tier) for m in player.shop if m is not None]
    changed = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(changed) == 1
    assert changed[0][0][1] == changed[0][1][1]  # same Tier
    assert hero_passives.can_use_hero_power(player, 5)  # twice a turn


def test_hooktusk_discovers_one_tier_below_the_body_it_removed(patch):
    tier_four = sorted(c for c in patch.pool_ids if patch.templates[c].tier == 4)[0]
    body = patch.make_minion(tier_four)
    player = _armed(patch, "TB_BaconShop_HERO_67", board=[body])
    _press(patch, player)
    assert player.board == []
    assert {patch.templates[o].tier for o in player.pending_choice.options} == {3}


def test_jandice_trades_rather_than_taking(patch):
    mine = patch.make_minion("BGS_119")
    player = _stock(patch, _armed(patch, "TB_BaconShop_HERO_71", board=[mine]))
    _press(patch, player)
    assert len(player.board) == 1 and player.board[0] is not mine
    assert any(m is mine for m in player.shop if m is not None)


def test_inge_pays_the_seats_tier(patch):
    for tier in (2, 5):
        body = _minion("mine")
        player = _armed(patch, "BG26_HERO_102", tier=tier, board=[body])
        _press(patch, player)
        assert body.raw_attack == 1 + tier, tier


def test_voljin_gives_each_the_others_attack_until_next_turn(patch):
    from src.bg_recruitment.spellcraft import expire_temporary_buffs

    a, b = _minion("a", 3, 9), _minion("b", 7, 9)
    player = _armed(patch, "BG20_HERO_201", board=[a, b])
    _press(patch, player)
    assert (a.raw_attack, b.raw_attack) == (10, 10)
    expire_temporary_buffs(player)
    assert (a.raw_attack, b.raw_attack) == (3, 7)


# --------------------------------------------------------------------------- #
# What the audit found: seven powers doing something other than what they print
# --------------------------------------------------------------------------- #


def test_toki_stops_at_the_top_of_the_tavern(patch):
    """"A Tier higher than yours" has nowhere to go at Tier 6. The tier-7 cards
    the pool keeps out of the counter are not this power's to hand out."""
    from src.bg_recruitment.shop import refresh_shop

    cap = patch.meta.ruleset.max_tier
    for seed in range(8):
        player = _armed(patch, "TB_BaconShop_HERO_28", tier=cap)
        refresh_shop(player, None, rng=np.random.default_rng(seed), patch=patch)
        _press(patch, player, seed=seed)
        assert all(m.tier <= cap for m in player.shop if m is not None), seed


def test_toki_offers_the_tier_above_while_there_is_one(patch):
    from src.bg_recruitment.shop import refresh_shop

    player = _armed(patch, "TB_BaconShop_HERO_28", tier=4)
    refresh_shop(player, None, rng=np.random.default_rng(3), patch=patch)
    _press(patch, player, seed=3)
    assert sum(1 for m in player.shop if m is not None and m.tier == 5) >= 2


def test_toki_leaves_a_frozen_slot_alone(patch):
    """A frozen slot is not one of the slots the refresh rerolled, so it is not
    one of the slots the higher-Tier pair replaces either."""
    from src.bg_recruitment.shop import refresh_shop

    player = _armed(patch, "TB_BaconShop_HERO_28", tier=3)
    refresh_shop(player, None, rng=np.random.default_rng(5), patch=patch)
    player.shop_frozen = tuple(
        [True, True] + [False] * (len(player.shop_frozen) - 2)
    )
    kept = [player.shop[0].card_id, player.shop[1].card_id]
    _press(patch, player, seed=5)
    assert [player.shop[0].card_id, player.shop[1].card_id] == kept


def test_edwins_power_improves_with_the_buys_it_counts(patch):
    """"Improves after you buy 4 cards" reads the plain count of buys, which
    belongs to the purchase rather than to any one passive that asks for it."""
    for buys, wanted in ((0, 2), (3, 2), (4, 4), (8, 6)):
        player = _armed(
            patch, "TB_BaconShop_HERO_01", board=[patch.make_minion("BGS_115")]
        )
        player.hero_buy_count = buys
        body = player.board[0]
        before = (body.raw_attack, body.max_health)
        _press(patch, player)
        assert (body.raw_attack - before[0], body.max_health - before[1]) == (
            wanted,
            wanted,
        ), buys


def test_a_buy_is_counted_once_however_many_passives_read_it(patch):
    """The counter is the purchase's, not a passive's: a hero with no
    every-Nth-buy passive still counts what it bought."""
    player = _armed(patch, "TB_BaconShop_HERO_01", gold=30)
    for _ in range(3):
        hero_passives.apply_hero_on_bought(
            patch.make_minion("BGS_115"), player, patch=patch
        )
    assert player.hero_buy_count == 3
    assert player.hero_tiers_bought == 3 * patch.templates["BGS_115"].tier


def test_inge_swaps_which_stat_she_pays(patch):
    """"(Swaps to Health next turn!)" — the same number, the other half of the
    body, every other turn."""
    player = _armed(patch, "BG26_HERO_102", tier=5, board=[patch.make_minion("BGS_115")])
    body = player.board[0]
    seen = []
    for rnd in (1, 2, 3, 4):
        player.round_number = rnd
        player.hero_power_uses_this_turn = 0
        before = (body.raw_attack, body.max_health)
        hero_passives.use_hero_power(
            player, rng=np.random.default_rng(0), patch=patch, round_number=rnd
        )
        seen.append(
            (body.raw_attack - before[0], body.max_health - before[1])
        )
    assert seen == [(5, 0), (0, 5), (5, 0), (0, 5)]


def test_the_lich_kings_power_is_not_a_spell_cast_on_the_body(patch):
    """Nothing was played from hand, so the cards that read a cast do not read
    this: no +1 Health from a spell watcher, and Lava Lurker's "the first
    Spellcraft spell on this is permanent" is not spent on it."""
    watcher = next(
        (
            c
            for c in sorted(patch.pool_ids)
            if any(
                ab.trigger is Trigger.ON_TARGETED_BY_SPELL
                for ab in patch.templates[c].abilities
            )
        ),
        None,
    )
    player = _armed(patch, "TB_BaconShop_HERO_22", board=[patch.make_minion("BGS_115")])
    body = player.board[0]
    before = (body.raw_attack, body.max_health)
    _press(patch, player)
    assert Keyword.REBORN in body.all_keywords
    assert (body.raw_attack, body.max_health) == before
    if watcher is not None:
        player = _armed(
            patch, "TB_BaconShop_HERO_22", board=[patch.make_minion(watcher)]
        )
        body = player.board[0]
        before = (body.raw_attack, body.max_health)
        _press(patch, player)
        assert (body.raw_attack, body.max_health) == before


def test_bazhial_and_armor(patch):
    """Covered by ``test_bazhial_steals_and_bleeds``; kept here as the shape of
    the rule — the shop-phase damage path is the combat one."""
    from src.bg_recruitment.shop_triggers import ShopTriggers

    player = _seat(patch)
    player.armor = 5
    ShopTriggers(np.random.default_rng(0), patch=patch).damage_hero(player, 3)
    assert (player.health, player.armor) == (patch.meta.ruleset.starting_health, 2)
    ShopTriggers(np.random.default_rng(0), patch=patch).damage_hero(player, 4)
    assert (player.health, player.armor) == (patch.meta.ruleset.starting_health - 2, 0)


def test_a_fallback_pick_is_one_the_effect_can_use(patch):
    """With no seat to ask, the target is drawn from the bodies the effect can
    actually help — Reno has one press for the whole game to spend."""
    for seed in range(20):
        player = _armed(
            patch,
            "TB_BaconShop_HERO_41",
            board=[patch.make_minion("BGS_115"), patch.make_minion("BGS_115")],
        )
        player.board[0].is_golden = True
        _press(patch, player, seed=seed)
        assert player.board[1].is_golden, seed

    for seed in range(20):
        player = _armed(
            patch,
            "TB_BaconShop_HERO_15",
            board=[patch.make_minion("BGS_115"), patch.make_minion("BGS_115")],
        )
        player.board[0].has_shield = True
        player.board[0].granted_keywords = frozenset({Keyword.SHIELD})
        _press(patch, player, seed=seed)
        assert player.board[1].has_shield, seed


def test_a_press_that_could_do_nothing_is_not_offered(patch):
    """The tavern greys the button out rather than taking the gold for nothing:
    no Undead to destroy, no counter to swap with, no room for the card."""
    from src.bg_recruitment.shop import refresh_shop

    def armed(hero_id, **kw):
        return _armed(patch, hero_id, round_number=9, **kw)

    beast = next(
        c for c in sorted(patch.pool_ids) if patch.templates[c].race is Race.BEAST
    )
    # The Jailer: "Destroy a friendly Undead" with no Undead on the board.
    jailer = armed("TB_BaconShop_HERO_702", board=[patch.make_minion(beast)])
    assert not hero_passives.can_use_hero_power(jailer, 9)
    undead = next(
        c for c in sorted(patch.pool_ids) if patch.templates[c].race is Race.UNDEAD
    )
    jailer.board.append(patch.make_minion(undead))
    assert hero_passives.can_use_hero_power(jailer, 9)
    # ...and with nowhere for the Undead it gets to land.
    jailer.hand = [patch.make_minion(beast) for _ in jailer.hand]
    assert not hero_passives.can_use_hero_power(jailer, 9)

    # Jandice and Malygos both need something on the counter.
    for hero_id in ("TB_BaconShop_HERO_71", "TB_BaconShop_HERO_58"):
        player = armed(hero_id, board=[patch.make_minion(beast)])
        assert not hero_passives.can_use_hero_power(player, 9), hero_id
        refresh_shop(player, None, rng=np.random.default_rng(0), patch=patch)
        assert hero_passives.can_use_hero_power(player, 9), hero_id

    # Mutanus and Vol'jin need two bodies: one to spend and one to receive.
    for hero_id in ("BG20_HERO_301", "BG20_HERO_201"):
        player = armed(hero_id, board=[patch.make_minion(beast)])
        assert not hero_passives.can_use_hero_power(player, 9), hero_id
        player.board.append(patch.make_minion(beast))
        assert hero_passives.can_use_hero_power(player, 9), hero_id

    # Tess reads the last warband she fought.
    tess = armed("TB_BaconShop_HERO_50")
    assert not hero_passives.can_use_hero_power(tess, 9)
    tess.last_opponent_board = [patch.make_minion(beast)]
    assert hero_passives.can_use_hero_power(tess, 9)

    # Elise Discovers into the hand, and pays more every time — a press with a
    # full hand would climb the price for nothing.
    elise = armed("TB_BaconShop_HERO_42", tier=4)
    assert hero_passives.can_use_hero_power(elise, 9)
    elise.hand = [patch.make_minion(beast) for _ in elise.hand]
    assert not hero_passives.can_use_hero_power(elise, 9)


# --------------------------------------------------------------------------- #
# What the state audit found
# --------------------------------------------------------------------------- #


def test_a_tier_discover_that_could_not_open_is_still_owed(patch):
    """The queue is what makes a chain open one modal at a time. Taking the
    head off before knowing it opened threw the whole chain away the moment the
    hand happened to be full."""
    beast = next(
        c for c in sorted(patch.pool_ids) if patch.templates[c].race is Race.BEAST
    )
    player = _seat(patch, "BG22_HERO_201")  # Ambassador Faelin
    player.tavern_tier = 6
    player.hero_pending_tier_discovers = (6, 4, 2)
    player.hand = [patch.make_minion(beast) for _ in player.hand]
    hero_passives.flush_hero_tier_discovers(
        player, rng=np.random.default_rng(0), patch=patch
    )
    assert player.pending_choice is None
    assert player.hero_pending_tier_discovers == (6, 4, 2)

    player.hand[0] = None
    hero_passives.flush_hero_tier_discovers(
        player, rng=np.random.default_rng(0), patch=patch
    )
    assert player.pending_choice is not None
    assert player.hero_pending_tier_discovers == (4, 2)


def test_a_hero_power_discover_only_offers_heroes_that_have_one(patch):
    """A passive hero has no power to offer, so picking one used to *delete*
    the seat's rather than replace it."""
    for seed in range(30):
        player = _seat(patch, "BG20_HERO_202")  # Master Nguyen
        hero_passives._open_hero_power_discover(
            player, rng=np.random.default_rng(seed), patch=patch, options=2
        )
        options = player.pending_choice.options
        assert len(options) == 2, seed
        assert all(patch.heroes[o].has_power() for o in options), seed
        assert player.hero.hero_id not in options, seed


def test_picking_a_power_leaves_the_hero_that_picked_it(patch):
    """Master Nguyen Discovers "at the start of every turn" — which he can only
    do while he is still the hero doing it."""
    from src.bg_recruitment.discover import resolve_discover_pick

    player = _seat(patch, "BG20_HERO_202")
    for rnd in (1, 2, 3):
        hero_passives.apply_hero_on_turn_start(
            player, round_number=rnd, patch=patch, rng=np.random.default_rng(rnd)
        )
        assert player.pending_choice is not None, rnd
        taken = patch.heroes[player.pending_choice.options[0]]
        resolve_discover_pick(
            player,
            0,
            None,
            rng=np.random.default_rng(0),
            on_after_placed=lambda *_: None,
            patch=patch,
        )
        assert player.hero.hero_id == "BG20_HERO_202", rnd
        assert player.hero.power == taken.power, rnd
        assert player.hero.passives == patch.heroes["BG20_HERO_202"].passives, rnd


def test_a_swapped_in_power_does_not_settle_a_backlog_it_never_earned(patch):
    """The countdown counters belong to the seat. When a Discover replaced the
    whole hero, one turn start paid out every reward the new hero's passive
    would have owed for a game it did not play."""
    from src.bg_recruitment.discover import resolve_discover_pick

    player = _seat(patch, "BG20_HERO_202")
    player.hero_deaths = 40
    player.hero_attacks = 90
    hero_passives.apply_hero_on_turn_start(
        player, round_number=1, patch=patch, rng=np.random.default_rng(0)
    )
    resolve_discover_pick(
        player,
        0,
        None,
        rng=np.random.default_rng(0),
        on_after_placed=lambda *_: None,
        patch=patch,
    )
    hero_passives.apply_hero_on_turn_start(
        player, round_number=2, patch=patch, rng=np.random.default_rng(0)
    )
    assert _hand(player) == [] or all(
        c is None for c in player.hand[1:]
    ), "a power swap paid out a countdown the seat never ran"


def test_thorim_pays_when_the_gold_is_spent(patch):
    """"After you spend 60 Gold" is a threshold the purchase crosses, so the
    card arrives in the hand that just paid for it."""
    from src.bg_recruitment.economy import note_gold_spent

    player = _seat(patch, "BG27_HERO_801")
    assert player.hero_promised_card
    promised = player.hero_promised_card
    for _ in range(20):
        note_gold_spent(player, 3, patch=patch)
    assert promised in _hand(player)
    assert player.hero_gold_paid == 1
    before = len(_hand(player))
    for _ in range(20):
        note_gold_spent(player, 3, patch=patch)
    assert len(_hand(player)) == before


def test_yogg_casts_into_the_tavern_the_seat_will_be_shown(patch):
    """Half of what a Tavern spell does is to the counter, so casting before
    the turn's refresh threw the spell away on every turn but the ones that
    happened not to touch the shop."""
    from src.bg_recruitment.shop import refresh_shop

    hits = 0
    for seed in range(12):
        player = _seat(patch, "TB_BaconShop_HERO_35")
        player.tavern_tier = 5
        hero_passives.apply_hero_on_turn_start(
            player, round_number=6, patch=patch, rng=np.random.default_rng(seed)
        )
        refresh_shop(player, None, rng=np.random.default_rng(seed), patch=patch)
        before = [
            (m.card_id, m.raw_attack, m.max_health) if m else None for m in player.shop
        ]
        hero_passives.apply_hero_after_shop_refresh(
            player, 6, patch=patch, rng=np.random.default_rng(seed)
        )
        after = [
            (m.card_id, m.raw_attack, m.max_health) if m else None for m in player.shop
        ]
        assert player.last_tavern_spell_cast is not None, seed
        hits += before != after
    # Not every random spell touches the counter, but some must now be able to.
    assert hits > 0


# --------------------------------------------------------------------------- #
# What the fuzzer found
# --------------------------------------------------------------------------- #


def test_the_structured_path_can_press_the_power(patch):
    """The flat mask offered it and the structured one did not, which on the
    training path is a power the policy can never reach."""
    from src.envs.bglike.action_map import struct_action_to_game_action
    from src.envs.bglike import actions as A
    from src.envs.minibg.structured_actions import (
        StructAction,
        StructActionType,
        validate_struct_action,
    )

    token = StructAction(StructActionType.HERO_POWER, ())
    validate_struct_action(token, hand_size=10, board_size=7, max_shop_slots=7)
    assert struct_action_to_game_action(token) == int(A.Action.HERO_POWER)
    with pytest.raises(ValueError):
        validate_struct_action(
            StructAction(StructActionType.HERO_POWER, (0,)),
            hand_size=10,
            board_size=7,
            max_shop_slots=7,
        )


def test_tess_copies_a_warband_that_fought_with_a_token(patch):
    """Fishbait is built in code and has no printing, so building the offer
    from its card id raised. Any seat can buy one and take it to a combat."""
    from src.bg_lobby.shared_pool import build_initial_shared_pool
    from src.bg_recruitment.fishbait import make_fishbait
    from src.bg_recruitment.shop import refresh_shop

    pool = build_initial_shared_pool(patch=patch)
    player = _armed(patch, "TB_BaconShop_HERO_50", round_number=8)
    refresh_shop(player, None, rng=np.random.default_rng(1), patch=patch, shared_pool=pool)
    plain = next(c for c in sorted(patch.pool_ids) if patch.templates[c].tier == 1)
    player.last_opponent_board = [make_fishbait(), patch.make_minion(plain)]
    before = pool.remaining_copies(plain)
    hero_passives.use_hero_power(
        player,
        rng=np.random.default_rng(1),
        patch=patch,
        round_number=8,
        shared_pool=pool,
    )
    offers = [m.card_id for m in player.shop if m is not None]
    assert "BG36_205" in offers
    # And the copy on the counter is one the lobby lent, not one minted.
    assert pool.remaining_copies(plain) == before - 1


def test_a_destroyed_minion_goes_back_to_the_lobby(patch):
    """Hooktusk presses this for free, once a turn, all game — every press used
    to take a copy out of the shared pool for good."""
    from src.bg_lobby.shared_pool import build_initial_shared_pool

    plain = next(c for c in sorted(patch.pool_ids) if patch.templates[c].tier == 1)
    pool = build_initial_shared_pool(patch=patch)
    player = _armed(
        patch, "TB_BaconShop_HERO_67", round_number=8, board=[patch.make_minion(plain)]
    )
    before = pool.remaining_copies(plain)
    hero_passives.use_hero_power(
        player,
        rng=np.random.default_rng(0),
        patch=patch,
        round_number=8,
        shared_pool=pool,
    )
    assert player.board == []
    assert pool.remaining_copies(plain) == before + 1


def test_a_borrowed_power_brings_its_own_charges(patch):
    """``power_charges`` belongs to the power. A once-per-game power picked up
    by a seat that had already pressed one arrived unusable."""
    from src.bg_recruitment.discover import resolve_discover_pick

    player = _armed(patch, "BG20_HERO_202", round_number=8)
    player.hero_power_uses_game = 4
    player.hero_power_uses_this_turn = 1
    player.hero_power_cost_delta = 3
    hero_passives._open_hero_power_discover(
        player, rng=np.random.default_rng(0), patch=patch, options=2
    )
    resolve_discover_pick(
        player,
        0,
        None,
        rng=np.random.default_rng(0),
        on_after_placed=lambda *_: None,
        patch=patch,
    )
    assert player.hero_power_uses_game == 0
    assert player.hero_power_uses_this_turn == 0
    assert player.hero_power_cost_delta == 0


def test_a_hero_the_lobbys_tribes_cannot_serve_is_not_dealt(patch):
    """Alexstrasza with no Dragons pays a gold for a Discover that cannot be
    filled; the tavern leaves her out of the offer instead."""
    from collections import Counter

    dragon_locked = {"Alexstrasza", "Ysera"}
    seen = Counter()
    for i in range(200):
        player = _seat(patch)
        hero_passives.assign_random_hero(
            player,
            patch=patch,
            rng=np.random.default_rng(i),
            shop_excluded_race=(Race.DRAGON,),
        )
        seen[player.hero.name] += 1
    assert not (dragon_locked & set(seen))
    assert len(seen) > 40  # and the rest of the pool is still dealt

    open_pool = Counter()
    for i in range(200):
        player = _seat(patch)
        hero_passives.assign_random_hero(
            player, patch=patch, rng=np.random.default_rng(i)
        )
        open_pool[player.hero.name] += 1
    assert dragon_locked & set(open_pool)


def test_a_gem_needs_somewhere_to_go_and_reborn_needs_a_body_without_it(patch):
    beast = next(
        c for c in sorted(patch.pool_ids) if patch.templates[c].race is Race.BEAST
    )
    gems = _armed(
        patch, "BG20_HERO_103", round_number=8, board=[patch.make_minion(beast)]
    )
    assert hero_passives.can_use_hero_power(gems, 8)
    gems.hand = [patch.make_minion(beast) for _ in gems.hand]
    assert not hero_passives.can_use_hero_power(gems, 8)

    king = _armed(
        patch, "TB_BaconShop_HERO_22", round_number=8, board=[patch.make_minion(beast)]
    )
    assert hero_passives.can_use_hero_power(king, 8)
    king.board[0].granted_keywords = frozenset({Keyword.REBORN})
    assert not hero_passives.can_use_hero_power(king, 8)


def test_pressing_the_power_costs_none_of_the_turns_budget(patch):
    """It is its own resource, spent by its own per-turn count and its own
    price — not by one of the turn's shop actions."""
    from src.bg_player_turn.context import PlayerTurnContext
    from src.bg_player_turn.engine import PlayerTurnEngine
    from src.bg_recruitment.shop_triggers import ShopTriggers
    from src.envs.bglike import actions as A

    player = _armed(patch, "BG32_HERO_001", round_number=8)  # Cenarius, no target
    rng = np.random.default_rng(0)
    ctx = PlayerTurnContext(
        rng=rng, triggers=ShopTriggers(rng, patch=patch), patch=patch, round_number=8
    )
    engine = PlayerTurnEngine(A)
    assert int(A.Action.HERO_POWER) in engine.legal_actions(player, patch.meta.ruleset)
    assert engine.apply(player, int(A.Action.HERO_POWER), ctx) is False


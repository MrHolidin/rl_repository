"""Tavern spells the tavern offers, played rather than inspected.

Every one of these was offerable and inert — the seat could buy it and casting
it did nothing. They are bindings on top of effects built for minions that say
the same sentence, so what is worth pinning is that each one *lands*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PendingChoiceKind, PlayerPhase, PlayerState
from src.bg_recruitment.standing_bonuses import settle_standing_bonuses
from src.bg_recruitment.tavern_spells import cast_tavern_spell

PATCH_DIR = Path("data/bgcore/36_2_0_248348")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


def _player(patch, board=(), **kw) -> PlayerState:
    base = dict(
        health=30, gold=10, tavern_tier=6, board=list(board), shop=[None] * 7,
        hand=[None] * 10, phase=PlayerPhase.SHOP, shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _m(card_id="m", atk=1, hp=1, race=None, keywords=frozenset()):
    return Minion(
        card_id=card_id, base_attack=atk, base_health=hp, tier=1,
        race=race, keywords=keywords,
    )


def _cast(patch, spell_id, player, target=None):
    cast_tavern_spell(
        player,
        patch.tavern_spells[spell_id],
        rng=np.random.default_rng(0),
        patch=patch,
        target=target,
    )
    return player


def test_no_offerable_spell_is_inert(patch):
    """The count only goes down. A spell the tavern offers and that does
    nothing is a card the seat can waste gold on."""
    inert = [s for s in patch.tavern_spells.values() if s.in_pool and not s.abilities]
    assert len(inert) <= 16


def test_defenders_rites(patch):
    target = _m()
    _cast(patch, "BG28_825", _player(patch, [target]), target)
    assert (target.raw_attack, target.max_health) == (8, 8)
    assert Keyword.TAUNT in target.all_keywords


def test_sacred_gift(patch):
    target = _m()
    _cast(patch, "BG28_507", _player(patch, [target]), target)
    assert Keyword.SHIELD in target.all_keywords and target.has_shield


def test_perfect_vision_sets_rather_than_adds(patch):
    target = _m(atk=9, hp=9)
    _cast(patch, "BG28_838", _player(patch, [target]), target)
    assert (target.raw_attack, target.max_health) == (20, 20)


def test_might_of_stormwind_pays_four_of_six(patch):
    board = [_m(f"x{i}") for i in range(6)]
    _cast(patch, "BG35_951", _player(patch, board))
    assert sum(1 for x in board if x.max_health > 1) == 4


def test_sanctify_pays_only_divine_shields(patch):
    shielded = _m("s", keywords=frozenset({Keyword.SHIELD}))
    plain = _m("p")
    _cast(patch, "BG33_817", _player(patch, [shielded, plain]))
    assert (shielded.raw_attack, plain.raw_attack) == (7, 1)


def test_queens_command_pays_naga_twice(patch):
    naga = _m("n", race=Race.NAGA)
    other = _m("o")
    _cast(patch, "BG35_922", _player(patch, [naga, other]))
    assert (naga.raw_attack, other.raw_attack) == (5, 3)


def test_azerite_empowerment_lands_twice(patch):
    target = _m("a")
    _cast(patch, "BG28_169", _player(patch, [target]))
    assert (target.raw_attack, target.max_health) == (5, 5)


def test_strike_oil_raises_the_gold_cap(patch):
    """It rewrites the seat's own ruleset, which is where the cap lives."""
    before = patch.meta.ruleset.gold_cap
    player = _cast(patch, "BG28_805", _player(patch))
    assert player.ruleset.gold_cap == before + 1
    assert patch.meta.ruleset.gold_cap == before  # the package is untouched


def test_leaf_through_the_pages_gives_two_free_refreshes(patch):
    player = _cast(patch, "BG28_827", _player(patch))
    assert player.free_roll_charges == 2


def test_careful_investment_pays_next_turn(patch):
    player = _cast(patch, "BG28_800", _player(patch))
    assert player.gold_next_turn == 2


def test_staff_of_enrichment_pays_the_tavern_for_the_game(patch):
    player = _player(patch)
    player.shop[0] = _m("s0")
    _cast(patch, "BG28_886", player)
    settle_standing_bonuses(player)
    assert (player.shop[0].raw_attack, player.shop[0].max_health) == (3, 3)


def test_weapons_forge_hands_over_three_arrows(patch):
    player = _cast(patch, "BG36_884", _player(patch))
    assert [c.card_id for c in player.hand if c is not None] == ["EBG_Spell_014"] * 3


def test_tomb_turning_discovers_undead(patch):
    player = _cast(patch, "BG34_888", _player(patch))
    pc = player.pending_choice
    assert pc is not None and pc.kind is PendingChoiceKind.DISCOVER_TRIBE
    assert all(patch.templates[cid].race is Race.UNDEAD for cid in pc.options)


@pytest.mark.parametrize(
    "spell_id,eats", [("BG28_607", 3), ("BG36_880", 2)]
)
def test_a_demon_eats_off_the_counter(patch, spell_id, eats):
    """The spell names the Demon, so the target is the eater."""
    demon = _m("d", race=Race.DEMON)
    player = _player(patch, [demon])
    for i in range(4):
        player.shop[i] = _m(f"v{i}", 2, 2)
    _cast(patch, spell_id, player, demon)
    assert (demon.raw_attack, demon.max_health) == (1 + 2 * eats, 1 + 2 * eats)
    assert sum(1 for x in player.shop if x is not None) == 4 - eats


# --------------------------------------------------------------------------- #
# Discovers narrowed by something that is not a tribe
# --------------------------------------------------------------------------- #


def _catalog_mechanics(patch):
    import json

    rows = json.load(open(PATCH_DIR / "catalog.json"))["minions"]
    return {r["id"]: set(r.get("mechanics") or ()) for r in rows}


def test_contracted_corpse_offers_only_deathrattles(patch):
    tags = _catalog_mechanics(patch)
    player = _cast(patch, "BG28_882", _player(patch))
    options = player.pending_choice.options
    assert options
    assert all("DEATHRATTLE" in tags[cid] for cid in options)


def test_hired_headhunter_offers_only_battlecries(patch):
    """"A Battlecry minion" is the catalog tag *or* a binding that fires on
    play — the tag is missing on a few, Choose One cards among them."""
    from src.envs.minibg.summon_pool import record_has_battlecry

    tags = _catalog_mechanics(patch)
    player = _cast(patch, "BG28_GIL_836", _player(patch))
    options = player.pending_choice.options
    assert options
    assert all(
        record_has_battlecry(cid, frozenset(tags[cid]), patch.effects)
        for cid in options
    )


def test_planar_telescope_reads_the_board_for_its_tribe(patch):
    """"your most common type" is not named on the card — it is counted."""
    board = [_m("a", race=Race.MURLOC), _m("b", race=Race.MURLOC), _m("c", race=Race.BEAST)]
    player = _cast(patch, "BG28_521", _player(patch, board))
    assert all(
        patch.templates[cid].race is Race.MURLOC
        for cid in player.pending_choice.options
    )


def test_planar_telescope_on_an_empty_board_offers_nothing(patch):
    player = _cast(patch, "BG28_521", _player(patch))
    assert player.pending_choice is None


def test_search_through_time_offers_exactly_your_tier(patch):
    """A bare Discover is your tier *or below*; this one says "of your Tier"."""
    for tier in (3, 4, 6):
        player = _cast(patch, "BG34_330", _player(patch, tavern_tier=tier))
        assert {patch.templates[c].tier for c in player.pending_choice.options} == {tier}


def test_armor_stash_sets_rather_than_adds(patch):
    player = _player(patch)
    player.armor = 12
    _cast(patch, "BG28_500", player)
    assert player.armor == 5


# ------------------------------------- the buffs that read their target


def test_tricky_trousers_gives_taunt(patch):
    target = _m()
    _cast(patch, "BG28_520", _player(patch, [target]), target)
    assert (target.raw_attack, target.max_health) == (2, 3)
    assert Keyword.TAUNT in target.all_keywords


def test_tricky_trousers_takes_taunt_off_a_minion_that_has_it(patch):
    """"If it already has Taunt, remove it" — the stats still land."""
    target = _m(keywords=frozenset({Keyword.TAUNT}))
    _cast(patch, "BG28_520", _player(patch, [target]), target)
    assert (target.raw_attack, target.max_health) == (2, 3)
    assert Keyword.TAUNT not in target.all_keywords


def test_shifting_tide_is_two_buffs(patch):
    target = _m(race=Race.BEAST)
    _cast(patch, "BG32_815", _player(patch, [target]), target)
    assert (target.raw_attack, target.max_health) == (3, 3)


def test_shifting_tide_repeats_on_a_naga(patch):
    target = _m(race=Race.NAGA)
    _cast(patch, "BG32_815", _player(patch, [target]), target)
    assert (target.raw_attack, target.max_health) == (5, 5)


def test_eonars_favor_reads_the_targets_tribe(patch):
    """The scope is not printed on the card — it is whatever was chosen."""
    target = _m(race=Race.BEAST)
    player = _player(patch, [target])
    player.shop[0] = _m("beast-in-shop", race=Race.BEAST)
    player.shop[1] = _m("murloc-in-shop", race=Race.MURLOC)
    _cast(patch, "BG35_912", player, target)
    settle_standing_bonuses(player)
    assert (player.shop[0].raw_attack, player.shop[0].max_health) == (4, 4)
    assert (player.shop[1].raw_attack, player.shop[1].max_health) == (1, 1)


def test_eonars_favor_leaves_the_board_alone(patch):
    """"in the Tavern" — the minion named is not itself paid."""
    target = _m(race=Race.BEAST)
    player = _cast(patch, "BG35_912", _player(patch, [target]), target)
    settle_standing_bonuses(player)
    assert (target.raw_attack, target.max_health) == (1, 1)


def test_eonars_favor_on_a_tribeless_minion_scopes_nothing(patch):
    target = _m(race=None)
    player = _player(patch, [target])
    player.shop[0] = _m("in-shop", race=Race.BEAST)
    _cast(patch, "BG35_912", player, target)
    settle_standing_bonuses(player)
    assert (player.shop[0].raw_attack, player.shop[0].max_health) == (1, 1)


def test_wave_of_gold_pays_golden_minions_twice(patch):
    plain, golden = _m("plain"), _m("golden")
    golden.is_golden = True
    _cast(patch, "BG34_990", _player(patch, [plain, golden]))
    assert (plain.raw_attack, plain.max_health) == (4, 3)
    assert (golden.raw_attack, golden.max_health) == (7, 5)


def test_menagerie_tableware_repeats_per_minion_type(patch):
    board = [_m("a", race=Race.BEAST), _m("b", race=Race.BEAST), _m("c", race=Race.MURLOC)]
    _cast(patch, "BG34_272", _player(patch, board))
    for m in board:
        assert (m.raw_attack, m.max_health) == (7, 7)


def test_menagerie_tableware_pays_nothing_without_a_type(patch):
    """Same reading every other "Repeat for each" card gets: no count, no buff."""
    target = _m(race=None)
    _cast(patch, "BG34_272", _player(patch, [target]))
    assert (target.raw_attack, target.max_health) == (1, 1)


def test_cloning_conch_hands_over_the_same_murloc_twice(patch):
    player = _cast(patch, "BG28_601", _player(patch))
    got = [c for c in player.hand if c is not None]
    assert len(got) == 2
    assert got[0].card_id == got[1].card_id
    assert patch.templates[got[0].card_id].race is Race.MURLOC


def test_spitescale_special_draws_from_the_spellcraft_pool(patch):
    """Spellcraft spells are minted by Nagas, never offered on the counter."""
    from src.bg_recruitment.tavern_spells import spellcraft_spell_ids

    player = _cast(patch, "BG28_606", _player(patch))
    got = [c for c in player.hand if c is not None]
    assert len(got) == 3
    assert all(c.card_id in spellcraft_spell_ids(patch) for c in got)
    assert not any(patch.tavern_spells[c.card_id].in_pool for c in got)


# ------------------------------- a body traded for what it becomes


def _inert(patch, card_id="BGS_119"):
    """A body with no text of its own — Crackling Cyclone is keywords only.

    Worth insisting on: the first draft of these probes used Charging Czarina,
    whose "whenever you cast a Tavern spell" buffs the board every time and
    made every number below look wrong.
    """
    return patch.make_minion(card_id)


def test_golden_touch_makes_a_tavern_minion_golden(patch):
    offer = _inert(patch)
    player = _player(patch)
    player.shop[0] = offer
    printed = patch.templates[offer.card_id]
    _cast(patch, "BG28_830", player)
    assert offer.is_golden
    assert offer.raw_attack == printed.base_attack * 2
    assert offer.max_health == printed.base_health * 2


def test_a_golden_offer_takes_the_two_copies_it_now_stands_for(patch):
    """The slot reserved one when it was filled; clearing it releases three."""
    from src.bg_lobby.shared_pool import build_initial_shared_pool

    pool = build_initial_shared_pool(patch=patch)
    offer = _inert(patch)
    player = _player(patch)
    player.shop[0] = offer
    before = pool.remaining_copies(offer.card_id)
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG28_830"],
        rng=np.random.default_rng(0),
        patch=patch,
        shared_pool=pool,
    )
    assert pool.remaining_copies(offer.card_id) == before - 2


def test_a_made_golden_keeps_the_plain_card_id(patch):
    """A forged Golden carries it, and the pool and the triple scan are keyed
    on it — ``is_golden`` is the whole difference between the printings."""
    from src.bg_recruitment.targeted_battlecry import make_golden

    body = _inert(patch)
    make_golden(body, patch=patch)
    assert body.is_golden and body.card_id == "BGS_119"


def test_eyes_of_the_earth_mother_respects_the_printed_cap(patch):
    low = _inert(patch)  # Tier 1
    player = _player(patch, [low])
    _cast(patch, "EBG_Spell_017", player, low)
    assert low.is_golden

    high = patch.make_minion("BG25_354")  # Tier 5
    player = _player(patch, [high])
    _cast(patch, "EBG_Spell_017", player, high)
    assert not high.is_golden


def test_robust_evolution_keeps_the_stats_and_takes_the_card(patch):
    body = _inert(patch)
    body.bonus_attack, body.bonus_health = 10, 10
    attack, health = body.raw_attack, body.max_health
    _cast(patch, "BG30_804", _player(patch, [body]), body)
    assert body.card_id != "BGS_119"
    assert patch.templates[body.card_id].tier == 2
    assert (body.raw_attack, body.max_health) == (attack, health)


def test_robust_evolution_on_a_tier_seven_body_does_nothing(patch):
    top = next(c for c in patch.pool_ids if patch.templates[c].tier == 7)
    body = patch.make_minion(top)
    _cast(patch, "BG30_804", _player(patch, [body]), body)
    assert body.card_id == top


def test_mounting_avalanche_sells_and_pays_the_left_most_elemental(patch):
    elemental = next(
        c for c in patch.pool_ids if patch.templates[c].race is Race.ELEMENTAL
    )
    left, right = patch.make_minion(elemental), patch.make_minion(elemental)
    victim = _inert(patch)
    victim.bonus_attack, victim.bonus_health = 5, 5
    attack, health = victim.raw_attack, victim.max_health
    printed = patch.templates[elemental]
    player = _player(patch, [left, victim, right])
    gold = player.gold

    _cast(patch, "BG33_899", player, victim)
    assert victim not in player.board
    assert player.gold == gold + 1  # sold, not destroyed
    assert (left.raw_attack, left.max_health) == (
        printed.base_attack + attack,
        printed.base_health + health,
    )
    assert (right.raw_attack, right.max_health) == (
        printed.base_attack,
        printed.base_health,
    )


def test_mounting_avalanche_still_sells_with_no_elemental_to_pay(patch):
    victim, other = _inert(patch), _inert(patch)
    player = _player(patch, [victim, other])
    gold = player.gold
    _cast(patch, "BG33_899", player, victim)
    assert player.board == [other]
    assert player.gold == gold + 1


def test_channel_the_devourer_pays_a_friendly_that_is_left(patch):
    victim, heir = _inert(patch), _inert(patch)
    victim.bonus_attack, victim.bonus_health = 7, 7
    attack, health = victim.raw_attack, victim.max_health
    printed = patch.templates[heir.card_id]
    player = _player(patch, [victim, heir])
    _cast(patch, "EBG_Spell_032", player, victim)
    assert player.board == [heir]
    assert (heir.raw_attack, heir.max_health) == (
        printed.base_attack + attack,
        printed.base_health + health,
    )


# --------------------------- Start of Combat, bought a turn early


def _start_of_combat(patch, player, mine, theirs, seed=1):
    """Fire only the Start of Combat window, and hand back both sides.

    Not a whole fight: what these spells promise happens before a blow is
    struck, and a fight would bury the result under the swings that follow.
    """
    from src.bg_combat.battle.engine import _fire_start_of_combat
    from src.bg_combat.battle.seat import RecordingSeat
    from src.bg_combat.battle.sides import _build_side
    from src.bg_combat.battle.state import BattleSide, _CombatRuntime
    from src.bg_recruitment.combat_seat import PlayerCombatSeat

    rt = _CombatRuntime(
        sides=(BattleSide([]), BattleSide([])),
        rng=np.random.default_rng(seed),
        combat_board_max=7,
        damage_cap=15,
        patch=patch,
        seats=(PlayerCombatSeat(player), RecordingSeat()),
    )
    rt.sides = (_build_side(mine, rt), _build_side(theirs, rt))
    _fire_start_of_combat(rt)
    return list(rt.side(0).minions), list(rt.side(1).minions)


def test_a_start_of_combat_spell_does_nothing_when_it_is_cast(patch):
    """The seat holds the promise; the next fight is what reads it."""
    player = _player(patch)
    _cast(patch, "BG34_889", player)
    assert len(player.start_combat_promises) == 1


def test_the_promise_is_spent_by_one_fight(patch):
    from src.bg_recruitment.shop_triggers import ShopTriggers

    player = _player(patch)
    _cast(patch, "BG28_573", player)
    ShopTriggers(np.random.default_rng(0), patch=patch).fire_on_turn_start(player)
    assert player.start_combat_promises == ()


def test_upper_hand_writes_an_enemys_health_to_one(patch):
    player = _player(patch)
    _cast(patch, "BG28_573", player)
    _, theirs = _start_of_combat(patch, player, [_m("mine", 1, 1)], [_m("big", 5, 50)])
    assert theirs[0].max_health == 1
    assert theirs[0].raw_attack == 5  # Health only, and nothing was dealt


def test_upper_hand_without_the_spell_leaves_the_enemy_alone(patch):
    _, theirs = _start_of_combat(
        patch, _player(patch), [_m("mine", 1, 1)], [_m("big", 5, 50)]
    )
    assert theirs[0].max_health == 50


def test_brood_of_nozdormu_doubles_only_the_left_most(patch):
    player = _player(patch)
    _cast(patch, "BG34_889", player)
    mine, _ = _start_of_combat(
        patch, player, [_m("left", 5, 9), _m("right", 5, 9)], [_m("foe", 1, 1)]
    )
    assert (mine[0].raw_attack, mine[1].raw_attack) == (10, 5)


def test_sharing_is_caring_takes_the_stats_of_the_body_opposite(patch):
    player = _player(patch)
    _cast(patch, "BG31_889", player)
    mine, _ = _start_of_combat(patch, player, [_m("left", 1, 100)], [_m("foe", 7, 9)])
    assert (mine[0].raw_attack, mine[0].max_health) == (8, 109)


def test_a_start_of_combat_spell_does_not_write_to_the_seats_board(patch):
    """Combat runs on copies; a promise is no exception."""
    player = _player(patch)
    _cast(patch, "BG34_889", player)
    body = _m("left", 5, 9)
    _start_of_combat(patch, player, [body], [_m("foe", 1, 1)])
    assert body.raw_attack == 5

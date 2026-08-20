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
    assert len(inert) <= 36


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
    tags = _catalog_mechanics(patch)
    player = _cast(patch, "BG28_GIL_836", _player(patch))
    options = player.pending_choice.options
    assert options
    assert all("BATTLECRY" in tags[cid] for cid in options)


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

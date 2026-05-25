"""Triple-forged golden scaling for patch 74257 (implicit + catalog hints)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_catalog.triple_effects import resolve_triple_forged_abilities
from src.bg_core.effects import (
    AddFromLastOpponentBoardEffect,
    BuffSelf,
    BuffSelfFromFriendlyTribeCount,
    BuffSelfFromHeroDamageTaken,
    BuffAllOtherOfTribe,
    DealDamageRandomEnemyMinion,
    DealHeroDamage,
    DiscoverMurlocEffect,
    IncrementShopTribeBonusEffect,
    Keyword,
    MultiplySelfAttackEffect,
    ReduceUpgradeCostEffect,
    StartOfCombatDamagePerFriendlyTribe,
    TransformIntoShopMinionEffect,
    TriggerRandomFriendlyDeathrattleEffect,
)
from src.bg_recruitment.triples import merge_three_non_golden_into_golden
from src.envs.minibg.state import Minion
from dataclasses import replace

PATCH_74257 = Path("data/bgcore/19_6_0_74257")


@pytest.fixture(scope="module")
def ctx_74257() -> PatchContext:
    return PatchContext.load(PATCH_74257)


def _forged(card_id: str, ctx: PatchContext):
    return resolve_triple_forged_abilities(
        card_id, ctx.effects, catalog_path=ctx.patch_dir / "catalog.json"
    )


def test_forged_nomi_doubles_shop_bonus(ctx_74257):
    abs_ = _forged("BGS_104", ctx_74257)
    eff = abs_[0].effect
    assert isinstance(eff, IncrementShopTribeBonusEffect)
    assert eff.attack == 2 and eff.health == 2


def test_forged_hangry_doubles_buff(ctx_74257):
    eff = _forged("BGS_033", ctx_74257)[0].effect
    assert isinstance(eff, BuffSelf)
    assert eff.attack == 4 and eff.health == 4


def test_forged_wrath_preserves_hero_damage_doubles_self_buff(ctx_74257):
    forged = _forged("BGS_004", ctx_74257)
    dmg = forged[0].effect
    buff = forged[1].effect
    assert isinstance(dmg, DealHeroDamage) and dmg.amount == 1
    assert isinstance(buff, BuffSelf) and buff.attack == 4 and buff.health == 4


def test_forged_soul_juggler_twice_not_double_amount(ctx_74257):
    eff = _forged("BGS_002", ctx_74257)[0].effect
    assert isinstance(eff, DealDamageRandomEnemyMinion)
    assert eff.amount == 3 and eff.repeats == 2


def test_forged_red_whelp_soc_twice(ctx_74257):
    eff = _forged("BGS_019", ctx_74257)[0].effect
    assert isinstance(eff, StartOfCombatDamagePerFriendlyTribe)
    assert eff.amount_per_match == 1 and eff.repeats == 2


def test_forged_macaw_doubles_dr_triggers(ctx_74257):
    eff = _forged("BGS_078", ctx_74257)[0].effect
    assert isinstance(eff, TriggerRandomFriendlyDeathrattleEffect)
    assert eff.repeats == 2


def test_forged_glyph_triple_attack_factor(ctx_74257):
    eff = _forged("BGS_045", ctx_74257)[0].effect
    assert isinstance(eff, MultiplySelfAttackEffect)
    assert eff.factor == 3


def test_forged_primalfin_two_discovers(ctx_74257):
    eff = _forged("BGS_020", ctx_74257)[0].effect
    assert isinstance(eff, DiscoverMurlocEffect)
    assert eff.repeats == 2


def test_forged_annihilan_double_health_per_damage(ctx_74257):
    eff = _forged("BGS_010", ctx_74257)[0].effect
    assert isinstance(eff, BuffSelfFromHeroDamageTaken)
    assert eff.health_per_damage == 2


def test_forged_deck_swabbie_reduces_two(ctx_74257):
    eff = _forged("BGS_055", ctx_74257)[0].effect
    assert isinstance(eff, ReduceUpgradeCostEffect)
    assert eff.amount == 2


def test_forged_razorgore_doubles_per_dragon(ctx_74257):
    eff = _forged("BGS_036", ctx_74257)[0].effect
    assert isinstance(eff, BuffSelfFromFriendlyTribeCount)
    assert eff.attack_per == 2 and eff.health_per == 2


def test_merge_nomi_golden_uses_forged_abilities(ctx_74257):
    tpl = ctx_74257.templates["BGS_104"]
    a = replace(ctx_74257.make_minion("BGS_104"), abilities=tpl.abilities)
    b = replace(a)
    c = replace(a)
    g = merge_three_non_golden_into_golden("BGS_104", a, b, c, patch=ctx_74257)
    eff = g.abilities[0].effect
    assert isinstance(eff, IncrementShopTribeBonusEffect)
    assert eff.attack == 2


def test_forged_murozond_make_golden(ctx_74257):
    eff = _forged("BGS_043", ctx_74257)[0].effect
    assert isinstance(eff, AddFromLastOpponentBoardEffect)
    assert eff.make_golden


def test_forged_faceless_copy_golden(ctx_74257):
    eff = _forged("BGS_113", ctx_74257)[0].effect
    assert isinstance(eff, TransformIntoShopMinionEffect)
    assert eff.copy_golden


def test_forged_seabreaker_authored_overkill(ctx_74257):
    eff = _forged("BGS_080", ctx_74257)[0].effect
    assert isinstance(eff, BuffAllOtherOfTribe)
    assert eff.attack == 4 and eff.health == 4


def test_merge_cyclone_mega_windfury(ctx_74257):
    tpl = ctx_74257.templates["BGS_119"]
    a = replace(ctx_74257.make_minion("BGS_119"), abilities=tpl.abilities)
    g = merge_three_non_golden_into_golden(
        "BGS_119", a, replace(a), replace(a), patch=ctx_74257
    )
    assert Keyword.MEGA_WINDFURY in g.all_keywords
    assert Keyword.WINDFURY in g.all_keywords


def test_forged_imprisoner_summons_2_2_imp(ctx_74257):
    from src.bg_core.effects import SummonEffect

    eff = _forged("BGS_014", ctx_74257)[0].effect
    assert isinstance(eff, SummonEffect)
    assert eff.token_id == "TB_BaconUps_030t" and eff.count == 1


def test_forged_refreshing_anomaly_two_free_rolls(ctx_74257):
    from src.bg_core.effects import SetNextRollCostEffect

    eff = _forged("BGS_116", ctx_74257)[0].effect
    assert isinstance(eff, SetNextRollCostEffect)
    assert eff.cost == 0 and eff.uses == 2


def test_forged_wildfire_both_adjacent(ctx_74257):
    from src.bg_core.effects import DealExcessDamageToAdjacentEffect

    eff = _forged("BGS_126", ctx_74257)[0].effect
    assert isinstance(eff, DealExcessDamageToAdjacentEffect)
    assert eff.both_adjacent

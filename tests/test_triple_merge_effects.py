"""Triple-forged golden must not stack three copies of the same ability."""

from dataclasses import replace

from src.envs.minibg.card_pool import EFFECTS, triple_merge_golden_abilities
from src.envs.minibg.effects import SummonEffect, SummonMultiplierAura
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.state import Minion


def test_triple_merge_brann_abilities_match_authored_golden_row():
    assert triple_merge_golden_abilities("LOE_077") == EFFECTS["TB_BaconUps_045"]


def test_triple_merge_brann_singleton_factor_three_not_three_copies():
    merged = triple_merge_golden_abilities("LOE_077")
    assert len(merged) == 1
    assert merged[0].effect.factor == 3


def test_triple_merge_pack_rat_deathrattle_uses_waves_not_triplicate_ability_rows():
    abs_ = triple_merge_golden_abilities("CFM_316")
    assert len(abs_) == 1
    eff = abs_[0].effect
    assert isinstance(eff, SummonEffect)
    assert eff.count_from_source_attack
    assert eff.dr_wave_count == 2


def test_merge_three_non_golden_into_golden_replaces_abilities_from_canonical_tables():
    a = Minion(
        card_id="FP1_031",
        base_attack=1,
        base_health=7,
        tier=5,
        name="Baron Rivendare",
        abilities=EFFECTS["FP1_031"],
    )
    b = replace(a)
    c = replace(a)
    g = MiniBGGame._merge_three_non_golden_into_golden("FP1_031", a, b, c)
    assert g.abilities == EFFECTS["TB_BaconUps_055"]
    assert len(g.abilities) == 1


def test_merge_three_summon_multiplier_is_singleton_golden_three():
    a = Minion(
        card_id="DAL_575",
        base_attack=2,
        base_health=2,
        tier=3,
        name="Khadgar",
        abilities=EFFECTS["DAL_575"],
    )
    b = replace(a)
    c = replace(a)
    g = MiniBGGame._merge_three_non_golden_into_golden("DAL_575", a, b, c)
    assert len(g.abilities) == 1
    assert isinstance(g.abilities[0].effect, SummonMultiplierAura)
    assert g.abilities[0].effect.factor == 3

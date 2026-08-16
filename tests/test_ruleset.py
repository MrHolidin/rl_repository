"""DEFAULT_RULESET must reproduce the pre-existing bglike/minibg action-module
constants exactly, and meta.json parsing must round-trip both a missing
"ruleset" key (existing patch packages) and a partial/full override."""

import pytest

from src.bg_catalog.ruleset import DamageCapStep, DEFAULT_RULESET, Ruleset, ruleset_from_meta
from src.envs.bglike import actions as bglike_actions


def test_default_ruleset_matches_bglike_actions_constants():
    r = DEFAULT_RULESET
    assert r.max_tier == bglike_actions.MAX_TIER
    assert dict(r.level_up_costs) == bglike_actions.LEVEL_UP_COSTS
    assert r.level_up_discount_per_round == bglike_actions.LEVEL_UP_DISCOUNT_PER_ROUND
    assert dict(r.gold_per_round) == bglike_actions.GOLD_PER_ROUND
    assert r.gold_cap == bglike_actions.GOLD_AT_CAP
    assert r.buy_cost == bglike_actions.BUY_COST
    assert r.sell_reward == bglike_actions.SELL_REWARD
    assert r.roll_cost == bglike_actions.ROLL_COST
    assert r.starting_health == bglike_actions.STARTING_HEALTH
    assert r.max_rounds == bglike_actions.MAX_ROUNDS
    # Flat cap forever, matching the old single DAMAGE_CAP scalar.
    for round_number in (1, 4, 5, 8, 9, 50):
        assert r.damage_cap_for_round(round_number) == bglike_actions.DAMAGE_CAP
    # Lift disabled by default (old-patch behavior: cap never lifts).
    assert r.effective_damage_cap(50, 1) == bglike_actions.DAMAGE_CAP


def test_gold_for_round_matches_helper():
    for round_number in range(1, 12):
        assert DEFAULT_RULESET.gold_for_round(round_number) == bglike_actions.gold_for_round(
            round_number
        )


def test_level_up_cost_matches_dict_lookup():
    for tier in (1, 2, 3, 4, 5):
        assert DEFAULT_RULESET.level_up_cost(tier) == bglike_actions.LEVEL_UP_COSTS[tier]
    # Tier 6 (max) has no upgrade cost entry — falls back to 0, same as the
    # `.get(tier, 0)` pattern call sites used before this module existed.
    assert DEFAULT_RULESET.level_up_cost(6) == 0


def test_ruleset_from_meta_none_or_missing_key_is_default():
    assert ruleset_from_meta(None) is DEFAULT_RULESET
    assert ruleset_from_meta({}) is DEFAULT_RULESET


def test_ruleset_from_meta_partial_override():
    r = ruleset_from_meta({"starting_health": 30, "roll_cost": 2})
    assert r.starting_health == 30
    assert r.roll_cost == 2
    # Untouched fields still fall back to the same defaults.
    assert r.gold_cap == DEFAULT_RULESET.gold_cap
    assert dict(r.level_up_costs) == dict(DEFAULT_RULESET.level_up_costs)


def test_raising_max_tier_without_pricing_the_new_upgrade_is_rejected():
    """The tier-7 trap: level_up_cost falls back to 0, so the upgrade is free."""
    with pytest.raises(ValueError, match=r"no price for tiers \[6\]"):
        ruleset_from_meta({"max_tier": 7})


def test_raising_max_tier_with_the_new_price_is_accepted():
    r = ruleset_from_meta(
        {
            "max_tier": 7,
            "level_up_costs": {"1": 5, "2": 7, "3": 8, "4": 9, "5": 10, "6": 11},
        }
    )
    assert r.max_tier == 7
    assert r.level_up_cost(6) == 11


def test_ruleset_from_meta_damage_cap_schedule_and_lift():
    r = ruleset_from_meta(
        {
            "damage_cap_schedule": [[4, 5], [8, 10], [None, 15]],
            "damage_cap_lifted_at_alive": 4,
        }
    )
    assert r.damage_cap_schedule == (
        DamageCapStep(4, 5),
        DamageCapStep(8, 10),
        DamageCapStep(None, 15),
    )
    # Ramp boundaries.
    assert r.damage_cap_for_round(1) == 5
    assert r.damage_cap_for_round(4) == 5
    assert r.damage_cap_for_round(5) == 10
    assert r.damage_cap_for_round(8) == 10
    assert r.damage_cap_for_round(9) == 15
    assert r.damage_cap_for_round(50) == 15
    # Lift: uncapped once alive_count <= 4, still ramped above that.
    assert r.effective_damage_cap(9, 5) == 15
    assert r.effective_damage_cap(9, 4) > 15
    assert r.effective_damage_cap(1, 4) > 15
    assert r.effective_damage_cap(1, 8) == 5


def test_ruleset_is_frozen_and_comparable():
    a = Ruleset()
    b = Ruleset()
    assert a == b
    try:
        a.max_tier = 7  # frozen dataclass — must reject mutation
    except (AttributeError, TypeError) as exc:
        # dataclasses.FrozenInstanceError subclasses AttributeError
        return
    raise AssertionError("Ruleset should be immutable")

"""Step-3 DvD metrics: preset columns + agent metric emission.

The CSV logger drops any key absent from the resolved fieldnames, so the
``ppo_dvd`` preset must carry every ``dvd_*`` key the agent emits.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.agents import PPODvDAgent  # noqa: F401 (registers "ppo")
from src.bg_catalog.patch_context import load_patch_context
from src.bg_core.minion import Minion, Race
from src.envs.bglike.actions import NUM_ACTIONS
from src.envs.bglike.board_descriptor import board_descriptor
from src.envs.bglike.obs_v5 import OBS_DIM_V5
from src.registry import make_agent
from src.training.metrics_presets import (
    METRICS_PRESET_PPO,
    resolve_metrics_csv_fieldnames,
)

_PATCH = "data/bgcore/15_6_2_36393"
_N_ID = 4


def _m(race):
    return Minion(card_id=f"t_{race.name}", base_attack=3, base_health=4, tier=2, race=race)


def _state(board, tavern_tier=3):
    return SimpleNamespace(players=[SimpleNamespace(board=board, tavern_tier=tavern_tier)])


def _make_agent(diversity_coef=0.3):
    ctx = load_patch_context(_PATCH)
    return make_agent(
        "ppo",
        network_type="bglike_structured_v7",
        num_identities=_N_ID,
        diversity_coef=diversity_coef,
        num_actions=NUM_ACTIONS,
        observation_shape=(OBS_DIM_V5,),
        observation_type="vector",
        num_pool_indices=ctx.num_pool_indices,
        device="cpu",
    )


def test_preset_superset_of_ppo_and_has_dvd_columns():
    cols = resolve_metrics_csv_fieldnames("ppo", preset="ppo_dvd")
    # Strict superset of the plain PPO preset (no regression for existing tools).
    assert set(METRICS_PRESET_PPO).issubset(set(cols))
    for key in (
        "dvd_pop_diversity",
        "dvd_identity_coverage",
        "dvd_distinct_tribes",
        "dvd_placement_best",
        "dvd_mean_bonus",
        "dvd_bonus_place_ratio",
        "dvd_identity_contrib_norm",
        "dvd_place_0",
        "dvd_tribe_0",
    ):
        assert key in cols


def test_emitted_metric_keys_are_all_in_preset():
    """Every key the agent emits must be loggable (else silently dropped)."""
    agent = _make_agent(diversity_coef=0.3)
    cols = set(resolve_metrics_csv_fieldnames("ppo", preset="ppo_dvd"))

    # Populate two identities with distinct builds + placements.
    agent._active_identity = 0
    agent._desc_by_seat[0] = board_descriptor(_state([_m(Race.MECHANICAL)] * 5), 0)
    agent.close_segment(seat=0, terminal_reward=0.6)  # uses empty buffer → no-op stamp
    agent._active_identity = 1
    agent._desc_by_seat[0] = board_descriptor(_state([_m(Race.ELEMENTAL)] * 5), 0)
    agent.close_segment(seat=0, terminal_reward=-0.2)

    metrics = agent._dvd_metrics()
    assert metrics  # non-empty
    unknown = set(metrics) - cols
    assert not unknown, f"emitted keys not in preset: {unknown}"


def test_metrics_values_and_accumulator_reset():
    agent = _make_agent(diversity_coef=0.5)
    # Per-seat mode: two seats with explicitly assigned distinct identities.
    agent._identity_by_seat = {0: 0, 1: 1}
    agent._desc_by_seat[0] = board_descriptor(_state([_m(Race.MECHANICAL)] * 5), 0)
    agent.close_segment(seat=0, terminal_reward=1.0)
    agent._desc_by_seat[1] = board_descriptor(_state([_m(Race.MURLOC)] * 5), 0)
    agent.close_segment(seat=1, terminal_reward=0.0)

    m = agent._dvd_metrics()
    assert m["dvd_identity_coverage"] == 2.0 / _N_ID
    assert m["dvd_pop_diversity"] > 0.0
    assert m["dvd_distinct_tribes"] == 2.0
    assert m["dvd_place_0"] == 1.0
    assert m["dvd_place_1"] == 0.0
    # accumulators reset after emit
    assert agent._acc_n == 0
    assert agent._acc_bonus == 0.0
    assert agent._acc_abs_place == 0.0

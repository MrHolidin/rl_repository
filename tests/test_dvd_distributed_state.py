"""Distributed DvD-state transport: workers accumulate φ, host merges + emits.

Regression for the bug where dist_ppo runs logged all-zero dvd_* metrics: the
host runs ``update()`` / metric emission but has no DvD state of its own, so a
worker snapshot must be merged in first.
"""

from __future__ import annotations

import numpy as np

from src.agents import PPODvDAgent  # noqa: F401 (registers "ppo")
from src.bg_catalog.patch_context import load_patch_context
from src.bg_core.minion import Minion, Race
from src.envs.bglike.actions import NUM_ACTIONS
from src.envs.bglike.board_descriptor import board_descriptor
from src.envs.bglike.obs_v5 import OBS_DIM_V5
from src.registry import make_agent

_PATCH = "data/bgcore/15_6_2_36393"
_N_ID = 4


def _agent(diversity_coef=0.3):
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


def _state(board, tavern_tier=3):
    from types import SimpleNamespace

    return SimpleNamespace(
        players=[SimpleNamespace(board=board, tavern_tier=tavern_tier)]
    )


def _m(race):
    return Minion(card_id=f"t_{race.name}", base_attack=3, base_health=4, tier=2, race=race)


def _populate(agent, seat, identity, race, place):
    agent._identity_by_seat = {seat: identity}
    agent._desc_by_seat[seat] = board_descriptor(_state([_m(race)] * 5), 0)
    agent.close_segment(seat=seat, terminal_reward=place)


def test_snapshot_roundtrips_into_fresh_host():
    worker = _agent()
    _populate(worker, seat=0, identity=0, race=Race.MECHANICAL, place=0.8)
    _populate(worker, seat=1, identity=1, race=Race.ELEMENTAL, place=-0.4)

    snap = worker.dvd_state_snapshot()
    host = _agent()
    assert not host._phi_seen.any()  # fresh host has nothing
    host.merge_dvd_state([snap])

    # Host now reflects the worker's identities + placements.
    assert host._phi_seen[0] and host._phi_seen[1]
    assert np.isclose(host._placement_ema[0], 0.8)
    assert np.isclose(host._placement_ema[1], -0.4)
    m = host._dvd_metrics()
    assert m["dvd_pop_diversity"] > 0.0
    assert m["dvd_distinct_tribes"] == 2.0


def test_merge_averages_two_workers_per_identity():
    w1 = _agent()
    _populate(w1, seat=0, identity=0, race=Race.MECHANICAL, place=1.0)
    w2 = _agent()
    _populate(w2, seat=0, identity=0, race=Race.MECHANICAL, place=0.0)

    host = _agent()
    host.merge_dvd_state([w1.dvd_state_snapshot(), w2.dvd_state_snapshot()])
    # Same identity seen by both workers → placement averaged.
    assert np.isclose(host._placement_ema[0], 0.5)
    # Window accumulators sum across workers.
    assert host._acc_n == 2


def test_merge_empty_is_noop():
    host = _agent()
    host.merge_dvd_state([])
    assert not host._phi_seen.any()

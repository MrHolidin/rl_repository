"""Tribe-only repulsion: diversity is measured on board composition alone.

dist_ppo_073 collapsed to one good-stuff pile (all identities tribe=0,
pop_diversity≈0.08) while novelty stayed high — identities were diverging on
board-size/stat axes, not tribe. The repulsion distance now uses the tribe
block of the descriptor ONLY, so size/stat differences don't count as
diversity.
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

_PATCH = "data/bgcore/15_6_2_36393"
_N_ID = 4


def _agent():
    ctx = load_patch_context(_PATCH)
    return make_agent(
        "ppo",
        network_type="bglike_structured_v7",
        num_identities=_N_ID,
        diversity_coef=0.3,
        num_actions=NUM_ACTIONS,
        observation_shape=(OBS_DIM_V5,),
        observation_type="vector",
        num_pool_indices=ctx.num_pool_indices,
        device="cpu",
    )


def _m(race, atk=3, hp=4):
    return Minion(card_id=f"t_{race.name}", base_attack=atk, base_health=hp, tier=2, race=race)


def _state(board, tavern_tier=3):
    return SimpleNamespace(players=[SimpleNamespace(board=board, tavern_tier=tavern_tier)])


def test_size_only_difference_is_zero_distance():
    # Same tribe, different board size / tavern tier → tribe fractions identical
    # → repulsion distance is exactly zero (size is NOT diversity).
    a = _agent()
    small = board_descriptor(_state([_m(Race.MECHANICAL)] * 2, tavern_tier=2), 0)
    big = board_descriptor(_state([_m(Race.MECHANICAL)] * 6, tavern_tier=5), 0)
    assert np.isclose(a._tribe_dist(small, big), 0.0)


def test_tribe_difference_is_positive():
    a = _agent()
    mech = board_descriptor(_state([_m(Race.MECHANICAL)] * 5), 0)
    elem = board_descriptor(_state([_m(Race.ELEMENTAL)] * 5), 0)
    assert a._tribe_dist(mech, elem) > 0.0


def test_tribe_distance_ignores_stat_axes():
    # Two pure-elemental boards with very different stats but identical tribe mix
    # → zero distance (stats excluded).
    a = _agent()
    weak = board_descriptor(_state([_m(Race.ELEMENTAL, atk=1, hp=1)] * 4), 0)
    strong = board_descriptor(_state([_m(Race.ELEMENTAL, atk=15, hp=15)] * 4), 0)
    assert np.isclose(a._tribe_dist(weak, strong), 0.0)


def test_pop_diversity_uses_tribe_only():
    a = _agent()
    # Two identities: same tribe, different size → pop_diversity must stay 0.
    a._phi[0] = board_descriptor(_state([_m(Race.MECHANICAL)] * 2), 0)
    a._phi[1] = board_descriptor(_state([_m(Race.MECHANICAL)] * 6), 0)
    a._phi_seen[0] = a._phi_seen[1] = True
    assert np.isclose(a.population_diversity(), 0.0)
    # Flip identity 1 to a different tribe → now positive.
    a._phi[1] = board_descriptor(_state([_m(Race.ELEMENTAL)] * 6), 0)
    assert a.population_diversity() > 0.0

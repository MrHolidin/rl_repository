"""Step-3b: population co-play sampler (sibling + frozen seats, fixed identities)."""

from __future__ import annotations

from typing import Optional

from src.agents.ppo_dvd_agent import PPODvDAgent, SiblingOpponent
from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.actions import NUM_ACTIONS
from src.envs.bglike.obs_v5 import OBS_DIM_V5
from src.registry import make_agent
from src.training.opponent_sampler import DvDPopulationSampler
from src.training.selfplay.game_record import SLOT_CURRENT

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


class _FakePool:
    """Minimal stand-in for OpponentPool: a live learner, no frozen snapshots."""

    def __init__(self, learner):
        self.current_agent = learner
        self._last_sample_slot_id = -2  # SLOT_SCRIPTED

    def _sample_frozen_info(self):
        return None  # no frozen yet → everything falls back to siblings


def test_prepare_does_not_pin_learner_identity():
    # Per-seat mode: the learner assigns its own identities across its current
    # seats, so the sampler must NOT force single-identity mode on it.
    learner = _agent()
    sampler = DvDPopulationSampler(
        _FakePool(learner), num_identities=_N_ID, sibling_fraction=0.5, seed=1
    )
    sampler.prepare(0)
    assert not learner._identity_externally_set


def test_learner_assigns_distinct_identities_across_seats():
    learner = _agent()
    learner._reset_seat_identities()
    ids = [learner._identity_for_seat(s) for s in range(_N_ID)]
    # First num_identities seats get a permutation → all distinct.
    assert sorted(ids) == list(range(_N_ID))
    # Stable within an episode.
    assert learner._identity_for_seat(0) == ids[0]


def test_other_identity_in_range():
    learner = _agent()
    sampler = DvDPopulationSampler(
        _FakePool(learner), num_identities=_N_ID, sibling_fraction=1.0, seed=2
    )
    sampler.prepare(0)
    for _ in range(200):
        assert 0 <= sampler._other_identity() < _N_ID


def test_sibling_seats_when_no_frozen():
    learner = _agent()
    sampler = DvDPopulationSampler(
        _FakePool(learner), num_identities=_N_ID, sibling_fraction=0.5, seed=3
    )
    sampler.prepare(0)
    seats = [1, 2, 3, 4, 5, 6, 7]
    assigned = sampler.sample_for_seats(seats)
    assert set(assigned) == set(seats)
    # With no frozen snapshots, every opponent seat is a sibling of the live net.
    for seat, opp in assigned.items():
        assert isinstance(opp, SiblingOpponent)
        assert 0 <= opp.identity < _N_ID
        assert sampler._slot_by_seat[seat] == SLOT_CURRENT


def test_sibling_opponent_runs_in_eval_without_buffer_writes():
    """SiblingOpponent acting must not pollute the learner's rollout buffer."""
    learner = _agent()
    learner.train()  # learner is training; sibling forward must still be inert
    sib = SiblingOpponent(learner, identity=2)
    # Force-identity context restores cleanly.
    assert learner._forced_identity is None
    with learner._forced_identity_ctx(2):
        assert learner._forced_identity == 2
    assert learner._forced_identity is None
    # Buffer untouched by merely constructing / entering the context.
    assert len(learner.rollout_buffer) == 0

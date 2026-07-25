"""Tier-milestone shaping: parsing, payout, and per-seat attribution.

The failure mode this guards is silent: a milestone that never pays, or pays to
the wrong seat's segment, is indistinguishable from "the shaping did nothing" --
which is exactly the hypothesis the shaping run is meant to test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.training.bglike_perspective import (
    BGLikeAgentPerspectiveEnv,
    parse_tier_milestones,
)


def _env(milestones=None, decay=0.8, n_seats=8):
    env = BGLikeAgentPerspectiveEnv.__new__(BGLikeAgentPerspectiveEnv)
    env._tier_milestones = milestones if milestones is not None else {5: (0.20, 8)}
    env._tier_milestone_decay = decay
    env._tier_paid = {}
    players = [SimpleNamespace(tavern_tier=1) for _ in range(n_seats)]
    env._bg_base = SimpleNamespace(
        state=SimpleNamespace(players=players, round_number=1)
    )
    return env


def _step(env, seat, tier, rnd):
    env._bg_base.state.players[seat].tavern_tier = tier
    env._bg_base.state.round_number = rnd
    return env._tier_milestone_reward({"acting_seat": seat})


# --- parsing ---------------------------------------------------------------


def test_parse_accepts_mapping_and_pair_forms():
    assert parse_tier_milestones({5: {"base": 0.2, "target_round": 8}}) == {5: (0.2, 8)}
    assert parse_tier_milestones({6: [0.15, 11]}) == {6: (0.15, 11)}


def test_parse_empty_is_disabled():
    assert parse_tier_milestones(None) == {}
    assert parse_tier_milestones({}) == {}


@pytest.mark.parametrize(
    "bad",
    [
        {9: {"base": 0.2, "target_round": 3}},   # tier outside 2..6
        {1: {"base": 0.2, "target_round": 3}},   # starting tier
        {5: {"base": 0.2, "target_round": 0}},   # round < 1
    ],
)
def test_parse_rejects_unusable_spec(bad):
    with pytest.raises(ValueError):
        parse_tier_milestones(bad)


def test_parse_rejects_missing_base():
    with pytest.raises(KeyError):
        parse_tier_milestones({5: {"target_round": 8}})


# --- payout ----------------------------------------------------------------


def test_no_payout_below_milestone_tier():
    env = _env()
    assert _step(env, 0, 4, 7) == 0.0


def test_on_target_round_pays_full_base():
    env = _env()
    assert _step(env, 0, 5, 8) == pytest.approx(0.20)


def test_late_arrival_decays_but_early_is_not_boosted():
    env = _env()
    assert _step(env, 0, 5, 12) == pytest.approx(0.20 * 0.8**4)
    # reaching it early must not pay more than base
    assert _step(_env(), 1, 5, 5) == pytest.approx(0.20)


def test_paid_once_per_seat_per_lobby():
    env = _env()
    assert _step(env, 0, 5, 8) > 0.0
    assert _step(env, 0, 5, 9) == 0.0
    assert _step(env, 0, 6, 10) == 0.0  # t6 not configured here


def test_attribution_does_not_leak_between_seats():
    env = _env()
    _step(env, 0, 5, 8)
    # seat 3 has its own entitlement, unaffected by seat 0 having been paid
    assert _step(env, 3, 5, 12) == pytest.approx(0.20 * 0.8**4)


def test_skipping_a_tier_pays_every_milestone_crossed():
    env = _env({5: (0.20, 8), 6: (0.15, 11)})
    assert _step(env, 0, 6, 11) == pytest.approx(0.20 * 0.8**3 + 0.15)


def test_lobby_boundary_restores_entitlement():
    env = _env()
    assert _step(env, 0, 5, 8) > 0.0
    env._tier_paid = {}  # what notify_episode_end does
    assert _step(env, 0, 5, 8) == pytest.approx(0.20)


def test_disabled_when_unconfigured():
    env = _env({})
    assert _step(env, 0, 6, 8) == 0.0


def test_missing_acting_seat_is_not_an_error():
    env = _env()
    assert env._tier_milestone_reward({}) == 0.0

"""League scores from BGLike placement."""

from __future__ import annotations

from src.envs.bglike.placement import (
    placement_reward,
    placement_reward_to_score,
    placement_score,
)
from src.training.selfplay.league_state import LeagueController, normalize_agent_score


def test_placement_score_bounds_and_monotone():
    scores = [placement_score(p) for p in range(1, 9)]
    assert scores[0] == 1.0
    assert scores[-1] == 0.0
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_placement_reward_to_score_matches_place():
    for place in range(1, 9):
        assert placement_reward_to_score(placement_reward(place)) == placement_score(place)


def test_legacy_int_outcomes_normalize():
    assert normalize_agent_score(1) == 1.0
    assert normalize_agent_score(-1) == 0.0
    assert normalize_agent_score(0) == 0.5


def test_league_ema_uses_continuous_placement():
    league = LeagueController(ema_beta=1.0)
    league.add_frozen_bytes(b"x", episode=0)
    slot_id = 0
    league.apply_outcomes([(slot_id, placement_score(1))])
    assert league.get_slot(slot_id).ema_win_rate == 0.0
    league.apply_outcomes([(slot_id, placement_score(8))])
    assert league.get_slot(slot_id).ema_win_rate == 1.0
    league.apply_outcomes([(slot_id, placement_score(4))])
    assert abs(league.get_slot(slot_id).ema_win_rate - (1.0 - placement_score(4))) < 1e-9

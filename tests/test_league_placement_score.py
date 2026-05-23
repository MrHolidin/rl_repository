"""League scores from BGLike placement."""

from __future__ import annotations

from src.envs.bglike.placement import (
    pairwise_learner_score,
    placement_reward,
    placement_reward_to_score,
    placement_score,
)
from src.training.selfplay.game_record import (
    GameRecord,
    ParticipantOutcome,
    league_updates_from_record,
)
from src.training.selfplay.league_state import SLOT_CURRENT, normalize_agent_score


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


def test_pairwise_learner_score():
    assert pairwise_learner_score(2, 5) == 1.0
    assert pairwise_learner_score(5, 2) == 0.0
    assert pairwise_learner_score(4, 4) == 0.5


def test_league_updates_from_record_matches_pairwise_score():
    slot_id = 0
    record = GameRecord(
        (
            ParticipantOutcome(SLOT_CURRENT, 4),
            ParticipantOutcome(slot_id, 5),
        )
    )
    updates = league_updates_from_record(record)
    assert len(updates) == 1
    assert updates[0] == (slot_id, pairwise_learner_score(4, 5))

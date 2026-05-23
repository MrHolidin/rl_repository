"""TrueSkill rating backend."""

from __future__ import annotations

import pytest

trueskill = pytest.importorskip("trueskill")

from src.training.selfplay.game_record import GameRecord, ParticipantOutcome, SLOT_CURRENT, build_scripted_slot_map
from src.training.selfplay.league_state import LeagueController
from src.training.selfplay.rating_system import TrueSkillRating, make_rating_system


def test_trueskill_updates_mu_from_full_placement():
    rating = TrueSkillRating(draw_probability=0.0)
    rating.register(SLOT_CURRENT)
    rating.register(0)
    record = GameRecord(
        (
            ParticipantOutcome(SLOT_CURRENT, 1),
            ParticipantOutcome(0, 3),
        )
    )
    assert rating.update(record) is True
    learner = rating.summary(SLOT_CURRENT)
    opp = rating.summary(0)
    assert learner["mu"] > opp["mu"]
    assert opp["games"] == 1


def test_trueskill_league_controller_via_config():
    league = LeagueController(rating_kind="trueskill", trueskill={"draw_probability": 0.0})
    league.register_meta_slot(SLOT_CURRENT)
    league.register_scripted_slots(build_scripted_slot_map(["random", "structured"]))
    sid = league.add_frozen_bytes(b"x", episode=0)
    league.submit(
        GameRecord(
            (
                ParticipantOutcome(SLOT_CURRENT, 2),
                ParticipantOutcome(sid, 5),
            )
        )
    )
    payload = league.get_status_file_data()
    assert payload["rating_system"] == "trueskill"
    agents = payload["agents"]
    assert len(agents) == 4
    current_row = next(r for r in agents if r["kind"] == "current")
    assert current_row["slot_id"] == SLOT_CURRENT
    assert current_row["label"] == "learner"
    assert current_row["games"] == 1
    assert "match_quality_vs_current" not in current_row
    frozen_row = next(r for r in agents if r["kind"] == "frozen")
    assert frozen_row["mu"] is not None
    assert frozen_row["sigma"] is not None
    assert frozen_row["games"] == 1
    assert "match_quality_vs_current" in frozen_row
    assert "win_rate" not in frozen_row
    assert "wins" not in frozen_row
    scripted_rows = [r for r in agents if r["kind"] == "scripted"]
    assert {r["label"] for r in scripted_rows} == {"random", "structured"}
    for row in agents:
        assert "ema_win_rate" not in row
        assert "selection_probability" not in row


def test_trueskill_status_excludes_ema_fields():
    league = LeagueController(rating_kind="trueskill", trueskill={"draw_probability": 0.0})
    league.register_meta_slot(SLOT_CURRENT)
    stats = league.get_pool_stats_for_status()
    assert len(stats) == 1
    assert stats[0]["kind"] == "current"
    assert stats[0]["games"] == 0
    assert "match_quality_vs_current" not in stats[0]


def test_trueskill_eight_player_record():
    rating = make_rating_system("trueskill", trueskill={"draw_probability": 0.0})
    participants = tuple(
        ParticipantOutcome(i, place)
        for i, place in enumerate([1, 2, 3, 4, 5, 6, 7, 8], start=-1)
    )
    assert rating.update(GameRecord(participants)) is True
    assert rating.rating(-1) > rating.rating(0)


def test_trueskill_duplicate_slot_averages_updates():
    rating = TrueSkillRating(draw_probability=0.0)
    rating.register(0)
    rating.register(SLOT_CURRENT)
    record = GameRecord(
        (
            ParticipantOutcome(SLOT_CURRENT, 1),
            ParticipantOutcome(0, 3),
            ParticipantOutcome(0, 4),
        )
    )
    rating.update(record)
    assert rating.summary(0)["games"] == 1

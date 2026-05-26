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


def test_trueskill_frozen_slot_can_copy_current_mu_only():
    league = LeagueController(rating_kind="trueskill", trueskill={"draw_probability": 0.0})
    league.register_meta_slot(SLOT_CURRENT)
    scripted_slots = build_scripted_slot_map(["random"])
    league.register_scripted_slots(scripted_slots)
    scripted_slot = scripted_slots["random"]
    league.submit(
        GameRecord(
            (
                ParticipantOutcome(SLOT_CURRENT, 1),
                ParticipantOutcome(scripted_slot, 2),
            )
        )
    )

    current_before = next(
        row for row in league.get_status_file_data()["agents"] if row["kind"] == "current"
    )
    sid = league.add_frozen_bytes(b"x", episode=0, copy_current_mu=True)
    frozen_row = next(
        row
        for row in league.get_status_file_data()["agents"]
        if row["kind"] == "frozen" and row["slot_id"] == sid
    )

    assert frozen_row["mu"] == current_before["mu"]
    assert frozen_row["sigma"] != current_before["sigma"]
    assert frozen_row["sigma"] == pytest.approx(8.3333, abs=1e-3)


def _play_series(rating: TrueSkillRating, *, n: int, learner_wins: bool) -> None:
    """Run ``n`` 2-player matches between SLOT_CURRENT and slot 0."""
    for _ in range(n):
        a, b = (1, 2) if learner_wins else (2, 1)
        record = GameRecord(
            (
                ParticipantOutcome(SLOT_CURRENT, a),
                ParticipantOutcome(0, b),
            )
        )
        rating.update(record)


def _play_alternating(rating: TrueSkillRating, *, n: int) -> None:
    """Run ``n`` alternating-winner 2-player matches. Outcomes stay informative
    (50/50 win rate) so σ updates don't saturate to "expected result"."""
    for i in range(n):
        if i % 2 == 0:
            rec = GameRecord(
                (ParticipantOutcome(SLOT_CURRENT, 1), ParticipantOutcome(0, 2))
            )
        else:
            rec = GameRecord(
                (ParticipantOutcome(SLOT_CURRENT, 2), ParticipantOutcome(0, 1))
            )
        rating.update(rec)


def test_frozen_sigma_keeps_shrinking_below_tau_floor():
    """With the τ²-undo for non-learner slots, frozen σ must not plateau at the
    τ-induced floor (~τ × const) the way it does for the live learner.

    Uses alternating-winner matches so each game carries information (a 100%
    predictable result barely updates σ regardless of τ — it's the informative
    games that reveal the floor effect).
    """
    rating = TrueSkillRating(draw_probability=0.0)
    rating.register(SLOT_CURRENT)
    rating.register(0)
    _play_alternating(rating, n=500)

    learner_sigma = rating.summary(SLOT_CURRENT)["sigma"]
    frozen_sigma = rating.summary(0)["sigma"]

    # The learner still feels τ each match → σ plateaus well above zero.
    assert learner_sigma > 0.5
    # The frozen slot is now allowed to converge below that floor.
    assert frozen_sigma < learner_sigma * 0.6


def test_frozen_mu_still_moves_from_default_sigma_prior():
    """A freshly-added frozen agent should still get fast μ updates in the first
    few games (the point of NOT copying σ): high prior σ → big step size."""
    rating = TrueSkillRating(draw_probability=0.0)
    rating.register(SLOT_CURRENT)
    # Pin learner at a high μ; frozen starts at default μ=25.
    rating.set_mu(SLOT_CURRENT, 40.0)
    rating.register(0)

    mu_before = rating.summary(0)["mu"]
    # Frozen wins 5 in a row → μ should rise sharply because σ is still ~8.3.
    _play_series(rating, n=5, learner_wins=False)
    mu_after_5 = rating.summary(0)["mu"]

    # Step size should be on the order of σ²/(2β²+Σσ²) × outcome_density ≈ 1.0+ mu
    # per game in the high-σ regime.
    assert mu_after_5 - mu_before > 5.0


def test_tau_undo_clamps_to_floor_does_not_negativise_sigma():
    """Edge-case: if σ² drops below τ² after rate(), the floor must clamp it
    instead of producing a NaN sqrt."""
    rating = TrueSkillRating(draw_probability=0.0, tau=5.0)  # absurdly large τ
    rating.register(SLOT_CURRENT)
    rating.register(0)
    rating._ratings[0].sigma = 0.01  # below τ; would go negative without clamp
    _play_series(rating, n=1, learner_wins=True)
    assert rating.summary(0)["sigma"] > 0.0
    assert rating.summary(0)["sigma"] < 0.01  # floor, not NaN

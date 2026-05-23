"""SlotRegistry + RatingSystem + LeagueController glue."""

from __future__ import annotations

from src.training.selfplay.game_record import GameRecord, ParticipantOutcome, SLOT_CURRENT
from src.training.selfplay.league_state import LeagueController
from src.training.selfplay.rating_system import EmaPairwiseRating, make_rating_system
from src.training.selfplay.slot_registry import SlotRegistry


def test_league_controller_delegates_registry_and_rating():
    registry = SlotRegistry()
    rating = EmaPairwiseRating(ema_beta=1.0)
    league = LeagueController(registry=registry, rating=rating)
    league.register_meta_slot(SLOT_CURRENT)

    sid = league.add_frozen_bytes(b"w", episode=5)
    assert registry.get(sid) is not None
    assert rating.get_stats(sid) is not None

    league.submit(
        GameRecord(
            (
                ParticipantOutcome(SLOT_CURRENT, 1),
                ParticipantOutcome(sid, 2),
            )
        )
    )
    slot = league.get_slot(sid)
    assert slot is not None
    assert slot.games == 1
    assert slot.ema_win_rate == 0.0
    assert league.win_rates()[sid] == 0.0


def test_make_rating_system_ema_default():
    rating = make_rating_system("ema", ema_beta=0.1)
    assert isinstance(rating, EmaPairwiseRating)


def test_make_rating_system_unknown_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown rating"):
        make_rating_system("elo")

"""GameRecord league update conversion."""

from __future__ import annotations

from src.training.selfplay.game_record import (
    GameRecord,
    ParticipantOutcome,
    league_updates_from_record,
    minibg_record_from_learner_score,
)
from src.training.selfplay.league_state import SLOT_CURRENT, SLOT_SCRIPTED


def test_minibg_record_win_loss_draw():
    win = minibg_record_from_learner_score(3, 1.0)
    assert league_updates_from_record(win) == [(3, 1.0)]
    loss = minibg_record_from_learner_score(3, 0.0)
    assert league_updates_from_record(loss) == [(3, 0.0)]
    draw = minibg_record_from_learner_score(3, 0.5)
    assert league_updates_from_record(draw) == [(3, 0.5)]


def test_league_updates_pairwise_per_opponent_seat():
    record = GameRecord(
        participants=(
            ParticipantOutcome(SLOT_CURRENT, 2),
            ParticipantOutcome(SLOT_SCRIPTED, 1),
            ParticipantOutcome(7, 8),
            ParticipantOutcome(7, 8),
            ParticipantOutcome(SLOT_SCRIPTED, 4),
        )
    )
    assert league_updates_from_record(record) == [
        (SLOT_SCRIPTED, 0.0),
        (7, 1.0),
        (7, 1.0),
        (SLOT_SCRIPTED, 1.0),
    ]


def test_game_record_for_lobby_end():
    from src.training.selfplay.game_record import game_record_for_lobby_end

    record = game_record_for_lobby_end(
        current_seats=(0, 1),
        slot_by_seat={4: -2, 5: 0},
        placements_full={0: 3, 1: 5, 4: 1, 5: 8, 2: 2, 3: 4, 6: 6, 7: 7},
        slot_id_to_scripted_key={-2: "random"},
    )
    assert record is not None
    current_places = [p.placement for p in record.participants if p.slot_id == SLOT_CURRENT]
    assert current_places == [3, 5]
    assert any(p.slot_id == -2 and p.placement == 1 for p in record.participants)
    assert any(p.slot_id == 0 and p.placement == 8 for p in record.participants)


def test_league_placement_ema_from_lobby_record():
    from src.training.selfplay.league_state import LeagueController

    league = LeagueController(rating_kind="trueskill", trueskill={"draw_probability": 0.0})
    league.register_meta_slot(SLOT_CURRENT)
    sid = league.add_frozen_bytes(b"x", episode=0)
    league.submit(
        GameRecord(
            (
                ParticipantOutcome(SLOT_CURRENT, 2),
                ParticipantOutcome(sid, 5),
                ParticipantOutcome(-2, 1),
            )
        )
    )
    stats = league.get_status_file_data()["agents"]
    frozen_row = next(r for r in stats if r["kind"] == "frozen")
    assert frozen_row["placement_ema"] == 5.0
    assert frozen_row["games"] == 1


def test_league_ema_from_game_records():
    from src.training.selfplay.league_state import LeagueController

    league = LeagueController(ema_beta=1.0)
    league.add_frozen_bytes(b"x", episode=0)
    slot_id = 0
    league.submit(
        GameRecord(
            (
                ParticipantOutcome(SLOT_CURRENT, 1),
                ParticipantOutcome(slot_id, 2),
            )
        )
    )
    assert league.get_slot(slot_id).ema_win_rate == 0.0
    league.submit(
        GameRecord(
            (
                ParticipantOutcome(SLOT_CURRENT, 8),
                ParticipantOutcome(slot_id, 1),
            )
        )
    )
    assert league.get_slot(slot_id).ema_win_rate == 1.0
    league.submit(
        GameRecord(
            (
                ParticipantOutcome(SLOT_CURRENT, 4),
                ParticipantOutcome(slot_id, 5),
            )
        )
    )
    assert league.get_slot(slot_id).ema_win_rate == 0.0

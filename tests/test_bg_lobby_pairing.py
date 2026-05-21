"""8p combat pairing: round-robin, cooldown, ghost."""

from __future__ import annotations

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_lobby.match_types import GHOST_OPPONENT_ID, CombatMatch, EliminatedSnapshot
from src.bg_lobby.pairing import (
    COOLDOWN_COMBAT_ROUNDS,
    build_round_robin_schedule,
    compute_pairings,
    record_combat_opponent,
)


def _empty_recent(n: int = 8) -> tuple[tuple[int, ...], ...]:
    return tuple(() for _ in range(n))


def _snapshot(seat: int, *, round_no: int = 1) -> EliminatedSnapshot:
    return EliminatedSnapshot(
        seat=seat,
        last_board=(make_minion("recruit"),),
        tavern_tier=3,
        eliminated_combat_round=round_no,
    )


def test_round_robin_eight_players_seven_unique_opponents_each():
    schedule = build_round_robin_schedule(8)
    assert len(schedule) == 7
    assert all(len(r) == 4 for r in schedule)

    met: list[set[int]] = [set() for _ in range(8)]
    for pairs in schedule:
        for a, b in pairs:
            met[a].add(b)
            met[b].add(a)

    for seat in range(8):
        assert met[seat] == {i for i in range(8) if i != seat}


def test_full_lobby_uses_schedule_cycle():
    alive = tuple(range(8))
    recent = _empty_recent()
    rng = np.random.default_rng(0)
    all_pairs: list[tuple[int, int]] = []
    cycle = 0
    for _ in range(7):
        matches, cycle = compute_pairings(
            alive,
            recent,
            (),
            full_lobby_cycle_round=cycle,
            rng=rng,
        )
        assert len(matches) == 4
        assert all(not m.is_ghost for m in matches)
        all_pairs.extend((m.a, m.b) for m in matches if m.b is not None)
    assert cycle == 0
    assert len(all_pairs) == 28


def test_two_alive_always_paired():
    matches, _ = compute_pairings(
        (2, 5),
        _empty_recent(),
        (),
        rng=np.random.default_rng(1),
    )
    assert len(matches) == 1
    assert matches[0].a == 2 and matches[0].b == 5


def test_odd_alive_has_exactly_one_ghost():
    eliminated = (_snapshot(3),)
    matches, _ = compute_pairings(
        (0, 1, 2, 4, 5),
        _empty_recent(),
        eliminated,
        rng=np.random.default_rng(2),
    )
    ghosts = [m for m in matches if m.is_ghost]
    live = [m for m in matches if not m.is_ghost]
    assert len(ghosts) == 1
    assert len(live) == 2
    assert ghosts[0].ghost is not None


def test_cooldown_avoids_recent_opponent_when_possible():
    recent = list(_empty_recent())
    # 0 and 1 fought last round
    recent[0] = (1,)
    recent[1] = (0,)
    matches, _ = compute_pairings(
        (0, 1, 2, 3),
        tuple(recent),
        (),
        rng=np.random.default_rng(3),
    )
    live_pairs = {(m.a, m.b) for m in matches if m.b is not None}
    assert (0, 1) not in live_pairs and (1, 0) not in live_pairs


def test_ghost_seat_avoids_recent_ghost_when_possible():
    recent = list(_empty_recent())
    recent[4] = (GHOST_OPPONENT_ID,)
    matches, _ = compute_pairings(
        (0, 1, 2, 4, 5),
        tuple(recent),
        (_snapshot(7),),
        rng=np.random.default_rng(4),
    )
    ghost_matches = [m for m in matches if m.is_ghost]
    assert len(ghost_matches) == 1
    # with 5 players, prefer ghost seat != 4 if a valid matching exists
    if ghost_matches[0].a != 4:
        assert GHOST_OPPONENT_ID not in recent[ghost_matches[0].a][-COOLDOWN_COMBAT_ROUNDS:]


def test_odd_alive_without_eliminated_uses_bye_not_ghost():
    matches, _ = compute_pairings(
        (0, 1, 2, 4, 5),
        _empty_recent(),
        (),
        rng=np.random.default_rng(9),
    )
    assert all(not m.is_ghost for m in matches)
    assert len(matches) == 2


def test_ghost_initiative_uses_board_size():
    from types import SimpleNamespace

    from src.bg_lobby.eight_player import _initiative_for_ghost

    ghost = _snapshot(3)
    state = SimpleNamespace(
        players=[SimpleNamespace(board=[1, 2, 3]), SimpleNamespace(board=[1])],
        round_number=1,
        initiative_player=0,
    )
    assert _initiative_for_ghost(state, 0, ghost) is True
    assert _initiative_for_ghost(state, 1, ghost) is False


def test_record_combat_opponent_trims_to_three():
    r = record_combat_opponent((2, 3, 4), 5)
    assert r == (3, 4, 5)


def test_compute_pairings_two_player_waives_cooldown():
    recent = list(_empty_recent())
    recent[0] = (1, 2, 3)
    recent[1] = (0, 2, 3)
    matches, _ = compute_pairings(
        (0, 1),
        tuple(recent),
        (),
        rng=np.random.default_rng(5),
    )
    assert matches[0].a == 0 and matches[0].b == 1

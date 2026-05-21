"""Placement reward table for 8-player lobby."""

from __future__ import annotations

import pytest

from src.bg_lobby.match_types import EliminatedSnapshot
from src.bg_catalog.cards import make_minion
from src.envs.bglike.placement import (
    placement_for_seat,
    placement_reward,
    placement_reward_for_seat,
)
from src.envs.bglike.state import BGLikeState
from src.bg_lobby.player import PlayerState, PlayerPhase


def _player() -> PlayerState:
    return PlayerState(
        health=0,
        hero_damage_taken_total=0,
        gold=0,
        tavern_tier=1,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * 6,
        hand=[None] * 10,
        phase=PlayerPhase.DONE,
        shop_actions_used=0,
    )


def _state(*, winner: int | None, eliminated_seats: tuple[int, ...]) -> BGLikeState:
    snaps = tuple(
        EliminatedSnapshot(
            seat=s,
            last_board=(make_minion("recruit"),),
            tavern_tier=1,
            eliminated_combat_round=i + 1,
        )
        for i, s in enumerate(eliminated_seats)
    )
    players = tuple(_player() for _ in range(8))
    return BGLikeState(
        players=players,
        alive=() if winner is None else (winner,),
        round_number=10,
        combat_round=7,
        full_lobby_cycle_round=0,
        current_player_index=0,
        shop_turn_order=(0,),
        recent_opponents=tuple(() for _ in range(8)),
        eliminated=snaps,
        pairings=(),
        initiative_player=0,
        winner=winner,
        done=True,
    )


@pytest.mark.parametrize(
    "place, expected",
    [
        (1, 1.0),
        (2, 5 / 7),
        (3, 3 / 7),
        (4, 1 / 7),
        (5, -1 / 7),
        (6, -3 / 7),
        (7, -5 / 7),
        (8, -1.0),
    ],
)
def test_placement_reward_normalized(place: int, expected: float) -> None:
    assert placement_reward(place) == pytest.approx(expected)


def test_placement_for_winner_and_first_out() -> None:
    elim = tuple(range(7, 0, -1))
    s = _state(winner=0, eliminated_seats=elim)
    assert placement_for_seat(s, 0) == 1
    assert placement_for_seat(s, 7) == 8
    assert placement_for_seat(s, 3) == 4
    assert placement_reward_for_seat(s, 3) == pytest.approx(1 / 7)

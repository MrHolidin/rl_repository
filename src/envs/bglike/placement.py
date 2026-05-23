"""Lobby placement rank and normalized placement reward."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import BGLikeState

PLACEMENT_REWARD_SCALE = 7.0


def placement_for_seat(state: BGLikeState, seat: int) -> int:
    """Final standing 1 (winner) .. 8 (first eliminated)."""
    if state.winner == seat:
        return 1
    for i, snap in enumerate(state.eliminated):
        if snap.seat == seat:
            return 8 - i
    raise ValueError(f"seat {seat} has no final placement (lobby not finished for seat)")


def placement_reward(place: int) -> float:
    """Normalized reward in [-1, 1]: ``(9 - 2*place) / 7``."""
    if not 1 <= place <= 8:
        raise ValueError(f"place must be 1..8, got {place}")
    return (9 - 2 * place) / PLACEMENT_REWARD_SCALE


def placement_score(place: int) -> float:
    """League outcome in [0, 1]: 1st -> 1.0, 8th -> 0.0 (linear in placement)."""
    return (placement_reward(place) + 1.0) / 2.0


def pairwise_learner_score(learner_place: int, opponent_place: int) -> float:
    """Learner performance vs one opponent: 1.0 win, 0.0 loss, 0.5 tie (lower place is better)."""
    if learner_place < opponent_place:
        return 1.0
    if learner_place > opponent_place:
        return 0.0
    return 0.5


def placement_reward_to_score(reward: float) -> float:
    """Map training placement reward [-1, 1] to league score [0, 1]."""
    return max(0.0, min(1.0, (float(reward) + 1.0) / 2.0))


def placement_reward_for_seat(state: BGLikeState, seat: int) -> float:
    return placement_reward(placement_for_seat(state, seat))


def is_seat_eliminated(state: BGLikeState, seat: int) -> bool:
    return seat not in state.alive or state.players[seat].health <= 0


def is_seat_finished(state: BGLikeState, seat: int) -> bool:
    if state.winner == seat:
        return True
    return any(snap.seat == seat for snap in state.eliminated)


__all__ = [
    "PLACEMENT_REWARD_SCALE",
    "is_seat_eliminated",
    "is_seat_finished",
    "placement_for_seat",
    "placement_reward",
    "placement_reward_for_seat",
    "placement_reward_to_score",
    "placement_score",
    "pairwise_learner_score",
]

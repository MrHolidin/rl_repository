"""Board ordering primitives for MiniBG (adjacent swaps during shop)."""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

from .state import MiniBGState, PlayerPhase

S = TypeVar("S", bound=MiniBGState)


def reorder_board(
    state: S,
    player_idx: int,
    perm: Sequence[int],
    *,
    board_size: int,
    copy_state: Callable[[S], S],
) -> S:
    if state.done:
        raise ValueError("Cannot reorder in terminal state")
    if len(perm) != board_size:
        raise ValueError(f"perm must have length {board_size}, got {len(perm)}")
    if sorted(perm) != list(range(board_size)):
        raise ValueError(
            f"perm must be a permutation of 0..{board_size - 1}: got {tuple(perm)}"
        )

    new_state = copy_state(state)
    player = new_state.players[player_idx]
    k = len(player.board)
    player.board = [player.board[p] for p in perm if p < k]
    return new_state


def swap_board_adjacent(
    state: S,
    player_idx: int,
    i: int,
    *,
    copy_state: Callable[[S], S],
) -> S:
    if state.done:
        raise ValueError("Cannot swap board in terminal state")
    new_state = copy_state(state)
    player = new_state.players[player_idx]
    if player.phase != PlayerPhase.SHOP:
        raise ValueError("swap_board_adjacent only valid in SHOP phase")
    b = player.board
    if not (0 <= i < len(b) - 1):
        raise ValueError(f"swap index {i} invalid for board length {len(b)}")
    b[i], b[i + 1] = b[i + 1], b[i]
    return new_state

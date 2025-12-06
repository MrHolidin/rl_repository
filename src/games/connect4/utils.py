"""Shared utilities for Connect4 game logic."""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Sequence

import numpy as np

from .connect4_state import Connect4State
from src.games.turn_based_game import TurnBasedGame

# Default board dimensions
CONNECT4_ROWS = 6
CONNECT4_COLS = 7


def build_state_dict(
    state: Connect4State,
    game: TurnBasedGame[Connect4State],
    legal_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Build the state dictionary expected by ObservationBuilder.

    Args:
        state: Current game state.
        game: Game rules instance.
        legal_mask: Optional precomputed legal actions mask.

    Returns:
        Dictionary with board, current_player_token, last_move, legal_actions_mask.
    """
    current_token = game.current_player(state)

    if legal_mask is None:
        legal_actions = game.legal_actions(state)
        legal_mask = np.zeros(state.board.shape[1], dtype=bool)
        for a in legal_actions:
            if 0 <= a < len(legal_mask):
                legal_mask[a] = True

    return {
        "board": state.board,
        "current_player_token": current_token,
        "last_move": state.last_move,
        "legal_actions_mask": legal_mask,
    }


def check_n_in_row(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    n: int,
    rows: int,
    cols: int,
) -> bool:
    """
    Check if there are at least n pieces in a row for the given player
    passing through (row, col).

    Args:
        board: Game board array.
        row: Row position to check from.
        col: Column position to check from.
        player: Player token (1 or -1).
        n: Number of pieces in a row to check for.
        rows: Total number of rows.
        cols: Total number of columns.

    Returns:
        True if player has at least n in a row through (row, col).
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        count = 1
        # Check positive direction
        for i in range(1, n):
            r, c = row + dr * i, col + dc * i
            if 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                count += 1
            else:
                break
        # Check negative direction
        for i in range(1, n):
            r, c = row - dr * i, col - dc * i
            if 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                count += 1
            else:
                break

        if count >= n:
            return True

    return False


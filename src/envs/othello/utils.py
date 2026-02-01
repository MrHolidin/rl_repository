"""Shared utilities for Othello game logic."""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from .state import OthelloState
from src.games.turn_based_game import TurnBasedGame

OTHELLO_SIZE = 8


def build_state_dict(
    state: OthelloState,
    game: TurnBasedGame[OthelloState],
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
        legal_mask = np.zeros(state.board.shape[0] * state.board.shape[1], dtype=bool)
        for a in legal_actions:
            if 0 <= a < len(legal_mask):
                legal_mask[a] = True

    return {
        "board": state.board,
        "current_player_token": current_token,
        "last_move": state.last_move,
        "legal_actions_mask": legal_mask,
    }


def get_flips(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    size: int = OTHELLO_SIZE,
) -> List[Tuple[int, int]]:
    """
    Get all pieces that would be flipped by placing a piece at (row, col).

    Args:
        board: Game board array.
        row: Row position.
        col: Column position.
        player: Player token (1 or -1).
        size: Board size.

    Returns:
        List of (row, col) positions that would be flipped.
    """
    if board[row, col] != 0:
        return []

    opponent = -player
    flips = []
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for dr, dc in directions:
        temp_flips = []
        r, c = row + dr, col + dc

        while 0 <= r < size and 0 <= c < size and board[r, c] == opponent:
            temp_flips.append((r, c))
            r += dr
            c += dc

        if 0 <= r < size and 0 <= c < size and board[r, c] == player and temp_flips:
            flips.extend(temp_flips)

    return flips


def is_valid_move(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    size: int = OTHELLO_SIZE,
) -> bool:
    """
    Check if placing a piece at (row, col) is a valid move.

    Args:
        board: Game board array.
        row: Row position.
        col: Column position.
        player: Player token (1 or -1).
        size: Board size.

    Returns:
        True if the move is valid.
    """
    return len(get_flips(board, row, col, player, size)) > 0


def count_pieces(board: np.ndarray) -> Tuple[int, int]:
    """
    Count pieces for each player.

    Args:
        board: Game board array.

    Returns:
        Tuple of (player1_count, player2_count) where player1 is token 1, player2 is token -1.
    """
    player1_count = np.sum(board == 1)
    player2_count = np.sum(board == -1)
    return int(player1_count), int(player2_count)

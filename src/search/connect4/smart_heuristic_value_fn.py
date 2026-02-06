"""Smart heuristic value function for Connect4 minimax."""

from __future__ import annotations

import math

import numpy as np

from src.envs.connect4 import Connect4State, connect4_terminal_evaluator
from src.games.turn_based_game import TurnBasedGame
from ..value_fn import StateValueFn


def _score_window_fast(w0: int, w1: int, w2: int, w3: int, player: int) -> float:
    """Score 4 cells. Pure Python, no numpy."""
    opp = -player
    player_count = (w0 == player) + (w1 == player) + (w2 == player) + (w3 == player)
    opp_count = (w0 == opp) + (w1 == opp) + (w2 == opp) + (w3 == opp)

    if player_count > 0 and opp_count > 0:
        return 0.0

    empty_count = 4 - player_count - opp_count

    if player_count == 4:
        return 1000.0
    if player_count == 3 and empty_count == 1:
        return 50.0
    if player_count == 2 and empty_count == 2:
        return 5.0
    if player_count == 1 and empty_count == 3:
        return 1.0

    if opp_count == 4:
        return -1000.0
    if opp_count == 3 and empty_count == 1:
        return -50.0
    if opp_count == 2 and empty_count == 2:
        return -5.0
    if opp_count == 1 and empty_count == 3:
        return -1.0

    return 0.0


def evaluate_board_fast(board: np.ndarray, player: int) -> float:
    """Evaluate board position. Optimized pure Python loops."""
    rows, cols = board.shape
    score = 0.0
    opp = -player

    # Center column bonus
    center_col = cols // 2
    for r in range(rows):
        v = int(board[r, center_col])
        if v == player:
            score += 3.0
        elif v == opp:
            score -= 3.0

    # Horizontal windows
    for r in range(rows):
        for c in range(cols - 3):
            score += _score_window_fast(
                int(board[r, c]), int(board[r, c + 1]),
                int(board[r, c + 2]), int(board[r, c + 3]), player
            )

    # Vertical windows
    for r in range(rows - 3):
        for c in range(cols):
            score += _score_window_fast(
                int(board[r, c]), int(board[r + 1, c]),
                int(board[r + 2, c]), int(board[r + 3, c]), player
            )

    # Positive diagonal (bottom-left to top-right)
    for r in range(3, rows):
        for c in range(cols - 3):
            score += _score_window_fast(
                int(board[r, c]), int(board[r - 1, c + 1]),
                int(board[r - 2, c + 2]), int(board[r - 3, c + 3]), player
            )

    # Negative diagonal (top-left to bottom-right)
    for r in range(rows - 3):
        for c in range(cols - 3):
            score += _score_window_fast(
                int(board[r, c]), int(board[r + 1, c + 1]),
                int(board[r + 2, c + 2]), int(board[r + 3, c + 3]), player
            )

    return score


class Connect4SmartHeuristicValueFn(StateValueFn[Connect4State]):
    """Evaluates Connect4 positions using pattern-based heuristics."""

    def __init__(self, normalize: bool = True) -> None:
        self._normalize = normalize

    def evaluate(
        self,
        game: TurnBasedGame[Connect4State],
        state: Connect4State,
    ) -> float:
        current_token = game.current_player(state)

        if game.is_terminal(state):
            return connect4_terminal_evaluator(game, state, root_player=current_token)

        raw_score = evaluate_board_fast(state.board, current_token)

        if self._normalize:
            return math.tanh(raw_score / 100.0)

        return raw_score

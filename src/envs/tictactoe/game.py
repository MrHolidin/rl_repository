"""TicTacToe game rules."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.games.turn_based_game import TurnBasedGame, Action
from .state import TicTacToeState

PLAYER_TOKENS = [1, -1]

_WIN_LINES = [
    # rows
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    # cols
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    # diagonals
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)],
]


class TicTacToeGame(TurnBasedGame[TicTacToeState]):

    def initial_state(self) -> TicTacToeState:
        return TicTacToeState(
            board=np.zeros((3, 3), dtype=np.int8),
            current_player_index=0,
            winner=None,
            done=False,
        )

    def legal_actions(self, state: TicTacToeState) -> Sequence[Action]:
        return [int(r * 3 + c) for r in range(3) for c in range(3) if state.board[r, c] == 0]

    def apply_action(self, state: TicTacToeState, action: Action) -> TicTacToeState:
        if state.done:
            raise ValueError("Cannot apply action in terminal state")

        r, c = divmod(action, 3)
        if state.board[r, c] != 0:
            raise ValueError(f"Cell ({r},{c}) is already occupied")

        board = state.board.copy()
        token = PLAYER_TOKENS[state.current_player_index]
        board[r, c] = token

        winner = self._check_winner(board, r, c, token)
        done = winner is not None or not np.any(board == 0)
        if done and winner is None:
            winner = 0  # draw

        return TicTacToeState(
            board=board,
            current_player_index=1 - state.current_player_index,
            winner=winner,
            done=done,
        )

    def current_player(self, state: TicTacToeState) -> int:
        return PLAYER_TOKENS[state.current_player_index]

    def is_terminal(self, state: TicTacToeState) -> bool:
        return state.done

    def winner(self, state: TicTacToeState) -> Optional[int]:
        return state.winner

    def _check_winner(self, board: np.ndarray, r: int, c: int, token: int) -> Optional[int]:
        for line in _WIN_LINES:
            if all(board[lr, lc] == token for lr, lc in line):
                return token
        return None

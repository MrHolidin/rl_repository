"""Connect4 game rules (immutable state, for search algorithms)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .state import Connect4State
from .utils import CONNECT4_COLS, CONNECT4_ROWS, check_n_in_row
from src.games.turn_based_game import TurnBasedGame, Action


class Connect4Game(TurnBasedGame[Connect4State]):
    """
    Чистые правила Connect4 без окружения: только переходы по состояниям.
    """

    def __init__(
        self, rows: int = CONNECT4_ROWS, cols: int = CONNECT4_COLS
    ) -> None:
        self.rows = rows
        self.cols = cols
        self._player_tokens = np.array([1, -1], dtype=np.int8)

    def initial_state(self) -> Connect4State:
        board = np.zeros((self.rows, self.cols), dtype=np.int8)
        return Connect4State(
            board=board,
            current_player_index=0,
            winner=None,
            done=False,
        )

    def legal_actions(self, state: Connect4State) -> Sequence[Action]:
        top_row = state.board[0]
        return [col for col in range(self.cols) if top_row[col] == 0]

    def apply_action(self, state: Connect4State, action: Action) -> Connect4State:
        if state.done:
            raise ValueError("Cannot apply action in terminal state")

        if action < 0 or action >= self.cols:
            raise ValueError(f"Illegal action: {action}")

        if action not in self.legal_actions(state):
            raise ValueError(f"Column {action} is full")

        board = state.board.copy()
        current_token = int(self._player_tokens[state.current_player_index])
        row = self._drop_piece(board, action, current_token)
        assert row is not None

        winner: Optional[int] = None
        done = False

        if self._check_win(board, row, action, current_token):
            winner = current_token
            done = True
        elif self._is_board_full(board):
            winner = 0
            done = True

        next_player_index = 1 - state.current_player_index
        return Connect4State(
            board=board,
            current_player_index=next_player_index,
            winner=winner,
            done=done,
            last_move=(row, action),
        )

    def current_player(self, state: Connect4State) -> int:
        return int(self._player_tokens[state.current_player_index])

    def is_terminal(self, state: Connect4State) -> bool:
        return state.done

    def winner(self, state: Connect4State) -> Optional[int]:
        return state.winner

    def _drop_piece(self, board: np.ndarray, col: int, player: int) -> Optional[int]:
        for row in range(self.rows - 1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return row
        return None

    def _check_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        return check_n_in_row(board, row, col, player, n=4, rows=self.rows, cols=self.cols)

    def _is_board_full(self, board: np.ndarray) -> bool:
        return np.all(board != 0)

"""Othello game rules (immutable state, for search algorithms)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .state import OthelloState
from .utils import OTHELLO_SIZE, get_flips, is_valid_move, count_pieces
from src.games.turn_based_game import TurnBasedGame, Action


class OthelloGame(TurnBasedGame[OthelloState]):
    """
    Pure Othello rules without environment: only state transitions.
    """

    def __init__(self, size: int = OTHELLO_SIZE) -> None:
        self.size = size
        self._player_tokens = np.array([1, -1], dtype=np.int8)

    def initial_state(self) -> OthelloState:
        board = np.zeros((self.size, self.size), dtype=np.int8)
        
        mid = self.size // 2
        board[mid - 1, mid - 1] = -1
        board[mid - 1, mid] = 1
        board[mid, mid - 1] = 1
        board[mid, mid] = -1

        return OthelloState(
            board=board,
            current_player_index=0,
            winner=None,
            done=False,
        )

    def legal_actions(self, state: OthelloState) -> Sequence[Action]:
        if state.done:
            return []

        current_token = int(self._player_tokens[state.current_player_index])
        legal = []

        for row in range(self.size):
            for col in range(self.size):
                if is_valid_move(state.board, row, col, current_token, self.size):
                    legal.append(row * self.size + col)

        return legal

    def apply_action(self, state: OthelloState, action: Action) -> OthelloState:
        if state.done:
            raise ValueError("Cannot apply action in terminal state")

        if action < 0 or action >= self.size * self.size:
            raise ValueError(f"Illegal action: {action}")

        row = action // self.size
        col = action % self.size

        current_token = int(self._player_tokens[state.current_player_index])
        flips = get_flips(state.board, row, col, current_token, self.size)

        if not flips:
            raise ValueError(f"Invalid move at ({row}, {col})")

        board = state.board.copy()
        board[row, col] = current_token
        for flip_row, flip_col in flips:
            board[flip_row, flip_col] = current_token

        next_player_index = 1 - state.current_player_index
        next_token = int(self._player_tokens[next_player_index])

        has_next_moves = any(
            is_valid_move(board, r, c, next_token, self.size)
            for r in range(self.size)
            for c in range(self.size)
        )

        if not has_next_moves:
            has_current_moves = any(
                is_valid_move(board, r, c, current_token, self.size)
                for r in range(self.size)
                for c in range(self.size)
            )

            if not has_current_moves:
                p1_count, p2_count = count_pieces(board)
                if p1_count > p2_count:
                    winner = 1
                elif p2_count > p1_count:
                    winner = -1
                else:
                    winner = 0

                return OthelloState(
                    board=board,
                    current_player_index=next_player_index,
                    winner=winner,
                    done=True,
                    last_move=(row, col),
                )
            else:
                next_player_index = state.current_player_index

        return OthelloState(
            board=board,
            current_player_index=next_player_index,
            winner=None,
            done=False,
            last_move=(row, col),
        )

    def current_player(self, state: OthelloState) -> int:
        return int(self._player_tokens[state.current_player_index])

    def is_terminal(self, state: OthelloState) -> bool:
        return state.done

    def winner(self, state: OthelloState) -> Optional[int]:
        return state.winner

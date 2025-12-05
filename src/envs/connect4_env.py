"""Connect Four environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from ..features.observation_builder import BoardChannels, ObservationBuilder
from .base import StepResult, TurnBasedEnv
from .reward_config import RewardConfig


@dataclass
class Connect4State:
    """
    Complete Connect Four game state kept outside the env itself.

    Keeping the board/turn/winner together makes it easier to explicitly copy
    the state or share it with planning modules instead of scattering the
    attributes on the env directly.
    """

    board: np.ndarray
    current_player_index: int
    winner: Optional[int]
    done: bool


class Connect4Env(TurnBasedEnv):
    """
    Connect Four environment (6 rows × 7 columns).

    Players drop pieces into columns. Win condition: 4 in a row
    (horizontal, vertical, or diagonal).
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        reward_config: Optional[RewardConfig] = None,
        observation_builder: Optional[ObservationBuilder] = None,
    ):
        self.rows = rows
        self.cols = cols

        self._state: Optional[Connect4State] = None
        self._last_move: Optional[Tuple[int, int]] = None
        self._player_tokens = np.array([1, -1], dtype=np.int8)
        self._rng = np.random.default_rng()

        if observation_builder is None:
            observation_builder = BoardChannels(board_shape=(rows, cols))
        self._observation_builder = observation_builder

        if reward_config is None:
            reward_config = RewardConfig()
        self.reward_config = reward_config

        self.reset()

    # ------------------------------------------------------------------
    # TurnBasedEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self._state = Connect4State(
            board=board,
            current_player_index=0,
            winner=None,
            done=False,
        )
        self._last_move = None
        return self._get_obs()

    def step(self, action: int) -> StepResult:
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        legal_mask = self.legal_actions_mask
        if action < 0 or action >= self.cols or not legal_mask[action]:
            info = {
                "winner": None,
                "termination_reason": "illegal",
                "invalid_action": True,
                "three_in_row": False,
                "opponent_three_in_row": False,
            }
            return StepResult(
                obs=self._get_obs(),
                reward=self.reward_config.invalid_action,
                terminated=False,
                truncated=False,
                info=info,
            )

        current_token = self.current_player_token
        row = self._drop_piece(action, current_token)
        if row is None:
            info = {
                "winner": None,
                "termination_reason": "illegal",
                "invalid_action": True,
                "three_in_row": False,
                "opponent_three_in_row": False,
            }
            return StepResult(
                obs=self._get_obs(),
                reward=self.reward_config.invalid_action,
                terminated=False,
                truncated=False,
                info=info,
            )

        assert self._state is not None
        self._last_move = (row, action)

        won = self._check_win(row, action, current_token)

        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            "winner": None,
            "termination_reason": None,
            "invalid_action": False,
            "three_in_row": False,
            "opponent_three_in_row": False,
        }

        if won:
            self._state.winner = current_token
            self._state.done = True
            terminated = True
            reward = self.reward_config.win
            info["winner"] = self._state.winner
            info["termination_reason"] = "win"
        elif self._is_board_full():
            self._state.winner = 0
            self._state.done = True
            terminated = True
            reward = self.reward_config.draw
            info["winner"] = 0
            info["termination_reason"] = "draw"
        else:
            if self.reward_config.three_in_row != 0.0:
                if self._has_any_three(current_token):
                    reward += self.reward_config.three_in_row
                    info["three_in_row"] = True
                else:
                    info["three_in_row"] = False

            if self.reward_config.opponent_three_in_row != 0.0:
                if self._has_any_three(-current_token):
                    reward -= self.reward_config.opponent_three_in_row
                    info["opponent_three_in_row"] = True
                else:
                    info["opponent_three_in_row"] = False

        assert self._state is not None
        self._state.current_player_index = 1 - self._state.current_player_index

        return StepResult(
            obs=self._get_obs(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    # ------------------------------------------------------------------
    # Internal game logic (operates on self._state.board)
    # ------------------------------------------------------------------

    def _drop_piece(self, col: int, player: int) -> Optional[int]:
        board = self.board
        for row in range(self.rows - 1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return row
        return None

    def _check_win(self, row: int, col: int, player: int) -> bool:
        return self._check_n_in_row(row, col, player, n=4)

    def _check_three_in_row(self, row: int, col: int, player: int) -> bool:
        has_three = self._check_n_in_row(row, col, player, n=3)
        has_four = self._check_n_in_row(row, col, player, n=4)
        return has_three and not has_four

    def _has_any_three(self, player: int) -> bool:
        board = self.board
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r, c] == player:
                    if self._check_three_in_row(r, c, player):
                        return True
        return False

    def _check_n_in_row(self, row: int, col: int, player: int, n: int) -> bool:
        board = self.board
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            for i in range(1, n):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.rows and 0 <= c < self.cols and board[r, c] == player:
                    count += 1
                else:
                    break
            for i in range(1, n):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.rows and 0 <= c < self.cols and board[r, c] == player:
                    count += 1
                else:
                    break

            if count >= n:
                return True

        return False

    def _is_board_full(self) -> bool:
        return np.all(self.board != 0)

    # ------------------------------------------------------------------
    # Legal actions & state API
    # ------------------------------------------------------------------

    def get_legal_actions(self) -> List[int]:
        board = self.board
        return [col for col in range(self.cols) if board[0, col] == 0]

    @property
    def legal_actions_mask(self) -> np.ndarray:
        board = self.board
        return (board[0] == 0).astype(bool)

    def current_player(self) -> int:
        assert self._state is not None
        return self._state.current_player_index

    @property
    def current_player_token(self) -> int:
        assert self._state is not None
        return int(self._player_tokens[self._state.current_player_index])

    # ------------------------------------------------------------------
    # State/board helpers
    # ------------------------------------------------------------------

    @property
    def board(self) -> np.ndarray:
        assert self._state is not None
        return self._state.board

    @property
    def winner(self) -> Optional[int]:
        assert self._state is not None
        return self._state.winner

    @property
    def done(self) -> bool:
        assert self._state is not None
        return self._state.done

    @property
    def observation_builder(self) -> ObservationBuilder:
        return self._observation_builder

    def _get_obs(self) -> np.ndarray:
        assert self._state is not None
        state = {
            "board": self.board,
            "current_player_token": self.current_player_token,
            "last_move": self._last_move,
            "legal_actions_mask": self.legal_actions_mask,
        }
        return self._observation_builder.build(state)

    # ------------------------------------------------------------------
    # Utilities & debugging
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[str]:
        if mode == "human":
            print("\n" + "=" * (self.cols * 2 + 1))
            print("  " + " ".join(str(i) for i in range(self.cols)))
            print("=" * (self.cols * 2 + 1))

            for row in range(self.rows):
                row_str = "|"
                for col in range(self.cols):
                    if self.board[row, col] == 1:
                        row_str += "X|"
                    elif self.board[row, col] == -1:
                        row_str += "O|"
                    else:
                        row_str += " |"
                print(row_str)

            print("=" * (self.cols * 2 + 1))

            if self.done:
                if self.winner == 1:
                    print("Player X wins!")
                elif self.winner == -1:
                    print("Player O wins!")
                else:
                    print("Draw!")
            else:
                player_symbol = "X" if self.current_player_token == 1 else "O"
                print(f"Current player: {player_symbol}")

            print()

        return None

    def get_state_hash(self) -> str:
        return ",".join(str(x) for x in self.board.flatten())

    def get_state(self) -> Connect4State:
        assert self._state is not None
        return Connect4State(
            board=self.board.copy(),
            current_player_index=self._state.current_player_index,
            winner=self._state.winner,
            done=self._state.done,
        )

    def set_state(self, state: Connect4State) -> None:
        self._state = Connect4State(
            board=state.board.copy(),
            current_player_index=state.current_player_index,
            winner=state.winner,
            done=state.done,
        )
        self._last_move = None

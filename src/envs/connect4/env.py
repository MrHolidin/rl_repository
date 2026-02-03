"""Connect Four environment implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from ...features.observation_builder import BoardChannels, ObservationBuilder
from ..base import StepResult, TurnBasedEnv
from ..reward_config import RewardConfig
from .state import Connect4State
from .game import Connect4Game
from .utils import CONNECT4_COLS, CONNECT4_ROWS, build_state_dict, check_n_in_row


class Connect4Env(TurnBasedEnv):
    """
    Connect Four environment (6 rows × 7 columns).

    Players drop pieces into columns. Win condition: 4 in a row
    (horizontal, vertical, or diagonal).
    """

    def __init__(
        self,
        rows: int = CONNECT4_ROWS,
        cols: int = CONNECT4_COLS,
        reward_config: Optional[RewardConfig] = None,
        observation_builder: Optional[ObservationBuilder] = None,
    ):
        self.rows = rows
        self.cols = cols

        self._game = Connect4Game(rows=rows, cols=cols)
        self._state: Optional[Connect4State] = None
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

        self._state = self._game.initial_state()
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
        assert self._state is not None
        new_state = self._game.apply_action(self._state, action)
        self._state = new_state

        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            "winner": self._state.winner,
            "termination_reason": None,
            "invalid_action": False,
            "three_in_row": False,
            "opponent_three_in_row": False,
        }

        if self._state.done:
            terminated = True
            if self._state.winner == current_token:
                reward = self.reward_config.win
                info["termination_reason"] = "win"
            elif self._state.winner == 0:
                reward = self.reward_config.draw
                info["termination_reason"] = "draw"
            else:
                reward = self.reward_config.loss
                info["termination_reason"] = "loss"
        else:
            three = self.reward_config.three_in_row
            opp_three = self.reward_config.opponent_three_in_row
            if three != 0.0 or opp_three != 0.0:
                if three != 0.0 and self._has_any_three(current_token):
                    reward += three
                    info["three_in_row"] = True
                if opp_three != 0.0 and self._has_any_three(-current_token):
                    reward -= opp_three
                    info["opponent_three_in_row"] = True

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

    def _check_three_in_row(self, row: int, col: int, player: int) -> bool:
        board = self.board
        has_three = check_n_in_row(board, row, col, player, n=3, rows=self.rows, cols=self.cols)
        has_four = check_n_in_row(board, row, col, player, n=4, rows=self.rows, cols=self.cols)
        return has_three and not has_four

    def _has_any_three(self, player: int) -> bool:
        board = self.board
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r, c] == player:
                    if self._check_three_in_row(r, c, player):
                        return True
        return False

    # ------------------------------------------------------------------
    # Legal actions & state API
    # ------------------------------------------------------------------

    def get_legal_actions(self) -> List[int]:
        assert self._state is not None
        return list(self._game.legal_actions(self._state))

    @property
    def legal_actions_mask(self) -> np.ndarray:
        assert self._state is not None
        return self._state.board[0] == 0

    def current_player(self) -> int:
        assert self._state is not None
        return self._state.current_player_index

    @property
    def current_player_token(self) -> int:
        assert self._state is not None
        return self._game.current_player(self._state)

    # ------------------------------------------------------------------
    # State/board helpers
    # ------------------------------------------------------------------

    @property
    def board(self) -> np.ndarray:
        assert self._state is not None
        return self._state.board

    @property
    def last_move(self) -> Optional[Tuple[int, int]]:
        assert self._state is not None
        return self._state.last_move

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
        state_dict = build_state_dict(
            self._state, self._game, legal_mask=self.legal_actions_mask
        )
        return self._observation_builder.build(state_dict)

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
            last_move=self._state.last_move,
        )

    def set_state(self, state: Connect4State) -> None:
        self._state = Connect4State(
            board=state.board.copy(),
            current_player_index=state.current_player_index,
            winner=state.winner,
            done=state.done,
            last_move=state.last_move,
        )

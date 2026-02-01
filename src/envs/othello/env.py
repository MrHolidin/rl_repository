"""Othello environment implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from ...features.observation_builder import BoardChannels, ObservationBuilder
from ..base import StepResult, TurnBasedEnv
from ..reward_config import RewardConfig
from .state import OthelloState
from .game import OthelloGame
from .utils import OTHELLO_SIZE, build_state_dict


class OthelloEnv(TurnBasedEnv):
    """
    Othello (Reversi) environment (8x8 board).

    Players place pieces on the board, flipping opponent pieces between
    the placed piece and existing pieces. Win condition: most pieces when
    no more moves are available.
    """

    def __init__(
        self,
        size: int = OTHELLO_SIZE,
        reward_config: Optional[RewardConfig] = None,
        observation_builder: Optional[ObservationBuilder] = None,
    ):
        self.size = size

        self._game = OthelloGame(size=size)
        self._state: Optional[OthelloState] = None

        if observation_builder is None:
            observation_builder = BoardChannels(board_shape=(size, size))
        self._observation_builder = observation_builder

        if reward_config is None:
            reward_config = RewardConfig()
        self.reward_config = reward_config

        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._state = self._game.initial_state()
        self._last_move = None
        return self._get_obs()

    def step(self, action: int) -> StepResult:
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        legal_mask = self.legal_actions_mask
        if action < 0 or action >= self.size * self.size or not legal_mask[action]:
            info = {
                "winner": None,
                "termination_reason": "illegal",
                "invalid_action": True,
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

        return StepResult(
            obs=self._get_obs(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_legal_actions(self) -> List[int]:
        assert self._state is not None
        return list(self._game.legal_actions(self._state))

    @property
    def legal_actions_mask(self) -> np.ndarray:
        assert self._state is not None
        legal_actions = self._game.legal_actions(self._state)
        mask = np.zeros(self.size * self.size, dtype=bool)
        for action in legal_actions:
            mask[action] = True
        return mask

    def current_player(self) -> int:
        assert self._state is not None
        return self._state.current_player_index

    @property
    def current_player_token(self) -> int:
        assert self._state is not None
        return self._game.current_player(self._state)

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

    def render(self, mode: str = "human") -> Optional[str]:
        if mode == "human":
            print("\n" + "=" * (self.size * 2 + 3))
            print("  " + " ".join(str(i) for i in range(self.size)))
            print("=" * (self.size * 2 + 3))

            for row in range(self.size):
                row_str = f"{row}|"
                for col in range(self.size):
                    if self.board[row, col] == 1:
                        row_str += "X|"
                    elif self.board[row, col] == -1:
                        row_str += "O|"
                    else:
                        row_str += " |"
                print(row_str)

            print("=" * (self.size * 2 + 3))

            if self.done:
                p1_count = np.sum(self.board == 1)
                p2_count = np.sum(self.board == -1)
                print(f"X: {p1_count}, O: {p2_count}")
                if self.winner == 1:
                    print("Player X wins!")
                elif self.winner == -1:
                    print("Player O wins!")
                else:
                    print("Draw!")
            else:
                player_symbol = "X" if self.current_player_token == 1 else "O"
                p1_count = np.sum(self.board == 1)
                p2_count = np.sum(self.board == -1)
                print(f"X: {p1_count}, O: {p2_count}")
                print(f"Current player: {player_symbol}")

            print()

        return None

    def get_state_hash(self) -> str:
        return ",".join(str(x) for x in self.board.flatten())

    def get_state(self) -> OthelloState:
        assert self._state is not None
        return OthelloState(
            board=self.board.copy(),
            current_player_index=self._state.current_player_index,
            winner=self._state.winner,
            done=self._state.done,
            last_move=self._state.last_move,
        )

    def set_state(self, state: OthelloState) -> None:
        self._state = OthelloState(
            board=state.board.copy(),
            current_player_index=state.current_player_index,
            winner=state.winner,
            done=state.done,
            last_move=state.last_move,
        )

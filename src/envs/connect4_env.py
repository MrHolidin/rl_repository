"""Connect Four environment implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..features.observation_builder import BoardChannels, ObservationBuilder
from .base import StepResult, TurnBasedEnv
from .reward_config import RewardConfig


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
        """
        Initialize Connect Four environment.
        
        Args:
            rows: Number of rows (default: 6)
            cols: Number of columns (default: 7)
            reward_config: RewardConfig object (default: RewardConfig with default values)
            observation_builder: ObservationBuilder instance for feature extraction
        """
        self.rows = rows
        self.cols = cols
        self.board: Optional[np.ndarray] = None
        self.winner: Optional[int] = None
        self.done = False
        self._rng = np.random.default_rng()
        self._player_tokens = np.array([1, -1], dtype=np.int8)
        self._current_player_index = 0
        self._last_move: Optional[Tuple[int, int]] = None

        if observation_builder is None:
            observation_builder = BoardChannels(board_shape=(rows, cols))
        self._observation_builder = observation_builder

        # Reward configuration
        if reward_config is None:
            reward_config = RewardConfig()
        self.reward_config = reward_config
        
        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self._current_player_index = 0
        self.winner = None
        self.done = False
        self._last_move = None
        return self._get_obs()

    def step(self, action: int) -> StepResult:
        """
        Execute one step in the environment.
        
        Args:
            action: Column index (0 to cols-1)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
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

        # Drop piece in the column
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

        # Track last move for feature builders
        self._last_move = (row, action)

        # Check for win
        won = self._check_win(row, action, current_token)
        
        # Initialize reward and info
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
            self.winner = current_token
            self.done = True
            terminated = True
            reward = self.reward_config.win
            info["winner"] = self.winner
            info["termination_reason"] = "win"
        elif self._is_board_full():
            self.winner = 0
            self.done = True
            terminated = True
            reward = self.reward_config.draw
            info["winner"] = 0
            info["termination_reason"] = "draw"
        else:
            # Shaping rewards: check all threes on board before switching player
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
        
        # Switch player turn (even if the game has ended, to keep observation POV consistent)
        self._current_player_index = 1 - self._current_player_index

        return StepResult(
            obs=self._get_obs(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _drop_piece(self, col: int, player: int) -> Optional[int]:
        """
        Drop a piece in the specified column.
        
        Args:
            col: Column index
            player: Player ID (1 or -1)
            
        Returns:
            Row index where piece was placed, or None if column is full
        """
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = player
                return row
        return None

    def _check_win(self, row: int, col: int, player: int) -> bool:
        """
        Check if the last move resulted in a win (4 in a row).
        
        Args:
            row: Row of last move
            col: Column of last move
            player: Player ID
            
        Returns:
            True if player won
        """
        return self._check_n_in_row(row, col, player, n=4)
    
    def _check_three_in_row(self, row: int, col: int, player: int) -> bool:
        """
        Check if the last move resulted in 3 in a row.
        
        Args:
            row: Row of last move
            col: Column of last move
            player: Player ID
            
        Returns:
            True if player has 3 in a row (but not 4)
        """
        has_three = self._check_n_in_row(row, col, player, n=3)
        has_four = self._check_n_in_row(row, col, player, n=4)
        return has_three and not has_four
    
    def _has_any_three(self, player: int) -> bool:
        """
        Check if player has any 3 in a row anywhere on the board (but not 4).
        
        Args:
            player: Player ID (1 or -1)
            
        Returns:
            True if player has at least one 3 in a row (but not 4)
        """
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == player:
                    if self._check_three_in_row(r, c, player):
                        return True
        return False
    
    def _check_n_in_row(self, row: int, col: int, player: int, n: int) -> bool:
        """
        Check if the last move resulted in n pieces in a row.
        
        Args:
            row: Row of last move
            col: Column of last move
            player: Player ID
            n: Number of pieces in a row to check for
            
        Returns:
            True if player has n pieces in a row
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            for i in range(1, n):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    break
            for i in range(1, n):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= n:
                return True
        
        return False

    def _is_board_full(self) -> bool:
        """Check if the board is full."""
        return np.all(self.board != 0)

    def get_legal_actions(self) -> list[int]:
        """
        Get list of legal actions (columns that are not full).
        
        Returns:
            List of column indices
        """
        return [col for col in range(self.cols) if self.board[0, col] == 0]

    @property
    def legal_actions_mask(self) -> np.ndarray:
        """Boolean mask of available actions."""
        return (self.board[0] == 0).astype(bool)

    def current_player(self) -> int:
        """Index (0 or 1) of the player whose turn it is."""
        return self._current_player_index

    @property
    def current_player_token(self) -> int:
        """Return board token (1 or -1) for the current player."""
        return int(self._player_tokens[self._current_player_index])

    @property
    def observation_builder(self) -> ObservationBuilder:
        """Return the observation builder used by the environment."""
        return self._observation_builder

    def _get_obs(self) -> np.ndarray:
        """
        Get observation representation.
        
        Returns:
            3D array of shape (3, rows, cols):
            - Channel 0: Current player's pieces (1/0)
            - Channel 1: Opponent's pieces (1/0)
            - Channel 2: Current player indicator (1 for player 1, 0 for player -1)
        """
        state = {
            "board": self.board,
            "current_player_token": self.current_player_token,
            "last_move": self._last_move,
            "legal_actions_mask": self.legal_actions_mask,
        }
        return self._observation_builder.build(state)

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode ('human' for console output)
            
        Returns:
            String representation if mode is 'rgb_array', None otherwise
        """
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
        """
        Get a hashable representation of the current state.
        Useful for tabular methods.
        
        Returns:
            String representation of the board state
        """
        return ",".join(str(x) for x in self.board.flatten())


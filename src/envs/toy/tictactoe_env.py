"""Simple TicTacToe environment for testing Trainer with adversarial games."""

from typing import Optional

import numpy as np

from ..base import StepResult, TurnBasedEnv


class TicTacToeEnv(TurnBasedEnv):
    """
    Minimal TicTacToe (3x3) for testing DQN through Trainer.
    
    Observation: (3, 3, 3) tensor
        - channel 0: current player's pieces (1 where placed)
        - channel 1: opponent's pieces (1 where placed)
        - channel 2: turn indicator (all 1s if player token=1, all 0s if token=-1)
    
    Actions: 0-8 (flattened board positions)
    Rewards: +1 win, -1 loss, 0 draw/continue
    """
    
    def __init__(self):
        self._board = np.zeros((3, 3), dtype=np.int8)  # 0=empty, 1=X, -1=O
        self._current_token = 1  # X starts
        self._done = False
        self._winner = None  # 1, -1, or 0 (draw)
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self._board = np.zeros((3, 3), dtype=np.int8)
        self._current_token = 1
        self._done = False
        self._winner = None
        return self._get_obs()
    
    def step(self, action: int) -> StepResult:
        if self._done:
            return StepResult(
                obs=self._get_obs(),
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"winner": self._winner},
            )
        
        row, col = divmod(action, 3)
        
        # Invalid move check
        if self._board[row, col] != 0:
            return StepResult(
                obs=self._get_obs(),
                reward=-0.1,  # Small penalty for invalid
                terminated=False,
                truncated=False,
                info={"invalid": True, "winner": None},
            )
        
        # Place piece
        self._board[row, col] = self._current_token
        
        # Check win
        if self._check_win(self._current_token):
            self._done = True
            self._winner = self._current_token
            reward = 1.0  # Current player won
        # Check draw
        elif np.all(self._board != 0):
            self._done = True
            self._winner = 0
            reward = 0.0
        else:
            reward = 0.0
        
        # Switch player
        self._current_token = -self._current_token
        
        return StepResult(
            obs=self._get_obs(),
            reward=reward,
            terminated=self._done,
            truncated=False,
            info={"winner": self._winner},
        )
    
    @property
    def legal_actions_mask(self) -> np.ndarray:
        return (self._board.flatten() == 0)
    
    def current_player(self) -> int:
        return 0 if self._current_token == 1 else 1
    
    def render(self) -> None:
        symbols = {0: ".", 1: "X", -1: "O"}
        for row in self._board:
            print(" ".join(symbols[cell] for cell in row))
        print()
    
    def _get_obs(self) -> np.ndarray:
        """Build observation from current player's perspective."""
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        
        # Channel 0: current player's pieces
        obs[0] = (self._board == self._current_token).astype(np.float32)
        
        # Channel 1: opponent's pieces
        obs[1] = (self._board == -self._current_token).astype(np.float32)
        
        # Channel 2: turn indicator (1 if token=1, 0 if token=-1)
        obs[2] = 1.0 if self._current_token == 1 else 0.0
        
        return obs
    
    def _check_win(self, token: int) -> bool:
        """Check if token has won."""
        b = self._board
        
        # Rows
        for r in range(3):
            if np.all(b[r, :] == token):
                return True
        
        # Columns
        for c in range(3):
            if np.all(b[:, c] == token):
                return True
        
        # Diagonals
        if b[0, 0] == b[1, 1] == b[2, 2] == token:
            return True
        if b[0, 2] == b[1, 1] == b[2, 0] == token:
            return True
        
        return False

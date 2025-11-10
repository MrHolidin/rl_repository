"""Heuristic agent implementation."""

import random
from typing import Optional

import numpy as np

from .base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    """
    Heuristic agent that uses simple rules:
    1. Win if possible
    2. Block opponent from winning
    3. Otherwise random
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize heuristic agent.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Select action using heuristic rules."""
        if legal_mask is None:
            raise ValueError("HeuristicAgent requires legal_mask to be provided.")
        legal_actions = np.flatnonzero(legal_mask).tolist()
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # Extract board from observation
        # obs shape: (3, rows, cols)
        board = np.zeros((obs.shape[1], obs.shape[2]), dtype=np.int8)
        board[obs[0] == 1] = 1  # Current player's pieces
        board[obs[1] == 1] = -1  # Opponent's pieces
        
        current_player = 1 if obs[2, 0, 0] == 1 else -1
        
        # Try to win
        for action in legal_actions:
            if self._would_win(board, action, current_player):
                return action
        
        # Try to block opponent
        for action in legal_actions:
            if self._would_win(board, action, -current_player):
                return action
        
        # Otherwise random
        return int(random.choice(legal_actions))

    def _would_win(self, board: np.ndarray, col: int, player: int) -> bool:
        """
        Check if dropping a piece in column would result in a win.
        
        Args:
            board: Current board state
            col: Column index
            player: Player ID (1 or -1)
            
        Returns:
            True if move would result in win
        """
        # Find row where piece would land
        row = None
        for r in range(board.shape[0] - 1, -1, -1):
            if board[r, col] == 0:
                row = r
                break
        
        if row is None:
            return False
        
        # Temporarily place piece
        board[row, col] = player
        
        # Check for win
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal /
            (1, -1),  # diagonal \
        ]

        for dr, dc in directions:
            count = 1
            # Check in positive direction
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] == player:
                    count += 1
                else:
                    break
            # Check in negative direction
            for i in range(1, 4):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= 4:
                board[row, col] = 0  # Restore
                return True
        
        board[row, col] = 0  # Restore
        return False

    def save(self, path: str) -> None:
        """Heuristic agent has no persistent state."""
        return None

    @classmethod
    def load(cls, path: str, **kwargs: object) -> "HeuristicAgent":
        """Return a new heuristic agent."""
        return cls(**kwargs)


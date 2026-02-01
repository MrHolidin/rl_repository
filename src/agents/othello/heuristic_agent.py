"""Heuristic agent implementation for Othello."""

import random
from typing import Optional

import numpy as np

from ..base_agent import BaseAgent


class OthelloHeuristicAgent(BaseAgent):
    """
    Heuristic agent for Othello using positional weights.
    
    Strategy:
    1. Corners are most valuable (stable positions)
    2. Edges are valuable
    3. Positions adjacent to corners are dangerous (X-squares, C-squares)
    4. Maximize mobility and disc count
    """

    POSITION_WEIGHTS_8x8 = np.array([
        [100, -20,  10,   5,   5,  10, -20, 100],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [ 10,  -2,   5,   1,   1,   5,  -2,  10],
        [  5,  -2,   1,   0,   0,   1,  -2,   5],
        [  5,  -2,   1,   0,   0,   1,  -2,   5],
        [ 10,  -2,   5,   1,   1,   5,  -2,  10],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [100, -20,  10,   5,   5,  10, -20, 100],
    ], dtype=np.float32)

    def __init__(self, seed: Optional[int] = None, board_size: int = 8):
        """
        Initialize heuristic agent.
        
        Args:
            seed: Random seed for reproducibility
            board_size: Size of the board (default 8x8)
        """
        self._rng = random.Random(seed)
        self.board_size = board_size
        
        if board_size == 8:
            self.position_weights = self.POSITION_WEIGHTS_8x8
        else:
            self.position_weights = np.zeros((board_size, board_size), dtype=np.float32)

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Select action using positional heuristics."""
        if legal_mask is None:
            raise ValueError("OthelloHeuristicAgent requires legal_mask to be provided.")
        
        legal_actions = np.flatnonzero(legal_mask).tolist()
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        board = np.zeros((obs.shape[1], obs.shape[2]), dtype=np.int8)
        board[obs[0] == 1] = 1
        board[obs[1] == 1] = -1
        
        best_score = float('-inf')
        best_actions = []
        
        for action in legal_actions:
            row = action // self.board_size
            col = action % self.board_size
            
            score = self._evaluate_move(board, row, col, action)
            
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        
        if deterministic:
            return best_actions[0]
        return int(self._rng.choice(best_actions))

    def _evaluate_move(self, board: np.ndarray, row: int, col: int, action: int) -> float:
        """
        Evaluate a move based on multiple heuristics.
        
        Args:
            board: Current board state
            row: Row of the move
            col: Column of the move
            action: Action index
            
        Returns:
            Score for the move (higher is better)
        """
        score = 0.0
        
        score += self.position_weights[row, col]
        
        num_flips = self._count_flips(board, row, col, player=1)
        score += num_flips * 2
        
        if self._is_corner(row, col):
            score += 50
        elif self._is_edge(row, col):
            score += 5
        
        return score

    def _count_flips(self, board: np.ndarray, row: int, col: int, player: int) -> int:
        """Count how many pieces would be flipped by this move."""
        if board[row, col] != 0:
            return 0

        opponent = -player
        total_flips = 0
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
            flips = 0
            r, c = row + dr, col + dc

            while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == opponent:
                flips += 1
                r += dr
                c += dc

            if 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player and flips > 0:
                total_flips += flips

        return total_flips

    def _is_corner(self, row: int, col: int) -> bool:
        """Check if position is a corner."""
        corners = [(0, 0), (0, self.board_size - 1), (self.board_size - 1, 0), (self.board_size - 1, self.board_size - 1)]
        return (row, col) in corners

    def _is_edge(self, row: int, col: int) -> bool:
        """Check if position is on an edge (but not corner)."""
        if self._is_corner(row, col):
            return False
        return row == 0 or row == self.board_size - 1 or col == 0 or col == self.board_size - 1

    def save(self, path: str) -> None:
        """Heuristic agent has no persistent state."""
        return None

    @classmethod
    def load(cls, path: str, **kwargs: object) -> "OthelloHeuristicAgent":
        """Return a new heuristic agent."""
        return cls(**kwargs)

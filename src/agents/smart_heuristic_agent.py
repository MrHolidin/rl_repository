"""Smart heuristic agent implementation for Connect Four."""

import random
from typing import List, Optional, Tuple
import numpy as np

from .base_agent import BaseAgent


class SmartHeuristicAgent(BaseAgent):
    """
    Smart heuristic agent with advanced Connect Four strategy:
    1. Win if possible
    2. Block opponent from winning
    3. Create threats (3 in a row with open ends)
    4. Block opponent threats
    5. Play center columns (more valuable)
    6. Avoid giving opponent opportunities
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize smart heuristic agent.
        
        Args:
            seed: Random seed for reproducibility (uses local RNG, not global)
        """
        self._rng = random.Random(seed)

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Select action using smart heuristic rules."""
        if legal_mask is None:
            raise ValueError("SmartHeuristicAgent requires legal_mask to be provided.")
        legal_actions = np.flatnonzero(legal_mask).tolist()
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # Extract board from observation (relative coords: current player = +1)
        board = np.zeros((obs.shape[1], obs.shape[2]), dtype=np.int8)
        board[obs[0] == 1] = 1
        board[obs[1] == 1] = -1
        current_player = 1
        opponent = -1
        
        # Priority 1: Win if possible
        for action in legal_actions:
            if self._would_win(board, action, current_player):
                return action
        
        # Priority 2: Block opponent from winning
        for action in legal_actions:
            if self._would_win(board, action, opponent):
                return action
        
        # Priority 3: Create a winning threat (3 in a row with open end)
        threat_scores = {}
        for action in legal_actions:
            threat_score = self._evaluate_threat(board, action, current_player)
            if threat_score > 0:
                threat_scores[action] = threat_score
        
        if threat_scores:
            # Choose action with highest threat score
            best_actions = [action for action, score in threat_scores.items() if score == max(threat_scores.values())]
            return self._rng.choice(best_actions)
        
        # Priority 4: Block opponent's threats
        block_scores = {}
        for action in legal_actions:
            block_score = self._evaluate_threat(board, action, opponent)
            if block_score > 0:
                block_scores[action] = block_score
        
        if block_scores:
            best_score = max(block_scores.values())
            best_actions = [action for action, score in block_scores.items() if score == best_score]
            return self._rng.choice(best_actions)
        
        # Priority 5: Play center columns (more valuable)
        center_preference = self._get_center_preference(legal_actions, board.shape[1])
        if center_preference:
            return center_preference
        
        # Priority 6: Avoid giving opponent opportunities
        safe_actions = self._get_safe_actions(board, legal_actions, current_player)
        if safe_actions:
            return int(self._rng.choice(safe_actions))

        # Fallback: random from legal actions
        return int(self._rng.choice(legal_actions))

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
        row = self._get_drop_row(board, col)
        if row is None:
            return False
        
        # Temporarily place piece
        board[row, col] = player
        
        # Check for win
        won = self._check_win(board, row, col, player)
        
        # Restore
        board[row, col] = 0
        
        return won

    def _evaluate_threat(self, board: np.ndarray, col: int, player: int) -> float:
        """
        Evaluate how threatening a move is (3 in a row with open ends).
        Returns a score: 0 = no threat, higher = more threatening.
        
        Args:
            board: Current board state
            col: Column index
            player: Player ID (1 or -1)
            
        Returns:
            Threat score (0.0 to 1.0)
        """
        row = self._get_drop_row(board, col)
        if row is None:
            return 0.0
        
        # Temporarily place piece
        board[row, col] = player
        
        # Check for threats in all directions
        threat_score = 0.0
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal /
            (1, -1),  # diagonal \
        ]
        
        for dr, dc in directions:
            # Count consecutive pieces in this direction
            count = 1  # The piece we just placed
            open_ends = 0
            
            # Check positive direction
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                    if board[r, c] == player:
                        count += 1
                    elif board[r, c] == 0:
                        # Check if this position is reachable (not floating)
                        # For vertical direction, check if below is filled
                        # For horizontal/diagonal, check if below is filled
                        if dr == 1:  # Vertical - check if below is filled
                            if r == board.shape[0] - 1 or (r + 1 < board.shape[0] and board[r + 1, c] != 0):
                                open_ends += 1
                                break
                        else:  # Horizontal or diagonal - check if below is filled
                            if r == board.shape[0] - 1 or (r + 1 < board.shape[0] and board[r + 1, c] != 0):
                                open_ends += 1
                                break
                    else:
                        break
                else:
                    break
            
            # Check negative direction
            for i in range(1, 4):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                    if board[r, c] == player:
                        count += 1
                    elif board[r, c] == 0:
                        # Check if this position is reachable
                        if dr == 1:  # Vertical - check if below is filled
                            if r == board.shape[0] - 1 or (r + 1 < board.shape[0] and board[r + 1, c] != 0):
                                open_ends += 1
                                break
                        else:  # Horizontal or diagonal - check if below is filled
                            if r == board.shape[0] - 1 or (r + 1 < board.shape[0] and board[r + 1, c] != 0):
                                open_ends += 1
                                break
                    else:
                        break
                else:
                    break
            
            # Evaluate threat based on count and open ends
            if count >= 3 and open_ends > 0:
                # 3 in a row with at least one open end is a strong threat
                threat_score = max(threat_score, 0.5 + 0.3 * open_ends)
            elif count == 2 and open_ends >= 2:
                # 2 in a row with 2 open ends is also good
                threat_score = max(threat_score, 0.3)
        
        # Restore
        board[row, col] = 0
        
        return threat_score

    def _get_center_preference(self, legal_actions: List[int], num_cols: int) -> int:
        """
        Prefer center columns (more valuable in Connect Four).
        
        Args:
            legal_actions: List of legal actions
            num_cols: Number of columns
            
        Returns:
            Preferred action or None
        """
        center = num_cols // 2
        center_cols = [center]
        if num_cols % 2 == 0:
            center_cols.append(center - 1)
        
        # Prefer center columns
        center_candidates = [col for col in center_cols if col in legal_actions]
        if center_candidates:
            return self._rng.choice(center_candidates)
        
        # Prefer columns near center
        ring_candidates = []
        for offset in range(1, num_cols // 2 + 1):
            for col in center_cols:
                for side in (-offset, offset):
                    candidate = col + side
                    if candidate in legal_actions:
                        ring_candidates.append(candidate)
        if ring_candidates:
            return self._rng.choice(ring_candidates)
        
        return None

    def _get_safe_actions(self, board: np.ndarray, legal_actions: List[int], player: int) -> List[int]:
        """
        Get actions that don't give opponent immediate opportunities.
        
        Args:
            board: Current board state
            legal_actions: List of legal actions
            player: Current player ID
            
        Returns:
            List of safe actions
        """
        opponent = -player
        safe_actions = []
        
        for action in legal_actions:
            row = self._get_drop_row(board, action)
            if row is None:
                continue
            
            # Temporarily place piece
            board[row, action] = player
            
            # Check if this gives opponent a winning opportunity
            is_safe = True
            for opp_action in range(board.shape[1]):
                if self._would_win(board, opp_action, opponent):
                    is_safe = False
                    break
            
            # Restore
            board[row, action] = 0
            
            if is_safe:
                safe_actions.append(action)
        
        return safe_actions if safe_actions else legal_actions

    def _get_drop_row(self, board: np.ndarray, col: int) -> int:
        """
        Get the row where a piece would land in a column.
        
        Args:
            board: Current board state
            col: Column index
            
        Returns:
            Row index where piece would land, or None if column is full
        """
        for row in range(board.shape[0] - 1, -1, -1):
            if board[row, col] == 0:
                return row
        return None

    def _check_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """
        Check if the last move resulted in a win.
        
        Args:
            board: Current board state
            row: Row of last move
            col: Column of last move
            player: Player ID
            
        Returns:
            True if player won
        """
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal /
            (1, -1),  # diagonal \
        ]

        for dr, dc in directions:
            count = 1  # Count the piece just placed
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
                return True
        
        return False

    def save(self, path: str) -> None:
        """Smart heuristic agent has no persistent state."""
        return None

    @classmethod
    def load(cls, path: str, **kwargs: object) -> "SmartHeuristicAgent":
        """Return a new smart heuristic agent instance."""
        return cls(**kwargs)


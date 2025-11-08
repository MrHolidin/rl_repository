"""Connect Four environment implementation."""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class Connect4Env:
    """
    Connect Four environment (6 rows × 7 columns).
    
    Players drop pieces into columns. Win condition: 4 in a row
    (horizontal, vertical, or diagonal).
    """

    def __init__(
        self, 
        rows: int = 6, 
        cols: int = 7,
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_draw: float = 0.0,
        reward_three_in_row: float = 0.0,
        reward_opponent_three_in_row: float = 0.0,
        reward_invalid_action: float = -0.1,
    ):
        """
        Initialize Connect Four environment.
        
        Args:
            rows: Number of rows (default: 6)
            cols: Number of columns (default: 7)
            reward_win: Reward for winning (default: 1.0)
            reward_loss: Reward for losing (default: -1.0)
            reward_draw: Reward for draw (default: 0.0)
            reward_three_in_row: Reward for getting 3 in a row (default: 0.0)
            reward_opponent_three_in_row: Penalty for opponent having 3 in a row (default: 0.0)
            reward_invalid_action: Reward for invalid action (default: -0.1)
        """
        self.rows = rows
        self.cols = cols
        self.board = None
        self.current_player = 1  # 1 or -1
        self.winner = None
        self.done = False
        
        # Reward configuration
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.reward_draw = reward_draw
        self.reward_three_in_row = reward_three_in_row
        self.reward_opponent_three_in_row = reward_opponent_three_in_row
        self.reward_invalid_action = reward_invalid_action
        
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.winner = None
        self.done = False
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Column index (0 to cols-1)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")
        
        if action not in self.get_legal_actions():
            # Invalid action: return negative reward and don't change state
            return self._get_obs(), self.reward_invalid_action, self.done, {"invalid_action": True}

        # Drop piece in the column
        row = self._drop_piece(action, self.current_player)
        if row is None:
            # Column is full
            return self._get_obs(), self.reward_invalid_action, self.done, {"invalid_action": True}

        # Check for win
        won = self._check_win(row, action, self.current_player)
        
        # Initialize reward and info
        reward = 0.0
        info = {"winner": None, "reason": None, "three_in_row": False, "opponent_three_in_row": False}
        
        if won:
            self.winner = self.current_player
            self.done = True
            reward = self.reward_win
            info = {"winner": self.current_player, "reason": "win", "three_in_row": False, "opponent_three_in_row": False}
        elif self._is_board_full():
            self.winner = 0
            self.done = True
            reward = self.reward_draw
            info = {"winner": 0, "reason": "draw", "three_in_row": False, "opponent_three_in_row": False}
        else:
            # Shaping rewards: check all threes on board before switching player
            if self.reward_three_in_row != 0.0:
                if self._has_any_three(self.current_player):
                    reward += self.reward_three_in_row
                    info["three_in_row"] = True
                else:
                    info["three_in_row"] = False
            
            if self.reward_opponent_three_in_row != 0.0:
                if self._has_any_three(-self.current_player):
                    reward -= self.reward_opponent_three_in_row
                    info["opponent_three_in_row"] = True
                else:
                    info["opponent_three_in_row"] = False

        self.current_player *= -1

        return self._get_obs(), reward, self.done, info

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

    def get_legal_actions(self) -> List[int]:
        """
        Get list of legal actions (columns that are not full).
        
        Returns:
            List of column indices
        """
        return [col for col in range(self.cols) if self.board[0, col] == 0]

    def _get_obs(self) -> np.ndarray:
        """
        Get observation representation.
        
        Returns:
            3D array of shape (3, rows, cols):
            - Channel 0: Current player's pieces (1/0)
            - Channel 1: Opponent's pieces (1/0)
            - Channel 2: Current player indicator (1 for player 1, 0 for player -1)
        """
        obs = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        obs[0] = (self.board == self.current_player).astype(np.float32)
        obs[1] = (self.board == -self.current_player).astype(np.float32)
        obs[2] = np.full((self.rows, self.cols), 1.0 if self.current_player == 1 else 0.0, dtype=np.float32)
        
        return obs

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
                player_symbol = "X" if self.current_player == 1 else "O"
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


"""Q-learning agent implementation (tabular)."""

import pickle
import random
from typing import Dict, List, Optional

import numpy as np

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Tabular Q-learning agent.
    
    Uses state hashing for state representation.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int = None,
    ):
        """
        Initialize Q-learning agent.
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial epsilon for epsilon-greedy
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            seed: Random seed
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table: Dict[str, Dict[int, float]] = {}
        self.training = True
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Select action using epsilon-greedy policy."""
        if legal_mask is None:
            raise ValueError("QLearningAgent requires legal_mask to be provided.")
        legal_actions = np.flatnonzero(legal_mask).tolist()
        if not legal_actions:
            raise ValueError("No legal actions available")

        state_hash = self._obs_to_hash(obs)
        
        # Initialize Q-values for this state if not seen before
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {action: 0.0 for action in legal_actions}
        
        explore = not deterministic and self.training and random.random() < self.epsilon

        if explore:
            return int(random.choice(legal_actions))

        # Select best action from legal actions
        q_values = self.q_table[state_hash]
        best_value = max(q_values.get(a, 0.0) for a in legal_actions)
        best_actions = [a for a in legal_actions if q_values.get(a, 0.0) == best_value]
        return int(random.choice(best_actions))

    def observe(self, transition) -> Dict[str, float]:
        """
        Update Q-values using Q-learning update rule.
        
        Args:
            transition: Tuple of (obs, action, reward, next_obs, done, info)
        """
        if not self.training:
            return {}

        if hasattr(transition, "obs"):
            obs = transition.obs
            action = transition.action
            reward = transition.reward
            next_obs = transition.next_obs
            done = transition.terminated or transition.truncated
        else:
            obs, action, reward, next_obs, done, *_ = transition
        
        state_hash = self._obs_to_hash(obs)
        next_state_hash = self._obs_to_hash(next_obs)
        
        # Initialize Q-values if needed
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}
        if next_state_hash not in self.q_table:
            self.q_table[next_state_hash] = {}
        
        # Get current Q-value
        current_q = self.q_table[state_hash].get(action, 0.0)
        
        if done:
            # Terminal state: Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
            target_q = reward
        else:
            # Non-terminal: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
            next_legal_actions = self._get_legal_actions_from_obs(next_obs)
            if next_legal_actions:
                max_next_q = max(
                    self.q_table[next_state_hash].get(a, 0.0)
                    for a in next_legal_actions
                )
            else:
                max_next_q = 0.0
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        self.q_table[state_hash][action] = current_q + self.learning_rate * (target_q - current_q)
        
        # NOTE: Epsilon decay is now handled per episode in training loop, not per step
        # This prevents epsilon from decaying too quickly
        return {}

    def _obs_to_hash(self, obs: np.ndarray) -> str:
        """
        Convert observation to hashable state representation.
        
        Args:
            obs: Observation array
            
        Returns:
            State hash string
        """
        # Reconstruct board from observation
        # obs shape: (3, rows, cols)
        board = np.zeros((obs.shape[1], obs.shape[2]), dtype=np.int8)
        board[obs[0] == 1] = 1  # Current player's pieces
        board[obs[1] == 1] = -1  # Opponent's pieces
        
        # Include current player in hash
        current_player = 1 if obs[2, 0, 0] == 1 else -1
        
        # Create hash: board state + current player
        return ",".join(str(x) for x in board.flatten()) + f",p{current_player}"

    def _get_legal_actions_from_obs(self, obs: np.ndarray) -> List[int]:
        """
        Extract legal actions from observation.
        
        Args:
            obs: Observation array
            
        Returns:
            List of legal action indices
        """
        # Reconstruct board
        board = np.zeros((obs.shape[1], obs.shape[2]), dtype=np.int8)
        board[obs[0] == 1] = 1
        board[obs[1] == 1] = -1
        
        # Legal actions are columns that are not full (top row is empty)
        return [col for col in range(board.shape[1]) if board[0, col] == 0]

    def train(self) -> None:
        """Set agent to training mode."""
        self.training = True

    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self.training = False

    def save(self, path: str) -> None:
        """
        Save Q-table to file.
        
        Args:
            path: Path to save file
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
            }, f)

    @classmethod
    def load(cls, path: str, **kwargs: object) -> "QLearningAgent":
        """
        Load Q-table from file and return a new agent instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(**kwargs)
        agent.q_table = data["q_table"]
        agent.epsilon = data.get("epsilon", agent.epsilon)
        agent.learning_rate = data.get("learning_rate", agent.learning_rate)
        agent.discount_factor = data.get("discount_factor", agent.discount_factor)
        return agent


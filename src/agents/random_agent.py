"""Random agent implementation."""

import random
from typing import List
import numpy as np

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects actions randomly from legal actions."""

    def __init__(self, seed: int = None):
        """
        Initialize random agent.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """
        Select a random action from legal actions.
        
        Args:
            obs: Current observation (not used)
            legal_actions: List of legal action indices
            
        Returns:
            Randomly selected action index
        """
        if not legal_actions:
            raise ValueError("No legal actions available")
        return random.choice(legal_actions)


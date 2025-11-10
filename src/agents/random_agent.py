"""Random agent implementation."""

import random
from typing import Optional

import numpy as np

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects actions randomly from legal actions."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random agent.

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
        """
        Select a random action from legal actions.

        Args:
            obs: Current observation (unused)
            legal_mask: Boolean mask of legal actions
            deterministic: Ignored for random policy

        Returns:
            Randomly selected action index
        """
        if legal_mask is None:
            raise ValueError("RandomAgent requires legal_mask to be provided.")
        legal_indices = np.flatnonzero(legal_mask)
        if legal_indices.size == 0:
            raise ValueError("No legal actions available")
        return int(np.random.choice(legal_indices))

    def save(self, path: str) -> None:
        """Random agent has no state to persist."""
        return None

    @classmethod
    def load(cls, path: str, **kwargs: object) -> "RandomAgent":
        """Return a new random agent instance."""
        return cls(**kwargs)


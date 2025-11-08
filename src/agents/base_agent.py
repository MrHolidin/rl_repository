"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
import numpy as np


class BaseAgent(ABC):
    """Base class for all agents."""

    @abstractmethod
    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """
        Select an action given an observation.
        
        Args:
            obs: Current observation
            legal_actions: List of legal action indices
            
        Returns:
            Selected action index
        """
        raise NotImplementedError

    def observe(self, transition: Tuple[Any, ...]) -> None:
        """
        Observe a transition (for learning).
        
        Args:
            transition: Tuple of (obs, action, reward, next_obs, done, info)
        """
        pass

    def save(self, path: str) -> None:
        """
        Save agent to file.
        
        Args:
            path: Path to save file
        """
        pass

    def load(self, path: str) -> None:
        """
        Load agent from file.
        
        Args:
            path: Path to load file from
        """
        pass

    def train(self) -> None:
        """Set agent to training mode."""
        pass

    def eval(self) -> None:
        """Set agent to evaluation mode."""
        pass


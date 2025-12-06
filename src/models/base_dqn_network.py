"""Abstract base class for DQN networks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseDQNNetwork(nn.Module, ABC):
    """
    Abstract base class for all DQN networks.
    
    This class defines the interface that all game-specific DQN networks must implement.
    Each network must handle:
    - Forward pass with optional legal action masking
    - Input observation processing appropriate for the game
    - Serialization of constructor arguments for checkpoint save/load
    """

    def __init__(
        self,
        num_actions: int,
        dueling: bool = False,
    ):
        """
        Initialize base DQN network.
        
        Args:
            num_actions: Number of discrete actions.
            dueling: Whether to use dueling architecture.
        """
        super().__init__()
        self._num_actions = num_actions
        self._dueling = dueling

    @property
    def num_actions(self) -> int:
        """Number of discrete actions."""
        return self._num_actions

    @property
    def dueling(self) -> bool:
        """Whether this network uses dueling architecture."""
        return self._dueling

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input observation tensor.
            legal_mask: Optional boolean mask of legal actions (batch, num_actions).
            
        Returns:
            Q-values tensor of shape (batch, num_actions).
        """
        pass

    @abstractmethod
    def get_constructor_kwargs(self) -> Dict[str, Any]:
        """
        Return kwargs needed to recreate this network.
        
        Used for serialization - the returned dict will be saved in checkpoint
        and passed to __init__ when loading.
        
        Returns:
            Dictionary of constructor arguments.
        """
        pass

    @classmethod
    def get_class_name(cls) -> str:
        """
        Return the class name for registry lookup.
        
        By default returns the class name. Override if needed.
        
        Returns:
            Class name string.
        """
        return cls.__name__


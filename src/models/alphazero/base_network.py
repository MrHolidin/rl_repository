"""Base class for AlphaZero networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseAlphaZeroNetwork(nn.Module, ABC):
    """
    Base class for AlphaZero-style dual-head networks.

    Outputs:
        - policy_logits: (batch, num_actions)
        - value: (batch, 1) in [-1, 1]
    """

    def __init__(self, num_actions: int):
        super().__init__()
        self._num_actions = num_actions

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            (policy_logits, value) tuple
        """
        pass

    @abstractmethod
    def get_constructor_kwargs(self) -> Dict[str, Any]:
        pass

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

    def predict(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy probabilities and value."""
        policy_logits, value = self.forward(x, legal_mask)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))

        policy_probs = torch.softmax(policy_logits, dim=-1)
        return policy_probs, value

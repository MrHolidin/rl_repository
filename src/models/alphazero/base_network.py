"""Base class for AlphaZero networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with batch normalization used in AlphaZero trunks."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


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

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

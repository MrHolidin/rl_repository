"""Connect4-specific DQN network implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dqn_network import BaseDQNNetwork
from .dueling_utils import dueling_aggregate


class Connect4DQN(BaseDQNNetwork):
    """
    Deep Q-Network specialized for Connect Four.

    Simple architecture:
    - Two conv layers (no normalization for stable gradients)
    - Flatten → FC layers
    - Optional dueling head
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        in_channels: int = 3,
        num_actions: int = 7,
        dueling: bool = False,
    ):
        super().__init__(num_actions=num_actions, dueling=dueling)

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels

        # Conv layers (no normalization - keeps gradients stable)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Flatten size: 128 channels * rows * cols
        self.flatten_size = 128 * rows * cols

        # FC layers
        if dueling:
            self.fc_shared = nn.Linear(self.flatten_size, 256)

            # Value stream
            self.fc_value_hidden = nn.Linear(256, 128)
            self.fc_value = nn.Linear(128, 1)

            # Advantage stream
            self.fc_adv_hidden = nn.Linear(256, 128)
            self.fc_advantage = nn.Linear(128, num_actions)
        else:
            self.fc1 = nn.Linear(self.flatten_size, 256)
            self.fc2 = nn.Linear(256, num_actions)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, rows, cols)
            legal_mask: Optional[bool] (batch, num_actions) — used in dueling_aggregate
        Returns:
            Q-values: (batch, num_actions)
        """
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.view(x.size(0), -1)

        if self._dueling:
            # Shared layer
            x = F.relu(self.fc_shared(x))

            # Value stream
            value = F.relu(self.fc_value_hidden(x))
            value = self.fc_value(value)

            # Advantage stream
            advantage = F.relu(self.fc_adv_hidden(x))
            advantage = self.fc_advantage(advantage)

            # Aggregation considering legal actions
            q_values = dueling_aggregate(value, advantage, legal_mask)
        else:
            x = F.relu(self.fc1(x))
            q_values = self.fc2(x)

        return q_values

    def get_constructor_kwargs(self) -> dict:
        """Return kwargs needed to recreate this network."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "in_channels": self.in_channels,
            "num_actions": self._num_actions,
            "dueling": self._dueling,
        }


# Backward compatibility alias
DQN = Connect4DQN

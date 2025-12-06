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

    Architecture features:
    - Conv block with GroupNorm (better for RL, PER, self-play than BatchNorm)
    - Two residual blocks with 128 channels
    - Global Average Pooling over the board (6x7) → compact feature vector
    - Dueling head with LayerNorm (optional)
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

        # --- Conv block ---
        # 3×6×7 → 64×6×7
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64)

        # 64×6×7 → 128×6×7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=128)

        # Two residual blocks with 128 channels
        self.res_block1 = self._make_res_block(128)
        self.res_block2 = self._make_res_block(128)

        # After conv block the feature map size: (B, 128, rows, cols)
        # We apply Global Average Pooling → (B, 128)
        conv_out_channels = 128
        head_in_dim = conv_out_channels  # after GAP

        # --- Fully connected layers ---
        if dueling:
            # Shared layer before value/advantage split
            self.fc_shared = nn.Linear(head_in_dim, 256)
            self.shared_norm = nn.LayerNorm(256)
            self.shared_dropout = nn.Dropout(p=0.05)

            # Value stream: V(s)
            self.fc_value_hidden = nn.Linear(256, 128)
            self.fc_value = nn.Linear(128, 1)

            # Advantage stream: A(s, a)
            self.fc_adv_hidden = nn.Linear(256, 128)
            self.fc_advantage = nn.Linear(128, num_actions)
        else:
            # Standard DQN head
            self.fc1 = nn.Linear(head_in_dim, 256)
            self.fc1_norm = nn.LayerNorm(256)
            self.fc2 = nn.Linear(256, num_actions)

    @staticmethod
    def _make_res_block(channels: int) -> nn.Module:
        """One residual block: Conv-GN-ReLU-Conv-GN with external skip connection."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16 if channels >= 128 else 8, num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16 if channels >= 128 else 8, num_channels=channels),
        )

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

        # --- Conv block with residual layers ---
        # 3×6×7 → 64×6×7
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)

        # 64×6×7 → 128×6×7
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)

        # Residual block 1
        res = x
        out = self.res_block1(x)
        x = F.relu(res + out)

        # Residual block 2
        res = x
        out = self.res_block2(x)
        x = F.relu(res + out)

        # --- Global Average Pooling over the board ---
        # (B, 128, rows, cols) → (B, 128)
        x = x.mean(dim=(2, 3))

        if self._dueling:
            # Shared layer
            x = self.fc_shared(x)
            x = self.shared_norm(x)
            x = F.relu(x)
            x = self.shared_dropout(x)

            # Value stream
            value = F.relu(self.fc_value_hidden(x))   # (B, 128)
            value = self.fc_value(value)              # (B, 1)

            # Advantage stream
            advantage = F.relu(self.fc_adv_hidden(x))  # (B, 128)
            advantage = self.fc_advantage(advantage)   # (B, num_actions)

            # Proper aggregation considering legal actions
            q_values = dueling_aggregate(value, advantage, legal_mask)
        else:
            # Standard DQN head
            x = self.fc1(x)
            x = self.fc1_norm(x)
            x = F.relu(x)
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

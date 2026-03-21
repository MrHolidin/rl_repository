"""AlphaZero network for Connect4."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base_network import BaseAlphaZeroNetwork, ResidualBlock


class Connect4AlphaZeroNetwork(BaseAlphaZeroNetwork):
    """
    AlphaZero-style network for Connect4.

    Architecture:
        - Convolutional stem
        - Residual tower
        - Policy head (per-column output)
        - Value head (scalar output with tanh)
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        in_channels: int = 2,
        num_actions: int = 7,
        trunk_channels: int = 64,
        num_res_blocks: int = 4,
        policy_channels: int = 16,
        value_channels: int = 16,
        value_hidden: int = 64,
    ):
        super().__init__(num_actions=num_actions)

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.trunk_channels = trunk_channels
        self.num_res_blocks = num_res_blocks
        self.policy_channels = policy_channels
        self.value_channels = value_channels
        self.value_hidden = value_hidden

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(),
        )

        self.res_tower = nn.Sequential(
            *[ResidualBlock(trunk_channels) for _ in range(num_res_blocks)]
        )

        self.policy_conv = nn.Sequential(
            nn.Conv2d(trunk_channels, policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
        )
        self.policy_fc = nn.Conv1d(policy_channels, 1, kernel_size=1)

        self.value_conv = nn.Sequential(
            nn.Conv2d(trunk_channels, value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(value_channels * rows * cols, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.res_tower(h)

        p = self.policy_conv(h)
        p = p.mean(dim=2)
        p = self.policy_fc(p).squeeze(1)

        if legal_mask is not None:
            p = p.masked_fill(~legal_mask, float("-inf"))

        v = self.value_conv(h)
        v = v.flatten(1)
        v = self.value_fc(v)

        return p, v

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "in_channels": self.in_channels,
            "num_actions": self._num_actions,
            "trunk_channels": self.trunk_channels,
            "num_res_blocks": self.num_res_blocks,
            "policy_channels": self.policy_channels,
            "value_channels": self.value_channels,
            "value_hidden": self.value_hidden,
        }

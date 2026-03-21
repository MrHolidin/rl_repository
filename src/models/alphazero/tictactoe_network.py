"""AlphaZero network for TicTacToe."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseAlphaZeroNetwork
from .connect4_network import ResidualBlock


class TicTacToeAlphaZeroNetwork(BaseAlphaZeroNetwork):
    """
    AlphaZero-style network for TicTacToe (3x3 board, 9 actions).

    Policy head outputs one logit per cell (not per column).
    """

    def __init__(
        self,
        trunk_channels: int = 64,
        num_res_blocks: int = 2,
        policy_channels: int = 16,
        value_channels: int = 16,
        value_hidden: int = 64,
    ):
        super().__init__(num_actions=9)

        self.trunk_channels = trunk_channels
        self.num_res_blocks = num_res_blocks
        self.policy_channels = policy_channels
        self.value_channels = value_channels
        self.value_hidden = value_hidden

        self.stem = nn.Sequential(
            nn.Conv2d(2, trunk_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(),
        )

        self.res_tower = nn.Sequential(
            *[ResidualBlock(trunk_channels) for _ in range(num_res_blocks)]
        )

        # Policy head: flatten and output 9 logits
        self.policy_conv = nn.Sequential(
            nn.Conv2d(trunk_channels, policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(policy_channels * 9, 9)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(trunk_channels, value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(value_channels * 9, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.res_tower(h)

        p = self.policy_conv(h)
        p = p.flatten(1)
        p = self.policy_fc(p)

        if legal_mask is not None:
            p = p.masked_fill(~legal_mask, float("-inf"))

        v = self.value_conv(h)
        v = v.flatten(1)
        v = self.value_fc(v)

        return p, v

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        return {
            "trunk_channels": self.trunk_channels,
            "num_res_blocks": self.num_res_blocks,
            "policy_channels": self.policy_channels,
            "value_channels": self.value_channels,
            "value_hidden": self.value_hidden,
        }

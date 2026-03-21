"""Shared residual block for DQN models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidBlock(nn.Module):
    """Pre-activation residual block (no BatchNorm) used in DQN trunks."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(x + h)

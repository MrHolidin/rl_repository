"""DQN network for TicTacToe (3x3, flat action space)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dqn_network import BaseDQNNetwork
from .dueling_utils import dueling_aggregate
from .noisy_layers import NoisyLinear


class TicTacToeDQN(BaseDQNNetwork):
    """
    Small CNN + MLP DQN for TicTacToe.

    Input: (B, in_channels, 3, 3)
    Output: (B, num_actions)  -- flat over all 9 cells
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_actions: int = 9,
        dueling: bool = True,
        hidden: int = 128,
        use_noisy: bool = False,
        noisy_sigma: float = 0.5,
    ):
        super().__init__(num_actions=num_actions, dueling=dueling)
        self.in_channels = in_channels
        self.hidden = hidden
        self.use_noisy = use_noisy
        self.noisy_sigma = noisy_sigma

        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        lkw = {"sigma_init": noisy_sigma} if use_noisy else {}

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        flat_dim = 64 * 3 * 3  # after two convs on 3x3

        if dueling:
            self.adv1 = LinearLayer(flat_dim, hidden, **lkw)
            self.adv2 = LinearLayer(hidden, num_actions, **lkw)
            self.val1 = LinearLayer(flat_dim, hidden, **lkw)
            self.val2 = LinearLayer(hidden, 1, **lkw)
        else:
            self.q1 = LinearLayer(flat_dim, hidden, **lkw)
            self.q2 = LinearLayer(hidden, num_actions, **lkw)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.relu(self.conv1(x))   # (B, 32, 4, 4)
        h = F.relu(self.conv2(h))   # (B, 64, 3, 3)
        h = h.flatten(1)            # (B, 576)

        if self._dueling:
            a = F.relu(self.adv1(h))
            a = self.adv2(a)          # (B, 9)
            v = F.relu(self.val1(h))
            v = self.val2(v)          # (B, 1)
            q = dueling_aggregate(v, a, legal_mask)
        else:
            q = F.relu(self.q1(h))
            q = self.q2(q)

        if legal_mask is not None:
            q = q.masked_fill(~legal_mask.bool(), -1e9)

        return q

    def get_constructor_kwargs(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "num_actions": self._num_actions,
            "dueling": self._dueling,
            "hidden": self.hidden,
            "use_noisy": self.use_noisy,
            "noisy_sigma": self.noisy_sigma,
        }

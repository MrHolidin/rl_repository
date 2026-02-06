"""Othello DQN network with dueling and noisy net support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_dqn_network import BaseDQNNetwork
from .dueling_utils import dueling_aggregate
from .noisy_layers import NoisyLinear, NoisyConv2d


class ResidBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(x + h)


class OthelloDQN(BaseDQNNetwork):
    """
    Othello/Reversi DQN (8x8, 64 actions).

    Input channels:
      - Channel 0: current player's stones
      - Channel 1: opponent's stones
      - Optional CoordConv channels

    Output:
      - Q-values for each cell in row-major order: action = row * 8 + col.

    Architecture:
      - Stem conv -> ResBlocks trunk
      - Dueling: per-cell advantage head + global value head
      - Non-dueling: per-cell Q head
      - Optional noisy layers in heads for exploration
    """

    def __init__(
        self,
        board_size: int = 8,
        in_channels: int = 2,
        num_actions: int = 64,
        dueling: bool = True,
        trunk_channels: int = 96,
        num_res_blocks: int = 3,
        use_coord_channels: bool = True,
        head_hidden: int = 64,
        val_hidden: int = 128,
        use_noisy: bool = False,
        noisy_sigma: float = 0.5,
    ):
        super().__init__(num_actions=num_actions, dueling=dueling)

        assert board_size == 8, "This network is for standard 8x8 Othello."
        assert num_actions == board_size * board_size

        self.board_size = board_size
        self.in_channels = in_channels
        self.trunk_channels = trunk_channels
        self.num_res_blocks = num_res_blocks
        self.use_coord_channels = use_coord_channels
        self.head_hidden = head_hidden
        self.val_hidden = val_hidden
        self.use_noisy = use_noisy
        self.noisy_sigma = noisy_sigma

        extra = 2 if use_coord_channels else 0
        cin = in_channels + extra

        # Trunk (always regular layers)
        self.stem = nn.Conv2d(cin, trunk_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList(
            [ResidBlock(trunk_channels) for _ in range(num_res_blocks)]
        )

        # Head layers: noisy or regular
        Conv2dLayer = NoisyConv2d if use_noisy else nn.Conv2d
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        conv_kwargs = {"sigma_init": noisy_sigma} if use_noisy else {}
        linear_kwargs = {"sigma_init": noisy_sigma} if use_noisy else {}

        if self._dueling:
            # Advantage head: per-cell (B, C, 8, 8) -> (B, 64)
            self.adv1 = Conv2dLayer(trunk_channels, head_hidden, kernel_size=1, **conv_kwargs)
            self.adv2 = Conv2dLayer(head_hidden, 1, kernel_size=1, **conv_kwargs)

            # Value head: global pooled (B, C) -> (B, 1)
            self.val1 = LinearLayer(trunk_channels, val_hidden, **linear_kwargs)
            self.val2 = LinearLayer(val_hidden, 1, **linear_kwargs)
        else:
            # Non-dueling: per-cell Q head
            self.q1 = Conv2dLayer(trunk_channels, head_hidden, kernel_size=1, **conv_kwargs)
            self.q2 = Conv2dLayer(head_hidden, head_hidden, kernel_size=3, padding=1, **conv_kwargs)
            self.q3 = Conv2dLayer(head_hidden, 1, kernel_size=1, **conv_kwargs)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (NoisyLinear, NoisyConv2d)):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def reset_noise(self) -> None:
        """Resample noise in all noisy layers."""
        for m in self.modules():
            if isinstance(m, (NoisyLinear, NoisyConv2d)):
                m.reset_noise()

    def _add_coord(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        rr = torch.linspace(-1, 1, steps=h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        cc = torch.linspace(-1, 1, steps=w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, rr, cc], dim=1)

    def forward(
        self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, 8, 8)
            legal_mask: (B, 64) bool/float mask, True/1 = legal

        Returns:
            Q-values (B, 64) in row-major order
        """
        if self.use_coord_channels:
            x = self._add_coord(x)

        h = F.relu(self.stem(x))
        for blk in self.res_blocks:
            h = blk(h)  # (B, C, 8, 8)

        if self._dueling:
            # Advantage per cell
            a = F.relu(self.adv1(h))
            a = self.adv2(a).flatten(start_dim=1)  # (B, 64)

            # Value scalar
            h_global = h.mean(dim=(2, 3))  # (B, C)
            v = F.relu(self.val1(h_global))
            v = self.val2(v)  # (B, 1)

            q = dueling_aggregate(v, a, legal_mask)
        else:
            q = F.relu(self.q1(h))
            q = F.relu(self.q2(q))
            q = self.q3(q).flatten(start_dim=1)  # (B, 64)

        if legal_mask is not None:
            lm = legal_mask.bool()
            q = q.masked_fill(~lm, -1e9)

        return q

    def get_constructor_kwargs(self) -> dict:
        return {
            "board_size": self.board_size,
            "in_channels": self.in_channels,
            "num_actions": self._num_actions,
            "dueling": self._dueling,
            "trunk_channels": self.trunk_channels,
            "num_res_blocks": self.num_res_blocks,
            "use_coord_channels": self.use_coord_channels,
            "head_hidden": self.head_hidden,
            "val_hidden": self.val_hidden,
            "use_noisy": self.use_noisy,
            "noisy_sigma": self.noisy_sigma,
        }


class OthelloQRDQN(OthelloDQN):
    """
    Quantile Regression DQN for Othello.
    Outputs (B, 64, n_quantiles). Q(s,a) = mean over quantiles.
    """

    def __init__(
        self,
        board_size: int = 8,
        in_channels: int = 2,
        num_actions: int = 64,
        n_quantiles: int = 32,
        dueling: bool = True,
        trunk_channels: int = 96,
        num_res_blocks: int = 3,
        use_coord_channels: bool = True,
        head_hidden: int = 64,
        val_hidden: int = 128,
        use_noisy: bool = False,
        noisy_sigma: float = 0.5,
    ):
        super().__init__(
            board_size=board_size,
            in_channels=in_channels,
            num_actions=num_actions,
            dueling=dueling,
            trunk_channels=trunk_channels,
            num_res_blocks=num_res_blocks,
            use_coord_channels=use_coord_channels,
            head_hidden=head_hidden,
            val_hidden=val_hidden,
            use_noisy=use_noisy,
            noisy_sigma=noisy_sigma,
        )
        self.n_quantiles = n_quantiles

        Conv2dLayer = NoisyConv2d if use_noisy else nn.Conv2d
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        conv_kw = {"sigma_init": noisy_sigma} if use_noisy else {}
        lin_kw = {"sigma_init": noisy_sigma} if use_noisy else {}

        if self._dueling:
            del self.adv2, self.val2
            self.adv2_quantile = Conv2dLayer(head_hidden, n_quantiles, kernel_size=1, **conv_kw)
            self.val2_quantile = LinearLayer(val_hidden, n_quantiles, **lin_kw)
        else:
            del self.q3
            self.q3_quantile = Conv2dLayer(head_hidden, n_quantiles, kernel_size=1, **conv_kw)

    def forward(
        self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_coord_channels:
            x = self._add_coord(x)

        h = F.relu(self.stem(x))
        for blk in self.res_blocks:
            h = blk(h)

        if self._dueling:
            # Advantage: (B, head_hidden, 8, 8) -> (B, n_quantiles, 8, 8) -> (B, 64, n_quantiles)
            a = F.relu(self.adv1(h))
            a = self.adv2_quantile(a)  # (B, n_quantiles, 8, 8)
            a = a.flatten(2).permute(0, 2, 1)  # (B, 64, n_quantiles)

            # Value: (B, C) -> (B, n_quantiles) -> (B, 1, n_quantiles)
            h_global = h.mean(dim=(2, 3))
            v = F.relu(self.val1(h_global))
            v = self.val2_quantile(v).unsqueeze(1)  # (B, 1, n_quantiles)

            quantiles = v + (a - a.mean(dim=1, keepdim=True))
        else:
            q = F.relu(self.q1(h))
            q = F.relu(self.q2(q))
            quantiles = self.q3_quantile(q)  # (B, n_quantiles, 8, 8)
            quantiles = quantiles.flatten(2).permute(0, 2, 1)  # (B, 64, n_quantiles)

        if legal_mask is not None:
            lm = legal_mask.bool().unsqueeze(-1)
            quantiles = quantiles.masked_fill(~lm, -1e9)

        return quantiles

    def get_constructor_kwargs(self) -> dict:
        d = super().get_constructor_kwargs()
        d["n_quantiles"] = self.n_quantiles
        return d


DQN = OthelloDQN

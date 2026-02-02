"""Othello DQN network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_dqn_network import BaseDQNNetwork


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
    Othello/Reversi DQN (8x8, 64 actions, non-dueling).

    Input channels:
      - Channel 0: current player's stones
      - Channel 1: opponent's stones
      - Optional extras via in_channels param.

    Output:
      - Q-values for each cell in row-major order: action = row * 8 + col.

    Architecture:
      - Stem conv -> ResBlocks trunk -> per-cell Q head (Conv2d)
      - Optional CoordConv channels for positional awareness
    """

    def __init__(
        self,
        board_size: int = 8,
        in_channels: int = 2,
        num_actions: int = 64,
        trunk_channels: int = 96,
        num_res_blocks: int = 3,
        use_coord_channels: bool = True,
        head_hidden: int = 64,
    ):
        super().__init__(num_actions=num_actions, dueling=False)

        assert board_size == 8, "This network is for standard 8x8 Othello."
        assert num_actions == board_size * board_size

        self.board_size = board_size
        self.in_channels = in_channels
        self.trunk_channels = trunk_channels
        self.num_res_blocks = num_res_blocks
        self.use_coord_channels = use_coord_channels
        self.head_hidden = head_hidden

        extra = 2 if use_coord_channels else 0
        cin = in_channels + extra

        self.stem = nn.Conv2d(cin, trunk_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList(
            [ResidBlock(trunk_channels) for _ in range(num_res_blocks)]
        )

        self.q1 = nn.Conv2d(trunk_channels, head_hidden, kernel_size=1)
        self.q2 = nn.Conv2d(head_hidden, head_hidden, kernel_size=3, padding=1)
        self.q3 = nn.Conv2d(head_hidden, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
            h = blk(h)

        q = F.relu(self.q1(h))
        q = F.relu(self.q2(q))
        q = self.q3(q)
        q = q.flatten(start_dim=1)  # (B, 64) row-major

        if legal_mask is not None:
            lm = legal_mask.bool()
            q = q.masked_fill(~lm, -1e9)

        return q

    def get_constructor_kwargs(self) -> dict:
        return {
            "board_size": self.board_size,
            "in_channels": self.in_channels,
            "num_actions": self._num_actions,
            "trunk_channels": self.trunk_channels,
            "num_res_blocks": self.num_res_blocks,
            "use_coord_channels": self.use_coord_channels,
            "head_hidden": self.head_hidden,
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
        trunk_channels: int = 96,
        num_res_blocks: int = 3,
        use_coord_channels: bool = True,
        head_hidden: int = 64,
    ):
        super().__init__(
            board_size=board_size,
            in_channels=in_channels,
            num_actions=num_actions,
            trunk_channels=trunk_channels,
            num_res_blocks=num_res_blocks,
            use_coord_channels=use_coord_channels,
            head_hidden=head_hidden,
        )
        self.n_quantiles = n_quantiles
        self.q3 = nn.Conv2d(head_hidden, n_quantiles, kernel_size=1)

    def forward(
        self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_coord_channels:
            x = self._add_coord(x)

        h = F.relu(self.stem(x))
        for blk in self.res_blocks:
            h = blk(h)

        q = F.relu(self.q1(h))
        q = F.relu(self.q2(q))
        quantiles = self.q3(q)  # (B, n_quantiles, 8, 8)
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

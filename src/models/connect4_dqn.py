import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_dqn_network import BaseDQNNetwork
from .dueling_utils import dueling_aggregate
from .noisy_layers import NoisyLinear, NoisyConv1d


class ResidBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(x + h)


class Connect4DQN(BaseDQNNetwork):
    """
    Geometry-aware Connect4 DQN:
    - Trunk outputs (B, C, 6, 7)
    - Advantage head works per-column (action = column):
        pool over rows -> (B, C, 7) -> small Conv1d head -> (B, 7)
    - Value head is global pooled -> small MLP -> (B, 1)
    - Optional CoordConv channels
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        in_channels: int = 2,
        num_actions: int = 7,
        dueling: bool = True,
        trunk_channels: int = 96,
        num_res_blocks: int = 2,
        use_coord_channels: bool = True,
        adv_hidden: int = 64,   # маленькая 1D голова
        val_hidden: int = 128,  # маленькая value MLP
        use_noisy: bool = False,
        noisy_sigma: float = 0.5,
    ):
        super().__init__(num_actions=num_actions, dueling=dueling)

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.trunk_channels = trunk_channels
        self.num_res_blocks = num_res_blocks
        self.use_coord_channels = use_coord_channels
        self.adv_hidden = adv_hidden
        self.val_hidden = val_hidden
        self.use_noisy = use_noisy
        self.noisy_sigma = noisy_sigma

        extra = 2 if use_coord_channels else 0
        cin = in_channels + extra

        # Trunk (always regular layers)
        self.stem = nn.Conv2d(cin, trunk_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResidBlock(trunk_channels) for _ in range(num_res_blocks)])

        # Head layers: noisy or regular depending on use_noisy
        Conv1dLayer = NoisyConv1d if use_noisy else nn.Conv1d
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        conv_kwargs = {"sigma_init": noisy_sigma} if use_noisy else {}
        linear_kwargs = {"sigma_init": noisy_sigma} if use_noisy else {}

        if self._dueling:
            # Advantage head: (B, C, 7) -> (B, 7)
            self.adv1 = Conv1dLayer(trunk_channels, adv_hidden, kernel_size=1, **conv_kwargs)
            self.adv2 = Conv1dLayer(adv_hidden, 1, kernel_size=1, **conv_kwargs)

            # Value head: (B, C) -> (B, 1)
            self.val1 = LinearLayer(trunk_channels, val_hidden, **linear_kwargs)
            self.val2 = LinearLayer(val_hidden, 1, **linear_kwargs)
        else:
            # Non-dueling: still per-column head (B, C, 7) -> (B, 7)
            self.q1 = Conv1dLayer(trunk_channels, adv_hidden, kernel_size=1, **conv_kwargs)
            self.q2 = Conv1dLayer(adv_hidden, 1, kernel_size=1, **conv_kwargs)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            # Skip noisy layers - they have their own initialization
            if isinstance(m, (NoisyLinear, NoisyConv1d)):
                continue
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def reset_noise(self) -> None:
        """Resample noise in all noisy layers."""
        for m in self.modules():
            if isinstance(m, (NoisyLinear, NoisyConv1d)):
                m.reset_noise()

    def _add_coord(self, x: torch.Tensor) -> torch.Tensor:
        b, _, r, c = x.shape
        rr = torch.linspace(-1, 1, steps=r, device=x.device).view(1, 1, r, 1).expand(b, 1, r, c)
        cc = torch.linspace(-1, 1, steps=c, device=x.device).view(1, 1, 1, c).expand(b, 1, r, c)
        return torch.cat([x, rr, cc], dim=1)

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_coord_channels:
            x = self._add_coord(x)

        h = F.relu(self.stem(x))
        for blk in self.res_blocks:
            h = blk(h)  # (B, C, 6, 7)

        # Column pooling: action = column, so keep per-column representation
        h_col = h.mean(dim=2)  # pool over rows -> (B, C, 7)

        if self._dueling:
            # Advantage per column
            a = F.relu(self.adv1(h_col))   # (B, adv_hidden, 7)
            a = self.adv2(a).squeeze(1)    # (B, 7)

            # Value scalar
            h_global = h.mean(dim=(2, 3))  # (B, C)
            v = F.relu(self.val1(h_global))
            v = self.val2(v)              # (B, 1)

            q = dueling_aggregate(v, a, legal_mask)
        else:
            q = F.relu(self.q1(h_col))
            q = self.q2(q).squeeze(1)     # (B, 7)

        # Mask illegal actions with -inf for correct argmax
        if legal_mask is not None:
            q = q.masked_fill(~legal_mask.bool(), -1e9)

        return q

    def get_constructor_kwargs(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "in_channels": self.in_channels,
            "num_actions": self._num_actions,
            "dueling": self._dueling,
            "trunk_channels": self.trunk_channels,
            "num_res_blocks": self.num_res_blocks,
            "use_coord_channels": self.use_coord_channels,
            "adv_hidden": self.adv_hidden,
            "val_hidden": self.val_hidden,
            "use_noisy": self.use_noisy,
            "noisy_sigma": self.noisy_sigma,
        }


class Connect4QRDQN(Connect4DQN):
    """
    Quantile Regression DQN for Connect4.
    Outputs (B, num_actions, n_quantiles) quantile predictions.
    Q(s,a) = mean over quantiles for action selection.
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        in_channels: int = 2,
        num_actions: int = 7,
        n_quantiles: int = 32,
        trunk_channels: int = 96,
        num_res_blocks: int = 2,
        use_coord_channels: bool = True,
        adv_hidden: int = 64,
        val_hidden: int = 128,
        use_noisy: bool = False,
        noisy_sigma: float = 0.5,
    ):
        super().__init__(
            rows=rows,
            cols=cols,
            in_channels=in_channels,
            num_actions=num_actions,
            dueling=False,
            trunk_channels=trunk_channels,
            num_res_blocks=num_res_blocks,
            use_coord_channels=use_coord_channels,
            adv_hidden=adv_hidden,
            val_hidden=val_hidden,
            use_noisy=use_noisy,
            noisy_sigma=noisy_sigma,
        )
        self.n_quantiles = n_quantiles
        # q2_quantile is also noisy if use_noisy
        Conv1dLayer = NoisyConv1d if use_noisy else nn.Conv1d
        conv_kwargs = {"sigma_init": noisy_sigma} if use_noisy else {}
        self.q2_quantile = Conv1dLayer(adv_hidden, n_quantiles, kernel_size=1, **conv_kwargs)

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_coord_channels:
            x = self._add_coord(x)

        h = F.relu(self.stem(x))
        for blk in self.res_blocks:
            h = blk(h)

        h_col = h.mean(dim=2)
        q = F.relu(self.q1(h_col))
        quantiles = self.q2_quantile(q)  # (B, n_quantiles, 7)
        quantiles = quantiles.permute(0, 2, 1)  # (B, 7, n_quantiles)

        if legal_mask is not None:
            lm = legal_mask.bool().unsqueeze(-1)
            quantiles = quantiles.masked_fill(~lm, -1e9)

        return quantiles

    def get_constructor_kwargs(self) -> dict:
        d = super().get_constructor_kwargs()
        d.pop("dueling", None)
        d["n_quantiles"] = self.n_quantiles
        return d


DQN = Connect4DQN

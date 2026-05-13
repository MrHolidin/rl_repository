"""MiniBG DQN torso: slot-wise Conv1d over board/shop/enemy strips + globals."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dqn_network import BaseDQNNetwork
from .dueling_utils import dueling_aggregate
from .noisy_layers import NoisyLinear

from src.envs.minibg.actions import BOARD_SIZE, SHOP_SIZE
from src.envs.minibg.obs import (
    GLOBAL_DIM as _GLOBAL_DIM,
    HAND_LEN as _HAND_LEN,
    LAST_BATTLE_DIM as _LAST_BATTLE_DIM,
    OBS_DIM as _OBS_DIM,
    PHASE_DIM as _PHASE_DIM,
    SLOT_DIM as _SLOT_DIM,
)

_OWN_LEN = BOARD_SIZE
_SHOP_LEN = SHOP_SIZE
_HAND_LEN = _HAND_LEN
_ENEMY_LEN = BOARD_SIZE

_TOTAL_SLOTS = _OWN_LEN + _SHOP_LEN + _HAND_LEN + _ENEMY_LEN


class MiniBGSlotEncoderNet(BaseDQNNetwork):
    """
    Unpacks the MiniBG vector into own board (4×25), shop (3×25), hand (3×25),
    enemy (4×25). Runs a shared Conv1d encoder over the slot axis (kernel=1 then
    optional kernel=3 with padding on ``region_conv2``) and **flattens** per-slot features so slot
    identity is preserved — required so that slot-indexed action heads
    (BUY_SLOT_*, SELL_BOARD_*, PLACE_HAND_*, SELECT_ORDER_*) can read
    "what's in slot i" rather than the regional mean.
    """

    def __init__(
        self,
        num_actions: int,
        *,
        slot_hidden: int = 16,
        trunk_hidden: int = 256,
        dueling: bool = True,
        use_noisy: bool = False,
        noisy_sigma: float = 0.5,
        region_conv2_kernel: int = 1,
    ) -> None:
        super().__init__(num_actions=num_actions, dueling=dueling)
        self.slot_hidden = int(slot_hidden)
        self.trunk_hidden = int(trunk_hidden)
        self.use_noisy = bool(use_noisy)
        self.noisy_sigma = float(noisy_sigma)
        k2 = int(region_conv2_kernel)
        if k2 not in (1, 3):
            raise ValueError("region_conv2_kernel must be 1 or 3")
        self.region_conv2_kernel = k2

        self.region_conv1 = nn.Conv1d(_SLOT_DIM, self.slot_hidden, kernel_size=1)
        if k2 == 3:
            self.region_conv2 = nn.Conv1d(
                self.slot_hidden, self.slot_hidden, kernel_size=3, padding=1
            )
        else:
            self.region_conv2 = nn.Conv1d(
                self.slot_hidden, self.slot_hidden, kernel_size=1
            )

        trunk_in = (
            _TOTAL_SLOTS * self.slot_hidden
            + _GLOBAL_DIM
            + _LAST_BATTLE_DIM
            + _PHASE_DIM
        )
        self.trunk_fc1 = nn.Linear(trunk_in, self.trunk_hidden)
        self.trunk_fc2 = nn.Linear(self.trunk_hidden, self.trunk_hidden)

        HeadLayer = NoisyLinear if self.use_noisy else nn.Linear
        head_kw = {"sigma_init": self.noisy_sigma} if self.use_noisy else {}

        if dueling:
            hv = max(1, self.trunk_hidden // 2)
            self.value_fc = HeadLayer(self.trunk_hidden, hv, **head_kw)
            self.value_out = HeadLayer(hv, 1, **head_kw)
            self.adv_fc = HeadLayer(self.trunk_hidden, hv, **head_kw)
            self.adv_out = HeadLayer(hv, num_actions, **head_kw)
        else:
            self.q_out = HeadLayer(self.trunk_hidden, num_actions, **head_kw)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def _unpack(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.shape[1] != _OBS_DIM:
            raise ValueError(f"Expected obs dim {_OBS_DIM}, got {x.shape[1]}")
        g = x[:, :_GLOBAL_DIM]
        i = _GLOBAL_DIM
        own = x[:, i : i + _OWN_LEN * _SLOT_DIM].view(-1, _OWN_LEN, _SLOT_DIM)
        i += _OWN_LEN * _SLOT_DIM
        shop = x[:, i : i + _SHOP_LEN * _SLOT_DIM].view(-1, _SHOP_LEN, _SLOT_DIM)
        i += _SHOP_LEN * _SLOT_DIM
        hand = x[:, i : i + _HAND_LEN * _SLOT_DIM].view(-1, _HAND_LEN, _SLOT_DIM)
        i += _HAND_LEN * _SLOT_DIM
        enemy = x[:, i : i + _ENEMY_LEN * _SLOT_DIM].view(-1, _ENEMY_LEN, _SLOT_DIM)
        i += _ENEMY_LEN * _SLOT_DIM
        lb = x[:, i : i + _LAST_BATTLE_DIM]
        i += _LAST_BATTLE_DIM
        phase = x[:, i : i + _PHASE_DIM]
        return g, own, shop, hand, enemy, lb, phase

    def _encode_region(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, SLOT_DIM) -> (B, L * slot_hidden); slot identity preserved."""
        h = z.transpose(1, 2)
        h = F.relu(self.region_conv1(h))
        h = F.relu(self.region_conv2(h))
        return h.flatten(1)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g, own, shop, hand, enemy, lb, phase = self._unpack(x)
        e_own = self._encode_region(own)
        e_shop = self._encode_region(shop)
        e_hand = self._encode_region(hand)
        e_enemy = self._encode_region(enemy)
        feat = torch.cat([e_own, e_shop, e_hand, e_enemy, g, lb, phase], dim=1)
        t = F.relu(self.trunk_fc1(feat))
        t = F.relu(self.trunk_fc2(t))

        if self._dueling:
            v = F.relu(self.value_fc(t))
            value = self.value_out(v)
            a = F.relu(self.adv_fc(t))
            advantage = self.adv_out(a)
            return dueling_aggregate(value, advantage, legal_mask)
        return self.q_out(t)

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        return {
            "num_actions": self._num_actions,
            "slot_hidden": self.slot_hidden,
            "trunk_hidden": self.trunk_hidden,
            "dueling": self._dueling,
            "use_noisy": self.use_noisy,
            "noisy_sigma": self.noisy_sigma,
            "region_conv2_kernel": self.region_conv2_kernel,
        }


__all__ = ["MiniBGSlotEncoderNet", "_OBS_DIM"]

"""MiniBG DQN torso: slot-wise Conv1d over board/shop/enemy strips + globals."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dqn_network import BaseDQNNetwork
from .dueling_utils import dueling_aggregate

# Must match src/envs/minibg/obs.py layout (flat concat order).
_GLOBAL_DIM = 10
_SLOT_DIM = 25
_OWN_LEN = 4
_SHOP_LEN = 3
_ENEMY_LEN = 4
_LAST_BATTLE_DIM = 1
_OBS_DIM = (
    _GLOBAL_DIM
    + _OWN_LEN * _SLOT_DIM
    + _SHOP_LEN * _SLOT_DIM
    + _ENEMY_LEN * _SLOT_DIM
    + _LAST_BATTLE_DIM
)


class MiniBGSlotEncoderNet(BaseDQNNetwork):
    """
    Unpacks the 286-d MiniBG vector into own board (4×25), shop (3×25), enemy (4×25),
    runs shared Conv1d over the slot axis, mean-pools per region, concatenates globals,
    then a small MLP + dueling heads.
    """

    def __init__(
        self,
        num_actions: int,
        *,
        slot_hidden: int = 64,
        trunk_hidden: int = 256,
        dueling: bool = True,
    ) -> None:
        super().__init__(num_actions=num_actions, dueling=dueling)
        self.slot_hidden = int(slot_hidden)
        self.trunk_hidden = int(trunk_hidden)

        self.region_conv1 = nn.Conv1d(_SLOT_DIM, self.slot_hidden, kernel_size=1)
        self.region_conv2 = nn.Conv1d(
            self.slot_hidden, self.slot_hidden, kernel_size=3, padding=1
        )

        trunk_in = 3 * self.slot_hidden + _GLOBAL_DIM + _LAST_BATTLE_DIM
        self.trunk_fc1 = nn.Linear(trunk_in, self.trunk_hidden)
        self.trunk_fc2 = nn.Linear(self.trunk_hidden, self.trunk_hidden)

        if dueling:
            hv = max(1, self.trunk_hidden // 2)
            self.value_fc = nn.Linear(self.trunk_hidden, hv)
            self.value_out = nn.Linear(hv, 1)
            self.adv_fc = nn.Linear(self.trunk_hidden, hv)
            self.adv_out = nn.Linear(hv, num_actions)
        else:
            self.q_out = nn.Linear(self.trunk_hidden, num_actions)

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
        enemy = x[:, i : i + _ENEMY_LEN * _SLOT_DIM].view(-1, _ENEMY_LEN, _SLOT_DIM)
        i += _ENEMY_LEN * _SLOT_DIM
        lb = x[:, i : i + _LAST_BATTLE_DIM]
        return g, own, shop, enemy, lb

    def _encode_region(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, SLOT_DIM) -> (B, slot_hidden)"""
        h = z.transpose(1, 2)
        h = F.relu(self.region_conv1(h))
        h = F.relu(self.region_conv2(h))
        return h.mean(dim=2)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g, own, shop, enemy, lb = self._unpack(x)
        e_own = self._encode_region(own)
        e_shop = self._encode_region(shop)
        e_enemy = self._encode_region(enemy)
        feat = torch.cat([e_own, e_shop, e_enemy, g, lb], dim=1)
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
        }


__all__ = ["MiniBGSlotEncoderNet", "_OBS_DIM"]

"""MiniBG actor-critic (PPO): same slot trunk as MiniBGSlotEncoderNet, separate policy and value heads."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Layout must match src/envs/minibg/obs.py and minibg_slot_net.py.
_GLOBAL_DIM = 10
_SLOT_DIM = 25
_OWN_LEN = 4
_SHOP_LEN = 3
_HAND_LEN = 3
_ENEMY_LEN = 4
_LAST_BATTLE_DIM = 1
_PHASE_DIM = 1
_OBS_DIM = (
    _GLOBAL_DIM
    + _OWN_LEN * _SLOT_DIM
    + _SHOP_LEN * _SLOT_DIM
    + _HAND_LEN * _SLOT_DIM
    + _ENEMY_LEN * _SLOT_DIM
    + _LAST_BATTLE_DIM
    + _PHASE_DIM
)
_TOTAL_SLOTS = _OWN_LEN + _SHOP_LEN + _HAND_LEN + _ENEMY_LEN


class MiniBGSlotActorCritic(nn.Module):
    """Shared slot encoder; policy logits and scalar V(s)."""

    def __init__(
        self,
        num_actions: int,
        *,
        slot_hidden: int = 16,
        trunk_hidden: int = 256,
        region_conv2_kernel: int = 1,
    ) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.slot_hidden = int(slot_hidden)
        self.trunk_hidden = int(trunk_hidden)
        k2 = int(region_conv2_kernel)
        if k2 not in (1, 3):
            raise ValueError("region_conv2_kernel must be 1 or 3")
        self._region_conv2_kernel = k2

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
        self.policy_head = nn.Linear(self.trunk_hidden, self.num_actions)
        self.value_head = nn.Linear(self.trunk_hidden, 1)

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        return {
            "num_actions": self.num_actions,
            "slot_hidden": self.slot_hidden,
            "trunk_hidden": self.trunk_hidden,
            "region_conv2_kernel": self._region_conv2_kernel,
        }

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
        h = z.transpose(1, 2)
        h = F.relu(self.region_conv1(h))
        h = F.relu(self.region_conv2(h))
        return h.flatten(1)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g, own, shop, hand, enemy, lb, phase = self._unpack(x)
        e_own = self._encode_region(own)
        e_shop = self._encode_region(shop)
        e_hand = self._encode_region(hand)
        e_enemy = self._encode_region(enemy)
        feat = torch.cat([e_own, e_shop, e_hand, e_enemy, g, lb, phase], dim=1)
        t = F.relu(self.trunk_fc1(feat))
        t = F.relu(self.trunk_fc2(t))

        logits = self.policy_head(t)
        values = self.value_head(t).squeeze(-1)

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))

        return logits, values


__all__ = ["MiniBGSlotActorCritic", "_OBS_DIM"]

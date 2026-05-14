"""MiniBG actor-critic (PPO): slot trunk with card embedding, separate policy/value heads."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.minibg.actions import BOARD_SIZE, MAX_SHOP_SLOTS
from src.envs.minibg.obs import (
    CARD_IDX_OFFSET as _CARD_IDX_OFFSET,
    GLOBAL_DIM as _GLOBAL_DIM,
    HAND_LEN as _HAND_LEN,
    LAST_BATTLE_DIM as _LAST_BATTLE_DIM,
    NUM_POOL_INDICES as _NUM_POOL_INDICES,
    OBS_DIM as _OBS_DIM,
    PENDING_CHOICE_DIM as _PENDING_CHOICE_DIM,
    PENDING_DISCOVER_IDX_DIM as _PENDING_DISCOVER_IDX_DIM,
    PENDING_DISCOVER_IDX_OFFSET as _PENDING_DISCOVER_IDX_OFFSET,
    PHASE_DIM as _PHASE_DIM,
    SLOT_DIM as _SLOT_DIM,
    PENDING_HEADER_OFFSET as _PENDING_HEADER_OFFSET,
    PENDING_OPTIONS_OFFSET as _PENDING_OPTIONS_OFFSET,
)
from src.envs.minibg.discover_pool import ADAPT_KEYS_ALL

_OWN_LEN = BOARD_SIZE
_SHOP_LEN = MAX_SHOP_SLOTS
_HAND_LEN = _HAND_LEN
_ENEMY_LEN = BOARD_SIZE
_TOTAL_SLOTS = _OWN_LEN + _SHOP_LEN + _HAND_LEN + _ENEMY_LEN
_SLOT_CONT_DIM = _SLOT_DIM - 1
_PENDING_CONT_DIM = _PENDING_CHOICE_DIM - _PENDING_DISCOVER_IDX_DIM
_PENDING_IS_ADAPT_CH = _PENDING_HEADER_OFFSET + 1


def _pending_three_option_emb(
    pending: torch.Tensor,
    card_emb: nn.Embedding,
    adapt_emb: nn.Embedding,
) -> torch.Tensor:
    """``(B, 3, D)`` per-modal option vectors; mirrors ``encode_pending_choice``."""

    is_adapt = pending[..., _PENDING_IS_ADAPT_CH] > 0.5
    opt_scaled = pending[
        ...,
        _PENDING_OPTIONS_OFFSET : _PENDING_OPTIONS_OFFSET + _PENDING_DISCOVER_IDX_DIM,
    ]
    adapt_idx_raw = (opt_scaled * 9.0).round().long().clamp(0, len(ADAPT_KEYS_ALL) - 1)
    disc_idx = pending[
        ...,
        _PENDING_DISCOVER_IDX_OFFSET : _PENDING_DISCOVER_IDX_OFFSET + _PENDING_DISCOVER_IDX_DIM,
    ].long().clamp_(min=0, max=_NUM_POOL_INDICES)
    ae = adapt_emb(adapt_idx_raw + 1)
    de = card_emb(disc_idx)
    mask = is_adapt.unsqueeze(-1).unsqueeze(-1).expand_as(de)
    return torch.where(mask, ae, de)


def _encode_pending_with_card_emb(
    pending: torch.Tensor,
    card_emb: nn.Embedding,
    adapt_emb: nn.Embedding,
) -> torch.Tensor:
    """Flatten pending + per-option embeddings (discover vs ADAPT‑key table)."""

    cont = pending[..., :_PENDING_DISCOVER_IDX_OFFSET]
    emb_stack = _pending_three_option_emb(pending, card_emb, adapt_emb)
    return torch.cat([cont, emb_stack.flatten(-2)], dim=-1)


def _split_card_idx_and_cont(
    z: torch.Tensor, card_emb: nn.Embedding
) -> torch.Tensor:
    """``(B, L, SLOT_DIM)`` → ``(B, L, (SLOT_DIM-1) + card_emb_dim)`` by replacing card_idx channel with its embedding."""
    cid_f = z[..., _CARD_IDX_OFFSET]
    cid_long = cid_f.long().clamp_(min=0, max=_NUM_POOL_INDICES)
    emb = card_emb(cid_long)
    cont = torch.cat(
        [
            z[..., :_CARD_IDX_OFFSET],
            z[..., _CARD_IDX_OFFSET + 1 :],
        ],
        dim=-1,
    )
    return torch.cat([cont, emb], dim=-1)


class MiniBGSlotActorCritic(nn.Module):
    """Shared slot encoder; policy logits and scalar V(s)."""

    def __init__(
        self,
        num_actions: int,
        *,
        slot_hidden: int = 32,
        trunk_hidden: int = 256,
        region_conv2_kernel: int = 1,
        card_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.slot_hidden = int(slot_hidden)
        self.trunk_hidden = int(trunk_hidden)
        self.card_emb_dim = int(card_emb_dim)
        k2 = int(region_conv2_kernel)
        if k2 not in (1, 3):
            raise ValueError("region_conv2_kernel must be 1 or 3")
        self._region_conv2_kernel = k2

        self.card_emb = nn.Embedding(
            _NUM_POOL_INDICES + 1, self.card_emb_dim, padding_idx=0
        )
        self.adapt_choice_emb = nn.Embedding(
            len(ADAPT_KEYS_ALL) + 1, self.card_emb_dim, padding_idx=0
        )
        # Smaller-than-default init: 102 random rows at N(0, 1) into a Conv1d alongside
        # bounded continuous features causes large early-step grads in region_conv1.
        nn.init.normal_(self.card_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.card_emb.weight[0].zero_()
        nn.init.normal_(self.adapt_choice_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.adapt_choice_emb.weight[0].zero_()

        conv_in = _SLOT_CONT_DIM + self.card_emb_dim
        self.region_conv1 = nn.Conv1d(conv_in, self.slot_hidden, kernel_size=1)
        if k2 == 3:
            self.region_conv2 = nn.Conv1d(
                self.slot_hidden, self.slot_hidden, kernel_size=3, padding=1
            )
        else:
            self.region_conv2 = nn.Conv1d(
                self.slot_hidden, self.slot_hidden, kernel_size=1
            )

        pending_feat_dim = _PENDING_CONT_DIM + _PENDING_DISCOVER_IDX_DIM * self.card_emb_dim
        trunk_in = (
            _TOTAL_SLOTS * self.slot_hidden
            + _GLOBAL_DIM
            + _LAST_BATTLE_DIM
            + _PHASE_DIM
            + pending_feat_dim
        )
        self._pending_feat_dim = pending_feat_dim
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
            "card_emb_dim": self.card_emb_dim,
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
        i += _PHASE_DIM
        pending = x[:, i : i + _PENDING_CHOICE_DIM]
        return g, own, shop, hand, enemy, lb, phase, pending

    def _encode_region(self, z: torch.Tensor) -> torch.Tensor:
        z = _split_card_idx_and_cont(z, self.card_emb)
        h = z.transpose(1, 2)
        h = F.relu(self.region_conv1(h))
        h = F.relu(self.region_conv2(h))
        return h.flatten(1)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g, own, shop, hand, enemy, lb, phase, pending = self._unpack(x)
        e_own = self._encode_region(own)
        e_shop = self._encode_region(shop)
        e_hand = self._encode_region(hand)
        e_enemy = self._encode_region(enemy)
        pending_feat = _encode_pending_with_card_emb(
            pending, self.card_emb, self.adapt_choice_emb
        )
        feat = torch.cat(
            [e_own, e_shop, e_hand, e_enemy, g, lb, phase, pending_feat], dim=1
        )
        t = F.relu(self.trunk_fc1(feat))
        t = F.relu(self.trunk_fc2(t))

        logits = self.policy_head(t)
        values = self.value_head(t).squeeze(-1)

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))

        return logits, values


__all__ = [
    "MiniBGSlotActorCritic",
    "_OBS_DIM",
    "_PENDING_CHOICE_DIM",
    "_PENDING_CONT_DIM",
    "_PENDING_DISCOVER_IDX_DIM",
    "_split_card_idx_and_cont",
    "_encode_pending_with_card_emb",
    "_pending_three_option_emb",
]

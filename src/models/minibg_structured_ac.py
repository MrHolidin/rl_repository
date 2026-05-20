"""MiniBG actor-critic: structured legal actions + optional autoregressive board-order head."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.envs.minibg.actions import BOARD_SIZE
from src.envs.minibg.structured_actions import StructAction, StructActionType

from src.bg_recruitment.discover_pool import ADAPT_KEYS_ALL

from src.envs.minibg.obs import (
    EFFECT_OFFSET as _EFFECT_OFFSET,
    NUM_EFFECT_CHANNELS as _NUM_EFFECT_CHANNELS,
    NUM_POOL_INDICES as _NUM_POOL_INDICES,
    NUM_TRIGGER_CHANNELS as _NUM_TRIGGER_CHANNELS,
    TRIGGER_OFFSET as _TRIGGER_OFFSET,
)

from src.envs.minibg.obs import (
    PENDING_CHOICE_DIM as _PENDING_CHOICE_DIM,
    PENDING_DISCOVER_IDX_OFFSET as _PENDING_DISCOVER_IDX_OFFSET,
    PENDING_IS_APPLY_OFFSET as _PENDING_IS_APPLY_OFFSET,
)

from .minibg_slot_ac import (
    _ENEMY_LEN,
    _GLOBAL_DIM,
    _HAND_LEN,
    _LAST_BATTLE_DIM,
    _OBS_DIM,
    _OWN_LEN,
    _PENDING_CHOICE_DIM,
    _PENDING_CONT_DIM,
    _PENDING_DISCOVER_IDX_DIM,
    _PHASE_DIM,
    _SHOP_LEN,
    _SLOT_CONT_DIM,
    _SLOT_DIM,
    _TOTAL_SLOTS,
    _pending_three_option_emb,
    _split_card_idx_and_cont,
)

_EXT_DIM = _NUM_TRIGGER_CHANNELS + _NUM_EFFECT_CHANNELS
_EXT_END = _EFFECT_OFFSET + _NUM_EFFECT_CHANNELS

_NUM_STRUCT_TYPES = len(StructActionType)
_ROLE_NONE = 0
_ROLE_SHOP = 1
_ROLE_BOARD = 2
_ROLE_HAND = 3
_ROLE_PENDING = 4
_NUM_ROLES = 5

# Region "kinds" used by the batched action embedding gather. Index 0 is reserved
# for null-entity actions (ROLL/LEVEL_UP/COMPLETE_TURN) so padded slots also map there.
# PENDING covers discover/adapt option indices [0..2].
_REGION_NULL = 0
_REGION_SHOP = 1
_REGION_OWN = 2
_REGION_HAND = 3
_REGION_PENDING = 4
_NUM_REGIONS = 5
_PENDING_LEN = 3
_MAX_REGION_LEN = max(_OWN_LEN, _SHOP_LEN, _HAND_LEN, _PENDING_LEN, 1)

# Globals (gold, round, healths, actions_left, ...) + last_battle + phase routed
# explicitly into action / order scoring heads so the actor stops ignoring them.
_GLOBALS_FULL_DIM = _GLOBAL_DIM + _LAST_BATTLE_DIM + _PHASE_DIM

# Indices into ``slot_region_emb`` (additive post-conv geometry). Not the same as
# ``_REGION_*`` action-token region kinds used by the legal-action gather.
REG_OWN = 0
REG_SHOP = 1
REG_HAND = 2
REG_ENEMY = 3
REG_PENDING = 4


class EntityAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 4,
        ff_mult: int = 2,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.ln_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.ln_ff = nn.LayerNorm(dim)
        ff_dim = ff_mult * dim
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

        self.attn_scale = nn.Parameter(torch.full((dim,), float(init_scale)))
        self.ff_scale = nn.Parameter(torch.full((dim,), float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_attn(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a * self.attn_scale.view(1, 1, -1)

        h = self.ln_ff(x)
        x = x + self.ff(h) * self.ff_scale.view(1, 1, -1)

        return x


def role_for_struct(a: StructAction) -> int:
    t = a.type
    if t == StructActionType.BUY:
        return _ROLE_SHOP
    if t == StructActionType.SELL:
        return _ROLE_BOARD
    if t == StructActionType.PLACE or t == StructActionType.MAGNET:
        return _ROLE_HAND
    if t == StructActionType.DISCOVER_PICK:
        return _ROLE_PENDING
    if t == StructActionType.APPLY_EFFECT:
        return _ROLE_BOARD
    if t == StructActionType.APPLY_EFFECT_SKIP:
        return _ROLE_BOARD
    return _ROLE_NONE


def _struct_action_codes(a: StructAction) -> Tuple[int, int, int, int, int, int]:
    """Return ``(type_id, role_id, src_region, src_slot, tgt_region, tgt_slot)``.

    Two-entity tokens so MAGNET carries both source (hand) and target (board), and
    DISCOVER_PICK carries the picked pending option as its source entity.
    """
    t = a.type
    if t == StructActionType.BUY:
        return (int(t), _ROLE_SHOP, _REGION_SHOP, int(a.args[0]), _REGION_NULL, 0)
    if t == StructActionType.SELL:
        return (int(t), _ROLE_BOARD, _REGION_OWN, int(a.args[0]), _REGION_NULL, 0)
    if t == StructActionType.PLACE:
        return (int(t), _ROLE_HAND, _REGION_HAND, int(a.args[0]), _REGION_NULL, 0)
    if t == StructActionType.MAGNET:
        return (
            int(t),
            _ROLE_HAND,
            _REGION_HAND,
            int(a.args[0]),
            _REGION_OWN,
            int(a.args[1]),
        )
    if t == StructActionType.DISCOVER_PICK:
        return (int(t), _ROLE_PENDING, _REGION_PENDING, int(a.args[0]), _REGION_NULL, 0)
    if t == StructActionType.APPLY_EFFECT:
        return (
            int(t),
            _ROLE_BOARD,
            _REGION_NULL,
            0,
            _REGION_OWN,
            int(a.args[0]),
        )
    if t == StructActionType.APPLY_EFFECT_SKIP:
        return (int(t), _ROLE_BOARD, _REGION_NULL, 0, _REGION_NULL, 0)
    # ROLL / LEVEL_UP / COMPLETE_TURN: null entity on both sides.
    return (int(t), _ROLE_NONE, _REGION_NULL, 0, _REGION_NULL, 0)


def _build_action_tokens(
    legal_actions: Sequence[Sequence[StructAction]],
    Lmax: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Pack (B, Lmax) int64 token arrays + bool mask. Padded slots stay at NULL/0.

    Returns (type_ids, role_ids, src_region_kinds, src_region_slots,
    tgt_region_kinds, tgt_region_slots, mask).
    """
    B = len(legal_actions)
    type_ids = np.zeros((B, Lmax), dtype=np.int64)
    role_ids = np.zeros((B, Lmax), dtype=np.int64)
    src_region_kinds = np.zeros((B, Lmax), dtype=np.int64)
    src_region_slots = np.zeros((B, Lmax), dtype=np.int64)
    tgt_region_kinds = np.zeros((B, Lmax), dtype=np.int64)
    tgt_region_slots = np.zeros((B, Lmax), dtype=np.int64)
    mask = np.zeros((B, Lmax), dtype=bool)
    for b in range(B):
        row = legal_actions[b]
        L = len(row)
        if L == 0:
            continue
        for l in range(L):
            t, r, sk, ss, tk, ts = _struct_action_codes(row[l])
            type_ids[b, l] = t
            role_ids[b, l] = r
            src_region_kinds[b, l] = sk
            src_region_slots[b, l] = ss
            tgt_region_kinds[b, l] = tk
            tgt_region_slots[b, l] = ts
        mask[b, :L] = True
    return (
        type_ids,
        role_ids,
        src_region_kinds,
        src_region_slots,
        tgt_region_kinds,
        tgt_region_slots,
        mask,
    )


class MiniBGStructuredActorCritic(nn.Module):
    """
    Shared slot encoder → state embedding; post-conv additive position + region
    embeddings per tensor (own/shop/hand/enemy/pending) for action/order heads.
    Policy logits over variable legal sets via concat(state, action_emb, state×action,
    globals); optional autoregressive board-order head. Value head reads trunk ``t``;
    actor/policy sides still use ``state_proj(t)``.
    """

    def __init__(
        self,
        *,
        slot_hidden: int = 32,
        trunk_hidden: int = 256,
        state_dim: int = 128,
        action_dim: int = 64,
        interaction_dim: int = 64,
        order_hidden: int = 64,
        order_pos_dim: int = 16,
        score_hidden: int = 128,
        order_score_hidden: int = 64,
        critic_hidden: int = 128,
        region_conv2_kernel: int = 1,
        card_emb_dim: int = 16,
        entity_attention_layers: int = 0,
        entity_attention_heads: int = 4,
        entity_attention_ff_mult: int = 2,
        entity_attention_init_scale: float = 0.1,
        use_global_entity_token: bool = True,
    ) -> None:
        super().__init__()
        self.slot_hidden = int(slot_hidden)
        self.trunk_hidden = int(trunk_hidden)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.interaction_dim = int(interaction_dim)
        self.order_hidden = int(order_hidden)
        self.order_pos_dim = int(order_pos_dim)
        self.score_hidden = int(score_hidden)
        self.order_score_hidden = int(order_score_hidden)
        self.critic_hidden = int(critic_hidden)
        self.card_emb_dim = int(card_emb_dim)
        self.entity_attention_layers = int(entity_attention_layers)
        self.entity_attention_heads = int(entity_attention_heads)
        self.entity_attention_ff_mult = int(entity_attention_ff_mult)
        self.entity_attention_init_scale = float(entity_attention_init_scale)
        self.use_global_entity_token = bool(use_global_entity_token)

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
        # Tight init (matches slot AC): keeps Conv1d activations bounded on step 0
        # despite 102 random rows fanning into 50 continuous channels.
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

        self.own_pos_emb = nn.Embedding(_OWN_LEN, self.slot_hidden)
        self.shop_pos_emb = nn.Embedding(_SHOP_LEN, self.slot_hidden)
        self.hand_pos_emb = nn.Embedding(_HAND_LEN, self.slot_hidden)
        self.enemy_pos_emb = nn.Embedding(_ENEMY_LEN, self.slot_hidden)
        self.pending_pos_emb = nn.Embedding(_PENDING_LEN, self.slot_hidden)
        self.slot_region_emb = nn.Embedding(5, self.slot_hidden)
        for emb in (
            self.own_pos_emb,
            self.shop_pos_emb,
            self.hand_pos_emb,
            self.enemy_pos_emb,
            self.pending_pos_emb,
            self.slot_region_emb,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        if self.entity_attention_layers < 0:
            raise ValueError("entity_attention_layers must be >= 0")

        if self.entity_attention_layers > 0:
            self.entity_attn = nn.ModuleList(
                [
                    EntityAttentionBlock(
                        self.slot_hidden,
                        num_heads=self.entity_attention_heads,
                        ff_mult=self.entity_attention_ff_mult,
                        init_scale=self.entity_attention_init_scale,
                    )
                    for _ in range(self.entity_attention_layers)
                ]
            )
        else:
            self.entity_attn = nn.ModuleList()

        if self.entity_attention_layers > 0 and self.use_global_entity_token:
            self.global_to_entity_token = nn.Linear(_GLOBALS_FULL_DIM, self.slot_hidden)
        else:
            self.global_to_entity_token = None

        self._pending_feat_dim = (
            _PENDING_CHOICE_DIM + _PENDING_DISCOVER_IDX_DIM * self.card_emb_dim
        )
        trunk_in = (
            _TOTAL_SLOTS * self.slot_hidden
            + _GLOBAL_DIM
            + _LAST_BATTLE_DIM
            + _PHASE_DIM
            + self._pending_feat_dim
        )
        self.trunk_fc1 = nn.Linear(trunk_in, self.trunk_hidden)
        self.trunk_fc2 = nn.Linear(self.trunk_hidden, self.trunk_hidden)

        # Drop the redundant pooled-mean stream: it sits in span(trunk) already.
        self.state_proj = nn.Linear(self.trunk_hidden, self.state_dim)

        self.type_emb = nn.Embedding(_NUM_STRUCT_TYPES, self.action_dim)
        self.role_emb = nn.Embedding(_NUM_ROLES, self.action_dim)
        # Source entity projection (used by BUY/SELL/PLACE/MAGNET/DISCOVER_PICK).
        self.entity_to_action = nn.Linear(self.slot_hidden, self.action_dim)
        # Target entity projection — only fires for MAGNET (board target). Separate
        # weights from source so the model doesn't have to learn an arbitrary tag
        # to distinguish "the hand source" from "the board target" inside one Linear.
        self.entity_to_action_tgt = nn.Linear(self.slot_hidden, self.action_dim)
        # Direct bypass for trigger/effect bits — avoids the slot_hidden bottleneck
        # so action scoring distinguishes Brann/Khadgar/Baron via the effect channels
        # without depending on the conv head retaining that information.
        self.ent_extras = nn.Linear(_EXT_DIM, self.action_dim)
        self.ent_extras_tgt = nn.Linear(_EXT_DIM, self.action_dim)
        # Discover/adapt option entity: pending tail only has card_idx — project the
        # shared card embedding into slot_hidden so the standard E_regions gather
        # path covers PENDING the same way it covers SHOP/OWN/HAND.
        self.pending_to_slot = nn.Linear(self.card_emb_dim, self.slot_hidden)
        self.null_entity_action = nn.Parameter(torch.zeros(self.action_dim))

        # Pairwise state×action interaction (Hadamard after shared projections) so logits
        # are not purely additive in ``state_emb`` vs ``action_emb``.
        self.state_to_interact = nn.Linear(self.state_dim, self.interaction_dim, bias=False)
        self.action_to_interact = nn.Linear(self.action_dim, self.interaction_dim, bias=False)

        # Non-linear scoring head with explicit globals routing.
        self.score_fc = nn.Sequential(
            nn.Linear(
                self.state_dim + self.action_dim + self.interaction_dim + _GLOBALS_FULL_DIM,
                self.score_hidden,
            ),
            nn.ReLU(),
            nn.Linear(self.score_hidden, 1),
        )

        # V(s) from trunk ``t`` — avoids inflicting the actor's state_dim bottleneck on value.
        self.critic = nn.Sequential(
            nn.LayerNorm(self.trunk_hidden),
            nn.Linear(self.trunk_hidden, self.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.critic_hidden, 1),
        )

        self.order_pos_emb = nn.Embedding(BOARD_SIZE, self.order_pos_dim)
        self.order_start = nn.Parameter(torch.zeros(self.slot_hidden))
        # Initialize GRU hidden from state_emb so the first pick already sees board context.
        self.order_init = nn.Linear(self.state_dim, self.order_hidden)
        gru_in = self.slot_hidden + self.order_pos_dim
        self.order_gru = nn.GRUCell(gru_in, self.order_hidden)
        # Same additivity fix for the order ranking head.
        order_score_in = (
            self.state_dim
            + self.slot_hidden
            + self.order_hidden
            + self.order_pos_dim
            + _GLOBALS_FULL_DIM
        )
        self.order_score = nn.Sequential(
            nn.Linear(order_score_in, self.order_score_hidden),
            nn.ReLU(),
            nn.Linear(self.order_score_hidden, 1),
        )

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        return {
            "slot_hidden": self.slot_hidden,
            "trunk_hidden": self.trunk_hidden,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "interaction_dim": self.interaction_dim,
            "order_hidden": self.order_hidden,
            "order_pos_dim": self.order_pos_dim,
            "score_hidden": self.score_hidden,
            "order_score_hidden": self.order_score_hidden,
            "critic_hidden": self.critic_hidden,
            "region_conv2_kernel": self._region_conv2_kernel,
            "card_emb_dim": self.card_emb_dim,
            "entity_attention_layers": self.entity_attention_layers,
            "entity_attention_heads": self.entity_attention_heads,
            "entity_attention_ff_mult": self.entity_attention_ff_mult,
            "entity_attention_init_scale": self.entity_attention_init_scale,
            "use_global_entity_token": self.use_global_entity_token,
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

    def _encode_region_slots(self, z: torch.Tensor) -> torch.Tensor:
        z = _split_card_idx_and_cont(z, self.card_emb)
        h = z.transpose(1, 2)
        h = F.relu(self.region_conv1(h))
        h = F.relu(self.region_conv2(h))
        return h.transpose(1, 2).contiguous()

    def _add_pos_region(
        self,
        E: torch.Tensor,
        pos_emb: nn.Embedding,
        region_id: int,
    ) -> torch.Tensor:
        _, L, H = E.shape
        device = E.device
        pos_ids = torch.arange(L, device=device, dtype=torch.long)
        pe = pos_emb(pos_ids).unsqueeze(0)
        re = self.slot_region_emb(
            torch.tensor(region_id, device=device, dtype=torch.long)
        ).view(1, 1, H)
        return E + pe + re

    def _contextualize_entities(
        self,
        E_own: torch.Tensor,
        E_shop: torch.Tensor,
        E_hand: torch.Tensor,
        E_enemy: torch.Tensor,
        E_pending: torch.Tensor,
        g_full: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.entity_attn) == 0:
            return E_own, E_shop, E_hand, E_enemy, E_pending

        pieces = [E_own, E_shop, E_hand, E_enemy, E_pending]

        if self.global_to_entity_token is not None:
            g_tok = self.global_to_entity_token(g_full).unsqueeze(1)
            E_all = torch.cat([g_tok] + pieces, dim=1)
            offset = 1
        else:
            E_all = torch.cat(pieces, dim=1)
            offset = 0

        for block in self.entity_attn:
            E_all = block(E_all)

        if offset:
            E_all = E_all[:, offset:, :]

        i = 0
        E_own = E_all[:, i : i + _OWN_LEN]
        i += _OWN_LEN

        E_shop = E_all[:, i : i + _SHOP_LEN]
        i += _SHOP_LEN

        E_hand = E_all[:, i : i + _HAND_LEN]
        i += _HAND_LEN

        E_enemy = E_all[:, i : i + _ENEMY_LEN]
        i += _ENEMY_LEN

        E_pending = E_all[:, i : i + _PENDING_LEN]

        return E_own, E_shop, E_hand, E_enemy, E_pending

    def encode_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        g, own, shop, hand, enemy, lb, phase, pending = self._unpack(x)
        E_own = self._add_pos_region(
            self._encode_region_slots(own), self.own_pos_emb, REG_OWN
        )
        E_shop = self._add_pos_region(
            self._encode_region_slots(shop), self.shop_pos_emb, REG_SHOP
        )
        E_hand = self._add_pos_region(
            self._encode_region_slots(hand), self.hand_pos_emb, REG_HAND
        )
        E_enemy = self._add_pos_region(
            self._encode_region_slots(enemy), self.enemy_pos_emb, REG_ENEMY
        )

        # Raw trigger+effect bits per slot — fed directly into action embedding so
        # multipliers/auras keep an identity-strong signal even with slot_hidden compression.
        EXT_own = own[..., _TRIGGER_OFFSET:_EXT_END]
        EXT_shop = shop[..., _TRIGGER_OFFSET:_EXT_END]
        EXT_hand = hand[..., _TRIGGER_OFFSET:_EXT_END]

        # Pending entities (3 discover or ADAPT options). Indices route through ``card_emb``
        # or dedicated ``adapt_choice_emb`` matching ``encode_pending_choice``.
        B = x.size(0)
        device = x.device
        dtype = E_own.dtype
        cont = pending[..., :_PENDING_CHOICE_DIM]
        opt_stack = _pending_three_option_emb(
            pending, self.card_emb, self.adapt_choice_emb
        )
        is_apply = pending[..., _PENDING_IS_APPLY_OFFSET : _PENDING_IS_APPLY_OFFSET + 1] > 0.5
        opt_stack = opt_stack.masked_fill(is_apply.unsqueeze(-1), 0.0)
        pending_feat = torch.cat([cont, opt_stack.flatten(-2)], dim=-1)
        E_pending = self._add_pos_region(
            self.pending_to_slot(opt_stack), self.pending_pos_emb, REG_PENDING
        )
        EXT_pending = torch.zeros(B, _PENDING_LEN, _EXT_DIM, device=device, dtype=dtype)

        g_full = torch.cat([g, lb, phase], dim=-1)
        E_own, E_shop, E_hand, E_enemy, E_pending = self._contextualize_entities(
            E_own,
            E_shop,
            E_hand,
            E_enemy,
            E_pending,
            g_full,
        )

        feat_flat = torch.cat(
            [
                E_own.flatten(1),
                E_shop.flatten(1),
                E_hand.flatten(1),
                E_enemy.flatten(1),
                g,
                lb,
                phase,
                pending_feat,
            ],
            dim=1,
        )
        t = F.relu(self.trunk_fc2(F.relu(self.trunk_fc1(feat_flat))))

        state_emb = self.state_proj(t)
        cache = {
            "E_own": E_own,
            "E_shop": E_shop,
            "E_hand": E_hand,
            "E_enemy": E_enemy,
            "E_pending": E_pending,
            "EXT_own": EXT_own,
            "EXT_shop": EXT_shop,
            "EXT_hand": EXT_hand,
            "EXT_pending": EXT_pending,
            "trunk": t,
            "g_full": g_full,
            "pending_feat": pending_feat,
        }
        return state_emb, cache

    def _encode_actions(
        self,
        type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        src_region_kinds: torch.Tensor,
        src_region_slots: torch.Tensor,
        tgt_region_kinds: torch.Tensor,
        tgt_region_slots: torch.Tensor,
        cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Batched action embedding for ``(B, Lmax)`` token tensors → ``(B, Lmax, action_dim)``.

        Source entity covers BUY/SELL/PLACE shopper, MAGNET's hand card, DISCOVER_PICK's
        pending option. Target entity covers MAGNET's board target (everything else: NULL).
        """
        B, Lmax = type_ids.shape
        device = type_ids.device
        dtype = cache["E_own"].dtype

        type_e = self.type_emb(type_ids)
        role_e = self.role_emb(role_ids)

        # Pack region features in a single padded tensor so one gather covers all rows/actions.
        # Region 0 is the null region (zeros); padded slots map here and get overwritten by
        # null masking below, so their exact value does not matter.
        E_regions = torch.zeros(
            B,
            _NUM_REGIONS,
            _MAX_REGION_LEN,
            self.slot_hidden,
            device=device,
            dtype=dtype,
        )
        E_regions[:, _REGION_SHOP, :_SHOP_LEN] = cache["E_shop"]
        E_regions[:, _REGION_OWN, :_OWN_LEN] = cache["E_own"]
        E_regions[:, _REGION_HAND, :_HAND_LEN] = cache["E_hand"]
        E_regions[:, _REGION_PENDING, :_PENDING_LEN] = cache["E_pending"]
        E_regions_flat = E_regions.view(B, _NUM_REGIONS * _MAX_REGION_LEN, self.slot_hidden)

        EXT_regions = torch.zeros(
            B,
            _NUM_REGIONS,
            _MAX_REGION_LEN,
            _EXT_DIM,
            device=device,
            dtype=dtype,
        )
        EXT_regions[:, _REGION_SHOP, :_SHOP_LEN] = cache["EXT_shop"]
        EXT_regions[:, _REGION_OWN, :_OWN_LEN] = cache["EXT_own"]
        EXT_regions[:, _REGION_HAND, :_HAND_LEN] = cache["EXT_hand"]
        EXT_regions[:, _REGION_PENDING, :_PENDING_LEN] = cache["EXT_pending"]
        EXT_regions_flat = EXT_regions.view(B, _NUM_REGIONS * _MAX_REGION_LEN, _EXT_DIM)

        def _gather_entity(
            region_kinds: torch.Tensor,
            region_slots: torch.Tensor,
            ent_lin: nn.Linear,
            ext_lin: nn.Linear,
        ) -> torch.Tensor:
            flat_idx = region_kinds * _MAX_REGION_LEN + region_slots
            ent_idx = flat_idx.unsqueeze(-1).expand(-1, -1, self.slot_hidden)
            ext_idx = flat_idx.unsqueeze(-1).expand(-1, -1, _EXT_DIM)
            ent_slot = torch.gather(E_regions_flat, dim=1, index=ent_idx)
            ext_slot = torch.gather(EXT_regions_flat, dim=1, index=ext_idx)
            ent_from_slot = ent_lin(ent_slot) + ext_lin(ext_slot)
            # Null-region tokens (e.g. padded slots, ROLL/LEVEL_UP/COMPLETE_TURN sources,
            # tgt of all single-entity actions) must zero out — otherwise the Linear bias
            # term leaks into the action embedding for every "empty" entity slot.
            is_null = (region_kinds == _REGION_NULL).unsqueeze(-1)
            return ent_from_slot.masked_fill(is_null, 0.0)

        src_ent = _gather_entity(
            src_region_kinds, src_region_slots, self.entity_to_action, self.ent_extras
        )
        tgt_ent = _gather_entity(
            tgt_region_kinds,
            tgt_region_slots,
            self.entity_to_action_tgt,
            self.ent_extras_tgt,
        )

        # Tokens whose source AND target are both NULL (ROLL/LEVEL_UP/COMPLETE_TURN
        # plus padding) get the null entity parameter so the network can still learn a
        # type-conditioned bias for "non-entity" actions.
        ent_null = self.null_entity_action.view(1, 1, -1).expand(B, Lmax, -1)
        both_null = (
            (src_region_kinds == _REGION_NULL) & (tgt_region_kinds == _REGION_NULL)
        ).unsqueeze(-1)
        ent = torch.where(both_null, ent_null, src_ent + tgt_ent)

        return type_e + role_e + ent

    def _logits_from_state_and_tokens(
        self,
        state_emb: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        src_region_kinds: torch.Tensor,
        src_region_slots: torch.Tensor,
        tgt_region_kinds: torch.Tensor,
        tgt_region_slots: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, Lmax = type_ids.shape
        ae = self._encode_actions(
            type_ids,
            role_ids,
            src_region_kinds,
            src_region_slots,
            tgt_region_kinds,
            tgt_region_slots,
            cache,
        )
        s_exp = state_emb.unsqueeze(1).expand(-1, Lmax, -1)
        g_exp = cache["g_full"].unsqueeze(1).expand(-1, Lmax, -1)

        s_int = torch.tanh(self.state_to_interact(state_emb))
        a_int = torch.tanh(self.action_to_interact(ae))
        interaction = s_int.unsqueeze(1) * a_int

        h_all = torch.cat([s_exp, ae, interaction, g_exp], dim=-1)
        logits = self.score_fc(h_all.reshape(B * Lmax, -1)).squeeze(-1).view(B, Lmax)
        return logits.masked_fill(~mask, float("-inf"))

    def policy_logits_value_from_tokens(
        self,
        obs: torch.Tensor,
        type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        src_region_kinds: torch.Tensor,
        src_region_slots: torch.Tensor,
        tgt_region_kinds: torch.Tensor,
        tgt_region_slots: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """Same shape contract as :meth:`policy_logits_and_value`, but skips token build (hot path)."""
        state_emb, cache = self.encode_state(obs)
        logits = self._logits_from_state_and_tokens(
            state_emb,
            cache,
            type_ids,
            role_ids,
            src_region_kinds,
            src_region_slots,
            tgt_region_kinds,
            tgt_region_slots,
            mask,
        )
        values = self.critic(cache["trunk"]).squeeze(-1)
        if return_cache:
            cache_out = dict(cache)
            cache_out["state_emb"] = state_emb
            return logits, mask, values, cache_out
        return logits, mask, values

    def policy_logits_and_value(
        self,
        obs: torch.Tensor,
        legal_actions: List[List[StructAction]],
        *,
        return_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """
        Returns:
            logits: (B, max_L) with -inf padding
            mask: (B, max_L) True for valid slots
            values: (B,)
        """
        B = obs.size(0)
        device = obs.device
        if len(legal_actions) != B:
            raise ValueError(
                f"legal_actions has {len(legal_actions)} rows but obs batch size is {B}"
            )
        Lmax = max((len(row) for row in legal_actions), default=0)
        if Lmax == 0:
            raise ValueError(
                "policy_logits_and_value: every legal_actions row is empty — invalid for structured policy."
            )

        (
            t_np,
            r_np,
            src_k_np,
            src_s_np,
            tgt_k_np,
            tgt_s_np,
            m_np,
        ) = _build_action_tokens(legal_actions, Lmax)
        type_ids = torch.from_numpy(t_np).to(device, non_blocking=True)
        role_ids = torch.from_numpy(r_np).to(device, non_blocking=True)
        src_region_kinds = torch.from_numpy(src_k_np).to(device, non_blocking=True)
        src_region_slots = torch.from_numpy(src_s_np).to(device, non_blocking=True)
        tgt_region_kinds = torch.from_numpy(tgt_k_np).to(device, non_blocking=True)
        tgt_region_slots = torch.from_numpy(tgt_s_np).to(device, non_blocking=True)
        mask = torch.from_numpy(m_np).to(device, non_blocking=True)

        return self.policy_logits_value_from_tokens(
            obs,
            type_ids,
            role_ids,
            src_region_kinds,
            src_region_slots,
            tgt_region_kinds,
            tgt_region_slots,
            mask,
            return_cache=return_cache,
        )

    def order_logits_step(
        self,
        state_emb: torch.Tensor,
        E_own: torch.Tensor,
        g_full: torch.Tensor,
        hidden: torch.Tensor,
        pos_e_row: torch.Tensor,
    ) -> torch.Tensor:
        """Logits over BOARD_SIZE slots: (B, BOARD_SIZE). ``pos_e_row`` is the (B, order_pos_dim) row."""
        B, K, _ = E_own.shape
        sh = hidden.unsqueeze(1).expand(-1, K, -1)
        pe = pos_e_row.unsqueeze(1).expand(-1, K, -1)
        se = state_emb.unsqueeze(1).expand(-1, K, -1)
        gx = g_full.unsqueeze(1).expand(-1, K, -1)
        h_cat = torch.cat([se, E_own, sh, pe, gx], dim=-1)
        return self.order_score(h_cat).squeeze(-1)

    def _order_init_hidden(self, state_emb: torch.Tensor) -> torch.Tensor:
        """Initial GRU hidden from state — gives the pointer head board context at pos=0."""
        return torch.tanh(self.order_init(state_emb))

    def sample_board_order(
        self,
        state_emb: torch.Tensor,
        E_own: torch.Tensor,
        g_full: torch.Tensor,
        occupied_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressive pointer over occupied board slots.

        Args:
            state_emb: (B, state_dim)
            E_own: (B, BOARD_SIZE, slot_hidden) — use first BOARD_SIZE own-board slots from encoder
            occupied_mask: (B, BOARD_SIZE) bool, True if board slot index currently holds a minion

        Returns:
            picked_slots: (B, BOARD_SIZE) int64, -1 padding after all picks done
            logprob_sum: (B,)
            remaining_after: (B, BOARD_SIZE) bool — should be all False unless input had >BOARD_SIZE True
        """
        B, K, _ = E_own.shape
        if K != BOARD_SIZE:
            raise ValueError(f"E_own must have BOARD_SIZE={BOARD_SIZE} slots, got {K}")
        device = E_own.device
        remaining = occupied_mask.clone()
        hidden = self._order_init_hidden(state_emb)
        slot_in = self.order_start.unsqueeze(0).expand(B, -1)

        logprob_sum = torch.zeros(B, device=device, dtype=E_own.dtype)
        picked = torch.full((B, BOARD_SIZE), -1, device=device, dtype=torch.long)
        batch_arange = torch.arange(B, device=device)

        # Precompute positional embeddings for all steps in one lookup (no per-step host→device sync).
        pos_table = self.order_pos_emb(torch.arange(BOARD_SIZE, device=device))

        # Fixed-length loop over BOARD_SIZE: cheap (<=4) and removes both `.item()` and `.any()`
        # CPU syncs that would otherwise stall every COMPLETE_TURN call.
        for pos in range(BOARD_SIZE):
            active = remaining.any(dim=1)
            pos_e_row = pos_table[pos].unsqueeze(0).expand(B, -1)

            gru_in = torch.cat([slot_in, pos_e_row], dim=-1)
            hidden_new = self.order_gru(gru_in, hidden)
            hidden = torch.where(active.unsqueeze(-1), hidden_new, hidden)

            logits = self.order_logits_step(state_emb, E_own, g_full, hidden, pos_e_row)
            logits = logits.masked_fill(~remaining, float("-inf"))
            # Fully-finished rows have all -inf — keep Categorical numerically valid.
            logits = logits.masked_fill((~active).unsqueeze(-1), 0.0)

            dist = Categorical(logits=logits)
            if deterministic:
                idx = logits.argmax(dim=-1)
            else:
                idx = dist.sample()

            lp = dist.log_prob(idx)
            logprob_sum = logprob_sum + active.to(logprob_sum.dtype) * lp

            picked[batch_arange, pos] = torch.where(active, idx, torch.full_like(idx, -1))

            sel_emb = E_own[batch_arange, idx]
            slot_in = torch.where(active.unsqueeze(-1), sel_emb, slot_in)

            mask_pick = F.one_hot(idx, num_classes=K).bool() & active.unsqueeze(-1)
            remaining = remaining & ~mask_pick

        return picked, logprob_sum, remaining

    def order_logprob_given_sequence(
        self,
        state_emb: torch.Tensor,
        E_own: torch.Tensor,
        g_full: torch.Tensor,
        occupied_mask: torch.Tensor,
        picked_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced log prob sum for PPO/training (B,). picked_slots (B, max_steps) indices or -1."""
        B, K, _ = E_own.shape
        device = E_own.device
        remaining = occupied_mask.clone()
        hidden = self._order_init_hidden(state_emb)
        slot_in = self.order_start.unsqueeze(0).expand(B, -1)
        logprob_sum = torch.zeros(B, device=device, dtype=E_own.dtype)
        batch_arange = torch.arange(B, device=device)

        max_steps = int(picked_slots.size(1))
        # Pre-build positional embeddings; saves a host→device sync per step.
        pos_table = self.order_pos_emb(torch.arange(max(max_steps, 1), device=device))

        for pos in range(max_steps):
            active = remaining.any(dim=1)
            idx = picked_slots[:, pos]
            valid_pick = active & (idx >= 0)

            pos_e_row = pos_table[pos].unsqueeze(0).expand(B, -1)
            gru_in = torch.cat([slot_in, pos_e_row], dim=-1)
            hidden_new = self.order_gru(gru_in, hidden)
            # Advance GRU only when the teacher row has a real pick at this step. Advancing while
            # active but idx==-1 desynchronizes teacher replay and can explode hidden → NaN logits.
            hidden = torch.where(valid_pick.unsqueeze(-1), hidden_new, hidden)

            logits = self.order_logits_step(state_emb, E_own, g_full, hidden, pos_e_row)
            logits = logits.masked_fill(~remaining, float("-inf"))
            # Two failure modes need 0-filled logits to keep Categorical valid: finished rows
            # (~active) and padding gap rows (active & idx<0). `~valid_pick` covers both.
            logits = logits.masked_fill((~valid_pick).unsqueeze(-1), 0.0)

            dist = Categorical(logits=logits)
            idx_eval = idx.clamp(min=0)
            lp = dist.log_prob(idx_eval)
            logprob_sum = logprob_sum + valid_pick.to(logprob_sum.dtype) * lp

            sel_emb = E_own[batch_arange, idx_eval]
            slot_in = torch.where(valid_pick.unsqueeze(-1), sel_emb, slot_in)

            mask_pick = F.one_hot(idx_eval, num_classes=K).bool() & valid_pick.unsqueeze(-1)
            remaining = remaining & ~mask_pick

        return logprob_sum


__all__ = ["MiniBGStructuredActorCritic", "_OBS_DIM"]


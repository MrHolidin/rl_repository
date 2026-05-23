"""Structured actor-critic, v2.

Differences from v1 ([minibg_structured_ac.py](src/models/minibg_structured_ac.py)):

  - **Trunk killed.** v1 flattens every entity embedding and runs the result
    through a ~200K-parameter ``Linear(827 -> 256)`` followed by another
    ``Linear(256 -> 256)``. That FC stack repeats work the entity attention
    already did. v2 uses a learnable **[CLS] token** that joins the entity
    sequence, attends with all entities, and its output IS the state summary —
    the standard pattern from AlphaStar, BERT, ViT.
  - **Wider per-card representation.** ``slot_hidden`` default is 48 (vs 32).
    32 dims were a bottleneck for 12 trigger channels + 15 effect classes +
    tribes + keywords + stats; the v1 ``ent_extras`` bypass exists precisely
    because of this.
  - **Deeper entity reasoning.** ``entity_attention_layers`` default is 2 (vs 1).
    One attention layer = 1-hop synergy; multi-card-multi-effect interactions
    need ≥2 hops.

Net effect (rough):
  - Parameters: ~450K (v1) → ~220K (v2). Half the weights, more reasoning.
  - Forward FLOPs: roughly the same (FC trunk removed, one extra attention
    layer added; both small compared to the (B × Lmax) action-scoring head).

Cache contract is the same as v1 (see ``StructuredActorCriticProtocol``):
``E_own``/``E_shop``/``E_hand``/``E_pending``/``E_enemy``, their ``EXT_*``
counterparts, ``g_full``, ``trunk`` (the input to the critic), ``state_emb``.

Shared building blocks (``EntityAttentionBlock``, action token helpers, region
and role constants) are imported from v1's module so we don't duplicate them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.bg_recruitment.discover_pool import ADAPT_KEYS_ALL
from src.envs.minibg.actions import BOARD_SIZE
from src.envs.minibg.structured_actions import (
    StructAction,
    StructActionType,
    slot_pick_sequence_to_perm,
)

# Reuse v1's frozen building blocks. These do not depend on the v1 class's layer
# layout — just module-level helpers and a generic attention block.
from .minibg_structured_ac import (
    EntityAttentionBlock,
    _build_action_tokens,
    _EXT_DIM,
    _EXT_END,
    _GLOBALS_FULL_DIM,
    _MAX_REGION_LEN,
    _NUM_REGIONS,
    _NUM_ROLES,
    _NUM_STRUCT_TYPES,
    _PENDING_LEN,
    _REGION_HAND,
    _REGION_NULL,
    _REGION_OWN,
    _REGION_PENDING,
    _REGION_SHOP,
    REG_HAND,
    REG_OWN,
    REG_PENDING,
    REG_SHOP,
)
from .minibg_slot_ac import (
    _GLOBAL_DIM,
    _HAND_LEN,
    _LAST_BATTLE_DIM,
    _OBS_DIM,
    _OWN_LEN,
    _PHASE_DIM,
    _SHOP_LEN,
    _SLOT_CONT_DIM,
    _SLOT_DIM,
    _pending_three_option_emb,
    _split_card_idx_and_cont,
)
from src.envs.minibg.obs import (
    NUM_POOL_INDICES as _NUM_POOL_INDICES,
    PENDING_CHOICE_DIM as _PENDING_CHOICE_DIM,
    PENDING_DISCOVER_IDX_OFFSET as _PENDING_DISCOVER_IDX_OFFSET,
    PENDING_IS_APPLY_OFFSET as _PENDING_IS_APPLY_OFFSET,
    TRIGGER_OFFSET as _TRIGGER_OFFSET,
)


class BGLikeStructuredV2(nn.Module):
    """v2 of the structured actor-critic.

    Implements the same ``StructuredActorCriticProtocol`` surface as v1 (the
    structured PPO agent talks to either via duck-typing), but with a smaller,
    cleaner internal layout.
    """

    def __init__(
        self,
        *,
        slot_hidden: int = 48,
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
        entity_attention_layers: int = 2,
        entity_attention_heads: int = 4,
        entity_attention_ff_mult: int = 2,
        entity_attention_init_scale: float = 0.1,
        obs_layout: str = "bglike",
        num_pool_indices: Optional[int] = None,
    ) -> None:
        super().__init__()

        layout = obs_layout.strip().lower()
        if layout == "bglike":
            from src.envs.bglike.actions import (
                BOARD_SIZE as _layout_board,
                HAND_SIZE as _layout_hand,
                MAX_SHOP_SLOTS as _layout_shop,
            )
            from src.envs.bglike.obs import OBS_DIM as _layout_obs_dim

            self.obs_layout = "bglike"
            self.obs_dim = int(_layout_obs_dim)
            self.own_len = int(_layout_board)
            self.hand_len = int(_layout_hand)
            self.shop_len = int(_layout_shop)
            self.enemy_len = 0
            self.board_size = int(_layout_board)
        elif layout == "minibg":
            self.obs_layout = "minibg"
            self.obs_dim = _OBS_DIM
            self.own_len = _OWN_LEN
            self.hand_len = _HAND_LEN
            self.shop_len = _SHOP_LEN
            self.enemy_len = 0  # v2 still does not consume enemy boards
            self.board_size = BOARD_SIZE
        else:
            raise ValueError(f"obs_layout must be 'minibg' or 'bglike', got {obs_layout!r}")
        self.pending_len = _PENDING_LEN
        self.entity_slots = (
            self.own_len + self.shop_len + self.hand_len + self.enemy_len
        )
        self.total_slots = self.entity_slots
        self.max_region_len = max(
            self.own_len, self.shop_len, self.hand_len, self.pending_len, 1
        )

        if entity_attention_layers < 1:
            raise ValueError("v2 requires entity_attention_layers >= 1 (CLS-token aggregation needs attention)")

        self.slot_hidden = int(slot_hidden)
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
        if num_pool_indices is None:
            raise ValueError("num_pool_indices is required")
        self.num_pool_indices = int(num_pool_indices)

        k2 = int(region_conv2_kernel)
        if k2 not in (1, 3):
            raise ValueError("region_conv2_kernel must be 1 or 3")
        self._region_conv2_kernel = k2

        # --- Card / pending embeddings -----------------------------------
        self.card_emb = nn.Embedding(
            self.num_pool_indices + 1, self.card_emb_dim, padding_idx=0
        )
        self.adapt_choice_emb = nn.Embedding(
            len(ADAPT_KEYS_ALL) + 1, self.card_emb_dim, padding_idx=0
        )
        nn.init.normal_(self.card_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.card_emb.weight[0].zero_()
        nn.init.normal_(self.adapt_choice_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.adapt_choice_emb.weight[0].zero_()

        # --- Slot encoder (per-region conv stack on continuous features) --
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

        # Region + position embeddings
        self.own_pos_emb = nn.Embedding(self.own_len, self.slot_hidden)
        self.shop_pos_emb = nn.Embedding(self.shop_len, self.slot_hidden)
        self.hand_pos_emb = nn.Embedding(self.hand_len, self.slot_hidden)
        self.pending_pos_emb = nn.Embedding(self.pending_len, self.slot_hidden)
        # 4 regions in v2 (no enemy). Index unused-by-v2-but-reserved keeps shape parity.
        self.slot_region_emb = nn.Embedding(5, self.slot_hidden)
        for emb in (
            self.own_pos_emb,
            self.shop_pos_emb,
            self.hand_pos_emb,
            self.pending_pos_emb,
            self.slot_region_emb,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        # --- Entity attention with a [CLS] token from globals ------------
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
        # CLS token is built from globals (gold/hp/round/...) so by the time the
        # attention runs, the [CLS] slot already carries non-card context. Its
        # post-attention output is the state summary that replaces v1's trunk.
        self.cls_from_globals = nn.Linear(_GLOBALS_FULL_DIM, self.slot_hidden)

        self.pending_to_slot = nn.Linear(self.card_emb_dim, self.slot_hidden)

        # --- State summary -> state_emb (no flatten trunk!) --------------
        # Header info that didn't go into the CLS token via globals: the
        # pending-choice header (has_pending / is_adapt / extras_after).
        self._pending_header_dim = 3  # has_pending, is_adapt, extras_modals_after
        state_summary_dim = (
            self.slot_hidden                # CLS token output
            + _GLOBAL_DIM
            + _LAST_BATTLE_DIM
            + _PHASE_DIM
            + self._pending_header_dim
        )
        self._state_summary_dim = int(state_summary_dim)
        self.state_summary_ln = nn.LayerNorm(state_summary_dim)
        self.state_proj = nn.Linear(state_summary_dim, self.state_dim)

        # --- Action embedding (same shape contract as v1) ---------------
        self.type_emb = nn.Embedding(_NUM_STRUCT_TYPES, self.action_dim)
        self.role_emb = nn.Embedding(_NUM_ROLES, self.action_dim)
        self.entity_to_action = nn.Linear(self.slot_hidden, self.action_dim)
        self.entity_to_action_tgt = nn.Linear(self.slot_hidden, self.action_dim)
        self.ent_extras = nn.Linear(_EXT_DIM, self.action_dim)
        self.ent_extras_tgt = nn.Linear(_EXT_DIM, self.action_dim)
        self.null_entity_action = nn.Parameter(torch.zeros(self.action_dim))

        self.state_to_interact = nn.Linear(self.state_dim, self.interaction_dim, bias=False)
        self.action_to_interact = nn.Linear(self.action_dim, self.interaction_dim, bias=False)

        self.score_fc = nn.Sequential(
            nn.Linear(
                self.state_dim + self.action_dim + self.interaction_dim + _GLOBALS_FULL_DIM,
                self.score_hidden,
            ),
            nn.ReLU(),
            nn.Linear(self.score_hidden, 1),
        )

        # --- Critic reads the state-summary (post-LN) -------------------
        self.critic = nn.Sequential(
            nn.Linear(state_summary_dim, self.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.critic_hidden, 1),
        )

        # --- Board-order head (same GRU-based pointer as v1) ------------
        self.order_pos_emb = nn.Embedding(self.board_size, self.order_pos_dim)
        self.order_start = nn.Parameter(torch.zeros(self.slot_hidden))
        self.order_init = nn.Linear(self.state_dim, self.order_hidden)
        gru_in = self.slot_hidden + self.order_pos_dim
        self.order_gru = nn.GRUCell(gru_in, self.order_hidden)
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
            "obs_layout": self.obs_layout,
            "num_pool_indices": self.num_pool_indices,
        }

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _unpack(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs dim {self.obs_dim}, got {x.shape[1]}")
        g = x[:, :_GLOBAL_DIM]
        i = _GLOBAL_DIM
        own = x[:, i : i + self.own_len * _SLOT_DIM].view(-1, self.own_len, _SLOT_DIM)
        i += self.own_len * _SLOT_DIM
        shop = x[:, i : i + self.shop_len * _SLOT_DIM].view(-1, self.shop_len, _SLOT_DIM)
        i += self.shop_len * _SLOT_DIM
        hand = x[:, i : i + self.hand_len * _SLOT_DIM].view(-1, self.hand_len, _SLOT_DIM)
        i += self.hand_len * _SLOT_DIM
        # enemy region: only present in minibg V1 layout; v2 skips it (enemy_len=0)
        if self.enemy_len > 0:
            enemy = x[:, i : i + self.enemy_len * _SLOT_DIM].view(
                -1, self.enemy_len, _SLOT_DIM
            )
            i += self.enemy_len * _SLOT_DIM
        else:
            enemy = x.new_zeros(x.size(0), 0, _SLOT_DIM)
        lb = x[:, i : i + _LAST_BATTLE_DIM]
        i += _LAST_BATTLE_DIM
        phase = x[:, i : i + _PHASE_DIM]
        i += _PHASE_DIM
        pending = x[:, i : i + _PENDING_CHOICE_DIM]
        return g, own, shop, hand, enemy, lb, phase, pending

    def _encode_region_slots(self, z: torch.Tensor) -> torch.Tensor:
        z = _split_card_idx_and_cont(
            z, self.card_emb, max_card_idx=self.num_pool_indices
        )
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

    def encode_state(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        g, own, shop, hand, _enemy, lb, phase, pending = self._unpack(x)

        # Per-region slot encoding + position/region embeddings.
        E_own = self._add_pos_region(
            self._encode_region_slots(own), self.own_pos_emb, REG_OWN
        )
        E_shop = self._add_pos_region(
            self._encode_region_slots(shop), self.shop_pos_emb, REG_SHOP
        )
        E_hand = self._add_pos_region(
            self._encode_region_slots(hand), self.hand_pos_emb, REG_HAND
        )

        # Raw trigger+effect bits — bypass for action scoring (matches v1).
        EXT_own = own[..., _TRIGGER_OFFSET:_EXT_END]
        EXT_shop = shop[..., _TRIGGER_OFFSET:_EXT_END]
        EXT_hand = hand[..., _TRIGGER_OFFSET:_EXT_END]

        # Pending: 3 discover / adapt options.
        B = x.size(0)
        device = x.device
        dtype = E_own.dtype
        opt_stack = _pending_three_option_emb(
            pending,
            self.card_emb,
            self.adapt_choice_emb,
            max_card_idx=self.num_pool_indices,
        )
        is_apply = pending[..., _PENDING_IS_APPLY_OFFSET : _PENDING_IS_APPLY_OFFSET + 1] > 0.5
        opt_stack = opt_stack.masked_fill(is_apply.unsqueeze(-1), 0.0)
        E_pending = self._add_pos_region(
            self.pending_to_slot(opt_stack), self.pending_pos_emb, REG_PENDING
        )
        EXT_pending = torch.zeros(B, self.pending_len, _EXT_DIM, device=device, dtype=dtype)

        # Pending header (3 dims: has_pending / is_adapt / extras_modals_after) —
        # carry into the state summary directly (not blurred by attention).
        pending_header = pending[..., :self._pending_header_dim]

        g_full = torch.cat([g, lb, phase], dim=-1)  # (B, _GLOBALS_FULL_DIM)

        # Prepend [CLS] (built from globals) and run entity attention. The
        # post-attention CLS output IS the v2 state summary.
        cls_tok = self.cls_from_globals(g_full).unsqueeze(1)  # (B, 1, slot_hidden)
        E_all = torch.cat([cls_tok, E_own, E_shop, E_hand, E_pending], dim=1)
        for block in self.entity_attn:
            E_all = block(E_all)

        cls_out = E_all[:, 0]  # (B, slot_hidden)
        idx = 1
        E_own = E_all[:, idx : idx + self.own_len]
        idx += self.own_len
        E_shop = E_all[:, idx : idx + self.shop_len]
        idx += self.shop_len
        E_hand = E_all[:, idx : idx + self.hand_len]
        idx += self.hand_len
        E_pending = E_all[:, idx : idx + self.pending_len]
        E_enemy = E_all.new_zeros(B, 0, self.slot_hidden)

        state_summary = torch.cat(
            [cls_out, g, lb, phase, pending_header], dim=-1
        )  # (B, state_summary_dim)
        state_summary_n = self.state_summary_ln(state_summary)
        state_emb = self.state_proj(state_summary_n)

        cache: Dict[str, torch.Tensor] = {
            "E_own": E_own,
            "E_shop": E_shop,
            "E_hand": E_hand,
            "E_enemy": E_enemy,
            "E_pending": E_pending,
            "EXT_own": EXT_own,
            "EXT_shop": EXT_shop,
            "EXT_hand": EXT_hand,
            "EXT_pending": EXT_pending,
            # Critic reads `trunk`. In v2, trunk == post-LN state summary.
            "trunk": state_summary_n,
            "g_full": g_full,
        }
        return state_emb, cache

    # ------------------------------------------------------------------
    # Action scoring (same contract as v1; logic mirrors v1._encode_actions)
    # ------------------------------------------------------------------

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
        B, Lmax = type_ids.shape
        device = type_ids.device
        dtype = cache["E_own"].dtype

        type_e = self.type_emb(type_ids)
        role_e = self.role_emb(role_ids)

        E_regions = torch.zeros(
            B,
            _NUM_REGIONS,
            self.max_region_len,
            self.slot_hidden,
            device=device,
            dtype=dtype,
        )
        E_regions[:, _REGION_SHOP, : self.shop_len] = cache["E_shop"]
        E_regions[:, _REGION_OWN, : self.own_len] = cache["E_own"]
        E_regions[:, _REGION_HAND, : self.hand_len] = cache["E_hand"]
        E_regions[:, _REGION_PENDING, : self.pending_len] = cache["E_pending"]
        E_regions_flat = E_regions.view(
            B, _NUM_REGIONS * self.max_region_len, self.slot_hidden
        )

        EXT_regions = torch.zeros(
            B,
            _NUM_REGIONS,
            self.max_region_len,
            _EXT_DIM,
            device=device,
            dtype=dtype,
        )
        EXT_regions[:, _REGION_SHOP, : self.shop_len] = cache["EXT_shop"]
        EXT_regions[:, _REGION_OWN, : self.own_len] = cache["EXT_own"]
        EXT_regions[:, _REGION_HAND, : self.hand_len] = cache["EXT_hand"]
        EXT_regions[:, _REGION_PENDING, : self.pending_len] = cache["EXT_pending"]
        EXT_regions_flat = EXT_regions.view(
            B, _NUM_REGIONS * self.max_region_len, _EXT_DIM
        )

        def _gather_entity(
            region_kinds: torch.Tensor,
            region_slots: torch.Tensor,
            ent_lin: nn.Linear,
            ext_lin: nn.Linear,
        ) -> torch.Tensor:
            flat_idx = region_kinds * self.max_region_len + region_slots
            ent_idx = flat_idx.unsqueeze(-1).expand(-1, -1, self.slot_hidden)
            ext_idx = flat_idx.unsqueeze(-1).expand(-1, -1, _EXT_DIM)
            ent_slot = torch.gather(E_regions_flat, dim=1, index=ent_idx)
            ext_slot = torch.gather(EXT_regions_flat, dim=1, index=ext_idx)
            ent_from_slot = ent_lin(ent_slot) + ext_lin(ext_slot)
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
        B = obs.size(0)
        if len(legal_actions) != B:
            raise ValueError(
                f"legal_actions has {len(legal_actions)} rows but obs batch size is {B}"
            )
        Lmax = max((len(row) for row in legal_actions), default=0)
        if Lmax == 0:
            raise ValueError("policy_logits_and_value: every legal_actions row is empty")

        (
            t_np,
            r_np,
            src_k_np,
            src_s_np,
            tgt_k_np,
            tgt_s_np,
            m_np,
        ) = _build_action_tokens(legal_actions, Lmax)
        device = obs.device
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

    # ------------------------------------------------------------------
    # Board-order pointer (identical math to v1)
    # ------------------------------------------------------------------

    def _order_logits_step(
        self,
        state_emb: torch.Tensor,
        E_own: torch.Tensor,
        g_full: torch.Tensor,
        hidden: torch.Tensor,
        pos_e_row: torch.Tensor,
    ) -> torch.Tensor:
        B, K, _ = E_own.shape
        sh = hidden.unsqueeze(1).expand(-1, K, -1)
        pe = pos_e_row.unsqueeze(1).expand(-1, K, -1)
        se = state_emb.unsqueeze(1).expand(-1, K, -1)
        gx = g_full.unsqueeze(1).expand(-1, K, -1)
        h_cat = torch.cat([se, E_own, sh, pe, gx], dim=-1)
        return self.order_score(h_cat).squeeze(-1)

    def _order_init_hidden(self, state_emb: torch.Tensor) -> torch.Tensor:
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
        B, K, _ = E_own.shape
        if K != self.board_size:
            raise ValueError(
                f"E_own must have board_size={self.board_size} slots, got {K}"
            )
        device = E_own.device
        remaining = occupied_mask.clone()
        hidden = self._order_init_hidden(state_emb)
        slot_in = self.order_start.unsqueeze(0).expand(B, -1)

        logprob_sum = torch.zeros(B, device=device, dtype=E_own.dtype)
        picked = torch.full((B, self.board_size), -1, device=device, dtype=torch.long)
        batch_arange = torch.arange(B, device=device)

        pos_table = self.order_pos_emb(torch.arange(self.board_size, device=device))

        for pos in range(self.board_size):
            active = remaining.any(dim=1)
            pos_e_row = pos_table[pos].unsqueeze(0).expand(B, -1)

            gru_in = torch.cat([slot_in, pos_e_row], dim=-1)
            hidden_new = self.order_gru(gru_in, hidden)
            hidden = torch.where(active.unsqueeze(-1), hidden_new, hidden)

            logits = self._order_logits_step(state_emb, E_own, g_full, hidden, pos_e_row)
            logits = logits.masked_fill(~remaining, float("-inf"))
            logits = logits.masked_fill((~active).unsqueeze(-1), 0.0)

            dist = Categorical(logits=logits)
            if deterministic:
                idx = logits.argmax(dim=-1)
            else:
                idx = dist.sample()

            lp = dist.log_prob(idx)
            logprob_sum = logprob_sum + active.to(logprob_sum.dtype) * lp

            picked[batch_arange, pos] = torch.where(
                active, idx, torch.full_like(idx, -1)
            )

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
        B, K, _ = E_own.shape
        device = E_own.device
        remaining = occupied_mask.clone()
        hidden = self._order_init_hidden(state_emb)
        slot_in = self.order_start.unsqueeze(0).expand(B, -1)
        logprob_sum = torch.zeros(B, device=device, dtype=E_own.dtype)
        batch_arange = torch.arange(B, device=device)

        max_steps = int(picked_slots.size(1))
        pos_table = self.order_pos_emb(torch.arange(max(max_steps, 1), device=device))

        for pos in range(max_steps):
            active = remaining.any(dim=1)
            idx = picked_slots[:, pos]
            valid_pick = active & (idx >= 0)

            pos_e_row = pos_table[pos].unsqueeze(0).expand(B, -1)
            gru_in = torch.cat([slot_in, pos_e_row], dim=-1)
            hidden_new = self.order_gru(gru_in, hidden)
            hidden = torch.where(valid_pick.unsqueeze(-1), hidden_new, hidden)

            logits = self._order_logits_step(state_emb, E_own, g_full, hidden, pos_e_row)
            logits = logits.masked_fill(~remaining, float("-inf"))
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


__all__ = ["BGLikeStructuredV2"]

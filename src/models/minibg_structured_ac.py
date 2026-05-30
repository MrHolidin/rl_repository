"""MiniBG actor-critic: structured legal actions + optional autoregressive board-order head.

FROZEN — v1 of the structured architecture. Do **not** modify the layer layout or
forward signatures: this would silently break loading of existing v1 checkpoints
(state_dict keys must remain stable). For architectural changes, create a new
file (e.g. ``bglike_structured_v2.py``) with a fresh class and register it under
a new ``ppo_network_type`` id. The shared building blocks (``EntityAttentionBlock``,
``role_for_struct``, ``_struct_action_codes``, ``_build_action_tokens``, region/role
constants) are intentionally module-level so future versions can import them.

Pure additions are OK (new methods that do not touch ``state_dict`` keys).
Renames, layer dim changes, removals are NOT OK.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.envs.minibg.actions import BOARD_SIZE
from src.envs.minibg.structured_actions import StructAction, StructActionType

from src.bg_recruitment.discover_pool import ADAPT_KEYS_ALL

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

# Shared helpers / constants live in ``structured_common``. They are re-imported
# here so external code that already does
# ``from src.models.minibg_structured_ac import EntityAttentionBlock`` (tests,
# benches, v2 historically) keeps working without modification.
from .structured_common import (  # noqa: F401  (re-exported)
    EntityAttentionBlock,
    _NUM_REGIONS,
    _NUM_ROLES,
    _NUM_STRUCT_TYPES,
    _PENDING_LEN,
    _REGION_HAND,
    _REGION_NULL,
    _REGION_OWN,
    _REGION_PENDING,
    _REGION_SHOP,
    _ROLE_BOARD,
    _ROLE_HAND,
    _ROLE_NONE,
    _ROLE_PENDING,
    _ROLE_SHOP,
    REG_ENEMY,
    REG_HAND,
    REG_OWN,
    REG_PENDING,
    REG_SHOP,
    _build_action_tokens,
    _struct_action_codes,
    role_for_struct,
)

_MAX_REGION_LEN = max(_OWN_LEN, _SHOP_LEN, _HAND_LEN, _PENDING_LEN, 1)

# Globals (gold, round, healths, actions_left, ...) + last_battle + phase routed
# explicitly into action / order scoring heads so the actor stops ignoring them.
_GLOBALS_FULL_DIM = _GLOBAL_DIM + _LAST_BATTLE_DIM + _PHASE_DIM


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
        obs_layout: str = "minibg",
        num_pool_indices: Optional[int] = None,
    ) -> None:
        super().__init__()
        layout = obs_layout.strip().lower()
        if layout == "bglike":
            from src.envs.bglike.obs import OBS_DIM as _layout_obs_dim
            from src.envs.bglike.actions import (
                BOARD_SIZE as _layout_board,
                HAND_SIZE as _layout_hand,
                MAX_SHOP_SLOTS as _layout_shop,
            )

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
            self.enemy_len = _ENEMY_LEN
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
        if num_pool_indices is None:
            raise ValueError("num_pool_indices is required")
        self.num_pool_indices = int(num_pool_indices)

        k2 = int(region_conv2_kernel)
        if k2 not in (1, 3):
            raise ValueError("region_conv2_kernel must be 1 or 3")
        self._region_conv2_kernel = k2

        self.card_emb = nn.Embedding(
            self.num_pool_indices + 1, self.card_emb_dim, padding_idx=0
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

        self.own_pos_emb = nn.Embedding(self.own_len, self.slot_hidden)
        self.shop_pos_emb = nn.Embedding(self.shop_len, self.slot_hidden)
        self.hand_pos_emb = nn.Embedding(self.hand_len, self.slot_hidden)
        if self.enemy_len > 0:
            self.enemy_pos_emb = nn.Embedding(self.enemy_len, self.slot_hidden)
        else:
            self.enemy_pos_emb = None
        self.pending_pos_emb = nn.Embedding(self.pending_len, self.slot_hidden)
        self.slot_region_emb = nn.Embedding(5, self.slot_hidden)
        for emb in (
            self.own_pos_emb,
            self.shop_pos_emb,
            self.hand_pos_emb,
            self.pending_pos_emb,
            self.slot_region_emb,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        if self.enemy_pos_emb is not None:
            nn.init.normal_(self.enemy_pos_emb.weight, mean=0.0, std=0.02)

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
            self.total_slots * self.slot_hidden
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

        self.order_pos_emb = nn.Embedding(self.board_size, self.order_pos_dim)
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
            "obs_layout": self.obs_layout,
            "num_pool_indices": self.num_pool_indices,
        }

    def _unpack(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
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

        pieces = [E_own, E_shop, E_hand]
        if self.enemy_len > 0:
            pieces.append(E_enemy)
        pieces.append(E_pending)

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
        E_own = E_all[:, i : i + self.own_len]
        i += self.own_len

        E_shop = E_all[:, i : i + self.shop_len]
        i += self.shop_len

        E_hand = E_all[:, i : i + self.hand_len]
        i += self.hand_len

        if self.enemy_len > 0:
            E_enemy = E_all[:, i : i + self.enemy_len]
            i += self.enemy_len
        else:
            E_enemy = E_all[:, i : i + 0]

        E_pending = E_all[:, i : i + self.pending_len]

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
        if self.enemy_len > 0:
            assert self.enemy_pos_emb is not None
            E_enemy = self._add_pos_region(
                self._encode_region_slots(enemy), self.enemy_pos_emb, REG_ENEMY
            )
        else:
            E_enemy = enemy.new_zeros(enemy.size(0), 0, self.slot_hidden)

        # Pending entities (3 discover or ADAPT options). Indices route through ``card_emb``
        # or dedicated ``adapt_choice_emb`` matching ``encode_pending_choice``.
        B = x.size(0)
        device = x.device
        dtype = E_own.dtype
        cont = pending[..., :_PENDING_CHOICE_DIM]
        opt_stack = _pending_three_option_emb(
            pending,
            self.card_emb,
            self.adapt_choice_emb,
            max_card_idx=self.num_pool_indices,
        )
        is_apply = pending[..., _PENDING_IS_APPLY_OFFSET : _PENDING_IS_APPLY_OFFSET + 1] > 0.5
        opt_stack = opt_stack.masked_fill(is_apply.unsqueeze(-1), 0.0)
        pending_feat = torch.cat([cont, opt_stack.flatten(-2)], dim=-1)
        E_pending = self._add_pos_region(
            self.pending_to_slot(opt_stack), self.pending_pos_emb, REG_PENDING
        )

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

        def _gather_entity(
            region_kinds: torch.Tensor,
            region_slots: torch.Tensor,
            ent_lin: nn.Linear,
        ) -> torch.Tensor:
            flat_idx = region_kinds * self.max_region_len + region_slots
            ent_idx = flat_idx.unsqueeze(-1).expand(-1, -1, self.slot_hidden)
            ent_slot = torch.gather(E_regions_flat, dim=1, index=ent_idx)
            ent_from_slot = ent_lin(ent_slot)
            # Null-region tokens (padded slots, ROLL/LEVEL_UP/COMPLETE_TURN sources,
            # tgt of all single-entity actions) must zero out — otherwise the Linear
            # bias term leaks into the action embedding for every "empty" entity slot.
            is_null = (region_kinds == _REGION_NULL).unsqueeze(-1)
            return ent_from_slot.masked_fill(is_null, 0.0)

        src_ent = _gather_entity(
            src_region_kinds, src_region_slots, self.entity_to_action
        )
        tgt_ent = _gather_entity(
            tgt_region_kinds, tgt_region_slots, self.entity_to_action_tgt
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
        board_size = self.board_size
        if K != board_size:
            raise ValueError(f"E_own must have board_size={board_size} slots, got {K}")
        device = E_own.device
        remaining = occupied_mask.clone()
        hidden = self._order_init_hidden(state_emb)
        slot_in = self.order_start.unsqueeze(0).expand(B, -1)

        logprob_sum = torch.zeros(B, device=device, dtype=E_own.dtype)
        picked = torch.full((B, board_size), -1, device=device, dtype=torch.long)
        batch_arange = torch.arange(B, device=device)

        pos_table = self.order_pos_emb(torch.arange(board_size, device=device))

        for pos in range(board_size):
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


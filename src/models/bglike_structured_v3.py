"""Structured actor-critic, v3.

Difference from v2: **action tokens cross-attend to the entity sequence**.

Motivation: in v2 each legal action's embedding is built from its own
src/tgt entity (gather) + type/role embeddings. Whether "buy this Murloc"
is a good action depends on what else is on the board (other Murlocs,
synergy pieces, gold), but that context only reaches the action token
through whatever the entity self-attention has already baked into the
gathered ``E_shop[idx]`` slot. v3 lets each action token explicitly query
the post-self-attention entity sequence (``E_own | E_shop | E_hand |
E_pending``) before being scored.

Implementation:
  - Inherits from ``BGLikeStructuredV2`` so all encoding (slot conv, CLS
    token, entity self-attention, state summary, critic, board-order
    pointer head) is reused unchanged.
  - Adds ``action_to_slot`` (action_dim → slot_hidden) + a
    ``CrossAttentionBlock`` over (Q = action tokens, K/V = entities) +
    ``slot_to_action`` (slot_hidden → action_dim).
  - ``slot_to_action`` is **zero-initialized** so v3 starts with exactly
    v2's behaviour — the cross-attention head must learn to use the
    extra signal, it can't make things worse at step 0.

Checkpoint contract: v3 has its own ``ppo_network_type``
(``bglike_structured_v3``). v2 and v3 checkpoints are not interchangeable
because v3 adds new state_dict keys.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .bglike_structured_v2 import BGLikeStructuredV2
from .structured_common import CrossAttentionBlock


class BGLikeStructuredV3(BGLikeStructuredV2):
    """v2 + cross-attention from action tokens to the entity sequence."""

    def __init__(
        self,
        *,
        action_cross_attn_heads: int = 4,
        action_cross_attn_ff_mult: int = 2,
        action_cross_attn_init_scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if self.slot_hidden % int(action_cross_attn_heads) != 0:
            raise ValueError(
                f"slot_hidden={self.slot_hidden} must be divisible by "
                f"action_cross_attn_heads={action_cross_attn_heads}"
            )

        self.action_cross_attn_heads = int(action_cross_attn_heads)
        self.action_cross_attn_ff_mult = int(action_cross_attn_ff_mult)
        self.action_cross_attn_init_scale = float(action_cross_attn_init_scale)

        # Bridge action_dim ↔ slot_hidden so cross-attention can run at the
        # entity-token width (where CLS/board/shop/etc. all live).
        self.action_to_slot = nn.Linear(self.action_dim, self.slot_hidden)
        self.action_cross_attn = CrossAttentionBlock(
            self.slot_hidden,
            num_heads=self.action_cross_attn_heads,
            ff_mult=self.action_cross_attn_ff_mult,
            init_scale=self.action_cross_attn_init_scale,
        )
        self.slot_to_action = nn.Linear(self.slot_hidden, self.action_dim)

        # Zero-init the projection back to action_dim so at step 0 the cross-attn
        # head contributes exactly nothing — v3 reproduces v2's outputs and
        # training has to actively pull weight onto this path.
        nn.init.zeros_(self.slot_to_action.weight)
        nn.init.zeros_(self.slot_to_action.bias)

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        kw = super().get_constructor_kwargs()
        kw["action_cross_attn_heads"] = self.action_cross_attn_heads
        kw["action_cross_attn_ff_mult"] = self.action_cross_attn_ff_mult
        kw["action_cross_attn_init_scale"] = self.action_cross_attn_init_scale
        return kw

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
        ae = super()._encode_actions(
            type_ids,
            role_ids,
            src_region_kinds,
            src_region_slots,
            tgt_region_kinds,
            tgt_region_slots,
            cache,
        )

        # Entity K/V sequence: post-self-attention entities from all regions
        # the action head cares about. CLS isn't stored separately in cache,
        # but every E_* token already carries global context via the CLS-mixed
        # self-attention in encode_state, so this is sufficient.
        entity_seq = torch.cat(
            [cache["E_own"], cache["E_shop"], cache["E_hand"], cache["E_pending"]],
            dim=1,
        )  # (B, N_ent, slot_hidden)

        ae_q = self.action_to_slot(ae)  # (B, Lmax, slot_hidden)
        ae_attended = self.action_cross_attn(ae_q, entity_seq)  # (B, Lmax, slot_hidden)
        ae_extra = self.slot_to_action(ae_attended)  # (B, Lmax, action_dim); zero at init
        return ae + ae_extra


__all__ = ["BGLikeStructuredV3"]

"""Structured actor-critic, v7 = v6 + per-policy *identity* conditioning.

Difference from v6: the network can be told which member of a co-trained
population is acting, and adds a learned identity contribution into the CLS
token, so a single shared trunk can express several distinct build strategies
— the substrate for population-diversity training (DvD-style behavioural
repulsion across identities) and, later, a meta-selector that picks a style.

How identity arrives: as a **one-hot tail appended to the observation vector**,
exactly mirroring how v5/v6 appended the ability block to the v3 obs. This is
deliberate — it means the identity rides through the rollout buffer, the PPO
update, the GAE bootstrap and every other obs-carrying path *for free*, with no
new argument threaded through the (large, shared) structured agent. The agent
only has to widen the obs it feeds the net.

Accepted input widths (``encode_state`` / forward):
  * ``OBS_DIM_V5``                      → no identity, **exact v6 behaviour**;
  * ``OBS_DIM_V5 + num_identities``     → last ``num_identities`` floats are a
                                          one-hot identity; its contribution is
                                          added to the CLS token.

Mechanically:
  * ``identity_proj`` : ``Linear(num_identities, slot_hidden)``, **zero
    initialised** (weight and bias). Applied to the one-hot tail and added to
    the v6 CLS token before self-attention, so the identity signal propagates
    to every slot. Zero-init ⇒ v7 reproduces v6 byte-for-byte at step 0
    (the added term is exactly zero); gradient still flows on the first update.

Why add into the CLS token (and not widen an inherited layer): keeps every v6
weight-tensor shape identical, so a v6 checkpoint warm-starts v7 with
``strict=False`` (only ``identity_proj`` is missing) and behaves identically
until trained.

Checkpoint contract: own ``ppo_network_type`` (``bglike_structured_v7``); not
interchangeable with v6 (extra ``identity_proj`` params, wider obs).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.envs.bglike.obs_v5 import OBS_DIM_V5

from .bglike_structured_v6 import BGLikeStructuredV6
from .structured_common import (
    REG_HAND,
    REG_OWN,
    REG_PENDING,
    REG_SHOP,
)


class BGLikeStructuredV7(BGLikeStructuredV6):
    """v6 + per-policy identity conditioning via a one-hot observation tail."""

    def __init__(
        self,
        *,
        num_identities: int = 8,
        ability_emb_dim: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(ability_emb_dim=ability_emb_dim, **kwargs)

        self.num_identities = int(num_identities)

        # Full input width the agent must feed: v6 obs + one-hot identity tail.
        # (The v6 region/offset bookkeeping computed in the parent stays sized
        # to ``OBS_DIM_V5``; encode_state slices the core off the front.)
        self.obs_dim = int(OBS_DIM_V5) + self.num_identities

        # Zero-init projection of the one-hot tail into the CLS token. Both
        # weight and bias are zero so identity contributes nothing at step 0.
        self.identity_proj = nn.Linear(self.num_identities, self.slot_hidden)
        nn.init.zeros_(self.identity_proj.weight)
        nn.init.zeros_(self.identity_proj.bias)

        # Per-slot identity GATE: identity × slot-content interaction.
        # Additive and FiLM both apply the *same* transform to every slot, so they
        # can't say "prefer THIS slot because it's my tribe" — BUY logits compete
        # between slots and are ~invariant to a shared transform (causal probe:
        # TV≈0.03 between identities even with huge identity weights). The gate is
        # computed per slot from concat(slot_encoding, identity_embedding): it can
        # up/down-weight each slot individually based on whether its content
        # matches the acting identity's tribe. ``identity_emb_proj`` maps the
        # one-hot to an embedding; ``identity_slot_gate`` maps (2H→H). Zero-init
        # gate ⇒ modulation 0 at step 0 (v6 warm-start preserved); the agent's
        # identity_init_std re-inits non-zero to break symmetry from step 1.
        self.identity_emb_proj = nn.Linear(self.num_identities, self.slot_hidden)
        nn.init.normal_(self.identity_emb_proj.weight, std=0.02)
        nn.init.zeros_(self.identity_emb_proj.bias)
        self.identity_slot_gate = nn.Linear(2 * self.slot_hidden, self.slot_hidden)
        nn.init.zeros_(self.identity_slot_gate.weight)
        nn.init.zeros_(self.identity_slot_gate.bias)

    # ------------------------------------------------------------------
    # Split helper: (core v6 obs, one-hot identity tail | None).
    # ------------------------------------------------------------------
    def _split_identity(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        w = x.shape[1]
        if w == OBS_DIM_V5:
            return x, None
        if w == OBS_DIM_V5 + self.num_identities:
            return x[:, :OBS_DIM_V5], x[:, OBS_DIM_V5:]
        raise ValueError(
            f"v7 expected obs dim {OBS_DIM_V5} or "
            f"{OBS_DIM_V5 + self.num_identities}, got {w}"
        )

    # ------------------------------------------------------------------
    # encode_state: v6's body verbatim on the core obs, with the single change
    # that the CLS token gains an additive (zero-init) identity term decoded
    # from the one-hot tail. A core-width obs (no tail) takes the exact v6 path.
    # ------------------------------------------------------------------
    def encode_state(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from src.envs.bglike.obs import OBS_DIM as _OBS_DIM_V3
        from src.envs.minibg.obs import (
            PENDING_IS_APPLY_OFFSET as _PENDING_IS_APPLY_OFFSET,
        )
        from .minibg_slot_ac import _pending_three_option_emb

        xc, id_onehot = self._split_identity(x)
        head = xc[:, :_OBS_DIM_V3]

        # ---- v3-equivalent obs unpacking ----
        saved_obs_dim = self.obs_dim
        self.obs_dim = _OBS_DIM_V3
        try:
            g, own, shop, hand, _enemy, lb, phase, pending = self._unpack(head)
        finally:
            self.obs_dim = saved_obs_dim

        # ---- Per-region ability summaries (unified encoder input) ----
        abil_raw = self._unpack_v6_abilities(xc)
        abil_own = self._per_host_ability_summary(
            abil_raw, self._abil_own_offset, self.own_len
        )
        abil_shop = self._per_host_ability_summary(
            abil_raw, self._abil_shop_offset, self.shop_len
        )
        abil_hand = self._per_host_ability_summary(
            abil_raw, self._abil_hand_offset, self.hand_len
        )
        abil_pending = self._per_host_ability_summary(
            abil_raw, self._abil_pending_offset, self.pending_len
        )

        # ---- Per-minion encoder: stats / card / abilities together ----
        E_own = self._add_pos_region(
            self._encode_region_slots(own, abil_own), self.own_pos_emb, REG_OWN
        )
        E_shop = self._add_pos_region(
            self._encode_region_slots(shop, abil_shop), self.shop_pos_emb, REG_SHOP
        )
        E_hand = self._add_pos_region(
            self._encode_region_slots(hand, abil_hand), self.hand_pos_emb, REG_HAND
        )

        B = xc.size(0)

        # ---- Pending options: card_emb + ability_summary → widened Linear ----
        opt_stack = _pending_three_option_emb(
            pending,
            self.card_emb,
            self.adapt_choice_emb,
            max_card_idx=self.num_pool_indices,
        )  # (B, PENDING_LEN, card_emb_dim)
        is_apply = (
            pending[..., _PENDING_IS_APPLY_OFFSET : _PENDING_IS_APPLY_OFFSET + 1] > 0.5
        )
        opt_stack = opt_stack.masked_fill(is_apply.unsqueeze(-1), 0.0)
        pending_input = torch.cat([opt_stack, abil_pending], dim=-1)
        E_pending = self._add_pos_region(
            self.pending_to_slot(pending_input), self.pending_pos_emb, REG_PENDING
        )

        pending_header = pending[..., : self._pending_header_dim]
        g_full = torch.cat([g, lb, phase], dim=-1)

        # ---- CLS token (+ zero-init identity term decoded from the tail) ----
        cls_base = self.cls_from_globals(g_full)
        if id_onehot is not None:
            cls_base = cls_base + self.identity_proj(id_onehot)
        cls_tok = cls_base.unsqueeze(1)

        # ---- Per-slot identity gate: identity × each slot's content ----
        if id_onehot is not None:
            id_e = self.identity_emb_proj(id_onehot)  # (B, H)

            def _gate(E):
                Bn, Ln, Hn = E.shape
                cat = torch.cat([E, id_e.unsqueeze(1).expand(Bn, Ln, Hn)], dim=-1)
                return E * (1.0 + self.identity_slot_gate(cat))  # per-slot modulation

            E_own = _gate(E_own)
            E_shop = _gate(E_shop)
            E_hand = _gate(E_hand)
            E_pending = _gate(E_pending)

        E_all = torch.cat([cls_tok, E_own, E_shop, E_hand, E_pending], dim=1)
        for block in self.entity_attn:
            E_all = block(E_all)

        cls_out = E_all[:, 0]
        idx = 1
        E_own = E_all[:, idx : idx + self.own_len]
        idx += self.own_len
        E_shop = E_all[:, idx : idx + self.shop_len]
        idx += self.shop_len
        E_hand = E_all[:, idx : idx + self.hand_len]
        idx += self.hand_len
        E_pending = E_all[:, idx : idx + self.pending_len]
        E_enemy = E_all.new_zeros(B, 0, self.slot_hidden)

        state_summary = torch.cat([cls_out, g, lb, phase, pending_header], dim=-1)
        state_summary_n = self.state_summary_ln(state_summary)
        state_emb = self.state_proj(state_summary_n)

        cache: Dict[str, torch.Tensor] = {
            "E_own": E_own,
            "E_shop": E_shop,
            "E_hand": E_hand,
            "E_enemy": E_enemy,
            "E_pending": E_pending,
            "trunk": state_summary_n,
            "g_full": g_full,
        }
        return state_emb, cache

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        kw = super().get_constructor_kwargs()
        kw["num_identities"] = self.num_identities
        return kw


__all__ = ["BGLikeStructuredV7"]

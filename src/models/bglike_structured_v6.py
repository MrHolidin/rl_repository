"""Structured actor-critic, v6 = v3 + abilities as extra channels of the
per-minion encoder.

Difference from v3: abilities are **part of the input** to the same conv that
encodes the rest of the minion's properties (stats / race / keywords /
triggers-1hot / effect-1hot / card_emb). The slot encoder produces one unified
vector per minion that already accounts for "DiscoverMurloc with condition
OTHER_TRIBE_ON_BOARD: MURLOC + 3/2 stats + Murloc race" as a single entity,
instead of computing the base slot vector and bolting an ability summary on
top.

Mechanically:
  * Per-host ability summary: encode K=4 raw ability tokens via
    ``AbilityTokenEncoder``, pool to one ``slot_hidden`` vector with a single
    learned-query softmax pool (padding via ``effect_id == 0`` mask).
  * ``region_conv1`` input is widened to ``conv_in_card + slot_hidden`` —
    abilities become extra channels alongside stats / card_emb. Conv (and its
    downstream relu+conv2+relu nonlinearities) can directly learn interactions
    between stats and abilities inside one minion.
  * ``pending_to_slot`` is widened the same way for pending options that carry
    a discoverable card (DISCOVER_MURLOC / TRIPLE_REWARD_DISCOVER); adapt /
    apply options have zero ability summary by construction (no minion).
  * Both widened layers have **zero-initialized weights** for the new ability
    channels, so v6 reproduces v3 byte-for-byte at step 0. Gradient still
    flows on the first update — dL/dW = output_grad @ input.T is non-zero
    even when the ability slice of W is zero, so weights move immediately.

Why this over v5 (separate region) and over the earlier residual-bolt-on
attempt: in BG, abilities are **structurally** a property of a card, not a
free-floating relational thing. The model shouldn't need to learn the
binding from positional cues (v5) or learn that a separate residual stream is
related to a particular slot (residual-bolt-on). Threading abilities through
the same conv that encodes the rest of the minion is the architecture that
matches the domain: one card → one encoder pass → one vector that knows
everything.

Obs contract: same as v5 (``OBS_DIM_V5``, ``obs_kind="bglike_v5"``).

What v6 does NOT change:
  * ``BattlePredictionHead`` still calls ``_encode_region_slots`` without an
    ability summary (combat snapshots don't carry ability blocks on the env
    side yet). With ``ability_summary=None`` the override fills the
    ability-channel input with zeros — since the corresponding conv1 weights
    are zero-initialized, the head's output at step 0 == v3. After training
    those ability-channel weights are no longer zero, so the head's behaviour
    will shift slightly even without seeing real ability data (it'll see
    zeros in the ability channels). Acceptable for an auxiliary aux head; if
    it matters, wire abilities into ``predict_battle`` as a follow-up.

Checkpoint contract: v6 has its own ``ppo_network_type``
(``bglike_structured_v6``). Not interchangeable with v3/v4/v5 — widened
``region_conv1`` and ``pending_to_slot`` weight shapes differ from v3.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.bglike.obs_v5 import (
    ABIL_FEAT_DIM,
    ABIL_OFF_EFFECT,
    K_ABIL,
    OBS_DIM_V5,
    PENDING_LEN,
)
from src.envs.bglike.actions import (
    BOARD_SIZE as _BG_BOARD_SIZE,
    HAND_SIZE as _BG_HAND_SIZE,
    MAX_SHOP_SLOTS as _BG_SHOP_SLOTS,
)

from .bglike_structured_v3 import BGLikeStructuredV3
from .bglike_structured_v5 import AbilityTokenEncoder
from .minibg_slot_ac import _split_card_idx_and_cont
from .structured_common import (
    REG_HAND,
    REG_OWN,
    REG_PENDING,
    REG_SHOP,
)


class BGLikeStructuredV6(BGLikeStructuredV3):
    """v3 + abilities as extra input channels of the per-minion encoder."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        ability_emb_dim: int = 8,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("obs_layout", "bglike")
        if kwargs["obs_layout"] != "bglike":
            raise ValueError(
                f"BGLikeStructuredV6 requires obs_layout='bglike' (slot encoder "
                f"layout unchanged); v6 reads the ability block tail of the v5 "
                f"obs. Got {kwargs['obs_layout']!r}."
            )
        super().__init__(**kwargs)

        self.obs_dim = int(OBS_DIM_V5)
        self.ability_emb_dim = int(ability_emb_dim)
        self.k_abil = int(K_ABIL)
        self.abil_feat_dim = int(ABIL_FEAT_DIM)
        self._ability_summary_dim = int(self.slot_hidden)

        if (
            self.own_len != _BG_BOARD_SIZE
            or self.shop_len != _BG_SHOP_SLOTS
            or self.hand_len != _BG_HAND_SIZE
            or self.pending_len != PENDING_LEN
        ):
            raise RuntimeError(
                "v6 region sizes diverged from obs_v5 expectations: "
                f"own={self.own_len} shop={self.shop_len} hand={self.hand_len} "
                f"pending={self.pending_len}"
            )

        # ---- Ability encoder + pool ------------------------------------
        self.ability_encoder = AbilityTokenEncoder(
            slot_hidden=self.slot_hidden,
            emb_dim=self.ability_emb_dim,
            card_emb_dim=self.card_emb_dim,
        )
        # Single learned query for the per-host softmax pool. Random init so
        # the pool starts with non-uniform attention; the *conv input weights*
        # for the ability channels are zero-init (below), which is what
        # actually controls "v6 == v3 at step 0".
        self.ability_pool_query = nn.Parameter(torch.empty(self.slot_hidden))
        nn.init.normal_(self.ability_pool_query, mean=0.0, std=0.02)

        # ---- Widen region_conv1 to take ability summary as extra channels.
        # The parent's region_conv1 was sized for `conv_in_card = SLOT_CONT_DIM
        # + card_emb_dim`. We expand the input by `slot_hidden` (= ability
        # summary dim), preserve original-card weights, zero-init ability
        # weights → step-0 output identical to v3's conv1 on the same input.
        self._conv_in_card = int(self.region_conv1.weight.shape[1])
        self.region_conv1 = self._widen_conv1d_input(
            self.region_conv1,
            extra_in_channels=self._ability_summary_dim,
            extra_zero_init=True,
        )

        # ---- Widen pending_to_slot the same way (it's a Linear, not Conv1d).
        # Pending options that resolve to a discoverable card now go through a
        # single encoder that sees both card_emb and the option's ability
        # summary at once. Adapt/apply options carry zero ability summary.
        self._pending_in_card = int(self.pending_to_slot.weight.shape[1])
        self.pending_to_slot = self._widen_linear_input(
            self.pending_to_slot,
            extra_in_features=self._ability_summary_dim,
            extra_zero_init=True,
        )

        # ---- Per-region ability-block offsets in the flat ability tail.
        self._abil_own_offset = 0
        self._abil_shop_offset = self._abil_own_offset + self.own_len * self.k_abil
        self._abil_hand_offset = self._abil_shop_offset + self.shop_len * self.k_abil
        self._abil_pending_offset = self._abil_hand_offset + self.hand_len * self.k_abil
        self._abil_total_tokens = self._abil_pending_offset + self.pending_len * self.k_abil

    # ------------------------------------------------------------------
    # Layer widening helpers (preserve original weights, zero new channels)
    # ------------------------------------------------------------------

    @staticmethod
    def _widen_conv1d_input(
        conv: nn.Conv1d,
        *,
        extra_in_channels: int,
        extra_zero_init: bool,
    ) -> nn.Conv1d:
        """Return a new Conv1d with ``in_channels + extra_in_channels`` input
        channels. Copies original weights into the first ``in_channels`` slice;
        new channels are zero-init (so the layer's output on
        ``concat(orig_input, zeros)`` matches the original conv exactly).
        """
        old_w = conv.weight.data  # (out, in, k)
        old_b = conv.bias.data if conv.bias is not None else None
        out_dim, in_dim_old, k = old_w.shape
        in_dim_new = in_dim_old + int(extra_in_channels)
        new_conv = nn.Conv1d(in_dim_new, out_dim, kernel_size=k, bias=(old_b is not None))
        with torch.no_grad():
            new_conv.weight.data.zero_()
            new_conv.weight.data[:, :in_dim_old, :].copy_(old_w)
            if not extra_zero_init:
                # Fall back to standard init for the new slice
                nn.init.kaiming_uniform_(
                    new_conv.weight.data[:, in_dim_old:, :], a=5 ** 0.5
                )
            if old_b is not None:
                new_conv.bias.data.copy_(old_b)
        return new_conv

    @staticmethod
    def _widen_linear_input(
        layer: nn.Linear,
        *,
        extra_in_features: int,
        extra_zero_init: bool,
    ) -> nn.Linear:
        """Return a new Linear with ``in_features + extra_in_features`` input.
        Original weights preserved on the first slice; new slice zero-init."""
        old_w = layer.weight.data  # (out, in)
        old_b = layer.bias.data if layer.bias is not None else None
        out_dim, in_dim_old = old_w.shape
        in_dim_new = in_dim_old + int(extra_in_features)
        new_layer = nn.Linear(in_dim_new, out_dim, bias=(old_b is not None))
        with torch.no_grad():
            new_layer.weight.data.zero_()
            new_layer.weight.data[:, :in_dim_old].copy_(old_w)
            if not extra_zero_init:
                nn.init.kaiming_uniform_(
                    new_layer.weight.data[:, in_dim_old:], a=5 ** 0.5
                )
            if old_b is not None:
                new_layer.bias.data.copy_(old_b)
        return new_layer

    # ------------------------------------------------------------------
    # Ability summary helpers
    # ------------------------------------------------------------------

    def _unpack_v6_abilities(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, num_abil_tokens, ABIL_FEAT_DIM)`` slice off the obs tail."""
        if x.shape[1] != OBS_DIM_V5:
            raise ValueError(
                f"v6 expected obs dim {OBS_DIM_V5}, got {x.shape[1]}"
            )
        from src.envs.bglike.obs_v5 import ABIL_BLOCK_OFFSET

        tail = x[:, ABIL_BLOCK_OFFSET:]
        B = x.shape[0]
        return tail.view(B, self._abil_total_tokens, self.abil_feat_dim)

    def _per_host_ability_summary(
        self,
        abil_raw: torch.Tensor,
        region_start: int,
        region_len: int,
    ) -> torch.Tensor:
        """``(B, region_len, slot_hidden)``: K abilities of each host pooled
        into one summary vector via single-query softmax attention with
        ``effect_id == 0`` padding mask. All-padded hosts produce a zero
        contribution (NaN-safe). No additional projection — the widened
        ``region_conv1`` / ``pending_to_slot`` will project as part of the
        per-minion encoder pass.
        """
        B = abil_raw.shape[0]
        region_block = abil_raw[
            :, region_start : region_start + region_len * self.k_abil
        ]
        region_block = region_block.view(
            B, region_len, self.k_abil, self.abil_feat_dim
        )

        encoded = self.ability_encoder(region_block, self.card_emb)  # (B, L, K, slot_hidden)
        pad_mask = region_block[..., ABIL_OFF_EFFECT] < 0.5  # (B, L, K) True=pad
        all_pad = pad_mask.all(dim=-1, keepdim=True)  # (B, L, 1)

        query = self.ability_pool_query.view(1, 1, 1, -1)
        scores = (encoded * query).sum(dim=-1)
        # For all-pad hosts leave scores untouched (uniform); we zero the
        # pooled output explicitly below. For partially-padded hosts mask the
        # pad slots so they get -inf and drop out of the softmax.
        scores = torch.where(
            all_pad.expand_as(scores),
            scores,
            scores.masked_fill(pad_mask, float("-inf")),
        )
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, K, 1)
        pooled = (encoded * weights).sum(dim=-2)  # (B, L, slot_hidden)
        pooled = pooled.masked_fill(all_pad.expand_as(pooled), 0.0)
        return pooled

    # ------------------------------------------------------------------
    # Overridden per-minion encoder: takes an optional ability_summary as
    # extra input channels. Callers that don't have an ability summary
    # (e.g. ``predict_battle``) get zero-filled channels — the widened
    # conv1's ability-channel weights are zero-init, so the conv output
    # on (orig_input, zeros) is bit-identical to v3's conv1(orig_input)
    # at step 0.
    # ------------------------------------------------------------------

    def _encode_region_slots(
        self,
        z: torch.Tensor,
        ability_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = _split_card_idx_and_cont(
            z, self.card_emb, max_card_idx=self.num_pool_indices
        )  # (B, L, conv_in_card)

        B, L, _ = z.shape
        if ability_summary is None:
            ability_summary = z.new_zeros(B, L, self._ability_summary_dim)
        else:
            if ability_summary.shape != (B, L, self._ability_summary_dim):
                raise ValueError(
                    f"ability_summary shape {tuple(ability_summary.shape)} != "
                    f"expected ({B}, {L}, {self._ability_summary_dim})"
                )

        # Per-minion encoder sees: (orig_slot_features, ability_summary)
        # as one contiguous channel vector. Conv1's ability-channel weights
        # are zero-init → step-0 output identical to v3.
        z_full = torch.cat([z, ability_summary], dim=-1)
        h = z_full.transpose(1, 2)
        h = F.relu(self.region_conv1(h))
        h = F.relu(self.region_conv2(h))
        return h.transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # ``encode_state`` — same as v2/v3, but threads per-host ability
    # summaries into the slot/pending encoders.
    # ------------------------------------------------------------------

    def encode_state(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from src.envs.bglike.obs import OBS_DIM as _OBS_DIM_V3
        from src.envs.minibg.obs import (
            PENDING_IS_APPLY_OFFSET as _PENDING_IS_APPLY_OFFSET,
        )
        from .minibg_slot_ac import _pending_three_option_emb

        if x.shape[1] != OBS_DIM_V5:
            raise ValueError(
                f"v6 expected obs dim {OBS_DIM_V5}, got {x.shape[1]}"
            )
        head = x[:, :_OBS_DIM_V3]

        # ---- v3-equivalent obs unpacking ----
        saved_obs_dim = self.obs_dim
        self.obs_dim = _OBS_DIM_V3
        try:
            g, own, shop, hand, _enemy, lb, phase, pending = self._unpack(head)
        finally:
            self.obs_dim = saved_obs_dim

        # ---- Per-region ability summaries (unified encoder input) ----
        abil_raw = self._unpack_v6_abilities(x)
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

        B = x.size(0)
        device = x.device
        dtype = E_own.dtype

        # ---- Pending options: card_emb + ability_summary → widened Linear ----
        opt_stack = _pending_three_option_emb(
            pending,
            self.card_emb,
            self.adapt_choice_emb,
            max_card_idx=self.num_pool_indices,
        )  # (B, PENDING_LEN, card_emb_dim)
        is_apply = pending[..., _PENDING_IS_APPLY_OFFSET : _PENDING_IS_APPLY_OFFSET + 1] > 0.5
        opt_stack = opt_stack.masked_fill(is_apply.unsqueeze(-1), 0.0)
        # Pending encoder input = (card_emb, ability_summary). For apply / adapt
        # options abil_pending is naturally zero (those modal kinds don't
        # carry minion-typed options — see obs_v5.encode_pending_option_abilities).
        pending_input = torch.cat([opt_stack, abil_pending], dim=-1)
        E_pending = self._add_pos_region(
            self.pending_to_slot(pending_input), self.pending_pos_emb, REG_PENDING
        )

        pending_header = pending[..., : self._pending_header_dim]
        g_full = torch.cat([g, lb, phase], dim=-1)

        cls_tok = self.cls_from_globals(g_full).unsqueeze(1)
        E_all = torch.cat([cls_tok, E_own, E_shop, E_hand, E_pending], dim=1)
        # Sequence length stays at v3's ~27 tokens (no separate ability region).
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

        state_summary = torch.cat(
            [cls_out, g, lb, phase, pending_header], dim=-1
        )
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

    # ------------------------------------------------------------------
    # Action cross-attn (inherited from v3) — entity_seq slots are already
    # encoded with abilities baked in. No override needed.
    # ------------------------------------------------------------------

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        kw = super().get_constructor_kwargs()
        kw["ability_emb_dim"] = self.ability_emb_dim
        return kw


__all__ = ["BGLikeStructuredV6"]

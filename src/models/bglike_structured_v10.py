"""Structured actor-critic, v10 = v9 + opponent tokens + post-attention thinking core.

Two orthogonal additions over v9, both motivated by the diagnosis that bglike
training **saturates** (~5M steps, meta ~2M) — the bottleneck is capacity /
architecture in the *post-aggregation* path, not the training budget.

(A) Opponents become attention TOKENS instead of 77 flat globals.
    v2..v9 feed the 77-dim lobby panel (7 opponents × [hp, alive, tribe-9])
    into the encoder only as raw floats inside ``g`` — consumed flat by
    ``cls_from_globals``, by the ``score_fc`` ``g_full`` slice, and verbatim in
    the state summary. 63 of those 77 dims are sparse tribe one-hots, and there
    is **no relational processing across opponents** (no "how many seats contest
    my tribe"). v10 reshapes the panel back into 7 per-opponent feature vectors
    (``[hp_i, alive_i, tribe_i(9)]`` → 11 dims), projects each to ``slot_hidden``,
    tags it with an opponent position + the reserved REG_ENEMY region embedding,
    and joins the 7 tokens to the main entity self-attention. The CLS token then
    pools opponents *learnedly* (matches the no-hand-aggregates obs philosophy),
    and the 77 raw dims are **removed from the flat path** (zeroed in ``g`` /
    ``g_full``) so the post-attention summary is genuinely compressed rather than
    carrying the panel twice. The economy slice that v9 injects into action
    queries (leading ``core + rotation`` = 25 floats) is BEFORE the panel offset,
    so v9's ``action_econ_proj`` is unaffected.

(B) A post-attention "thinking" core refines the trunk before the heads.
    After the CLS-token aggregation there is currently only ``state_proj`` (one
    Linear) and the ``score_fc`` MLP between the pooled state and the heads —
    no depth to *combine* the aggregated facts into a plan. v10 inserts a stack
    of ``thinking_blocks`` ReZero residual MLP blocks over the post-LN trunk
    (dim ``_state_summary_dim``); the refined trunk feeds BOTH the distributional
    critic (``value_from_trunk`` reads ``cache["trunk"]``) and the actor
    (``state_emb = state_proj(refined_trunk)``).

    ReZero gate is initialised **non-zero** (α=0.1), NOT zero: v10 is trained
    from scratch (no warm-start), so the zero-init-identity trick of v3/v7/v9 is
    not just unnecessary but counterproductive — a zero-init additive branch can
    sit at a lazy minimum and never wake up (cf. the DvD identity-pathway
    atrophy). α is per-feature and logged-friendly: if it stays ≈0 the core is
    dead, same diagnostic as ``attn_scale``/``ff_scale``.

What stays exactly v9: obs layout / width, ability channels (v6), identity
conditioning + per-slot gate (v7), distributional placement critic (v8),
economy-coloured action queries (v9), the board-order pointer head. The trunk
dimension is unchanged (panel is zeroed, not removed), so all inherited
head layers keep their shapes.

Checkpoint contract: own ``ppo_network_type`` (``bglike_structured_v10``); not
interchangeable with v9 (extra ``opp_proj`` / ``opp_pos_emb`` / ``thinking_core``
params). A v9 checkpoint can warm-start with ``strict=False`` (only the new
modules are missing) but behaviour will NOT match v9 at step 0 because the panel
is rerouted and ReZero α is non-zero.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bglike_structured_v9 import BGLikeStructuredV9
from .structured_common import REG_HAND, REG_OWN, REG_PENDING, REG_SHOP

# Region index for opponent tokens — reuse the slot_region_emb row reserved for
# enemy slots (index 4; see BGLikeStructuredV2.slot_region_emb shape == 5).
_REG_ENEMY_ROW = 4


class _ReZeroMLPBlock(nn.Module):
    """Pre-LN residual MLP block with a per-feature ReZero gate.

    ``x + alpha * fc2(gelu(fc1(LN(x))))``. ``alpha`` is a learned per-feature
    vector initialised to ``init_alpha`` (non-zero) so the block contributes
    from the first step but starts small. Mirrors the residual-gate style of
    ``EntityAttentionBlock`` (``attn_scale`` / ``ff_scale``).
    """

    def __init__(self, dim: int, hidden: int, *, init_alpha: float = 0.1) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.alpha = nn.Parameter(torch.full((dim,), float(init_alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + self.alpha.view(1, -1) * h


class BGLikeStructuredV10(BGLikeStructuredV9):
    """v9 + opponent attention tokens + ReZero post-attention thinking core."""

    def __init__(
        self,
        *,
        thinking_hidden: int = 256,
        thinking_blocks: int = 2,
        thinking_init_alpha: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.thinking_hidden = int(thinking_hidden)
        self.thinking_blocks = int(thinking_blocks)
        self.thinking_init_alpha = float(thinking_init_alpha)

        # ---- (A) opponent tokens --------------------------------------
        # Panel geometry, imported so we don't hardcode offsets.
        from src.envs.bglike.obs import (
            BGLIKE_GLOBAL_CORE_DIM,
            LOBBY_PANEL_DIM,
            MAX_OPPS,
        )
        from src.envs.minibg.obs import RACE_ONEHOT_DIM, SHOP_ROTATION_OBS_DIM

        self._has_panel = self.obs_layout == "bglike"
        self._panel_off = int(BGLIKE_GLOBAL_CORE_DIM) + int(SHOP_ROTATION_OBS_DIM)
        self._panel_dim = int(LOBBY_PANEL_DIM)
        self._max_opps = int(MAX_OPPS)
        self._race_onehot_dim = int(RACE_ONEHOT_DIM)
        # Per-opponent feature = [hp, alive, tribe-one-hot].
        self._opp_feat_dim = 2 + self._race_onehot_dim

        if self._has_panel:
            self.opp_proj = nn.Linear(self._opp_feat_dim, self.slot_hidden)
            nn.init.normal_(self.opp_proj.weight, std=0.02)
            nn.init.zeros_(self.opp_proj.bias)
            self.opp_pos_emb = nn.Embedding(self._max_opps, self.slot_hidden)
            nn.init.normal_(self.opp_pos_emb.weight, mean=0.0, std=0.02)

        # ---- (B) thinking core over the trunk -------------------------
        self.thinking_core = nn.Sequential(
            *[
                _ReZeroMLPBlock(
                    self._state_summary_dim,
                    self.thinking_hidden,
                    init_alpha=self.thinking_init_alpha,
                )
                for _ in range(self.thinking_blocks)
            ]
        )

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        kw = super().get_constructor_kwargs()
        kw["thinking_hidden"] = self.thinking_hidden
        kw["thinking_blocks"] = self.thinking_blocks
        kw["thinking_init_alpha"] = self.thinking_init_alpha
        return kw

    # ------------------------------------------------------------------
    # Opponent-token construction from the (original, un-masked) panel slice.
    # ------------------------------------------------------------------
    def _encode_opponent_tokens(self, g: torch.Tensor) -> torch.Tensor:
        """``g`` is the (B, global_dim) flat globals; returns (B, MAX_OPPS, H)."""
        B = g.size(0)
        panel = g[:, self._panel_off : self._panel_off + self._panel_dim]
        n = self._max_opps
        hp = panel[:, 0:n].unsqueeze(-1)  # (B, n, 1)
        alive = panel[:, n : 2 * n].unsqueeze(-1)  # (B, n, 1)
        tribe = panel[:, 2 * n :].view(B, n, self._race_onehot_dim)  # (B, n, 9)
        opp_feat = torch.cat([hp, alive, tribe], dim=-1)  # (B, n, opp_feat_dim)

        E_opp = self.opp_proj(opp_feat)  # (B, n, H)
        pe = self.opp_pos_emb.weight[:n].unsqueeze(0)
        re = self.slot_region_emb.weight[_REG_ENEMY_ROW].view(1, 1, -1)
        return E_opp + pe + re

    # ------------------------------------------------------------------
    # encode_state: mirrors BGLikeStructuredV7.encode_state verbatim, with two
    # v10 deltas marked ``# v10:`` — opponent tokens joined to the attention,
    # and a thinking core refining the trunk. v7/v6/v9 helpers are inherited.
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

        saved_obs_dim = self.obs_dim
        self.obs_dim = _OBS_DIM_V3
        try:
            g, own, shop, hand, _enemy, lb, phase, pending = self._unpack(head)
        finally:
            self.obs_dim = saved_obs_dim

        # ---- Per-region ability summaries (v6) ----
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

        # ---- Per-minion encoder (v6) ----
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

        opt_stack = _pending_three_option_emb(
            pending,
            self.card_emb,
            self.adapt_choice_emb,
            max_card_idx=self.num_pool_indices,
        )
        is_apply = (
            pending[..., _PENDING_IS_APPLY_OFFSET : _PENDING_IS_APPLY_OFFSET + 1] > 0.5
        )
        opt_stack = opt_stack.masked_fill(is_apply.unsqueeze(-1), 0.0)
        pending_input = torch.cat([opt_stack, abil_pending], dim=-1)
        E_pending = self._add_pos_region(
            self.pending_to_slot(pending_input), self.pending_pos_emb, REG_PENDING
        )

        pending_header = pending[..., : self._pending_header_dim]

        # v10: opponent tokens are built from the ORIGINAL panel, then the panel
        # is zeroed in the flat globals so it reaches the heads only via the
        # tokens → CLS attention path (compression, no flat double-count).
        if self._has_panel:
            E_opp = self._encode_opponent_tokens(g)
            g = g.clone()
            g[:, self._panel_off : self._panel_off + self._panel_dim] = 0.0
        else:
            E_opp = E_own.new_zeros(B, 0, self.slot_hidden)

        g_full = torch.cat([g, lb, phase], dim=-1)  # panel-zeroed

        # ---- CLS token (+ zero-init identity term, v7) ----
        cls_base = self.cls_from_globals(g_full)
        if id_onehot is not None:
            cls_base = cls_base + self.identity_proj(id_onehot)
        cls_tok = cls_base.unsqueeze(1)

        # ---- Per-slot identity gate (v7) ----
        if id_onehot is not None:
            id_e = self.identity_emb_proj(id_onehot)

            def _gate(E):
                Bn, Ln, Hn = E.shape
                cat = torch.cat([E, id_e.unsqueeze(1).expand(Bn, Ln, Hn)], dim=-1)
                return E * (1.0 + self.identity_slot_gate(cat))

            E_own = _gate(E_own)
            E_shop = _gate(E_shop)
            E_hand = _gate(E_hand)
            E_pending = _gate(E_pending)

        # v10: opponent tokens appended at the TAIL of E_all so the existing
        # CLS / own / shop / hand / pending slice indices are unchanged.
        E_all = torch.cat(
            [cls_tok, E_own, E_shop, E_hand, E_pending, E_opp], dim=1
        )
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
        idx += self.pending_len
        E_opp = E_all[:, idx : idx + E_opp.shape[1]]
        E_enemy = E_all.new_zeros(B, 0, self.slot_hidden)

        state_summary = torch.cat([cls_out, g, lb, phase, pending_header], dim=-1)
        state_summary_n = self.state_summary_ln(state_summary)

        # v10: thinking core refines the trunk; both heads read the refined trunk.
        trunk = self.thinking_core(state_summary_n)
        state_emb = self.state_proj(trunk)

        cache: Dict[str, torch.Tensor] = {
            "E_own": E_own,
            "E_shop": E_shop,
            "E_hand": E_hand,
            "E_enemy": E_enemy,
            "E_pending": E_pending,
            "E_opp": E_opp,
            "trunk": trunk,
            "g_full": g_full,
        }
        return state_emb, cache


__all__ = ["BGLikeStructuredV10"]

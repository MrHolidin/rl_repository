"""Structured actor-critic, v11_heroes — v11 + hero-power awareness.

Thin subclass of :class:`BGLikeStructuredV11` for the ``bglike_v5_heroes`` obs
(``OBS_DIM_V5_HEROES`` = obs_v5 + a hero block). It adds exactly two things, in
the spirit of v11's SET/SCALAR split:

  * **self-hero = a SCALAR modality** — ``hero_encoder`` MLP over the 35-dim
    self-hero features, late-fused into the trunk next to economy/combat/pending
    (no per-slot gate: the hero reaches the actor globally via ``state_emb`` and
    the action interaction term);
  * **opponent hero = extra opponent-token features** — each opponent's hero
    one-hot is concatenated onto its ``[hp, alive, tribe]`` token (the obs emits
    them in the same HP-sorted order as the panel).

Everything else (entity attention, distributional critic, actor cross-attn,
board-order head) is inherited unchanged. The trunk-sized layers
(``state_summary_ln`` / ``thinking_core`` / ``state_proj`` / ``critic_dist``) are
rebuilt to absorb the extra ``hero_out`` width.

Obs contract: ``OBS_DIM_V5_HEROES`` (+ optional ``num_identities`` one-hot tail).
``obs_kind="bglike_v5_heroes"``; ``ppo_network_type="bglike_structured_v11_heroes"``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.bglike.obs import OBS_DIM as _OBS_DIM_HEAD
from src.envs.bglike.obs_v5 import OBS_DIM_V5
from src.envs.bglike.obs_v5_heroes import (
    HERO_BLOCK_DIM,
    HERO_SELF_DIM,
    NUM_HERO_OBS_IDS,
    OBS_DIM_V5_HEROES,
)
from src.envs.minibg.obs import (
    PENDING_DISCOVER_IDX_DIM,
    PENDING_DISCOVER_IDX_OFFSET,
    PENDING_IS_APPLY_OFFSET,
)

from .bglike_structured_v11 import BGLikeStructuredV11, NUM_PLACEMENTS, _ReZeroMLPBlock


class BGLikeStructuredV11Heroes(BGLikeStructuredV11):
    """v11 + hero scalar modality + opponent-hero token features."""

    def __init__(
        self,
        *,
        hero_hidden: int = 48,
        hero_out: int = 24,
        **v11_kwargs: Any,
    ) -> None:
        super().__init__(**v11_kwargs)
        self.hero_hidden = int(hero_hidden)
        self.hero_out = int(hero_out)
        self.hero_self_dim = int(HERO_SELF_DIM)
        self.num_hero_obs = int(NUM_HERO_OBS_IDS)
        self.obs_dim = int(OBS_DIM_V5_HEROES)

        # Self-hero scalar encoder (late-fused into the trunk).
        self.hero_encoder = nn.Sequential(
            nn.Linear(self.hero_self_dim, self.hero_hidden),
            nn.GELU(),
            nn.Linear(self.hero_hidden, self.hero_out),
        )

        # Opponent tokens gain the opponent's hero one-hot.
        self.opp_proj = nn.Linear(self._opp_feat_dim + self.num_hero_obs, self.slot_hidden)
        nn.init.normal_(self.opp_proj.weight, std=0.02)
        nn.init.zeros_(self.opp_proj.bias)

        # Trunk grows by hero_out → rebuild the trunk-sized layers.
        self._state_summary_dim = self._state_summary_dim + self.hero_out
        self.state_summary_ln = nn.LayerNorm(self._state_summary_dim)
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
        self.state_proj = nn.Linear(self._state_summary_dim, self.state_dim)
        self.critic_dist = nn.Sequential(
            nn.Linear(self._state_summary_dim, self.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.critic_hidden, NUM_PLACEMENTS),
        )
        nn.init.zeros_(self.critic_dist[-1].weight)
        nn.init.zeros_(self.critic_dist[-1].bias)
        # Intrinsic-value head (if enabled) must also absorb the grown trunk.
        if self.with_value_int:
            self._build_value_int()

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        kw = super().get_constructor_kwargs()
        kw["hero_hidden"] = self.hero_hidden
        kw["hero_out"] = self.hero_out
        return kw

    # ------------------------------------------------------------------
    # Obs split: [ obs_v5 | hero_block | identity? ]
    # ------------------------------------------------------------------
    def _split_hero_identity(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        w = x.shape[1]
        base = OBS_DIM_V5_HEROES
        if w == base:
            rest, id_onehot = x, None
        elif w == base + self.num_identities:
            rest, id_onehot = x[:, :base], x[:, base:]
        else:
            raise ValueError(
                f"v11_heroes expected obs dim {base} or {base + self.num_identities}, got {w}"
            )
        xc = rest[:, :OBS_DIM_V5]
        hero_block = rest[:, OBS_DIM_V5:base]  # (B, HERO_BLOCK_DIM)
        return xc, hero_block, id_onehot

    def _encode_opponents_hero(self, g: torch.Tensor, opp_hero: torch.Tensor) -> torch.Tensor:
        B = g.size(0)
        panel = g[:, self._panel_off : self._panel_off + self._panel_dim]
        n = self._max_opps
        hp = panel[:, 0:n].unsqueeze(-1)
        alive = panel[:, n : 2 * n].unsqueeze(-1)
        tribe = panel[:, 2 * n :].view(B, n, self._race_onehot_dim)
        feat = torch.cat([hp, alive, tribe, opp_hero], dim=-1)
        return self.opp_proj(feat) + self.opp_pos_emb.weight[:n].unsqueeze(0)

    # ------------------------------------------------------------------
    # State encoder (v11 body + hero scalar + opponent-hero tokens).
    # ------------------------------------------------------------------
    def encode_state(self, x: torch.Tensor):
        xc, hero_block, id_onehot = self._split_hero_identity(x)
        head = xc[:, :_OBS_DIM_HEAD]
        g, own, shop, hand, lb, phase, pending = self._unpack_head(head)
        B = xc.size(0)

        abil = self._unpack_abilities(xc)
        o_own = 0
        o_shop = o_own + self.own_len * self.k_abil
        o_hand = o_shop + self.shop_len * self.k_abil
        o_pend = o_hand + self.hand_len * self.k_abil

        E_own = self._encode_region(own, self._ability_summary(abil, o_own, self.own_len), self.own_pos_emb)
        E_shop = self._encode_region(shop, self._ability_summary(abil, o_shop, self.shop_len), self.shop_pos_emb)
        E_hand = self._encode_region(hand, self._ability_summary(abil, o_hand, self.hand_len), self.hand_pos_emb)

        disc_idx = pending[
            ..., PENDING_DISCOVER_IDX_OFFSET : PENDING_DISCOVER_IDX_OFFSET + PENDING_DISCOVER_IDX_DIM
        ].long().clamp_(min=0, max=self.num_pool_indices)
        opt = self.card_emb(disc_idx)
        is_apply = pending[..., PENDING_IS_APPLY_OFFSET : PENDING_IS_APPLY_OFFSET + 1] > 0.5
        opt = opt.masked_fill(is_apply.unsqueeze(-1), 0.0)
        abil_pend = self._ability_summary(abil, o_pend, self.pending_len)
        E_pending = F.relu(self.pending_to_slot(torch.cat([opt, abil_pend], dim=-1)))
        E_pending = E_pending + self.pending_pos_emb.weight[: self.pending_len].unsqueeze(0)

        # Opponents → set tokens, with each opponent's hero one-hot fused in.
        self_hero = hero_block[:, :HERO_SELF_DIM]
        opp_hero = hero_block[:, HERO_SELF_DIM:HERO_BLOCK_DIM].view(B, self._max_opps, self.num_hero_obs)
        E_opp = self._encode_opponents_hero(g, opp_hero)

        # Scalar modalities (clean named slices; no carve/rejoin).
        econ = g[:, : self._econ_dim]
        battle_hist = g[:, self._panel_off + self._panel_dim :]
        combat = torch.cat([battle_hist, lb, phase], dim=-1)
        econ_emb = self.economy_encoder(econ)
        combat_emb = self.combat_proj(combat)
        pctx_emb = self.pending_ctx_proj(pending)
        hero_emb = self.hero_encoder(self_hero)  # self-hero scalar modality

        k = self.summary_queries
        query_tok = self.summary_query_emb.weight.unsqueeze(0).expand(B, -1, -1)

        if id_onehot is not None:
            id_e = self.identity_emb_proj(id_onehot)

            def _gate(E):
                Bn, Ln, Hn = E.shape
                cat = torch.cat([E, id_e.unsqueeze(1).expand(Bn, Ln, Hn)], dim=-1)
                return E * (1.0 + self.identity_slot_gate(cat))

            E_own, E_shop, E_hand, E_pending = _gate(E_own), _gate(E_shop), _gate(E_hand), _gate(E_pending)
            E_opp = _gate(E_opp)

        E_all = torch.cat([query_tok, E_own, E_shop, E_hand, E_pending, E_opp], dim=1)
        for block in self.entity_attn:
            E_all = block(E_all)

        summary = E_all[:, :k].reshape(B, k * self.slot_hidden)
        idx = k
        E_own = E_all[:, idx : idx + self.own_len]; idx += self.own_len
        E_shop = E_all[:, idx : idx + self.shop_len]; idx += self.shop_len
        E_hand = E_all[:, idx : idx + self.hand_len]; idx += self.hand_len
        E_pending = E_all[:, idx : idx + self.pending_len]; idx += self.pending_len
        E_opp = E_all[:, idx : idx + self._max_opps]

        # Late-fuse scalars: pooled set summary ⊕ economy ⊕ combat ⊕ pending ⊕ hero.
        trunk_in = torch.cat([summary, econ_emb, combat_emb, pctx_emb, hero_emb], dim=-1)
        state_summary_n = self.state_summary_ln(trunk_in)
        trunk = self.thinking_core(state_summary_n)
        state_emb = self.state_proj(trunk)

        cache: Dict[str, torch.Tensor] = {
            "E_own": E_own,
            "E_shop": E_shop,
            "E_hand": E_hand,
            "E_pending": E_pending,
            "E_opp": E_opp,
            "E_enemy": E_all.new_zeros(B, 0, self.slot_hidden),
            "trunk": trunk,
            "econ_emb": econ_emb,
            "g_full": econ_emb,
        }
        return state_emb, cache


__all__ = ["BGLikeStructuredV11Heroes"]

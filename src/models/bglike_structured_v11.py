"""Structured actor-critic, v11 — flat, consolidated CLEAN version.

v11 is a single self-contained module (NOT a subclass of v2..v10). It is the
clean consolidation of the v2→v10 delta chain for the *bglike* layout, trained
**from scratch** — so all the warm-start scaffolding that justified the chain
(layer widening + zero-init deltas) is gone. The final architecture only:

  Encoder
    * per-minion encoder = ONE Linear over [slot continuous ⊕ card_emb ⊕
      per-host ability summary]  (v6's abilities-as-channels, but a single
      projection instead of 2 stacked kernel-1 "convs", and a real Linear
      instead of Conv1d — cross-slot mixing is the attention's job);
    * opponents are 7 attention TOKENS (v10), NOT 77 flat globals;
    * per-region position embeddings tag region identity (no separate
      slot_region_emb — the region is implied by which pos table is used);
    * SETS vs SCALARS split (AlphaStar-style), not "everything a token":
        - SET tokens in the self-attention: minions, opponents, the 3 discover
          options, + k PURE learned summary queries (PMA / Perceiver) that pool
          them. k post-attention query outputs = the set summary (k·H).
        - SCALAR modalities get their own MLP and are LATE-FUSED into the trunk
          (concat), never forced into the set-attention: ``economy_encoder``
          (core+rotation, 25 → economy_out=32 via a 2-layer MLP — the dedicated
          economy encoder), ``combat_proj`` (battle history + last_battle +
          phase, 15 → small), ``pending_ctx_proj`` (full pending: apply effect /
          masks / header, 31 → small).
    * per-region position embeddings tag region identity (no slot_region_emb);
      per-slot identity GATE (v7) — additive identity_proj (near-no-op, TV≈0.03)
      DROPPED; entity self-attention, **2 layers** (halved from 4-layer configs).

  Trunk
    * trunk = [ set_summary(k·H) | economy_emb(32) | combat_emb | pending_ctx_emb ].
      Sets pooled by attention; scalars encoded by their own MLPs and late-fused
      (encoded-then-concat is fine — AlphaStar; the old sin was RAW concat). No
      raw flat globals/pending anywhere → no LN skew.
    * ReZero "thinking" core (v10), non-zero α — mixes the set summary with the
      scalar embeddings and refines the trunk for both heads.

  Heads
    * distributional 8-way placement critic (v8); the unused scalar critic is gone;
    * actor: action token = type+role+gathered-entity + economy term (v9, now
      from the ENCODED economy embedding — single source of truth), cross-attends
      to the entity sequence (v3); score MLP = [state_emb, ae, interaction], no g_full;
    * board-order GRU pointer (v2) — g_full arg kept in the signature for the
      shared agent but ignored internally (no flat g).

Removed vs v10: flat lobby panel (77), scalar critic, battle-prediction head,
enemy-region plumbing, minibg layout branch, additive identity_proj, 2nd conv
layer + kernel=3 option, slot_region_emb, adapt support (adapt_choice_emb and
the adapt branch of the pending encoder — patch 74257 has no adapt cards).

NOTE on attention depth: a saturated checkpoint showed all 4 layers alive
(gates 0.10–0.15, last layer most active) — the 4→2 cut is a deliberate
override of that evidence for speed, to be confirmed by a 2-vs-4 ablation.

Obs contract: ``OBS_DIM_V5`` (+ optional ``num_identities`` one-hot tail).
``obs_kind="bglike_v5"``. Own ``ppo_network_type`` (``bglike_structured_v11``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.envs.bglike.actions import (
    BOARD_SIZE as _BOARD_SIZE,
    HAND_SIZE as _HAND_SIZE,
    MAX_SHOP_SLOTS as _SHOP_SLOTS,
)
from src.envs.bglike.obs import (
    BGLIKE_GLOBAL_CORE_DIM,
    BGLIKE_GLOBAL_DIM,
    LOBBY_PANEL_DIM,
    MAX_OPPS,
    OBS_DIM as _OBS_DIM_HEAD,
)
from src.envs.bglike.obs_v5 import (
    ABIL_BLOCK_OFFSET,
    ABIL_FEAT_DIM,
    ABIL_OFF_EFFECT,
    K_ABIL,
    OBS_DIM_V5,
    PENDING_LEN,
)
from src.envs.minibg.obs import (
    PENDING_CHOICE_DIM,
    PENDING_DISCOVER_IDX_DIM,
    PENDING_DISCOVER_IDX_OFFSET,
    PENDING_IS_APPLY_OFFSET,
    RACE_ONEHOT_DIM,
    SHOP_ROTATION_OBS_DIM,
)
from src.envs.minibg.structured_actions import StructAction

from .bglike_structured_v5 import AbilityTokenEncoder
from .minibg_slot_ac import (
    _LAST_BATTLE_DIM,
    _PHASE_DIM,
    _SLOT_CONT_DIM,
    _SLOT_DIM,
    _split_card_idx_and_cont,
)
from .structured_common import (
    CrossAttentionBlock,
    EntityAttentionBlock,
    _build_action_tokens,
    _NUM_REGIONS,
    _NUM_ROLES,
    _NUM_STRUCT_TYPES,
    _REGION_HAND,
    _REGION_NULL,
    _REGION_OWN,
    _REGION_PENDING,
    _REGION_SHOP,
)

NUM_PLACEMENTS = 8


def _placement_reward_vec() -> torch.Tensor:
    from src.envs.bglike.placement import placement_reward

    return torch.tensor(
        [placement_reward(p) for p in range(1, NUM_PLACEMENTS + 1)],
        dtype=torch.float32,
    )


class _ReZeroMLPBlock(nn.Module):
    """``x + alpha * fc2(gelu(fc1(LN(x))))`` with a per-feature ReZero gate
    initialised non-zero (from-scratch training: a zero-init branch can sit at a
    lazy minimum and never wake up)."""

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


class BGLikeStructuredV11(nn.Module):
    """Flat clean structured actor-critic (bglike, obs_v5)."""

    def __init__(
        self,
        *,
        slot_hidden: int = 64,
        state_dim: int = 128,
        action_dim: int = 64,
        interaction_dim: int = 64,
        order_hidden: int = 64,
        order_pos_dim: int = 16,
        score_hidden: int = 128,
        order_score_hidden: int = 64,
        critic_hidden: int = 128,
        card_emb_dim: int = 16,
        ability_emb_dim: int = 8,
        entity_attention_layers: int = 2,
        entity_attention_heads: int = 4,
        entity_attention_ff_mult: int = 4,
        entity_attention_init_scale: float = 0.1,
        action_cross_attn_heads: int = 4,
        action_cross_attn_ff_mult: int = 2,
        action_cross_attn_init_scale: float = 0.1,
        thinking_hidden: int = 256,
        thinking_blocks: int = 1,
        thinking_init_alpha: float = 0.1,
        summary_queries: int = 2,
        economy_hidden: int = 64,
        economy_out: int = 32,
        combat_out: int = 16,
        pending_ctx_out: int = 16,
        num_identities: int = 4,
        num_pool_indices: Optional[int] = None,
        # --- accepted-and-ignored (obsolete in v11; kept so the shared agent /
        #     existing configs can pass them without erroring) ---
        region_conv2_kernel: Any = None,
        obs_layout: Any = None,
        battle_pred_config: Any = None,
    ) -> None:
        super().__init__()
        if num_pool_indices is None:
            raise ValueError("num_pool_indices is required")

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
        self.ability_emb_dim = int(ability_emb_dim)
        self.entity_attention_layers = int(entity_attention_layers)
        self.entity_attention_heads = int(entity_attention_heads)
        self.entity_attention_ff_mult = int(entity_attention_ff_mult)
        self.entity_attention_init_scale = float(entity_attention_init_scale)
        self.action_cross_attn_heads = int(action_cross_attn_heads)
        self.action_cross_attn_ff_mult = int(action_cross_attn_ff_mult)
        self.action_cross_attn_init_scale = float(action_cross_attn_init_scale)
        self.thinking_hidden = int(thinking_hidden)
        self.thinking_blocks = int(thinking_blocks)
        self.thinking_init_alpha = float(thinking_init_alpha)
        self.summary_queries = int(summary_queries)
        self.economy_hidden = int(economy_hidden)
        self.economy_out = int(economy_out)
        self.combat_out = int(combat_out)
        self.pending_ctx_out = int(pending_ctx_out)
        self.num_identities = int(num_identities)
        self.num_pool_indices = int(num_pool_indices)
        self.k_abil = int(K_ABIL)
        self.abil_feat_dim = int(ABIL_FEAT_DIM)

        # ---- Region geometry (bglike obs_v5) --------------------------
        self.own_len = int(_BOARD_SIZE)
        self.shop_len = int(_SHOP_SLOTS)
        self.hand_len = int(_HAND_SIZE)
        self.pending_len = int(PENDING_LEN)
        self.board_size = int(_BOARD_SIZE)
        self.max_region_len = max(
            self.own_len, self.shop_len, self.hand_len, self.pending_len, 1
        )

        # v3-head globals geometry + panel slice.
        self.global_dim = int(BGLIKE_GLOBAL_DIM)              # 115
        self._panel_off = int(BGLIKE_GLOBAL_CORE_DIM) + int(SHOP_ROTATION_OBS_DIM)  # 25
        self._panel_dim = int(LOBBY_PANEL_DIM)                # 77
        self._max_opps = int(MAX_OPPS)                        # 7
        self._race_onehot_dim = int(RACE_ONEHOT_DIM)          # 9
        self._opp_feat_dim = 2 + self._race_onehot_dim        # hp, alive, tribe
        # Scalar modalities (sliced cleanly from the flat globals, no carve/rejoin):
        #   economy = core + rotation  (prefix)
        #   combat  = battle_history (after the panel) + last_battle + phase
        self._econ_dim = int(BGLIKE_GLOBAL_CORE_DIM) + int(SHOP_ROTATION_OBS_DIM)  # 25
        self._battle_hist_dim = self.global_dim - (self._panel_off + self._panel_dim)  # 13
        self._combat_dim = self._battle_hist_dim + int(_LAST_BATTLE_DIM) + int(_PHASE_DIM)  # 15
        self._pending_dim = int(PENDING_CHOICE_DIM)           # 31

        self.obs_dim = int(OBS_DIM_V5)

        # ---- Embeddings -----------------------------------------------
        self.card_emb = nn.Embedding(
            self.num_pool_indices + 1, self.card_emb_dim, padding_idx=0
        )
        nn.init.normal_(self.card_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.card_emb.weight[0].zero_()

        # ---- Per-minion encoder: ONE Linear over [cont ⊕ card_emb ⊕ ability]
        self._slot_in_card = _SLOT_CONT_DIM + self.card_emb_dim          # 35+16=51
        self.slot_proj = nn.Linear(self._slot_in_card + self.slot_hidden, self.slot_hidden)

        # Abilities: encode K tokens, single-query softmax pool → slot_hidden.
        self.ability_encoder = AbilityTokenEncoder(
            slot_hidden=self.slot_hidden,
            emb_dim=self.ability_emb_dim,
            card_emb_dim=self.card_emb_dim,
        )
        self.ability_pool_query = nn.Parameter(torch.empty(self.slot_hidden))
        nn.init.normal_(self.ability_pool_query, mean=0.0, std=0.02)

        # Pending option encoder: [discover card_emb ⊕ ability summary] → slot.
        self.pending_to_slot = nn.Linear(self.card_emb_dim + self.slot_hidden, self.slot_hidden)

        # Opponent token encoder.
        self.opp_proj = nn.Linear(self._opp_feat_dim, self.slot_hidden)
        nn.init.normal_(self.opp_proj.weight, std=0.02)
        nn.init.zeros_(self.opp_proj.bias)

        # ---- Position embeddings (region implied by which table is used) ----
        self.own_pos_emb = nn.Embedding(self.own_len, self.slot_hidden)
        self.shop_pos_emb = nn.Embedding(self.shop_len, self.slot_hidden)
        self.hand_pos_emb = nn.Embedding(self.hand_len, self.slot_hidden)
        self.pending_pos_emb = nn.Embedding(self.pending_len, self.slot_hidden)
        self.opp_pos_emb = nn.Embedding(self._max_opps, self.slot_hidden)
        for emb in (
            self.own_pos_emb, self.shop_pos_emb, self.hand_pos_emb,
            self.pending_pos_emb, self.opp_pos_emb,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        # ---- Summary queries: PURE learned PMA seeds (no globals seed) -----
        # k content-free learned query tokens pool the SET tokens (minions /
        # opponents / discover-options) into a k×H summary. Globals are NOT a
        # seed anymore — they're separate scalar modalities, late-fused below.
        self.summary_query_emb = nn.Embedding(self.summary_queries, self.slot_hidden)
        nn.init.normal_(self.summary_query_emb.weight, mean=0.0, std=0.02)

        # ---- Scalar modality encoders (AlphaStar-style; late-fused into trunk)
        # Economy gets a real 2-layer MLP at full width (high-value for the
        # critic + the actor's economy-conditioned queries). Combat-history and
        # pending-context are secondary / mostly-inactive → small outputs.
        self.economy_encoder = nn.Sequential(
            nn.Linear(self._econ_dim, self.economy_hidden),
            nn.GELU(),
            nn.Linear(self.economy_hidden, self.economy_out),
        )
        self.combat_proj = nn.Linear(self._combat_dim, self.combat_out)
        self.pending_ctx_proj = nn.Linear(self._pending_dim, self.pending_ctx_out)
        # Per-slot identity gate: identity × each slot's content.
        self.identity_emb_proj = nn.Linear(self.num_identities, self.slot_hidden)
        nn.init.normal_(self.identity_emb_proj.weight, std=0.02)
        nn.init.zeros_(self.identity_emb_proj.bias)
        self.identity_slot_gate = nn.Linear(2 * self.slot_hidden, self.slot_hidden)
        nn.init.normal_(self.identity_slot_gate.weight, std=0.02)
        nn.init.zeros_(self.identity_slot_gate.bias)

        # ---- Entity self-attention (2 layers) -----------------------------
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

        # ---- Trunk: k pooled set-summary ⊕ late-fused scalar embeddings ----
        #   [ summary(k·H) | economy(H) | combat(combat_out) | pending_ctx(pctx_out) ]
        self._state_summary_dim = (
            self.summary_queries * self.slot_hidden
            + self.economy_out
            + self.combat_out
            + self.pending_ctx_out
        )
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

        # ---- Distributional placement critic ------------------------------
        self.critic_dist = nn.Sequential(
            nn.Linear(self._state_summary_dim, self.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.critic_hidden, NUM_PLACEMENTS),
        )
        nn.init.zeros_(self.critic_dist[-1].weight)
        nn.init.zeros_(self.critic_dist[-1].bias)
        self.register_buffer("placement_reward_vec", _placement_reward_vec())

        # ---- Action embedding + economy term + cross-attn ----------------
        self.type_emb = nn.Embedding(_NUM_STRUCT_TYPES, self.action_dim)
        self.role_emb = nn.Embedding(_NUM_ROLES, self.action_dim)
        self.entity_to_action = nn.Linear(self.slot_hidden, self.action_dim)
        self.entity_to_action_tgt = nn.Linear(self.slot_hidden, self.action_dim)
        self.null_entity_action = nn.Parameter(torch.zeros(self.action_dim))
        # Economy-conditioned action queries read the ENCODED economy embedding
        # (same economy_encoder output as the trunk — single source of truth).
        self.action_econ_proj = nn.Linear(self.economy_out, self.action_dim)

        self.action_to_slot = nn.Linear(self.action_dim, self.slot_hidden)
        self.action_cross_attn = CrossAttentionBlock(
            self.slot_hidden,
            num_heads=self.action_cross_attn_heads,
            ff_mult=self.action_cross_attn_ff_mult,
            init_scale=self.action_cross_attn_init_scale,
        )
        self.slot_to_action = nn.Linear(self.slot_hidden, self.action_dim)

        self.state_to_interact = nn.Linear(self.state_dim, self.interaction_dim, bias=False)
        self.action_to_interact = nn.Linear(self.action_dim, self.interaction_dim, bias=False)

        # Score MLP: NO flat g_full (globals via state_emb, economy via token).
        self.score_fc = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim + self.interaction_dim, self.score_hidden),
            nn.ReLU(),
            nn.Linear(self.score_hidden, 1),
        )

        # ---- Board-order pointer head (no flat g_full) --------------------
        self.order_pos_emb = nn.Embedding(self.board_size, self.order_pos_dim)
        self.order_start = nn.Parameter(torch.zeros(self.slot_hidden))
        self.order_init = nn.Linear(self.state_dim, self.order_hidden)
        self.order_gru = nn.GRUCell(self.slot_hidden + self.order_pos_dim, self.order_hidden)
        self.order_score = nn.Sequential(
            nn.Linear(
                self.state_dim + self.slot_hidden + self.order_hidden + self.order_pos_dim,
                self.order_score_hidden,
            ),
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
            "card_emb_dim": self.card_emb_dim,
            "ability_emb_dim": self.ability_emb_dim,
            "entity_attention_layers": self.entity_attention_layers,
            "entity_attention_heads": self.entity_attention_heads,
            "entity_attention_ff_mult": self.entity_attention_ff_mult,
            "entity_attention_init_scale": self.entity_attention_init_scale,
            "action_cross_attn_heads": self.action_cross_attn_heads,
            "action_cross_attn_ff_mult": self.action_cross_attn_ff_mult,
            "action_cross_attn_init_scale": self.action_cross_attn_init_scale,
            "thinking_hidden": self.thinking_hidden,
            "thinking_blocks": self.thinking_blocks,
            "thinking_init_alpha": self.thinking_init_alpha,
            "summary_queries": self.summary_queries,
            "economy_hidden": self.economy_hidden,
            "economy_out": self.economy_out,
            "combat_out": self.combat_out,
            "pending_ctx_out": self.pending_ctx_out,
            "num_identities": self.num_identities,
            "num_pool_indices": self.num_pool_indices,
        }

    # ------------------------------------------------------------------
    # Obs unpacking
    # ------------------------------------------------------------------
    def _split_identity(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        w = x.shape[1]
        if w == OBS_DIM_V5:
            return x, None
        if w == OBS_DIM_V5 + self.num_identities:
            return x[:, :OBS_DIM_V5], x[:, OBS_DIM_V5:]
        raise ValueError(
            f"v11 expected obs dim {OBS_DIM_V5} or {OBS_DIM_V5 + self.num_identities}, got {w}"
        )

    def _unpack_head(self, head: torch.Tensor):
        """Slice the v3-head (``OBS_DIM``) into g / own / shop / hand / lb / phase / pending."""
        if head.shape[1] != _OBS_DIM_HEAD:
            raise ValueError(f"expected head dim {_OBS_DIM_HEAD}, got {head.shape[1]}")
        g = head[:, : self.global_dim]
        i = self.global_dim
        own = head[:, i : i + self.own_len * _SLOT_DIM].view(-1, self.own_len, _SLOT_DIM)
        i += self.own_len * _SLOT_DIM
        shop = head[:, i : i + self.shop_len * _SLOT_DIM].view(-1, self.shop_len, _SLOT_DIM)
        i += self.shop_len * _SLOT_DIM
        hand = head[:, i : i + self.hand_len * _SLOT_DIM].view(-1, self.hand_len, _SLOT_DIM)
        i += self.hand_len * _SLOT_DIM
        lb = head[:, i : i + _LAST_BATTLE_DIM]
        i += _LAST_BATTLE_DIM
        phase = head[:, i : i + _PHASE_DIM]
        i += _PHASE_DIM
        pending = head[:, i : i + self._pending_dim]
        return g, own, shop, hand, lb, phase, pending

    def _unpack_abilities(self, x_full: torch.Tensor) -> torch.Tensor:
        tail = x_full[:, ABIL_BLOCK_OFFSET:]
        total = (
            self.own_len + self.shop_len + self.hand_len + self.pending_len
        ) * self.k_abil
        return tail.view(x_full.shape[0], total, self.abil_feat_dim)

    def _ability_summary(
        self, abil_raw: torch.Tensor, region_start: int, region_len: int
    ) -> torch.Tensor:
        B = abil_raw.shape[0]
        block = abil_raw[:, region_start : region_start + region_len * self.k_abil]
        block = block.view(B, region_len, self.k_abil, self.abil_feat_dim)
        encoded = self.ability_encoder(block, self.card_emb)  # (B, L, K, H)
        pad = block[..., ABIL_OFF_EFFECT] < 0.5  # (B, L, K) True=pad
        all_pad = pad.all(dim=-1, keepdim=True)
        q = self.ability_pool_query.view(1, 1, 1, -1)
        scores = (encoded * q).sum(dim=-1)
        scores = torch.where(
            all_pad.expand_as(scores), scores, scores.masked_fill(pad, float("-inf"))
        )
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (encoded * w).sum(dim=-2)
        return pooled.masked_fill(all_pad.expand_as(pooled), 0.0)

    # ------------------------------------------------------------------
    # Per-region encoders
    # ------------------------------------------------------------------
    def _encode_region(
        self, z_slots: torch.Tensor, ability_summary: torch.Tensor, pos_emb: nn.Embedding
    ) -> torch.Tensor:
        z = _split_card_idx_and_cont(z_slots, self.card_emb, max_card_idx=self.num_pool_indices)
        h = F.relu(self.slot_proj(torch.cat([z, ability_summary], dim=-1)))
        L = h.shape[1]
        return h + pos_emb.weight[:L].unsqueeze(0)

    def _encode_opponents(self, g: torch.Tensor) -> torch.Tensor:
        B = g.size(0)
        panel = g[:, self._panel_off : self._panel_off + self._panel_dim]
        n = self._max_opps
        hp = panel[:, 0:n].unsqueeze(-1)
        alive = panel[:, n : 2 * n].unsqueeze(-1)
        tribe = panel[:, 2 * n :].view(B, n, self._race_onehot_dim)
        feat = torch.cat([hp, alive, tribe], dim=-1)
        return self.opp_proj(feat) + self.opp_pos_emb.weight[:n].unsqueeze(0)

    # ------------------------------------------------------------------
    # State encoder
    # ------------------------------------------------------------------
    def encode_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        xc, id_onehot = self._split_identity(x)
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

        # Pending options (discover only; adapt removed).
        disc_idx = pending[
            ..., PENDING_DISCOVER_IDX_OFFSET : PENDING_DISCOVER_IDX_OFFSET + PENDING_DISCOVER_IDX_DIM
        ].long().clamp_(min=0, max=self.num_pool_indices)
        opt = self.card_emb(disc_idx)  # (B, 3, card_emb_dim)
        is_apply = pending[..., PENDING_IS_APPLY_OFFSET : PENDING_IS_APPLY_OFFSET + 1] > 0.5
        opt = opt.masked_fill(is_apply.unsqueeze(-1), 0.0)
        abil_pend = self._ability_summary(abil, o_pend, self.pending_len)
        E_pending = F.relu(self.pending_to_slot(torch.cat([opt, abil_pend], dim=-1)))
        E_pending = E_pending + self.pending_pos_emb.weight[: self.pending_len].unsqueeze(0)

        # Opponents → set tokens.
        E_opp = self._encode_opponents(g)

        # ---- Scalar modalities (clean named slices; no carve/rejoin) ----
        econ = g[:, : self._econ_dim]                                   # core+rot (25)
        battle_hist = g[:, self._panel_off + self._panel_dim :]         # after panel (13)
        combat = torch.cat([battle_hist, lb, phase], dim=-1)           # (15)
        econ_emb = self.economy_encoder(econ)                          # (B, H)
        combat_emb = self.combat_proj(combat)                          # (B, combat_out)
        pctx_emb = self.pending_ctx_proj(pending)                      # (B, pctx_out)

        # k PURE learned summary queries (content-free PMA seeds).
        k = self.summary_queries
        query_tok = self.summary_query_emb.weight.unsqueeze(0).expand(B, -1, -1)  # (B,k,H)

        if id_onehot is not None:
            id_e = self.identity_emb_proj(id_onehot)

            def _gate(E):
                Bn, Ln, Hn = E.shape
                cat = torch.cat([E, id_e.unsqueeze(1).expand(Bn, Ln, Hn)], dim=-1)
                return E * (1.0 + self.identity_slot_gate(cat))

            E_own, E_shop, E_hand, E_pending = _gate(E_own), _gate(E_shop), _gate(E_hand), _gate(E_pending)
            E_opp = _gate(E_opp)

        # Attention over SET tokens only (queries pool minions/shop/hand/options/opp).
        E_all = torch.cat([query_tok, E_own, E_shop, E_hand, E_pending, E_opp], dim=1)
        for block in self.entity_attn:
            E_all = block(E_all)

        summary = E_all[:, :k].reshape(B, k * self.slot_hidden)  # k pooled vectors
        idx = k
        E_own = E_all[:, idx : idx + self.own_len]; idx += self.own_len
        E_shop = E_all[:, idx : idx + self.shop_len]; idx += self.shop_len
        E_hand = E_all[:, idx : idx + self.hand_len]; idx += self.hand_len
        E_pending = E_all[:, idx : idx + self.pending_len]; idx += self.pending_len
        E_opp = E_all[:, idx : idx + self._max_opps]

        # Late-fuse scalars (AlphaStar-style): pooled set summary ⊕ scalar embeddings.
        trunk_in = torch.cat([summary, econ_emb, combat_emb, pctx_emb], dim=-1)
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
            "econ_emb": econ_emb,            # → economy-conditioned action queries
            "g_full": econ_emb,             # placeholder for the agent's order-head arg (ignored)
        }
        return state_emb, cache

    # ------------------------------------------------------------------
    # Distributional critic
    # ------------------------------------------------------------------
    def placement_logits(self, trunk: torch.Tensor) -> torch.Tensor:
        return self.critic_dist(trunk)

    def value_from_trunk(self, trunk: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.placement_logits(trunk), dim=-1)
        return (probs * self.placement_reward_vec).sum(dim=-1)

    # ------------------------------------------------------------------
    # Action encoding (base + economy + cross-attn)
    # ------------------------------------------------------------------
    def _encode_actions(
        self,
        type_ids, role_ids, src_region_kinds, src_region_slots,
        tgt_region_kinds, tgt_region_slots, cache,
    ) -> torch.Tensor:
        B, Lmax = type_ids.shape
        device = type_ids.device
        dtype = cache["E_own"].dtype

        type_e = self.type_emb(type_ids)
        role_e = self.role_emb(role_ids)

        E_regions = torch.zeros(
            B, _NUM_REGIONS, self.max_region_len, self.slot_hidden, device=device, dtype=dtype
        )
        E_regions[:, _REGION_SHOP, : self.shop_len] = cache["E_shop"]
        E_regions[:, _REGION_OWN, : self.own_len] = cache["E_own"]
        E_regions[:, _REGION_HAND, : self.hand_len] = cache["E_hand"]
        E_regions[:, _REGION_PENDING, : self.pending_len] = cache["E_pending"]
        E_flat = E_regions.view(B, _NUM_REGIONS * self.max_region_len, self.slot_hidden)

        def _gather(kinds, slots, lin):
            flat_idx = kinds * self.max_region_len + slots
            ent_idx = flat_idx.unsqueeze(-1).expand(-1, -1, self.slot_hidden)
            ent = torch.gather(E_flat, dim=1, index=ent_idx)
            return lin(ent).masked_fill((kinds == _REGION_NULL).unsqueeze(-1), 0.0)

        src_ent = _gather(src_region_kinds, src_region_slots, self.entity_to_action)
        tgt_ent = _gather(tgt_region_kinds, tgt_region_slots, self.entity_to_action_tgt)
        ent_null = self.null_entity_action.view(1, 1, -1).expand(B, Lmax, -1)
        both_null = (
            (src_region_kinds == _REGION_NULL) & (tgt_region_kinds == _REGION_NULL)
        ).unsqueeze(-1)
        ent = torch.where(both_null, ent_null, src_ent + tgt_ent)

        ae = type_e + role_e + ent
        # Economy term: the ENCODED economy embedding (single source of truth,
        # same economy_encoder output that feeds the trunk) → action query.
        ae = ae + self.action_econ_proj(cache["econ_emb"]).unsqueeze(1)

        # Cross-attend to the entity sequence (own/shop/hand/pending).
        entity_seq = torch.cat(
            [cache["E_own"], cache["E_shop"], cache["E_hand"], cache["E_pending"]], dim=1
        )
        ae_q = self.action_to_slot(ae)
        ae_attended = self.action_cross_attn(ae_q, entity_seq)
        return ae + self.slot_to_action(ae_attended)

    def _logits_from_state_and_tokens(
        self, state_emb, cache, type_ids, role_ids, src_region_kinds,
        src_region_slots, tgt_region_kinds, tgt_region_slots, mask,
    ) -> torch.Tensor:
        B, Lmax = type_ids.shape
        ae = self._encode_actions(
            type_ids, role_ids, src_region_kinds, src_region_slots,
            tgt_region_kinds, tgt_region_slots, cache,
        )
        s_exp = state_emb.unsqueeze(1).expand(-1, Lmax, -1)
        s_int = torch.tanh(self.state_to_interact(state_emb))
        a_int = torch.tanh(self.action_to_interact(ae))
        interaction = s_int.unsqueeze(1) * a_int
        h_all = torch.cat([s_exp, ae, interaction], dim=-1)
        logits = self.score_fc(h_all.reshape(B * Lmax, -1)).squeeze(-1).view(B, Lmax)
        return logits.masked_fill(~mask, float("-inf"))

    def policy_logits_value_from_tokens(
        self, obs, type_ids, role_ids, src_region_kinds, src_region_slots,
        tgt_region_kinds, tgt_region_slots, mask, *, return_cache: bool = False,
    ):
        state_emb, cache = self.encode_state(obs)
        logits = self._logits_from_state_and_tokens(
            state_emb, cache, type_ids, role_ids, src_region_kinds,
            src_region_slots, tgt_region_kinds, tgt_region_slots, mask,
        )
        values = self.value_from_trunk(cache["trunk"])
        if return_cache:
            cache_out = dict(cache)
            cache_out["state_emb"] = state_emb
            return logits, mask, values, cache_out
        return logits, mask, values

    def policy_logits_and_value(
        self, obs, legal_actions: List[List[StructAction]], *, return_cache: bool = False,
    ):
        B = obs.size(0)
        if len(legal_actions) != B:
            raise ValueError(f"legal_actions has {len(legal_actions)} rows but obs batch is {B}")
        Lmax = max((len(row) for row in legal_actions), default=0)
        if Lmax == 0:
            raise ValueError("policy_logits_and_value: every legal_actions row is empty")

        t_np, r_np, src_k, src_s, tgt_k, tgt_s, m_np = _build_action_tokens(legal_actions, Lmax)
        dev = obs.device
        return self.policy_logits_value_from_tokens(
            obs,
            torch.from_numpy(t_np).to(dev, non_blocking=True),
            torch.from_numpy(r_np).to(dev, non_blocking=True),
            torch.from_numpy(src_k).to(dev, non_blocking=True),
            torch.from_numpy(src_s).to(dev, non_blocking=True),
            torch.from_numpy(tgt_k).to(dev, non_blocking=True),
            torch.from_numpy(tgt_s).to(dev, non_blocking=True),
            torch.from_numpy(m_np).to(dev, non_blocking=True),
            return_cache=return_cache,
        )

    # ------------------------------------------------------------------
    # Board-order pointer head. ``g_full`` is accepted for signature parity
    # with the shared agent but deliberately UNUSED (no flat globals).
    # ------------------------------------------------------------------
    def _order_logits_step(self, state_emb, E_own, hidden, pos_e_row) -> torch.Tensor:
        B, K, _ = E_own.shape
        sh = hidden.unsqueeze(1).expand(-1, K, -1)
        pe = pos_e_row.unsqueeze(1).expand(-1, K, -1)
        se = state_emb.unsqueeze(1).expand(-1, K, -1)
        h_cat = torch.cat([se, E_own, sh, pe], dim=-1)
        return self.order_score(h_cat).squeeze(-1)

    def _order_init_hidden(self, state_emb: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.order_init(state_emb))

    def sample_board_order(
        self, state_emb, E_own, g_full, occupied_mask, *, deterministic: bool = False,
    ):
        B, K, _ = E_own.shape
        if K != self.board_size:
            raise ValueError(f"E_own must have board_size={self.board_size} slots, got {K}")
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
            logits = self._order_logits_step(state_emb, E_own, hidden, pos_e_row)
            logits = logits.masked_fill(~remaining, float("-inf"))
            logits = logits.masked_fill((~active).unsqueeze(-1), 0.0)
            dist = Categorical(logits=logits)
            idx = logits.argmax(dim=-1) if deterministic else dist.sample()
            lp = dist.log_prob(idx)
            logprob_sum = logprob_sum + active.to(logprob_sum.dtype) * lp
            picked[batch_arange, pos] = torch.where(active, idx, torch.full_like(idx, -1))
            sel_emb = E_own[batch_arange, idx]
            slot_in = torch.where(active.unsqueeze(-1), sel_emb, slot_in)
            mask_pick = F.one_hot(idx, num_classes=K).bool() & active.unsqueeze(-1)
            remaining = remaining & ~mask_pick

        return picked, logprob_sum, remaining

    def order_logprob_given_sequence(
        self, state_emb, E_own, g_full, occupied_mask, picked_slots,
    ):
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
            logits = self._order_logits_step(state_emb, E_own, hidden, pos_e_row)
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


__all__ = ["BGLikeStructuredV11", "NUM_PLACEMENTS"]

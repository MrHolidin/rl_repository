"""Structured actor-critic, v5 = v3 + per-ability tokens.

Difference from v3: every minion (in own / shop / hand) and every minion-typed
pending discover option expands into up to ``K_ABIL`` extra **ability tokens**
that join the main entity self-attention. Each ability token packs
``(effect_class, trigger, condition_kind, condition_arg_race, filter_race,
combat_only, atk, hp, amount, repeats, count)`` — see
:mod:`src.envs.bglike.obs_v5` for the obs-side packing.

Motivation: v2/v3 encode abilities as a per-slot one-hot bitmap inside the slot
vector (``EFFECT_OFFSET`` block). That collapses **all** abilities on a minion
into a presence/absence bit per effect class and throws away parameters,
trigger filter, and ability-level condition gates. For sparse abilities like
``DiscoverMurlocEffect`` with ``OTHER_TRIBE_ON_BOARD: MURLOC`` (Primalfin
Lookout), the model can't see the gate, has to infer it indirectly, and ends
up never engaging — see the discover audit on the v3 checkpoint.

v5 lifts each ability into its own attention token so:
  * Conditions / filters / numeric params are first-class features instead of
    summary bits.
  * Multi-ability minions (Brann + something, AdaptSelfRandom + AddToken) get
    one token per ability instead of two bits in the same slot vector.
  * Discover options expose abilities the same way live minions do — so the
    DISCOVER_PICK action token attends to "this would give me a 4/3 with
    DiscoverMurloc", not just to ``card_emb[idx]``.

Implementation:
  * Obs adds a fixed ``(own + shop + hand + pending) * K_ABIL * ABIL_FEAT_DIM``
    float block at the tail (strict superset of v3 obs). ``obs_layout="bglike_v5"``.
  * Ability tokens are encoded via ``AbilityTokenEncoder`` (embedding lookups
    for the 5 categorical ids + linear projection on numerics) and tagged with
    region / host-slot / ability-index positional embeddings. The host-slot
    positional embeddings are **shared** with v3's per-region ``*_pos_emb``
    tables so the model can directly tie an ability back to its minion.
  * Padded ability tokens (effect_id == 0) are masked out of attention via
    ``key_padding_mask``. ``EntityAttentionBlock`` / ``CrossAttentionBlock``
    accept the mask (added in this commit, default ``None`` keeps v2/v3 forward
    behaviour byte-identical).
  * ``_encode_actions`` extends v3's ``entity_seq`` with ``E_abil`` and passes
    the matching key_padding_mask so action queries attend to live abilities
    only.

Checkpoint contract: v5 has its own ``ppo_network_type``
(``bglike_structured_v5``). Not interchangeable with v2/v3/v4 — different
obs shape AND new state_dict keys.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.envs.bglike.obs_v5 import (
    ABIL_BLOCK_OFFSET,
    ABIL_FEAT_DIM,
    ABIL_HAND_OFFSET,
    ABIL_OFF_ATK,
    ABIL_OFF_COMBAT_ONLY,
    ABIL_OFF_COND_ARG_RACE,
    ABIL_OFF_COND_KIND,
    ABIL_OFF_COUNT,
    ABIL_OFF_EFFECT,
    ABIL_OFF_EFFECT_KEYWORD,
    ABIL_OFF_EFFECT_TRIBE,
    ABIL_OFF_FILTER_RACE,
    ABIL_OFF_FILTER_VICTIM_KW,
    ABIL_OFF_SUMMON_TOKEN,
    ABIL_OFF_TRIGGER,
    ABIL_OWN_OFFSET,
    ABIL_PENDING_OFFSET,
    ABIL_SHOP_OFFSET,
    K_ABIL,
    NUM_CONDITION_KIND_IDS,
    NUM_EFFECT_IDS,
    NUM_KEYWORD_IDS,
    NUM_RACE_IDS,
    NUM_TRIGGER_IDS,
    OBS_DIM_V5,
    PENDING_LEN,
)
from src.envs.bglike.actions import (
    BOARD_SIZE as _BG_BOARD_SIZE,
    HAND_SIZE as _BG_HAND_SIZE,
    MAX_SHOP_SLOTS as _BG_SHOP_SLOTS,
)

from .bglike_structured_v3 import BGLikeStructuredV3
from .structured_common import (
    REG_HAND,
    REG_OWN,
    REG_PENDING,
    REG_SHOP,
)


class AbilityTokenEncoder(nn.Module):
    """Encode an ``(B, N, ABIL_FEAT_DIM)`` raw ability tensor → ``(B, N, slot_hidden)``.

    Categorical ids are looked up via ``Embedding(..., padding_idx=0)`` so
    padded tokens contribute zero:
      * effect, trigger, cond_kind — own tables.
      * cond_arg_race, filter_race, effect_tribe — share one ``race_emb``
        table (same vocab) but are concatenated separately so each role stays
        distinguishable downstream.
      * effect_keyword, filter_victim_keyword — share one ``kw_emb`` table.
      * summon_token — looked up in the **shared model ``card_emb``** (passed
        to ``forward``) so a summoned minion shares its representation with
        the same card in shop/hand/board. This is why ``forward`` needs the
        card embedding handed in.

    Numeric channels (combat_only + 5 stat/param floats) are concatenated raw
    and projected jointly with the embeddings.
    """

    def __init__(
        self,
        *,
        slot_hidden: int,
        emb_dim: int = 8,
        card_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.slot_hidden = int(slot_hidden)
        self.emb_dim = int(emb_dim)
        self.card_emb_dim = int(card_emb_dim)

        self.effect_emb = nn.Embedding(NUM_EFFECT_IDS, self.emb_dim, padding_idx=0)
        self.trigger_emb = nn.Embedding(NUM_TRIGGER_IDS, self.emb_dim, padding_idx=0)
        self.cond_kind_emb = nn.Embedding(
            NUM_CONDITION_KIND_IDS, self.emb_dim, padding_idx=0
        )
        self.race_emb = nn.Embedding(NUM_RACE_IDS, self.emb_dim, padding_idx=0)
        self.kw_emb = nn.Embedding(NUM_KEYWORD_IDS, self.emb_dim, padding_idx=0)

        for emb in (
            self.effect_emb,
            self.trigger_emb,
            self.cond_kind_emb,
            self.race_emb,
            self.kw_emb,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()

        # Embedding lookups concatenated for the projection:
        #   effect, trigger, cond_kind, cond_arg_race, filter_race,
        #   effect_tribe, effect_keyword, filter_victim_keyword  → 8 * emb_dim
        #   summon_token (shared card_emb)                        → card_emb_dim
        #   combat_only flag                                      → 1
        #   atk/hp/amount/repeats/count                           → 5
        feat_in = 8 * self.emb_dim + self.card_emb_dim + 1 + 5
        self.proj = nn.Linear(feat_in, self.slot_hidden)

    def forward(self, raw: torch.Tensor, card_emb: nn.Embedding) -> torch.Tensor:
        """``raw``: (..., ABIL_FEAT_DIM) float32. Returns (..., slot_hidden).

        ``card_emb`` is the model's shared card embedding table, used for the
        summon-token channel so summoned minions reuse the global card space.
        """
        eff = raw[..., ABIL_OFF_EFFECT].long().clamp_(min=0, max=NUM_EFFECT_IDS - 1)
        trg = raw[..., ABIL_OFF_TRIGGER].long().clamp_(min=0, max=NUM_TRIGGER_IDS - 1)
        ck = raw[..., ABIL_OFF_COND_KIND].long().clamp_(
            min=0, max=NUM_CONDITION_KIND_IDS - 1
        )
        car = raw[..., ABIL_OFF_COND_ARG_RACE].long().clamp_(
            min=0, max=NUM_RACE_IDS - 1
        )
        fr = raw[..., ABIL_OFF_FILTER_RACE].long().clamp_(
            min=0, max=NUM_RACE_IDS - 1
        )
        et = raw[..., ABIL_OFF_EFFECT_TRIBE].long().clamp_(min=0, max=NUM_RACE_IDS - 1)
        ekw = raw[..., ABIL_OFF_EFFECT_KEYWORD].long().clamp_(
            min=0, max=NUM_KEYWORD_IDS - 1
        )
        fvk = raw[..., ABIL_OFF_FILTER_VICTIM_KW].long().clamp_(
            min=0, max=NUM_KEYWORD_IDS - 1
        )
        st = raw[..., ABIL_OFF_SUMMON_TOKEN].long().clamp_(
            min=0, max=int(card_emb.num_embeddings) - 1
        )

        e_eff = self.effect_emb(eff)
        e_trg = self.trigger_emb(trg)
        e_ck = self.cond_kind_emb(ck)
        e_car = self.race_emb(car)
        e_fr = self.race_emb(fr)
        e_et = self.race_emb(et)
        e_ekw = self.kw_emb(ekw)
        e_fvk = self.kw_emb(fvk)
        e_st = card_emb(st)

        combat_only = raw[..., ABIL_OFF_COMBAT_ONLY : ABIL_OFF_COMBAT_ONLY + 1]
        # 5 numerics: ATK (0), HP (1), AMOUNT (2), REPEATS (3), COUNT (4).
        numerics = raw[..., ABIL_OFF_ATK : ABIL_OFF_COUNT + 1]

        feat = torch.cat(
            [e_eff, e_trg, e_ck, e_car, e_fr, e_et, e_ekw, e_fvk, e_st, combat_only, numerics],
            dim=-1,
        )
        return self.proj(feat)


class BGLikeStructuredV5(BGLikeStructuredV3):
    """v3 + per-ability token region inside entity self-attention."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        ability_emb_dim: int = 8,
        ability_attention_init_scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        # v3 only knows obs_layout in ("bglike", "minibg"). v5 always reads the
        # bglike v5 obs; we force the underlying init to "bglike" so the slot
        # encoder dims stay identical (slot vector itself is unchanged — the
        # ability block is appended *after* the existing obs).
        kwargs.setdefault("obs_layout", "bglike")
        if kwargs["obs_layout"] != "bglike":
            raise ValueError(
                f"BGLikeStructuredV5 requires obs_layout='bglike' (the underlying "
                f"slot encoder layout); v5 ability block is appended on top. Got "
                f"{kwargs['obs_layout']!r}."
            )
        super().__init__(**kwargs)

        # Override the obs_dim the parent stored to account for the appended
        # ability block. ``_unpack_v5`` reads the tail off the same flat vector.
        self.obs_dim = int(OBS_DIM_V5)
        self.ability_emb_dim = int(ability_emb_dim)
        self.ability_attention_init_scale = float(ability_attention_init_scale)

        self.k_abil = int(K_ABIL)
        self.abil_feat_dim = int(ABIL_FEAT_DIM)

        # Per-region slot counts the ability block carries (must match obs_v5).
        if (
            self.own_len != _BG_BOARD_SIZE
            or self.shop_len != _BG_SHOP_SLOTS
            or self.hand_len != _BG_HAND_SIZE
            or self.pending_len != PENDING_LEN
        ):
            raise RuntimeError(
                "v5 region sizes diverged from obs_v5 expectations: "
                f"own={self.own_len} shop={self.shop_len} hand={self.hand_len} "
                f"pending={self.pending_len}"
            )

        self.ability_encoder = AbilityTokenEncoder(
            slot_hidden=self.slot_hidden,
            emb_dim=self.ability_emb_dim,
            card_emb_dim=self.card_emb_dim,
        )

        # Ability index inside its host minion (0..K-1) — separate from the host
        # slot position so the model can tell apart "first ability of this card"
        # vs "second ability of same card".
        self.ability_index_emb = nn.Embedding(self.k_abil, self.slot_hidden)
        nn.init.normal_(self.ability_index_emb.weight, mean=0.0, std=0.02)

        # ``ability_pos_index`` (cached, non-learned) maps each ability token to
        # its host (region, slot_within_region, ability_index). We use it to
        # additively combine the parent's per-region positional embeddings with
        # the ability-index embedding when running attention.
        region_ids, host_slots, abil_idxs = self._build_ability_position_table()
        self.register_buffer("_abil_region_ids", region_ids, persistent=False)
        self.register_buffer("_abil_host_slots", host_slots, persistent=False)
        self.register_buffer("_abil_indices", abil_idxs, persistent=False)
        self._num_abil_tokens = int(region_ids.shape[0])

        # v5 cross-attention contribution: ``slot_to_action`` already exists from
        # v3 and is zero-initialized — v3 cross-attn now sees ability tokens too
        # (we extend ``entity_seq`` in ``_encode_actions``), so this path also
        # warms up from zero contribution and learns to use them.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_ability_position_table(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(region_ids, host_slot_ids, ability_indices)`` as int64
        tensors of length ``total_abil_tokens``. Order matches obs_v5 layout:
        own region first, then shop, hand, pending; for each minion slot we
        emit K_ABIL consecutive ability tokens with ability_index 0..K-1."""
        region_ids = []
        host_slots = []
        abil_idxs = []
        for region_id, region_len in (
            (REG_OWN, self.own_len),
            (REG_SHOP, self.shop_len),
            (REG_HAND, self.hand_len),
            (REG_PENDING, self.pending_len),
        ):
            for slot in range(region_len):
                for k in range(self.k_abil):
                    region_ids.append(region_id)
                    host_slots.append(slot)
                    abil_idxs.append(k)
        return (
            torch.tensor(region_ids, dtype=torch.long),
            torch.tensor(host_slots, dtype=torch.long),
            torch.tensor(abil_idxs, dtype=torch.long),
        )

    def _ability_position_emb(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """``(num_abil_tokens, slot_hidden)`` positional sum:
        ``slot_region_emb[region] + host_pos_emb[slot] + ability_index_emb[k]``.
        Uses the **same** per-region positional tables as the parent — so an
        ability token at (REG_SHOP, slot=2, k=0) shares its host-slot
        positional tag with the shop slot-2 entity token."""
        region_e = self.slot_region_emb(self._abil_region_ids.to(device))

        # Per-region host-slot positional emb. The parent stores 4 separate
        # embeddings; gather from the right table per region.
        host_e = torch.zeros(
            self._num_abil_tokens, self.slot_hidden, device=device, dtype=dtype
        )

        # Build the (own, shop, hand, pending) chunk index ranges once.
        # Iterating per region keeps the table writeable in-place.
        cursor = 0
        for region_len, pos_emb in (
            (self.own_len, self.own_pos_emb),
            (self.shop_len, self.shop_pos_emb),
            (self.hand_len, self.hand_pos_emb),
            (self.pending_len, self.pending_pos_emb),
        ):
            n_tokens = region_len * self.k_abil
            slot_ids = self._abil_host_slots[cursor : cursor + n_tokens].to(device)
            host_e[cursor : cursor + n_tokens] = pos_emb(slot_ids)
            cursor += n_tokens

        idx_e = self.ability_index_emb(self._abil_indices.to(device))
        return (region_e + host_e + idx_e).to(dtype)

    def _unpack_v5_abilities(self, x: torch.Tensor) -> torch.Tensor:
        """Slice the ability block off the end of the obs vector.

        Returns a single flat ``(B, num_abil_tokens, ABIL_FEAT_DIM)`` tensor
        in (own, shop, hand, pending) order — the same order as the position
        table built in ``_build_ability_position_table``."""
        if x.shape[1] != OBS_DIM_V5:
            raise ValueError(
                f"v5 expected obs dim {OBS_DIM_V5}, got {x.shape[1]}"
            )
        tail = x[:, ABIL_BLOCK_OFFSET:]
        B = x.shape[0]
        return tail.view(B, self._num_abil_tokens, self.abil_feat_dim)

    @staticmethod
    def _ability_pad_mask(abil_raw: torch.Tensor) -> torch.Tensor:
        """``(B, num_abil_tokens)`` bool, True where the token is padding.

        Convention: ``effect_id == 0`` (the +1-shifted pad id) means "no
        ability here" — the only signal we need to mask attention with.
        """
        return abil_raw[..., ABIL_OFF_EFFECT] < 0.5

    # ------------------------------------------------------------------
    # ``encode_state`` — extended for ability tokens
    # ------------------------------------------------------------------

    def encode_state(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Slice the v3 prefix and the v5 ability tail. v3's _unpack reads from
        # ``self.obs_dim``; we override that to OBS_DIM_V5, so feed v3's logic
        # only the head explicitly. The cleanest route is to re-implement
        # encode_state here since the ability tokens must join the same
        # attention pass as the slot tokens (otherwise CLS doesn't see them).

        # Imports kept local: pulling them at module load triggers the heavy
        # minibg path eagerly.
        from src.envs.minibg.obs import (
            PENDING_IS_APPLY_OFFSET as _PENDING_IS_APPLY_OFFSET,
        )
        from .minibg_slot_ac import _pending_three_option_emb

        if x.shape[1] != OBS_DIM_V5:
            raise ValueError(
                f"v5 expected obs dim {OBS_DIM_V5}, got {x.shape[1]}"
            )
        from src.envs.bglike.obs import OBS_DIM as _OBS_DIM_V3

        head = x[:, :_OBS_DIM_V3]

        # ---- v3-equivalent encoding (mirrors BGLikeStructuredV2.encode_state) ----
        # Temporarily fake the parent's obs_dim so _unpack accepts the v3-sized head.
        saved_obs_dim = self.obs_dim
        self.obs_dim = _OBS_DIM_V3
        try:
            g, own, shop, hand, _enemy, lb, phase, pending = self._unpack(head)
        finally:
            self.obs_dim = saved_obs_dim

        E_own = self._add_pos_region(
            self._encode_region_slots(own), self.own_pos_emb, REG_OWN
        )
        E_shop = self._add_pos_region(
            self._encode_region_slots(shop), self.shop_pos_emb, REG_SHOP
        )
        E_hand = self._add_pos_region(
            self._encode_region_slots(hand), self.hand_pos_emb, REG_HAND
        )

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

        pending_header = pending[..., : self._pending_header_dim]
        g_full = torch.cat([g, lb, phase], dim=-1)

        cls_tok = self.cls_from_globals(g_full).unsqueeze(1)

        # ---- v5: ability tokens ----
        abil_raw = self._unpack_v5_abilities(x)  # (B, N_abil, FEAT)
        abil_feat = self.ability_encoder(abil_raw, self.card_emb)  # (B, N_abil, slot_hidden)
        pos_emb = self._ability_position_emb(device=device, dtype=dtype)  # (N_abil, H)
        E_abil = abil_feat + pos_emb.unsqueeze(0)
        abil_pad_mask = self._ability_pad_mask(abil_raw)  # (B, N_abil)

        # ---- joint self-attention over CLS + slot tokens + ability tokens ----
        n_slot_tokens = 1 + self.own_len + self.shop_len + self.hand_len + self.pending_len
        E_all = torch.cat([cls_tok, E_own, E_shop, E_hand, E_pending, E_abil], dim=1)

        # CLS + slot tokens are always valid; only ability tail can be padded.
        slot_pad_mask = torch.zeros(B, n_slot_tokens, dtype=torch.bool, device=device)
        full_pad_mask = torch.cat([slot_pad_mask, abil_pad_mask], dim=1)

        for block in self.entity_attn:
            E_all = block(E_all, key_padding_mask=full_pad_mask)

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
        E_abil_out = E_all[:, idx : idx + self._num_abil_tokens]
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
            "E_abil": E_abil_out,
            "abil_pad_mask": abil_pad_mask,
        }
        return state_emb, cache

    # ------------------------------------------------------------------
    # Action scoring — extend cross-attn K/V with ability tokens
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
        # v3's _encode_actions = (v2 entity-gather) + v3 cross-attn over
        # (E_own | E_shop | E_hand | E_pending). v5 reuses the entity-gather
        # path from v2 but extends the cross-attn K/V with E_abil so action
        # tokens can read live ability tokens directly. The base ``ae`` (entity
        # gather + type/role) is identical to v3's pre-cross-attn output, so we
        # call up through the grandparent for that piece and re-implement the
        # cross-attn step here with the wider K/V.

        from .bglike_structured_v2 import BGLikeStructuredV2

        ae = BGLikeStructuredV2._encode_actions(
            self,
            type_ids,
            role_ids,
            src_region_kinds,
            src_region_slots,
            tgt_region_kinds,
            tgt_region_slots,
            cache,
        )

        entity_seq = torch.cat(
            [
                cache["E_own"],
                cache["E_shop"],
                cache["E_hand"],
                cache["E_pending"],
                cache["E_abil"],
            ],
            dim=1,
        )  # (B, N_ent + N_abil, slot_hidden)

        B = ae.shape[0]
        device = ae.device
        # Slot tokens always valid; only the ability tail carries padding.
        n_slot = (
            cache["E_own"].shape[1]
            + cache["E_shop"].shape[1]
            + cache["E_hand"].shape[1]
            + cache["E_pending"].shape[1]
        )
        slot_pad = torch.zeros(B, n_slot, dtype=torch.bool, device=device)
        kv_pad = torch.cat([slot_pad, cache["abil_pad_mask"]], dim=1)

        ae_q = self.action_to_slot(ae)
        ae_attended = self.action_cross_attn(ae_q, entity_seq, key_padding_mask=kv_pad)
        ae_extra = self.slot_to_action(ae_attended)
        return ae + ae_extra

    # ------------------------------------------------------------------
    # Constructor kwargs (for checkpoint restore)
    # ------------------------------------------------------------------

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        kw = super().get_constructor_kwargs()
        kw["ability_emb_dim"] = self.ability_emb_dim
        kw["ability_attention_init_scale"] = self.ability_attention_init_scale
        return kw


__all__ = ["BGLikeStructuredV5", "AbilityTokenEncoder"]

"""Structured actor-critic, v12 — card facts from a frozen static table.

v12 is v11_heroes with one change, made in both the observation and the
encoder: **everything a card template determines is looked up, not observed.**

  v11_heroes:  slot = [ continuous ⊕ learned card_emb(card_idx) ⊕ pooled
                        ability summary over K=4 padded ability tokens ]
  v12:         slot = [ continuous ⊕ static_table[card_idx, is_golden] ]

where ``static_table`` row = ``[ frozen rules-text embedding | ability
magnitudes from the DSL ]`` (see :mod:`src.envs.bglike.card_static`). The
table is not trained.

What this buys:

* the 1560-float ability tail leaves the observation (2683 → 1123, −58%) —
  it was re-encoding static card facts on every step;
* ``ability_encoder``, the single most expensive module in the v11 forward
  (19.5% of ``act_structured`` measured on CPU), collapses into one gather;
* the text vector groups cards by *mechanic* rather than identity, which the
  effect-id encoding cannot do — 73% of effect classes in this patch appear on
  exactly one card, so "effect" was an alias for "which card". Foe Reaper and
  Cave Hydra sit at cos 0.97 in text space; the three multiplier auras (Brann /
  Baron / Khadgar), which a kind/scope taxonomy maps to one point, separate at
  0.44–0.65.

What text does **not** carry, and therefore stays an explicit channel: tavern
tier (a linear probe scores 0.150 vs a 0.213 majority baseline — the tier is
not written on the card), magnitudes (changing a digit moves the sentence
embedding by 0.98 cosine, so the build script masks digits and the numbers
come from the DSL block), and everything runtime — keywords, divine shield,
current stats, golden, frozen.

``card_text_mode="random"`` swaps the text half for a fixed random frozen
matrix of the same shape. That run is the control: it isolates "the text
carries mechanics" from "restructuring card features into a static table
helped". A v12-vs-v11 win without it is unattributable.

Known blind spot: a magnetised mech carries abilities merged in from another
card, so its ``card_id`` no longer describes it (~3.9% of observed minions
after the golden half of the table is accounted for). See
:func:`src.envs.bglike.card_static.magnetic_divergence_note`.

Obs contract: ``OBS_DIM_V6_HEROES`` (+ optional ``num_identities`` one-hot
tail). ``obs_kind="bglike_v6_heroes"``; ``ppo_network_type="bglike_structured_v12"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.bglike.card_static import (
    NUM_DIM as _CARD_NUM_DIM,
    TEXT_MODE_RANDOM,
    TEXT_MODE_TEXT,
    TEXT_MODES,
    build_card_static_table,
)
from src.envs.bglike.obs import OBS_DIM as _OBS_DIM_HEAD
from src.envs.bglike.obs_v5_heroes import HERO_BLOCK_DIM, HERO_SELF_DIM
from src.envs.bglike.obs_v6_heroes import OBS_DIM_V6_HEROES
from src.envs.minibg.obs import (
    CARD_IDX_OFFSET as _CARD_IDX_OFFSET,
    GOLDEN_OFFSET as _GOLDEN_OFFSET,
    PENDING_DISCOVER_IDX_DIM,
    PENDING_DISCOVER_IDX_OFFSET,
    PENDING_IS_APPLY_OFFSET,
)

from .bglike_structured_v11 import _SLOT_CONT_DIM
from .bglike_structured_v11_heroes import BGLikeStructuredV11Heroes
from .structured_common import BattlePredictionHead

DEFAULT_CARD_TEXT_DIM = 32

# Battle labels are squashed as tanh(damage / DEFAULT_DAMAGE_NORM). Measured on
# 330 combats of a trained policy: mean |damage| 4.49, p90 11.0, max 16.0, so a
# norm of 10 maps the bulk of the distribution onto |t| <= 0.8 -- inside tanh's
# responsive range rather than against its flat tails.
DEFAULT_DAMAGE_NORM = 10.0
# Huber delta in that squashed space. Typical residuals are ~0.3, so this keeps
# most of the loss quadratic, matching what delta=5.0 did on raw damage.
DEFAULT_HUBER_DELTA = 0.33


class BGLikeStructuredV12(BGLikeStructuredV11Heroes):
    """v11_heroes with the ability tail replaced by a frozen per-card table."""

    def __init__(
        self,
        *,
        card_text_mode: str = TEXT_MODE_TEXT,
        card_text_dim: int = DEFAULT_CARD_TEXT_DIM,
        card_static_seed: int = 0,
        card_patch_dir: Optional[str] = None,
        battle_pred_config: Optional[Dict[str, Any]] = None,
        **v11_heroes_kwargs: Any,
    ) -> None:
        v11_heroes_kwargs.pop("battle_pred_config", None)  # v11 accepts-and-ignores
        super().__init__(**v11_heroes_kwargs)

        if card_text_mode not in TEXT_MODES:
            raise ValueError(f"card_text_mode={card_text_mode!r} not in {TEXT_MODES}")
        self.card_text_mode = str(card_text_mode)
        self.card_text_dim = int(card_text_dim)
        self.card_static_seed = int(card_static_seed)
        self.card_patch_dir = str(card_patch_dir) if card_patch_dir else None
        self.obs_dim = int(OBS_DIM_V6_HEROES)

        self._static_rows = self.num_pool_indices + 1
        self.card_row_dim = self.card_text_dim + int(_CARD_NUM_DIM)

        # The learned card embedding and the whole ability-token encoder are
        # what this version exists to replace — drop them rather than leave
        # dead parameters in the optimizer state.
        del self.card_emb
        del self.ability_encoder
        del self.ability_pool_query

        table, meta, ready = self._build_static_table()
        # persistent=True: the checkpoint carries the exact table it trained
        # with, so reloading needs neither the patch package nor the encoder.
        self.register_buffer("card_static", table, persistent=True)
        self._static_table_ready = ready
        self.card_static_meta = meta

        # The frozen row feeds the slot encoder directly — no projection in
        # between. A Linear(row_dim → k) before slot_proj would be either a
        # pure rank bottleneck (slot_proj is already linear over the
        # concatenation, so it can apply any linear map to the card part) or,
        # with an activation, an extra nonlinearity that v11 does not have in
        # its card path: there, AbilityTokenEncoder.proj is un-activated and
        # the single ReLU sits on slot_proj. Matching that depth keeps the
        # v11-vs-v12 comparison about the representation and nothing else.
        #
        # Sharing is unchanged too: v11 shared the card *table* (card_emb)
        # across slots and discover options while keeping slot_proj and
        # pending_to_slot separate. The frozen table is the shared part here.
        self.slot_proj = nn.Linear(_SLOT_CONT_DIM + self.card_row_dim, self.slot_hidden)
        self.pending_to_slot = nn.Linear(self.card_row_dim, self.slot_hidden)

        # ---- Auxiliary battle-outcome head (v11 dropped it; v12 restores it)
        # It predicts the signed uncapped damage of the resolved combat from
        # the two boards alone. The env already snapshots the boards and the
        # label on every combat-resolution step, and the agent already
        # back-fills them into the buffer — only the model side was missing.
        #
        # Why it fits v12 in particular: the slot encoder reads any raw
        # SLOT_DIM vector through the frozen card table, so the *opponent's*
        # board encodes with the same weights and needs no observation change.
        # The head therefore regularises exactly the slot encoder — the place
        # where card facts are read — and not the attention stack, which sits
        # downstream of it.
        self.battle_pred_config = dict(battle_pred_config or {})
        # The head predicts tanh(damage / damage_norm), not raw damage, and the
        # agent squashes the label the same way. Both defaults live here so the
        # agent (which reads battle_pred_config off the model) cannot drift out
        # of step with predict_battle.
        #
        # Why: with a raw-damage target the head has to grow its own weights to
        # reach outputs of +/-15, and the gradient it pushes into the shared
        # slot encoder grows with them. Measured on a trained head, the Huber
        # gradient on slot_proj was 25.9 against 0.0121 at initialisation -- a
        # factor of ~2100, which makes aux_coef impossible to calibrate up
        # front: a value chosen from an untrained net is wrong by three orders
        # of magnitude by the time the head has learned anything.
        #
        # tanh on both sides bounds the loss (and hence the gradient) whatever
        # the head has learned, so aux_coef means the same thing at step 0 and
        # at step 5M. tanh rather than clip(x/norm) keeps the target monotone in
        # damage: a 30-damage blowout still ranks above a 16-damage one.
        self.battle_pred_config.setdefault("damage_norm", DEFAULT_DAMAGE_NORM)
        self.battle_pred_config.setdefault("huber_delta", DEFAULT_HUBER_DELTA)
        self._battle_pred_enabled = bool(self.battle_pred_config.get("enabled", False))
        if self._battle_pred_enabled:
            self.battle_head = BattlePredictionHead(
                slot_hidden=self.slot_hidden,
                board_size=self.board_size,
                head_hidden=int(self.battle_pred_config.get("head_hidden", 128)),
                n_heads=int(
                    self.battle_pred_config.get(
                        "head_attn_heads", self.entity_attention_heads
                    )
                ),
                attn_init_scale=float(
                    self.battle_pred_config.get("attn_init_scale", 0.1)
                ),
            )
        else:
            self.battle_head = None

    # ------------------------------------------------------------------
    # Auxiliary battle-outcome head
    # ------------------------------------------------------------------
    def predict_battle(
        self,
        own_board_obs: torch.Tensor,
        opp_board_obs: torch.Tensor,
        attack_first: torch.Tensor,
        *,
        detach_features: bool = False,
    ) -> torch.Tensor:
        """Signed uncapped damage predicted from the two boards alone.

        ``own_board_obs`` / ``opp_board_obs``: ``(B, BOARD_SIZE, SLOT_DIM)``, as
        produced by :func:`src.envs.bglike.obs.encode_board_minions` from the
        combat snapshot. ``attack_first``: ``(B,)`` or ``(B, 1)`` float 0/1.

        Board-only by design: shop, hand and battle history must not leak in,
        because the label is a function of the two boards and who swings first.

        ``detach_features=True`` cuts the gradient to the slot encoder, leaving
        the head as a pure probe of the representation.
        """
        if not self._battle_pred_enabled or self.battle_head is None:
            raise RuntimeError(
                "predict_battle called but battle_head is disabled. Pass "
                "battle_pred_config={'enabled': True, ...} at construction."
            )
        # Own board keeps the shared own-position table; the enemy board gets
        # the head's own, because combat position means something different on
        # the other side of the board.
        e_own = self._encode_region(own_board_obs, None, self.own_pos_emb)
        e_enemy = self._encode_region(
            opp_board_obs, None, self.battle_head.enemy_pos_emb
        )
        if detach_features:
            e_own = e_own.detach()
            e_enemy = e_enemy.detach()
        # Squashed to (-1, 1). The label is squashed the same way (the agent
        # applies tanh(damage / damage_norm)), so both sides of the loss live
        # in a bounded space with a known scale -- see DEFAULT_DAMAGE_NORM.
        return torch.tanh(self.battle_head(e_own, e_enemy, attack_first))

    # ------------------------------------------------------------------
    # Static table
    # ------------------------------------------------------------------
    def _build_static_table(self) -> Tuple[torch.Tensor, Dict[str, Any], bool]:
        """Build the frozen table, or a zero placeholder when the patch is absent.

        A checkpoint reconstructed on a machine that lacks the patch package
        (paths differ between the training host and eval boxes) allocates zeros
        here; ``load_state_dict`` then restores the real table. Training from
        scratch with an unusable table is caught in :meth:`encode_state` rather
        than silently learning on zeros.
        """
        shape = (2 * self._static_rows, self.card_row_dim)
        unbuilt = (torch.zeros(shape, dtype=torch.float32), {"source": "unbuilt"}, False)
        if self.card_patch_dir is None:
            return unbuilt

        # A *missing* patch package is the normal restore case, not an error:
        # card_patch_dir is stored as the absolute path of the machine that
        # trained the run, and training happens on rented boxes while evaluation
        # happens elsewhere. The checkpoint carries the table in its state_dict,
        # so fall back to the placeholder and let load_state_dict fill it. A
        # fresh net that never loads one still fails loudly, in encode_state.
        #
        # A patch that *is* readable but disagrees (pool size, row shape) is a
        # real misconfiguration and still raises below.
        if not (Path(self.card_patch_dir) / "catalog.json").is_file():
            return unbuilt

        from src.bg_catalog.patch_context import PatchContext

        patch = PatchContext.load(Path(self.card_patch_dir))
        if int(patch.num_pool_indices) != int(self.num_pool_indices):
            raise ValueError(
                f"num_pool_indices={self.num_pool_indices} but patch "
                f"{self.card_patch_dir} has {patch.num_pool_indices}"
            )
        table, meta = build_card_static_table(
            patch,
            text_mode=self.card_text_mode,
            text_dim=self.card_text_dim,
            random_seed=self.card_static_seed,
        )
        if table.shape != shape:
            raise ValueError(f"static table {table.shape} != expected {shape}")
        return torch.from_numpy(np.ascontiguousarray(table)), dict(meta), True

    def load_state_dict(self, state_dict, strict: bool = True, **kw):  # type: ignore[override]
        out = super().load_state_dict(state_dict, strict=strict, **kw)
        if "card_static" in state_dict:
            self._static_table_ready = True
        return out

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        kw = super().get_constructor_kwargs()
        kw.update(
            {
                "card_text_mode": self.card_text_mode,
                "card_text_dim": self.card_text_dim,
                "card_static_seed": self.card_static_seed,
                "card_patch_dir": self.card_patch_dir,
                "battle_pred_config": dict(self.battle_pred_config),
            }
        )
        # Meaningless in v12 (no learned card embedding, no ability tokens);
        # dropped so a reconstructed net cannot look like it still has them.
        for dead in ("card_emb_dim", "use_card_emb", "ability_emb_dim"):
            kw.pop(dead, None)
        return kw

    # ------------------------------------------------------------------
    # Obs split: [ base obs | hero block | identity? ]
    # ------------------------------------------------------------------
    def _split_hero_identity(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        w = x.shape[1]
        base = OBS_DIM_V6_HEROES
        if w == base:
            rest, id_onehot = x, None
        elif w == base + self.num_identities:
            rest, id_onehot = x[:, :base], x[:, base:]
        else:
            raise ValueError(
                f"v12 expected obs dim {base} or {base + self.num_identities}, got {w}"
            )
        head = rest[:, :_OBS_DIM_HEAD]
        hero_block = rest[:, _OBS_DIM_HEAD:base]  # (B, HERO_BLOCK_DIM)
        return head, hero_block, id_onehot

    # ------------------------------------------------------------------
    # Card facts
    # ------------------------------------------------------------------
    def _card_features(self, card_idx: torch.Tensor, is_golden: torch.Tensor) -> torch.Tensor:
        """``(..., card_row_dim)`` — the frozen row for this (card, golden) pair.

        Empty slots carry ``card_idx == 0``, whose row (and its golden twin) is
        all-zero by construction, so absent minions contribute nothing.
        """
        return self.card_static[card_idx + self._static_rows * is_golden]

    def _encode_region(
        self, z_slots: torch.Tensor, _unused_ability_summary, pos_emb: nn.Embedding
    ) -> torch.Tensor:
        cid = z_slots[..., _CARD_IDX_OFFSET].long().clamp(min=0, max=self.num_pool_indices)
        golden = (z_slots[..., _GOLDEN_OFFSET] > 0.5).long()
        cont = torch.cat(
            [z_slots[..., :_CARD_IDX_OFFSET], z_slots[..., _CARD_IDX_OFFSET + 1 :]],
            dim=-1,
        )
        card = self._card_features(cid, golden)
        h = F.relu(self.slot_proj(torch.cat([cont, card], dim=-1)))
        return h + pos_emb.weight[: h.shape[1]].unsqueeze(0)

    # ------------------------------------------------------------------
    # State encoder (v11_heroes body, ability block removed)
    # ------------------------------------------------------------------
    def encode_state(self, x: torch.Tensor):
        if not self._static_table_ready:
            raise RuntimeError(
                "card_static table was never built: construct with "
                "card_patch_dir=<patch package> or load a checkpoint that "
                "carries the buffer."
            )

        head, hero_block, id_onehot = self._split_hero_identity(x)
        g, own, shop, hand, lb, phase, pending = self._unpack_head(head)
        B = head.size(0)

        E_own = self._encode_region(own, None, self.own_pos_emb)
        E_shop = self._encode_region(shop, None, self.shop_pos_emb)
        E_hand = self._encode_region(hand, None, self.hand_pos_emb)

        # Pending discover options: card facts only (they are offers, not
        # board minions — never golden, no runtime state of their own).
        disc_idx = (
            pending[
                ...,
                PENDING_DISCOVER_IDX_OFFSET : PENDING_DISCOVER_IDX_OFFSET
                + PENDING_DISCOVER_IDX_DIM,
            ]
            .long()
            .clamp(min=0, max=self.num_pool_indices)
        )
        opt = self._card_features(disc_idx, torch.zeros_like(disc_idx))
        is_apply = pending[..., PENDING_IS_APPLY_OFFSET : PENDING_IS_APPLY_OFFSET + 1] > 0.5
        opt = opt.masked_fill(is_apply.unsqueeze(-1), 0.0)
        E_pending = F.relu(self.pending_to_slot(opt))
        E_pending = E_pending + self.pending_pos_emb.weight[: self.pending_len].unsqueeze(0)

        # Opponents → set tokens, with each opponent's hero one-hot fused in.
        self_hero = hero_block[:, :HERO_SELF_DIM]
        opp_hero = hero_block[:, HERO_SELF_DIM:HERO_BLOCK_DIM].view(
            B, self._max_opps, self.num_hero_obs
        )
        E_opp = self._encode_opponents_hero(g, opp_hero)

        # Scalar modalities.
        econ = g[:, : self._econ_dim]
        battle_hist = g[:, self._panel_off + self._panel_dim :]
        combat = torch.cat([battle_hist, lb, phase], dim=-1)
        econ_emb = self.economy_encoder(econ)
        combat_emb = self.combat_proj(combat)
        pctx_emb = self.pending_ctx_proj(pending)
        hero_emb = self.hero_encoder(self_hero)

        k = self.summary_queries
        query_tok = self.summary_query_emb.weight.unsqueeze(0).expand(B, -1, -1)

        if id_onehot is not None:
            id_e = self.identity_emb_proj(id_onehot)

            def _gate(E):
                Bn, Ln, Hn = E.shape
                cat = torch.cat([E, id_e.unsqueeze(1).expand(Bn, Ln, Hn)], dim=-1)
                return E * (1.0 + self.identity_slot_gate(cat))

            E_own, E_shop, E_hand, E_pending = (
                _gate(E_own),
                _gate(E_shop),
                _gate(E_hand),
                _gate(E_pending),
            )
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


__all__ = [
    "BGLikeStructuredV12",
    "DEFAULT_CARD_TEXT_DIM",
    "DEFAULT_DAMAGE_NORM",
    "DEFAULT_HUBER_DELTA",
]

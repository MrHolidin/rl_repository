"""Building blocks shared by the structured actor-critics (v1 / v2 / v3).

Source of truth for pure helpers and constants that the per-architecture modules
need to agree on (region/role ids, action-token packing, attention blocks).

Why this module exists:
  - ``minibg_structured_ac.py`` (v1) is FROZEN: its state_dict keys cannot move.
    But the *helpers* it defines (``EntityAttentionBlock``, ``_build_action_tokens``,
    constants) aren't part of the state_dict — they're just module-level
    functions/classes/ints. We host them here, and v1 re-exports them for
    backwards compatibility so existing import sites keep working.
  - v2 and v3 import directly from here.

Layout:
  * Role / region constants used by the legal-action gather and the post-conv
    additive region embeddings. Two different namespaces — see comments below.
  * ``EntityAttentionBlock`` — self-attention block used in entity-token stacks.
  * ``CrossAttentionBlock`` — Q attends to a separate K/V stream. Added for v3
    so action tokens can attend to the entity sequence at score time.
  * Pure functions: ``role_for_struct``, ``_struct_action_codes``,
    ``_build_action_tokens``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.envs.minibg.obs import (
    EFFECT_OFFSET as _EFFECT_OFFSET,
    NUM_EFFECT_CHANNELS as _NUM_EFFECT_CHANNELS,
    NUM_TRIGGER_CHANNELS as _NUM_TRIGGER_CHANNELS,
)
from src.envs.minibg.structured_actions import StructAction, StructActionType


# ---------------------------------------------------------------------------
# Shape / vocabulary constants
# ---------------------------------------------------------------------------

# Raw trigger + effect-class bit window inside a slot vector — used by the
# ent_extras bypass so the action head sees these bits directly.
_EXT_DIM = _NUM_TRIGGER_CHANNELS + _NUM_EFFECT_CHANNELS
_EXT_END = _EFFECT_OFFSET + _NUM_EFFECT_CHANNELS

_NUM_STRUCT_TYPES = len(StructActionType)

# Role ids: used by the action token's role embedding.
_ROLE_NONE = 0
_ROLE_SHOP = 1
_ROLE_BOARD = 2
_ROLE_HAND = 3
_ROLE_PENDING = 4
_NUM_ROLES = 5

# Region "kinds" used by the batched action-token gather. Index 0 is reserved
# for null-entity actions (ROLL / LEVEL_UP / COMPLETE_TURN) so padded slots map
# there as well.
_REGION_NULL = 0
_REGION_SHOP = 1
_REGION_OWN = 2
_REGION_HAND = 3
_REGION_PENDING = 4
_NUM_REGIONS = 5

# How many pending option slots we expose to the model (discover / adapt
# offer at most 3 options).
_PENDING_LEN = 3

# Region ids used by ``slot_region_emb`` — additive post-conv geometry tag.
# Distinct from ``_REGION_*`` above: those are gather kinds for the legal-action
# token path, these are positional tags inside the entity sequence.
REG_OWN = 0
REG_SHOP = 1
REG_HAND = 2
REG_ENEMY = 3
REG_PENDING = 4


# ---------------------------------------------------------------------------
# Attention building blocks
# ---------------------------------------------------------------------------


class EntityAttentionBlock(nn.Module):
    """Self-attention block over an entity sequence.

    Pre-LN MHA + pre-LN FFN, both with per-feature learned residual gates
    (``attn_scale`` / ``ff_scale``) initialized to a small constant so the
    block starts close to identity.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 4,
        ff_mult: int = 2,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.ln_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.ln_ff = nn.LayerNorm(dim)
        ff_dim = ff_mult * dim
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

        self.attn_scale = nn.Parameter(torch.full((dim,), float(init_scale)))
        self.ff_scale = nn.Parameter(torch.full((dim,), float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_attn(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a * self.attn_scale.view(1, 1, -1)

        h = self.ln_ff(x)
        x = x + self.ff(h) * self.ff_scale.view(1, 1, -1)

        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block: a query stream attends to a separate key/value stream.

    Same pre-LN + scaled-residual scheme as ``EntityAttentionBlock``, but the K/V
    side is read-only (no FFN on K/V, no residual on K/V). Used by v3's action
    head: each action token attends to the post-self-attention entity sequence
    to pick up board-wide context before scoring.

    ``embed_dim`` must match both Q and K/V — we don't add a projection here on
    purpose, leaving the caller to bridge dims explicitly.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int = 4,
        ff_mult: int = 2,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.ln_ff = nn.LayerNorm(embed_dim)
        ff_dim = ff_mult * embed_dim
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

        self.attn_scale = nn.Parameter(torch.full((embed_dim,), float(init_scale)))
        self.ff_scale = nn.Parameter(torch.full((embed_dim,), float(init_scale)))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_ln = self.ln_q(q)
        kv_ln = self.ln_kv(kv)
        a, _ = self.attn(q_ln, kv_ln, kv_ln, need_weights=False)
        q = q + a * self.attn_scale.view(1, 1, -1)

        h = self.ln_ff(q)
        q = q + self.ff(h) * self.ff_scale.view(1, 1, -1)

        return q


class BattlePredictionHead(nn.Module):
    """Predicts signed uncapped board-damage from (own_board, opp_board, attack_first).

    Input policy: **board-only**. The head deliberately does NOT consume
    ``state_emb`` (shop / hand / past battle history / opponent panel etc.)
    so the prediction target is purely a function of the two boards + who
    attacks first. This keeps the head's training signal aligned with the
    label (which is also board-only, see :func:`simulate_battle`).

    Inputs (pre-encoded by the parent model via shared conv weights):
    - ``e_own``:   (B, board_size, slot_hidden) — own board in post-reorder
      combat order, uses parent's ``own_pos_emb``.
    - ``e_enemy``: (B, board_size, slot_hidden) — opp board in post-reorder
      combat order, uses the head's own ``enemy_pos_emb`` (combat-position
      semantics differ between sides).
    - ``attack_first``: (B,) or (B, 1) float 0/1 — did this seat strike first.

    Aggregation: a learnable CLS token is prepended to ``[e_own, e_enemy]``,
    runs through :class:`EntityAttentionBlock`, and its post-attention vector
    is concatenated with ``attack_first`` and projected to a scalar.

    Reuse from the parent model:
    - Conv-encoder weights (``region_conv1``/``region_conv2``) produce e_own
      and e_enemy. When ``aux_coef > 0`` and ``detach_features=False``,
      gradient from the head's Huber loss flows back into them, acting as a
      board-understanding regularizer for the actor encoder.
    - Head-local: ``enemy_pos_emb``, learnable CLS, attention block, MLP.
    """

    def __init__(
        self,
        *,
        slot_hidden: int,
        board_size: int,
        head_hidden: int = 128,
        n_heads: int = 4,
        attn_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.slot_hidden = int(slot_hidden)
        self.board_size = int(board_size)
        self.enemy_pos_emb = nn.Embedding(self.board_size, self.slot_hidden)
        nn.init.normal_(self.enemy_pos_emb.weight, mean=0.0, std=0.02)
        # Learnable CLS token. Initialized small (std=0.02) so attention starts
        # close to "CLS is just a fresh aggregator", not biased toward any
        # specific board configuration.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.slot_hidden))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.attn = EntityAttentionBlock(
            self.slot_hidden,
            num_heads=int(n_heads),
            ff_mult=2,
            init_scale=float(attn_init_scale),
        )
        self.proj = nn.Sequential(
            nn.Linear(self.slot_hidden + 1, int(head_hidden)),
            nn.GELU(),
            nn.Linear(int(head_hidden), 1),
        )

    def forward(
        self,
        e_own: torch.Tensor,
        e_enemy: torch.Tensor,
        attack_first: torch.Tensor,
    ) -> torch.Tensor:
        B = e_own.size(0)
        cls = self.cls_token.expand(B, 1, -1)                              # (B, 1, slot_hidden)
        seq = torch.cat([cls, e_own, e_enemy], dim=1)                      # (B, 1+2*board_size, slot_hidden)
        seq = self.attn(seq)
        cls_out = seq[:, 0]                                                # (B, slot_hidden)
        if attack_first.dim() == 1:
            attack_first = attack_first.unsqueeze(-1)
        feat = torch.cat([cls_out, attack_first], dim=-1)
        return self.proj(feat).squeeze(-1)                                 # (B,)


# ---------------------------------------------------------------------------
# Action-token packing
# ---------------------------------------------------------------------------


def role_for_struct(a: StructAction) -> int:
    t = a.type
    if t == StructActionType.BUY:
        return _ROLE_SHOP
    if t == StructActionType.SELL:
        return _ROLE_BOARD
    if t == StructActionType.PLACE or t == StructActionType.MAGNET:
        return _ROLE_HAND
    if t == StructActionType.DISCOVER_PICK:
        return _ROLE_PENDING
    if t == StructActionType.APPLY_EFFECT:
        return _ROLE_BOARD
    if t == StructActionType.APPLY_EFFECT_SKIP:
        return _ROLE_BOARD
    return _ROLE_NONE


def _struct_action_codes(a: StructAction) -> Tuple[int, int, int, int, int, int]:
    """Return ``(type_id, role_id, src_region, src_slot, tgt_region, tgt_slot)``.

    Two-entity tokens so MAGNET carries both source (hand) and target (board), and
    DISCOVER_PICK carries the picked pending option as its source entity.
    """
    t = a.type
    if t == StructActionType.BUY:
        return (int(t), _ROLE_SHOP, _REGION_SHOP, int(a.args[0]), _REGION_NULL, 0)
    if t == StructActionType.SELL:
        return (int(t), _ROLE_BOARD, _REGION_OWN, int(a.args[0]), _REGION_NULL, 0)
    if t == StructActionType.PLACE:
        return (int(t), _ROLE_HAND, _REGION_HAND, int(a.args[0]), _REGION_NULL, 0)
    if t == StructActionType.MAGNET:
        return (
            int(t),
            _ROLE_HAND,
            _REGION_HAND,
            int(a.args[0]),
            _REGION_OWN,
            int(a.args[1]),
        )
    if t == StructActionType.DISCOVER_PICK:
        return (int(t), _ROLE_PENDING, _REGION_PENDING, int(a.args[0]), _REGION_NULL, 0)
    if t == StructActionType.APPLY_EFFECT:
        return (
            int(t),
            _ROLE_BOARD,
            _REGION_NULL,
            0,
            _REGION_OWN,
            int(a.args[0]),
        )
    if t == StructActionType.APPLY_EFFECT_SKIP:
        return (int(t), _ROLE_BOARD, _REGION_NULL, 0, _REGION_NULL, 0)
    # ROLL / LEVEL_UP / COMPLETE_TURN: null entity on both sides.
    return (int(t), _ROLE_NONE, _REGION_NULL, 0, _REGION_NULL, 0)


def _build_action_tokens(
    legal_actions: Sequence[Sequence[StructAction]],
    Lmax: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Pack ``(B, Lmax)`` int64 token arrays + bool mask. Padded slots stay at NULL/0.

    Returns ``(type_ids, role_ids, src_region_kinds, src_region_slots,
    tgt_region_kinds, tgt_region_slots, mask)``.
    """
    B = len(legal_actions)
    type_ids = np.zeros((B, Lmax), dtype=np.int64)
    role_ids = np.zeros((B, Lmax), dtype=np.int64)
    src_region_kinds = np.zeros((B, Lmax), dtype=np.int64)
    src_region_slots = np.zeros((B, Lmax), dtype=np.int64)
    tgt_region_kinds = np.zeros((B, Lmax), dtype=np.int64)
    tgt_region_slots = np.zeros((B, Lmax), dtype=np.int64)
    mask = np.zeros((B, Lmax), dtype=bool)
    for b in range(B):
        row = legal_actions[b]
        L = len(row)
        if L == 0:
            continue
        for l in range(L):
            t, r, sk, ss, tk, ts = _struct_action_codes(row[l])
            type_ids[b, l] = t
            role_ids[b, l] = r
            src_region_kinds[b, l] = sk
            src_region_slots[b, l] = ss
            tgt_region_kinds[b, l] = tk
            tgt_region_slots[b, l] = ts
        mask[b, :L] = True
    return (
        type_ids,
        role_ids,
        src_region_kinds,
        src_region_slots,
        tgt_region_kinds,
        tgt_region_slots,
        mask,
    )


__all__ = [
    "EntityAttentionBlock",
    "CrossAttentionBlock",
    "role_for_struct",
    "_struct_action_codes",
    "_build_action_tokens",
    "_EXT_DIM",
    "_EXT_END",
    "_NUM_STRUCT_TYPES",
    "_ROLE_NONE",
    "_ROLE_SHOP",
    "_ROLE_BOARD",
    "_ROLE_HAND",
    "_ROLE_PENDING",
    "_NUM_ROLES",
    "_REGION_NULL",
    "_REGION_SHOP",
    "_REGION_OWN",
    "_REGION_HAND",
    "_REGION_PENDING",
    "_NUM_REGIONS",
    "_PENDING_LEN",
    "REG_OWN",
    "REG_SHOP",
    "REG_HAND",
    "REG_ENEMY",
    "REG_PENDING",
]

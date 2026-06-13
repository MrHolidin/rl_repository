"""Structured actor-critic, v9 = v8 + economy-conditioned action queries.

Difference from v8: each action token's embedding gains an additive economy
term (core globals + tribe rotation) **before** it cross-attends to the entity
sequence. In v3..v8 the per-action query into the board/shop tokens is built
only from the action's type/role/src/tgt-entity; the economy (gold, income,
tier-up cost, roll cost, action budget, HP, which tribes are in rotation)
reaches the action only at the final ``score_fc`` MLP, *after* attention. So a
``LEVEL_UP`` token — which carries no entity at all — queries the board the
same way whether tier-up costs 4 or 11 gold.

v9 projects the economy slice of ``g_full`` (the first
``BGLIKE_GLOBAL_CORE_DIM + SHOP_ROTATION_OBS_DIM`` floats — core economy +
rotation, NOT the 77-dim opponent panel / battle history) into ``action_dim``
and adds it to every action token's base embedding. The cross-attention query
is then economically coloured: the action can decide *what on the board to look
at* as a function of the economy, and ``LEVEL_UP`` finally has a non-trivial
representation that conditions on its own cost and the gold available.

Mechanically:
  * ``action_econ_proj`` : ``Linear(econ_dim -> action_dim)``, **zero-init**
    (weight and bias). Added to the v2 base action embedding before v3's
    cross-attention. Zero-init ⇒ v9 reproduces v8 byte-for-byte at step 0;
    gradient flows on the first update.

v9 keeps the v8 distributional placement critic and the whole obs_v5 + identity
surface unchanged, so a v8 checkpoint warm-starts v9 with ``strict=False`` (only
``action_econ_proj.*`` is new).

Checkpoint contract: own ``ppo_network_type`` (``bglike_structured_v9``); not
interchangeable with v8 (extra ``action_econ_proj`` params).
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .bglike_structured_v8 import BGLikeStructuredV8


class BGLikeStructuredV9(BGLikeStructuredV8):
    """v8 + economy (core globals + tribe rotation) injected into action queries."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        from src.envs.bglike.obs import BGLIKE_GLOBAL_CORE_DIM
        from src.envs.minibg.obs import SHOP_ROTATION_OBS_DIM

        # Economy slice = leading [core | rotation] of g_full. The remaining
        # g_full dims (opponent panel, battle history, last_battle, phase) are
        # deliberately excluded — they already saturate score_fc, and the
        # diagnosed gap is economy/availability in the action *query*.
        self._econ_dim = int(BGLIKE_GLOBAL_CORE_DIM) + int(SHOP_ROTATION_OBS_DIM)

        self.action_econ_proj = nn.Linear(self._econ_dim, self.action_dim)
        nn.init.zeros_(self.action_econ_proj.weight)
        nn.init.zeros_(self.action_econ_proj.bias)

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
        # v2 base action embedding (type + role + gathered entity), BEFORE v3's
        # cross-attention. ``super(BGLikeStructuredV8, ...)`` is v7→v6→v3; none
        # of v6/v7/v8 override ``_encode_actions``, so this resolves to v3's,
        # which itself first builds the v2 base and then applies cross-attn — we
        # must NOT call that here (it would attend with an un-coloured query).
        # Build the v2 base directly via the v2 implementation.
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

        # Economy-colour the base action embedding (zero at init).
        econ = cache["g_full"][:, : self._econ_dim]  # (B, econ_dim)
        ae = ae + self.action_econ_proj(econ).unsqueeze(1)  # broadcast over Lmax

        # v3's cross-attention, now querying with the economy-coloured tokens.
        # (Mirrors BGLikeStructuredV3._encode_actions; v3 is left untouched.)
        entity_seq = torch.cat(
            [cache["E_own"], cache["E_shop"], cache["E_hand"], cache["E_pending"]],
            dim=1,
        )
        ae_q = self.action_to_slot(ae)
        ae_attended = self.action_cross_attn(ae_q, entity_seq)
        ae_extra = self.slot_to_action(ae_attended)  # zero at init
        return ae + ae_extra


__all__ = ["BGLikeStructuredV9"]

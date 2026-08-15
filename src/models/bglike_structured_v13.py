"""Structured actor-critic, v13 = v12 + the seat's tribe-preference vector.

    obs = [ base(976) | hero block(147) | tribe pref(7) ]   (OBS_DIM_V7_PREF)

The vector is a hand-written stand-in for a DvD identity: drawn per seat at
game start, uniform in [-1, 1] per tribe, and the trainer pays
``coef * v[tribe]`` for every minion of that tribe the seat buys. To act on it
the net has to read it, so it enters on two paths:

* the per-slot gate inherited from v7 — ``num_identities`` is pinned to the
  tribe count so ``identity_emb_proj`` is exactly the projection this vector
  needs, and every slot embedding is modulated by it;
* concatenated (through a small encoder) into the state summary, so the critic
  and the action head see it directly rather than only through gated slots.

The second path is the point of the version. The gate alone is the pathway the
v11 notes call near-no-op (TV≈0.03) and the one the DvD experiment watched
atrophy under a dominant meta; a vector that only reaches the policy through a
multiplicative slot gate can be ignored, and an ignored vector makes the whole
arm inert while still looking like it ran.

Trunk-sized layers are rebuilt for the widened summary, exactly as v11_heroes
does for its hero block.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.envs.bglike.obs import OBS_DIM as _OBS_DIM_HEAD
from src.envs.bglike.obs_v6_heroes import OBS_DIM_V6_HEROES
from src.envs.bglike.obs_v7_pref import OBS_DIM_V7_PREF, TRIBE_PREF_DIM

from .bglike_structured_v11 import NUM_PLACEMENTS, _ReZeroMLPBlock
from .bglike_structured_v12 import BGLikeStructuredV12

DEFAULT_PREF_HIDDEN = 32
DEFAULT_PREF_OUT = 16


class BGLikeStructuredV13(BGLikeStructuredV12):
    """v12 conditioned on a per-seat tribe-preference vector."""

    def __init__(
        self,
        *,
        pref_hidden: int = DEFAULT_PREF_HIDDEN,
        pref_out: int = DEFAULT_PREF_OUT,
        **v12_kwargs: Any,
    ) -> None:
        # The gate projection is sized by num_identities; pinning it to the
        # tribe count turns the inherited identity path into the preference
        # path with no extra parameters. Reject a config that sets it to
        # something else rather than silently ignoring the value.
        requested = v12_kwargs.pop("num_identities", TRIBE_PREF_DIM)
        if int(requested) != TRIBE_PREF_DIM:
            raise ValueError(
                f"v13 pins num_identities to the tribe count {TRIBE_PREF_DIM}; "
                f"got {requested}"
            )
        super().__init__(num_identities=TRIBE_PREF_DIM, **v12_kwargs)

        self.pref_hidden = int(pref_hidden)
        self.pref_out = int(pref_out)
        self.obs_dim = int(OBS_DIM_V7_PREF)

        self.pref_encoder = nn.Sequential(
            nn.Linear(TRIBE_PREF_DIM, self.pref_hidden),
            nn.ReLU(),
            nn.Linear(self.pref_hidden, self.pref_out),
        )

        # Trunk grows by pref_out → rebuild the trunk-sized layers (same
        # treatment v11_heroes gives the hero block).
        self._state_summary_dim = self._state_summary_dim + self.pref_out
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
        self.critic_shape = nn.Linear(self._state_summary_dim, 1)
        nn.init.zeros_(self.critic_shape.weight)
        nn.init.zeros_(self.critic_shape.bias)

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        kw = super().get_constructor_kwargs()
        kw["pref_hidden"] = self.pref_hidden
        kw["pref_out"] = self.pref_out
        return kw

    # ------------------------------------------------------------------
    # Obs split: [ base obs | hero block | tribe pref ]
    # ------------------------------------------------------------------
    def _split_hero_identity(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        w = x.shape[1]
        base = OBS_DIM_V6_HEROES
        if w == base + TRIBE_PREF_DIM:
            rest, pref = x[:, :base], x[:, base:]
        elif w == base:
            # A preference-free observation is legal and means "no preference":
            # zeros gate to a no-op and encode to a constant.
            rest, pref = x, x.new_zeros(x.shape[0], TRIBE_PREF_DIM)
        else:
            raise ValueError(
                f"v13 expected obs dim {base + TRIBE_PREF_DIM} or {base}, got {w}"
            )
        head = rest[:, :_OBS_DIM_HEAD]
        hero_block = rest[:, _OBS_DIM_HEAD:base]
        return head, hero_block, pref

    def _trunk_extra(self, id_tail: Optional[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        if id_tail is None:
            raise RuntimeError("v13 always carries a preference tail")
        return (self.pref_encoder(id_tail),)


__all__ = ["BGLikeStructuredV13", "DEFAULT_PREF_HIDDEN", "DEFAULT_PREF_OUT"]

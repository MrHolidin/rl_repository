"""Structured actor-critic, v8 = v7 + distributional placement critic.

Difference from v7: the critic predicts a **distribution over final placement**
(8-way softmax) instead of a scalar value. The scalar value consumed by
PPO/GAE is its expectation under the fixed placement-reward vector:

    V(s) = sum_k softmax(critic_dist(trunk))_k * placement_reward(k+1)

Why: with terminal-only placement reward and gamma=1, the return from ANY step
of a seat segment is exactly ``placement_reward(final_place)`` — a
deterministic function of an 8-way categorical outcome. Scalar MSE stops
producing gradient once the conditional mean is matched ("either top-2 or
bottom-2" and "always 4th-5th" have the same mean), while cross-entropy on the
final placement keeps shaping the full conditional distribution. The agent
(``MiniBGPPOStructuredAgent``) detects this head via ``placement_logits`` and
replaces the value-MSE with CE on backfilled per-segment placement labels;
GAE/advantages consume the expectation unchanged.

Init contract: the dist head's final layer is **zero-initialised** → uniform
placement distribution → V(s) = mean(placement_reward) = exactly 0, matching a
fresh scalar critic. A v7 checkpoint warm-starts with ``strict=False`` (only
``critic_dist.*`` keys are new; the inherited scalar ``critic`` stays in the
state_dict but is no longer used for values).

Consistency caveat: if the terminal reward carries non-placement terms (board
shaping, DvD bonus, battle shaping), V misses them by construction — run v8
with those at 0, or add a residual scalar head before turning them on.

Checkpoint contract: own ``ppo_network_type`` (``bglike_structured_v8``).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from .bglike_structured_v7 import BGLikeStructuredV7

NUM_PLACEMENTS = 8


def _placement_reward_vec() -> torch.Tensor:
    from src.envs.bglike.placement import placement_reward

    return torch.tensor(
        [placement_reward(p) for p in range(1, NUM_PLACEMENTS + 1)],
        dtype=torch.float32,
    )


class BGLikeStructuredV8(BGLikeStructuredV7):
    """v7 + 8-way placement-distribution critic (value = expectation)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Inherited scalar ``self.critic`` is kept (unused) so v7 state_dicts
        # load without unexpected keys; values come from ``critic_dist``.
        self.critic_dist = nn.Sequential(
            nn.Linear(self._state_summary_dim, self.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.critic_hidden, NUM_PLACEMENTS),
        )
        # Zero-init the final layer: uniform over placements → V = 0 exactly
        # (the placement-reward vector is symmetric), matching a fresh scalar
        # critic at step 0. The first hidden layer keeps standard init so
        # gradient flows immediately.
        nn.init.zeros_(self.critic_dist[-1].weight)
        nn.init.zeros_(self.critic_dist[-1].bias)
        self.register_buffer("placement_reward_vec", _placement_reward_vec())

    # ------------------------------------------------------------------
    # Distributional critic surface (consumed by the PPO agent)
    # ------------------------------------------------------------------

    def placement_logits(self, trunk: torch.Tensor) -> torch.Tensor:
        """(B, 8) logits over final placement, from the post-LN state summary."""
        return self.critic_dist(trunk)

    def value_from_trunk(self, trunk: torch.Tensor) -> torch.Tensor:
        """Scalar value = expected placement reward under the predicted distribution."""
        probs = torch.softmax(self.placement_logits(trunk), dim=-1)
        return (probs * self.placement_reward_vec).sum(dim=-1)

    # ------------------------------------------------------------------
    # Values now come from the dist head; everything else is v7 verbatim.
    # (``policy_logits_and_value`` delegates here, so one override covers both.)
    # ------------------------------------------------------------------

    def policy_logits_value_from_tokens(
        self,
        obs: torch.Tensor,
        type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        src_region_kinds: torch.Tensor,
        src_region_slots: torch.Tensor,
        tgt_region_kinds: torch.Tensor,
        tgt_region_slots: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        state_emb, cache = self.encode_state(obs)
        logits = self._logits_from_state_and_tokens(
            state_emb,
            cache,
            type_ids,
            role_ids,
            src_region_kinds,
            src_region_slots,
            tgt_region_kinds,
            tgt_region_slots,
            mask,
        )
        values = self.value_from_trunk(cache["trunk"])
        if return_cache:
            cache_out = dict(cache)
            cache_out["state_emb"] = state_emb
            return logits, mask, values, cache_out
        return logits, mask, values


__all__ = ["BGLikeStructuredV8", "NUM_PLACEMENTS"]

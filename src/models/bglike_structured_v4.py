"""Structured actor-critic, v4 = v3 + round-level recurrent state (GRU).

Difference from v3: a per-seat hidden state ``h`` is carried **across rounds**
within an episode. ``h`` is updated once per round via ``GRUCell(state_emb, h)``
at the agent's ``COMPLETE_TURN`` action, and concatenated into the state
summary that feeds both actor and critic. Within a round all shop steps see
the same ``h``.

Motivation: v1/v2/v3 are fully Markovian — every forward pass is computed
from the current observation alone. Multi-round planning (commit to a tribe
on R3, payoff on R8) has nowhere to "live" in the model. v4 adds the simplest
form of episodic memory: 1 hidden vector updated once per round.

Shape contract is the same as v3 plus an extra optional ``h_prev`` argument on
``encode_state``, ``policy_logits_and_value``, and ``policy_logits_value_from_tokens``.
When ``h_prev`` is ``None``, zeros are substituted. State summary, projection
and critic input widths are wider than v3 by ``recurrent_hidden_dim``, so v4
checkpoints are not interchangeable with v3.

Trainer responsibilities (out of scope for this module):
  * Maintain a per-seat ``h`` during rollout collection (zeros at episode reset,
    updated by ``step_round_hidden`` on the seat's COMPLETE_TURN step).
  * Store ``h_prev`` at every step in the rollout buffer.
  * During PPO update: group buffer steps by ``(episode_id, seat_id)`` into
    contiguous sequences and replay the GRU forward chain per sequence for
    full BPTT.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.envs.minibg.obs import (
    PENDING_IS_APPLY_OFFSET as _PENDING_IS_APPLY_OFFSET,
    TRIGGER_OFFSET as _TRIGGER_OFFSET,
)
from .bglike_structured_v3 import BGLikeStructuredV3
from .minibg_slot_ac import (
    _LAST_BATTLE_DIM,
    _PHASE_DIM,
    _pending_three_option_emb,
)
from .structured_common import (
    REG_HAND,
    REG_OWN,
    REG_PENDING,
    REG_SHOP,
    _EXT_DIM,
    _EXT_END,
    _build_action_tokens,
)


class BGLikeStructuredV4(BGLikeStructuredV3):
    """v3 + round-level recurrence.

    Args (in addition to v3):
        recurrent_hidden_dim: width of the GRU hidden state. 128 is a sane default.
        round_gru_init_scale: gain for xavier init of GRU weights. 0.1 keeps
            early-training updates small so the recurrent head doesn't
            destabilise the policy before it can learn what to remember.
    """

    def __init__(
        self,
        *,
        recurrent_hidden_dim: int = 128,
        round_gru_init_scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.round_gru_init_scale = float(round_gru_init_scale)

        self.round_gru = nn.GRUCell(self.state_dim, self.recurrent_hidden_dim)
        for name, param in self.round_gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param, gain=self.round_gru_init_scale)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Widen state_summary by recurrent_hidden_dim. Replace LN / state_proj /
        # critic to match — v4 state dicts are intentionally incompatible with v3.
        base_dim = (
            self.slot_hidden
            + self.global_dim
            + _LAST_BATTLE_DIM
            + _PHASE_DIM
            + self._pending_header_dim
        )
        new_state_summary_dim = base_dim + self.recurrent_hidden_dim
        self._state_summary_dim = int(new_state_summary_dim)
        self._state_summary_base_dim = int(base_dim)

        self.state_summary_ln = nn.LayerNorm(new_state_summary_dim)
        self.state_proj = nn.Linear(new_state_summary_dim, self.state_dim)
        self.critic = nn.Sequential(
            nn.Linear(new_state_summary_dim, self.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.critic_hidden, 1),
        )

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        kw = super().get_constructor_kwargs()
        kw["recurrent_hidden_dim"] = self.recurrent_hidden_dim
        kw["round_gru_init_scale"] = self.round_gru_init_scale
        return kw

    # ------------------------------------------------------------------
    # Hidden-state helpers
    # ------------------------------------------------------------------

    def zero_hidden(
        self,
        batch_size: int,
        *,
        device=None,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """Allocate ``(batch_size, recurrent_hidden_dim)`` zeros on the model's device."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.recurrent_hidden_dim, device=device, dtype=dtype)

    def step_round_hidden(
        self, state_emb: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        """One GRU step: ``h_t = GRU(state_emb_t, h_{t-1})``.

        Called by the trainer on the agent's COMPLETE_TURN steps. Within a round
        all shop steps reuse the same ``h_{t-1}`` as input (no in-round update).
        """
        return self.round_gru(state_emb, h_prev)

    # ------------------------------------------------------------------
    # Encoding (overridden to consume h_prev)
    # ------------------------------------------------------------------

    def encode_entities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Heavy, h-independent part of ``encode_state``.

        Runs per-slot conv encoders, builds the [CLS]+entity sequence and the
        entity self-attention stack. Returns a dict with all tensors needed to
        (a) finalize state_summary with an h_prev (via :meth:`state_summary_and_emb`),
        (b) score actions (E_* and EXT_*, g_full).

        Splitting this out lets the trainer batch (B*T) across BPTT timesteps
        through the expensive attention work in one forward, leaving only cheap
        per-step ops (concat + LN + state_proj + GRU + critic linear) in the
        Python sequential loop.
        """
        g, own, shop, hand, _enemy, lb, phase, pending = self._unpack(x)

        E_own = self._add_pos_region(
            self._encode_region_slots(own), self.own_pos_emb, REG_OWN
        )
        E_shop = self._add_pos_region(
            self._encode_region_slots(shop), self.shop_pos_emb, REG_SHOP
        )
        E_hand = self._add_pos_region(
            self._encode_region_slots(hand), self.hand_pos_emb, REG_HAND
        )

        EXT_own = own[..., _TRIGGER_OFFSET:_EXT_END]
        EXT_shop = shop[..., _TRIGGER_OFFSET:_EXT_END]
        EXT_hand = hand[..., _TRIGGER_OFFSET:_EXT_END]

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
        EXT_pending = torch.zeros(B, self.pending_len, _EXT_DIM, device=device, dtype=dtype)

        pending_header = pending[..., :self._pending_header_dim]
        g_full = torch.cat([g, lb, phase], dim=-1)

        cls_tok = self.cls_from_globals(g_full).unsqueeze(1)
        E_all = torch.cat([cls_tok, E_own, E_shop, E_hand, E_pending], dim=1)
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

        return {
            "cls_out": cls_out,
            "E_own": E_own,
            "E_shop": E_shop,
            "E_hand": E_hand,
            "E_enemy": E_enemy,
            "E_pending": E_pending,
            "EXT_own": EXT_own,
            "EXT_shop": EXT_shop,
            "EXT_hand": EXT_hand,
            "EXT_pending": EXT_pending,
            "g_full": g_full,
            "g": g,
            "lb": lb,
            "phase": phase,
            "pending_header": pending_header,
        }

    def state_summary_and_emb(
        self,
        cls_out: torch.Tensor,
        g: torch.Tensor,
        lb: torch.Tensor,
        phase: torch.Tensor,
        pending_header: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Light, h-dependent part: concat + LN + state_proj.

        Returns ``(state_emb, state_summary_n)`` — ``state_summary_n`` is the
        critic's input (the post-LN trunk).
        """
        state_summary = torch.cat(
            [cls_out, g, lb, phase, pending_header, h_prev], dim=-1
        )
        state_summary_n = self.state_summary_ln(state_summary)
        state_emb = self.state_proj(state_summary_n)
        return state_emb, state_summary_n

    def encode_state(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        parts = self.encode_entities(x)

        B = x.size(0)
        device = x.device
        dtype = parts["cls_out"].dtype
        if h_prev is None:
            h_prev = self.zero_hidden(B, device=device, dtype=dtype)
        else:
            if h_prev.dim() != 2 or h_prev.shape[0] != B or h_prev.shape[1] != self.recurrent_hidden_dim:
                raise ValueError(
                    f"h_prev shape {tuple(h_prev.shape)} != expected "
                    f"({B}, {self.recurrent_hidden_dim})"
                )
            if h_prev.device != device:
                h_prev = h_prev.to(device)
            if h_prev.dtype != dtype:
                h_prev = h_prev.to(dtype)

        state_emb, state_summary_n = self.state_summary_and_emb(
            parts["cls_out"],
            parts["g"],
            parts["lb"],
            parts["phase"],
            parts["pending_header"],
            h_prev,
        )

        cache: Dict[str, torch.Tensor] = {
            "E_own": parts["E_own"],
            "E_shop": parts["E_shop"],
            "E_hand": parts["E_hand"],
            "E_enemy": parts["E_enemy"],
            "E_pending": parts["E_pending"],
            "EXT_own": parts["EXT_own"],
            "EXT_shop": parts["EXT_shop"],
            "EXT_hand": parts["EXT_hand"],
            "EXT_pending": parts["EXT_pending"],
            "trunk": state_summary_n,
            "g_full": parts["g_full"],
            "h_prev": h_prev,
        }
        return state_emb, cache

    # ------------------------------------------------------------------
    # Forward APIs (overridden to thread h_prev through)
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
        h_prev: Optional[torch.Tensor] = None,
        return_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        state_emb, cache = self.encode_state(obs, h_prev=h_prev)
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
        values = self.critic(cache["trunk"]).squeeze(-1)
        if return_cache:
            cache_out = dict(cache)
            cache_out["state_emb"] = state_emb
            return logits, mask, values, cache_out
        return logits, mask, values

    def policy_logits_and_value(
        self,
        obs: torch.Tensor,
        legal_actions,
        *,
        h_prev: Optional[torch.Tensor] = None,
        return_cache: bool = False,
    ):
        B = obs.size(0)
        if len(legal_actions) != B:
            raise ValueError(
                f"legal_actions has {len(legal_actions)} rows but obs batch size is {B}"
            )
        Lmax = max((len(row) for row in legal_actions), default=0)
        if Lmax == 0:
            raise ValueError("policy_logits_and_value: every legal_actions row is empty")

        (
            t_np,
            r_np,
            src_k_np,
            src_s_np,
            tgt_k_np,
            tgt_s_np,
            m_np,
        ) = _build_action_tokens(legal_actions, Lmax)
        device = obs.device
        type_ids = torch.from_numpy(t_np).to(device, non_blocking=True)
        role_ids = torch.from_numpy(r_np).to(device, non_blocking=True)
        src_region_kinds = torch.from_numpy(src_k_np).to(device, non_blocking=True)
        src_region_slots = torch.from_numpy(src_s_np).to(device, non_blocking=True)
        tgt_region_kinds = torch.from_numpy(tgt_k_np).to(device, non_blocking=True)
        tgt_region_slots = torch.from_numpy(tgt_s_np).to(device, non_blocking=True)
        mask = torch.from_numpy(m_np).to(device, non_blocking=True)

        return self.policy_logits_value_from_tokens(
            obs,
            type_ids,
            role_ids,
            src_region_kinds,
            src_region_slots,
            tgt_region_kinds,
            tgt_region_slots,
            mask,
            h_prev=h_prev,
            return_cache=return_cache,
        )


__all__ = ["BGLikeStructuredV4"]

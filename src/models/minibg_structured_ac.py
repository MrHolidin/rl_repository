"""MiniBG actor-critic: structured legal actions + optional autoregressive board-order head."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.envs.minibg.actions import BOARD_SIZE
from src.envs.minibg.structured_actions import StructAction, StructActionType

from .minibg_slot_ac import (
    _ENEMY_LEN,
    _GLOBAL_DIM,
    _HAND_LEN,
    _LAST_BATTLE_DIM,
    _OBS_DIM,
    _OWN_LEN,
    _PHASE_DIM,
    _SHOP_LEN,
    _SLOT_DIM,
    _TOTAL_SLOTS,
)

_NUM_STRUCT_TYPES = len(StructActionType)
_ROLE_NONE = 0
_ROLE_SHOP = 1
_ROLE_BOARD = 2
_ROLE_HAND = 3
_NUM_ROLES = 4

# Region "kinds" used by the batched action embedding gather. Index 0 is reserved
# for null-entity actions (ROLL/LEVEL_UP/COMPLETE_TURN) so padded slots also map there.
_REGION_NULL = 0
_REGION_SHOP = 1
_REGION_OWN = 2
_REGION_HAND = 3
_NUM_REGIONS = 4
_MAX_REGION_LEN = max(_OWN_LEN, _SHOP_LEN, _HAND_LEN, 1)

# Globals (gold, round, healths, actions_left, ...) + last_battle + phase routed
# explicitly into action / order scoring heads so the actor stops ignoring them.
_GLOBALS_FULL_DIM = _GLOBAL_DIM + _LAST_BATTLE_DIM + _PHASE_DIM


def role_for_struct(a: StructAction) -> int:
    if a.type == StructActionType.BUY:
        return _ROLE_SHOP
    if a.type == StructActionType.SELL:
        return _ROLE_BOARD
    if a.type == StructActionType.PLACE:
        return _ROLE_HAND
    return _ROLE_NONE


def _struct_action_codes(a: StructAction) -> Tuple[int, int, int, int]:
    """Return ``(type_id, role_id, region_kind, region_slot)`` for batched embedding."""
    t = a.type
    if t == StructActionType.BUY:
        return (int(t), _ROLE_SHOP, _REGION_SHOP, int(a.args[0]))
    if t == StructActionType.SELL:
        return (int(t), _ROLE_BOARD, _REGION_OWN, int(a.args[0]))
    if t == StructActionType.PLACE:
        return (int(t), _ROLE_HAND, _REGION_HAND, int(a.args[0]))
    # ROLL / LEVEL_UP / COMPLETE_TURN: null entity in region 0.
    return (int(t), _ROLE_NONE, _REGION_NULL, 0)


def _build_action_tokens(
    legal_actions: Sequence[Sequence[StructAction]],
    Lmax: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack (B, Lmax) int64 token arrays + bool mask. Padded slots stay at NULL/0."""
    B = len(legal_actions)
    type_ids = np.zeros((B, Lmax), dtype=np.int64)
    role_ids = np.zeros((B, Lmax), dtype=np.int64)
    region_kinds = np.zeros((B, Lmax), dtype=np.int64)
    region_slots = np.zeros((B, Lmax), dtype=np.int64)
    mask = np.zeros((B, Lmax), dtype=bool)
    for b in range(B):
        row = legal_actions[b]
        L = len(row)
        if L == 0:
            continue
        for l in range(L):
            t, r, k, s = _struct_action_codes(row[l])
            type_ids[b, l] = t
            role_ids[b, l] = r
            region_kinds[b, l] = k
            region_slots[b, l] = s
        mask[b, :L] = True
    return type_ids, role_ids, region_kinds, region_slots, mask


class MiniBGStructuredActorCritic(nn.Module):
    """
    Shared slot encoder → state embedding.
    Policy logits over variable legal sets via concat(state, action_emb).
    Separate GRU pointer head over board slots for COMPLETE_TURN ordering.
    """

    def __init__(
        self,
        *,
        slot_hidden: int = 16,
        trunk_hidden: int = 256,
        state_dim: int = 128,
        action_dim: int = 64,
        order_hidden: int = 64,
        order_pos_dim: int = 16,
        score_hidden: int = 128,
        order_score_hidden: int = 64,
        region_conv2_kernel: int = 1,
    ) -> None:
        super().__init__()
        self.slot_hidden = int(slot_hidden)
        self.trunk_hidden = int(trunk_hidden)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.order_hidden = int(order_hidden)
        self.order_pos_dim = int(order_pos_dim)
        self.score_hidden = int(score_hidden)
        self.order_score_hidden = int(order_score_hidden)

        k2 = int(region_conv2_kernel)
        if k2 not in (1, 3):
            raise ValueError("region_conv2_kernel must be 1 or 3")
        self._region_conv2_kernel = k2

        self.region_conv1 = nn.Conv1d(_SLOT_DIM, self.slot_hidden, kernel_size=1)
        if k2 == 3:
            self.region_conv2 = nn.Conv1d(
                self.slot_hidden, self.slot_hidden, kernel_size=3, padding=1
            )
        else:
            self.region_conv2 = nn.Conv1d(
                self.slot_hidden, self.slot_hidden, kernel_size=1
            )

        trunk_in = (
            _TOTAL_SLOTS * self.slot_hidden
            + _GLOBAL_DIM
            + _LAST_BATTLE_DIM
            + _PHASE_DIM
        )
        self.trunk_fc1 = nn.Linear(trunk_in, self.trunk_hidden)
        self.trunk_fc2 = nn.Linear(self.trunk_hidden, self.trunk_hidden)

        # Drop the redundant pooled-mean stream: it sits in span(trunk) already.
        self.state_proj = nn.Linear(self.trunk_hidden, self.state_dim)

        self.type_emb = nn.Embedding(_NUM_STRUCT_TYPES, self.action_dim)
        self.role_emb = nn.Embedding(_NUM_ROLES, self.action_dim)
        self.entity_to_action = nn.Linear(self.slot_hidden, self.action_dim)
        self.null_entity_action = nn.Parameter(torch.zeros(self.action_dim))

        # Non-linear scoring head with explicit globals routing. Without the ReLU the head
        # decomposes as `W_s·state + W_a·ae`, which cancels in softmax and forces the
        # actor to ignore state (only slot features drive action ranking).
        self.score_fc = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim + _GLOBALS_FULL_DIM, self.score_hidden),
            nn.ReLU(),
            nn.Linear(self.score_hidden, 1),
        )

        self.critic = nn.Linear(self.state_dim, 1)

        self.order_pos_emb = nn.Embedding(BOARD_SIZE, self.order_pos_dim)
        self.order_start = nn.Parameter(torch.zeros(self.slot_hidden))
        # Initialize GRU hidden from state_emb so the first pick already sees board context.
        self.order_init = nn.Linear(self.state_dim, self.order_hidden)
        gru_in = self.slot_hidden + self.order_pos_dim
        self.order_gru = nn.GRUCell(gru_in, self.order_hidden)
        # Same additivity fix for the order ranking head.
        order_score_in = (
            self.state_dim
            + self.slot_hidden
            + self.order_hidden
            + self.order_pos_dim
            + _GLOBALS_FULL_DIM
        )
        self.order_score = nn.Sequential(
            nn.Linear(order_score_in, self.order_score_hidden),
            nn.ReLU(),
            nn.Linear(self.order_score_hidden, 1),
        )

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        return {
            "slot_hidden": self.slot_hidden,
            "trunk_hidden": self.trunk_hidden,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "order_hidden": self.order_hidden,
            "order_pos_dim": self.order_pos_dim,
            "score_hidden": self.score_hidden,
            "order_score_hidden": self.order_score_hidden,
            "region_conv2_kernel": self._region_conv2_kernel,
        }

    def _unpack(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.shape[1] != _OBS_DIM:
            raise ValueError(f"Expected obs dim {_OBS_DIM}, got {x.shape[1]}")
        g = x[:, :_GLOBAL_DIM]
        i = _GLOBAL_DIM
        own = x[:, i : i + _OWN_LEN * _SLOT_DIM].view(-1, _OWN_LEN, _SLOT_DIM)
        i += _OWN_LEN * _SLOT_DIM
        shop = x[:, i : i + _SHOP_LEN * _SLOT_DIM].view(-1, _SHOP_LEN, _SLOT_DIM)
        i += _SHOP_LEN * _SLOT_DIM
        hand = x[:, i : i + _HAND_LEN * _SLOT_DIM].view(-1, _HAND_LEN, _SLOT_DIM)
        i += _HAND_LEN * _SLOT_DIM
        enemy = x[:, i : i + _ENEMY_LEN * _SLOT_DIM].view(-1, _ENEMY_LEN, _SLOT_DIM)
        i += _ENEMY_LEN * _SLOT_DIM
        lb = x[:, i : i + _LAST_BATTLE_DIM]
        i += _LAST_BATTLE_DIM
        phase = x[:, i : i + _PHASE_DIM]
        return g, own, shop, hand, enemy, lb, phase

    def _encode_region_slots(self, z: torch.Tensor) -> torch.Tensor:
        h = z.transpose(1, 2)
        h = F.relu(self.region_conv1(h))
        h = F.relu(self.region_conv2(h))
        return h.transpose(1, 2).contiguous()

    def encode_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        g, own, shop, hand, enemy, lb, phase = self._unpack(x)
        E_own = self._encode_region_slots(own)
        E_shop = self._encode_region_slots(shop)
        E_hand = self._encode_region_slots(hand)
        E_enemy = self._encode_region_slots(enemy)

        feat_flat = torch.cat(
            [
                E_own.flatten(1),
                E_shop.flatten(1),
                E_hand.flatten(1),
                E_enemy.flatten(1),
                g,
                lb,
                phase,
            ],
            dim=1,
        )
        t = F.relu(self.trunk_fc2(F.relu(self.trunk_fc1(feat_flat))))

        state_emb = self.state_proj(t)
        g_full = torch.cat([g, lb, phase], dim=-1)
        cache = {
            "E_own": E_own,
            "E_shop": E_shop,
            "E_hand": E_hand,
            "E_enemy": E_enemy,
            "trunk": t,
            "g_full": g_full,
        }
        return state_emb, cache

    def _encode_actions(
        self,
        type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        region_kinds: torch.Tensor,
        region_slots: torch.Tensor,
        cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Batched action embedding for ``(B, Lmax)`` token tensors → ``(B, Lmax, action_dim)``."""
        B, Lmax = type_ids.shape
        device = type_ids.device
        dtype = cache["E_own"].dtype

        type_e = self.type_emb(type_ids)
        role_e = self.role_emb(role_ids)

        # Pack region features in a single padded tensor so one gather covers all rows/actions.
        # Region 0 is the null region (zeros); padded slots map here and get overwritten by
        # null_entity_action below, so their exact value does not matter.
        E_regions = torch.zeros(
            B,
            _NUM_REGIONS,
            _MAX_REGION_LEN,
            self.slot_hidden,
            device=device,
            dtype=dtype,
        )
        E_regions[:, _REGION_SHOP, :_SHOP_LEN] = cache["E_shop"]
        E_regions[:, _REGION_OWN, :_OWN_LEN] = cache["E_own"]
        E_regions[:, _REGION_HAND, :_HAND_LEN] = cache["E_hand"]
        E_regions_flat = E_regions.view(B, _NUM_REGIONS * _MAX_REGION_LEN, self.slot_hidden)

        flat_idx = region_kinds * _MAX_REGION_LEN + region_slots
        flat_idx_exp = flat_idx.unsqueeze(-1).expand(-1, -1, self.slot_hidden)
        ent_slot = torch.gather(E_regions_flat, dim=1, index=flat_idx_exp)
        ent_from_slot = self.entity_to_action(ent_slot)

        ent_null = self.null_entity_action.view(1, 1, -1).expand(B, Lmax, -1)
        is_null = (region_kinds == _REGION_NULL).unsqueeze(-1)
        ent = torch.where(is_null, ent_null, ent_from_slot)

        return type_e + role_e + ent

    def _logits_from_state_and_tokens(
        self,
        state_emb: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        region_kinds: torch.Tensor,
        region_slots: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, Lmax = type_ids.shape
        ae = self._encode_actions(type_ids, role_ids, region_kinds, region_slots, cache)
        s_exp = state_emb.unsqueeze(1).expand(-1, Lmax, -1)
        g_exp = cache["g_full"].unsqueeze(1).expand(-1, Lmax, -1)
        h_all = torch.cat([s_exp, ae, g_exp], dim=-1)
        logits = self.score_fc(h_all.reshape(B * Lmax, -1)).squeeze(-1).view(B, Lmax)
        return logits.masked_fill(~mask, float("-inf"))

    def policy_logits_value_from_tokens(
        self,
        obs: torch.Tensor,
        type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        region_kinds: torch.Tensor,
        region_slots: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """Same shape contract as :meth:`policy_logits_and_value`, but skips token build (hot path)."""
        state_emb, cache = self.encode_state(obs)
        logits = self._logits_from_state_and_tokens(
            state_emb, cache, type_ids, role_ids, region_kinds, region_slots, mask
        )
        values = self.critic(state_emb).squeeze(-1)
        if return_cache:
            cache_out = dict(cache)
            cache_out["state_emb"] = state_emb
            return logits, mask, values, cache_out
        return logits, mask, values

    def policy_logits_and_value(
        self,
        obs: torch.Tensor,
        legal_actions: List[List[StructAction]],
        *,
        return_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """
        Returns:
            logits: (B, max_L) with -inf padding
            mask: (B, max_L) True for valid slots
            values: (B,)
        """
        B = obs.size(0)
        device = obs.device
        if len(legal_actions) != B:
            raise ValueError(
                f"legal_actions has {len(legal_actions)} rows but obs batch size is {B}"
            )
        Lmax = max((len(row) for row in legal_actions), default=0)
        if Lmax == 0:
            raise ValueError(
                "policy_logits_and_value: every legal_actions row is empty — invalid for structured policy."
            )

        t_np, r_np, k_np, s_np, m_np = _build_action_tokens(legal_actions, Lmax)
        type_ids = torch.from_numpy(t_np).to(device, non_blocking=True)
        role_ids = torch.from_numpy(r_np).to(device, non_blocking=True)
        region_kinds = torch.from_numpy(k_np).to(device, non_blocking=True)
        region_slots = torch.from_numpy(s_np).to(device, non_blocking=True)
        mask = torch.from_numpy(m_np).to(device, non_blocking=True)

        return self.policy_logits_value_from_tokens(
            obs,
            type_ids,
            role_ids,
            region_kinds,
            region_slots,
            mask,
            return_cache=return_cache,
        )

    def order_logits_step(
        self,
        state_emb: torch.Tensor,
        E_own: torch.Tensor,
        g_full: torch.Tensor,
        hidden: torch.Tensor,
        pos_e_row: torch.Tensor,
    ) -> torch.Tensor:
        """Logits over BOARD_SIZE slots: (B, BOARD_SIZE). ``pos_e_row`` is the (B, order_pos_dim) row."""
        B, K, _ = E_own.shape
        sh = hidden.unsqueeze(1).expand(-1, K, -1)
        pe = pos_e_row.unsqueeze(1).expand(-1, K, -1)
        se = state_emb.unsqueeze(1).expand(-1, K, -1)
        gx = g_full.unsqueeze(1).expand(-1, K, -1)
        h_cat = torch.cat([se, E_own, sh, pe, gx], dim=-1)
        return self.order_score(h_cat).squeeze(-1)

    def _order_init_hidden(self, state_emb: torch.Tensor) -> torch.Tensor:
        """Initial GRU hidden from state — gives the pointer head board context at pos=0."""
        return torch.tanh(self.order_init(state_emb))

    def sample_board_order(
        self,
        state_emb: torch.Tensor,
        E_own: torch.Tensor,
        g_full: torch.Tensor,
        occupied_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressive pointer over occupied board slots.

        Args:
            state_emb: (B, state_dim)
            E_own: (B, BOARD_SIZE, slot_hidden) — use first BOARD_SIZE own-board slots from encoder
            occupied_mask: (B, BOARD_SIZE) bool, True if board slot index currently holds a minion

        Returns:
            picked_slots: (B, BOARD_SIZE) int64, -1 padding after all picks done
            logprob_sum: (B,)
            remaining_after: (B, BOARD_SIZE) bool — should be all False unless input had >BOARD_SIZE True
        """
        B, K, _ = E_own.shape
        if K != BOARD_SIZE:
            raise ValueError(f"E_own must have BOARD_SIZE={BOARD_SIZE} slots, got {K}")
        device = E_own.device
        remaining = occupied_mask.clone()
        hidden = self._order_init_hidden(state_emb)
        slot_in = self.order_start.unsqueeze(0).expand(B, -1)

        logprob_sum = torch.zeros(B, device=device, dtype=E_own.dtype)
        picked = torch.full((B, BOARD_SIZE), -1, device=device, dtype=torch.long)
        batch_arange = torch.arange(B, device=device)

        # Precompute positional embeddings for all steps in one lookup (no per-step host→device sync).
        pos_table = self.order_pos_emb(torch.arange(BOARD_SIZE, device=device))

        # Fixed-length loop over BOARD_SIZE: cheap (<=4) and removes both `.item()` and `.any()`
        # CPU syncs that would otherwise stall every COMPLETE_TURN call.
        for pos in range(BOARD_SIZE):
            active = remaining.any(dim=1)
            pos_e_row = pos_table[pos].unsqueeze(0).expand(B, -1)

            gru_in = torch.cat([slot_in, pos_e_row], dim=-1)
            hidden_new = self.order_gru(gru_in, hidden)
            hidden = torch.where(active.unsqueeze(-1), hidden_new, hidden)

            logits = self.order_logits_step(state_emb, E_own, g_full, hidden, pos_e_row)
            logits = logits.masked_fill(~remaining, float("-inf"))
            # Fully-finished rows have all -inf — keep Categorical numerically valid.
            logits = logits.masked_fill((~active).unsqueeze(-1), 0.0)

            dist = Categorical(logits=logits)
            if deterministic:
                idx = logits.argmax(dim=-1)
            else:
                idx = dist.sample()

            lp = dist.log_prob(idx)
            logprob_sum = logprob_sum + active.to(logprob_sum.dtype) * lp

            picked[batch_arange, pos] = torch.where(active, idx, torch.full_like(idx, -1))

            sel_emb = E_own[batch_arange, idx]
            slot_in = torch.where(active.unsqueeze(-1), sel_emb, slot_in)

            mask_pick = F.one_hot(idx, num_classes=K).bool() & active.unsqueeze(-1)
            remaining = remaining & ~mask_pick

        return picked, logprob_sum, remaining

    def order_logprob_given_sequence(
        self,
        state_emb: torch.Tensor,
        E_own: torch.Tensor,
        g_full: torch.Tensor,
        occupied_mask: torch.Tensor,
        picked_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced log prob sum for PPO/training (B,). picked_slots (B, max_steps) indices or -1."""
        B, K, _ = E_own.shape
        device = E_own.device
        remaining = occupied_mask.clone()
        hidden = self._order_init_hidden(state_emb)
        slot_in = self.order_start.unsqueeze(0).expand(B, -1)
        logprob_sum = torch.zeros(B, device=device, dtype=E_own.dtype)
        batch_arange = torch.arange(B, device=device)

        max_steps = int(picked_slots.size(1))
        # Pre-build positional embeddings; saves a host→device sync per step.
        pos_table = self.order_pos_emb(torch.arange(max(max_steps, 1), device=device))

        for pos in range(max_steps):
            active = remaining.any(dim=1)
            idx = picked_slots[:, pos]
            valid_pick = active & (idx >= 0)

            pos_e_row = pos_table[pos].unsqueeze(0).expand(B, -1)
            gru_in = torch.cat([slot_in, pos_e_row], dim=-1)
            hidden_new = self.order_gru(gru_in, hidden)
            # Advance GRU only when the teacher row has a real pick at this step. Advancing while
            # active but idx==-1 desynchronizes teacher replay and can explode hidden → NaN logits.
            hidden = torch.where(valid_pick.unsqueeze(-1), hidden_new, hidden)

            logits = self.order_logits_step(state_emb, E_own, g_full, hidden, pos_e_row)
            logits = logits.masked_fill(~remaining, float("-inf"))
            # Two failure modes need 0-filled logits to keep Categorical valid: finished rows
            # (~active) and padding gap rows (active & idx<0). `~valid_pick` covers both.
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


__all__ = ["MiniBGStructuredActorCritic", "_OBS_DIM"]


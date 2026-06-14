"""PPO agent for MiniBG with structured legal actions + composite log-prob (main + order head)."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent
from .rollout_segments import (
    acting_seat_from_info,
    close_rollout_segment,
    compute_gae_advantages,
    seat_ids_array,
)
from ..envs.base import StepResult
from ..envs.minibg.actions import BOARD_SIZE
from ..envs.minibg.obs import SLOT_DIM
from ..envs.minibg.structured_actions import (
    StructAction,
    StructActionType,
    slot_pick_sequence_to_perm,
)
from ..features.action_space import DiscreteActionSpace
from ..features.observation_builder import ObservationType
from ..models.minibg_structured_ac import (
    MiniBGStructuredActorCritic,
    _build_action_tokens,
)
from ..models.structured_ac_protocol import StructuredActorCriticProtocol
from ..models.ppo_policy_factory import (
    PPO_NETWORK_MINIBG_STRUCTURED,
    default_ppo_network_kwargs,
    ppo_network_type_for_save,
    restore_ppo_actor_critic,
)

INFO_STRUCT_LEGAL = "minibg_struct_legal"
INFO_STRUCT_NEXT_LEGAL = "minibg_struct_next_legal"

# Phase 2: action-dim (Lmax) bucket granularity for the PPO update. Bucketing
# the per-round action tensor width to a fixed grid keeps the compiled update at
# a handful of static shapes (foundation for CUDA-graph capture). Padded slots
# are masked, so it is bit-identical to the exact Lmax.
_LMAX_BUCKET = 16


def _stack_and_release(arrs: List[np.ndarray]) -> np.ndarray:
    """Stack equal-shape arrays into one contiguous array, releasing each
    source as it's copied.

    ``np.stack`` holds both the input list and the new contiguous array at
    peak — for the obs buffer that's ~2x the obs footprint (~1 GB at 32k
    steps / v6 obs). Copying row-by-row and dropping each source reference
    keeps the peak at ~1x (the result array) plus a shrinking remainder.
    Mutates ``arrs`` in place (entries set to ``None``); the caller's buffer
    is cleared right after the update anyway.
    """
    n = len(arrs)
    if n == 0:
        raise ValueError("_stack_and_release: empty list")
    out = np.empty((n, *np.asarray(arrs[0]).shape), dtype=np.float32)
    for i in range(n):
        out[i] = arrs[i]
        arrs[i] = None  # type: ignore[call-overload]
    return out


class StructuredMiniBGRolloutBuffer:
    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.legal_lists: List[List[StructAction]] = []
        self.action_indices: List[int] = []
        self.complete_turn: List[bool] = []
        self.occupied_masks: List[np.ndarray] = []
        self.order_picks: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        # Only the *last* transition's next_obs is ever consumed (GAE bootstrap
        # of the final step — every interior step uses the next step's stored
        # value via GAE, never its own next_obs). Storing the full per-step
        # next_obs duplicated the entire obs buffer (~0.5 GB at 32k steps / v6
        # obs) for nothing, so we keep a single rolling slot instead.
        self.last_next_obs: Optional[np.ndarray] = None
        self.seat_ids: List[int] = []
        # v4 recurrent fields (optional — populated only by recurrent agents):
        # ``h_prev[t]`` = hidden state used by the model at decision time at step ``t``;
        # ``episode_ids[t]`` = monotonic worker-local episode id grouping steps into
        # (episode, seat) BPTT sequences during the PPO update.
        self.h_prev: List[np.ndarray] = []
        self.episode_ids: List[int] = []
        # Battle-prediction-head fields. On FINISH steps we reserve zero
        # placeholders; on the next combat-resolution step ``observe`` walks
        # back and backfills the actual snapshot + label.
        self.own_board_obs: List[np.ndarray] = []
        self.opp_board_obs: List[np.ndarray] = []
        self.attack_first: List[float] = []
        self.battle_target: List[float] = []
        self.battle_target_valid: List[bool] = []
        # Final placement (1..8) of the row's seat segment; -1 until the segment
        # closes (``close_rollout_segment`` backfills the whole segment). CE
        # target for the v8 distributional critic; ignored by scalar critics.
        self.placement_label: List[int] = []

    def add(
        self,
        *,
        obs: np.ndarray,
        legal_list: List[StructAction],
        action_index: int,
        complete_turn: bool,
        occupied_mask: np.ndarray,
        order_pick_row: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        next_obs: np.ndarray,
        next_legal_list: List[StructAction],
        seat_id: int = -1,
        h_prev: Optional[np.ndarray] = None,
        episode_id: int = 0,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.legal_lists.append(list(legal_list))
        self.action_indices.append(int(action_index))
        self.complete_turn.append(bool(complete_turn))
        self.occupied_masks.append(np.asarray(occupied_mask, dtype=bool))
        self.order_picks.append(np.asarray(order_pick_row, dtype=np.int64))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        # Overwrite the single rolling slot; only the final step's value is read
        # for the GAE bootstrap. ``next_legal_list`` is accepted for caller
        # compatibility but no longer stored (it was never read by the update).
        self.last_next_obs = np.asarray(next_obs, dtype=np.float32)
        self.seat_ids.append(int(seat_id))
        if h_prev is not None:
            self.h_prev.append(np.asarray(h_prev, dtype=np.float32))
        self.episode_ids.append(int(episode_id))
        # Battle-pred placeholders: always appended so list lengths stay in
        # sync with ``self.obs``. Shape is inferred from the first add when the
        # head is enabled; until then we just use empty arrays.
        self.own_board_obs.append(np.zeros(0, dtype=np.float32))
        self.opp_board_obs.append(np.zeros(0, dtype=np.float32))
        self.attack_first.append(0.0)
        self.battle_target.append(0.0)
        self.battle_target_valid.append(False)
        self.placement_label.append(-1)

    def __len__(self) -> int:
        return len(self.obs)

    def clear(self) -> None:
        self.obs.clear()
        self.legal_lists.clear()
        self.action_indices.clear()
        self.complete_turn.clear()
        self.occupied_masks.clear()
        self.order_picks.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.last_next_obs = None
        self.seat_ids.clear()
        self.h_prev.clear()
        self.episode_ids.clear()
        self.own_board_obs.clear()
        self.opp_board_obs.clear()
        self.attack_first.clear()
        self.battle_target.clear()
        self.battle_target_valid.clear()
        self.placement_label.clear()


class MiniBGPPOStructuredAgent(BaseAgent):
    """On-policy PPO with ``MiniBGStructuredActorCritic``; use with ``Trainer`` structured MiniBG branch."""

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        observation_type: ObservationType,
        num_actions: int,
        network: nn.Module,
        *,
        ppo_network_type: str = PPO_NETWORK_MINIBG_STRUCTURED,
        ppo_network_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_clip_eps: float = 0.2,
        clip_value_loss: bool = True,
        value_clip_eps: Optional[float] = None,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        rollout_steps: int = 1024,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        model_config: Optional[Dict] = None,
        compute_detailed_metrics: bool = True,
        patch_build: Optional[int] = None,
    ) -> None:
        if not isinstance(network, StructuredActorCriticProtocol):
            raise TypeError(
                "MiniBGPPOStructuredAgent: network must satisfy StructuredActorCriticProtocol "
                f"(got {type(network).__name__}). Check methods: encode_state, "
                "policy_logits_and_value, policy_logits_value_from_tokens, sample_board_order, "
                "order_logprob_given_sequence, get_constructor_kwargs."
            )
        self.observation_shape = observation_shape
        self.observation_type = observation_type
        self.num_actions = int(num_actions)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy_net = network.to(self.device)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.ppo_clip_eps = ppo_clip_eps
        self.clip_value_loss = clip_value_loss
        self.value_clip_eps = value_clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.model_config = model_config
        self.compute_detailed_metrics = compute_detailed_metrics
        self.patch_build = int(patch_build) if patch_build is not None else None
        self.action_space = DiscreteActionSpace(self.num_actions)

        self.training = True
        self.step_count = 0
        self.epsilon = 0.0
        self.epsilon_decay = 1.0
        self.epsilon_min = 0.0

        self._ppo_network_type = ppo_network_type_for_save(ppo_network_type)
        self._ppo_network_kwargs = dict(
            ppo_network_kwargs or default_ppo_network_kwargs(ppo_network_type, self.policy_net)
        )

        # Fused Adam collapses the per-parameter optimizer kernels into one CUDA
        # launch — ~6x faster .step() on this many-small-tensor net (measured
        # 12.9ms -> 2.1ms), i.e. ~2.8s/round off the host PPO update. CUDA-only;
        # workers (CPU) fall back to the standard implementation.
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
            fused=(self.device.type == "cuda"),
        )
        self.rollout_buffer = StructuredMiniBGRolloutBuffer()
        self._cache: Optional[Dict[str, Any]] = None

        # --- v4 recurrent state -----------------------------------------
        # Models that expose ``step_round_hidden`` + ``zero_hidden`` are
        # treated as round-level recurrent. v2/v3 don't, so the recurrent
        # bookkeeping is a no-op for them and the existing flat PPO update
        # path runs unchanged.
        self._is_recurrent: bool = (
            hasattr(self.policy_net, "step_round_hidden")
            and hasattr(self.policy_net, "zero_hidden")
            and hasattr(self.policy_net, "recurrent_hidden_dim")
        )
        self._recurrent_hidden_dim: int = (
            int(getattr(self.policy_net, "recurrent_hidden_dim", 0))
            if self._is_recurrent
            else 0
        )
        # Per-seat hidden state during rollout collection. Cleared on episode
        # end so the next episode starts from zeros.
        self._hidden_by_seat: Dict[int, np.ndarray] = {}
        # Monotonic worker-local episode counter; populated into each rollout
        # row so the PPO update can group steps into (episode, seat) sequences
        # for BPTT replay.
        self._episode_id: int = 0

        # --- Distributional placement critic (v8) ----------------------
        # Detected from the network surface; replaces value-MSE with CE on
        # per-segment final-placement labels (backfilled at segment close).
        self._distributional: bool = hasattr(self.policy_net, "placement_logits")
        if self._distributional and self._is_recurrent:
            raise ValueError(
                "distributional placement critic is not wired into the recurrent "
                "(v4 BPTT) update path; use a non-recurrent network"
            )

        # --- Battle-prediction-head hyperparameters --------------------
        # Mirror the values on the model (which carries them via constructor
        # kwargs). The agent reads them in observe() to gate backfill and in
        # _ppo_struct_update_flat to compute aux loss.
        self._battle_pred_enabled: bool = bool(
            getattr(self.policy_net, "_battle_pred_enabled", False)
        )
        bp_cfg: Dict[str, Any] = dict(
            getattr(self.policy_net, "battle_pred_config", {}) or {}
        )
        self._battle_pred_aux_coef: float = float(bp_cfg.get("aux_coef", 0.01))
        self._battle_pred_detach: bool = bool(bp_cfg.get("detach_features", False))
        self._battle_pred_huber_delta: float = float(bp_cfg.get("huber_delta", 5.0))

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        raise RuntimeError(
            "MiniBGPPOStructuredAgent: use Trainer structured branch + act_structured(...); "
            "flat act() is not supported."
        )

    def opponent_step(
        self,
        env: Any,
        obs: np.ndarray,
        *,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> StepResult:
        """Used by ``AgentPerspectiveEnv`` self-play drain (needs ``step_structured``, not flat ``step``)."""
        legal_list = env.legal_structured_actions()
        was_training = self.training
        self.training = False
        try:
            struct_act, board_perm, _idx = self.act_structured(
                obs,
                legal_list,
                env,
                deterministic=deterministic,
            )
        finally:
            self.training = was_training
        return env.step_structured(struct_act, board_perm=board_perm)

    def act_structured(
        self,
        obs: np.ndarray,
        legal_list: List[StructAction],
        env: Any,
        *,
        deterministic: bool = False,
    ) -> Tuple[StructAction, Optional[Tuple[int, ...]], int]:
        if not legal_list:
            raise ValueError("legal_list empty")

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = env.state
        cur_idx = state.current_player_index
        player = state.players[cur_idx]
        k_board = len(player.board)
        board_size = int(getattr(self.policy_net, "board_size", BOARD_SIZE))
        occupied_np = np.zeros(board_size, dtype=bool)
        if k_board > 0:
            occupied_np[:k_board] = True

        # v4: read per-seat hidden state (zeros if first time we see this seat
        # in the current episode). Stored as numpy in the cache so it can be
        # pickled into the rollout buffer at observe() time.
        h_prev_np: Optional[np.ndarray] = None
        h_prev_t: Optional[torch.Tensor] = None
        if self._is_recurrent:
            h_prev_np = self._hidden_by_seat.get(int(cur_idx))
            if h_prev_np is None:
                h_prev_np = np.zeros(self._recurrent_hidden_dim, dtype=np.float32)
            h_prev_t = torch.as_tensor(
                h_prev_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        with torch.no_grad():
            if self._is_recurrent:
                logits, mask, value, enc_cache = self.policy_net.policy_logits_and_value(
                    obs_t, [legal_list], h_prev=h_prev_t, return_cache=True
                )
            else:
                logits, mask, value, enc_cache = self.policy_net.policy_logits_and_value(
                    obs_t, [legal_list], return_cache=True
                )
            logits = logits.masked_fill(~mask, float("-inf"))

            if deterministic or not self.training:
                idx = int(logits.argmax(dim=-1).item())
                log_main = F.log_softmax(logits, dim=-1)[0, idx]
            else:
                dist = torch.distributions.Categorical(logits=logits)
                idx_t = dist.sample()
                idx = int(idx_t.item())
                log_main = dist.log_prob(idx_t)

            chosen = legal_list[idx]
            log_order = torch.zeros((), device=self.device, dtype=torch.float32)
            picks_np = np.full(board_size, -1, dtype=np.int64)
            board_perm: Optional[Tuple[int, ...]] = None

            if chosen.type in (
                StructActionType.COMPLETE_TURN,
                StructActionType.COMPLETE_TURN_FREEZE_SHOP,
            ):
                state_emb = enc_cache["state_emb"]
                e_own = enc_cache["E_own"]
                g_full = enc_cache["g_full"]
                occ_t = torch.as_tensor(occupied_np, dtype=torch.bool, device=self.device).unsqueeze(0)
                picked_t, log_order, _ = self.policy_net.sample_board_order(
                    state_emb,
                    e_own,
                    g_full,
                    occ_t,
                    deterministic=deterministic,
                )
                picks_np = picked_t[0].detach().cpu().numpy().astype(np.int64)
                seq = [int(x) for x in picks_np if int(x) >= 0][: max(k_board, 0)]
                board_perm = slot_pick_sequence_to_perm(seq, k_board, board_size=board_size)

            log_total = log_main + log_order.reshape(())

        if self.training:
            is_complete_turn = chosen.type in (
                StructActionType.COMPLETE_TURN,
                StructActionType.COMPLETE_TURN_FREEZE_SHOP,
            )
            cache_entry: Dict[str, Any] = {
                "obs": np.asarray(obs, dtype=np.float32),
                "legal_list": list(legal_list),
                "action_idx": idx,
                "value": float(value.squeeze(0).item()),
                "log_prob": float(log_total.item()),
                "complete_turn": is_complete_turn,
                "occupied_mask": occupied_np.copy(),
                "order_picks": picks_np.copy(),
                "seat": int(cur_idx),
            }
            if self._is_recurrent:
                cache_entry["h_prev"] = h_prev_np  # already numpy
                # For round-boundary steps we need state_emb to advance the GRU
                # in observe(). state_emb is (1, state_dim); store as 1-D numpy.
                if is_complete_turn:
                    state_emb_np = (
                        enc_cache["state_emb"].squeeze(0).detach().cpu().numpy().astype(np.float32)
                    )
                    cache_entry["state_emb_for_gru"] = state_emb_np
            self._cache = cache_entry
        # Opponent / eval forwards must not clear _cache — drain runs before observe.

        return chosen, board_perm, idx

    def observe(self, transition: Any, is_augmented: bool = False) -> Dict[str, float]:
        if not self.training:
            return {}
        if is_augmented:
            return {}

        assert hasattr(transition, "obs"), "MiniBG structured observe: expected Transition-like object"
        action = int(transition.action)
        reward = float(transition.reward)
        next_obs = transition.next_obs
        done = transition.terminated or transition.truncated
        info = transition.info or {}

        cache = self._cache
        assert cache is not None, "MiniBG structured observe: missing cache after act_structured"
        assert cache["action_idx"] == action, (
            f"MiniBG structured observe: action mismatch cache={cache['action_idx']} transition={action}"
        )

        legal_list = info.get(INFO_STRUCT_LEGAL)
        next_legal = info.get(INFO_STRUCT_NEXT_LEGAL)
        assert next_legal is not None, (
            "MiniBG structured observe: INFO_STRUCT_NEXT_LEGAL missing "
            "(trainer must attach legal next actions)"
        )
        assert legal_list is not None, (
            "MiniBG structured observe: INFO_STRUCT_LEGAL missing "
            "(trainer must attach legal list at decision)"
        )
        assert list(legal_list) == list(cache["legal_list"]), (
            "MiniBG structured observe: legal_list at transition differs from cached decision list"
        )

        if cache["complete_turn"]:
            occ = np.asarray(cache["occupied_mask"], dtype=bool)
            picks = np.asarray(cache["order_picks"], dtype=np.int64)
            k_occ = int(occ.sum())
            k_pick = int(np.sum(picks >= 0))
            assert k_occ == k_pick, (
                f"COMPLETE_TURN: occupied_slots={k_occ} != count(order_picks>=0)={k_pick}"
            )

        seat_id = acting_seat_from_info(info)
        if seat_id < 0:
            seat_id = cache.get("seat", -1)

        h_prev_for_buffer = cache.get("h_prev") if self._is_recurrent else None

        self.rollout_buffer.add(
            obs=cache["obs"],
            legal_list=cache["legal_list"],
            action_index=cache["action_idx"],
            complete_turn=cache["complete_turn"],
            occupied_mask=cache["occupied_mask"],
            order_pick_row=cache["order_picks"],
            reward=reward,
            done=done,
            value=cache["value"],
            log_prob=cache["log_prob"],
            next_obs=np.asarray(next_obs, dtype=np.float32),
            next_legal_list=list(next_legal),
            seat_id=seat_id,
            h_prev=h_prev_for_buffer,
            episode_id=self._episode_id,
        )

        # v4: advance the seat's round-level hidden state on COMPLETE_TURN.
        # GRU step uses the state_emb computed at decision time, so the seat's
        # next shop step in the next round will see the updated h.
        if self._is_recurrent and cache["complete_turn"]:
            seat_for_h = int(cache.get("seat", seat_id))
            state_emb_np = cache.get("state_emb_for_gru")
            h_prev_np = cache.get("h_prev")
            if state_emb_np is not None and h_prev_np is not None and seat_for_h >= 0:
                with torch.no_grad():
                    state_emb_t = torch.as_tensor(
                        state_emb_np, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    h_prev_t = torch.as_tensor(
                        h_prev_np, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    h_new_t = self.policy_net.step_round_hidden(state_emb_t, h_prev_t)
                self._hidden_by_seat[seat_for_h] = (
                    h_new_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )

        # On episode end: clear per-seat hidden state and bump episode id so
        # the next episode's rollout rows are tagged as a fresh BPTT group.
        if self._is_recurrent and done:
            self._hidden_by_seat.clear()
            self._episode_id += 1

        # Battle-prediction-head backfill: on combat-resolution steps the env
        # exposes per-seat battle data. For each seat we find the most recent
        # FINISH-row in the buffer without a valid battle target and stamp it
        # with the snapshot + label. ``observe`` is called once per env step,
        # so doing it here naturally picks up all combats from this resolution.
        if (
            self._battle_pred_enabled
            and info.get("combat_advanced")
            and info.get("battle_data_per_seat")
        ):
            self._backfill_battle_data(info["battle_data_per_seat"])

        self._cache = None
        self.step_count += 1
        return {}

    def _backfill_battle_data(self, battle_data: Dict[int, Dict[str, Any]]) -> None:
        buf = self.rollout_buffer
        for seat, data in battle_data.items():
            seat_i = int(seat)
            # Walk back the buffer to find the most recent FINISH-row for this
            # seat with battle_target still unfilled. There can be at most one
            # such row per combat-resolution event per seat by construction.
            for idx in range(len(buf) - 1, -1, -1):
                if buf.seat_ids[idx] != seat_i:
                    continue
                if not buf.complete_turn[idx]:
                    continue
                if buf.battle_target_valid[idx]:
                    # Already filled — older FINISH already paired with its
                    # combat. Stop the walk: anything earlier is also filled.
                    break
                buf.own_board_obs[idx] = np.asarray(data["own_board_obs"], dtype=np.float32)
                buf.opp_board_obs[idx] = np.asarray(data["opp_board_obs"], dtype=np.float32)
                buf.attack_first[idx] = float(data["attack_first"])
                buf.battle_target[idx] = float(data["damage_signed_uncapped"])
                buf.battle_target_valid[idx] = True
                break

    def close_segment(
        self, seat: int, terminal_reward: float, placement: Optional[int] = None
    ) -> bool:
        """Mark the last rollout step for ``seat`` as segment-terminal with ``terminal_reward``.

        ``placement`` (1..8, bglike) backfills the distributional-critic CE
        label over the whole segment; ``None`` keeps scalar-critic behaviour.
        """
        if not self.training:
            return False
        return close_rollout_segment(
            self.rollout_buffer, seat, terminal_reward, placement=placement
        )

    def update(self) -> Dict[str, float]:
        if not self.training:
            return {}
        if len(self.rollout_buffer) < self.rollout_steps:
            return {
                "rollout_size": len(self.rollout_buffer),
                "rollout_capacity": self.rollout_steps,
                "buffer_utilization": len(self.rollout_buffer) / float(self.rollout_steps),
            }
        metrics = self._ppo_struct_update()
        self.rollout_buffer.clear()
        return metrics

    def _value_only(self, obs_1: torch.Tensor) -> torch.Tensor:
        _, enc_cache = self.policy_net.encode_state(obs_1)
        if self._distributional:
            return self.policy_net.value_from_trunk(enc_cache["trunk"]).reshape(-1)
        return self.policy_net.critic(enc_cache["trunk"]).reshape(-1)

    def _ppo_struct_update(self) -> Dict[str, float]:
        if self._is_recurrent:
            return self._ppo_struct_update_recurrent()
        return self._ppo_struct_update_flat()

    def _ppo_struct_update_flat(self) -> Dict[str, float]:
        buf = self.rollout_buffer
        device = self.device

        obs_arr = _stack_and_release(buf.obs)
        rewards_arr = np.array(buf.rewards, dtype=np.float32)
        dones_arr = np.array(buf.dones, dtype=np.bool_)
        values_arr = np.array(buf.values, dtype=np.float32)
        log_probs_old_arr = np.array(buf.log_probs, dtype=np.float32)
        actions_arr = np.array(buf.action_indices, dtype=np.int64)
        complete_arr = np.array(buf.complete_turn, dtype=np.bool_)
        occupied_arr = np.stack(buf.occupied_masks, axis=0)
        picks_arr = np.stack(buf.order_picks, axis=0)

        N = obs_arr.shape[0]
        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
        if buf.last_next_obs is None:
            raise RuntimeError("rollout buffer has no last_next_obs for bootstrap")
        next_obs_last = torch.as_tensor(
            buf.last_next_obs[None], dtype=torch.float32, device=device
        )
        values_tensor = torch.as_tensor(values_arr, dtype=torch.float32, device=device)
        log_probs_old_tensor = torch.as_tensor(log_probs_old_arr, dtype=torch.float32, device=device)
        actions_tensor = torch.as_tensor(actions_arr, dtype=torch.long, device=device)
        complete_tensor = torch.as_tensor(complete_arr, dtype=torch.bool, device=device)
        occupied_tensor = torch.as_tensor(occupied_arr, dtype=torch.bool, device=device)
        picks_tensor = torch.as_tensor(picks_arr, dtype=torch.long, device=device)
        # v8 distributional critic: per-row final-placement labels (1..8, -1 = no
        # label — only possible for segments cut at the buffer edge; masked out).
        placement_tensor = (
            torch.as_tensor(
                np.array(buf.placement_label, dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            if self._distributional
            else None
        )

        # GAE on CPU/numpy: zero bootstrap across seat segment boundaries.
        if bool(dones_arr[-1]):
            last_bootstrap = 0.0
        else:
            with torch.no_grad():
                last_bootstrap = float(self._value_only(next_obs_last).reshape(()).item())

        seat_ids_arr = seat_ids_array(buf.seat_ids, N)
        adv_np, ret_np = compute_gae_advantages(
            rewards_arr,
            values_arr,
            dones_arr,
            seat_ids_arr,
            discount_factor=self.discount_factor,
            gae_lambda=self.gae_lambda,
            last_next_value=last_bootstrap,
        )

        advantages = torch.from_numpy(adv_np).to(device)
        returns = torch.from_numpy(ret_np).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        return_mean = returns.mean().item()
        adv_mean = advantages.mean().item()
        adv_std = advantages.std(unbiased=False).item()
        # Explained variance of the pre-update critic on this rollout's returns:
        # 1 - Var(R - V)/Var(R). Self-normalizes against the return distribution
        # of the current opponent pool, so it stays comparable as the league
        # strengthens (unlike raw value_loss).
        ret_var = float(np.var(ret_np))
        explained_variance = (
            1.0 - float(np.var(ret_np - values_arr)) / ret_var if ret_var > 1e-12 else 0.0
        )

        # Pre-tokenize the entire rollout buffer's legal lists. One H2D copy per update
        # replaces O(N * Lmax * mini_batches * epochs) per-action kernel launches that were
        # the dominant cost of `update()`.
        Lmax_global = max((len(row) for row in buf.legal_lists), default=0)
        if Lmax_global == 0:
            raise ValueError(
                "structured PPO update: empty legal_lists in rollout buffer (should be unreachable)"
            )
        # Phase 2 (static-shape foundation): bucket the action dim up to a fixed
        # grid so the compiled update sees only a handful of Lmax shapes (the
        # prerequisite for CUDA-graph capture). The extra slots are NULL and get
        # masked to -inf in the action head — the SAME padding the model already
        # applies to rows shorter than Lmax_global — so logits/loss/grads are
        # bit-identical to using the exact Lmax_global.
        Lmax_padded = (
            (Lmax_global + _LMAX_BUCKET - 1) // _LMAX_BUCKET
        ) * _LMAX_BUCKET
        (
            t_np,
            r_np,
            src_k_np,
            src_s_np,
            tgt_k_np,
            tgt_s_np,
            m_np,
        ) = _build_action_tokens(buf.legal_lists, Lmax_padded)
        type_ids_all = torch.from_numpy(t_np).to(device, non_blocking=True)
        role_ids_all = torch.from_numpy(r_np).to(device, non_blocking=True)
        src_region_kinds_all = torch.from_numpy(src_k_np).to(device, non_blocking=True)
        src_region_slots_all = torch.from_numpy(src_s_np).to(device, non_blocking=True)
        tgt_region_kinds_all = torch.from_numpy(tgt_k_np).to(device, non_blocking=True)
        tgt_region_slots_all = torch.from_numpy(tgt_s_np).to(device, non_blocking=True)
        mask_all = torch.from_numpy(m_np).to(device, non_blocking=True)

        # Battle-prediction-head tensors. Uniform-shape (N, BOARD_SIZE, SLOT_DIM);
        # invalid rows stay zeros and are masked by ``bp_valid_tensor``.
        if self._battle_pred_enabled:
            board_size = int(getattr(self.policy_net, "board_size", BOARD_SIZE))
            bp_own = np.zeros((N, board_size, SLOT_DIM), dtype=np.float32)
            bp_opp = np.zeros((N, board_size, SLOT_DIM), dtype=np.float32)
            for i in range(N):
                if buf.battle_target_valid[i]:
                    bp_own[i] = buf.own_board_obs[i]
                    bp_opp[i] = buf.opp_board_obs[i]
            bp_own_tensor = torch.from_numpy(bp_own).to(device, non_blocking=True)
            bp_opp_tensor = torch.from_numpy(bp_opp).to(device, non_blocking=True)
            bp_attack_first_tensor = torch.from_numpy(
                np.array(buf.attack_first, dtype=np.float32)
            ).to(device, non_blocking=True)
            bp_target_tensor = torch.from_numpy(
                np.array(buf.battle_target, dtype=np.float32)
            ).to(device, non_blocking=True)
            bp_valid_tensor = torch.from_numpy(
                np.array(buf.battle_target_valid, dtype=np.bool_)
            ).to(device, non_blocking=True)
        else:
            bp_own_tensor = None
            bp_opp_tensor = None
            bp_attack_first_tensor = None
            bp_target_tensor = None
            bp_valid_tensor = None

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_batches = 0
        total_clip_frac = 0.0
        total_battle_loss = 0.0
        total_battle_mae = 0.0
        total_battle_corr = 0.0
        total_battle_sign_acc = 0.0
        total_battle_batches = 0
        total_battle_corr_batches = 0  # corr undefined for single-sample minibatches
        total_placement_acc = 0.0
        total_placement_batches = 0
        grad_norm: torch.Tensor | float = 0.0

        # GPU-side scalar accumulators: sum per-minibatch metrics on-device and
        # sync ONCE after the loop, instead of one .item() per minibatch (each a
        # GPU->CPU stall). These metrics are logging-only — they never feed back
        # into training — so on-device summation is equivalent. Also a
        # prerequisite for CUDA-graph capture (no mid-loop host syncs).
        _zeros = lambda: torch.zeros((), device=device)
        acc_policy, acc_value, acc_entropy = _zeros(), _zeros(), _zeros()
        acc_kl, acc_clip, acc_place = _zeros(), _zeros(), _zeros()

        indices = np.arange(N, dtype=np.int64)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                if mb_idx.size == 0:
                    continue

                mb_idx_t = torch.from_numpy(mb_idx).to(device)
                obs_mb = obs_tensor[mb_idx_t]
                advantages_mb = advantages[mb_idx_t]
                returns_mb = returns[mb_idx_t]
                log_probs_old_mb = log_probs_old_tensor[mb_idx_t]
                actions_mb = actions_tensor[mb_idx_t]
                complete_mb = complete_tensor[mb_idx_t]
                occupied_mb = occupied_tensor[mb_idx_t]
                picks_mb = picks_tensor[mb_idx_t]
                values_old_mb = values_tensor[mb_idx_t]

                type_ids_mb = type_ids_all[mb_idx_t]
                role_ids_mb = role_ids_all[mb_idx_t]
                src_region_kinds_mb = src_region_kinds_all[mb_idx_t]
                src_region_slots_mb = src_region_slots_all[mb_idx_t]
                tgt_region_kinds_mb = tgt_region_kinds_all[mb_idx_t]
                tgt_region_slots_mb = tgt_region_slots_all[mb_idx_t]
                mask_mb = mask_all[mb_idx_t]

                logits, mask, values_new_mb, cache = (
                    self.policy_net.policy_logits_value_from_tokens(
                        obs_mb,
                        type_ids_mb,
                        role_ids_mb,
                        src_region_kinds_mb,
                        src_region_slots_mb,
                        tgt_region_kinds_mb,
                        tgt_region_slots_mb,
                        mask_mb,
                        return_cache=True,
                    )
                )

                B = obs_mb.shape[0]
                # `logits` is already masked with -inf on padded slots inside the network.
                log_sm = F.log_softmax(logits, dim=-1)
                batch_ar = torch.arange(B, device=device)
                lp_main = log_sm[batch_ar, actions_mb]

                # Mask illegal slots before p*log_sm: 0*(-inf) is NaN and poisons entropy backward.
                log_sm_safe = log_sm.masked_fill(~mask, 0.0)
                p_safe = log_sm.exp().masked_fill(~mask, 0.0)
                ent_row = -(p_safe * log_sm_safe).sum(dim=-1)
                entropy_mb = ent_row.mean()

                # Mix in autoregressive order-head log-prob without a `.any().item()` sync.
                # `order_logprob_given_sequence` already masks rows that don't have a real pick,
                # so calling it on the whole minibatch + masking with `complete_mb` is safe.
                lp_ord_all = self.policy_net.order_logprob_given_sequence(
                    cache["state_emb"],
                    cache["E_own"],
                    cache["g_full"],
                    occupied_mb,
                    picks_mb,
                )
                log_probs_mb = lp_main + complete_mb.to(lp_main.dtype) * lp_ord_all

                ratio = torch.exp(log_probs_mb - log_probs_old_mb)
                with torch.no_grad():
                    clipped = (ratio < 1.0 - self.ppo_clip_eps) | (ratio > 1.0 + self.ppo_clip_eps)
                    clip_frac_t = clipped.to(ratio.dtype).mean()

                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                values_old_flat = values_old_mb.reshape(-1)
                returns_flat = returns_mb.reshape(-1)
                if self._distributional:
                    # CE on final placement replaces the value regression; the
                    # scalar V (its expectation) still feeds GAE unchanged.
                    # Value clipping is a regression concept — not applied.
                    place_mb = placement_tensor[mb_idx_t]
                    place_valid = place_mb >= 1
                    pl_logits = self.policy_net.placement_logits(cache["trunk"])
                    if bool(place_valid.any()):
                        value_loss = F.cross_entropy(
                            pl_logits[place_valid], place_mb[place_valid] - 1
                        )
                        with torch.no_grad():
                            acc_t = (
                                pl_logits[place_valid].argmax(dim=-1) + 1
                                == place_mb[place_valid]
                            ).float().mean()
                        acc_place += acc_t.detach()
                        total_placement_batches += 1
                    else:
                        value_loss = pl_logits.sum() * 0.0
                elif self.clip_value_loss:
                    eps_v = (
                        float(self.value_clip_eps)
                        if self.value_clip_eps is not None
                        else float(self.ppo_clip_eps)
                    )
                    v_err_u = values_new_mb - returns_flat
                    v_clipped = values_old_flat + torch.clamp(
                        values_new_mb - values_old_flat, -eps_v, eps_v
                    )
                    v_err_c = v_clipped - returns_flat
                    value_loss = 0.5 * torch.maximum(v_err_u.pow(2), v_err_c.pow(2)).mean()
                else:
                    value_loss = 0.5 * (values_new_mb - returns_flat).pow(2).mean()

                approx_kl = 0.5 * (log_probs_old_mb - log_probs_mb).pow(2).mean()

                # ---- Auxiliary battle-prediction head ----
                battle_loss_term = None
                battle_loss_item = None
                battle_mae_item = None
                battle_corr_item = None
                battle_sign_acc_item = None
                if self._battle_pred_enabled and bp_valid_tensor is not None:
                    valid_mb = bp_valid_tensor[mb_idx_t]
                    if bool(valid_mb.any().item()):
                        # Board-only forward: shop/hand/past-battles deliberately
                        # don't enter via state_emb. Conv weights are shared with
                        # the actor encoder, so gradient still regularizes the
                        # backbone unless detach_features=True.
                        idx_valid = torch.nonzero(valid_mb, as_tuple=False).squeeze(-1)
                        own_v = bp_own_tensor[mb_idx_t][idx_valid]
                        opp_v = bp_opp_tensor[mb_idx_t][idx_valid]
                        af_v = bp_attack_first_tensor[mb_idx_t][idx_valid]
                        tgt_v = bp_target_tensor[mb_idx_t][idx_valid]
                        pred = self.policy_net.predict_battle(
                            own_v,
                            opp_v,
                            af_v,
                            detach_features=self._battle_pred_detach,
                        )
                        battle_loss_term = F.smooth_l1_loss(
                            pred, tgt_v, beta=self._battle_pred_huber_delta, reduction="mean"
                        )
                        with torch.no_grad():
                            err = pred - tgt_v
                            battle_mae_item = float(err.abs().mean().item())
                            # Pearson correlation in this minibatch (= cosim of
                            # centered prediction/target vectors). Bounded [-1, 1].
                            if pred.numel() >= 2:
                                p_c = pred - pred.mean()
                                t_c = tgt_v - tgt_v.mean()
                                denom = (p_c.norm() * t_c.norm()).clamp(min=1e-8)
                                battle_corr_item = float(((p_c * t_c).sum() / denom).item())
                            # Sign accuracy: did we get the win/loss direction right?
                            # Tie (target==0): considered correct iff pred is also 0;
                            # in practice pred is never exactly 0, so treat target==0
                            # as agreement when |pred| < huber_delta/4 (small).
                            sign_correct = torch.sign(pred) == torch.sign(tgt_v)
                            # Zero targets: count as correct if |pred| < delta/4.
                            zero_mask = tgt_v == 0
                            if zero_mask.any():
                                small_pred = pred.abs() < (self._battle_pred_huber_delta * 0.25)
                                sign_correct = torch.where(zero_mask, small_pred, sign_correct)
                            battle_sign_acc_item = float(sign_correct.float().mean().item())
                        battle_loss_item = float(battle_loss_term.item())

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mb
                if battle_loss_term is not None:
                    # ``aux_coef`` controls how strongly the head pulls on the
                    # backbone. We always add ``battle_loss_term`` so the head's
                    # own MLP gets gradient; if ``detach_features=True`` the
                    # gradient stops at the head and the backbone is untouched.
                    loss = loss + self._battle_pred_aux_coef * battle_loss_term

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                acc_policy += policy_loss.detach()
                acc_value += value_loss.detach()
                acc_entropy += entropy_mb.detach()
                acc_kl += approx_kl.detach()
                acc_clip += clip_frac_t.detach()
                total_batches += 1
                if battle_loss_item is not None:
                    total_battle_loss += battle_loss_item
                    total_battle_mae += battle_mae_item
                    total_battle_batches += 1
                    if battle_sign_acc_item is not None:
                        total_battle_sign_acc += battle_sign_acc_item
                    if battle_corr_item is not None:
                        total_battle_corr += battle_corr_item
                        total_battle_corr_batches += 1

        if total_batches == 0:
            return {}

        # Single GPU->CPU sync for all per-minibatch metrics (see accumulators above).
        total_policy_loss = float(acc_policy.item())
        total_value_loss = float(acc_value.item())
        total_entropy = float(acc_entropy.item())
        total_approx_kl = float(acc_kl.item())
        total_clip_frac = float(acc_clip.item())
        total_placement_acc = float(acc_place.item())

        gn = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        out_metrics = {
            "loss": (total_policy_loss + self.value_coef * total_value_loss - self.entropy_coef * total_entropy)
            / total_batches,
            "policy_loss": total_policy_loss / total_batches,
            "value_loss": total_value_loss / total_batches,
            "entropy": total_entropy / total_batches,
            "approx_kl": total_approx_kl / total_batches,
            "clip_frac": total_clip_frac / total_batches,
            "grad_norm": gn,
            "rollout_size": float(N),
            "rollout_capacity": float(self.rollout_steps),
            "buffer_utilization": float(N) / float(self.rollout_steps),
            "return_mean": return_mean,
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
            "explained_variance": explained_variance,
        }
        if total_battle_batches > 0:
            out_metrics["battle_pred_loss"] = total_battle_loss / total_battle_batches
            out_metrics["battle_pred_mae"] = total_battle_mae / total_battle_batches
            out_metrics["battle_pred_sign_acc"] = total_battle_sign_acc / total_battle_batches
            if total_battle_corr_batches > 0:
                out_metrics["battle_pred_corr"] = total_battle_corr / total_battle_corr_batches
        if total_placement_batches > 0:
            # value_loss above is the placement CE; acc is its readable companion
            # (uniform baseline = 0.125).
            out_metrics["placement_acc"] = total_placement_acc / total_placement_batches
        return out_metrics

    # ------------------------------------------------------------------
    # Recurrent (v4) PPO update with full BPTT per (episode, seat) sequence
    # ------------------------------------------------------------------

    def _ppo_struct_update_recurrent(self) -> Dict[str, float]:
        """PPO update for recurrent (v4+) models: chunked-sequence BPTT.

        Layout:
          * Buffer steps are grouped by ``(episode_id, seat_id)`` preserving
            buffer order — each group is one BPTT sequence (h_0 = zeros).
          * Each minibatch holds up to ``self._seqs_per_minibatch`` sequences,
            padded to the max length within the minibatch.
          * Forward is a Python for-loop over timesteps: encode_state with the
            current h, compute logits/value/entropy, and advance h via
            ``step_round_hidden`` on COMPLETE_TURN steps. Loss is summed over
            valid (non-pad) positions across all timesteps in the minibatch.
        """
        buf = self.rollout_buffer
        device = self.device

        obs_arr = _stack_and_release(buf.obs)
        rewards_arr = np.array(buf.rewards, dtype=np.float32)
        dones_arr = np.array(buf.dones, dtype=np.bool_)
        values_arr = np.array(buf.values, dtype=np.float32)
        log_probs_old_arr = np.array(buf.log_probs, dtype=np.float32)
        actions_arr = np.array(buf.action_indices, dtype=np.int64)
        complete_arr = np.array(buf.complete_turn, dtype=np.bool_)
        occupied_arr = np.stack(buf.occupied_masks, axis=0)
        picks_arr = np.stack(buf.order_picks, axis=0)
        episode_ids = np.array(buf.episode_ids, dtype=np.int64)
        seat_ids_np = seat_ids_array(buf.seat_ids, len(obs_arr))

        N = obs_arr.shape[0]

        # ---- GAE (same logic as flat) ----
        # Bootstrap value at the end of the rollout. Recurrent caveat: on the
        # host (PPO update side) we don't have the worker's per-seat ``h``
        # post-collection — best approximation is the stored ``h_prev`` of the
        # last buffer step, which is the h used to compute V at that step. If
        # the step was COMPLETE_TURN, the true bootstrap h would be one GRU
        # step ahead; we accept that bias rather than re-running the GRU here.
        if bool(dones_arr[-1]):
            last_bootstrap = 0.0
        else:
            if buf.last_next_obs is None:
                raise RuntimeError("rollout buffer has no last_next_obs for bootstrap")
            next_obs_last_t = torch.as_tensor(
                buf.last_next_obs[None], dtype=torch.float32, device=device
            )
            if buf.h_prev:
                h_last_t = torch.as_tensor(
                    buf.h_prev[-1], dtype=torch.float32, device=device
                ).unsqueeze(0)
            else:
                h_last_t = self.policy_net.zero_hidden(1, device=device)
            with torch.no_grad():
                _, c = self.policy_net.encode_state(next_obs_last_t, h_prev=h_last_t)
                last_bootstrap = float(self.policy_net.critic(c["trunk"]).reshape(()).item())

        adv_np, ret_np = compute_gae_advantages(
            rewards_arr,
            values_arr,
            dones_arr,
            seat_ids_np,
            discount_factor=self.discount_factor,
            gae_lambda=self.gae_lambda,
            last_next_value=last_bootstrap,
        )
        adv_t = torch.from_numpy(adv_np).to(device)
        ret_t = torch.from_numpy(ret_np).to(device)
        adv_t_norm = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
        return_mean = float(ret_t.mean().item())
        adv_mean = float(adv_t_norm.mean().item())
        adv_std = float(adv_t_norm.std(unbiased=False).item())
        ret_var = float(np.var(ret_np))
        explained_variance = (
            1.0 - float(np.var(ret_np - values_arr)) / ret_var if ret_var > 1e-12 else 0.0
        )

        # ---- Pre-tokenize legal_lists once ----
        Lmax_global = max((len(row) for row in buf.legal_lists), default=0)
        if Lmax_global == 0:
            raise ValueError(
                "structured PPO update (recurrent): empty legal_lists in rollout buffer"
            )
        (
            t_np,
            r_np,
            src_k_np,
            src_s_np,
            tgt_k_np,
            tgt_s_np,
            m_np,
        ) = _build_action_tokens(buf.legal_lists, Lmax_global)

        # ---- Group indices into sequences by (episode_id, seat_id) ----
        seq_map: Dict[Tuple[int, int], List[int]] = {}
        for i in range(N):
            key = (int(episode_ids[i]), int(seat_ids_np[i]))
            seq_map.setdefault(key, []).append(i)
        sequences: List[np.ndarray] = [np.asarray(idxs, dtype=np.int64) for idxs in seq_map.values()]
        n_seqs = len(sequences)
        if n_seqs == 0:
            return {}

        # Minibatch size in *sequences*. Aim for total padded step count near
        # ``self.minibatch_size`` so optimizer steps stay comparable to the
        # flat path. ``mean_seq_len`` is the average sequence length.
        mean_seq_len = max(1.0, float(N) / float(n_seqs))
        target_seqs = max(1, int(self.minibatch_size / mean_seq_len))
        seqs_per_mb = max(1, min(target_seqs, n_seqs))

        # Bulk tensors (kept on device, indexed by row id later).
        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
        values_tensor = torch.as_tensor(values_arr, dtype=torch.float32, device=device)
        log_probs_old_tensor = torch.as_tensor(log_probs_old_arr, dtype=torch.float32, device=device)
        actions_tensor = torch.as_tensor(actions_arr, dtype=torch.long, device=device)
        complete_tensor = torch.as_tensor(complete_arr, dtype=torch.bool, device=device)
        occupied_tensor = torch.as_tensor(occupied_arr, dtype=torch.bool, device=device)
        picks_tensor = torch.as_tensor(picks_arr, dtype=torch.long, device=device)
        type_ids_all = torch.from_numpy(t_np).to(device, non_blocking=True)
        role_ids_all = torch.from_numpy(r_np).to(device, non_blocking=True)
        src_region_kinds_all = torch.from_numpy(src_k_np).to(device, non_blocking=True)
        src_region_slots_all = torch.from_numpy(src_s_np).to(device, non_blocking=True)
        tgt_region_kinds_all = torch.from_numpy(tgt_k_np).to(device, non_blocking=True)
        tgt_region_slots_all = torch.from_numpy(tgt_s_np).to(device, non_blocking=True)
        mask_all = torch.from_numpy(m_np).to(device, non_blocking=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_batches = 0
        grad_norm: Union[torch.Tensor, float] = 0.0

        rng = np.random.default_rng()
        seq_order = np.arange(n_seqs, dtype=np.int64)
        net = self.policy_net

        for ep_idx in range(self.ppo_epochs):
            rng.shuffle(seq_order)
            for start in range(0, n_seqs, seqs_per_mb):
                mb_seq_ids = seq_order[start : start + seqs_per_mb]
                if mb_seq_ids.size == 0:
                    continue
                mb_seqs = [sequences[s] for s in mb_seq_ids]
                T_max = int(max(len(s) for s in mb_seqs))
                B = len(mb_seqs)
                if T_max == 0:
                    continue

                # Lay rows out as (T_max, B). Reshape to (T_max*B, ...) for the
                # one-shot heavy encode, then per-timestep slices are
                # ``arr[t*B:(t+1)*B]``.
                idx_grid = np.zeros((T_max, B), dtype=np.int64)
                valid_grid = np.zeros((T_max, B), dtype=np.bool_)
                for r, seq in enumerate(mb_seqs):
                    L = len(seq)
                    idx_grid[:L, r] = seq
                    if L > 0:
                        idx_grid[L:, r] = seq[-1]  # pad with last (masked anyway)
                    valid_grid[:L, r] = True

                idx_flat = torch.from_numpy(idx_grid.reshape(-1)).to(device)
                valid_flat = torch.from_numpy(valid_grid.reshape(-1)).to(device)
                BT = T_max * B

                # Gather rollout tensors once for the whole (B*T) batch.
                obs_BT = obs_tensor[idx_flat]
                adv_BT = adv_t_norm[idx_flat]
                ret_BT = ret_t[idx_flat]
                lp_old_BT = log_probs_old_tensor[idx_flat]
                act_BT = actions_tensor[idx_flat]
                complete_BT = complete_tensor[idx_flat]
                occ_BT = occupied_tensor[idx_flat]
                picks_BT = picks_tensor[idx_flat]
                v_old_BT = values_tensor[idx_flat]
                type_ids_BT = type_ids_all[idx_flat]
                role_ids_BT = role_ids_all[idx_flat]
                src_k_BT = src_region_kinds_all[idx_flat]
                src_s_BT = src_region_slots_all[idx_flat]
                tgt_k_BT = tgt_region_kinds_all[idx_flat]
                tgt_s_BT = tgt_region_slots_all[idx_flat]
                mask_BT = mask_all[idx_flat]

                # --- HEAVY: per-slot conv + entity self-attention on (B*T). ---
                parts = net.encode_entities(obs_BT)

                # --- LIGHT: sequential h chain over T_max, using already-encoded
                # cls_out / g / lb / phase / pending_header slices.
                h = net.zero_hidden(B, device=device)
                state_emb_list = []
                summary_n_list = []
                for t_step in range(T_max):
                    sl = slice(t_step * B, (t_step + 1) * B)
                    cls_t = parts["cls_out"][sl]
                    g_t = parts["g"][sl]
                    lb_t = parts["lb"][sl]
                    phase_t = parts["phase"][sl]
                    ph_t = parts["pending_header"][sl]
                    state_emb_t, summary_n_t = net.state_summary_and_emb(
                        cls_t, g_t, lb_t, phase_t, ph_t, h
                    )
                    state_emb_list.append(state_emb_t)
                    summary_n_list.append(summary_n_t)
                    complete_t = complete_BT[sl]
                    valid_t = valid_flat[sl]
                    h_new = net.step_round_hidden(state_emb_t, h)
                    advance = (complete_t & valid_t).unsqueeze(-1).to(h.dtype)
                    h = h_new * advance + h * (1.0 - advance)

                # Reassemble (B*T, ...) view. ``cat(dim=0)`` of length-T list of
                # (B, D) tensors gives (T*B, D) — same layout as idx_flat.
                state_emb_BT = torch.cat(state_emb_list, dim=0)
                summary_n_BT = torch.cat(summary_n_list, dim=0)

                # --- VECTORIZED ACTION SCORING on (B*T). ---
                cache_BT = {
                    "E_own": parts["E_own"],
                    "E_shop": parts["E_shop"],
                    "E_hand": parts["E_hand"],
                    "E_enemy": parts["E_enemy"],
                    "E_pending": parts["E_pending"],
                    "g_full": parts["g_full"],
                    "trunk": summary_n_BT,
                    "h_prev": None,
                }
                logits = net._logits_from_state_and_tokens(
                    state_emb_BT,
                    cache_BT,
                    type_ids_BT,
                    role_ids_BT,
                    src_k_BT,
                    src_s_BT,
                    tgt_k_BT,
                    tgt_s_BT,
                    mask_BT,
                )
                values_new_BT = net.critic(summary_n_BT).squeeze(-1)

                log_sm = F.log_softmax(logits, dim=-1)
                batch_ar = torch.arange(BT, device=device)
                lp_main = log_sm[batch_ar, act_BT]
                log_sm_safe = log_sm.masked_fill(~mask_BT, 0.0)
                p_safe = log_sm.exp().masked_fill(~mask_BT, 0.0)
                ent_row = -(p_safe * log_sm_safe).sum(dim=-1)

                lp_ord_all = net.order_logprob_given_sequence(
                    state_emb_BT,
                    parts["E_own"],
                    parts["g_full"],
                    occ_BT,
                    picks_BT,
                )
                log_probs_BT = lp_main + complete_BT.to(lp_main.dtype) * lp_ord_all

                ratio = torch.exp(log_probs_BT - lp_old_BT)
                with torch.no_grad():
                    clipped = (ratio < 1.0 - self.ppo_clip_eps) | (ratio > 1.0 + self.ppo_clip_eps)
                surr1 = ratio * adv_BT
                surr2 = torch.clamp(
                    ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps
                ) * adv_BT
                pol_row = -torch.min(surr1, surr2)

                if self.clip_value_loss:
                    eps_v = (
                        float(self.value_clip_eps)
                        if self.value_clip_eps is not None
                        else float(self.ppo_clip_eps)
                    )
                    v_err_u = values_new_BT - ret_BT
                    v_clipped = v_old_BT + torch.clamp(
                        values_new_BT - v_old_BT, -eps_v, eps_v
                    )
                    v_err_c = v_clipped - ret_BT
                    val_row = 0.5 * torch.maximum(v_err_u.pow(2), v_err_c.pow(2))
                else:
                    val_row = 0.5 * (values_new_BT - ret_BT).pow(2)

                kl_row = 0.5 * (lp_old_BT - log_probs_BT).pow(2)
                valid_f = valid_flat.to(pol_row.dtype)

                policy_loss_sum = (pol_row * valid_f).sum()
                value_loss_sum = (val_row * valid_f).sum()
                entropy_sum = (ent_row * valid_f).sum()
                kl_sum = (kl_row * valid_f).sum()
                clip_count = (clipped.to(pol_row.dtype) * valid_f).sum()
                valid_count = valid_f.sum()

                if valid_count.item() <= 0:
                    continue

                inv_v = 1.0 / valid_count
                policy_loss = policy_loss_sum * inv_v
                value_loss = value_loss_sum * inv_v
                entropy = entropy_sum * inv_v
                approx_kl = kl_sum * inv_v
                clip_frac = clip_count * inv_v

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_approx_kl += float(approx_kl.item())
                total_clip_frac += float(clip_frac.item())
                total_batches += 1

        if total_batches == 0:
            return {}

        gn = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        return {
            "loss": (
                total_policy_loss
                + self.value_coef * total_value_loss
                - self.entropy_coef * total_entropy
            )
            / total_batches,
            "policy_loss": total_policy_loss / total_batches,
            "value_loss": total_value_loss / total_batches,
            "entropy": total_entropy / total_batches,
            "approx_kl": total_approx_kl / total_batches,
            "clip_frac": total_clip_frac / total_batches,
            "grad_norm": gn,
            "rollout_size": float(N),
            "rollout_capacity": float(self.rollout_steps),
            "buffer_utilization": float(N) / float(self.rollout_steps),
            "return_mean": return_mean,
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
            "explained_variance": explained_variance,
            "bptt_sequences": float(n_seqs),
            "bptt_seqs_per_mb": float(seqs_per_mb),
            "bptt_mean_seq_len": float(mean_seq_len),
        }

    def train(self) -> None:
        self.training = True
        self.policy_net.train()

    def eval(self) -> None:
        self.training = False
        self.policy_net.eval()

    def save(self, path: str, save_epsilon: bool = True) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint: Dict[str, Any] = {
            "agent_kind": "ppo_minibg_structured",
            "policy_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "observation_shape": self.observation_shape,
            "observation_type": self.observation_type,
            "num_actions": self.num_actions,
            "model_config": self.model_config,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "gae_lambda": self.gae_lambda,
            "ppo_clip_eps": self.ppo_clip_eps,
            "clip_value_loss": self.clip_value_loss,
            "value_clip_eps": self.value_clip_eps,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "rollout_steps": self.rollout_steps,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_size": self.minibatch_size,
            "ppo_network_type": self._ppo_network_type,
            "ppo_network_kwargs": dict(self._ppo_network_kwargs),
        }
        if self.patch_build is not None:
            checkpoint["patch_build"] = int(self.patch_build)
        if save_epsilon:
            checkpoint["epsilon"] = self.epsilon
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, *, device: Optional[str] = None, **overrides: Any) -> "MiniBGPPOStructuredAgent":
        # Default CPU for eval/benchmarks; pass device="cuda" for GPU inference.
        eff_device = device if device is not None else "cpu"
        checkpoint = torch.load(path, map_location=eff_device)
        if checkpoint.get("agent_kind") != "ppo_minibg_structured":
            raise ValueError(f"checkpoint is not MiniBG structured PPO: {checkpoint.get('agent_kind')!r}")

        from src.training.patch_config import assert_checkpoint_patch_build

        expected_build = overrides.get("patch_build")
        assert_checkpoint_patch_build(checkpoint, expected_build)

        observation_shape = tuple(checkpoint["observation_shape"])
        observation_type = checkpoint.get("observation_type", "vector")
        num_actions = int(checkpoint["num_actions"])
        ppo_network_type = checkpoint.get("ppo_network_type", PPO_NETWORK_MINIBG_STRUCTURED)
        ppo_network_kwargs = dict(checkpoint.get("ppo_network_kwargs") or {})

        policy_net = restore_ppo_actor_critic(
            ppo_network_type,
            observation_shape,
            num_actions,
            ppo_network_kwargs,
        )
        if not isinstance(policy_net, StructuredActorCriticProtocol):
            raise TypeError(
                f"checkpoint network type {type(policy_net).__name__} is not "
                "StructuredActorCriticProtocol-compatible"
            )
        policy_net.load_state_dict(checkpoint["policy_state_dict"])

        base_kw: Dict[str, Any] = {
            "observation_shape": observation_shape,
            "observation_type": observation_type,
            "num_actions": num_actions,
            "network": policy_net,
            "ppo_network_type": ppo_network_type,
            "ppo_network_kwargs": ppo_network_kwargs,
            "learning_rate": checkpoint.get("learning_rate", 3e-4),
            "discount_factor": checkpoint.get("discount_factor", 0.99),
            "gae_lambda": checkpoint.get("gae_lambda", 0.95),
            "ppo_clip_eps": checkpoint.get("ppo_clip_eps", 0.2),
            "clip_value_loss": checkpoint.get("clip_value_loss", True),
            "value_clip_eps": checkpoint.get("value_clip_eps"),
            "entropy_coef": checkpoint.get("entropy_coef", 0.01),
            "value_coef": checkpoint.get("value_coef", 0.5),
            "max_grad_norm": checkpoint.get("max_grad_norm", 1.0),
            "rollout_steps": checkpoint.get("rollout_steps", 1024),
            "ppo_epochs": checkpoint.get("ppo_epochs", 4),
            "minibatch_size": checkpoint.get("minibatch_size", 256),
            "device": eff_device,
            "model_config": checkpoint.get("model_config"),
            "patch_build": checkpoint.get("patch_build", expected_build),
        }
        allowed_ov = frozenset(
            {
                "learning_rate",
                "discount_factor",
                "gae_lambda",
                "ppo_clip_eps",
                "clip_value_loss",
                "value_clip_eps",
                "entropy_coef",
                "value_coef",
                "max_grad_norm",
                "rollout_steps",
                "ppo_epochs",
                "minibatch_size",
                "device",
                "seed",
                "compute_detailed_metrics",
                "patch_build",
            }
        )
        for k, v in overrides.items():
            if k in allowed_ov:
                base_kw[k] = v

        agent = cls(**base_kw)
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.step_count = checkpoint.get("step_count", 0)
        agent.epsilon = checkpoint.get("epsilon", 0.0)
        return agent


__all__ = [
    "INFO_STRUCT_LEGAL",
    "INFO_STRUCT_NEXT_LEGAL",
    "MiniBGPPOStructuredAgent",
    "StructuredMiniBGRolloutBuffer",
]

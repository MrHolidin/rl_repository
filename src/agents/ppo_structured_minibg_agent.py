"""PPO agent for MiniBG with structured legal actions + composite log-prob (main + order head)."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

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
        self.next_obs: List[np.ndarray] = []
        self.next_legal_lists: List[List[StructAction]] = []
        self.seat_ids: List[int] = []

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
        self.next_obs.append(np.asarray(next_obs, dtype=np.float32))
        self.next_legal_lists.append(list(next_legal_list))
        self.seat_ids.append(int(seat_id))

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
        self.next_obs.clear()
        self.next_legal_lists.clear()
        self.seat_ids.clear()


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

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.rollout_buffer = StructuredMiniBGRolloutBuffer()
        self._cache: Optional[Dict[str, Any]] = None

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

        with torch.no_grad():
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
            self._cache = {
                "obs": np.asarray(obs, dtype=np.float32),
                "legal_list": list(legal_list),
                "action_idx": idx,
                "value": float(value.squeeze(0).item()),
                "log_prob": float(log_total.item()),
                "complete_turn": chosen.type
                in (
                    StructActionType.COMPLETE_TURN,
                    StructActionType.COMPLETE_TURN_FREEZE_SHOP,
                ),
                "occupied_mask": occupied_np.copy(),
                "order_picks": picks_np.copy(),
            }
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
        )
        self._cache = None
        self.step_count += 1
        return {}

    def close_segment(self, seat: int, terminal_reward: float) -> bool:
        """Mark the last rollout step for ``seat`` as segment-terminal with ``terminal_reward``."""
        if not self.training:
            return False
        return close_rollout_segment(self.rollout_buffer, seat, terminal_reward)

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
        return self.policy_net.critic(enc_cache["trunk"]).reshape(-1)

    def _ppo_struct_update(self) -> Dict[str, float]:
        buf = self.rollout_buffer
        device = self.device

        obs_arr = np.stack(buf.obs, axis=0)
        next_obs_arr = np.stack(buf.next_obs, axis=0)
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
        next_obs_last = torch.as_tensor(next_obs_arr[-1:], dtype=torch.float32, device=device)
        values_tensor = torch.as_tensor(values_arr, dtype=torch.float32, device=device)
        log_probs_old_tensor = torch.as_tensor(log_probs_old_arr, dtype=torch.float32, device=device)
        actions_tensor = torch.as_tensor(actions_arr, dtype=torch.long, device=device)
        complete_tensor = torch.as_tensor(complete_arr, dtype=torch.bool, device=device)
        occupied_tensor = torch.as_tensor(occupied_arr, dtype=torch.bool, device=device)
        picks_tensor = torch.as_tensor(picks_arr, dtype=torch.long, device=device)

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

        # Pre-tokenize the entire rollout buffer's legal lists. One H2D copy per update
        # replaces O(N * Lmax * mini_batches * epochs) per-action kernel launches that were
        # the dominant cost of `update()`.
        Lmax_global = max((len(row) for row in buf.legal_lists), default=0)
        if Lmax_global == 0:
            raise ValueError(
                "structured PPO update: empty legal_lists in rollout buffer (should be unreachable)"
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
        total_batches = 0
        total_clip_frac = 0.0
        grad_norm: torch.Tensor | float = 0.0

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
                if self.clip_value_loss:
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

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mb

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mb.item()
                total_approx_kl += approx_kl.item()
                total_batches += 1
                total_clip_frac += float(clip_frac_t.item())

        if total_batches == 0:
            return {}

        gn = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        return {
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

"""PPO agent implementation."""

import os
import random
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base_agent import BaseAgent
from ..features.action_space import ActionSpace, DiscreteActionSpace
from ..features.observation_builder import ObservationType
from ..models.author_critic_network import ActorCriticCNN
from ..models.ppo_policy_factory import (
    PPO_NETWORK_ACTOR_CRITIC_CNN,
    default_ppo_network_kwargs,
    ppo_network_type_for_save,
    restore_ppo_actor_critic,
)


class RolloutBuffer:
    """On-policy буфер для PPO (хранит только шаги агента)."""

    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.legal_masks: List[Optional[np.ndarray]] = []
        self.next_obs: List[np.ndarray] = []
        self.next_legal_masks: List[np.ndarray] = []
        self.seat_ids: List[int] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        legal_mask: Optional[np.ndarray],
        next_obs: np.ndarray,
        next_legal_mask: np.ndarray,
        seat_id: int = -1,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        if legal_mask is None:
            self.legal_masks.append(None)
        else:
            self.legal_masks.append(np.asarray(legal_mask, dtype=bool))
        self.next_obs.append(np.asarray(next_obs, copy=True, dtype=np.float32))
        self.next_legal_masks.append(np.asarray(next_legal_mask, dtype=bool))
        self.seat_ids.append(int(seat_id))

    def __len__(self) -> int:
        return len(self.obs)

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.legal_masks.clear()
        self.next_obs.clear()
        self.next_legal_masks.clear()
        self.seat_ids.clear()


def compute_gae_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    seat_ids: np.ndarray,
    *,
    discount_factor: float,
    gae_lambda: float,
    last_next_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """GAE with zero bootstrap across ``seat_id`` boundaries (independent learner segments)."""
    n = int(len(rewards))
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            if bool(dones[t]):
                next_value = 0.0
                cont = 0.0
            else:
                next_value = float(last_next_value)
                cont = 1.0
        elif bool(dones[t]) or int(seat_ids[t]) != int(seat_ids[t + 1]):
            next_value = 0.0
            cont = 0.0
        else:
            next_value = float(values[t + 1])
            cont = 1.0
        delta = rewards[t] + discount_factor * next_value * cont - values[t]
        gae = delta + discount_factor * gae_lambda * cont * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


class PPOAgent(BaseAgent):
    """
    PPO: board CNN (Connect Four / Othello-style), or injected actor-critic (e.g. MiniBG slot net).

    Интерфейс максимально совместим с DQNAgent:
    - act(...)
    - observe(...)
    - update(...)
    - train() / eval()
    - save() / load()
    """

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        observation_type: ObservationType,
        num_actions: int,
        network: Optional[nn.Module] = None,
        ppo_network_type: str = PPO_NETWORK_ACTOR_CRITIC_CNN,
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
        action_space: Optional[ActionSpace] = None,
        compute_detailed_metrics: bool = True,
    ):
        self.observation_shape = observation_shape
        self.observation_type = observation_type
        self.num_actions = num_actions

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
        self.action_space = action_space or DiscreteActionSpace(num_actions)
        self.compute_detailed_metrics = compute_detailed_metrics

        # Флаг тренировки — Trainer использует его
        self.training = True
        self.step_count = 0

        # Чтобы не ломать существующие коллбэки (epsilon decay / логгеры)
        self.epsilon = 0.0
        self.epsilon_decay = 1.0
        self.epsilon_min = 0.0

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if network is None:
            if len(observation_shape) != 3:
                raise ValueError(
                    "PPOAgent: pass `network=...` for non-(C,H,W) observations, "
                    "or build via make_agent('ppo', network_type='minibg_slot', ...)."
                )
            in_channels, rows, cols = observation_shape
            self.policy_net = ActorCriticCNN(
                rows=int(rows),
                cols=int(cols),
                in_channels=int(in_channels),
                num_actions=num_actions,
            ).to(self.device)
            self._ppo_network_type = ppo_network_type_for_save(PPO_NETWORK_ACTOR_CRITIC_CNN)
            self._ppo_network_kwargs = {
                "rows": int(rows),
                "cols": int(cols),
                "in_channels": int(in_channels),
            }
        else:
            self.policy_net = network.to(self.device)
            self._ppo_network_type = ppo_network_type_for_save(ppo_network_type)
            if ppo_network_kwargs is not None:
                self._ppo_network_kwargs = dict(ppo_network_kwargs)
            else:
                self._ppo_network_kwargs = dict(
                    default_ppo_network_kwargs(ppo_network_type, self.policy_net)
                )

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # On-policy буфер
        self.rollout_buffer = RolloutBuffer()
        # Кэш информации о последнем действии (чтобы связать act() и observe())
        self._last_action_cache: Optional[Dict[str, Any]] = None

        # Random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    # ---------- API ----------

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """
        Выбор действия.
        deterministic=True → argmax по policy (greedy),
        deterministic=False → сэмплирование из Categorical.
        """
        if legal_mask is None:
            if self.action_space is not None:
                legal_mask = np.ones(self.action_space.size, dtype=bool)
            else:
                legal_mask = np.ones(self.num_actions, dtype=bool)

        legal_mask_arr = np.asarray(legal_mask, dtype=bool)
        if not legal_mask_arr.any():
            # нет легальных действий — fallback
            legal_actions = np.arange(self.num_actions, dtype=int)
            return int(np.random.choice(legal_actions))

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        legal_mask_tensor = torch.as_tensor(legal_mask_arr, dtype=torch.bool, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.policy_net(obs_tensor, legal_mask=legal_mask_tensor)
            # logits уже замаскированы

            # Превращаем в распределение
            probs = torch.softmax(logits, dim=-1)  # (1, A)
            # На случай численной деградации
            probs = probs / probs.sum(dim=-1, keepdim=True)

            if deterministic or not self.training:
                # greedy-режим
                masked_probs = probs.clone()
                masked_probs[~legal_mask_tensor] = 0.0
                action = int(masked_probs.argmax(dim=-1).item())
                log_prob = torch.log(probs[0, action] + 1e-8)
            else:
                dist = Categorical(probs)
                action_t = dist.sample()
                action = int(action_t.item())
                log_prob = dist.log_prob(action_t)

        if self.training:
            self._last_action_cache = {
                "obs": np.array(obs, copy=True),
                "legal_mask": legal_mask_arr.copy(),
                "action": action,
                "value": float(value.squeeze(0).item()),
                "log_prob": float(log_prob.item()),
            }
        else:
            self._last_action_cache = None

        return action

    def observe(
        self, transition, is_augmented: bool = False
    ) -> Dict[str, float]:
        """
        Получаем кредиты за наш ход (через Trainer).
        Сохраняем шаг в rollout_buffer.
        """
        if not self.training:
            return {}

        # On-policy: synthetic augmentations (see Trainer.data_augment_fn) must
        # not pollute rollouts—they do not correspond to act() logits stored in cache.
        if is_augmented:
            return {}

        # Разбираем Transition, как в DQNAgent
        seat_id = -1
        if hasattr(transition, "obs"):
            obs = transition.obs
            action = transition.action
            reward = transition.reward
            next_obs = transition.next_obs
            done = transition.terminated or transition.truncated
            legal_mask = transition.legal_mask
            next_legal_mask = transition.next_legal_mask
            if isinstance(getattr(transition, "info", None), dict):
                acting = transition.info.get("acting_seat")
                if acting is not None:
                    seat_id = int(acting)
        else:
            obs, action, reward, next_obs, done, *_rest = transition
            if len(_rest) >= 2:
                legal_mask, next_legal_mask = _rest[:2]
            else:
                legal_mask = np.ones(self.num_actions, dtype=bool)
                next_legal_mask = np.ones(self.num_actions, dtype=bool)

        # Кэш мог отсутствовать (например, если эпизод завершился ходом соперника,
        # когда агент ещё не делал ходов) — такие переходы просто игнорируем.
        cache = self._last_action_cache
        if cache is None or cache["action"] != int(action):
            self.step_count += 1
            return {}

        buf = self.rollout_buffer
        if buf.seat_ids and seat_id >= 0:
            prev_seat = int(buf.seat_ids[-1])
            if prev_seat != int(seat_id) and not buf.dones[-1]:
                buf.dones[-1] = True
                buf.next_obs[-1] = np.asarray(buf.obs[-1], dtype=np.float32)

        if next_legal_mask is None:
            next_legal_mask = np.ones(self.num_actions, dtype=bool)
        else:
            next_legal_mask = np.asarray(next_legal_mask, dtype=bool)

        self.rollout_buffer.add(
            obs=cache["obs"],
            action=cache["action"],
            reward=reward,
            done=done,
            value=cache["value"],
            log_prob=cache["log_prob"],
            legal_mask=legal_mask,
            next_obs=np.asarray(next_obs, dtype=np.float32),
            next_legal_mask=next_legal_mask,
            seat_id=seat_id,
        )
        self._last_action_cache = None
        self.step_count += 1

        return {}  # все метрики отдаём из update()

    def close_segment(self, seat: int, terminal_reward: float) -> bool:
        """Mark the last rollout step for ``seat`` as segment-terminal with ``terminal_reward``.

        Returns True if a matching rollout step was found and updated.
        """
        if not self.training:
            return False
        buf = self.rollout_buffer
        seat = int(seat)
        for idx in range(len(buf.seat_ids) - 1, -1, -1):
            if buf.seat_ids[idx] == seat:
                buf.rewards[idx] = float(terminal_reward)
                buf.dones[idx] = True
                buf.next_obs[idx] = np.asarray(buf.obs[idx], dtype=np.float32)
                return True
        return False

    def update(self) -> Dict[str, float]:
        """
        Делает PPO-обновление, когда накоплено rollout_steps шагов.
        Вызывается Trainer'ом каждый шаг.
        """
        if not self.training:
            return {}

        if len(self.rollout_buffer) < self.rollout_steps:
            # Для логгера можно вернуть "заполненность" буфера
            return {
                "rollout_size": len(self.rollout_buffer),
                "rollout_capacity": self.rollout_steps,
                "buffer_utilization": len(self.rollout_buffer) / float(self.rollout_steps),
            }

        metrics = self._ppo_update()
        self.rollout_buffer.clear()
        return metrics

    # ---------- PPO training ----------

    def _ppo_update(self) -> Dict[str, float]:
        device = self.device

        obs_arr = np.stack(self.rollout_buffer.obs, axis=0)
        next_obs_arr = np.stack(self.rollout_buffer.next_obs, axis=0)
        next_legal_arr = np.stack(self.rollout_buffer.next_legal_masks, axis=0)
        actions_arr = np.array(self.rollout_buffer.actions, dtype=np.int64)
        rewards_arr = np.array(self.rollout_buffer.rewards, dtype=np.float32)
        dones_arr = np.array(self.rollout_buffer.dones, dtype=np.bool_)
        values_arr = np.array(self.rollout_buffer.values, dtype=np.float32)
        log_probs_arr = np.array(self.rollout_buffer.log_probs, dtype=np.float32)

        # legal_masks могут быть None, если env не отдавал маску.
        if any(m is None for m in self.rollout_buffer.legal_masks):
            legal_masks_arr = None
        else:
            legal_masks_arr = np.stack(self.rollout_buffer.legal_masks, axis=0)

        N = obs_arr.shape[0]

        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
        next_obs_tensor = torch.as_tensor(next_obs_arr, dtype=torch.float32, device=device)
        next_legal_tensor = torch.as_tensor(next_legal_arr, dtype=torch.bool, device=device)
        actions_tensor = torch.as_tensor(actions_arr, dtype=torch.long, device=device)
        rewards_tensor = torch.as_tensor(rewards_arr, dtype=torch.float32, device=device)
        dones_tensor = torch.as_tensor(dones_arr, dtype=torch.bool, device=device)
        values_tensor = torch.as_tensor(values_arr, dtype=torch.float32, device=device)
        log_probs_old_tensor = torch.as_tensor(log_probs_arr, dtype=torch.float32, device=device)

        if legal_masks_arr is not None:
            legal_mask_tensor = torch.as_tensor(legal_masks_arr, dtype=torch.bool, device=device)
        else:
            legal_mask_tensor = torch.ones(N, self.num_actions, dtype=torch.bool, device=device)

        # ---------- GAE (zero bootstrap across seat_id segment boundaries) ----------
        last_next_value = 0.0
        if N > 0 and not bool(dones_arr[-1]):
            with torch.no_grad():
                no = next_obs_tensor[N - 1 : N]
                nlm = next_legal_tensor[N - 1 : N]
                _, v_last = self.policy_net(no, legal_mask=nlm)
            last_next_value = float(v_last.reshape(()).item())

        seat_ids_arr = (
            np.array(self.rollout_buffer.seat_ids, dtype=np.int64)
            if self.rollout_buffer.seat_ids
            else np.full(N, -1, dtype=np.int64)
        )
        adv_np, ret_np = compute_gae_advantages(
            rewards_arr,
            values_arr,
            dones_arr,
            seat_ids_arr,
            discount_factor=self.discount_factor,
            gae_lambda=self.gae_lambda,
            last_next_value=last_next_value,
        )
        advantages = torch.as_tensor(adv_np, dtype=torch.float32, device=device)
        returns = torch.as_tensor(ret_np, dtype=torch.float32, device=device)
        # Нормируем advantages (классический трюк)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        return_mean = returns.mean().item()
        adv_mean = advantages.mean().item()
        adv_std = advantages.std(unbiased=False).item()

        # ---------- PPO epochs ----------
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_batches = 0
        total_clip_frac = 0.0

        indices = np.arange(N)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                if mb_idx.size == 0:
                    continue

                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=device)

                obs_mb = obs_tensor[mb_idx_t]
                actions_mb = actions_tensor[mb_idx_t]
                advantages_mb = advantages[mb_idx_t]
                returns_mb = returns[mb_idx_t]
                log_probs_old_mb = log_probs_old_tensor[mb_idx_t]
                legal_mask_mb = legal_mask_tensor[mb_idx_t]

                logits_mb, values_mb = self.policy_net(obs_mb, legal_mask=legal_mask_mb)
                probs_mb = torch.softmax(logits_mb, dim=-1)
                probs_mb = probs_mb / probs_mb.sum(dim=-1, keepdim=True)

                dist_mb = Categorical(probs_mb)
                log_probs_mb = dist_mb.log_prob(actions_mb)
                entropy_mb = dist_mb.entropy().mean()

                # ratio = π_new / π_old
                ratio = torch.exp(log_probs_mb - log_probs_old_mb)

                # clip fraction: доля сэмплов, где ratio вышел за допустимый диапазон
                with torch.no_grad():
                    clipped = (ratio < 1.0 - self.ppo_clip_eps) | (ratio > 1.0 + self.ppo_clip_eps)
                    batch_clip_frac = clipped.float().mean().item()

                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss: like CleanRL / OpenAI-style — clip delta to old V from buffer,
                # take max(unclipped, clipped) squared error, then 1/2 (same as SB3/CleanRL).
                values_old_mb = values_tensor[mb_idx_t].reshape(-1)
                values_new_mb = values_mb.reshape(-1)
                returns_flat = returns_mb.reshape(-1)
                if self.clip_value_loss:
                    eps_v = (
                        float(self.value_clip_eps)
                        if self.value_clip_eps is not None
                        else float(self.ppo_clip_eps)
                    )
                    v_err_u = values_new_mb - returns_flat
                    v_clipped = values_old_mb + torch.clamp(
                        values_new_mb - values_old_mb, -eps_v, eps_v
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
                total_clip_frac += batch_clip_frac

        if total_batches == 0:
            return {}

        metrics: Dict[str, float] = {
            "loss": (total_policy_loss + self.value_coef * total_value_loss - self.entropy_coef * total_entropy) / total_batches,
            "policy_loss": total_policy_loss / total_batches,
            "value_loss": total_value_loss / total_batches,
            "entropy": total_entropy / total_batches,
            "approx_kl": total_approx_kl / total_batches,
            "clip_frac": total_clip_frac / total_batches,
            "grad_norm": float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "rollout_size": float(N),
            "rollout_capacity": float(self.rollout_steps),
            "buffer_utilization": float(N) / float(self.rollout_steps),
            "return_mean": return_mean,
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
        }

        return metrics

    # ---------- режимы, сохранение ----------

    def train(self) -> None:
        self.training = True
        self.policy_net.train()

    def eval(self) -> None:
        self.training = False
        self.policy_net.eval()

    def save(self, path: str, save_epsilon: bool = True) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
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

        if save_epsilon:
            checkpoint["epsilon"] = self.epsilon

        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device: Optional[str] = None,
        **overrides: Any,
    ) -> "PPOAgent":
        map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=map_location)

        observation_shape = tuple(checkpoint["observation_shape"])
        observation_type = checkpoint.get("observation_type", "board")
        num_actions = checkpoint["num_actions"]
        ppo_network_type = checkpoint.get("ppo_network_type", PPO_NETWORK_ACTOR_CRITIC_CNN)
        ppo_network_kwargs = dict(checkpoint.get("ppo_network_kwargs") or {})

        policy_net = restore_ppo_actor_critic(
            ppo_network_type,
            observation_shape,
            num_actions,
            ppo_network_kwargs,
        )
        policy_net.load_state_dict(checkpoint["policy_state_dict"])

        base_kwargs: Dict[str, Any] = {
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
            "device": device,
            "model_config": checkpoint.get("model_config"),
        }
        base_kwargs.update(overrides)

        agent = cls(**base_kwargs)
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.step_count = checkpoint.get("step_count", 0)
        agent.epsilon = checkpoint.get("epsilon", 0.0)
        return agent

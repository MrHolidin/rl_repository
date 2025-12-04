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
from ..models.author_critic_network import ActorCriticCNN  # путь поправь под свой проект


class RolloutBuffer:
    """On-policy буфер для PPO (хранит только шаги агента)."""

    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.legal_masks: List[np.ndarray] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        legal_mask: Optional[np.ndarray],
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


class PPOAgent(BaseAgent):
    """
    PPO агент для Connect Four.

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
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        rollout_steps: int = 1024,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        model_config: Optional[Dict] = None,  # на будущее, как в DQN
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

        # Сеть: сейчас реализуем только board-наблюдения
        obs_kind = getattr(self.observation_type, "value", self.observation_type)
        if obs_kind != "board":
            raise NotImplementedError("PPOAgent пока реализован только для board-типов наблюдений")

        in_channels, rows, cols = observation_shape
        self.policy_net = ActorCriticCNN(
            rows=rows,
            cols=cols,
            in_channels=in_channels,
            num_actions=num_actions,
        ).to(self.device)

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

    def observe(self, transition) -> Dict[str, float]:
        """
        Получаем кредиты за наш ход (через Trainer).
        Сохраняем шаг в rollout_buffer.
        """
        if not self.training:
            return {}

        # Разбираем Transition, как в DQNAgent
        if hasattr(transition, "obs"):
            obs = transition.obs
            action = transition.action
            reward = transition.reward
            next_obs = transition.next_obs
            done = transition.terminated or transition.truncated
            legal_mask = transition.legal_mask
            next_legal_mask = transition.next_legal_mask
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

        self.rollout_buffer.add(
            obs=cache["obs"],
            action=cache["action"],
            reward=reward,
            done=done,
            value=cache["value"],
            log_prob=cache["log_prob"],
            legal_mask=legal_mask,
        )
        self._last_action_cache = None
        self.step_count += 1

        return {}  # все метрики отдаём из update()

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

        obs_arr = np.stack(self.rollout_buffer.obs, axis=0)  # (N, C, H, W)
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
        actions_tensor = torch.as_tensor(actions_arr, dtype=torch.long, device=device)
        rewards_tensor = torch.as_tensor(rewards_arr, dtype=torch.float32, device=device)
        dones_tensor = torch.as_tensor(dones_arr, dtype=torch.bool, device=device)
        values_tensor = torch.as_tensor(values_arr, dtype=torch.float32, device=device)
        log_probs_old_tensor = torch.as_tensor(log_probs_arr, dtype=torch.float32, device=device)

        if legal_masks_arr is not None:
            legal_mask_tensor = torch.as_tensor(legal_masks_arr, dtype=torch.bool, device=device)
        else:
            legal_mask_tensor = torch.ones(N, self.num_actions, dtype=torch.bool, device=device)

        # ---------- GAE ----------
        advantages = torch.zeros_like(rewards_tensor, device=device)
        gae = 0.0

        for t in reversed(range(N)):
            if t == N - 1:
                next_value = 0.0
                next_non_terminal = 0.0 if dones_tensor[t] else 1.0
            else:
                next_value = values_tensor[t + 1]
                next_non_terminal = 0.0 if dones_tensor[t] else 1.0

            delta = rewards_tensor[t] + self.discount_factor * next_value * next_non_terminal - values_tensor[t]
            gae = delta + self.discount_factor * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values_tensor
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

                value_loss = (values_mb - returns_mb).pow(2).mean()

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
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "rollout_steps": self.rollout_steps,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_size": self.minibatch_size,
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

        base_kwargs: Dict[str, Any] = {
            "observation_shape": observation_shape,
            "observation_type": observation_type,
            "num_actions": num_actions,
            "learning_rate": checkpoint.get("learning_rate", 3e-4),
            "discount_factor": checkpoint.get("discount_factor", 0.99),
            "gae_lambda": checkpoint.get("gae_lambda", 0.95),
            "ppo_clip_eps": checkpoint.get("ppo_clip_eps", 0.2),
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
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.step_count = checkpoint.get("step_count", 0)
        agent.epsilon = checkpoint.get("epsilon", 0.0)
        return agent

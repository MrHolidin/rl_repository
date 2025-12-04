"""DQN agent implementation."""

import copy
import os
import random
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from ..features.action_space import ActionSpace, DiscreteActionSpace
from ..features.observation_builder import ObservationType
from ..models.q_network_factory import build_q_network
from ..utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for Connect Four.
    
    Uses Double DQN algorithm to reduce overestimation bias:
    - Main network selects the best action
    - Target network evaluates the selected action
    """

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        observation_type: ObservationType,
        num_actions: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 32,
        replay_buffer_size: int = 10000,
        target_update_freq: int = 100,
        soft_update: bool = False,
        tau: float = 0.01,
        device: Optional[str] = None,
        seed: int = None,
        network_type: str = "dqn",
        model_config: Optional[Dict] = None,
        action_space: Optional[ActionSpace] = None,
        compute_detailed_metrics: bool = True,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        per_eps: float = 1e-6,
        n_step: int = 1,
    ):
        """
        Initialize DQN agent.
        
        Args:
            observation_shape: Shape of processed observation.
            observation_type: Type of observation ("board" or "vector").
            num_actions: Number of discrete actions.
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma)
            epsilon: Initial epsilon for epsilon-greedy
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            batch_size: Batch size for training
            replay_buffer_size: Size of replay buffer
            target_update_freq: Frequency of target network updates
            soft_update: Whether to use soft target network update
            tau: Soft update coefficient (only used if soft_update=True)
            device: Device to use ('cuda' or 'cpu')
            seed: Random seed
            network_type: Network architecture type ('dqn' or 'dueling_dqn')
            model_config: Optional dictionary with model hyper-parameters
            action_space: Optional ActionSpace instance; defaults to discrete space of size num_actions
            compute_detailed_metrics: If False, only compute loss and grad_norm (faster training)
        """
        self.observation_shape = observation_shape
        self.observation_type = observation_type
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.soft_update = soft_update
        self.tau = tau  # For soft update
        self.training = True
        self.step_count = 0
        self.model_config = copy.deepcopy(model_config) if model_config is not None else None
        self.action_space = action_space or DiscreteActionSpace(num_actions)
        self.compute_detailed_metrics = compute_detailed_metrics
        self.use_per = use_per
        self.per_eps = per_eps
        self.n_step = max(1, n_step)
        # Online n-step buffer storing recent transitions
        self._n_step_buffer: Optional[Deque[Tuple[Any, ...]]] = deque() if self.n_step > 1 else None

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        use_dueling = network_type == "dueling_dqn"
        self.q_network = build_q_network(
            observation_type=self.observation_type,
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            dueling=use_dueling,
            model_config=self.model_config,
        ).to(self.device)
        self.target_network = build_q_network(
            observation_type=self.observation_type,
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            dueling=use_dueling,
            model_config=self.model_config,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.network_type = network_type
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=replay_buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
                eps=per_eps,
            )
        else:
            self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        
        # Random seed
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
        """Select action using epsilon-greedy policy with optional masking."""
        if legal_mask is None:
            if self.action_space is not None:
                legal_mask = np.ones(self.action_space.size, dtype=bool)
            else:
                legal_mask = np.ones(self.num_actions, dtype=bool)

        legal_mask_arr = np.asarray(legal_mask, dtype=bool)
        legal_actions = np.flatnonzero(legal_mask_arr)
        if legal_actions.size == 0:
            raise ValueError("No legal actions available")

        explore = (not deterministic) and self.training and random.random() < self.epsilon
        if explore:
            return int(np.random.choice(legal_actions))

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            legal_mask_tensor = torch.as_tensor(
                legal_mask_arr,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)
            q_values = self.q_network(obs_tensor, legal_mask=legal_mask_tensor).cpu().numpy()[0]
            masked_q_values = q_values.copy()
            masked_q_values[~legal_mask_arr] = -np.inf
            best_action = int(np.argmax(masked_q_values))
            if masked_q_values[best_action] == -np.inf:
                # Handle numerical issue when all entries masked
                best_action = int(np.random.choice(legal_actions))
            return best_action

    def observe(self, transition) -> Dict[str, float]:
        """
        Store transition in replay buffer and train if enough samples.
        
        Args:
            transition: Tuple of (obs, action, reward, next_obs, done, info, legal_mask, next_legal_mask)
            
        Returns:
            Dictionary with training metrics if training occurred, empty dict otherwise
        """
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

        if self.n_step == 1 or self._n_step_buffer is None:
            self.replay_buffer.push(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)
        else:
            self._store_n_step_transition(
                obs,
                action,
                reward,
                next_obs,
                done,
                legal_mask,
                next_legal_mask,
            )
        self.step_count += 1

        return {}

    def _store_n_step_transition(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        legal_mask,
        next_legal_mask,
    ) -> None:
        """Accumulate transitions for the current episode and push n-step returns when it ends."""
        if self._n_step_buffer is None:
            return

        self._n_step_buffer.append(
            (obs, action, reward, next_obs, bool(done), legal_mask, next_legal_mask)
        )

        if len(self._n_step_buffer) >= self.n_step:
            self._push_n_step_transition()

        if done:
            while self._n_step_buffer:
                self._push_n_step_transition()

    def _push_n_step_transition(self) -> None:
        """Assemble a single n-step transition from the rolling buffer and push to replay."""
        if self._n_step_buffer is None or not self._n_step_buffer:
            return

        gamma = self.discount_factor

        obs_t, action_t, _r0, next_obs_0, done_0, legal_t, next_legal_0 = self._n_step_buffer[0]

        R = 0.0
        discount = 1.0
        next_obs_n = next_obs_0
        next_legal_n = next_legal_0
        done_n = done_0

        for idx, (_obs_i, _action_i, r_i, next_obs_i, done_i, _legal_i, next_legal_i) in enumerate(self._n_step_buffer):
            R += discount * r_i
            discount *= gamma

            next_obs_n = next_obs_i
            next_legal_n = next_legal_i
            done_n = done_i

            if done_i or (idx + 1) >= self.n_step:
                break

        self.replay_buffer.push(
            obs_t,
            action_t,
            R,
            next_obs_n,
            done_n,
            legal_t,
            next_legal_n,
        )

        self._n_step_buffer.popleft()

    def update(self) -> Dict[str, float]:
        """Perform a training step using a batch from the replay buffer."""
        # Return buffer status even if not enough samples for training
        if len(self.replay_buffer) < self.batch_size:
            return {
                "buffer_size": len(self.replay_buffer),
                "buffer_capacity": self.replay_buffer.capacity,
                "buffer_utilization": len(self.replay_buffer) / self.replay_buffer.capacity if self.replay_buffer.capacity > 0 else 0.0,
            }
        metrics = self._train()

        if self.target_update_freq > 0 and self.step_count % self.target_update_freq == 0:
            if self.soft_update:
                # Optimized soft update: target = (1-tau)*target + tau*main
                with torch.no_grad():
                    for target_param, main_param in zip(
                        self.target_network.parameters(), self.q_network.parameters()
                    ):
                        target_param.data.mul_(1.0 - self.tau).add_(main_param.data, alpha=self.tau)
            else:
                self.target_network.load_state_dict(self.q_network.state_dict())
            metrics["target_network_updated"] = True

        # Add buffer status to metrics
        metrics["buffer_size"] = len(self.replay_buffer)
        metrics["buffer_capacity"] = self.replay_buffer.capacity
        metrics["buffer_utilization"] = len(self.replay_buffer) / self.replay_buffer.capacity if self.replay_buffer.capacity > 0 else 0.0

        return metrics

    def _train(self) -> dict:
        """
        Train the Q-network on a batch from replay buffer.
        
        Returns:
            Dictionary with training metrics (loss, avg_q, max_q, grad_norm)
        """
        if not self.training:
            return {}
        
        # Sample batch
        if self.use_per:
            (
                obs_batch,
                action_batch,
                reward_batch,
                next_obs_batch,
                done_batch,
                legal_mask_batch,
                next_legal_mask_batch,
                indices,
                is_weights,
            ) = self.replay_buffer.sample(self.batch_size)
        else:
            (
                obs_batch,
                action_batch,
                reward_batch,
                next_obs_batch,
                done_batch,
                legal_mask_batch,
                next_legal_mask_batch,
            ) = self.replay_buffer.sample(self.batch_size)
            indices = None
            is_weights = np.ones(len(action_batch), dtype=np.float32)
        
        # Convert to tensors (directly on device for efficiency)
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(action_batch, dtype=torch.long, device=self.device)
        reward_tensor = torch.as_tensor(reward_batch, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(done_batch, dtype=torch.bool, device=self.device)
        legal_mask_tensor = torch.as_tensor(legal_mask_batch, dtype=torch.bool, device=self.device)
        next_legal_mask_tensor = torch.as_tensor(next_legal_mask_batch, dtype=torch.bool, device=self.device)
        is_weights_tensor = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)
        
        q_values = self.q_network(obs_tensor, legal_mask=legal_mask_tensor)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # Double DQN: main network selects action, target network evaluates
        with torch.no_grad():
            next_q_values_main = self.q_network(next_obs_tensor, legal_mask=next_legal_mask_tensor)
            # Mask invalid actions
            masked_next_q_values = next_q_values_main.masked_fill(~next_legal_mask_tensor, float("-inf"))
            # Handle states with no legal actions (all False) by replacing -inf with 0
            no_legal_actions = ~next_legal_mask_tensor.any(dim=1)
            masked_next_q_values[no_legal_actions] = 0.0

            next_actions_tensor = masked_next_q_values.argmax(dim=1)

            next_q_values_target = self.target_network(next_obs_tensor, legal_mask=next_legal_mask_tensor)
            target_next_q = next_q_values_target.gather(1, next_actions_tensor.unsqueeze(1)).squeeze(1)
            gamma_bootstrap = self.discount_factor ** self.n_step if self.n_step > 1 else self.discount_factor
            target_q_value = reward_tensor + (1 - done_tensor.float()) * gamma_bootstrap * target_next_q
        
        td_errors = target_q_value - q_value
        td_errors_abs = td_errors.detach().abs()

        loss_elements = (td_errors ** 2) * is_weights_tensor
        loss = loss_elements.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm efficiently using PyTorch's built-in function
        # This returns the total norm before clipping
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        if self.use_per and indices is not None:
            new_prios = td_errors_abs.cpu().numpy() + self.per_eps
            self.replay_buffer.update_priorities(indices, new_prios)
        
        # Compute metrics only if requested (reduces overhead)
        metrics = {"loss": loss.item(), "grad_norm": total_grad_norm.item()}
        
        if self.compute_detailed_metrics:
            # Batch all metric computations to reduce CPU-GPU sync overhead
            with torch.no_grad():
                avg_q = q_value.mean()
                max_q = q_value.max()
                min_q = q_value.min()
                avg_target_q = target_q_value.mean()
                mean_td_error = td_errors_abs.mean()
            
            # Single synchronization point for all detailed metrics
            metrics.update({
                "avg_q": avg_q.item(),
                "max_q": max_q.item(),
                "min_q": min_q.item(),
                "avg_target_q": avg_target_q.item(),
                "td_error": mean_td_error.item(),
            })
        
        return metrics

    def train(self) -> None:
        """Set agent to training mode."""
        self.training = True
        self.q_network.train()

    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self.training = False
        self.q_network.eval()

    def save(self, path: str, save_epsilon: bool = True) -> None:
        """
        Save agent to file.
        
        Args:
            path: Path to save file
            save_epsilon: Whether to save epsilon value (default: True)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "network_type": self.network_type,
            "observation_shape": self.observation_shape,
            "observation_type": self.observation_type,
            "num_actions": self.num_actions,
            "model_config": self.model_config,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "soft_update": self.soft_update,
            "tau": self.tau,
            "replay_buffer_capacity": self.replay_buffer.capacity,
            "n_step": self.n_step,
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
    ) -> "DQNAgent":
        """
        Load agent from file and return a new instance.
        """
        map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=map_location)

        observation_shape = tuple(checkpoint["observation_shape"])
        observation_type = checkpoint.get("observation_type", "board")
        num_actions = checkpoint["num_actions"]
        network_type = checkpoint.get("network_type", "dqn")

        base_kwargs: Dict[str, Any] = {
            "observation_shape": observation_shape,
            "observation_type": observation_type,
            "num_actions": num_actions,
            "learning_rate": checkpoint.get("learning_rate", 0.001),
            "discount_factor": checkpoint.get("discount_factor", 0.99),
            "epsilon": checkpoint.get("epsilon", 0.0),
            "epsilon_decay": checkpoint.get("epsilon_decay", 0.995),
            "epsilon_min": checkpoint.get("epsilon_min", 0.01),
            "batch_size": checkpoint.get("batch_size", 32),
            "replay_buffer_size": checkpoint.get("replay_buffer_capacity", 10000),
            "target_update_freq": checkpoint.get("target_update_freq", 100),
            "soft_update": checkpoint.get("soft_update", False),
            "tau": checkpoint.get("tau", 0.01),
            "device": device,
            "network_type": network_type,
            "model_config": checkpoint.get("model_config"),
            "n_step": checkpoint.get("n_step", 1),
        }
        base_kwargs.update(overrides)

        agent = cls(**base_kwargs)
        agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
        agent.step_count = checkpoint.get("step_count", 0)
        return agent
    
    @staticmethod
    def get_network_type_from_checkpoint(path: str) -> str:
        """
        Get network type from checkpoint file.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Network type ('dqn' or 'dueling_dqn')
        """
        checkpoint = torch.load(path, map_location='cpu')
        return checkpoint.get("network_type", "dqn")


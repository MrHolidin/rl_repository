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

from ..base_agent import BaseAgent
from . import action_scores, masked_argmax
from .algo import make_dqn_algo
from .learner import DqnLearner
from .targets import TargetCfg, OptimizeCfg
from ...features.action_space import ActionSpace, DiscreteActionSpace
from ...models.base_dqn_network import BaseDQNNetwork
from ...models.q_network_factory import create_network_from_checkpoint
from ...utils.batch import Batch, to_device
from ...utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent.
    
    Uses Double DQN algorithm to reduce overestimation bias:
    - Main network selects the best action
    - Target network evaluates the selected action
    
    The network must be provided at construction time.
    """

    def __init__(
        self,
        network: nn.Module,
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
        action_space: Optional[ActionSpace] = None,
        compute_detailed_metrics: bool = True,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        per_eps: float = 1e-6,
        n_step: int = 1,
        use_twin_q: bool = False,
        use_distributional: bool = False,
        n_quantiles: int = 32,
        target_q_clip: Optional[float] = None,
        value_reg_weight: float = 0.0,
        metrics_interval: int = 1,
        update_every: int = 1,
        updates_per_step: int = 1,
        grad_clip_norm: float = 1.0,
        use_noisy_nets: bool = False,
    ):
        """
        Initialize DQN agent.
        
        Args:
            network: Q-network to use for action selection and training.
            num_actions: Number of discrete actions.
            learning_rate: Learning rate for optimizer.
            discount_factor: Discount factor (gamma).
            epsilon: Initial epsilon for epsilon-greedy.
            epsilon_decay: Epsilon decay rate.
            epsilon_min: Minimum epsilon value.
            batch_size: Batch size for training.
            replay_buffer_size: Size of replay buffer.
            target_update_freq: Frequency of target network updates.
            soft_update: Whether to use soft target network update.
            tau: Soft update coefficient (only used if soft_update=True).
            device: Device to use ('cuda' or 'cpu').
            seed: Random seed.
            action_space: Optional ActionSpace instance; defaults to discrete space of size num_actions.
            compute_detailed_metrics: If False, only compute loss and grad_norm (faster training).
            use_per: Whether to use prioritized experience replay.
            per_alpha: PER alpha parameter.
            per_beta_start: PER beta start value.
            per_beta_frames: PER beta annealing frames.
            per_eps: Small constant for PER priorities.
            n_step: N-step returns (1 = standard TD).
            use_twin_q: Use Twin Q-networks (Clipped Double Q-Learning from TD3).
            use_distributional: Use Quantile Regression DQN (predict return distribution).
            n_quantiles: Number of quantiles for distributional RL (when use_distributional=True).
        """
        self.num_actions = num_actions
        self.use_twin_q = use_twin_q
        self.use_distributional = use_distributional
        self.n_quantiles = n_quantiles
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.soft_update = soft_update
        self.tau = tau
        self.training = True
        self.step_count = 0
        self.action_space = action_space or DiscreteActionSpace(num_actions)
        self.compute_detailed_metrics = compute_detailed_metrics
        self.metrics_interval = max(1, metrics_interval)
        self.update_every = max(1, update_every)
        self.updates_per_step = max(1, updates_per_step)
        self.grad_clip_norm = float(grad_clip_norm)
        self._train_steps = 0
        self.use_per = use_per
        self.per_eps = float(per_eps)
        self.n_step = max(1, n_step)
        self._n_step_buffer: Optional[Deque[Tuple[Any, ...]]] = deque() if self.n_step > 1 else None
        self.target_q_clip = target_q_clip
        self.value_reg_weight = float(value_reg_weight)
        self.use_noisy_nets = use_noisy_nets

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = network.to(self.device)
        self.target_network = copy.deepcopy(network).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Twin Q-networks (for Clipped Double Q-Learning)
        if self.use_twin_q:
            self.q_network2 = copy.deepcopy(network).to(self.device)
            self.target_network2 = copy.deepcopy(network).to(self.device)
            self.target_network2.load_state_dict(self.q_network2.state_dict())
            self.target_network2.eval()
        else:
            self.q_network2 = None
            self.target_network2 = None
        
        if self.use_twin_q:
            self.optimizer = optim.Adam(
                list(self.q_network.parameters()) + list(self.q_network2.parameters()),
                lr=learning_rate,
            )
        else:
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        algo = make_dqn_algo(
            use_distributional=self.use_distributional,
            use_twin_q=self.use_twin_q,
            n_quantiles=self.n_quantiles,
            device=self.device,
        )
        main_nets = [self.q_network]
        target_nets = [self.target_network]
        if self.use_twin_q:
            main_nets.append(self.q_network2)
            target_nets.append(self.target_network2)
        tgt_cfg = TargetCfg(
            gamma=self.discount_factor,
            n_step=self.n_step,
            target_q_clip=self.target_q_clip,
        )
        opt_cfg = OptimizeCfg(
            grad_clip_norm=self.grad_clip_norm,
            value_reg_weight=self.value_reg_weight,
        )
        self.learner = DqnLearner(
            main_nets,
            target_nets,
            self.optimizer,
            algo,
            tgt_cfg,
            opt_cfg,
        )

        # Replay buffer (store on GPU if available for faster sampling)
        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=replay_buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
                eps=self.per_eps,
                seed=seed,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=replay_buffer_size, seed=seed, device=self.device
            )
        

        # Random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

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

        # Skip epsilon exploration when using noisy nets (noise handles exploration)
        effective_epsilon = 0.0 if self.use_noisy_nets else self.epsilon
        explore = (not deterministic) and self.training and random.random() < effective_epsilon
        if explore:
            return int(np.random.choice(legal_actions))

        # Reset noise before forward pass (only when training and using noisy nets)
        if self.training and self.use_noisy_nets:
            self._reset_all_noise()

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            legal_mask_tensor = torch.as_tensor(
                legal_mask_arr,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)
            out = self.q_network(obs_tensor, legal_mask=legal_mask_tensor)
            scores = action_scores(out)
            best_action = int(masked_argmax(scores, legal_mask_tensor).item())
            return best_action

    def act_batch(
        self,
        obs_batch: np.ndarray,
        legal_mask_batch: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Select actions for a batch of observations. Uses epsilon-greedy when deterministic=False and epsilon > 0."""
        B = legal_mask_batch.shape[0]

        # Reset noise before forward pass (only when training and using noisy nets)
        if self.training and self.use_noisy_nets:
            self._reset_all_noise()

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
            legal_t = torch.as_tensor(legal_mask_batch, dtype=torch.bool, device=self.device)
            out = self.q_network(obs_t, legal_mask=legal_t)
            scores = action_scores(out)
            actions = masked_argmax(scores, legal_t).cpu().numpy().astype(np.int64)

        # Skip epsilon exploration when using noisy nets
        effective_epsilon = 0.0 if self.use_noisy_nets else self.epsilon
        if not deterministic and effective_epsilon > 0 and B > 0:
            legal_np = np.asarray(legal_mask_batch, dtype=bool)
            for i in range(B):
                if random.random() < effective_epsilon:
                    legal_actions = np.flatnonzero(legal_np[i])
                    if legal_actions.size > 0:
                        actions[i] = int(np.random.choice(legal_actions))
        return actions

    def observe(self, transition) -> Dict[str, float]:
        """
        Store transition in replay buffer.
        
        Args:
            transition: Tuple of (obs, action, reward, next_obs, done, ..., legal_mask, next_legal_mask)
            
        Returns:
            Empty dictionary (metrics returned by update()).
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
                legal_mask, next_legal_mask = _rest[-2], _rest[-1]
            else:
                legal_mask = np.ones(self.num_actions, dtype=bool)
                next_legal_mask = np.ones(self.num_actions, dtype=bool)

        if self.n_step == 1 or self._n_step_buffer is None:
            self.replay_buffer.push(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)
        else:
            self._store_n_step_transition(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)
        self.step_count += 1

        return {}

    def _reset_all_noise(self) -> None:
        """Reset noise in all networks that support it (for noisy nets exploration)."""
        if hasattr(self.q_network, "reset_noise"):
            self.q_network.reset_noise()
        if hasattr(self.target_network, "reset_noise"):
            self.target_network.reset_noise()
        if self.q_network2 is not None and hasattr(self.q_network2, "reset_noise"):
            self.q_network2.reset_noise()
        if self.target_network2 is not None and hasattr(self.target_network2, "reset_noise"):
            self.target_network2.reset_noise()

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
        """Accumulate transitions and push n-step returns."""
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
        """Assemble n-step transition and push to replay."""
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

        self.replay_buffer.push(obs_t, action_t, R, next_obs_n, done_n, legal_t, next_legal_n)
        self._n_step_buffer.popleft()

    def update(self) -> Dict[str, float]:
        """Perform a training step using a batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return {
                "buffer_size": len(self.replay_buffer),
                "buffer_capacity": self.replay_buffer.capacity,
                "buffer_utilization": len(self.replay_buffer) / self.replay_buffer.capacity if self.replay_buffer.capacity > 0 else 0.0,
            }

        if self.step_count % self.update_every != 0:
            return {
                "buffer_size": len(self.replay_buffer),
                "buffer_capacity": self.replay_buffer.capacity,
                "buffer_utilization": len(self.replay_buffer) / self.replay_buffer.capacity if self.replay_buffer.capacity > 0 else 0.0,
            }

        if not self.training:
            return {
                "buffer_size": len(self.replay_buffer),
                "buffer_capacity": self.replay_buffer.capacity,
                "buffer_utilization": len(self.replay_buffer) / self.replay_buffer.capacity if self.replay_buffer.capacity > 0 else 0.0,
            }

        metrics = {}
        for _ in range(self.updates_per_step):
            raw = self.replay_buffer.sample(self.batch_size)
            batch = to_device(raw, self.device)
            self._train_steps += 1
            detailed = self.compute_detailed_metrics and (
                self._train_steps % self.metrics_interval == 0
            )
            # Reset noise before training forward pass
            if self.use_noisy_nets:
                self._reset_all_noise()
            metrics, td_errors_abs = self.learner.train_on_batch(batch, detailed_metrics=detailed)

            if self.use_per and batch.indices is not None:
                new_prios = td_errors_abs.cpu().numpy() + self.per_eps
                self.replay_buffer.update_priorities(batch.indices, new_prios)

        if self.target_update_freq > 0 and self.step_count % self.target_update_freq == 0:
            if self.soft_update:
                with torch.no_grad():
                    for target_param, main_param in zip(
                        self.target_network.parameters(), self.q_network.parameters()
                    ):
                        target_param.data.mul_(1.0 - self.tau).add_(main_param.data, alpha=self.tau)
                    # Update second target network if using Twin Q
                    if self.use_twin_q:
                        for target_param, main_param in zip(
                            self.target_network2.parameters(), self.q_network2.parameters()
                        ):
                            target_param.data.mul_(1.0 - self.tau).add_(main_param.data, alpha=self.tau)
            else:
                self.target_network.load_state_dict(self.q_network.state_dict())
                if self.use_twin_q:
                    self.target_network2.load_state_dict(self.q_network2.state_dict())
            metrics["target_network_updated"] = True

        metrics["buffer_size"] = len(self.replay_buffer)
        metrics["buffer_capacity"] = self.replay_buffer.capacity
        metrics["buffer_utilization"] = len(self.replay_buffer) / self.replay_buffer.capacity if self.replay_buffer.capacity > 0 else 0.0

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
        
        The checkpoint contains all information needed to restore the agent,
        including the network architecture.
        
        Args:
            path: Path to save file.
            save_epsilon: Whether to save epsilon value.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Get network serialization info
        if isinstance(self.q_network, BaseDQNNetwork):
            network_class = self.q_network.get_class_name()
            network_kwargs = self.q_network.get_constructor_kwargs()
        else:
            raise TypeError(
                f"Cannot serialize network of type {type(self.q_network).__name__}. "
                "Network must inherit from BaseDQNNetwork."
            )
        
        checkpoint = {
            # Network architecture
            "network_class": network_class,
            "network_kwargs": network_kwargs,
            # Network weights
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # Agent state
            "step_count": self.step_count,
            "num_actions": self.num_actions,
            # Hyperparameters
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
            "update_every": self.update_every,
            "updates_per_step": self.updates_per_step,
            "grad_clip_norm": self.grad_clip_norm,
            "use_distributional": self.use_distributional,
            "n_quantiles": self.n_quantiles,
            "target_q_clip": self.target_q_clip,
            "value_reg_weight": self.value_reg_weight,
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
        Load agent from file.
        
        The network is automatically recreated from checkpoint metadata.
        
        Args:
            path: Path to checkpoint file.
            device: Device to use ('cuda' or 'cpu').
            **overrides: Additional keyword arguments to override saved parameters.
            
        Returns:
            Loaded DQNAgent instance.
        """
        map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=map_location)

        # Recreate network from checkpoint
        network_class = checkpoint["network_class"]
        network_kwargs = checkpoint["network_kwargs"]
        network = create_network_from_checkpoint(network_class, network_kwargs)

        load_optimizer = overrides.pop("load_optimizer", True)

        base_kwargs: Dict[str, Any] = {
            "network": network,
            "num_actions": checkpoint["num_actions"],
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
            "n_step": checkpoint.get("n_step", 1),
            "update_every": checkpoint.get("update_every", 1),
            "updates_per_step": checkpoint.get("updates_per_step", 1),
            "grad_clip_norm": checkpoint.get("grad_clip_norm", 1.0),
            "use_distributional": checkpoint.get("use_distributional", False),
            "n_quantiles": checkpoint.get("n_quantiles")
            or checkpoint.get("network_kwargs", {}).get("n_quantiles", 32),
            "target_q_clip": checkpoint.get("target_q_clip"),
            "value_reg_weight": checkpoint.get("value_reg_weight", 0.0),
        }
        base_kwargs.update(overrides)

        agent = cls(**base_kwargs)
        agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            try:
                agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, KeyError):
                pass  # skip if optimizer structure mismatches (e.g. eval-only load)
        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
        agent.step_count = checkpoint.get("step_count", 0)
        return agent
    
    @staticmethod
    def get_checkpoint_info(path: str) -> Dict[str, Any]:
        """
        Get info from checkpoint without loading weights.
        
        Args:
            path: Path to checkpoint file.
            
        Returns:
            Dictionary with checkpoint metadata.
        """
        checkpoint = torch.load(path, map_location='cpu')
        return {
            "network_class": checkpoint.get("network_class"),
            "network_kwargs": checkpoint.get("network_kwargs"),
            "num_actions": checkpoint.get("num_actions"),
            "step_count": checkpoint.get("step_count", 0),
            "epsilon": checkpoint.get("epsilon"),
            "n_step": checkpoint.get("n_step", 1),
        }

"""DQN agent implementation."""

import copy
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from ..features.action_space import ActionSpace, DiscreteActionSpace
from ..features.observation_builder import ObservationType
from ..models.q_network_factory import build_q_network
from ..utils.replay_buffer import ReplayBuffer


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

        legal_actions = np.flatnonzero(legal_mask)
        if legal_actions.size == 0:
            raise ValueError("No legal actions available")

        explore = (not deterministic) and self.training and random.random() < self.epsilon
        if explore:
            return int(np.random.choice(legal_actions))

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(obs_tensor).cpu().numpy()[0]
            masked_q_values = q_values.copy()
            masked_q_values[~legal_mask.astype(bool)] = -np.inf
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

        self.replay_buffer.push(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)
        self.step_count += 1

        return {}

    def update(self) -> Dict[str, float]:
        """Perform a training step using a batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        metrics = self._train()

        if self.target_update_freq > 0 and self.step_count % self.target_update_freq == 0:
            if self.soft_update:
                for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            else:
                self.target_network.load_state_dict(self.q_network.state_dict())
            metrics["target_network_updated"] = True

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
        (
            obs_batch,
            action_batch,
            reward_batch,
            next_obs_batch,
            done_batch,
            _legal_mask_batch,
            next_legal_mask_batch,
        ) = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        reward_tensor = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs_batch).to(self.device)
        done_tensor = torch.BoolTensor(done_batch).to(self.device)
        next_legal_mask_tensor = torch.BoolTensor(next_legal_mask_batch).to(self.device)
        
        q_values = self.q_network(obs_tensor)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # Double DQN: main network selects action, target network evaluates
        with torch.no_grad():
            next_q_values_main = self.q_network(next_obs_tensor)
            # Mask invalid actions
            masked_next_q_values = next_q_values_main.masked_fill(~next_legal_mask_tensor, float("-inf"))
            # Handle states with no legal actions (all False) by replacing -inf with 0
            no_legal_actions = ~next_legal_mask_tensor.any(dim=1)
            masked_next_q_values[no_legal_actions] = 0.0

            next_actions_tensor = masked_next_q_values.argmax(dim=1)

            next_q_values_target = self.target_network(next_obs_tensor)
            target_next_q = next_q_values_target.gather(1, next_actions_tensor.unsqueeze(1)).squeeze(1)
            target_q_value = reward_tensor + (1 - done_tensor.float()) * self.discount_factor * target_next_q
        
        loss = nn.MSELoss()(q_value, target_q_value)
        
        avg_q = q_value.mean().item()
        max_q = q_value.max().item()
        min_q = q_value.min().item()
        avg_target_q = target_q_value.mean().item()
        td_error = (target_q_value - q_value).abs().mean().item()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        total_grad_norm = 0.0
        for param in self.q_network.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1.0 / 2)
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        total_grad_norm_clipped = 0.0
        for param in self.q_network.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm_clipped += param_norm.item() ** 2
        total_grad_norm_clipped = total_grad_norm_clipped ** (1.0 / 2)
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "avg_q": avg_q,
            "max_q": max_q,
            "min_q": min_q,
            "avg_target_q": avg_target_q,
            "td_error": td_error,
            "grad_norm": total_grad_norm,
            "grad_norm_clipped": total_grad_norm_clipped,
        }

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


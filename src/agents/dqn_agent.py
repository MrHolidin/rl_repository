"""DQN agent implementation."""

import random
import os
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from ..models.dqn_network import DQN
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
        rows: int = 6,
        cols: int = 7,
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
    ):
        """
        Initialize DQN agent.
        
        Args:
            rows: Number of rows in board
            cols: Number of columns in board
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
        """
        self.rows = rows
        self.cols = cols
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
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = DQN(rows=rows, cols=cols, in_channels=3, num_actions=cols).to(self.device)
        self.target_network = DQN(rows=rows, cols=cols, in_channels=3, num_actions=cols).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        
        # Random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            obs: Current observation
            legal_actions: List of legal action indices
            
        Returns:
            Selected action index
        """
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # Epsilon-greedy
        if self.training and random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            # Select best action from Q-network
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values = self.q_network(obs_tensor).cpu().numpy()[0]
                
                # Mask illegal actions
                masked_q_values = q_values.copy()
                masked_q_values[[a for a in range(len(q_values)) if a not in legal_actions]] = -np.inf
                
                best_action = np.argmax(masked_q_values)
                return int(best_action)

    def observe(self, transition: Tuple) -> dict:
        """
        Store transition in replay buffer and train if enough samples.
        
        Args:
            transition: Tuple of (obs, action, reward, next_obs, done, info)
            
        Returns:
            Dictionary with training metrics if training occurred, empty dict otherwise
        """
        obs, action, reward, next_obs, done, info = transition
        
        # Store in replay buffer
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        
        # Train if enough samples
        metrics = {}
        if len(self.replay_buffer) >= self.batch_size:
            metrics = self._train()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            if self.soft_update:
                # Soft update: gradually blend weights
                for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            else:
                # Hard update: copy weights completely
                self.target_network.load_state_dict(self.q_network.state_dict())
            metrics["target_network_updated"] = True
        
        # NOTE: Epsilon decay is now handled per episode in training loop, not per step
        # This prevents epsilon from decaying too quickly
        
        return metrics

    def _get_legal_actions_from_obs(self, obs: np.ndarray) -> List[int]:
        """
        Extract legal actions from observation.
        
        Args:
            obs: Observation array of shape (3, rows, cols)
            
        Returns:
            List of legal action indices (columns that are not full)
        """
        # Legal actions are columns where the top row (row 0) is empty
        # Check if both current player's pieces and opponent's pieces are 0 in row 0
        top_row_current = obs[0, 0, :]  # Current player's pieces in top row
        top_row_opponent = obs[1, 0, :]  # Opponent's pieces in top row
        
        # Column is legal if top cell is empty (both channels are 0)
        legal_actions = [
            col for col in range(self.cols) 
            if top_row_current[col] == 0 and top_row_opponent[col] == 0
        ]
        return legal_actions

    def _train(self) -> dict:
        """
        Train the Q-network on a batch from replay buffer.
        
        Returns:
            Dictionary with training metrics (loss, avg_q, max_q, grad_norm)
        """
        if not self.training:
            return {}
        
        # Sample batch
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        reward_tensor = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs_batch).to(self.device)
        done_tensor = torch.BoolTensor(done_batch).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(obs_tensor)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use main network to select action, target network to evaluate
        with torch.no_grad():
            # Use main network to select best action (with legal action masking)
            next_q_values_main = self.q_network(next_obs_tensor)
            
            # Mask illegal actions for each next_obs in the batch
            # Convert to numpy for easier masking
            next_q_values_main_np = next_q_values_main.cpu().numpy()
            next_actions = []
            
            for i in range(next_obs_tensor.shape[0]):
                # Skip action selection for done states (will be masked by done_tensor anyway)
                if done_batch[i]:
                    next_actions.append(0)  # Dummy action, won't be used
                    continue
                
                next_obs_np = next_obs_tensor[i].cpu().numpy()
                legal_actions = self._get_legal_actions_from_obs(next_obs_np)
                
                if not legal_actions:
                    # If no legal actions (shouldn't happen, but handle gracefully)
                    next_actions.append(0)
                else:
                    # Mask illegal actions
                    masked_q_vals = next_q_values_main_np[i].copy()
                    masked_q_vals[[a for a in range(len(masked_q_vals)) if a not in legal_actions]] = -np.inf
                    next_actions.append(np.argmax(masked_q_vals))
            
            next_actions_tensor = torch.LongTensor(next_actions).to(self.device)
            
            # Use target network to evaluate the selected action
            next_q_values_target = self.target_network(next_obs_tensor)
            next_q_value = next_q_values_target.gather(1, next_actions_tensor.unsqueeze(1)).squeeze(1)
            
            target_q_value = reward_tensor + (1 - done_tensor.float()) * self.discount_factor * next_q_value
        
        # Compute loss
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # Compute metrics before optimization
        avg_q = q_value.mean().item()
        max_q = q_value.max().item()
        min_q = q_value.min().item()
        avg_target_q = target_q_value.mean().item()
        td_error = (target_q_value - q_value).abs().mean().item()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm before clipping
        total_grad_norm = 0.0
        for param in self.q_network.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1.0 / 2)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        # Compute gradient norm after clipping
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
            "rows": self.rows,
            "cols": self.cols,
        }
        
        if save_epsilon:
            checkpoint["epsilon"] = self.epsilon
        
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load agent from file.
        
        Args:
            path: Path to load file from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Default to 0.0 (pure exploitation) if epsilon not in checkpoint
        self.epsilon = checkpoint.get("epsilon", 0.0)
        self.step_count = checkpoint.get("step_count", 0)


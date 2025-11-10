"""Experience replay buffer for DQN."""

from collections import deque
from typing import Tuple, Optional
import numpy as np
import random


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        legal_mask: np.ndarray,
        next_legal_mask: np.ndarray,
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
        """
        self.buffer.append((obs, action, reward, next_obs, done, legal_mask, next_legal_mask))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (obs, actions, rewards, next_obs, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        obs, actions, rewards, next_obs, dones, legal_masks, next_legal_masks = zip(*batch)
        
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones),
            np.array(legal_masks),
            np.array(next_legal_masks),
        )

    def __len__(self) -> int:
        """Return current size of the buffer."""
        return len(self.buffer)


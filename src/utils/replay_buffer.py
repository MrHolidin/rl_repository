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
        
        # Use np.stack for observations (more efficient for arrays of same shape)
        # Use np.array for scalars and 1D arrays
        return (
            np.stack(obs) if obs else np.array(obs),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_obs) if next_obs else np.array(next_obs),
            np.array(dones, dtype=bool),
            np.stack(legal_masks) if legal_masks else np.array(legal_masks),
            np.stack(next_legal_masks) if next_legal_masks else np.array(next_legal_masks),
        )

    def __len__(self) -> int:
        """Return current size of the buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps

        self.pos = 0
        self.size = 0
        self.frame = 0

        self.obs_buf = [None] * capacity
        self.action_buf = np.zeros(capacity, dtype=np.int64)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.next_obs_buf = [None] * capacity
        self.done_buf = np.zeros(capacity, dtype=bool)
        self.legal_mask_buf = [None] * capacity
        self.next_legal_mask_buf = [None] * capacity

        self.priorities = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def _beta_by_frame(self) -> float:
        t = min(1.0, self.frame / max(1, self.beta_frames))
        return self.beta_start + t * (1.0 - self.beta_start)

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
        idx = self.pos

        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.legal_mask_buf[idx] = legal_mask
        self.next_legal_mask_buf[idx] = next_legal_mask

        self.action_buf[idx] = action
        self.reward_buf[idx] = reward
        self.done_buf[idx] = done

        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[idx] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size == 0:
            raise ValueError("PrioritizedReplayBuffer is empty")

        self.frame += 1
        beta = self._beta_by_frame()

        batch_size = min(batch_size, self.size)

        prios = self.priorities[:self.size]
        if prios.max() == 0.0:
            prios = np.ones_like(prios)

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        obs_batch = np.stack([self.obs_buf[i] for i in indices])
        next_obs_batch = np.stack([self.next_obs_buf[i] for i in indices])
        legal_mask_batch = np.stack([self.legal_mask_buf[i] for i in indices])
        next_legal_mask_batch = np.stack([self.next_legal_mask_buf[i] for i in indices])

        action_batch = self.action_buf[indices]
        reward_batch = self.reward_buf[indices]
        done_batch = self.done_buf[indices]

        N = self.size
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            obs_batch,
            action_batch,
            reward_batch,
            next_obs_batch,
            done_batch,
            legal_mask_batch,
            next_legal_mask_batch,
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, new_priorities) -> None:
        new_priorities = np.asarray(new_priorities, dtype=np.float32)
        self.priorities[indices] = np.maximum(new_priorities, self.eps)


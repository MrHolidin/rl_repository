"""Experience replay buffer for DQN."""

from typing import Optional, Tuple, Union
import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    Uses pre-allocated arrays. Supports GPU storage for faster sampling.
    """

    def __init__(
        self,
        capacity: int = 10000,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)
        
        # Device for storage (None = numpy/CPU, "cuda" = GPU tensors)
        if device is None:
            self.device = None
        else:
            self.device = torch.device(device)
        self._use_gpu = self.device is not None and self.device.type == "cuda"
        
        self._obs_buf = None
        self._next_obs_buf = None
        self._action_buf = None
        self._reward_buf = None
        self._done_buf = None
        self._legal_mask_buf = None
        self._next_legal_mask_buf = None

    def _allocate(self, obs: np.ndarray, legal_mask: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        legal_mask = np.asarray(legal_mask, dtype=bool)
        
        if self._use_gpu:
            self._obs_buf = torch.zeros((self.capacity, *obs.shape), dtype=torch.float32, device=self.device)
            self._next_obs_buf = torch.zeros((self.capacity, *obs.shape), dtype=torch.float32, device=self.device)
            self._action_buf = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
            self._reward_buf = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
            self._done_buf = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
            self._legal_mask_buf = torch.zeros((self.capacity, *legal_mask.shape), dtype=torch.bool, device=self.device)
            self._next_legal_mask_buf = torch.zeros((self.capacity, *legal_mask.shape), dtype=torch.bool, device=self.device)
        else:
            self._obs_buf = np.zeros((self.capacity, *obs.shape), dtype=np.float32)
            self._next_obs_buf = np.zeros((self.capacity, *obs.shape), dtype=np.float32)
            self._action_buf = np.zeros(self.capacity, dtype=np.int64)
            self._reward_buf = np.zeros(self.capacity, dtype=np.float32)
            self._done_buf = np.zeros(self.capacity, dtype=bool)
            self._legal_mask_buf = np.zeros((self.capacity, *legal_mask.shape), dtype=bool)
            self._next_legal_mask_buf = np.zeros((self.capacity, *legal_mask.shape), dtype=bool)

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
        if self._obs_buf is None:
            self._allocate(obs, legal_mask)

        idx = self.pos
        
        if self._use_gpu:
            self._obs_buf[idx] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            self._next_obs_buf[idx] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            self._legal_mask_buf[idx] = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device)
            self._next_legal_mask_buf[idx] = torch.as_tensor(next_legal_mask, dtype=torch.bool, device=self.device)
            self._action_buf[idx] = action
            self._reward_buf[idx] = reward
            self._done_buf[idx] = done
        else:
            self._obs_buf[idx] = obs
            self._next_obs_buf[idx] = next_obs
            self._legal_mask_buf[idx] = legal_mask
            self._next_legal_mask_buf[idx] = next_legal_mask
            self._action_buf[idx] = action
            self._reward_buf[idx] = reward
            self._done_buf[idx] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        batch_size = min(batch_size, self.size)
        indices = self._rng.choice(self.size, size=batch_size, replace=False)
        phys = (self.pos - self.size + indices + self.capacity) % self.capacity

        if self._use_gpu:
            phys_t = torch.from_numpy(phys).to(self.device)
            weights = torch.ones(batch_size, dtype=torch.float32, device=self.device)
            return (
                self._obs_buf[phys_t],
                self._action_buf[phys_t],
                self._reward_buf[phys_t],
                self._next_obs_buf[phys_t],
                self._done_buf[phys_t],
                self._legal_mask_buf[phys_t],
                self._next_legal_mask_buf[phys_t],
                None,
                weights,
            )
        else:
            weights = np.ones(batch_size, dtype=np.float32)
            return (
                self._obs_buf[phys],
                self._action_buf[phys],
                self._reward_buf[phys],
                self._next_obs_buf[phys],
                self._done_buf[phys],
                self._legal_mask_buf[phys],
                self._next_legal_mask_buf[phys],
                None,
                weights,
            )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    Uses pre-allocated numpy arrays for efficient sampling.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps: float = 1e-6,
        seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps

        self.pos = 0
        self.size = 0
        self.frame = 0

        self.action_buf = np.zeros(capacity, dtype=np.int64)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=bool)
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self._obs_buf: Optional[np.ndarray] = None
        self._next_obs_buf: Optional[np.ndarray] = None
        self._legal_mask_buf: Optional[np.ndarray] = None
        self._next_legal_mask_buf: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def _beta_by_frame(self) -> float:
        t = min(1.0, self.frame / max(1, self.beta_frames))
        return self.beta_start + t * (1.0 - self.beta_start)

    def _allocate(self, obs: np.ndarray, legal_mask: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        legal_mask = np.asarray(legal_mask, dtype=bool)
        self._obs_buf = np.zeros((self.capacity, *obs.shape), dtype=np.float32)
        self._next_obs_buf = np.zeros((self.capacity, *obs.shape), dtype=np.float32)
        self._legal_mask_buf = np.zeros((self.capacity, *legal_mask.shape), dtype=bool)
        self._next_legal_mask_buf = np.zeros((self.capacity, *legal_mask.shape), dtype=bool)

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
        if self._obs_buf is None:
            self._allocate(obs, legal_mask)

        idx = self.pos
        np.copyto(self._obs_buf[idx], np.asarray(obs, dtype=np.float32))
        np.copyto(self._next_obs_buf[idx], np.asarray(next_obs, dtype=np.float32))
        np.copyto(self._legal_mask_buf[idx], np.asarray(legal_mask, dtype=bool))
        np.copyto(self._next_legal_mask_buf[idx], np.asarray(next_legal_mask, dtype=bool))
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

        prios = self.priorities[: self.size].copy()
        if prios.max() == 0.0:
            prios[:] = 1.0

        probs = np.power(prios, self.alpha)
        probs /= probs.sum()

        logical_indices = self._rng.choice(self.size, size=batch_size, p=probs, replace=False)
        phys = (self.pos - self.size + logical_indices + self.capacity) % self.capacity

        obs_batch = self._obs_buf[phys]
        next_obs_batch = self._next_obs_buf[phys]
        legal_mask_batch = self._legal_mask_buf[phys]
        next_legal_mask_batch = self._next_legal_mask_buf[phys]
        action_batch = self.action_buf[phys]
        reward_batch = self.reward_buf[phys]
        done_batch = self.done_buf[phys]

        N = self.size
        weights = np.power(N * probs[logical_indices], -beta)
        weights /= weights.max()

        return (
            obs_batch,
            action_batch,
            reward_batch,
            next_obs_batch,
            done_batch,
            legal_mask_batch,
            next_legal_mask_batch,
            phys,
            weights.astype(np.float32),
        )

    def update_priorities(self, phys_indices, new_priorities) -> None:
        new_priorities = np.asarray(new_priorities, dtype=np.float32)
        self.priorities[phys_indices] = np.maximum(new_priorities, self.eps)


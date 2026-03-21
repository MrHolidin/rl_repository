"""Replay buffer for AlphaZero training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class AlphaZeroSample:
    """A single training sample: (state, target_policy, target_value)."""

    observation: np.ndarray
    legal_mask: np.ndarray
    target_policy: np.ndarray
    target_value: float


@dataclass
class AlphaZeroBatch:
    """Batch of samples for training."""

    observations: torch.Tensor
    legal_masks: torch.Tensor
    target_policies: torch.Tensor
    target_values: torch.Tensor


class AlphaZeroReplayBuffer:
    """
    Replay buffer for AlphaZero training.

    Stores (observation, legal_mask, target_policy, target_value) tuples.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)

        self._obs_buf: Optional[np.ndarray] = None
        self._legal_mask_buf: Optional[np.ndarray] = None
        self._policy_buf: Optional[np.ndarray] = None
        self._value_buf: Optional[np.ndarray] = None

        self._pos = 0
        self._size = 0

    def _allocate(self, sample: AlphaZeroSample) -> None:
        obs_shape = sample.observation.shape
        num_actions = sample.target_policy.shape[0]

        self._obs_buf = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self._legal_mask_buf = np.zeros((self.capacity, num_actions), dtype=bool)
        self._policy_buf = np.zeros((self.capacity, num_actions), dtype=np.float32)
        self._value_buf = np.zeros(self.capacity, dtype=np.float32)

    def push(self, sample: AlphaZeroSample) -> None:
        if self._obs_buf is None:
            self._allocate(sample)

        idx = self._pos
        self._obs_buf[idx] = sample.observation
        self._legal_mask_buf[idx] = sample.legal_mask
        self._policy_buf[idx] = sample.target_policy
        self._value_buf[idx] = sample.target_value

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_game(self, samples: List[AlphaZeroSample]) -> None:
        for sample in samples:
            self.push(sample)

    def sample(self, batch_size: int, device: torch.device) -> AlphaZeroBatch:
        if self._size == 0:
            raise ValueError("Buffer is empty")

        batch_size = min(batch_size, self._size)
        indices = self._rng.choice(self._size, size=batch_size, replace=False)

        return AlphaZeroBatch(
            observations=torch.as_tensor(self._obs_buf[indices], device=device),
            legal_masks=torch.as_tensor(self._legal_mask_buf[indices], device=device),
            target_policies=torch.as_tensor(self._policy_buf[indices], device=device),
            target_values=torch.as_tensor(
                self._value_buf[indices], device=device
            ).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._pos = 0
        self._size = 0

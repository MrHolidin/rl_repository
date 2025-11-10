"""Core interfaces for turn-based environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import numpy as np


@dataclass(slots=True)
class StepResult:
    """Container for step outcomes in turn-based environments."""

    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

    def __iter__(self) -> Iterator[Any]:
        """Allow unpacking like the classic Gym tuple."""
        yield self.obs
        yield self.reward
        yield self.done
        yield self.info

    @property
    def done(self) -> bool:
        """Return True when the episode ended for any reason."""
        return self.terminated or self.truncated


class TurnBasedEnv(ABC):
    """Abstract base class for synchronous, two-player turn-based games."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment state and return the initial observation."""

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Advance environment by one action from the current player."""

    @property
    @abstractmethod
    def legal_actions_mask(self) -> np.ndarray:
        """Boolean mask of shape (num_actions,) with True for legal moves."""

    @abstractmethod
    def current_player(self) -> int:
        """Index (0 or 1) of the player whose turn it is."""

    @abstractmethod
    def render(self) -> None:
        """Render the environment state."""


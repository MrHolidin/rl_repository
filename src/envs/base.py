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


class SingleAgentEnv(ABC):
    """Single-agent gym-style environment.

    `step` returns the obs / reward / done at the agent's next decision point.
    Used by `Trainer`. Two-player envs are wrapped via
    `src.training.agent_perspective_env.AgentPerspectiveEnv`.
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset and return the first observation at an agent decision point."""

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Advance to the next agent decision point or terminal."""

    @property
    @abstractmethod
    def legal_actions_mask(self) -> np.ndarray:
        """Legal actions mask at the current agent decision point."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """True when the episode has ended."""

    @property
    def agent_token(self) -> int:
        """Side of the agent (+1 / -1) for two-player wrappers; +1 for native single-agent."""
        return 1

    def notify_episode_end(self, info: dict) -> None:
        """Optional hook invoked by `Trainer` after a terminal step before the next reset."""
        return None


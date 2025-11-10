"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


class BaseAgent(ABC):
    """Base class for all agents."""

    @abstractmethod
    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Return an action for the given observation."""

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """Backward-compatible wrapper that routes to :meth:`act`."""
        legal_mask = None
        if legal_actions:
            if hasattr(self, "num_actions"):
                num_actions = getattr(self, "num_actions")
            elif hasattr(self, "action_space") and hasattr(self.action_space, "size"):
                num_actions = getattr(self.action_space, "size")
            else:
                num_actions = max(legal_actions) + 1
            legal_mask = np.zeros(num_actions, dtype=bool)
            legal_mask[legal_actions] = True
        return self.act(obs, legal_mask=legal_mask, deterministic=False)

    def observe(self, transition: Any) -> Dict[str, float]:
        """Observe a transition and optionally return training metrics."""
        return {}

    def update(self) -> Dict[str, float]:
        """Perform a single optimisation step, returning metrics if any."""
        return {}

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist agent state to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs: Any) -> "BaseAgent":
        """Create an agent instance from ``path``."""

    def train(self) -> None:
        """Set agent to training mode."""

    def eval(self) -> None:
        """Set agent to evaluation mode."""

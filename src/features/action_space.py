"""Action space abstractions for turn-based games."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np


class ActionSpace(ABC):
    """Abstract base class describing an agent's action space."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of discrete actions."""

    @abstractmethod
    def legal_actions(self, mask: np.ndarray) -> List[int]:
        """Return a list of legal action indices based on the provided mask."""


class DiscreteActionSpace(ActionSpace):
    """Simple discrete action space."""

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("Action space size must be positive.")
        self._n = n

    @property
    def size(self) -> int:
        return self._n

    def legal_actions(self, mask: np.ndarray) -> List[int]:
        if mask.shape[0] != self._n:
            raise ValueError("Mask shape does not match action space size.")
        return [idx for idx, allowed in enumerate(mask) if allowed]

    def iter_legal(self, mask: np.ndarray) -> Iterable[int]:
        """Iterate over legal action indices."""
        for idx, allowed in enumerate(mask):
            if allowed:
                yield idx


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import numpy as np

S = TypeVar("S")


class StateEncoder(ABC, Generic[S]):
    """
    Превращает RL-наблюдение (obs) и маску легальных ходов
    в внутреннее Game-состояние.
    """

    @abstractmethod
    def from_obs(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
    ) -> S:
        """
        Преобразовать наблюдение (и маску) в состояние типа `S`.
        """


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .turn_based_game import TurnBasedGame

S = TypeVar("S")


class StateEvaluator(ABC, Generic[S]):
    """
    Оценка нетерминального состояния с точки зрения root-игрока.
    """

    @abstractmethod
    def evaluate(
        self,
        game: TurnBasedGame[S],
        state: S,
        root_player: int,
    ) -> float:
        """
        Вернуть эвристическую оценку состояния.
        """


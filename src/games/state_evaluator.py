from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .turn_based_game import TurnBasedGame

S = TypeVar("S")


def terminal_value(
    game: TurnBasedGame[S],
    state: S,
    root_player: int,
) -> float:
    """Return +1 / -1 / 0 for terminal states; 0.0 for non-terminal."""
    if not game.is_terminal(state):
        return 0.0
    winner = game.winner(state)
    if winner is None or winner == 0:
        return 0.0
    if winner == root_player:
        return 1.0
    return -1.0


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


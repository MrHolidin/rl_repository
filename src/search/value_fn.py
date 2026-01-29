"""Abstract state value function for search algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.games.turn_based_game import TurnBasedGame

StateT = TypeVar("StateT")


class StateValueFn(Generic[StateT], ABC):
    """
    State evaluator that returns value for ``game.current_player(state)``.
    """

    @abstractmethod
    def evaluate(self, game: TurnBasedGame[StateT], state: StateT) -> float:
        """
        Higher is better for ``game.current_player(state)``.
        """
        ...

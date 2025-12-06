from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVar

from src.games.turn_based_game import Action, TurnBasedGame

S = TypeVar("S")


class ActionPolicy(ABC, Generic[S]):
    """
    Abstract policy that selects actions using the game rules and state.

    Knows only about:
      - ``TurnBasedGame[S]``
      - a concrete ``S`` game state
      - optionally a list of legal actions
    """

    @abstractmethod
    def select_action(
        self,
        game: TurnBasedGame[S],
        state: S,
        legal_actions: Optional[Sequence[Action]] = None,
    ) -> Action:
        """
        Choose an action for ``state``.

        Args:
            game: rules / transitions implementation.
            state: current state.
            legal_actions: optional cached legal moves (fallbacks to
                ``game.legal_actions`` when ``None``).
        """
        raise NotImplementedError


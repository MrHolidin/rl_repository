"""Negamax search policy with optional alpha-beta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, Sequence, TypeVar

import math
import numpy as np

from src.games.turn_based_game import Action, TurnBasedGame
from .action_policy import ActionPolicy
from .value_fn import StateValueFn

StateT = TypeVar("StateT")


@dataclass
class MinimaxConfig:
    depth: int = 2
    use_alpha_beta: bool = False
    random_tiebreak: bool = True
    value_epsilon: float = 1e-6


class MinimaxPolicy(ActionPolicy[StateT], Generic[StateT]):

    def __init__(
        self,
        value_fn: StateValueFn[StateT],
        config: Optional[MinimaxConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.value_fn = value_fn
        self.config = config or MinimaxConfig()
        self.rng = rng or np.random.default_rng()

    def select_action(
        self,
        game: TurnBasedGame[StateT],
        state: StateT,
        legal_actions: Optional[Sequence[Action]] = None,
    ) -> Action:
        if legal_actions is None:
            legal_actions = list(game.legal_actions(state))
        if not legal_actions:
            raise ValueError("No legal actions")

        best_value = -math.inf
        best_actions: list[Action] = []

        for action in legal_actions:
            next_state = game.apply_action(state, action)
            if self.config.use_alpha_beta:
                value = -self._negamax_ab(game, next_state, self.config.depth - 1, -math.inf, math.inf)
            else:
                value = -self._negamax(game, next_state, self.config.depth - 1)

            if value > best_value + self.config.value_epsilon:
                best_value = value
                best_actions = [action]
            elif abs(value - best_value) <= self.config.value_epsilon:
                best_actions.append(action)

        if not best_actions:
            return legal_actions[0]
        if len(best_actions) == 1 or not self.config.random_tiebreak:
            return best_actions[0]
        return self.rng.choice(best_actions)

    def _negamax(self, game: TurnBasedGame[StateT], state: StateT, depth: int) -> float:
        """Simple negamax without pruning."""
        if depth == 0 or game.is_terminal(state):
            return self.value_fn.evaluate(game, state)

        legal_actions = list(game.legal_actions(state))
        if not legal_actions:
            return self.value_fn.evaluate(game, state)

        best = -math.inf
        for action in legal_actions:
            next_state = game.apply_action(state, action)
            val = -self._negamax(game, next_state, depth - 1)
            if val > best:
                best = val
        return best

    def _negamax_ab(self, game: TurnBasedGame[StateT], state: StateT, depth: int, alpha: float, beta: float) -> float:
        """Negamax with alpha-beta pruning."""
        if depth == 0 or game.is_terminal(state):
            return self.value_fn.evaluate(game, state)

        legal_actions = list(game.legal_actions(state))
        if not legal_actions:
            return self.value_fn.evaluate(game, state)

        value = -math.inf
        for action in legal_actions:
            next_state = game.apply_action(state, action)
            child_val = -self._negamax_ab(game, next_state, depth - 1, -beta, -alpha)
            if child_val > value:
                value = child_val
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value

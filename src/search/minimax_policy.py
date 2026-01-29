"""Minimax search policy with alpha-beta pruning."""

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
    use_alpha_beta: bool = True
    random_tiebreak: bool = True
    value_epsilon: float = 1e-6


class MinimaxPolicy(ActionPolicy[StateT], Generic[StateT]):
    """Negamax-based minimax policy over TurnBasedGame + StateValueFn."""

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
        if self.config.depth <= 0:
            raise ValueError("Minimax depth must be >= 1")

        if legal_actions is None:
            legal_actions = list(game.legal_actions(state))

        if not legal_actions:
            raise ValueError("No legal actions available for minimax")

        best_value = -math.inf
        best_actions: list[Action] = []
        root_alpha = -math.inf  # Track best value for root-level pruning

        for action in legal_actions:
            next_state = game.apply_action(state, action)
            value = -self._search(
                game=game,
                state=next_state,
                depth=self.config.depth - 1,
                alpha=-math.inf,
                beta=-root_alpha,  # Tell child: beat -root_alpha or I don't care
            )

            if value > best_value + self.config.value_epsilon:
                best_value = value
                best_actions = [action]
            elif abs(value - best_value) <= self.config.value_epsilon:
                best_actions.append(action)

            # Update root_alpha for subsequent iterations
            if self.config.use_alpha_beta:
                root_alpha = max(root_alpha, value)

        # Fallback if all values were -inf (edge case)
        if not best_actions:
            return legal_actions[0]

        if len(best_actions) == 1 or not self.config.random_tiebreak:
            return best_actions[0]

        return self.rng.choice(best_actions)

    def _search(
        self,
        game: TurnBasedGame[StateT],
        state: StateT,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        if depth == 0 or game.is_terminal(state):
            return self.value_fn.evaluate(game, state)

        legal_actions = list(game.legal_actions(state))
        if not legal_actions:
            return self.value_fn.evaluate(game, state)

        value = -math.inf

        for action in legal_actions:
            next_state = game.apply_action(state, action)
            child_value = -self._search(
                game=game,
                state=next_state,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
            )

            if child_value > value:
                value = child_value

            if self.config.use_alpha_beta:
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break

        return value

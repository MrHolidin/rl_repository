"""MCTS as ActionPolicy adapter."""

from __future__ import annotations

from typing import Generic, Optional, Sequence, TypeVar

import numpy as np

from src.games.turn_based_game import Action, TurnBasedGame
from src.search.action_policy import ActionPolicy

from .config import MCTSConfig
from .tree import MCTS, NetworkEvaluator

S = TypeVar("S")


class MCTSPolicy(ActionPolicy[S], Generic[S]):
    """ActionPolicy that uses MCTS for action selection."""

    def __init__(
        self,
        game: TurnBasedGame[S],
        evaluator: NetworkEvaluator,
        config: MCTSConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        self.game = game
        self.evaluator = evaluator
        self.config = config
        self.mcts = MCTS(game, evaluator, config, rng)
        self._move_count = 0

    def select_action(
        self,
        game: TurnBasedGame[S],
        state: S,
        legal_actions: Optional[Sequence[Action]] = None,
    ) -> Action:
        root = self.mcts.search(state, add_dirichlet_noise=False)
        temperature = self.config.get_temperature(self._move_count)
        action, _ = self.mcts.get_action_probs(root, temperature)
        self._move_count += 1
        return action

    def reset_move_count(self) -> None:
        self._move_count = 0

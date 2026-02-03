"""Adapter that wraps MinimaxPolicy as BaseAgent for use in env evaluation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.connect4 import Connect4Game, Connect4State
from ..minimax_policy import MinimaxPolicy


class Connect4MinimaxEnvAdapter(BaseAgent):
    """
    Wraps a MinimaxPolicy so it can be used as a BaseAgent in play_single_game.
    Requires a get_state callback (e.g. lambda: env.get_state()) bound to the current env.
    """

    def __init__(
        self,
        game: Connect4Game,
        policy: MinimaxPolicy[Connect4State],
        get_state: Callable[[], Connect4State],
    ) -> None:
        self._game = game
        self._policy = policy
        self._get_state = get_state

    def act(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> int:
        state = self._get_state()
        legal_actions = list(self._game.legal_actions(state))
        if legal_mask is not None:
            legal_actions = [a for a in legal_actions if 0 <= a < len(legal_mask) and legal_mask[a]]
        return int(self._policy.select_action(self._game, state, legal_actions))

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str, **kwargs: object) -> BaseAgent:
        raise NotImplementedError("Minimax adapter has no checkpoint load")

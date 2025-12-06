from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from src.agents.dqn_agent import DQNAgent
from ..connect4_state import Connect4State
from ..utils import build_state_dict
from src.features.observation_builder import ObservationBuilder
from src.games.turn_based_game import Action, TurnBasedGame
from src.policies.action_policy import ActionPolicy


class Connect4DQNPolicy(ActionPolicy[Connect4State]):
    """
    Wraps ``DQNAgent`` so it can be used as an ``ActionPolicy[Connect4State]``.

    Builds the observation / legal-mask from ``(game, state)``
    and forwards the call to ``DQNAgent.act``.
    """

    def __init__(
        self,
        dqn_agent: DQNAgent,
        observation_builder: ObservationBuilder,
        num_actions: int,
        deterministic: bool = True,
    ) -> None:
        self.dqn_agent = dqn_agent
        self.observation_builder = observation_builder
        self.num_actions = int(num_actions)
        self.deterministic = deterministic

    def select_action(
        self,
        game: TurnBasedGame[Connect4State],
        state: Connect4State,
        legal_actions: Optional[Sequence[Action]] = None,
    ) -> Action:
        game_legal = list(game.legal_actions(state))
        if legal_actions is None:
            effective_legal = game_legal
        else:
            allowed = set(legal_actions)
            effective_legal = [a for a in game_legal if a in allowed]

        if not effective_legal:
            raise ValueError("Connect4DQNPolicy: no legal actions available")

        obs, legal_mask = self._build_obs_and_mask(game, state, effective_legal)
        action = self.dqn_agent.act(
            obs=obs,
            legal_mask=legal_mask,
            deterministic=self.deterministic,
        )
        return int(action)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs_and_mask(
        self,
        game: TurnBasedGame[Connect4State],
        state: Connect4State,
        legal_actions: Sequence[Action],
    ) -> Tuple[np.ndarray, np.ndarray]:
        legal_mask = np.zeros(self.num_actions, dtype=bool)
        for a in legal_actions:
            if 0 <= a < self.num_actions:
                legal_mask[a] = True

        state_dict = build_state_dict(state, game, legal_mask=legal_mask)
        obs = self.observation_builder.build(state_dict)
        return obs, legal_mask


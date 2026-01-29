"""DQN-based value function for Connect4 search."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.envs.connect4 import (
    Connect4Game,
    Connect4State,
    connect4_terminal_evaluator,
    build_state_dict,
)
from src.games.turn_based_game import TurnBasedGame
from src.features.observation_builder import ObservationBuilder
from src.agents.dqn_agent import DQNAgent
from ..value_fn import StateValueFn


class Connect4DQNValueFn(StateValueFn[Connect4State]):
    def __init__(
        self,
        dqn_agent: DQNAgent,
        observation_builder: ObservationBuilder,
        *,
        default_value_on_no_actions: float = 0.0,
    ) -> None:
        self._agent = dqn_agent
        self._observation_builder = observation_builder
        self._default_value_on_no_actions = default_value_on_no_actions

    def evaluate(
        self,
        game: TurnBasedGame[Connect4State],
        state: Connect4State,
    ) -> float:
        assert isinstance(game, Connect4Game), "Connect4DQNValueFn expects Connect4Game"

        # Handle terminal states using existing evaluator
        if game.is_terminal(state):
            current_token = game.current_player(state)
            return connect4_terminal_evaluator(game, state, root_player=current_token)

        legal_actions: Sequence[int] = game.legal_actions(state)
        if not legal_actions:
            return self._default_value_on_no_actions

        num_actions = self._agent.num_actions
        legal_mask = np.zeros(num_actions, dtype=bool)
        for a in legal_actions:
            if 0 <= a < num_actions:
                legal_mask[a] = True

        state_dict = build_state_dict(state, game, legal_mask=legal_mask)
        obs = self._observation_builder.build(state_dict)

        self._agent.q_network.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self._agent.device
            ).unsqueeze(0)
            legal_mask_tensor = torch.as_tensor(
                legal_mask, dtype=torch.bool, device=self._agent.device
            ).unsqueeze(0)

            q_values = self._agent.q_network(obs_tensor, legal_mask=legal_mask_tensor)[0]

        q_values_np = q_values.cpu().numpy().copy()
        q_values_np[~legal_mask] = -np.inf

        if not np.any(legal_mask):
            return self._default_value_on_no_actions

        value = float(np.max(q_values_np))
        return value

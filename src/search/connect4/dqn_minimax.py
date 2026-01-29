"""DQN-Minimax policy factory for Connect4."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.envs.connect4 import Connect4Game, Connect4State
from src.features.observation_builder import ObservationBuilder
from src.agents.dqn_agent import DQNAgent
from ..minimax_policy import MinimaxPolicy, MinimaxConfig
from .dqn_value_fn import Connect4DQNValueFn


def make_connect4_dqn_minimax_policy(
    *,
    game: Connect4Game,
    dqn_agent: DQNAgent,
    observation_builder: ObservationBuilder,
    depth: int = 2,
    random_tiebreak: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> MinimaxPolicy[Connect4State]:
    value_fn = Connect4DQNValueFn(
        dqn_agent=dqn_agent,
        observation_builder=observation_builder,
    )
    config = MinimaxConfig(
        depth=depth,
        use_alpha_beta=True,
        random_tiebreak=random_tiebreak,
    )
    return MinimaxPolicy[Connect4State](
        value_fn=value_fn,
        config=config,
        rng=rng,
    )

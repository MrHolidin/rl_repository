"""Heuristic minimax policy for Connect4."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from src.envs.connect4 import Connect4Game, Connect4State
from ..minimax_policy import MinimaxConfig, MinimaxPolicy
from .heuristic_value_fn import Connect4HeuristicValueFn
from .smart_heuristic_value_fn import Connect4SmartHeuristicValueFn


def make_connect4_heuristic_minimax_policy(
    game: Connect4Game,
    *,
    depth: int = 2,
    heuristic: Literal["trivial", "smart"] = "trivial",
    use_alpha_beta: bool = True,
    random_tiebreak: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> MinimaxPolicy[Connect4State]:
    if heuristic == "smart":
        value_fn = Connect4SmartHeuristicValueFn(normalize=True)
    else:
        value_fn = Connect4HeuristicValueFn()
    config = MinimaxConfig(
        depth=depth,
        use_alpha_beta=use_alpha_beta,
        random_tiebreak=random_tiebreak,
    )
    return MinimaxPolicy[Connect4State](
        value_fn=value_fn,
        config=config,
        rng=rng or np.random.default_rng(),
    )

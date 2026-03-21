"""Helpers to run evaluation games between agents."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.envs.base import TurnBasedEnv
from src.training.opponent_sampler import OpponentSampler
from src.training.random_opening import RandomOpeningConfig
from src.training.trainer import StartPolicy
from src.utils.agent_utils import freeze_agent
from src.utils.match import play_single_game  # noqa: F401  (re-exported via __init__)


def play_series_vs_sampler(
    env_factory: Callable[[], TurnBasedEnv],
    agent: BaseAgent,
    opponent_sampler: OpponentSampler,
    *,
    num_games: int,
    start_policy: StartPolicy = StartPolicy.RANDOM,
    random_opening_config: Optional[RandomOpeningConfig] = None,
    deterministic_agent: bool = True,
    deterministic_opponent: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Play a series of games against opponents drawn from ``opponent_sampler``.

    Args:
        env_factory: Callable returning a fresh environment per game.
        agent: Agent under evaluation.
        opponent_sampler: Sampler that provides opponents (uses prepare/sample/on_episode_end).
        num_games: Number of games to play.
        start_policy: Who goes first (agent, opponent, or random).
        random_opening_config: Optional random opening applied before each game.
        deterministic_agent: Whether the main agent acts deterministically.
        deterministic_opponent: Whether sampled opponents act deterministically.
        seed: Optional seed controlling start policy randomness and openings.

    Returns:
        Summary dictionary with fields ``wins``, ``losses``, ``draws`` and per-game ``history``.
    """
    rng = random.Random(seed) if seed is not None else None
    wins = losses = draws = 0
    history: List[Dict[str, Any]] = []

    for game_idx in range(num_games):
        env = env_factory()
        opponent_sampler.prepare(game_idx)
        opponent = opponent_sampler.sample()
        freeze_agent(opponent)

        per_game_seed = (rng.randrange(0, 2**32 - 1) if rng is not None else None)
        result = play_single_game(
            env,
            agent,
            opponent,
            start_policy=start_policy,
            random_opening_config=random_opening_config,
            deterministic_agent=deterministic_agent,
            deterministic_opponent=deterministic_opponent,
            seed=per_game_seed,
        )
        winner = result["winner"]
        if winner == result["agent_token"]:
            wins += 1
        elif winner == -result["agent_token"]:
            losses += 1
        else:
            draws += 1

        history.append(result)
        episode_info = {
            "reward": result["reward"],
            "length": result["steps"],
            "info": {"winner": winner},
        }
        opponent_sampler.on_episode_end(game_idx, episode_info)

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "history": history,
    }

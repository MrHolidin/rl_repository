"""Utilities for injecting randomized openings into training episodes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, Tuple


class SupportsRandomOpeningEnv(Protocol):
    """Protocol for environments that support randomized openings."""

    done: bool

    def get_legal_actions(self) -> Sequence[int]:
        ...

    def step(self, action: int):
        ...


@dataclass
class RandomOpeningConfig:
    """Configuration for randomized opening prologues."""

    probability: float = 0.5
    min_half_moves: int = 2
    max_half_moves: int = 6

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0.0 and 1.0")
        if self.min_half_moves < 0:
            raise ValueError("min_half_moves must be non-negative")
        if self.max_half_moves < self.min_half_moves:
            raise ValueError("max_half_moves must be greater than or equal to min_half_moves")


def maybe_apply_random_opening(
    env: SupportsRandomOpeningEnv,
    initial_obs,
    config: Optional[RandomOpeningConfig],
    rng: Optional[random.Random] = None,
) -> Tuple[Any, int, bool]:
    """
    Optionally apply a randomized opening prologue by sampling random legal moves.

    Args:
        env: Environment instance.
        initial_obs: Observation returned by env.reset().
        config: Random opening configuration; if None or disabled, the function is a no-op.
        rng: Optional random number generator; defaults to global random module.

    Returns:
        Tuple of (obs, moves_played, episode_done) where:
            obs: Observation after applying the opening (or initial_obs if no moves applied)
            moves_played: Number of half-moves executed during the prologue
            episode_done: Whether the env transitioned to a terminal state during prologue
    """
    if config is None:
        return initial_obs, 0, getattr(env, "done", False)

    rand = rng if rng is not None else random

    if rand.random() >= config.probability:
        return initial_obs, 0, getattr(env, "done", False)

    moves_to_play = rand.randint(config.min_half_moves, config.max_half_moves)
    moves_played = 0
    obs = initial_obs

    for _ in range(moves_to_play):
        if getattr(env, "done", False):
            break

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break

        action = rand.choice(legal_actions)
        obs, _, _, _ = env.step(action)
        moves_played += 1

        if getattr(env, "done", False):
            break

    return obs, moves_played, getattr(env, "done", False)


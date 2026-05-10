"""Wrap MiniBG ``HeuristicBot`` as ``BaseAgent`` for ``play_match`` / eval utilities."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.base import TurnBasedEnv
from src.envs.minibg.env import MiniBGEnv

from .bots import HeuristicBot


class MiniBGHeuristicAgent(BaseAgent):
    """
    Uses ``set_env`` (called from ``play_match``) then ignores ``obs`` in ``act``,
    delegating to ``HeuristicBot.choose_action(env)`` — same information as tournament play.
    """

    def __init__(self, bot: HeuristicBot) -> None:
        self._bot = bot
        self._env: Optional[MiniBGEnv] = None

    def set_env(self, env: TurnBasedEnv) -> None:
        if not isinstance(env, MiniBGEnv):
            raise TypeError(f"MiniBGHeuristicAgent requires MiniBGEnv, got {type(env)}")
        self._env = env

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        if self._env is None:
            raise RuntimeError("MiniBGHeuristicAgent: call set_env(env) before act().")
        return int(self._bot.choose_action(self._env))

    def observe(self, transition: Any, is_augmented: bool = False) -> dict:
        return {}

    def update(self) -> dict:
        return {}

    def save(self, path: str) -> None:
        return None

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "MiniBGHeuristicAgent":
        raise NotImplementedError("MiniBGHeuristicAgent is not loadable from disk")

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


__all__ = ["MiniBGHeuristicAgent"]

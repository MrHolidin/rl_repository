"""Wrap BGLike heuristic bots as ``BaseAgent`` for training / eval."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.bglike.lobby_env import BGLobbyMultiCurrentEnv

from .bots import HeuristicBot, make_bot
from .env_view import BGLikeHeuristicEnvView


class BGLikeHeuristicAgent(BaseAgent):
    def __init__(self, bot: HeuristicBot) -> None:
        self._bot = bot
        self._view: Optional[BGLikeHeuristicEnvView] = None

    def set_env(self, env: Any) -> None:
        if isinstance(env, BGLobbyMultiCurrentEnv):
            self._view = BGLikeHeuristicEnvView(env)
            return
        lobby = getattr(env, "lobby", None)
        if isinstance(lobby, BGLobbyMultiCurrentEnv):
            self._view = BGLikeHeuristicEnvView(lobby)
            return
        raise TypeError(
            f"BGLikeHeuristicAgent requires BGLobbyMultiCurrentEnv, got {type(env)}"
        )

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        del obs, deterministic
        if self._view is None:
            raise RuntimeError("BGLikeHeuristicAgent: call set_env() before act().")
        self._view.set_mask_override(legal_mask)
        try:
            return int(self._bot.choose_action(self._view))
        finally:
            self._view.set_mask_override(None)

    def observe(self, transition: Any, is_augmented: bool = False) -> dict:
        return {}

    def update(self) -> dict:
        return {}

    def save(self, path: str) -> None:
        return None

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "BGLikeHeuristicAgent":
        raise NotImplementedError("BGLikeHeuristicAgent is not loadable from disk")

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


def make_heuristic_agent(name: str, *, seed: Optional[int] = None) -> BGLikeHeuristicAgent:
    return BGLikeHeuristicAgent(make_bot(name, seed=seed))


__all__ = ["BGLikeHeuristicAgent", "make_heuristic_agent"]

from __future__ import annotations

from typing import Any, Generic, List, Optional, Sequence, TypeVar

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.base import TurnBasedEnv
from src.games.turn_based_game import Action, TurnBasedGame
from src.policies.action_policy import ActionPolicy

S = TypeVar("S")


class GamePolicyEnvAgent(BaseAgent, Generic[S]):
    """
    Wraps an ``ActionPolicy`` so it can play matches through ``BaseAgent``.

    ``play_match`` still owns Env + BaseAgent + obs, but this adapter ignores the
    provided observation and interrogates ``env.get_state()`` instead.
    """

    def __init__(
        self,
        *,
        game: TurnBasedGame[S],
        env: Optional[TurnBasedEnv] = None,
        policy: ActionPolicy[S],
        num_actions: int,
    ) -> None:
        self._game = game
        self._env = env
        self._policy = policy
        self.num_actions = num_actions

    def set_env(self, env: TurnBasedEnv) -> None:
        """Set the environment reference used for reading game state."""
        self._env = env

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        if self._env is None:
            raise RuntimeError(
                "GamePolicyEnvAgent: env is not set. "
                "Call set_env() before using the agent, or pass env= to __init__."
            )
        state: S = self._env.get_state()
        legal_actions = list(self._game.legal_actions(state))
        if not legal_actions:
            raise ValueError("GamePolicyEnvAgent: no legal actions in current state.")

        action = self._policy.select_action(
            self._game,
            state,
            legal_actions=legal_actions,
        )
        return int(action)

    def observe(self, transition: Any) -> dict:
        return {}

    def update(self) -> dict:
        return {}

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "GamePolicyEnvAgent[S]":
        raise NotImplementedError("GamePolicyEnvAgent.load is not implemented")

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


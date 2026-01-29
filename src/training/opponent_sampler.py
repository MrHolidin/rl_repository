from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.agents import RandomAgent

try:
    from src.training.selfplay import OpponentPool  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpponentPool = None  # type: ignore[misc]


class OpponentSampler(ABC):
    """Abstract opponent sampler for Trainer self-play."""

    def prepare(self, episode_index: int) -> None:
        """Prepare sampler for upcoming episode."""

    def on_episode_end(self, episode_index: int, info: Optional[dict] = None) -> None:
        """Hook invoked after an episode finishes."""

    def on_checkpoint(self, path: str | Path, episode_index: int) -> None:
        """Hook invoked when a new checkpoint is saved."""

    @abstractmethod
    def sample(self) -> BaseAgent:
        """Return an agent to be used as the opponent for the next episode."""


class RandomOpponentSampler(OpponentSampler):
    """Always returns a fresh random agent."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._episodes_sampled = 0

    def prepare(self, episode_index: int) -> None:  # pragma: no cover - trivial
        self._episodes_sampled = episode_index

    def sample(self) -> BaseAgent:
        agent = RandomAgent(seed=(self.seed if self.seed is not None else None))
        self._episodes_sampled += 1
        return agent


class OpponentPoolSampler(OpponentSampler):
    """Adapter around OpponentPool from legacy training pipeline."""

    def __init__(self, opponent_pool: OpponentPool):  # type: ignore[valid-type]
        if OpponentPool is None:
            raise RuntimeError("OpponentPool is unavailable; install self-play components.")
        self.opponent_pool = opponent_pool
        self._next_episode = 0
        self._current_opponent: Optional[BaseAgent] = None

    def prepare(self, episode_index: int) -> None:
        self._next_episode = episode_index

    def sample(self) -> BaseAgent:
        episode = self._next_episode
        opponent = self.opponent_pool.sample_opponent(episode)  # type: ignore[operator]
        self._current_opponent = opponent
        if hasattr(opponent, "eval"):
            opponent.eval()
        if hasattr(opponent, "epsilon"):
            setattr(opponent, "epsilon", 0.0)
        return opponent

    def on_episode_end(self, episode_index: int, info: Optional[dict] = None) -> None:
        if self._current_opponent is None or info is None:
            return

        agent_result = info.get("agent_result")
        if agent_result is None:
            return

        for frozen in self.opponent_pool.frozen_agents:
            if frozen.loaded_agent is self._current_opponent:
                frozen.games += 1
                if agent_result == 1:
                    frozen.losses += 1
                elif agent_result == -1:
                    frozen.wins += 1
                elif agent_result == 0:
                    frozen.draws += 1
                break

        self._current_opponent = None

    def on_checkpoint(self, path: str | Path, episode_index: int) -> None:
        try:
            self.opponent_pool.add_frozen_agent(str(path), episode_index)  # type: ignore[operator]
        except FileNotFoundError:  # pragma: no cover - safety
            pass


__all__ = [
    "OpponentSampler",
    "RandomOpponentSampler",
    "OpponentPoolSampler",
]

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Sequence

from src.agents.base_agent import BaseAgent
from src.agents import RandomAgent
from src.utils.agent_utils import freeze_agent

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

    def on_rollout_end(self, metrics: Optional[dict] = None) -> None:
        """Hook invoked after a PPO rollout update (stats flush point)."""

    def on_checkpoint(self, path: str | Path, episode_index: int) -> None:
        """Hook invoked when a new checkpoint is saved."""

    @abstractmethod
    def sample(self) -> BaseAgent:
        """Return an agent to be used as the opponent for the next episode."""

    def sample_for_seats(self, seats: Sequence[int]) -> Dict[int, BaseAgent]:
        """Independent opponent per lobby seat (BGLike training)."""
        return {int(seat): self.sample() for seat in seats}


class RandomOpponentSampler(OpponentSampler):
    """Returns a fresh random agent per episode with episode-varying seed for diversity."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._episode_index = 0

    def prepare(self, episode_index: int) -> None:
        self._episode_index = episode_index

    def sample(self) -> BaseAgent:
        episode = self._episode_index
        self._episode_index += 1
        # Episode-varying seed gives diverse opponents each episode (match OpponentPool)
        opp_seed = (self.seed + 100000 + episode) if self.seed is not None else None
        return RandomAgent(seed=opp_seed)


class OpponentPoolSampler(OpponentSampler):
    """Adapter around OpponentPool from legacy training pipeline."""

    def __init__(self, opponent_pool: OpponentPool):  # type: ignore[valid-type]
        if OpponentPool is None:
            raise RuntimeError("OpponentPool is unavailable; install self-play components.")
        self.opponent_pool = opponent_pool
        self._next_episode = 0
        self._current_opponent: Optional[BaseAgent] = None
        self._episode_slot_by_seat: Dict[int, int] = {}

    def prepare(self, episode_index: int) -> None:
        self._next_episode = episode_index
        self._episode_slot_by_seat = {}

    @property
    def last_slot_id(self) -> int:
        return int(self.opponent_pool._last_sample_slot_id)  # type: ignore[union-attr]

    def sample(self) -> BaseAgent:
        episode = self._next_episode
        opponent = self.opponent_pool.sample_opponent(episode)  # type: ignore[operator]
        self._current_opponent = opponent
        freeze_agent(opponent)
        return opponent

    def sample_for_seats(self, seats: Sequence[int]) -> Dict[int, BaseAgent]:
        agents: Dict[int, BaseAgent] = {}
        slots: Dict[int, int] = {}
        for seat in seats:
            s = int(seat)
            agents[s] = self.sample()
            slots[s] = self.last_slot_id
        self._episode_slot_by_seat = slots
        self._slot_by_seat = slots
        return agents

    def on_episode_end(self, episode_index: int, info: Optional[dict] = None) -> None:
        if info is None:
            return
        pool = self.opponent_pool
        if info.get("skip_league_record"):
            self._current_opponent = None
            self._episode_slot_by_seat = {}
            return
        league_outcomes = info.get("opponent_league_outcomes")
        if league_outcomes:
            for item in league_outcomes:
                pool.record_outcome_for_slot(  # type: ignore[operator]
                    int(item["slot_id"]),
                    item.get("agent_result"),
                )
            self._current_opponent = None
            self._episode_slot_by_seat = {}
            return
        agent_result = info.get("agent_result")
        pool.record_episode_outcome(agent_result)  # type: ignore[operator]
        self._current_opponent = None
        self._episode_slot_by_seat = {}

    def on_rollout_end(self, metrics: Optional[dict] = None) -> None:
        self.opponent_pool.flush_pending_outcomes()  # type: ignore[operator]

    def on_checkpoint(self, path: str | Path, episode_index: int) -> None:
        pool = self.opponent_pool
        if pool.self_play_config is None:  # type: ignore[union-attr]
            return
        try:
            pool.add_frozen_agent(str(path), episode_index)  # type: ignore[operator]
        except FileNotFoundError:  # pragma: no cover - safety
            pass


__all__ = [
    "OpponentSampler",
    "RandomOpponentSampler",
    "OpponentPoolSampler",
]

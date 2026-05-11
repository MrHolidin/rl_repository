from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence

import random as py_random

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

    def on_checkpoint(self, path: str | Path, episode_index: int) -> None:
        """Hook invoked when a new checkpoint is saved."""

    @abstractmethod
    def sample(self) -> BaseAgent:
        """Return an agent to be used as the opponent for the next episode."""


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


class MiniBGMixedOpponentSampler(OpponentSampler):
    """``RandomAgent`` share ``random_fraction``; else uniform over ``bots`` (MiniBG heuristics)."""

    def __init__(
        self,
        *,
        seed: Optional[int],
        bots: Sequence[str],
        random_fraction: Optional[float] = None,
        equal_opponent_mass: bool = False,
        learning_agent: Optional[BaseAgent] = None,
        self_play: Optional[dict] = None,
    ) -> None:
        from src.envs.minibg.heuristic_bots.bots import default_bot_constructors
        from src.envs.minibg.heuristic_bots.tournament import make_bot

        self.seed = seed
        if not bots:
            raise ValueError("minibg_mixed: non-empty bots list is required")
        valid = frozenset(default_bot_constructors().keys())
        cleaned: List[str] = []
        for b in bots:
            key = str(b).strip()
            if key == "random":
                raise ValueError(
                    "minibg_mixed: use random_fraction for random; do not list 'random' in bots"
                )
            if key not in valid:
                raise ValueError(f"minibg_mixed: unknown bot {key!r}; valid: {sorted(valid)}")
            cleaned.append(key)
        self._bots = cleaned
        self._make_bot = make_bot
        self._episode_index = 0

        sp = dict(self_play or {})
        csf = float(sp.get("current_self_fraction", 0.0))
        if csf < 0.0 or csf > 1.0:
            raise ValueError("minibg_mixed: self_play.current_self_fraction must be in [0, 1]")
        self._self_play_csf = csf
        self._self_play_start = int(sp.get("start_episode", 0))
        self._learning_agent = learning_agent
        if self._self_play_csf > 0.0 and learning_agent is None:
            raise ValueError(
                "minibg_mixed: self_play.current_self_fraction > 0 requires the learning agent"
            )

        if equal_opponent_mass:
            self.random_fraction = 1.0 / (1.0 + float(len(self._bots)))
        elif random_fraction is not None:
            self.random_fraction = min(1.0, max(0.0, float(random_fraction)))
        else:
            self.random_fraction = 0.5

    def prepare(self, episode_index: int) -> None:
        self._episode_index = episode_index

    def sample(self) -> BaseAgent:
        from src.envs.minibg.heuristic_bots.agent_adapter import MiniBGHeuristicAgent
        from src.training.selfplay.opponent_pool import SelfPlayOpponent

        episode = self._episode_index
        self._episode_index += 1
        rng_ep = (self.seed + 100000 + episode) if self.seed is not None else episode
        rnd = py_random.Random(rng_ep + 17)

        if (
            self._learning_agent is not None
            and self._self_play_csf > 0.0
            and episode >= self._self_play_start
            and rnd.random() < self._self_play_csf
        ):
            return SelfPlayOpponent(self._learning_agent, greedy=True)

        rnd2 = py_random.Random(rng_ep + 99).random()
        if rnd2 < self.random_fraction:
            opp_seed = (self.seed + 200000 + episode) if self.seed is not None else None
            return RandomAgent(seed=opp_seed)
        name = rnd.choice(self._bots)
        return MiniBGHeuristicAgent(self._make_bot(name, rng_ep + 31))


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
        freeze_agent(opponent)
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
    "MiniBGMixedOpponentSampler",
    "OpponentPoolSampler",
]

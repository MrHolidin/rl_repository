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


class DvDPopulationSampler(OpponentPoolSampler):
    """Population co-play sampler for DvD/v7 identity-conditioned agents.

    Lobby seats are filled from three sources, keeping league memory intact:
      * the *learner* (handled by the env on ``current_seats``) under a pinned
        per-episode identity ``i``;
      * **sibling** seats — the same live net under a fixed identity ``j != i``
        (live diversity / co-play);
      * **frozen** seats — past snapshots from the pool (league memory /
        anti-forgetting), each assigned an explicit identity at seating.

    Opponent seats split ~``sibling_fraction`` siblings / rest frozen. When no
    frozen snapshot exists yet (early training) the seat falls back to a
    sibling. Identity choice is uniform here; PFSP weighting is a later step.
    """

    def __init__(
        self,
        opponent_pool,  # type: ignore[valid-type]
        *,
        num_identities: int,
        sibling_fraction: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(opponent_pool=opponent_pool)
        import random as _random

        from src.agents.ppo_dvd_agent import SiblingOpponent
        from src.training.selfplay.game_record import SLOT_CURRENT

        self._SiblingOpponent = SiblingOpponent
        self._SLOT_CURRENT = int(SLOT_CURRENT)
        self.num_identities = int(num_identities)
        self.sibling_fraction = float(sibling_fraction)
        self._rng = _random.Random(seed)
        self._learner_identity = 0

    def prepare(self, episode_index: int) -> None:
        super().prepare(episode_index)
        # In per-seat mode the learner assigns its OWN identities across its
        # current seats (multiple co-learners share one net), so we must NOT pin
        # a single identity on it here. Sibling/frozen opponent seats still get
        # explicit identities below. ``_learner_identity`` is left unused for
        # exclusion; siblings simply draw a uniform identity.

    def _other_identity(self) -> int:
        # Uniform identity for an opponent seat (no exclusion: the learner spans
        # several identities across its seats, so there is no single "self" id).
        return self._rng.randrange(self.num_identities)

    def sample_for_seats(self, seats: Sequence[int]) -> Dict[int, BaseAgent]:
        agents: Dict[int, BaseAgent] = {}
        slots: Dict[int, int] = {}
        learner = getattr(self.opponent_pool, "current_agent", None)
        for seat in seats:
            s = int(seat)
            want_sibling = (
                learner is not None and self._rng.random() < self.sibling_fraction
            )
            agent = None
            if not want_sibling:
                agent = self._try_frozen_seat()  # league memory first
            if agent is not None:
                slots[s] = self.last_slot_id
            elif learner is not None:
                # Sibling seat: same live net under a different identity.
                agent = self._SiblingOpponent(
                    learner, identity=self._other_identity()
                )
                slots[s] = self._SLOT_CURRENT
            else:
                # No learner and no frozen → fall back to the base pool sampler.
                agent = self.sample()
                slots[s] = self.last_slot_id
            agents[s] = agent
        self._episode_slot_by_seat = slots
        self._slot_by_seat = slots
        return agents

    def _try_frozen_seat(self):
        """Return a frozen snapshot agent with an explicit identity, or None."""
        pool = self.opponent_pool
        info = pool._sample_frozen_info()  # type: ignore[attr-defined]
        if info is None:
            return None
        agent = pool._get_loaded_frozen_agent(  # type: ignore[attr-defined]
            info, step=self._next_episode
        )
        pool._last_sample_slot_id = info.slot_id
        # A frozen DvD snapshot carries all identities; pin one explicitly so it
        # doesn't replay the random identity from its __init__.
        if hasattr(agent, "set_episode_identity"):
            agent.set_episode_identity(self._rng.randrange(self.num_identities))
        return agent


__all__ = [
    "OpponentSampler",
    "RandomOpponentSampler",
    "OpponentPoolSampler",
    "DvDPopulationSampler",
]

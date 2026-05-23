"""League rating backends (EMA pairwise, TrueSkill, ...)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .game_record import SLOT_CURRENT, GameRecord, league_updates_from_record


@dataclass
class EmaSlotStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    ema_win_rate: float = 0.5


class RatingSystem(ABC):
    @abstractmethod
    def register(self, slot_id: int) -> None: ...

    @abstractmethod
    def remove(self, slot_id: int) -> None: ...

    @abstractmethod
    def update(self, record: GameRecord) -> bool:
        """Apply one game record. Returns True if any slot stats changed."""

    @abstractmethod
    def rating(self, slot_id: int) -> float:
        """Scalar strength for PFSP / status (higher = opponent beats learner more)."""

    @abstractmethod
    def summary(self, slot_id: int) -> Dict[str, Any]: ...

    @abstractmethod
    def eviction_sort_key(self, slot_id: int, *, episode: int) -> Tuple[float, int]:
        """Sort ascending; first entry is evicted first."""


def trueskill_match_quality(
    mu_a: float,
    sigma_a: float,
    mu_b: float,
    sigma_b: float,
    *,
    beta: float = 25.0 / 6.0,
) -> float:
    """Approximate 1v1 match quality: high when skills are close and certain."""
    try:
        import trueskill as ts

        env = ts.global_env()
        ra = env.create_rating(float(mu_a), float(sigma_a))
        rb = env.create_rating(float(mu_b), float(sigma_b))
        # ``quality`` expects a list of teams; each team is an iterable of ratings.
        # For 1v1 that's two teams of one — passing ``[ra], [rb]`` (one team of two)
        # raises ValueError("Need multiple rating groups").
        return float(env.quality([(ra,), (rb,)]))
    except ImportError:
        denom = math.sqrt(2.0 * beta ** 2 + sigma_a ** 2 + sigma_b ** 2)
        if denom <= 0.0:
            return 0.0
        delta = abs(mu_a - mu_b)
        return math.exp(-0.5 * (delta / denom) ** 2)


class EmaPairwiseRating(RatingSystem):
    """Pairwise learner-vs-opponent EMA (legacy league behavior)."""

    def __init__(self, *, ema_beta: float = 0.05) -> None:
        self._beta = float(ema_beta)
        self._stats: Dict[int, EmaSlotStats] = {}

    def register(self, slot_id: int) -> None:
        sid = int(slot_id)
        if sid not in self._stats:
            self._stats[sid] = EmaSlotStats()

    def remove(self, slot_id: int) -> None:
        self._stats.pop(int(slot_id), None)

    def update(self, record: GameRecord) -> bool:
        updates = league_updates_from_record(record)
        if not updates:
            return False
        for slot_id, score in updates:
            stats = self._stats.get(int(slot_id))
            if stats is None:
                continue
            stats.games += 1
            if score > 0.5:
                stats.losses += 1
            elif score < 0.5:
                stats.wins += 1
            else:
                stats.draws += 1
            y = 1.0 - score
            stats.ema_win_rate = (1.0 - self._beta) * stats.ema_win_rate + self._beta * y
        return True

    def rating(self, slot_id: int) -> float:
        stats = self._stats.get(int(slot_id))
        if stats is None:
            return 0.5
        return stats.ema_win_rate

    def summary(self, slot_id: int) -> Dict[str, Any]:
        stats = self._stats.get(int(slot_id), EmaSlotStats())
        return {
            "games": stats.games,
            "wins": stats.wins,
            "losses": stats.losses,
            "draws": stats.draws,
            "ema_win_rate": round(stats.ema_win_rate, 4),
            "cumulative_win_rate": round(
                stats.wins / stats.games if stats.games > 0 else 0.5, 4
            ),
            "win_rate": round(stats.ema_win_rate, 4),
            "rating": round(stats.ema_win_rate, 4),
        }

    def eviction_sort_key(self, slot_id: int, *, episode: int) -> Tuple[float, int]:
        return (self.rating(slot_id), int(episode))

    def get_stats(self, slot_id: int) -> Optional[EmaSlotStats]:
        return self._stats.get(int(slot_id))


@dataclass
class _TrueSkillState:
    mu: float
    sigma: float
    games: int = 0


class TrueSkillRating(RatingSystem):
    """Bayesian multi-player TrueSkill from full ``GameRecord`` placements."""

    def __init__(
        self,
        *,
        mu: float = 25.0,
        sigma: float = 25.0 / 3.0,
        beta: float = 25.0 / 6.0,
        tau: float = 25.0 / 300.0,
        draw_probability: float = 0.0,
    ) -> None:
        try:
            import trueskill as ts
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TrueSkill rating requires the 'trueskill' package. "
                "Install with: pip install trueskill"
            ) from exc

        self._ts = ts
        self._env = ts.TrueSkill(
            mu=float(mu),
            sigma=float(sigma),
            beta=float(beta),
            tau=float(tau),
            draw_probability=float(draw_probability),
        )
        self._ratings: Dict[int, _TrueSkillState] = {}

    def register(self, slot_id: int) -> None:
        sid = int(slot_id)
        if sid not in self._ratings:
            self._ratings[sid] = _TrueSkillState(
                mu=float(self._env.mu),
                sigma=float(self._env.sigma),
            )

    def remove(self, slot_id: int) -> None:
        self._ratings.pop(int(slot_id), None)

    def update(self, record: GameRecord) -> bool:
        participants = record.participants
        if len(participants) < 2:
            return False

        for p in participants:
            self.register(p.slot_id)

        rating_groups: List[Tuple[Any, ...]] = []
        ranks: List[int] = []
        slot_ids: List[int] = []
        for p in participants:
            state = self._ratings[int(p.slot_id)]
            rating_groups.append(
                (self._env.create_rating(state.mu, state.sigma),)
            )
            ranks.append(int(p.placement) - 1)
            slot_ids.append(int(p.slot_id))

        new_groups = self._env.rate(rating_groups, ranks=ranks)

        merged: Dict[int, List[Any]] = defaultdict(list)
        for sid, group in zip(slot_ids, new_groups):
            merged[sid].append(group[0])

        for sid, ratings in merged.items():
            mu = sum(r.mu for r in ratings) / len(ratings)
            sigma = sum(r.sigma for r in ratings) / len(ratings)
            state = self._ratings[sid]
            state.mu = float(mu)
            state.sigma = float(sigma)
            state.games += 1
        return True

    def _win_prob(self, winner: _TrueSkillState, loser: _TrueSkillState) -> float:
        delta = winner.mu - loser.mu
        denom = math.sqrt(
            2.0 * float(self._env.beta) ** 2 + winner.sigma ** 2 + loser.sigma ** 2
        )
        if denom <= 0.0:
            return 0.5
        return float(self._ts.global_env().cdf(delta / denom))

    def get_mu_sigma(self, slot_id: int) -> Tuple[float, float]:
        state = self._ratings.get(int(slot_id))
        if state is None:
            return float(self._env.mu), float(self._env.sigma)
        return float(state.mu), float(state.sigma)

    def match_quality_vs_current(self, slot_id: int) -> float:
        opp = self.get_mu_sigma(slot_id)
        learner = self.get_mu_sigma(SLOT_CURRENT)
        return trueskill_match_quality(
            learner[0],
            learner[1],
            opp[0],
            opp[1],
            beta=float(self._env.beta),
        )

    def rating(self, slot_id: int) -> float:
        """Estimated P(this slot beats the current learner)."""
        opp = self._ratings.get(int(slot_id))
        learner = self._ratings.get(SLOT_CURRENT)
        if opp is None:
            return 0.5
        if learner is None:
            return self._win_prob(opp, _TrueSkillState(mu=self._env.mu, sigma=self._env.sigma))
        return self._win_prob(opp, learner)

    def summary(self, slot_id: int) -> Dict[str, Any]:
        state = self._ratings.get(int(slot_id))
        if state is None:
            state = _TrueSkillState(mu=float(self._env.mu), sigma=float(self._env.sigma))
        win_rate = self.rating(slot_id)
        return {
            "games": state.games,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "mu": round(state.mu, 4),
            "sigma": round(state.sigma, 4),
            "ema_win_rate": round(win_rate, 4),
            "cumulative_win_rate": round(win_rate, 4),
            "win_rate": round(win_rate, 4),
            "rating": round(win_rate, 4),
        }

    def eviction_sort_key(self, slot_id: int, *, episode: int) -> Tuple[float, int]:
        state = self._ratings.get(int(slot_id))
        mu = state.mu if state is not None else float(self._env.mu)
        return (mu, int(episode))


def make_rating_system(
    kind: str,
    *,
    ema_beta: float = 0.05,
    trueskill: Optional[Dict[str, Any]] = None,
) -> RatingSystem:
    key = (kind or "ema").strip().lower()
    if key in ("ema", "ema_pairwise", "pairwise"):
        return EmaPairwiseRating(ema_beta=ema_beta)
    if key == "trueskill":
        params = dict(trueskill or {})
        return TrueSkillRating(**params)
    raise ValueError(f"Unknown rating system: {kind!r} (expected 'ema' or 'trueskill')")


__all__ = [
    "EmaPairwiseRating",
    "EmaSlotStats",
    "RatingSystem",
    "TrueSkillRating",
    "make_rating_system",
    "trueskill_match_quality",
]

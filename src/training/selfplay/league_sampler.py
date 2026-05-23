"""Opponent sampling strategies for distributed / league training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .game_record import SLOT_CURRENT, build_scripted_slot_map
from .league_config import LeagueSamplerConfig
from .league_policy import (
    OpponentKind,
    decide_opponent_kind,
    pfsp_sample,
    sample_scripted_key,
    trueskill_quality_sample,
)
from .rating_system import trueskill_match_quality

DEFAULT_TRUESKILL_MU = 25.0
DEFAULT_TRUESKILL_SIGMA = 25.0 / 3.0


@dataclass
class LeagueSyncState:
    frozen_pool: Dict[int, bytes]
    win_rates: Dict[int, float]
    rating_kind: str = "ema"
    trueskill: Dict[int, Tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class OpponentSample:
    slot_id: int
    scripted_key: Optional[str] = None


def _pool_slot_ids(
    *,
    scripted_slot_ids: Dict[str, int],
    frozen_pool: Dict[int, bytes],
) -> List[int]:
    ids = sorted(set(scripted_slot_ids.values()))
    ids.extend(sorted(frozen_pool.keys()))
    return ids


def _learner_mu_sigma(sync: LeagueSyncState) -> Tuple[float, float]:
    return sync.trueskill.get(SLOT_CURRENT, (DEFAULT_TRUESKILL_MU, DEFAULT_TRUESKILL_SIGMA))


def _opp_mu_sigma(sync: LeagueSyncState, slot_id: int) -> Tuple[float, float]:
    return sync.trueskill.get(int(slot_id), (DEFAULT_TRUESKILL_MU, DEFAULT_TRUESKILL_SIGMA))


def _pick_by_match_quality(
    pool_ids: Sequence[int],
    *,
    sync: LeagueSyncState,
    game_rng: Any,
) -> int:
    learner = _learner_mu_sigma(sync)
    qualities: List[float] = []
    for sid in pool_ids:
        opp = _opp_mu_sigma(sync, int(sid))
        qualities.append(trueskill_match_quality(learner[0], learner[1], opp[0], opp[1]))
    return trueskill_quality_sample(list(pool_ids), qualities, game_rng)


def _pick_from_pool(
    pool_ids: Sequence[int],
    *,
    sync: LeagueSyncState,
    sampler: LeagueSamplerConfig,
    game_rng: Any,
) -> int:
    if not pool_ids:
        raise ValueError("empty opponent pool")
    if sampler.uses_match_quality(sync.rating_kind):
        return _pick_by_match_quality(pool_ids, sync=sync, game_rng=game_rng)
    rates = [sync.win_rates.get(int(sid), 0.5) for sid in pool_ids]
    return pfsp_sample(list(pool_ids), rates, game_rng)


def sample_league_opponent(
    *,
    game_rng: Any,
    use_self_play: bool,
    sync: LeagueSyncState,
    sampler: LeagueSamplerConfig,
    scripted_distribution: Dict[str, float],
    scripted_slot_ids: Dict[str, int],
    slot_id_to_scripted_key: Dict[int, str],
    frozen_pool: Dict[int, bytes],
) -> OpponentSample:
    """Choose a league slot (and optional scripted bot key) for one opponent seat."""
    if not use_self_play:
        key = sample_scripted_key(scripted_distribution, game_rng)
        return OpponentSample(
            slot_id=scripted_slot_ids[key],
            scripted_key=key,
        )

    if sampler.is_unified_pool():
        if game_rng.random() < sampler.current_self_fraction:
            return OpponentSample(slot_id=SLOT_CURRENT)
        pool_ids = _pool_slot_ids(
            scripted_slot_ids=scripted_slot_ids,
            frozen_pool=frozen_pool,
        )
        slot_id = _pick_from_pool(
            pool_ids,
            sync=sync,
            sampler=sampler,
            game_rng=game_rng,
        )
        return OpponentSample(
            slot_id=slot_id,
            scripted_key=slot_id_to_scripted_key.get(slot_id),
        )

    kind = decide_opponent_kind(
        game_rng.random(),
        current_fraction=sampler.current_self_fraction,
        past_fraction=sampler.past_self_fraction,
        frozen_nonempty=bool(frozen_pool),
        has_current_agent=True,
    )
    if kind == OpponentKind.SCRIPTED:
        key = sample_scripted_key(scripted_distribution, game_rng)
        return OpponentSample(
            slot_id=scripted_slot_ids[key],
            scripted_key=key,
        )
    if kind == OpponentKind.FROZEN:
        slot_ids = list(frozen_pool.keys())
        slot_id = _pick_from_pool(
            slot_ids,
            sync=sync,
            sampler=sampler,
            game_rng=game_rng,
        )
        return OpponentSample(slot_id=int(slot_id))
    return OpponentSample(slot_id=SLOT_CURRENT)


__all__ = [
    "LeagueSyncState",
    "OpponentSample",
    "sample_league_opponent",
]

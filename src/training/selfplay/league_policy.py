"""Pure opponent-league sampling policy (no torch / multiprocessing)."""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Optional, Sequence


class OpponentKind(str, Enum):
    CURRENT = "current"
    FROZEN = "frozen"
    SCRIPTED = "scripted"


def self_play_enabled(*, episode: int, start_episode: int, has_self_play_config: bool) -> bool:
    return has_self_play_config and episode >= start_episode


def decide_opponent_kind(
    roll: float,
    *,
    current_fraction: float,
    past_fraction: float,
    frozen_nonempty: bool,
    has_current_agent: bool,
) -> OpponentKind:
    """Match ``OpponentPool.sample_opponent`` branch order and thresholds."""
    threshold_current = current_fraction
    threshold_past = current_fraction + past_fraction

    if has_current_agent and (roll < threshold_current or not frozen_nonempty):
        return OpponentKind.CURRENT
    if roll < threshold_past and frozen_nonempty:
        return OpponentKind.FROZEN
    return OpponentKind.SCRIPTED


def pfsp_weights(ema_rates: Sequence[float], *, eps: float = 1e-2) -> List[float]:
    return [max(float(p), eps) ** 2 for p in ema_rates]


def pfsp_sample(
    slot_ids: Sequence[int],
    ema_rates: Sequence[float],
    rng: random.Random,
    *,
    eps: float = 1e-2,
) -> int:
    if not slot_ids:
        raise ValueError("pfsp_sample requires at least one slot_id")
    weights = pfsp_weights(ema_rates, eps=eps)
    total = sum(weights)
    if total <= 0:
        return rng.choice(list(slot_ids))
    r = rng.random() * total
    acc = 0.0
    ids = list(slot_ids)
    for slot_id, weight in zip(ids, weights):
        acc += weight
        if r <= acc:
            return slot_id
    return ids[-1]


def sample_scripted_key(distribution: Dict[str, float], rng: random.Random) -> str:
    keys = list(distribution.keys())
    weights = [float(distribution[k]) for k in keys]
    total = sum(weights)
    if total <= 0:
        return keys[-1]
    r = rng.random() * total
    acc = 0.0
    for key, weight in zip(keys, weights):
        acc += weight
        if r <= acc:
            return key
    return keys[-1]


def selection_probabilities(ema_rates: Sequence[float], *, eps: float = 1e-2) -> List[float]:
    weights = pfsp_weights(ema_rates, eps=eps)
    total = sum(weights)
    n = len(ema_rates)
    if total <= 0:
        return [1.0 / n] * n if n else []
    return [w / total for w in weights]


__all__ = [
    "OpponentKind",
    "decide_opponent_kind",
    "pfsp_sample",
    "pfsp_weights",
    "sample_scripted_key",
    "selection_probabilities",
    "self_play_enabled",
]

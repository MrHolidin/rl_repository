"""Per-seat tribe-preference vector: a hand-written stand-in for a DvD identity.

Each seat draws one component per tribe at game start, uniform in [-1, 1]. The
vector is part of the seat's observation, and the trainer pays
``coef * v[tribe]`` for every minion of that tribe the seat buys — so half the
tribes are a bonus and half a penalty, and the seat has to read its own vector
to know which build it is being asked to try this game.

Unlike the DvD identity this carries no learned population term and no
diversity objective: the vector is random, fixed for the game, and the reward
is a fixed linear function of it.

``Race.ALL`` (Amalgam and friends) is genuinely every tribe, so it scores the
mean of the vector rather than a component of its own. A tribeless minion
scores nothing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.bg_catalog.cards import Race

# One component per real tribe, in a fixed order the observation depends on.
# Race.ALL is deliberately absent — see the module docstring.
TRIBES: Tuple[Race, ...] = (
    Race.BEAST,
    Race.DEMON,
    Race.MECHANICAL,
    Race.MURLOC,
    Race.DRAGON,
    Race.PIRATE,
    Race.ELEMENTAL,
)
NUM_TRIBES = len(TRIBES)
_TRIBE_INDEX: Dict[Race, int] = {r: i for i, r in enumerate(TRIBES)}


def draw_tribe_pref(rng: Any) -> Tuple[float, ...]:
    """One vector, uniform in [-1, 1] per tribe.

    Accepts either a ``random.Random`` or a numpy generator so the caller can
    keep using whichever RNG owns its seed.
    """
    uniform = getattr(rng, "uniform", None)
    if uniform is None:
        raise TypeError(f"rng {type(rng).__name__} has no uniform()")
    vals = uniform(-1.0, 1.0, NUM_TRIBES)
    if np.isscalar(vals):  # random.Random.uniform takes no size
        return tuple(float(rng.uniform(-1.0, 1.0)) for _ in range(NUM_TRIBES))
    return tuple(float(v) for v in np.asarray(vals).reshape(-1))


def tribe_index(race: Optional[Race]) -> Optional[int]:
    """Index into the vector, or None for ALL / tribeless."""
    if race is None:
        return None
    return _TRIBE_INDEX.get(race)


def pref_value(pref: Sequence[float], race: Optional[Race]) -> float:
    """Score one minion's tribe against the vector.

    ``Race.ALL`` takes the mean (it counts as every tribe), a tribeless minion
    takes 0, and an empty/short vector reads as "no preference configured".
    """
    if not pref or len(pref) < NUM_TRIBES:
        return 0.0
    if race is Race.ALL:
        return float(sum(pref[:NUM_TRIBES])) / NUM_TRIBES
    idx = tribe_index(race)
    if idx is None:
        return 0.0
    return float(pref[idx])


def pref_reward_for_counts(
    pref: Sequence[float],
    counts: Dict[Optional[Race], int],
) -> float:
    """Total preference score for a ``{race: count}`` bag of purchases."""
    if not pref or not counts:
        return 0.0
    return sum(pref_value(pref, race) * int(n) for race, n in counts.items())


# Stack form: a tribe's score grows faster than its count, so five of one tribe
# beats one of five. The cap keeps the top end finite; measured on real boards
# the biggest single-tribe stack averages 2.6 and reaches 5 rarely, so the cap
# is a guard rather than an active constraint.
STACK_POWER = 1.5
STACK_CAP = 5


def pref_stack_reward(
    pref: Sequence[float],
    counts: Dict[Optional[Race], int],
) -> float:
    """Superlinear preference score: ``sum_x min(cap, n_x) ** power * v[x]``.

    Counts are per tribe, not per minion, which is what makes concentration pay:
    3+2+2 scores 10.9 where 5+2 scores 14.0. ``Race.ALL`` counts as its own bag
    (it is every tribe, so ``pref_value`` already averages the vector for it).
    """
    if not pref or not counts:
        return 0.0
    total = 0.0
    for race, n in counts.items():
        n = int(n)
        if n <= 0:
            continue
        total += (min(STACK_CAP, n) ** STACK_POWER) * pref_value(pref, race)
    return total


__all__ = [
    "NUM_TRIBES",
    "STACK_CAP",
    "STACK_POWER",
    "TRIBES",
    "draw_tribe_pref",
    "pref_reward_for_counts",
    "pref_stack_reward",
    "pref_value",
    "tribe_index",
]

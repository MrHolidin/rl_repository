"""Board strength, and the relative-strength targets the auxiliary head predicts.

Strength is the plain sum of ``attack + max_health`` over a board. It is a
deliberately *unweighted* measure: any hand-tuned weighting of keywords or
tribes would be an opinion about what makes a board good, and the point of the
auxiliary task is to make the encoder learn that, not to be told.

The consequence is that the measure cannot tell a 20/1 from a 10/10 with taunt
and divine shield. That is acceptable here only because the head is *auxiliary*:
its gradient shapes the representation and never enters the objective. Adding
the same quantity to the reward would be the battle-shaping mistake again — that
run learned "do not lose health", i.e. "do not invest", and stalled two tiers
below the baseline.

Targets are **ratios**, ``strength(t + h) / strength(t)``, not differences.
A ratio is scale-free: going 40 -> 60 early and 200 -> 300 late are the same
event (a 1.5x round), which is what the policy can actually control, whereas the
absolute delta grows with the game and would make late rounds dominate the loss.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

# Ratio bounds. Below 0.1 or above 5.0 the exact value stops carrying a
# decision-relevant distinction (both are "the board changed completely"), and
# leaving the target unbounded would let rare blowouts dominate the loss.
STRENGTH_RATIO_LO = 0.1
STRENGTH_RATIO_HI = 5.0

# Horizons in *combat rounds*, not model decisions. A seat takes ~6 decisions
# per round (median; p90 12), so h=4 is roughly 24 decisions and a quarter of a
# lobby. Measured on a trained policy, ratio predictability rises with horizon
# (corr 0.46 / 0.55 / 0.65 at h=1/2/4): one round of growth is mostly shop luck,
# while four rounds are set by tier, economy and tribe commitment. That is the
# tempo-vs-scaling trade-off, and it is what this head exists to represent —
# a level-up costs strength at h=1 and pays it back by h=4.
DEFAULT_STRENGTH_HORIZONS = (1, 2, 4)

# Guards a division by an empty board (a seat can be wiped to nothing).
_MIN_STRENGTH = 1.0


def board_strength(board: Optional[Iterable]) -> float:
    """``sum(attack + max_health)`` over the minions on ``board``."""
    if not board:
        return 0.0
    total = 0.0
    for m in board:
        if m is None:
            continue
        total += float(m.raw_attack) + float(m.max_health)
    return total


def strength_ratio(before: float, after: float) -> float:
    """``after / before``, clamped to the representable band."""
    denom = max(float(before), _MIN_STRENGTH)
    ratio = max(float(after), 0.0) / denom
    return min(max(ratio, STRENGTH_RATIO_LO), STRENGTH_RATIO_HI)


def horizons_from_config(value: Optional[Sequence[int]]) -> tuple:
    """Normalise a configured horizon list; empty/None falls back to the default."""
    if not value:
        return tuple(DEFAULT_STRENGTH_HORIZONS)
    out = tuple(sorted({int(h) for h in value if int(h) > 0}))
    if not out:
        raise ValueError("strength horizons must contain at least one positive round count")
    return out


__all__ = [
    "STRENGTH_RATIO_LO",
    "STRENGTH_RATIO_HI",
    "DEFAULT_STRENGTH_HORIZONS",
    "board_strength",
    "strength_ratio",
    "horizons_from_config",
]

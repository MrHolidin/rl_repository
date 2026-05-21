"""Shop-phase turn order within a recruitment round."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sample_shop_turn_order(rng: np.random.Generator, n: int = 2) -> Tuple[int, ...]:
    """Random permutation of player indices ``0 .. n-1``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return tuple(int(x) for x in rng.permutation(n))


__all__ = ["sample_shop_turn_order"]

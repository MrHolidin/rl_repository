"""Behavioural descriptor of a player's board, for DvD diversity pressure.

A low-dimensional, label-free summary of *how* a board was built — tribe mix,
tech level, how wide / how big — computed directly from the live game state
(no hand-named archetypes). The DvD agent accumulates an EMA of this per
population identity and rewards an identity for ending up far from the others
in this space.

Kept deliberately small and board-derivable: anything richer (spells cast,
triples over the game) would need trajectory bookkeeping; the final board
already separates mech / elemental / murloc / beast and tempo-vs-scaling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.envs.bglike.actions import BOARD_SIZE
from src.envs.minibg.obs import RACE_ONEHOT_DIM, _RACE_ORDER

__all__ = ["BOARD_DESCRIPTOR_DIM", "board_descriptor"]

# Layout: [tribe fractions | tier | fill | mean_atk | mean_hp | golden_frac]
_EXTRA = 5
BOARD_DESCRIPTOR_DIM = RACE_ONEHOT_DIM + _EXTRA

_MAX_TIER = 6.0
_STAT_NORM = 20.0  # rough per-minion stat scale → keeps entries ~unit


def board_descriptor(state: Any, seat: int, *, board_size: int = BOARD_SIZE) -> np.ndarray:
    """``(BOARD_DESCRIPTOR_DIM,)`` float32 summary of ``state.players[seat]``'s board."""
    player = state.players[seat]
    board = list(player.board)
    v = np.zeros(BOARD_DESCRIPTOR_DIM, dtype=np.float32)
    n = len(board)

    if n > 0:
        for m in board:
            try:
                idx = _RACE_ORDER.index(m.race)
            except ValueError:
                idx = 0  # unknown race → the "None" bucket
            v[idx] += 1.0
        v[:RACE_ONEHOT_DIM] /= float(n)

    off = RACE_ONEHOT_DIM
    v[off + 0] = float(getattr(player, "tavern_tier", 1)) / _MAX_TIER
    v[off + 1] = float(n) / float(board_size)
    if n > 0:
        v[off + 2] = float(sum(int(m.raw_attack) for m in board)) / (n * _STAT_NORM)
        v[off + 3] = float(sum(int(m.max_health) for m in board)) / (n * _STAT_NORM)
        v[off + 4] = float(sum(1 for m in board if getattr(m, "is_golden", False))) / float(n)
    return v

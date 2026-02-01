"""Othello game state dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class OthelloState:
    board: np.ndarray
    current_player_index: int
    winner: Optional[int]
    done: bool
    last_move: Optional[Tuple[int, int]] = None

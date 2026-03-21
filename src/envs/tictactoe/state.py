"""TicTacToe game state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TicTacToeState:
    board: np.ndarray  # (3, 3), values: 0 empty, 1 X, -1 O
    current_player_index: int  # 0 = X (+1), 1 = O (-1)
    winner: Optional[int]  # 1, -1, 0 (draw), or None
    done: bool

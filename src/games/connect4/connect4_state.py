from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Connect4State:
    board: np.ndarray
    current_player_index: int
    winner: Optional[int]
    done: bool
    last_move: Optional[Tuple[int, int]] = None


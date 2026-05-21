"""Combat pairing and ghost snapshot types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.bg_core.minion import Minion

# Opponent id recorded in recent-opponent history for a Kel'Thuzad (ghost) fight.
GHOST_OPPONENT_ID = -1

__all__ = [
    "GHOST_OPPONENT_ID",
    "EliminatedSnapshot",
    "CombatMatch",
]


@dataclass(frozen=True)
class EliminatedSnapshot:
    """Board snapshot from an eliminated player (ghost target pool)."""

    seat: int
    last_board: Tuple[Minion, ...]
    tavern_tier: int
    eliminated_combat_round: int


@dataclass(frozen=True)
class CombatMatch:
    """One combat pairing for the round (live vs live or live vs ghost)."""

    a: int
    b: Optional[int]
    ghost: Optional[EliminatedSnapshot] = None

    @property
    def is_ghost(self) -> bool:
        return self.b is None and self.ghost is not None

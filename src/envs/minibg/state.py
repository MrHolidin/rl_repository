from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.bg_core.minion import Race
from src.bg_lobby.shared_pool import SharedCardPool
from src.bg_lobby.player import (
    CNT_ACTIVE_SHOP_TRIBES,
    CasterKind,
    CasterRef,
    Minion,
    PendingChoice,
    PendingChoiceKind,
    PlayerPhase,
    PlayerState,
    ROTATION_SHOP_TRIBES,
)

__all__ = [
    "Minion",
    "PlayerState",
    "MiniBGState",
    "PlayerPhase",
    "Race",
    "ROTATION_SHOP_TRIBES",
    "CNT_ACTIVE_SHOP_TRIBES",
    "PendingChoiceKind",
    "PendingChoice",
    "CasterKind",
    "CasterRef",
]


@dataclass
class MiniBGState:
    players: Tuple[PlayerState, PlayerState]
    round_number: int
    current_player_index: int
    initiative_player: int
    winner: Optional[int]
    done: bool
    shop_excluded_race: Optional[Race] = None
    shop_turn_order: Tuple[int, int] = (0, 1)
    shared_pool: Optional[SharedCardPool] = None

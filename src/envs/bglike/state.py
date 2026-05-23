"""BGLike lobby state (8 players, shared pool)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.bg_core.minion import Race
from src.bg_lobby.match_types import CombatMatch, EliminatedSnapshot
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
from src.bg_lobby.shared_pool import SharedCardPool

from .actions import NUM_PLAYERS

__all__ = [
    "BGLikeState",
    "NUM_PLAYERS",
    "Minion",
    "PlayerState",
    "PlayerPhase",
    "Race",
    "ROTATION_SHOP_TRIBES",
    "CNT_ACTIVE_SHOP_TRIBES",
    "PendingChoice",
    "PendingChoiceKind",
    "CasterKind",
    "CasterRef",
]


@dataclass
class BGLikeState:
    players: Tuple[PlayerState, ...]
    alive: Tuple[int, ...]
    round_number: int
    combat_round: int
    full_lobby_cycle_round: int
    current_player_index: int
    shop_turn_order: Tuple[int, ...]
    recent_opponents: Tuple[Tuple[int, ...], ...]
    eliminated: Tuple[EliminatedSnapshot, ...]
    pairings: Tuple[CombatMatch, ...]
    initiative_player: int
    winner: Optional[int]
    done: bool
    shop_excluded_race: Optional[Race] = None
    shared_pool: Optional[SharedCardPool] = None
    patch_build: Optional[int] = None

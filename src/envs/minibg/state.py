from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

from src.bg_core.minion import Minion, Race

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


class PlayerPhase(IntEnum):
    SHOP = 0
    DONE = 1


class PendingChoiceKind(IntEnum):
    DISCOVER_MURLOC = 0
    ADAPT = 1
    TRIPLE_REWARD_DISCOVER = 2


@dataclass
class PendingChoice:
    """Player must pick one of ``options`` (three card_ids or three adapt keys)."""

    kind: PendingChoiceKind
    options: Tuple[str, str, str]
    extra_modals_after: int


class CasterKind(IntEnum):
    NONE = 0
    BOARD = 1
    HAND = 2
    HERO = 3


@dataclass(frozen=True)
class CasterRef:
    """Who triggered a shop effect (replay / RL bookkeeping)."""

    kind: CasterKind
    board_idx: Optional[int] = None
    hand_idx: Optional[int] = None


ROTATION_SHOP_TRIBES: Tuple[Race, Race, Race, Race] = (
    Race.BEAST,
    Race.DEMON,
    Race.MECHANICAL,
    Race.MURLOC,
)
CNT_ACTIVE_SHOP_TRIBES = len(ROTATION_SHOP_TRIBES) - 1


@dataclass
class PlayerState:
    health: int
    gold: int
    tavern_tier: int
    next_tier_up_cost: int
    board: List[Minion]
    shop: List[Optional[Minion]]
    hand: List[Optional[Minion]]
    phase: PlayerPhase
    shop_actions_used: int
    shop_freeze_next_round: bool = False
    hero_damage_taken_total: int = 0
    pogo_hoppers_played: int = 0
    pending_choice: Optional["PendingChoice"] = None
    placed_minion_board_index: Optional[int] = None
    placed_minion_pending_after: Optional["Minion"] = None
    triple_reward_discover_pending: bool = False
    """True when triple-reward discover spell could not fit in hand; grant when a slot opens."""
    triple_reward_spell_tier: int = 0
    """Discover pool tier for a pending spell grant (``tavern_tier + 1`` at forge time)."""

    @property
    def shopping_finished(self) -> bool:
        return self.phase == PlayerPhase.DONE


@dataclass
class MiniBGState:
    players: Tuple[PlayerState, PlayerState]
    round_number: int
    current_player_index: int
    initiative_player: int
    winner: Optional[int]
    done: bool
    shop_excluded_race: Optional[Race] = None

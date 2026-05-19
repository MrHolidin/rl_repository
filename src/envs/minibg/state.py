from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List, Optional, Tuple

from src.bg_core.minion import Minion, Race

from src.bg_core.effects import Ability, Keyword

# Re-export core minion types for existing ``from .state import Minion, Race`` paths.
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


# Tavern pool: per episode one of these is excluded (see ``MiniBGState.shop_excluded_race``).
# Neutrals (`race is None`) and ``Race.ALL`` minions are never excluded.
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
    # Current tier-up gold price; drops by 1 each new round (after battle) until bought.
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
    # Replay snapshot; kept aligned with placed_minion_pending_after when that is set.
    placed_minion_board_index: Optional[int] = None
    # Same Minion instance as on board until ON_PLACE modals finish and AFTER_PLACE fires.
    placed_minion_pending_after: Optional["Minion"] = None
    # Queued after playing a golden from a triple when a murloc/adapt modal is active first.
    triple_reward_discover_pending: bool = False

    @property
    def shopping_finished(self) -> bool:
        """``True`` once the player finished shopping for this round (submitted board)."""
        return self.phase == PlayerPhase.DONE


@dataclass
class MiniBGState:
    players: Tuple[PlayerState, PlayerState]
    round_number: int
    current_player_index: int
    initiative_player: int
    winner: Optional[int]
    done: bool
    # ``None`` = all four rotation tribes in the tavern (no exclusion).
    shop_excluded_race: Optional[Race] = None

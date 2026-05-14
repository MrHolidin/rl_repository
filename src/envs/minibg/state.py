from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import List, Optional, Tuple

from .effects import Ability, Keyword


class PlayerPhase(IntEnum):
    SHOP = 0
    ORDER = 1
    DONE = 2


class PendingChoiceKind(IntEnum):
    DISCOVER_MURLOC = 0
    ADAPT = 1


@dataclass
class PendingChoice:
    """Player must pick one of ``options`` (three card_ids or three adapt keys)."""

    kind: PendingChoiceKind
    options: Tuple[str, str, str]
    extra_modals_after: int


class Race(Enum):
    BEAST = auto()
    DEMON = auto()
    MECHANICAL = auto()
    MURLOC = auto()
    ALL = auto()


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
class Minion:
    card_id: str
    base_attack: int
    base_health: int
    tier: int
    # Catalog / card display name — for replays, logs, heuristics (not used in obs).
    name: str = ""
    bonus_attack: int = 0
    bonus_health: int = 0
    race: Optional[Race] = None
    keywords: frozenset[Keyword] = field(default_factory=frozenset)
    granted_keywords: frozenset[Keyword] = field(default_factory=frozenset)
    abilities: Tuple[Ability, ...] = ()
    has_shield: bool = False
    is_token: bool = False
    is_golden: bool = False
    dbf_id: Optional[int] = None

    @property
    def all_keywords(self) -> frozenset[Keyword]:
        return self.keywords | self.granted_keywords

    @property
    def max_health(self) -> int:
        return self.base_health + self.bonus_health

    @property
    def raw_attack(self) -> int:
        return self.base_attack + self.bonus_attack


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
    hero_damage_taken_total: int = 0
    pogo_hoppers_played: int = 0
    pending_choice: Optional["PendingChoice"] = None
    placed_minion_board_index: Optional[int] = None

    @property
    def shopping_finished(self) -> bool:
        """``True`` once the player finished the recruitment order phase (submitted board)."""
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

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

from .effects import Ability, Keyword


class Race(Enum):
    BEAST = auto()
    DEMON = auto()
    MECHANICAL = auto()
    MURLOC = auto()
    ALL = auto()


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
    """Set when forged from three non-golden copies (golden minion only, not the reward spell)."""
    from_triple_merge: bool = False
    is_triple_reward_spell: bool = False
    """Hand spell: PLACE consumes it and opens triple-reward discover (see ``triple_discover_tier``)."""
    triple_discover_tier: int = 0
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


__all__ = ["Race", "Minion"]

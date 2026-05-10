from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .effects import Ability, Keyword


@dataclass
class Minion:
    card_id: str
    base_attack: int
    base_health: int
    tier: int
    bonus_attack: int = 0
    bonus_health: int = 0
    keywords: frozenset[Keyword] = field(default_factory=frozenset)
    abilities: Tuple[Ability, ...] = ()
    has_shield: bool = False
    is_token: bool = False

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
    board: List[Minion]
    shop: List[Optional[Minion]]
    shopping_finished: bool
    shop_actions_used: int


@dataclass
class MiniBGState:
    players: Tuple[PlayerState, PlayerState]
    round_number: int
    current_player_index: int
    initiative_player: int
    winner: Optional[int]
    done: bool


__all__ = ["Minion", "PlayerState", "MiniBGState"]

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Union


class Keyword(Enum):
    TAUNT = auto()
    SHIELD = auto()


class Trigger(Enum):
    ON_BUY = auto()
    ON_DEATH = auto()
    ON_TURN_END = auto()
    AURA = auto()


@dataclass(frozen=True)
class SummonEffect:
    token_id: str


@dataclass(frozen=True)
class BuffRandomFriendly:
    attack: int
    health: int
    exclude_self: bool = True


@dataclass(frozen=True)
class StatAura:
    attack: int = 0
    health: int = 0


Effect = Union[SummonEffect, BuffRandomFriendly, StatAura]


@dataclass(frozen=True)
class Ability:
    trigger: Trigger
    effect: Effect


__all__ = [
    "Keyword",
    "Trigger",
    "SummonEffect",
    "BuffRandomFriendly",
    "StatAura",
    "Effect",
    "Ability",
]

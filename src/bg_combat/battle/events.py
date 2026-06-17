"""Combat event dataclasses (the ``BattleEvent`` union processed by the engine)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class BeginAttackExchange:
    """Attacker and defender sides for this strike (board indices, not minion refs)."""

    attacker_side_idx: int
    defender_side_idx: int


@dataclass(frozen=True)
class ShieldLost:
    victim_side_idx: int
    victim_instance_id: int


@dataclass(frozen=True)
class DamageDealt:
    victim_side_idx: int
    victim_instance_id: int
    source_instance_id: int
    hp_loss: int
    set_hp_to_zero_from_poison: bool


@dataclass(frozen=True)
class Overkill:
    victim_side_idx: int
    victim_instance_id: int
    attacker_side_idx: int
    attacker_instance_id: int
    excess_damage: int


@dataclass(frozen=True)
class AttackCompleted:
    """Both combatants have applied their strike damage; enqueue death batch next."""

    attacker_side_idx: int
    attacker_instance_id: int


@dataclass(frozen=True)
class MinionDied:
    side_idx: int
    instance_id: int


@dataclass(frozen=True)
class MinionSummoned:
    side_idx: int
    instance_id: int
    template_card_id: str


@dataclass(frozen=True)
class DamageStrike:
    attacker_instance_id: int
    victim_instance_id: int
    victim_side_idx: int
    amount: int


BattleEvent = Union[
    BeginAttackExchange,
    ShieldLost,
    DamageDealt,
    Overkill,
    AttackCompleted,
    MinionDied,
    MinionSummoned,
    DamageStrike,
]

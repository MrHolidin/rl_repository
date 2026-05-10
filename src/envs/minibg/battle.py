from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .actions import BOARD_SIZE, DAMAGE_CAP
from .cards import make_minion
from .effects import Keyword, StatAura, SummonEffect, Trigger
from .state import Minion


@dataclass
class BattleMinion:
    template: Minion
    current_health: int
    shield_armed: bool
    deathrattle_fired: bool = False

    @property
    def alive(self) -> bool:
        return self.current_health > 0

    @property
    def raw_attack(self) -> int:
        return self.template.raw_attack

    @property
    def tier(self) -> int:
        return self.template.tier

    @classmethod
    def from_minion(cls, minion: Minion) -> "BattleMinion":
        return cls(
            template=minion,
            current_health=minion.max_health,
            shield_armed=minion.has_shield,
        )


@dataclass
class BattleSide:
    minions: List[BattleMinion] = field(default_factory=list)
    cursor: int = 0

    def alive_minions(self) -> List[BattleMinion]:
        return [m for m in self.minions if m.alive]

    def alive_count(self) -> int:
        return sum(1 for m in self.minions if m.alive)

    def has_alive(self) -> bool:
        return any(m.alive for m in self.minions)


def _build_side(board: List[Minion]) -> BattleSide:
    return BattleSide(minions=[BattleMinion.from_minion(m) for m in board])


def attack_with_auras(minion: BattleMinion, side: BattleSide) -> int:
    bonus = 0
    for other in side.minions:
        if other is minion or not other.alive:
            continue
        for ab in other.template.abilities:
            if ab.trigger == Trigger.AURA and isinstance(ab.effect, StatAura):
                bonus += ab.effect.attack
    return minion.raw_attack + bonus


def _pick_target(
    defender_side: BattleSide,
    rng: np.random.Generator,
) -> Optional[BattleMinion]:
    alive = defender_side.alive_minions()
    if not alive:
        return None
    taunts = [m for m in alive if Keyword.TAUNT in m.template.keywords]
    pool = taunts if taunts else alive
    idx = int(rng.integers(0, len(pool)))
    return pool[idx]


def _apply_damage(target: BattleMinion, damage: int) -> None:
    if damage <= 0:
        return
    if target.shield_armed and Keyword.SHIELD in target.template.keywords:
        target.shield_armed = False
        return
    target.current_health -= damage


def _apply_on_death(dead: BattleMinion, side: BattleSide) -> None:
    for ab in dead.template.abilities:
        if ab.trigger != Trigger.ON_DEATH:
            continue
        effect = ab.effect
        if isinstance(effect, SummonEffect):
            if side.alive_count() >= BOARD_SIZE:
                continue
            token = make_minion(effect.token_id)
            side.minions.append(BattleMinion.from_minion(token))


def _resolve_deaths(side: BattleSide) -> None:
    i = 0
    while i < len(side.minions):
        m = side.minions[i]
        if not m.alive and not m.deathrattle_fired:
            m.deathrattle_fired = True
            _apply_on_death(m, side)
        i += 1


def _next_attacker(side: BattleSide) -> Optional[BattleMinion]:
    n = len(side.minions)
    if n == 0:
        return None
    start = side.cursor % n
    for offset in range(n):
        idx = (start + offset) % n
        if side.minions[idx].alive:
            side.cursor = (idx + 1) % n
            return side.minions[idx]
    return None


def _decide_first_side(
    side0: BattleSide,
    side1: BattleSide,
    p0_has_initiative: bool,
) -> int:
    n0 = side0.alive_count()
    n1 = side1.alive_count()
    if n0 > n1:
        return 0
    if n1 > n0:
        return 1
    return 0 if p0_has_initiative else 1


def _winner_damage(side: BattleSide) -> int:
    raw = sum(m.tier for m in side.minions if m.alive)
    return min(DAMAGE_CAP, raw)


def simulate_battle(
    p0_board: List[Minion],
    p1_board: List[Minion],
    p0_has_initiative: bool,
    rng: np.random.Generator,
    max_attacks: int = 200,
) -> Tuple[int, int]:
    side0 = _build_side(p0_board)
    side1 = _build_side(p1_board)

    if not side0.has_alive() and not side1.has_alive():
        return 0, 0
    if not side0.has_alive():
        return _winner_damage(side1), 0
    if not side1.has_alive():
        return 0, _winner_damage(side0)

    attacker_idx = _decide_first_side(side0, side1, p0_has_initiative)
    sides = (side0, side1)

    attacks = 0
    while side0.has_alive() and side1.has_alive() and attacks < max_attacks:
        attacker_side = sides[attacker_idx]
        defender_side = sides[1 - attacker_idx]

        attacker = _next_attacker(attacker_side)
        if attacker is None:
            break

        target = _pick_target(defender_side, rng)
        if target is None:
            break

        a_dmg = attack_with_auras(attacker, attacker_side)
        d_dmg = attack_with_auras(target, defender_side)

        _apply_damage(target, a_dmg)
        _apply_damage(attacker, d_dmg)

        _resolve_deaths(side0)
        _resolve_deaths(side1)

        attacker_idx = 1 - attacker_idx
        attacks += 1

    p0_alive = side0.has_alive()
    p1_alive = side1.has_alive()
    if p0_alive and not p1_alive:
        return 0, _winner_damage(side0)
    if p1_alive and not p0_alive:
        return _winner_damage(side1), 0
    return 0, 0


__all__ = [
    "BattleMinion",
    "BattleSide",
    "attack_with_auras",
    "simulate_battle",
]

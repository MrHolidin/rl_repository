"""Mutable combat state: minions, sides, and the per-battle runtime."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion

from .events import BattleEvent


@dataclass
class BattleMinion:
    template: Minion
    current_health: int
    shield_armed: bool
    deathrattle_fired: bool = False
    reborn_consumed: bool = False
    instance_id: int = 0
    health_aura_snapshot: int = 0

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
    def from_minion(cls, minion: Minion, instance_id: int) -> "BattleMinion":
        armed = minion.has_shield and Keyword.SHIELD in minion.all_keywords
        return cls(
            template=minion,
            current_health=minion.max_health,
            shield_armed=armed,
            instance_id=instance_id,
        )


@dataclass
class BattleSide:
    minions: List[BattleMinion] = field(default_factory=list)
    cursor: int = 0
    # Flat Attack added to every minion on this side (Deathwing's global +Attack
    # aura; set equal on both sides since it buffs all minions in the combat).
    attack_aura_all: int = 0
    # Keywords granted to this side's left-most minion at Start of Combat (Al'Akir).
    start_combat_keywords: frozenset = field(default_factory=frozenset)

    def alive_minions(self) -> List[BattleMinion]:
        return [m for m in self.minions if m.alive]

    def alive_count(self) -> int:
        return sum(1 for m in self.minions if m.alive)

    def has_alive(self) -> bool:
        return any(m.alive for m in self.minions)


@dataclass
class _CombatRuntime:
    sides: Tuple[BattleSide, BattleSide]
    rng: np.random.Generator
    combat_board_max: int
    damage_cap: int
    patch: PatchContext
    queue: Deque[BattleEvent] = field(default_factory=deque)
    next_id: int = 1
    in_death_resolution: bool = False
    death_hook: Optional[Callable[[int, str], None]] = None
    mech_hook: Optional[Callable[[int, Minion], None]] = None
    swing_damage_survivors: List[Tuple[int, int]] = field(default_factory=list)
    bonus_attack_depth: int = 0
    combat_gold: List[int] = field(default_factory=lambda: [0, 0])
    combat_hand_adds: List[List[str]] = field(default_factory=lambda: [[], []])
    kill_attribution: dict[Tuple[int, int], Tuple[int, int]] = field(
        default_factory=dict
    )
    attacker_killed_this_swing: bool = False
    health_aura_dirty: List[bool] = field(default_factory=lambda: [True, True])
    health_aura_dr_snapshot: Optional[bool] = None

    def alloc_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    def side(self, idx: int) -> BattleSide:
        return self.sides[idx]

    def find_minion(self, side_idx: int, instance_id: int) -> Optional[BattleMinion]:
        for m in self.side(side_idx).minions:
            if m.instance_id == instance_id:
                return m
        return None

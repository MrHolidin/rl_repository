"""Attack targeting: taunt / Zapp / cleave selection."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.bg_core.effects import CleaveOnAttack, Keyword, Trigger, ZappTargeting

from .state import BattleMinion, BattleSide
from .auras import attack_value


def _attacker_has_zapp_targeting(attacker: BattleMinion) -> bool:
    for ab in attacker.template.abilities:
        if ab.trigger == Trigger.AURA and isinstance(ab.effect, ZappTargeting):
            return True
    return False


def _attacker_has_cleave(attacker: BattleMinion) -> bool:
    for ab in attacker.template.abilities:
        if ab.trigger == Trigger.AURA and isinstance(ab.effect, CleaveOnAttack):
            return True
    return False


def _cleave_victim_ids_at_swing_start(
    defender_side: BattleSide, primary: BattleMinion
) -> List[int]:
    try:
        idx = defender_side.minions.index(primary)
    except ValueError:
        return []
    out: List[int] = []
    if idx > 0:
        left = defender_side.minions[idx - 1]
        if left.alive:
            out.append(left.instance_id)
    if idx + 1 < len(defender_side.minions):
        right = defender_side.minions[idx + 1]
        if right.alive:
            out.append(right.instance_id)
    return out


def _pick_target(
    defender_side: BattleSide,
    rng: np.random.Generator,
    attacker: Optional[BattleMinion] = None,
    battle_field: Optional[Tuple[BattleSide, BattleSide]] = None,
) -> Optional[BattleMinion]:
    alive = defender_side.alive_minions()
    if not alive:
        return None
    taunts = [m for m in alive if Keyword.TAUNT in m.template.all_keywords]
    pool = taunts if taunts else alive
    if attacker is not None and _attacker_has_zapp_targeting(attacker):
        atk_vals = [
            attack_value(m, defender_side, death_resolution=False, battle_field=battle_field)
            for m in pool
        ]
        mna = min(atk_vals)
        pool = [m for m, av in zip(pool, atk_vals) if av == mna]
    idx = int(rng.integers(0, len(pool)))
    return pool[idx]

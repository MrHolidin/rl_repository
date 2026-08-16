"""Attack targeting: taunt / Zapp / cleave selection."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.bg_core.effects import CleaveOnAttack, Keyword, Trigger, ZappTargeting

from .state import BattleMinion, BattleSide
from .auras import attack_value


def _attacker_has_zapp_targeting(attacker: BattleMinion) -> bool:
    for ab in attacker.abilities:
        if ab.trigger == Trigger.AURA and isinstance(ab.effect, ZappTargeting):
            return True
    return False


def _attacker_has_cleave(attacker: BattleMinion) -> bool:
    for ab in attacker.abilities:
        if ab.trigger == Trigger.AURA and isinstance(ab.effect, CleaveOnAttack):
            return True
    return False


def _cleave_victim_ids_at_swing_start(
    defender_side: BattleSide, primary: BattleMinion
) -> List[int]:
    """Splash onto the target's neighbours.

    This used to filter the minion list, because corpses stayed in it and a
    body between two minions would swallow half the splash. Bodies leave the
    board now, so the list *is* the board and the filter is gone.
    """
    alive = defender_side.minions
    idx = next((i for i, m in enumerate(alive) if m is primary), None)
    if idx is None:
        return []
    out: List[int] = []
    if idx > 0:
        out.append(alive[idx - 1].instance_id)
    if idx + 1 < len(alive):
        out.append(alive[idx + 1].instance_id)
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
    taunts = [m for m in alive if Keyword.TAUNT in m.all_keywords]
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

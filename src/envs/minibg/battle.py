"""Re-export combat from ``src.bg_combat``; inject MiniBG limits when kwargs omitted."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from src.bg_combat import battle as _core
from src.bg_core.minion import Minion

from .actions import BOARD_SIZE, COMBAT_BOARD_MAX, DAMAGE_CAP

from src.bg_combat.battle import *  # noqa: F403

# Test / shop-adjacent hooks (not part of public ``bg_combat`` API).
_CombatRuntime = _core._CombatRuntime
_fire_deathrattle = _core._fire_deathrattle
_decide_first_side = _core._decide_first_side
_pick_target = _core._pick_target
_sync_health_aura_side = _core._sync_health_aura_side


def simulate_battle(
    p0_board: List[Minion],
    p1_board: List[Minion],
    *,
    p0_has_initiative: bool,
    rng: np.random.Generator,
    **kwargs: Any,
) -> Tuple[int, int]:
    kwargs.setdefault("combat_board_max", COMBAT_BOARD_MAX)
    kwargs.setdefault("damage_cap", DAMAGE_CAP)
    kwargs.setdefault("max_board_slots", BOARD_SIZE)
    return _core.simulate_battle(
        p0_board,
        p1_board,
        p0_has_initiative=p0_has_initiative,
        rng=rng,
        **kwargs,
    )


__all__ = list(_core.__all__) + [
    "COMBAT_BOARD_MAX",
    "_CombatRuntime",
    "_fire_deathrattle",
    "_decide_first_side",
    "_pick_target",
    "_sync_health_aura_side",
]

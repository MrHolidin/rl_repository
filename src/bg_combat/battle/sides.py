"""Side construction from shop boards."""
from __future__ import annotations

from copy import copy
from typing import List

import numpy as np

from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_core.board_helpers import has_attack_threshold_ability
from src.bg_core.minion import Minion, Race

from .state import BattleMinion, battle_copy, BattleSide, _CombatRuntime


def _is_mech_template(m: Minion) -> bool:
    return m.race in (Race.MECHANICAL, Race.ALL)


def _build_side(board: List[Minion], rt: _CombatRuntime) -> BattleSide:
    out: List[BattleMinion] = []
    for m in board:
        bid = rt.alloc_id()
        out.append(battle_copy(m, bid))
        if has_attack_threshold_ability(m):
            rt.watch_attack_thresholds = True
    return BattleSide(minions=out)


def build_battle_side(board: List[Minion], *, patch: PatchContext) -> BattleSide:
    """Build a battle line with fresh instance IDs (tests, tooling)."""
    ctx = require_patch(patch, where="battle.build_battle_side")
    rt = _CombatRuntime(
        sides=(BattleSide(), BattleSide()),
        rng=np.random.default_rng(0),
        combat_board_max=10**9,
        damage_cap=10**9,
        patch=ctx,
    )
    side = _build_side(board, rt)
    # Hand back a consistent side. Attack reads a stored aura contribution now,
    # so a side that has never been synced reports everyone's printed attack --
    # a quiet wrong answer for anything holding a Dire Wolf.
    side.sync_auras()
    return side

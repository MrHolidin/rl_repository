"""Shop-phase ability preconditions."""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.bg_lobby.player import PlayerState

from .effects import Ability, Condition, ConditionKind
from .minion import Minion
from .board_helpers import minion_matches_tribe


def ability_condition_met(
    ability: Ability,
    player: PlayerState,
    board: Sequence[Minion],
    *,
    placed: Optional[Minion] = None,
) -> bool:
    cond = ability.condition
    if cond is None:
        return True
    if cond.kind == ConditionKind.OTHER_TRIBE_ON_BOARD:
        for m in board:
            if placed is not None and m is placed:
                continue
            if minion_matches_tribe(m, cond.tribe):
                return True
        return False
    if cond.kind == ConditionKind.LAST_COMBAT_WON:
        return player.last_combat_won
    return True


__all__ = ["ability_condition_met"]

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
    return condition_met(cond, player, board, placed=placed)


def condition_met(
    cond: Condition,
    player: PlayerState,
    board: Sequence[Minion],
    *,
    placed: Optional[Minion] = None,
) -> bool:
    """Whether ``cond`` holds right now, ``negate`` included.

    Split out of ``ability_condition_met`` because a condition can hang off
    something other than an ability's trigger — Tortollan Blue Shell's price is
    a condition on an effect, asked at the moment of a sale.
    """
    if cond.kind == ConditionKind.OTHER_TRIBE_ON_BOARD:
        held = any(
            m is not placed and minion_matches_tribe(m, cond.tribe) for m in board
        )
        return held != cond.negate
    if cond.kind == ConditionKind.LAST_COMBAT_WON:
        return bool(player.last_combat_won) != cond.negate
    return True


__all__ = ["ability_condition_met", "condition_met"]

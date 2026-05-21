"""Immediate ON_PLACE targeted battlecries (no player choice). Used by the game engine only."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.bg_core.effects import (
    BuffAdjacentBattlecry,
    BuffTargetFriendlyBattlecry,
    Trigger,
)
from src.bg_core.minion import Minion
from src.bg_recruitment.effect_modal import (
    _apply_buff_target,
    caster_ref_from_board_minion,
    compute_eligible_buff_target,
)
from src.bg_lobby.player import PlayerState

from .shop_triggers import ShopTriggers


def apply_targeted_on_place_battlecries(
    triggers: ShopTriggers,
    player: PlayerState,
    placed: Minion,
    *,
    rng: np.random.Generator,
    forced_buff_target: Optional[Minion] = None,
) -> None:
    """Resolve BuffTarget / BuffAdjacent instantly (random target if several eligible).

    ``forced_buff_target``: when set (e.g. RL commit), BuffTarget battlecry buffs that
    minion ``mult`` times. Adjacent battlecries ignore this and use board adjacency.
    """
    mult = ShopTriggers.battlecry_multiplier(player.board)
    caster = caster_ref_from_board_minion(player.board, placed)
    for ab in placed.abilities:
        if ab.trigger != Trigger.ON_PLACE:
            continue
        e = ab.effect
        if isinstance(e, BuffAdjacentBattlecry):
            for _ in range(mult):
                triggers.apply_buff_adjacent(player, placed, e)
        elif isinstance(e, BuffTargetFriendlyBattlecry):
            if forced_buff_target is not None:
                if forced_buff_target not in player.board:
                    continue
                target = forced_buff_target
            else:
                eligible = compute_eligible_buff_target(player.board, caster, e)
                if not eligible:
                    continue
                pick = (
                    eligible[0]
                    if len(eligible) == 1
                    else eligible[int(rng.integers(0, len(eligible)))]
                )
                target = player.board[pick]
            for _ in range(mult):
                idx = player.board.index(target)
                _apply_buff_target(player.board, idx, e)

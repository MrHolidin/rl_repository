"""Activate — the seat spends gold to fire a minion's ability, once per turn.

Every other trigger in the engine is an event: something happens and the
listeners hear it. Activate is a *move* — the seat clicks a minion in the
warband during the recruit phase and pays for it — so nothing here is called by
the rules on their own; the caller decides when it happens, the way it decides
when to buy or to roll.

Blizzard's own test cards spell the rule out: "Activate: Give a minion +1/+1
(one per turn)" and "(cost 1 gold)". Seventeen minions in the live pool carry
it, tiers 1 through 6, at 1 or 2 gold.

There is deliberately no flat action for it. The action space is frozen while
the engine catches up with the modern patches, and giving Activate an index
would move numbers every trained checkpoint is wired to. This module is what
the RL side will call once that layout moves.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import (
    Ability,
    BuffTargetFriendlyBattlecry,
    PlaceFishbaitEffect,
    Trigger,
)
from src.bg_core.minion import Minion
from src.bg_lobby.player import PlayerPhase, PlayerState

__all__ = [
    "activate_abilities",
    "activate_cost",
    "can_activate",
    "activate_minion",
    "reset_activations",
]


class ActivateNotAllowed(ValueError):
    """The seat cannot fire this Activate right now, and why."""


def activate_abilities(minion: Minion) -> Tuple[Ability, ...]:
    return tuple(ab for ab in minion.abilities if ab.trigger is Trigger.ON_ACTIVATE)


def activate_cost(minion: Minion) -> Optional[int]:
    """Gold to fire this minion's Activate, or ``None`` if it has none.

    A minion carries at most one Activate on every printing so far; if one ever
    carries two, the first is the one this reports, and the caller pays once.
    """
    abilities = activate_abilities(minion)
    if not abilities:
        return None
    return max(0, int(abilities[0].activate_cost))


def can_activate(player: PlayerState, minion: Minion) -> bool:
    cost = activate_cost(minion)
    return (
        cost is not None
        and player.phase == PlayerPhase.SHOP
        and not minion.activate_used_this_turn
        and player.gold >= cost
    )


def reset_activations(player: PlayerState) -> None:
    """Give every minion on the board its Activate back (start of turn)."""
    for minion in player.board:
        minion.activate_used_this_turn = False


def activate_minion(
    player: PlayerState,
    board_index: int,
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    shared_pool=None,
    buff_target: Optional[Minion] = None,
    shop_target_index: Optional[int] = None,
) -> None:
    """Pay for and fire the Activate on the minion at ``board_index``.

    Refuses loudly rather than doing nothing: an Activate that silently failed
    would look exactly like one whose effect the dispatcher does not implement,
    which is the confusion the shop dispatcher was already cleaned up to avoid.
    """
    if not 0 <= board_index < len(player.board):
        raise ActivateNotAllowed(f"no minion at board index {board_index}")
    minion = player.board[board_index]
    cost = activate_cost(minion)
    if cost is None:
        raise ActivateNotAllowed(f"{minion.card_id} has no Activate ability")
    if player.phase != PlayerPhase.SHOP:
        raise ActivateNotAllowed("Activate is a recruit-phase move")
    if minion.activate_used_this_turn:
        raise ActivateNotAllowed(f"{minion.card_id} already used its Activate this turn")
    if player.gold < cost:
        raise ActivateNotAllowed(
            f"{minion.card_id} costs {cost} to activate; the seat has {player.gold}"
        )

    from src.bg_recruitment.fishbait import place_fishbait
    from src.bg_recruitment.shop_triggers import ShopTriggers
    from src.bg_recruitment.targeted_battlecry import apply_targeted_buff

    player.gold -= cost
    minion.activate_used_this_turn = True
    triggers = ShopTriggers(rng, patch=patch)
    for ability in activate_abilities(minion):
        effect = ability.effect
        if isinstance(effect, PlaceFishbaitEffect):
            # "Choose a card in the Tavern. Replace it with a Fishbait" — the
            # seat names the slot, so this is a targeted move like the buff
            # below, and the shop dispatcher has no way to be told which slot.
            if shop_target_index is not None:
                place_fishbait(player, shop_target_index)
            continue
        if isinstance(effect, BuffTargetFriendlyBattlecry):
            # "Activate (1): Give another minion +3/+3" names a friendly, so it
            # goes through the same target pick a battlecry does. The shop
            # dispatcher deliberately drops these (they are _HANDLED_ELSEWHERE),
            # so routing them there would spend the gold and do nothing.
            apply_targeted_buff(
                player, minion, effect, rng=rng, forced_buff_target=buff_target
            )
            continue
        triggers.apply_shop_effect(
            player, minion, effect, placed=None, shared_pool=shared_pool
        )

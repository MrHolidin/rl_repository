"""Immediate ON_PLACE targeted battlecries (no player choice). Used by the game engine only."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.bg_core.effects import (
    BuffAdjacentBattlecry,
    BuffTargetFriendlyBattlecry,
    BuffTargetFromPiratesBoughtBattlecry,
    ConsumeFriendlyBattlecry,
    ConsumeTavernMinionEffect,
    Trigger,
)
from src.bg_core.minion import Minion
from src.bg_recruitment.effect_modal import (
    _apply_buff_target,
    caster_ref_from_board_minion,
    compute_eligible_buff_target,
)
from src.bg_lobby.player import CasterKind, CasterRef, PlayerState

from .shop_auras import shop_effective_stats
from .shop_triggers import ShopTriggers


def _apply_consume_friendly(
    player: PlayerState,
    source: Minion,
    target: Minion,
    effect: ConsumeFriendlyBattlecry,
) -> None:
    if target not in player.board:
        return
    # The stats the shop shows for the target, not its bare body: a demon standing
    # next to Mal'Ganis is eaten at +2/+2, which is what the aura displays.
    atk, hp = shop_effective_stats(player.board, target)
    source.bonus_attack += atk * effect.stat_multiplier
    source.bonus_health += hp * effect.stat_multiplier
    player.gold += effect.gold_reward
    player.board.remove(target)


def _pick_eater(
    player: PlayerState,
    placed: Minion,
    effect: ConsumeTavernMinionEffect,
    *,
    rng: np.random.Generator,
    forced: Optional[Minion],
) -> Optional[Minion]:
    """Which friendly does the eating — the seat's pick, or a random eligible."""
    if forced is not None:
        return forced if forced in player.board else None
    caster = caster_ref_from_board_minion(player.board, placed)
    eligible = compute_eligible_buff_target(
        player.board,
        caster,
        BuffTargetFriendlyBattlecry(filter_race=effect.filter_race, exclude_self=False),
    )
    if not eligible:
        return None
    pick = eligible[0] if len(eligible) == 1 else eligible[int(rng.integers(0, len(eligible)))]
    return player.board[pick]


def consume_tavern_minion(
    player: PlayerState,
    eater: Minion,
    *,
    rng: np.random.Generator,
) -> Optional[Minion]:
    """``eater`` eats a random minion off the counter and takes its stats.

    The stats are the ones the tavern shows, auras included — the same reading
    ``ConsumeFriendlyBattlecry`` takes of a minion it eats off the board.
    """
    filled = [i for i, m in enumerate(player.shop) if m is not None]
    if not filled:
        return None
    idx = filled[int(rng.integers(0, len(filled)))]
    eaten = player.shop[idx]
    attack, health = shop_effective_stats([m for m in player.shop if m is not None], eaten)
    eater.bonus_attack += attack
    eater.bonus_health += health
    player.shop[idx] = None
    return eaten


def apply_targeted_buff(
    player: PlayerState,
    source: Optional[Minion],
    effect: BuffTargetFriendlyBattlecry,
    *,
    rng: np.random.Generator,
    repeats: int = 1,
    forced_buff_target: Optional[Minion] = None,
) -> None:
    """Resolve one "give a minion +X/+Y" against a friendly the seat picks.

    Shared by the ON_PLACE battlecry below and by Activate: both name a friendly
    and differ only in what fires them. ``repeats`` is the battlecry multiplier
    at the placement site and stays 1 for an Activate, which Brann does not
    double — he doubles Battlecries, and Activate is a move, not a battlecry.

    With no ``forced_buff_target`` (no seat to ask, as in a heuristic rollout) a
    random eligible friendly takes it, which is what the placement path already
    did.
    """
    # ``source is None`` is a Tavern spell: there is no body on the board, so
    # no "self" to exclude and no slot to read adjacency from.
    caster = (
        CasterRef(CasterKind.NONE)
        if source is None
        else caster_ref_from_board_minion(player.board, source)
    )
    if forced_buff_target is not None:
        if forced_buff_target not in player.board:
            return
        target = forced_buff_target
    else:
        eligible = compute_eligible_buff_target(player.board, caster, effect)
        if not eligible:
            return
        pick = (
            eligible[0] if len(eligible) == 1 else eligible[int(rng.integers(0, len(eligible)))]
        )
        target = player.board[pick]
    for _ in range(repeats):
        _apply_buff_target(player.board, player.board.index(target), effect)


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
            apply_targeted_buff(
                player,
                placed,
                e,
                rng=rng,
                repeats=mult,
                forced_buff_target=forced_buff_target,
            )
        elif isinstance(e, BuffTargetFromPiratesBoughtBattlecry):
            n = max(0, player.pirates_bought_this_turn)
            if n == 0:
                continue
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
                m = player.board[idx]
                m.bonus_attack += e.attack_per * n
                m.bonus_health += e.health_per * n
        elif isinstance(e, ConsumeTavernMinionEffect):
            target = _pick_eater(player, placed, e, rng=rng, forced=forced_buff_target)
            if target is None:
                continue
            for _ in range(max(1, e.count)):
                consume_tavern_minion(player, target, rng=rng)
        elif isinstance(e, ConsumeFriendlyBattlecry):
            if forced_buff_target is not None:
                if forced_buff_target not in player.board:
                    continue
                target = forced_buff_target
            else:
                eligible = compute_eligible_buff_target(
                    player.board,
                    caster,
                    BuffTargetFriendlyBattlecry(
                        filter_race=e.filter_race,
                        exclude_self=e.exclude_self,
                    ),
                )
                if not eligible:
                    continue
                pick = (
                    eligible[0]
                    if len(eligible) == 1
                    else eligible[int(rng.integers(0, len(eligible)))]
                )
                target = player.board[pick]
            _apply_consume_friendly(player, placed, target, e)

"""Immediate ON_PLACE targeted battlecries (no player choice). Used by the game engine only."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.bg_core.effects import (
    BuffAdjacentBattlecry,
    BuffTargetFriendlyBattlecry,
    BuffTargetFromPiratesBoughtBattlecry,
    ConsumeFriendlyBattlecry,
    BuffTargetPerGoldSpentEffect,
    ConsumeTavernMinionEffect,
    DestroyFriendlyEffect,
    MakeFriendlyGoldenEffect,
    Trigger,
)
from src.bg_catalog.patch_catalog import minion_by_id, minion_from_tavern_record
from src.bg_core.minion import Minion
from src.bg_recruitment.effect_modal import (
    _apply_buff_target,
    caster_ref_from_board_minion,
    compute_eligible_buff_target,
)
from src.bg_lobby.player import CasterKind, CasterRef, PlayerState

from .hand_slots import first_free_hand_slot
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
    exclude_self: bool = False,
) -> Optional[Minion]:
    """Which friendly does the eating — the seat's pick, or a random eligible."""
    if forced is not None:
        if forced not in player.board:
            return None
        return None if exclude_self and forced is placed else forced
    caster = caster_ref_from_board_minion(player.board, placed)
    eligible = compute_eligible_buff_target(
        player.board,
        caster,
        BuffTargetFriendlyBattlecry(
            filter_race=effect.filter_race, exclude_self=exclude_self
        ),
    )
    if not eligible:
        return None
    pick = eligible[0] if len(eligible) == 1 else eligible[int(rng.integers(0, len(eligible)))]
    return player.board[pick]


def make_golden(target: Minion, *, patch) -> None:
    """Turn ``target`` into its Golden printing, in place.

    Not a triple: nothing merged, so no Triple Reward is owed. What the body had
    gained rides along, because the card is being upgraded rather than replaced
    — the golden printing's doubled base plus the bonuses it already carried.

    Takes no seat, so it serves a card being made Golden in hand as readily as
    one standing on the board.
    """
    from src.bg_catalog.patch_catalog import golden_upgrade_card_id

    if target.is_golden:
        return
    golden_id = golden_upgrade_card_id(
        target.card_id, patch.patch_dir / "catalog.json"
    )
    if golden_id is None:
        return
    golden = minion_from_tavern_record(
        minion_by_id(patch.patch_dir / "catalog.json")[golden_id]
    )
    target.card_id = golden.card_id
    target.base_attack = golden.base_attack
    target.base_health = golden.base_health
    target.keywords = golden.keywords
    # The golden printing's own abilities, resolved the way a triple resolves
    # them — a made Golden is the same card as a forged one.
    target.abilities = patch.triple_merge_golden_abilities(
        golden_id[:-2] if golden_id.endswith("_G") else golden_id
    )
    target.dbf_id = golden.dbf_id
    target.is_golden = True
    target.has_shield = target.has_shield or golden.has_shield


def destroy_friendly(
    player: PlayerState,
    victim: Minion,
    *,
    patch,
    get_copy: bool = True,
    triggers=None,
) -> Optional[Minion]:
    """Destroy ``victim``, optionally handing its owner a plain copy.

    Two of the three things a body normally does on the way out happen here. It
    is *counted* as a death, so "for each Eternal Knight that died this game"
    sees it, and its deathrattle fires — the modern patch prices that
    explicitly, since Plaguerunner pays double "if triggered outside combat".
    Reborn is the one that does not: that is a combat rule.

    Returns the copy when one was made, and ``None`` otherwise — including when
    the trade could not happen at all, which is why callers that pay for the
    destruction check ``victim not in player.board`` rather than this.
    """
    from src.bg_catalog.cards import make_minion

    from .game_counts import bump_died

    if victim not in player.board:
        return None
    slot = first_free_hand_slot(player) if get_copy else None
    if get_copy and slot is None:
        # No room for what the trade pays, so the trade does not happen.
        return None
    player.board.remove(victim)
    bump_died(player, victim)
    if triggers is not None:
        triggers.fire_tavern_deathrattle(victim, player)
    if not get_copy:
        return None
    # Plain: built from the template, so nothing the body had gained rides along.
    copy = make_minion(victim.card_id, patch=patch)
    player.hand[slot] = copy
    return copy


def apply_destroy_friendly(
    player: PlayerState,
    source: Optional[Minion],
    effect,
    *,
    rng: np.random.Generator,
    forced: Optional[Minion] = None,
    triggers,
) -> None:
    """Destroy a friendly the seat picked, then pay what the card prints.

    One body for four cards. The payout runs only if a body actually left the
    board: a Discover with nothing to feed it is not a Discover, and the
    Bellringer that finds no Undead gains nothing.
    """
    victim = _pick_eater(
        player,
        source,
        ConsumeTavernMinionEffect(filter_race=effect.filter_race),
        rng=rng,
        forced=forced,
        exclude_self=effect.exclude_self,
    )
    if victim is None:
        return
    if effect.grant_keyword is not None:
        victim.granted_keywords = victim.granted_keywords | {effect.grant_keyword}
    destroy_friendly(
        player,
        victim,
        patch=triggers._patch,
        get_copy=effect.get_copy,
        triggers=triggers,
    )
    if victim in player.board:
        return  # the trade did not happen (no room for the copy it pays)
    if effect.then is not None and source is not None:
        triggers.apply_shop_effect(player, source, effect.then, placed=None)


def consume_tavern_minion(
    player: PlayerState,
    eater: Minion,
    *,
    rng: np.random.Generator,
    highest_health: bool = False,
    stat_multiplier: int = 1,
) -> Optional[Minion]:
    """``eater`` eats a minion off the counter and takes its stats.

    The stats are the ones the tavern shows, auras included — the same reading
    ``ConsumeFriendlyBattlecry`` takes of a minion it eats off the board. One at
    random unless the card names the biggest.

    ``stat_multiplier`` is the Golden printing that gains *double* the stats of
    one minion: it eats no more of them, it just takes more from the one.
    """
    filled = [i for i, m in enumerate(player.shop) if m is not None]
    if not filled:
        return None
    if highest_health:
        idx = max(filled, key=lambda i: player.shop[i].max_health)
    else:
        idx = filled[int(rng.integers(0, len(filled)))]
    eaten = player.shop[idx]
    attack, health = shop_effective_stats([m for m in player.shop if m is not None], eaten)
    factor = max(1, int(stat_multiplier))
    eater.bonus_attack += attack * factor
    eater.bonus_health += health * factor
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
        elif isinstance(e, BuffTargetPerGoldSpentEffect):
            target = _pick_eater(
                player,
                placed,
                ConsumeTavernMinionEffect(filter_race=e.filter_race),
                rng=rng,
                forced=forced_buff_target,
            )
            if target is not None:
                times = 1 + max(0, player.gold_spent_this_turn)
                target.bonus_attack += e.attack * times
                target.bonus_health += e.health * times
        elif isinstance(e, MakeFriendlyGoldenEffect):
            target = _pick_eater(
                player,
                placed,
                ConsumeTavernMinionEffect(),
                rng=rng,
                forced=forced_buff_target,
            )
            if target is not None and (not e.max_tier or target.tier <= e.max_tier):
                make_golden(target, patch=triggers._patch)
        elif isinstance(e, DestroyFriendlyEffect):
            apply_destroy_friendly(
                player,
                placed,
                e,
                rng=rng,
                forced=forced_buff_target,
                triggers=triggers,
            )
        elif isinstance(e, ConsumeTavernMinionEffect):
            target = _pick_eater(player, placed, e, rng=rng, forced=forced_buff_target)
            if target is None:
                continue
            for _ in range(max(1, e.count)):
                consume_tavern_minion(
                    player, target, rng=rng, stat_multiplier=e.stat_multiplier
                )
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

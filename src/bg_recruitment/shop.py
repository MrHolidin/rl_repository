"""Tavern shop refresh (card pool selection and offer slots)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from src.bg_core.board_helpers import minion_matches_tribe
from src.bg_catalog.cards import make_minion, shop_pool_for_tier
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion, Race
from src.bg_lobby.shared_pool import SharedCardPool

from src.bg_recruitment.hand_slots import first_free_hand_slot
from src.bg_recruitment.standing_bonuses import settle_standing_bonuses
from src.bg_recruitment.tavern_spells import offer_tavern_spells
from src.envs.minibg.actions import MAX_SHOP_SLOTS, shop_offers_count
from src.bg_lobby.player import PlayerState


def effective_shop_offers_count(player: PlayerState) -> int:
    """Offer count for a player's tavern tier, including any hero extra slots
    (Ysera: +1 Dragon), capped at the max visible shop size."""
    n = shop_offers_count(player.tavern_tier)
    if player.hero is not None:
        n += player.hero.extra_shop_slots()
    return min(MAX_SHOP_SLOTS, n)


def _hero_forced_slot_tribe(player: PlayerState, slot: int) -> Optional[Race]:
    """Ysera's extra slot(s) (beyond the tier's base count) are always Dragons."""
    h = player.hero
    if h is None or h.extra_shop_slots() <= 0:
        return None
    if slot >= shop_offers_count(player.tavern_tier):
        return Race.DRAGON
    return None


def _apply_hero_shop_tribe_buff(player: PlayerState, minion: Optional[Minion]) -> None:
    """Millificent: minions of a tribe get +atk/+hp while in the Tavern."""
    h = player.hero
    if h is None or minion is None:
        return
    buff = h.shop_tribe_buff()
    if buff is not None and minion_matches_tribe(minion, buff.race):
        minion.bonus_attack += buff.attack
        minion.bonus_health += buff.health


def shop_tribe_bonus_for(player: PlayerState, race: Optional[Race]) -> int:
    if race == Race.ELEMENTAL:
        return player.shop_elemental_bonus
    return 0


def apply_shop_tribe_bonus_to_minion(minion: Minion, player: PlayerState) -> None:
    bonus = shop_tribe_bonus_for(player, minion.race)
    if bonus > 0:
        minion.bonus_attack += bonus
        minion.bonus_health += bonus


def buff_shop_minions_of_tribe(
    player: PlayerState, tribe: Race, *, attack: int, health: int
) -> None:
    for m in player.shop:
        if m is None:
            continue
        if not minion_matches_tribe(m, tribe) or m.cannot_gain_stats:
            continue
        m.bonus_attack += attack
        m.bonus_health += health


def buff_all_shop_offers(player: PlayerState, *, attack: int, health: int) -> None:
    for m in player.shop:
        if m is None or m.cannot_gain_stats:
            continue
        m.bonus_attack += attack
        m.bonus_health += health


def toggle_shop_slot_frozen(player: PlayerState, slot: int) -> None:
    """Toggle per-slot freeze when the offer slot holds a minion."""
    if slot < 0 or slot >= len(player.shop):
        return
    if player.shop[slot] is None:
        return
    frozen = list(player.shop_frozen)
    while len(frozen) < MAX_SHOP_SLOTS:
        frozen.append(False)
    frozen[slot] = not frozen[slot]
    player.shop_frozen = tuple(frozen[:MAX_SHOP_SLOTS])


def add_random_minion_to_shop(
    player: PlayerState,
    tribe: Race,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
    freeze_slot: bool = False,
) -> None:
    """Fill the first empty active offer slot with a random ``tribe`` minion."""
    n = effective_shop_offers_count(player)
    slot: Optional[int] = None
    for i in range(min(n, MAX_SHOP_SLOTS)):
        if player.shop[i] is None:
            slot = i
            break
    if slot is None:
        return
    pool = [
        cid
        for cid in tavern_card_pool(player.tavern_tier, shop_excluded_race, patch=patch)
        if minion_matches_tribe(patch.templates[cid], tribe)
    ]
    if not pool:
        return
    card_id = pool[int(rng.integers(0, len(pool)))]
    if shared_pool is not None and not shared_pool.try_reserve_offer(card_id):
        return
    player.shop[slot] = make_minion(card_id, patch=patch)
    apply_shop_tribe_bonus_to_minion(player.shop[slot], player)
    _apply_hero_shop_tribe_buff(player, player.shop[slot])
    if freeze_slot:
        frozen = list(player.shop_frozen)
        while len(frozen) < MAX_SHOP_SLOTS:
            frozen.append(False)
        frozen[slot] = True
        player.shop_frozen = tuple(frozen[:MAX_SHOP_SLOTS])


def add_random_minion_to_hand(
    player: PlayerState,
    tribe: Optional[Race],
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    patch: PatchContext,
    tier: Optional[int] = None,
) -> None:
    """Add a random tavern-pool minion (optional ``tribe`` filter) to the first free hand slot.

    The source card stays in the pool: "add another random Elemental" means one
    more Elemental, not a different one, and Tavern Tempest handing over a copy
    of itself is a real outcome. What must not happen is that it is the *only*
    outcome — see ``tavern_card_pool``.

    ``tier`` names one tavern tier and draws from that tier alone, independent
    of the seat: River Skipper hands over a Tier 1 minion on turn 12 as readily
    as on turn 1, and a card naming a tier above the seat's would otherwise draw
    from an empty pool.
    """
    slot = first_free_hand_slot(player)
    if slot is None:
        return
    draw_tier = player.tavern_tier if tier is None else int(tier)
    pool = [
        cid
        for cid in tavern_card_pool(draw_tier, shop_excluded_race, patch=patch)
        if (tribe is None or minion_matches_tribe(patch.templates[cid], tribe))
        and (tier is None or patch.templates[cid].tier == int(tier))
    ]
    if not pool:
        return
    card_id = pool[int(rng.integers(0, len(pool)))]
    player.hand[slot] = make_minion(card_id, patch=patch)


def tavern_card_pool(
    tavern_tier: int,
    shop_excluded_race: Optional[Race],
    *,
    patch: PatchContext,
) -> List[str]:
    """Cards a random tavern draw can produce: everything up to ``tavern_tier``.

    The shop's own roll uses ``tier <= tavern_tier`` (``eligible_card_ids_for_tier``);
    this drew from the tier exactly, which is a different pool and in one place a
    degenerate one. Tavern Tempest is the only tier-5 Elemental, so its battlecry
    — "add another random Elemental to your hand" — could only ever hand over
    another Tavern Tempest: play it for the Nomi tick, sell it for a gold, play
    the copy, and the loop runs until the turn's action budget is gone. A bot
    hunting Elementals found it and took a turn from 10 gold to 21.
    """
    pool = [
        cid
        for tier in range(1, max(1, int(tavern_tier)) + 1)
        for cid in shop_pool_for_tier(
            tier, shop_excluded_race=shop_excluded_race, patch=patch
        )
    ]
    if not pool:
        pool = [
            cid
            for tier in range(1, max(1, int(tavern_tier)) + 1)
            for cid in shop_pool_for_tier(tier, shop_excluded_race=None, patch=patch)
        ]
    return pool


def clear_shop_slot(
    player: PlayerState,
    slot: int,
    shared_pool: Optional[SharedCardPool],
    *,
    release_to_pool: bool = True,
) -> None:
    """Clear a shop slot; optionally return reserved copy to the lobby pool (variant B)."""
    if slot < 0 or slot >= len(player.shop):
        return
    m = player.shop[slot]
    if m is not None and shared_pool is not None and release_to_pool:
        shared_pool.release_offer(m.card_id)
    player.shop[slot] = None
    # A freeze belongs to the minion, not to the slot: once that body leaves the
    # counter (bought, or rolled away) nothing is pinned there any more. Leaving
    # the flag up made the next roll drop a fresh offer into a slot the player
    # never froze and then refuse to reroll it. ``refresh_shop`` reads its
    # ``frozen_slots`` argument, an immutable snapshot, so clearing here cannot
    # disturb the roll that is in flight.
    if slot < len(player.shop_frozen) and player.shop_frozen[slot]:
        frozen = list(player.shop_frozen)
        frozen[slot] = False
        player.shop_frozen = tuple(frozen)


def fill_shop_slot(
    player: PlayerState,
    slot: int,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> None:
    """Roll one offer into ``slot``; shared pool reserves on display."""
    # Ysera: the extra slot(s) beyond the tier's base count are always Dragons.
    forced_tribe = _hero_forced_slot_tribe(player, slot)
    if forced_tribe is not None and _fill_forced_tribe_slot(
        player, slot, forced_tribe, shop_excluded_race, rng=rng, shared_pool=shared_pool, patch=patch
    ):
        return
    if shared_pool is not None:
        cid = shared_pool.roll_and_reserve_offer(
            player.tavern_tier, shop_excluded_race, rng
        )
        player.shop[slot] = (
            make_minion(cid, patch=patch) if cid is not None else None
        )
        if player.shop[slot] is not None:
            apply_shop_tribe_bonus_to_minion(player.shop[slot], player)
            _apply_hero_shop_tribe_buff(player, player.shop[slot])
        return
    pool = tavern_card_pool(player.tavern_tier, shop_excluded_race, patch=patch)
    card_id = pool[int(rng.integers(0, len(pool)))]
    player.shop[slot] = make_minion(card_id, patch=patch)
    apply_shop_tribe_bonus_to_minion(player.shop[slot], player)
    _apply_hero_shop_tribe_buff(player, player.shop[slot])


def _fill_forced_tribe_slot(
    player: PlayerState,
    slot: int,
    tribe: Race,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool],
    patch: PatchContext,
) -> bool:
    """Fill ``slot`` with a random minion of ``tribe`` (Ysera's extra Dragon).
    Returns False if no such minion is available so the caller rolls normally."""
    pool = [
        cid
        for cid in tavern_card_pool(player.tavern_tier, shop_excluded_race, patch=patch)
        if minion_matches_tribe(patch.templates[cid], tribe)
    ]
    if not pool:
        return False
    card_id = pool[int(rng.integers(0, len(pool)))]
    if shared_pool is not None and not shared_pool.try_reserve_offer(card_id):
        return False
    player.shop[slot] = make_minion(card_id, patch=patch)
    apply_shop_tribe_bonus_to_minion(player.shop[slot], player)
    _apply_hero_shop_tribe_buff(player, player.shop[slot])
    # A minion that has just landed on the counter is owed whatever "this game"
    # bonuses the seat is carrying, the same as one it already owned.
    settle_standing_bonuses(player)
    return True


def refresh_shop(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    frozen_slots: Optional[Sequence[bool]] = None,
    patch: PatchContext,
) -> None:
    """Full reroll of active offer slots (frozen slots kept).

    A new tavern also puts a Tavern spell on the counter, *beside* the minions
    rather than in place of one: a tier-1 tavern shows three minions and a
    spell. A frozen shop keeps the spell it was frozen with, the way it keeps
    its minions.
    """
    n = effective_shop_offers_count(player)
    while len(player.shop) < MAX_SHOP_SLOTS:
        player.shop.append(None)
    frozen = frozen_slots or (False,) * MAX_SHOP_SLOTS
    if not (any(frozen[:n]) and player.tavern_spell_offers):
        offer_tavern_spells(player, rng=rng, patch=patch)
    for i in range(MAX_SHOP_SLOTS):
        if i >= n:
            if player.shop[i] is not None:
                clear_shop_slot(player, i, shared_pool, release_to_pool=not frozen[i])
            else:
                player.shop[i] = None
        elif frozen[i] and player.shop[i] is not None:
            continue
        else:
            clear_shop_slot(player, i, shared_pool, release_to_pool=True)
            fill_shop_slot(
                player,
                i,
                shop_excluded_race,
                rng=rng,
                shared_pool=shared_pool,
                patch=patch,
            )
    # After the roll, not before: the fill above would have rolled straight
    # over a promised card.
    _pay_refresh_promises(player, n, frozen, shared_pool=shared_pool, patch=patch)


def _pay_refresh_promises(
    player: PlayerState,
    n: int,
    frozen: Sequence[bool],
    *,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> None:
    """Put each promised card into this roll, one copy per promise.

    "Add a Fodder to your next 3 Refreshes" is spent a roll at a time, so this
    runs before the slots are filled and the fill works around what it placed.
    A frozen slot is not this roll's to take, and the minion it displaces goes
    back to the shared pool the way any cleared slot's does.
    """
    if not player.refresh_promises:
        return
    for card_id, left in list(player.refresh_promises.items()):
        if left <= 0:
            player.refresh_promises.pop(card_id, None)
            continue
        slot = next(
            (i for i in range(n) if not frozen[i]),
            None,
        )
        if slot is None:
            return
        clear_shop_slot(player, slot, shared_pool, release_to_pool=True)
        player.shop[slot] = make_minion(card_id, patch=patch)
        if left - 1 <= 0:
            player.refresh_promises.pop(card_id, None)
        else:
            player.refresh_promises[card_id] = left - 1


def refresh_shop_fill_empty_slots(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    frozen_slots: Optional[Sequence[bool]] = None,
    patch: PatchContext,
) -> None:
    """Keep existing offers; fill only empty active slots; clear inactive tiers."""
    n = effective_shop_offers_count(player)
    frozen = frozen_slots or (False,) * MAX_SHOP_SLOTS
    while len(player.shop) < MAX_SHOP_SLOTS:
        player.shop.append(None)
    for i in range(MAX_SHOP_SLOTS):
        if i >= n:
            if player.shop[i] is not None:
                clear_shop_slot(player, i, shared_pool, release_to_pool=not frozen[i])
            else:
                player.shop[i] = None
        elif player.shop[i] is None and not frozen[i]:
            fill_shop_slot(
                player,
                i,
                shop_excluded_race,
                rng=rng,
                shared_pool=shared_pool,
                patch=patch,
            )

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
from src.envs.minibg.actions import MAX_SHOP_SLOTS, shop_offers_count
from src.bg_lobby.player import PlayerState


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
        if not minion_matches_tribe(m, tribe):
            continue
        m.bonus_attack += attack
        m.bonus_health += health


def buff_all_shop_offers(player: PlayerState, *, attack: int, health: int) -> None:
    for m in player.shop:
        if m is None:
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
    n = shop_offers_count(player.tavern_tier)
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
) -> None:
    """Add a random tavern-pool minion (optional ``tribe`` filter) to the first free hand slot."""
    slot = first_free_hand_slot(player)
    if slot is None:
        return
    pool = [
        cid
        for cid in tavern_card_pool(player.tavern_tier, shop_excluded_race, patch=patch)
        if tribe is None or minion_matches_tribe(patch.templates[cid], tribe)
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
    pool = shop_pool_for_tier(
        tavern_tier, shop_excluded_race=shop_excluded_race, patch=patch
    )
    if not pool:
        pool = shop_pool_for_tier(tavern_tier, shop_excluded_race=None, patch=patch)
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
    if shared_pool is not None:
        cid = shared_pool.roll_and_reserve_offer(
            player.tavern_tier, shop_excluded_race, rng
        )
        player.shop[slot] = (
            make_minion(cid, patch=patch) if cid is not None else None
        )
        if player.shop[slot] is not None:
            apply_shop_tribe_bonus_to_minion(player.shop[slot], player)
        return
    pool = tavern_card_pool(player.tavern_tier, shop_excluded_race, patch=patch)
    card_id = pool[int(rng.integers(0, len(pool)))]
    player.shop[slot] = make_minion(card_id, patch=patch)
    apply_shop_tribe_bonus_to_minion(player.shop[slot], player)


def refresh_shop(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    frozen_slots: Optional[Sequence[bool]] = None,
    patch: PatchContext,
) -> None:
    """Full reroll of active offer slots (frozen slots kept)."""
    n = shop_offers_count(player.tavern_tier)
    while len(player.shop) < MAX_SHOP_SLOTS:
        player.shop.append(None)
    frozen = frozen_slots or (False,) * MAX_SHOP_SLOTS
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
    n = shop_offers_count(player.tavern_tier)
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

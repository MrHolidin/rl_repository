"""Tavern shop refresh (card pool selection and offer slots)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from src.bg_catalog.cards import make_minion, shop_pool_for_tier
from src.bg_core.minion import Minion, Race
from src.bg_lobby.shared_pool import SharedCardPool

from src.envs.minibg.actions import MAX_SHOP_SLOTS, shop_offers_count
from src.bg_lobby.player import PlayerState


def tavern_card_pool(
    tavern_tier: int,
    shop_excluded_race: Optional[Race],
) -> List[str]:
    pool = shop_pool_for_tier(
        tavern_tier, shop_excluded_race=shop_excluded_race
    )
    if not pool:
        pool = shop_pool_for_tier(tavern_tier, shop_excluded_race=None)
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
) -> None:
    """Roll one offer into ``slot``; shared pool reserves on display."""
    if shared_pool is not None:
        cid = shared_pool.roll_and_reserve_offer(
            player.tavern_tier, shop_excluded_race, rng
        )
        player.shop[slot] = make_minion(cid) if cid is not None else None
        return
    pool = tavern_card_pool(player.tavern_tier, shop_excluded_race)
    card_id = pool[int(rng.integers(0, len(pool)))]
    player.shop[slot] = make_minion(card_id)


def refresh_shop(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    frozen_slots: Optional[Sequence[bool]] = None,
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
                player, i, shop_excluded_race, rng=rng, shared_pool=shared_pool
            )


def refresh_shop_fill_empty_slots(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    frozen_slots: Optional[Sequence[bool]] = None,
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
                player, i, shop_excluded_race, rng=rng, shared_pool=shared_pool
            )

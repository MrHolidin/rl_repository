"""Tavern shop refresh (card pool selection and offer slots)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.bg_catalog.cards import make_minion, shop_pool_for_tier
from src.bg_core.minion import Minion, Race

from src.envs.minibg.actions import MAX_SHOP_SLOTS, shop_offers_count
from src.envs.minibg.state import PlayerState


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


def refresh_shop(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
) -> None:
    """Full reroll of active offer slots (``player.shop`` replaced)."""
    n = shop_offers_count(player.tavern_tier)
    pool = tavern_card_pool(player.tavern_tier, shop_excluded_race)
    new_shop: List[Optional[Minion]] = [None] * MAX_SHOP_SLOTS
    for i in range(n):
        card_id = pool[int(rng.integers(0, len(pool)))]
        new_shop[i] = make_minion(card_id)
    player.shop = new_shop


def refresh_shop_fill_empty_slots(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
) -> None:
    """Keep existing offers; reroll only empty active slots; clear inactive tiers."""
    n = shop_offers_count(player.tavern_tier)
    pool = tavern_card_pool(player.tavern_tier, shop_excluded_race)
    while len(player.shop) < MAX_SHOP_SLOTS:
        player.shop.append(None)
    for i in range(MAX_SHOP_SLOTS):
        if i >= n:
            player.shop[i] = None
        elif player.shop[i] is None:
            card_id = pool[int(rng.integers(0, len(pool)))]
            player.shop[i] = make_minion(card_id)

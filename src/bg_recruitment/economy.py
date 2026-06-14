"""Shop economy: buy, sell, roll, level up."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion, Race

from src.envs.minibg.actions import (
    BUY_COST,
    LEVEL_UP_COSTS,
    MAX_SHOP_SLOTS,
    MAX_TIER,
    ROLL_COST,
    SELL_REWARD,
    shop_offers_count,
)
from src.bg_lobby.player import PlayerState
from src.bg_lobby.shared_pool import SharedCardPool

from .hand_slots import first_free_hand_slot
from .pool_ledger import on_sell_minion
from .shop import clear_shop_slot, fill_shop_slot, refresh_shop, tavern_card_pool


def effective_sell_reward(minion: Minion) -> int:
    if minion.sell_value is not None:
        return int(minion.sell_value)
    return SELL_REWARD


def effective_roll_cost(player: PlayerState) -> int:
    # Nozdormu: first refresh each turn is free (takes precedence).
    if player.hero_free_roll_pending:
        return 0
    if player.next_roll_cost_override is not None:
        return max(0, int(player.next_roll_cost_override))
    # Millhouse: every refresh costs a flat amount.
    if player.hero is not None:
        flat = player.hero.flat_refresh_cost()
        if flat is not None:
            return max(0, flat)
    return ROLL_COST


def effective_buy_cost(player: PlayerState) -> int:
    # Millhouse: minions cost a flat amount.
    if player.hero is not None:
        flat = player.hero.flat_buy_cost()
        if flat is not None:
            return max(0, flat)
    return BUY_COST


def effective_level_up_cost(player: PlayerState) -> int:
    base = player.next_tier_up_cost + player.upgrade_cost_delta
    if player.hero is not None:
        base += player.hero.upgrade_cost_surcharge()  # Millhouse: +1
        base -= player.hero_upgrade_discount  # Chenvaala: accumulated discount
    return max(0, base)


def sell_from_board(
    player: PlayerState,
    pos: int,
    *,
    on_sell: Callable[[Minion, PlayerState], None] | None = None,
    on_triples: Callable[[PlayerState], None],
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    sold = player.board[pos]
    if on_sell is not None:
        on_sell(sold, player)
    on_sell_minion(shared_pool, sold)
    del player.board[pos]
    player.gold += effective_sell_reward(sold)
    on_triples(player)


def buy_from_shop(
    player: PlayerState,
    slot: int,
    *,
    on_bought: Callable[[Minion, PlayerState], None],
    on_friendly_bought: Callable[[Minion, PlayerState], None] | None = None,
    on_triples: Callable[[PlayerState], None],
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    minion = player.shop[slot]
    assert minion is not None
    player.gold -= effective_buy_cost(player)
    clear_shop_slot(player, slot, shared_pool, release_to_pool=False)
    h = first_free_hand_slot(player)
    assert h is not None, "BUY illegal when hand is full (legal mask bug)"
    player.hand[h] = minion
    on_bought(minion, player)
    if on_friendly_bought is not None:
        on_friendly_bought(minion, player)
    on_triples(player)


def roll_shop(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> None:
    cost = effective_roll_cost(player)
    player.gold -= cost
    # Nozdormu: consume the free first refresh for this turn.
    if player.hero_free_roll_pending:
        player.hero_free_roll_pending = False
    if player.free_roll_charges > 0:
        player.free_roll_charges -= 1
        if player.free_roll_charges > 0:
            player.next_roll_cost_override = 0
        else:
            player.next_roll_cost_override = None
    elif player.next_roll_cost_override is not None:
        player.next_roll_cost_override = None
    refresh_shop(
        player,
        shop_excluded_race,
        rng=rng,
        shared_pool=shared_pool,
        frozen_slots=player.shop_frozen,
        patch=patch,
    )


def level_up_tavern(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> None:
    cost = effective_level_up_cost(player)
    player.gold -= cost
    player.upgrade_cost_delta = 0
    player.hero_upgrade_discount = 0  # Chenvaala: discount consumed by the upgrade
    old_tier = player.tavern_tier
    player.tavern_tier += 1
    if player.tavern_tier < MAX_TIER:
        player.next_tier_up_cost = LEVEL_UP_COSTS[player.tavern_tier]
    extra = player.hero.extra_shop_slots() if player.hero is not None else 0
    old_n = min(MAX_SHOP_SLOTS, shop_offers_count(old_tier) + extra)
    new_n = min(MAX_SHOP_SLOTS, shop_offers_count(player.tavern_tier) + extra)
    while len(player.shop) < MAX_SHOP_SLOTS:
        player.shop.append(None)
    for i in range(old_n, new_n):
        fill_shop_slot(
            player,
            i,
            shop_excluded_race,
            rng=rng,
            shared_pool=shared_pool,
            patch=patch,
        )

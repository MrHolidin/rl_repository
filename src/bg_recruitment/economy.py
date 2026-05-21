"""Shop economy: buy, sell, roll, level up."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

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


def buy_from_shop(
    player: PlayerState,
    slot: int,
    *,
    on_bought: Callable[[Minion, PlayerState], None],
    on_triples: Callable[[PlayerState], None],
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    minion = player.shop[slot]
    assert minion is not None
    player.gold -= BUY_COST
    clear_shop_slot(player, slot, shared_pool, release_to_pool=False)
    h = first_free_hand_slot(player)
    assert h is not None, "BUY illegal when hand is full (legal mask bug)"
    player.hand[h] = minion
    on_bought(minion, player)
    on_triples(player)


def sell_from_board(
    player: PlayerState,
    pos: int,
    *,
    on_triples: Callable[[PlayerState], None],
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    sold = player.board[pos]
    on_sell_minion(shared_pool, sold)
    del player.board[pos]
    player.gold += SELL_REWARD
    on_triples(player)


def roll_shop(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    player.gold -= ROLL_COST
    refresh_shop(
        player, shop_excluded_race, rng=rng, shared_pool=shared_pool
    )


def level_up_tavern(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
    shared_pool: Optional[SharedCardPool] = None,
) -> None:
    cost = player.next_tier_up_cost
    player.gold -= cost
    old_tier = player.tavern_tier
    player.tavern_tier += 1
    if player.tavern_tier < MAX_TIER:
        player.next_tier_up_cost = LEVEL_UP_COSTS[player.tavern_tier]
    old_n = shop_offers_count(old_tier)
    new_n = shop_offers_count(player.tavern_tier)
    while len(player.shop) < MAX_SHOP_SLOTS:
        player.shop.append(None)
    for i in range(old_n, new_n):
        fill_shop_slot(
            player,
            i,
            shop_excluded_race,
            rng=rng,
            shared_pool=shared_pool,
        )

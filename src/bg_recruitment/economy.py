"""Shop economy: buy, sell, roll, level up."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_core.minion import Minion, Race

from src.envs.minibg.actions import (
    BUY_COST,
    HAND_SIZE,
    LEVEL_UP_COSTS,
    MAX_SHOP_SLOTS,
    MAX_TIER,
    ROLL_COST,
    SELL_REWARD,
    shop_offers_count,
)
from src.envs.minibg.state import PlayerState

from .shop import refresh_shop, tavern_card_pool


def buy_from_shop(
    player: PlayerState,
    slot: int,
    *,
    on_bought: Callable[[Minion, PlayerState], None],
    on_triples: Callable[[PlayerState], None],
) -> None:
    minion = player.shop[slot]
    assert minion is not None
    player.gold -= BUY_COST
    player.shop[slot] = None
    h = next((i for i in range(HAND_SIZE) if player.hand[i] is None), None)
    assert h is not None, "BUY illegal when hand is full (legal mask bug)"
    player.hand[h] = minion
    on_bought(minion, player)
    on_triples(player)


def sell_from_board(
    player: PlayerState,
    pos: int,
    *,
    on_triples: Callable[[PlayerState], None],
) -> None:
    del player.board[pos]
    player.gold += SELL_REWARD
    on_triples(player)


def roll_shop(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
) -> None:
    player.gold -= ROLL_COST
    refresh_shop(player, shop_excluded_race, rng=rng)


def level_up_tavern(
    player: PlayerState,
    shop_excluded_race: Optional[Race],
    *,
    rng: np.random.Generator,
) -> None:
    cost = player.next_tier_up_cost
    player.gold -= cost
    old_tier = player.tavern_tier
    player.tavern_tier += 1
    if player.tavern_tier < MAX_TIER:
        player.next_tier_up_cost = LEVEL_UP_COSTS[player.tavern_tier]
    old_n = shop_offers_count(old_tier)
    new_n = shop_offers_count(player.tavern_tier)
    pool = tavern_card_pool(player.tavern_tier, shop_excluded_race)
    while len(player.shop) < MAX_SHOP_SLOTS:
        player.shop.append(None)
    for i in range(old_n, new_n):
        card_id = pool[int(rng.integers(0, len(pool)))]
        player.shop[i] = make_minion(card_id)

from __future__ import annotations

import itertools
from typing import List, Tuple

from .actions import Action as GameAction
from .actions import BOARD_SIZE, SHOP_SIZE


NUM_ENV_ACTIONS = 33

A_ROLL = 0
A_LEVEL_UP = 1
A_BUY_BASE = 2
A_SELL_BASE = 5
A_SELECT_ORDER_BASE = 9

NUM_PERMS = 24

PERMUTATIONS_4: Tuple[Tuple[int, int, int, int], ...] = tuple(
    itertools.permutations(range(BOARD_SIZE))
)
assert len(PERMUTATIONS_4) == NUM_PERMS


def is_buy(env_action: int) -> bool:
    return A_BUY_BASE <= env_action < A_BUY_BASE + SHOP_SIZE


def is_sell(env_action: int) -> bool:
    return A_SELL_BASE <= env_action < A_SELL_BASE + BOARD_SIZE


def is_select_order(env_action: int) -> bool:
    return A_SELECT_ORDER_BASE <= env_action < NUM_ENV_ACTIONS


def buy_slot(env_action: int) -> int:
    return env_action - A_BUY_BASE


def sell_pos(env_action: int) -> int:
    return env_action - A_SELL_BASE


def order_index(env_action: int) -> int:
    return env_action - A_SELECT_ORDER_BASE


def legal_order_indices(board_size: int) -> List[int]:
    """Indices of permutations whose tail beyond `board_size` is identity."""
    return [
        i
        for i, perm in enumerate(PERMUTATIONS_4)
        if all(perm[j] == j for j in range(board_size, BOARD_SIZE))
    ]


def env_action_to_game_action(env_action: int) -> int:
    """Map non-SELECT_FINAL_ORDER env actions to game actions."""
    if env_action == A_ROLL:
        return int(GameAction.ROLL)
    if env_action == A_LEVEL_UP:
        return int(GameAction.LEVEL_UP)
    if is_buy(env_action):
        return int(GameAction.BUY_SLOT_0) + buy_slot(env_action)
    if is_sell(env_action):
        return int(GameAction.SELL_BOARD_0) + sell_pos(env_action)
    raise ValueError(
        f"env_action {env_action} is SELECT_FINAL_ORDER or out of range"
    )


__all__ = [
    "NUM_ENV_ACTIONS",
    "A_ROLL",
    "A_LEVEL_UP",
    "A_BUY_BASE",
    "A_SELL_BASE",
    "A_SELECT_ORDER_BASE",
    "NUM_PERMS",
    "PERMUTATIONS_4",
    "is_buy",
    "is_sell",
    "is_select_order",
    "buy_slot",
    "sell_pos",
    "order_index",
    "legal_order_indices",
    "env_action_to_game_action",
]

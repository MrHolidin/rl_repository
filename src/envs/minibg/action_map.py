from __future__ import annotations

import itertools
from typing import List, Tuple

from .actions import Action as GameAction
from .actions import BOARD_SIZE, HAND_SIZE, SHOP_SIZE


# Action layout (52 total):
#   0..23      ROLL .. MAGNET (last magnet = 23)
#   24..26     DISCOVER_PICK_0..2  (pending Discover / Adapt)
#   27         FINISH
#   28..51     SELECT_ORDER_*
A_ROLL = 0
A_LEVEL_UP = 1
A_BUY_BASE = 2
A_SELL_BASE = A_BUY_BASE + SHOP_SIZE
A_PLACE_BASE = A_SELL_BASE + BOARD_SIZE
A_MAGNET_BASE = A_PLACE_BASE + HAND_SIZE
A_DISCOVER_BASE = A_MAGNET_BASE + HAND_SIZE * BOARD_SIZE
A_FINISH = A_DISCOVER_BASE + 3
A_SELECT_ORDER_BASE = A_FINISH + 1

NUM_PERMS = 24
NUM_ENV_ACTIONS = A_SELECT_ORDER_BASE + NUM_PERMS

PERMUTATIONS_4: Tuple[Tuple[int, int, int, int], ...] = tuple(
    itertools.permutations(range(BOARD_SIZE))
)
assert len(PERMUTATIONS_4) == NUM_PERMS


def is_buy(env_action: int) -> bool:
    return A_BUY_BASE <= env_action < A_BUY_BASE + SHOP_SIZE


def is_sell(env_action: int) -> bool:
    return A_SELL_BASE <= env_action < A_SELL_BASE + BOARD_SIZE


def is_place(env_action: int) -> bool:
    return A_PLACE_BASE <= env_action < A_PLACE_BASE + HAND_SIZE


def is_magnet(env_action: int) -> bool:
    return A_MAGNET_BASE <= env_action < A_MAGNET_BASE + HAND_SIZE * BOARD_SIZE


def is_finish(env_action: int) -> bool:
    return env_action == A_FINISH


def is_select_order(env_action: int) -> bool:
    return A_SELECT_ORDER_BASE <= env_action < NUM_ENV_ACTIONS


def buy_slot(env_action: int) -> int:
    return env_action - A_BUY_BASE


def sell_pos(env_action: int) -> int:
    return env_action - A_SELL_BASE


def place_slot(env_action: int) -> int:
    return env_action - A_PLACE_BASE


def magnet_hand_board(env_action: int) -> Tuple[int, int]:
    off = env_action - A_MAGNET_BASE
    return off // BOARD_SIZE, off % BOARD_SIZE


def is_discover_pick(env_action: int) -> bool:
    return A_DISCOVER_BASE <= env_action < A_DISCOVER_BASE + 3


def discover_pick_slot(env_action: int) -> int:
    return env_action - A_DISCOVER_BASE


def order_index(env_action: int) -> int:
    return env_action - A_SELECT_ORDER_BASE


def legal_order_indices(board_size: int) -> List[int]:
    """Indices into ``PERMUTATIONS_4`` whose tail beyond ``board_size`` is identity.

    Returns exactly ``max(1, board_size!)`` indices: one canonical perm per
    equivalence class under ``MiniBGGame.reorder_board`` (compact-after-permute).
    For ``board_size = 0 or 1`` this is just the identity perm; for
    ``board_size = 4`` it is the full set of 24.
    """
    return [
        i
        for i, perm in enumerate(PERMUTATIONS_4)
        if all(perm[j] == j for j in range(board_size, BOARD_SIZE))
    ]


def env_action_to_game_action(env_action: int) -> int:
    """Map non-SELECT_ORDER env actions to game actions."""
    if env_action == A_ROLL:
        return int(GameAction.ROLL)
    if env_action == A_LEVEL_UP:
        return int(GameAction.LEVEL_UP)
    if env_action == A_FINISH:
        return int(GameAction.FINISH)
    if is_buy(env_action):
        return int(GameAction.BUY_SLOT_0) + buy_slot(env_action)
    if is_sell(env_action):
        return int(GameAction.SELL_BOARD_0) + sell_pos(env_action)
    if is_place(env_action):
        return int(GameAction.PLACE_HAND_0) + place_slot(env_action)
    if is_magnet(env_action):
        h, b = magnet_hand_board(env_action)
        return int(GameAction.MAGNET_HAND_0_BOARD_0) + h * BOARD_SIZE + b
    if is_discover_pick(env_action):
        return int(GameAction.DISCOVER_PICK_0) + discover_pick_slot(env_action)
    raise ValueError(
        f"env_action {env_action} is SELECT_ORDER or out of range"
    )


__all__ = [
    "NUM_ENV_ACTIONS",
    "A_ROLL",
    "A_LEVEL_UP",
    "A_BUY_BASE",
    "A_SELL_BASE",
    "A_PLACE_BASE",
    "A_MAGNET_BASE",
    "A_DISCOVER_BASE",
    "A_FINISH",
    "A_SELECT_ORDER_BASE",
    "NUM_PERMS",
    "PERMUTATIONS_4",
    "is_buy",
    "is_sell",
    "is_place",
    "is_magnet",
    "is_discover_pick",
    "is_finish",
    "is_select_order",
    "buy_slot",
    "sell_pos",
    "place_slot",
    "magnet_hand_board",
    "discover_pick_slot",
    "order_index",
    "legal_order_indices",
    "env_action_to_game_action",
]

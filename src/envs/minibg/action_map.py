"""MiniBG: flat Gymnasium env index layout.

``0 .. NUM_ACTIONS-1`` mirror :class:`actions.Action`. Order phase uses adjacent
board swaps at ``NUM_ACTIONS .. NUM_ACTIONS + BOARD_SIZE - 2`` (no factorial action count).
"""

from __future__ import annotations

from typing import Iterable, Tuple

from .actions import MAX_SHOP_SLOTS, Action, BOARD_SIZE, HAND_SIZE, NUM_ACTIONS, STARTING_TIER

A_SHOP_SLOT_0 = int(Action.BUY_SLOT_0)
A_SHOP_SLOT_LAST = int(Action.BUY_SLOT_5)
A_SELL_BOARD_0 = int(Action.SELL_BOARD_0)
A_SELL_BOARD_LAST = int(Action.SELL_BOARD_0) + BOARD_SIZE - 1
A_ROLL = int(Action.ROLL)
A_LEVEL_UP = int(Action.LEVEL_UP)
A_FINISH = int(Action.FINISH)
A_FINISH_FREEZE_SHOP = int(Action.FINISH_FREEZE_SHOP)
A_PLACE_HAND_0 = int(Action.PLACE_HAND_0)
A_PLACE_HAND_LAST = int(Action.PLACE_HAND_0) + HAND_SIZE - 1
A_MAGNET_HAND_0_BOARD_0 = int(Action.MAGNET_HAND_0_BOARD_0)
A_MAGNET_HAND_LAST_BOARD_LAST = int(Action.MAGNET_HAND_4_BOARD_6)
A_DISCOVER_PICK_0 = int(Action.DISCOVER_PICK_0)
A_DISCOVER_PICK_LAST = int(Action.DISCOVER_PICK_2)
A_TARGET_BOARD_0 = int(Action.TARGET_BOARD_0)
A_TARGET_BOARD_LAST = int(Action.TARGET_BOARD_6)

NUM_SWAP_ADJ = BOARD_SIZE - 1
A_SWAP_BOARD_0 = NUM_ACTIONS
A_SWAP_BOARD_LAST = A_SWAP_BOARD_0 + NUM_SWAP_ADJ - 1

# RL-only: skip second adjacent target during staged place (not a game ``Action``).
A_APPLY_EFFECT_SKIP = A_SWAP_BOARD_LAST + 1

NUM_ENV_ACTIONS = A_APPLY_EFFECT_SKIP + 1

A_BUY_BASE = A_SHOP_SLOT_0
A_SELL_BASE = A_SELL_BOARD_0
A_PLACE_BASE = A_PLACE_HAND_0
A_MAGNET_BASE = A_MAGNET_HAND_0_BOARD_0
A_DISCOVER_BASE = A_DISCOVER_PICK_0
A_TARGET_BOARD_BASE = A_TARGET_BOARD_0

A_SELECT_ORDER_BASE = A_SWAP_BOARD_0


def is_swap_board(env_action: int) -> bool:
    a = int(env_action)
    return A_SWAP_BOARD_0 <= a <= A_SWAP_BOARD_LAST


def swap_adj_index_from_env_action(env_action: int) -> int:
    a = int(env_action)
    if not is_swap_board(a):
        raise ValueError(f"not a SWAP_BOARD action: {env_action}")
    return a - A_SWAP_BOARD_0


def shop_slot_from_env_action(env_action: int) -> int:
    a = int(env_action)
    if not (A_SHOP_SLOT_0 <= a <= A_SHOP_SLOT_LAST):
        raise ValueError(f"not a shop buy action: {env_action}")
    return a - A_SHOP_SLOT_0


def is_buy(env_action: int) -> bool:
    return A_BUY_BASE <= int(env_action) <= A_SHOP_SLOT_LAST


def buy_slot(env_action: int) -> int:
    return shop_slot_from_env_action(env_action)


def is_sell(env_action: int) -> bool:
    a = int(env_action)
    return A_SELL_BASE <= a <= A_SELL_BOARD_LAST


def sell_pos(env_action: int) -> int:
    a = int(env_action)
    if not is_sell(a):
        raise ValueError(f"not a sell action: {env_action}")
    return a - A_SELL_BASE


def is_place(env_action: int) -> bool:
    a = int(env_action)
    return A_PLACE_BASE <= a <= A_PLACE_HAND_LAST


def place_slot(env_action: int) -> int:
    a = int(env_action)
    if not is_place(a):
        raise ValueError(f"not a place action: {env_action}")
    return a - A_PLACE_BASE


def is_magnet(env_action: int) -> bool:
    a = int(env_action)
    return A_MAGNET_BASE <= a <= A_MAGNET_HAND_LAST_BOARD_LAST


def magnet_hand_board(env_action: int) -> Tuple[int, int]:
    a = int(env_action)
    if not is_magnet(a):
        raise ValueError(f"not a magnet action: {env_action}")
    off = a - A_MAGNET_BASE
    return divmod(off, BOARD_SIZE)


def is_discover_pick(env_action: int) -> bool:
    a = int(env_action)
    return A_DISCOVER_BASE <= a <= A_DISCOVER_PICK_LAST


def discover_pick_slot(env_action: int) -> int:
    a = int(env_action)
    if not is_discover_pick(a):
        raise ValueError(f"not a discover pick action: {env_action}")
    return a - A_DISCOVER_BASE


def is_apply_effect_skip(env_action: int) -> bool:
    return int(env_action) == A_APPLY_EFFECT_SKIP


def is_target_board(env_action: int) -> bool:
    a = int(env_action)
    return A_TARGET_BOARD_0 <= a <= A_TARGET_BOARD_LAST


def target_board_slot(env_action: int) -> int:
    a = int(env_action)
    if not is_target_board(a):
        raise ValueError(f"not a TARGET_BOARD action: {env_action}")
    return a - A_TARGET_BOARD_BASE


def is_finish(env_action: int) -> bool:
    return int(env_action) == A_FINISH


def is_finish_freeze_shop(env_action: int) -> bool:
    return int(env_action) == A_FINISH_FREEZE_SHOP


def env_action_to_game_action(env_action: int) -> int:
    a = int(env_action)
    if is_swap_board(a):
        raise ValueError(f"env action {a} is SWAP_BOARD, not a primitive game action")
    if is_apply_effect_skip(a):
        raise ValueError(f"env action {a} is APPLY_EFFECT_SKIP, not a primitive game action")
    if not (0 <= a < NUM_ACTIONS):
        raise ValueError(f"invalid env action index: {env_action} (NUM_ACTIONS={NUM_ACTIONS})")
    return a


def _tier_to_max_shop_actions(tier: int) -> int:
    t = int(tier)
    return MAX_SHOP_SLOTS if t >= 6 else max(1, min(MAX_SHOP_SLOTS, 3 + (t - 1)))


def _iter_primitive_actions(*, tavern_tier: int) -> Iterable[int]:
    max_shop = _tier_to_max_shop_actions(tavern_tier)
    yield from range(A_SHOP_SLOT_0, A_SHOP_SLOT_0 + max_shop)
    yield from range(A_SELL_BOARD_0, A_SELL_BOARD_LAST + 1)
    yield A_ROLL
    yield A_LEVEL_UP
    yield A_FINISH
    yield A_FINISH_FREEZE_SHOP
    yield from range(A_PLACE_HAND_0, A_PLACE_HAND_LAST + 1)
    yield from range(A_MAGNET_HAND_0_BOARD_0, A_MAGNET_HAND_LAST_BOARD_LAST + 1)
    yield from range(A_DISCOVER_PICK_0, A_DISCOVER_PICK_LAST + 1)
    yield from range(A_TARGET_BOARD_0, A_TARGET_BOARD_LAST + 1)


def build_fixed_env_legal_mask(*, tavern_tier: int) -> list[bool]:
    legal = [False] * int(NUM_ENV_ACTIONS)
    for a in _iter_primitive_actions(tavern_tier=int(tavern_tier)):
        legal[a] = True
    for j in range(int(NUM_SWAP_ADJ)):
        legal[A_SWAP_BOARD_0 + j] = True
    return legal


FIXED_ENV_LEGAL_MASK: list[bool] = build_fixed_env_legal_mask(tavern_tier=STARTING_TIER)


__all__ = [
    "A_APPLY_EFFECT_SKIP",
    "A_BUY_BASE",
    "A_DISCOVER_BASE",
    "A_TARGET_BOARD_BASE",
    "A_TARGET_BOARD_0",
    "A_TARGET_BOARD_LAST",
    "is_apply_effect_skip",
    "is_target_board",
    "target_board_slot",
    "A_FINISH",
    "A_FINISH_FREEZE_SHOP",
    "A_LEVEL_UP",
    "A_MAGNET_BASE",
    "A_MAGNET_HAND_LAST_BOARD_LAST",
    "A_PLACE_BASE",
    "A_ROLL",
    "A_SELECT_ORDER_BASE",
    "A_SELL_BASE",
    "A_SWAP_BOARD_0",
    "A_SWAP_BOARD_LAST",
    "A_SHOP_SLOT_LAST",
    "FIXED_ENV_LEGAL_MASK",
    "NUM_ENV_ACTIONS",
    "NUM_SWAP_ADJ",
    "build_fixed_env_legal_mask",
    "buy_slot",
    "discover_pick_slot",
    "env_action_to_game_action",
    "is_buy",
    "is_discover_pick",
    "is_finish",
    "is_finish_freeze_shop",
    "is_magnet",
    "is_place",
    "is_sell",
    "is_swap_board",
    "magnet_hand_board",
    "place_slot",
    "sell_pos",
    "shop_slot_from_env_action",
    "swap_adj_index_from_env_action",
]

"""BGLike flat action indices (same layout family as minibg ``action_map``)."""

from __future__ import annotations

from typing import Tuple

from .actions import (
    BOARD_SIZE,
    HAND_SIZE,
    MAX_SHOP_SLOTS,
    NUM_ACTIONS,
    Action,
    is_discover_pick_game_action,
    is_magnet_game_action,
    magnet_game_action,
    magnet_hand_board_from_game_action,
)

NUM_SWAP_ADJ = BOARD_SIZE - 1
A_SWAP_BOARD_0 = NUM_ACTIONS
A_SWAP_BOARD_LAST = A_SWAP_BOARD_0 + NUM_SWAP_ADJ - 1

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
A_MAGNET_HAND_LAST_BOARD_LAST = (
    int(Action.MAGNET_HAND_0_BOARD_0) + HAND_SIZE * BOARD_SIZE - 1
)
A_DISCOVER_PICK_0 = int(Action.DISCOVER_PICK_0)
A_DISCOVER_PICK_LAST = int(Action.DISCOVER_PICK_2)
A_TARGET_BOARD_0 = int(Action.TARGET_BOARD_0)
A_TARGET_BOARD_LAST = int(Action.TARGET_BOARD_0) + BOARD_SIZE - 1

# RL-only: skip second adjacent target during staged place (not a game ``Action``).
A_APPLY_EFFECT_SKIP = A_SWAP_BOARD_LAST + 1

NUM_ENV_ACTIONS = A_APPLY_EFFECT_SKIP + 1

A_BUY_BASE = A_SHOP_SLOT_0
A_SELL_BASE = A_SELL_BOARD_0
A_PLACE_BASE = A_PLACE_HAND_0
A_MAGNET_BASE = A_MAGNET_HAND_0_BOARD_0
A_DISCOVER_BASE = A_DISCOVER_PICK_0
A_TARGET_BOARD_BASE = A_TARGET_BOARD_0


def is_swap_board(env_action: int) -> bool:
    a = int(env_action)
    return A_SWAP_BOARD_0 <= a <= A_SWAP_BOARD_LAST


def swap_adj_index_from_env_action(env_action: int) -> int:
    a = int(env_action)
    if not is_swap_board(a):
        raise ValueError(f"not a SWAP_BOARD action: {env_action}")
    return a - A_SWAP_BOARD_0


def is_buy(env_action: int) -> bool:
    return A_BUY_BASE <= int(env_action) <= A_SHOP_SLOT_LAST


def buy_slot(env_action: int) -> int:
    a = int(env_action)
    if not is_buy(a):
        raise ValueError(f"not a shop buy action: {env_action}")
    return a - A_SHOP_SLOT_0


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
    return is_magnet_game_action(int(env_action))


def magnet_hand_board(env_action: int) -> Tuple[int, int]:
    return magnet_hand_board_from_game_action(int(env_action))


def is_discover_pick(env_action: int) -> bool:
    return is_discover_pick_game_action(int(env_action))


def discover_pick_slot(env_action: int) -> int:
    a = int(env_action)
    if not is_discover_pick(a):
        raise ValueError(f"not a discover pick action: {env_action}")
    return a - A_DISCOVER_BASE


def is_target_board(env_action: int) -> bool:
    a = int(env_action)
    return A_TARGET_BOARD_BASE <= a <= A_TARGET_BOARD_LAST


def is_apply_effect_skip(env_action: int) -> bool:
    return int(env_action) == A_APPLY_EFFECT_SKIP


def is_finish(env_action: int) -> bool:
    return int(env_action) == A_FINISH


def is_finish_freeze_shop(env_action: int) -> bool:
    return int(env_action) == A_FINISH_FREEZE_SHOP


def target_board_slot(env_action: int) -> int:
    a = int(env_action)
    if not is_target_board(a):
        raise ValueError(f"not a TARGET_BOARD action: {env_action}")
    return a - A_TARGET_BOARD_BASE


def struct_action_to_game_action(action) -> int:
    """Map structured shop token to flat game ``Action`` int."""
    from src.envs.minibg.structured_actions import StructAction, StructActionType

    if not isinstance(action, StructAction):
        raise TypeError(f"expected StructAction, got {type(action)!r}")
    if action.type == StructActionType.ROLL:
        return int(Action.ROLL)
    if action.type == StructActionType.LEVEL_UP:
        return int(Action.LEVEL_UP)
    if action.type == StructActionType.BUY:
        return int(Action.BUY_SLOT_0) + action.args[0]
    if action.type == StructActionType.SELL:
        return int(Action.SELL_BOARD_0) + action.args[0]
    if action.type == StructActionType.PLACE:
        return int(Action.PLACE_HAND_0) + action.args[0]
    if action.type == StructActionType.MAGNET:
        return int(magnet_game_action(action.args[0], action.args[1]))
    if action.type == StructActionType.DISCOVER_PICK:
        return int(Action.DISCOVER_PICK_0) + action.args[0]
    if action.type == StructActionType.APPLY_EFFECT:
        return int(Action.TARGET_BOARD_0) + action.args[0]
    raise ValueError(f"not a shop-phase structured action: {action}")


def struct_action_to_log_int(action) -> int:
    """Flat env action id for replay/logging after a structured step."""
    from src.envs.minibg.structured_actions import StructAction, StructActionType

    if action.type == StructActionType.COMPLETE_TURN:
        return int(Action.FINISH)
    if action.type == StructActionType.COMPLETE_TURN_FREEZE_SHOP:
        return int(Action.FINISH_FREEZE_SHOP)
    if action.type == StructActionType.APPLY_EFFECT_SKIP:
        return int(A_APPLY_EFFECT_SKIP)
    return struct_action_to_game_action(action)


__all__ = [
    "A_APPLY_EFFECT_SKIP",
    "A_BUY_BASE",
    "A_DISCOVER_BASE",
    "A_DISCOVER_PICK_0",
    "A_FINISH",
    "A_FINISH_FREEZE_SHOP",
    "A_LEVEL_UP",
    "A_MAGNET_BASE",
    "A_PLACE_BASE",
    "A_ROLL",
    "A_SELL_BASE",
    "A_SWAP_BOARD_0",
    "A_SWAP_BOARD_LAST",
    "A_TARGET_BOARD_BASE",
    "NUM_ENV_ACTIONS",
    "NUM_SWAP_ADJ",
    "buy_slot",
    "discover_pick_slot",
    "is_buy",
    "is_discover_pick",
    "is_magnet",
    "is_place",
    "is_sell",
    "is_apply_effect_skip",
    "is_finish",
    "is_finish_freeze_shop",
    "is_swap_board",
    "is_target_board",
    "magnet_hand_board",
    "struct_action_to_game_action",
    "struct_action_to_log_int",
    "swap_adj_index_from_env_action",
    "target_board_slot",
    "place_slot",
    "sell_pos",
]

from __future__ import annotations

from enum import IntEnum

SHOP_SIZE = 3
BOARD_SIZE = 4
HAND_SIZE = 3


class Action(IntEnum):
    BUY_SLOT_0 = 0
    BUY_SLOT_1 = 1
    BUY_SLOT_2 = 2
    SELL_BOARD_0 = 3
    SELL_BOARD_1 = 4
    SELL_BOARD_2 = 5
    SELL_BOARD_3 = 6
    ROLL = 7
    LEVEL_UP = 8
    FINISH = 9
    PLACE_HAND_0 = 10
    PLACE_HAND_1 = 11
    PLACE_HAND_2 = 12
    MAGNET_HAND_0_BOARD_0 = 13
    MAGNET_HAND_0_BOARD_1 = 14
    MAGNET_HAND_0_BOARD_2 = 15
    MAGNET_HAND_0_BOARD_3 = 16
    MAGNET_HAND_1_BOARD_0 = 17
    MAGNET_HAND_1_BOARD_1 = 18
    MAGNET_HAND_1_BOARD_2 = 19
    MAGNET_HAND_1_BOARD_3 = 20
    MAGNET_HAND_2_BOARD_0 = 21
    MAGNET_HAND_2_BOARD_1 = 22
    MAGNET_HAND_2_BOARD_2 = 23
    MAGNET_HAND_2_BOARD_3 = 24
    DISCOVER_PICK_0 = 25
    DISCOVER_PICK_1 = 26
    DISCOVER_PICK_2 = 27


NUM_ACTIONS = 28

MAGNET_ACTION_BASE = int(Action.MAGNET_HAND_0_BOARD_0)
NUM_MAGNET_ACTIONS = HAND_SIZE * BOARD_SIZE
MAX_SHOP_ACTIONS = 20

STARTING_HEALTH = 30
STARTING_GOLD = 3
STARTING_TIER = 1
MAX_TIER = 6
MAX_ROUNDS = 20

BUY_COST = 3
SELL_REWARD = 1
ROLL_COST = 1
# Base gold to upgrade from current tier T → T+1 (wiki.gg BG table). In-game price
# is ``PlayerState.next_tier_up_cost``, which ticks down by 1 each new round until bought.
LEVEL_UP_COSTS: dict[int, int] = {1: 5, 2: 7, 3: 8, 4: 11, 5: 11}
LEVEL_UP_DISCOUNT_PER_ROUND = 1

GOLD_PER_ROUND: dict[int, int] = {
    1: 3,
    2: 4,
    3: 5,
    4: 6,
    5: 7,
    6: 8,
    7: 9,
    8: 10,
    9: 10,
    10: 10,
}
GOLD_AT_CAP = 10

DAMAGE_CAP = 7


def gold_for_round(round_number: int) -> int:
    return GOLD_PER_ROUND.get(round_number, GOLD_AT_CAP)


def base_level_up_cost(current_tier: int) -> int:
    """Gold for the next tier-up at full base price (no waiting discount)."""
    return LEVEL_UP_COSTS[current_tier]


def buy_slot_action(slot: int) -> Action:
    return Action(Action.BUY_SLOT_0 + slot)


def sell_board_action(pos: int) -> Action:
    return Action(Action.SELL_BOARD_0 + pos)


def place_hand_action(slot: int) -> Action:
    return Action(Action.PLACE_HAND_0 + slot)


def is_magnet_game_action(action_int: int) -> bool:
    return MAGNET_ACTION_BASE <= action_int < MAGNET_ACTION_BASE + NUM_MAGNET_ACTIONS


def magnet_hand_board_from_game_action(action_int: int) -> tuple[int, int]:
    off = action_int - MAGNET_ACTION_BASE
    return off // BOARD_SIZE, off % BOARD_SIZE


def magnet_game_action(hand: int, board_pos: int) -> Action:
    return Action(MAGNET_ACTION_BASE + hand * BOARD_SIZE + board_pos)


def is_discover_pick_game_action(action_int: int) -> bool:
    return int(Action.DISCOVER_PICK_0) <= action_int <= int(Action.DISCOVER_PICK_2)


def discover_pick_index(action_int: int) -> int:
    return action_int - int(Action.DISCOVER_PICK_0)


__all__ = [
    "Action",
    "MAGNET_ACTION_BASE",
    "NUM_MAGNET_ACTIONS",
    "NUM_ACTIONS",
    "MAX_SHOP_ACTIONS",
    "SHOP_SIZE",
    "BOARD_SIZE",
    "HAND_SIZE",
    "STARTING_HEALTH",
    "STARTING_GOLD",
    "STARTING_TIER",
    "MAX_TIER",
    "MAX_ROUNDS",
    "BUY_COST",
    "SELL_REWARD",
    "ROLL_COST",
    "LEVEL_UP_COSTS",
    "LEVEL_UP_DISCOUNT_PER_ROUND",
    "base_level_up_cost",
    "GOLD_PER_ROUND",
    "GOLD_AT_CAP",
    "DAMAGE_CAP",
    "gold_for_round",
    "buy_slot_action",
    "sell_board_action",
    "place_hand_action",
    "is_magnet_game_action",
    "magnet_hand_board_from_game_action",
    "magnet_game_action",
    "is_discover_pick_game_action",
    "discover_pick_index",
]

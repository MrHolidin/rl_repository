from __future__ import annotations

from enum import IntEnum


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


NUM_ACTIONS = 13
MAX_SHOP_ACTIONS = 10
SHOP_SIZE = 3
BOARD_SIZE = 4
HAND_SIZE = 3

STARTING_HEALTH = 15
STARTING_GOLD = 3
STARTING_TIER = 1
MAX_TIER = 3
MAX_ROUNDS = 15

BUY_COST = 3
SELL_REWARD = 1
ROLL_COST = 1
LEVEL_UP_COSTS = {1: 4, 2: 6}

GOLD_PER_ROUND = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
GOLD_AT_CAP = 8

DAMAGE_CAP = 7


def gold_for_round(round_number: int) -> int:
    return GOLD_PER_ROUND.get(round_number, GOLD_AT_CAP)


def buy_slot_action(slot: int) -> Action:
    return Action(Action.BUY_SLOT_0 + slot)


def sell_board_action(pos: int) -> Action:
    return Action(Action.SELL_BOARD_0 + pos)


def place_hand_action(slot: int) -> Action:
    return Action(Action.PLACE_HAND_0 + slot)


__all__ = [
    "Action",
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
    "GOLD_PER_ROUND",
    "GOLD_AT_CAP",
    "DAMAGE_CAP",
    "gold_for_round",
    "buy_slot_action",
    "sell_board_action",
    "place_hand_action",
]

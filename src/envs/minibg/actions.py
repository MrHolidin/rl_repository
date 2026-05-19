from __future__ import annotations

from enum import IntEnum

# BG-style: max visible shop slots (tier 6); lower tiers use fewer (see ``shop_offers_count``).
MAX_SHOP_SLOTS = 6
BOARD_SIZE = 7
HAND_SIZE = 5

# Minions offered per refresh by tavern tier (Hearthstone Battlegrounds).
SHOP_OFFERS_BY_TIER: dict[int, int] = {
    1: 3,
    2: 4,
    3: 4,
    4: 5,
    5: 5,
    6: 6,
}


def shop_offers_count(tavern_tier: int) -> int:
    return SHOP_OFFERS_BY_TIER.get(int(tavern_tier), MAX_SHOP_SLOTS)


class Action(IntEnum):
    BUY_SLOT_0 = 0
    BUY_SLOT_1 = 1
    BUY_SLOT_2 = 2
    BUY_SLOT_3 = 3
    BUY_SLOT_4 = 4
    BUY_SLOT_5 = 5
    SELL_BOARD_0 = 6
    SELL_BOARD_1 = 7
    SELL_BOARD_2 = 8
    SELL_BOARD_3 = 9
    SELL_BOARD_4 = 10
    SELL_BOARD_5 = 11
    SELL_BOARD_6 = 12
    ROLL = 13
    LEVEL_UP = 14
    FINISH = 15
    PLACE_HAND_0 = 16
    PLACE_HAND_1 = 17
    PLACE_HAND_2 = 18
    PLACE_HAND_3 = 19
    PLACE_HAND_4 = 20
    MAGNET_HAND_0_BOARD_0 = 21
    MAGNET_HAND_0_BOARD_1 = 22
    MAGNET_HAND_0_BOARD_2 = 23
    MAGNET_HAND_0_BOARD_3 = 24
    MAGNET_HAND_0_BOARD_4 = 25
    MAGNET_HAND_0_BOARD_5 = 26
    MAGNET_HAND_0_BOARD_6 = 27
    MAGNET_HAND_1_BOARD_0 = 28
    MAGNET_HAND_1_BOARD_1 = 29
    MAGNET_HAND_1_BOARD_2 = 30
    MAGNET_HAND_1_BOARD_3 = 31
    MAGNET_HAND_1_BOARD_4 = 32
    MAGNET_HAND_1_BOARD_5 = 33
    MAGNET_HAND_1_BOARD_6 = 34
    MAGNET_HAND_2_BOARD_0 = 35
    MAGNET_HAND_2_BOARD_1 = 36
    MAGNET_HAND_2_BOARD_2 = 37
    MAGNET_HAND_2_BOARD_3 = 38
    MAGNET_HAND_2_BOARD_4 = 39
    MAGNET_HAND_2_BOARD_5 = 40
    MAGNET_HAND_2_BOARD_6 = 41
    MAGNET_HAND_3_BOARD_0 = 42
    MAGNET_HAND_3_BOARD_1 = 43
    MAGNET_HAND_3_BOARD_2 = 44
    MAGNET_HAND_3_BOARD_3 = 45
    MAGNET_HAND_3_BOARD_4 = 46
    MAGNET_HAND_3_BOARD_5 = 47
    MAGNET_HAND_3_BOARD_6 = 48
    MAGNET_HAND_4_BOARD_0 = 49
    MAGNET_HAND_4_BOARD_1 = 50
    MAGNET_HAND_4_BOARD_2 = 51
    MAGNET_HAND_4_BOARD_3 = 52
    MAGNET_HAND_4_BOARD_4 = 53
    MAGNET_HAND_4_BOARD_5 = 54
    MAGNET_HAND_4_BOARD_6 = 55
    DISCOVER_PICK_0 = 56
    DISCOVER_PICK_1 = 57
    DISCOVER_PICK_2 = 58
    FINISH_FREEZE_SHOP = 59


NUM_ACTIONS = 60

MAGNET_ACTION_BASE = int(Action.MAGNET_HAND_0_BOARD_0)
NUM_MAGNET_ACTIONS = HAND_SIZE * BOARD_SIZE
MAX_SHOP_ACTIONS = 20

STARTING_HEALTH = 40
STARTING_GOLD = 3
STARTING_TIER = 1
MAX_TIER = 6
MAX_ROUNDS = 20

BUY_COST = 3
SELL_REWARD = 1
ROLL_COST = 1
LEVEL_UP_COSTS: dict[int, int] = {1: 5, 2: 7, 3: 8, 4: 11, 5: 11}
LEVEL_UP_COST_MAX = max(LEVEL_UP_COSTS.values())
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

DAMAGE_CAP = 15
# Max living minions per side during combat (retail BG board space).
COMBAT_BOARD_MAX = 7


def gold_for_round(round_number: int) -> int:
    return GOLD_PER_ROUND.get(round_number, GOLD_AT_CAP)


def base_level_up_cost(current_tier: int) -> int:
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
    "MAX_SHOP_SLOTS",
    "NUM_MAGNET_ACTIONS",
    "NUM_ACTIONS",
    "MAX_SHOP_ACTIONS",
    "BOARD_SIZE",
    "HAND_SIZE",
    "SHOP_OFFERS_BY_TIER",
    "shop_offers_count",
    "STARTING_HEALTH",
    "STARTING_GOLD",
    "STARTING_TIER",
    "MAX_TIER",
    "MAX_ROUNDS",
    "BUY_COST",
    "SELL_REWARD",
    "ROLL_COST",
    "LEVEL_UP_COSTS",
    "LEVEL_UP_COST_MAX",
    "LEVEL_UP_DISCOUNT_PER_ROUND",
    "base_level_up_cost",
    "GOLD_PER_ROUND",
    "GOLD_AT_CAP",
    "DAMAGE_CAP",
    "COMBAT_BOARD_MAX",
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

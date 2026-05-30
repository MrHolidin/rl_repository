"""BGLike discrete game actions (10-card hand, 7 board slots)."""

from __future__ import annotations

from enum import IntEnum
from types import ModuleType
from typing import Dict

MAX_SHOP_SLOTS = 6
BOARD_SIZE = 7
HAND_SIZE = 10

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


def _build_action_enum() -> type[IntEnum]:
    members: Dict[str, int] = {}
    n = 0
    for i in range(MAX_SHOP_SLOTS):
        members[f"BUY_SLOT_{i}"] = n
        n += 1
    for i in range(BOARD_SIZE):
        members[f"SELL_BOARD_{i}"] = n
        n += 1
    members["ROLL"] = n
    n += 1
    members["LEVEL_UP"] = n
    n += 1
    members["FINISH"] = n
    n += 1
    for i in range(HAND_SIZE):
        members[f"PLACE_HAND_{i}"] = n
        n += 1
    for h in range(HAND_SIZE):
        for b in range(BOARD_SIZE):
            members[f"MAGNET_HAND_{h}_BOARD_{b}"] = n
            n += 1
    for i in range(3):
        members[f"DISCOVER_PICK_{i}"] = n
        n += 1
    members["FINISH_FREEZE_SHOP"] = n
    n += 1
    for i in range(BOARD_SIZE):
        members[f"TARGET_BOARD_{i}"] = n
        n += 1
    return IntEnum("Action", members)


Action = _build_action_enum()

NUM_ACTIONS = int(max(a.value for a in Action)) + 1
MAGNET_ACTION_BASE = int(Action.MAGNET_HAND_0_BOARD_0)
NUM_MAGNET_ACTIONS = HAND_SIZE * BOARD_SIZE
MAX_SHOP_ACTIONS = 20

NUM_PLAYERS = 8
STARTING_HEALTH = 40
STARTING_TIER = 1
MAX_TIER = 6
MAX_ROUNDS = 50

BUY_COST = 3
SELL_REWARD = 1
ROLL_COST = 1
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
}
GOLD_AT_CAP = 10

HIGH_MODE_START_ROUND = 9
HIGH_MODE_START_TIER = 5

DAMAGE_CAP = 15
COMBAT_BOARD_MAX = 7


def gold_for_round(round_number: int) -> int:
    return GOLD_PER_ROUND.get(round_number, GOLD_AT_CAP)


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
    "BOARD_SIZE",
    "BUY_COST",
    "COMBAT_BOARD_MAX",
    "DAMAGE_CAP",
    "GOLD_AT_CAP",
    "HIGH_MODE_START_ROUND",
    "HIGH_MODE_START_TIER",
    "HAND_SIZE",
    "LEVEL_UP_COSTS",
    "LEVEL_UP_DISCOUNT_PER_ROUND",
    "MAGNET_ACTION_BASE",
    "MAX_ROUNDS",
    "MAX_SHOP_ACTIONS",
    "MAX_SHOP_SLOTS",
    "MAX_TIER",
    "NUM_ACTIONS",
    "NUM_MAGNET_ACTIONS",
    "NUM_PLAYERS",
    "ROLL_COST",
    "SELL_REWARD",
    "SHOP_OFFERS_BY_TIER",
    "STARTING_HEALTH",
    "STARTING_TIER",
    "gold_for_round",
    "discover_pick_index",
    "is_discover_pick_game_action",
    "is_magnet_game_action",
    "magnet_game_action",
    "magnet_hand_board_from_game_action",
    "shop_offers_count",
]

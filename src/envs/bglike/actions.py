"""BGLike discrete game actions (10-card hand, 7 board slots)."""

from __future__ import annotations

from enum import IntEnum
from types import ModuleType
from typing import Dict

from src.bg_catalog.layout import DEFAULT_LAYOUT
from src.bg_catalog.ruleset import DEFAULT_RULESET

# Shop shape comes from the patch layout (src/bg_catalog/layout.py); these are
# the default layout's numbers, which is what every package shipped today
# declares. The flat action-space layout below is built from them.
MAX_SHOP_SLOTS = DEFAULT_LAYOUT.max_shop_slots
BOARD_SIZE = 7
HAND_SIZE = 10

SHOP_OFFERS_BY_TIER: dict[int, int] = dict(DEFAULT_LAYOUT.shop_offers_by_tier)

NUM_DISCOVER_PICKS = DEFAULT_LAYOUT.discover_picks


def shop_offers_count(tavern_tier: int) -> int:
    return SHOP_OFFERS_BY_TIER.get(int(tavern_tier), MAX_SHOP_SLOTS)


#: Env-only action ids ``action_map`` reserves directly above the game band:
#: ``BOARD_SIZE - 1`` adjacent board swaps and one APPLY_EFFECT_SKIP.
ENV_ONLY_ACTION_SLOTS = (BOARD_SIZE - 1) + 1


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
        members[f"PLAY_HAND_{i}"] = n
        n += 1
    for h in range(HAND_SIZE):
        for b in range(BOARD_SIZE):
            members[f"MAGNET_HAND_{h}_BOARD_{b}"] = n
            n += 1
    for i in range(NUM_DISCOVER_PICKS):
        members[f"DISCOVER_PICK_{i}"] = n
        n += 1
    members["FINISH_FREEZE_SHOP"] = n
    n += 1
    for i in range(BOARD_SIZE):
        members[f"TARGET_BOARD_{i}"] = n
        n += 1
    # Appended last, and only ever appended: every member before it keeps its
    # value, so a policy trained without this one reads every other action the
    # same way. The modern patch has 65 heroes whose power the seat presses;
    # the 2021 pool is passive and never offers it.
    #
    # The flat env carves its own actions out of this same integer space, directly
    # above the game ones: ``action_map`` puts the board swaps and APPLY_EFFECT_SKIP
    # there. They were already there before HERO_POWER, and a policy trained on the
    # 2021 patch has learned those ids — so HERO_POWER goes above *them*. Appending
    # it at the enum's end would have been an append for the enum and a shift for
    # every env id after it.
    members["HERO_POWER"] = n + ENV_ONLY_ACTION_SLOTS
    # Buying the Tavern spell beside the minion row. One id because a tavern
    # offers one spell (``ruleset.tavern_spells_per_roll``), and a package that
    # offers more is caught by the layout test rather than left with an offer
    # nothing can buy. Appended above HERO_POWER, which is itself above the
    # env-only band — see NUM_CORE_ACTIONS.
    members["BUY_TAVERN_SPELL"] = n + ENV_ONLY_ACTION_SLOTS + 1
    return IntEnum("Action", members)


Action = _build_action_enum()

#: The contiguous band of game actions, below the env-only ids. This is what
#: ``action_map`` stacks its own actions on top of, and it must never move.
NUM_CORE_ACTIONS = int(Action.HERO_POWER) - ENV_ONLY_ACTION_SLOTS
#: One past the highest action id, env-only reservations included.
NUM_ACTIONS = int(max(a.value for a in Action)) + 1
MAGNET_ACTION_BASE = int(Action.MAGNET_HAND_0_BOARD_0)
NUM_MAGNET_ACTIONS = HAND_SIZE * BOARD_SIZE
# Precomputed magnet Action members: index = hand * BOARD_SIZE + board_pos.
# magnet_game_action is called ~hundreds of thousands of times in legal-action
# generation; indexing this tuple avoids constructing the IntEnum each call
# (Enum.__call__/__new__/__hash__ churn).
_MAGNET_ACTIONS = tuple(Action(MAGNET_ACTION_BASE + i) for i in range(NUM_MAGNET_ACTIONS))
MAX_SHOP_ACTIONS = 30

NUM_PLAYERS = 8
STARTING_HEALTH = 40
STARTING_TIER = 1
# Tavern tier ceiling — the default ruleset's value (a patch package raises it
# via meta.json["ruleset"]["max_tier"], which is how tier 7 arrives).
MAX_TIER = DEFAULT_RULESET.max_tier
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
    return _MAGNET_ACTIONS[hand * BOARD_SIZE + board_pos]


DISCOVER_PICK_BASE = int(Action.DISCOVER_PICK_0)
DISCOVER_PICK_LAST = DISCOVER_PICK_BASE + NUM_DISCOVER_PICKS - 1


def is_discover_pick_game_action(action_int: int) -> bool:
    return DISCOVER_PICK_BASE <= action_int <= DISCOVER_PICK_LAST


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

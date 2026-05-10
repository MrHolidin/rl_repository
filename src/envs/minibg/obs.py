from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .actions import (
    BOARD_SIZE,
    GOLD_AT_CAP,
    MAX_ROUNDS,
    MAX_SHOP_ACTIONS,
    MAX_TIER,
    SHOP_SIZE,
    STARTING_HEALTH,
    gold_for_round,
)
from .effects import Keyword, Trigger
from .state import MiniBGState, Minion


CARD_IDS_FOR_ENCODING: tuple[str, ...] = (
    "recruit",
    "guard",
    "buffer",
    "bruiser",
    "shield_bot",
    "pack_rat",
    "big_guy",
    "commander",
    "summoner",
    "mentor",
)
NUM_CARD_IDS = len(CARD_IDS_FOR_ENCODING)
CARD_ID_TO_INDEX = {cid: i for i, cid in enumerate(CARD_IDS_FOR_ENCODING)}

# slot vector layout
#   [0]                     presence
#   [1 .. 1+NUM_CARD_IDS)   card_id one-hot (10 dims)
#   [11 .. 14)              tier one-hot (3 dims, tiers 1..3)
#   [14] base_attack / 5
#   [15] base_health / 5
#   [16] bonus_attack / 5
#   [17] bonus_health / 5
#   [18] keyword TAUNT
#   [19] keyword SHIELD
#   [20] runtime has_shield
#   [21] has ON_BUY ability
#   [22] has ON_DEATH ability
#   [23] has AURA ability
#   [24] has ON_TURN_END ability
SLOT_DIM = 25
GLOBAL_DIM = 10
LAST_BATTLE_DIM = 1
OBS_DIM = (
    GLOBAL_DIM
    + BOARD_SIZE * SLOT_DIM
    + SHOP_SIZE * SLOT_DIM
    + BOARD_SIZE * SLOT_DIM
    + LAST_BATTLE_DIM
)

_STAT_NORM = 5.0


def encode_minion(minion: Optional[Minion]) -> np.ndarray:
    v = np.zeros(SLOT_DIM, dtype=np.float32)
    if minion is None:
        return v
    v[0] = 1.0
    cid_idx = CARD_ID_TO_INDEX.get(minion.card_id)
    if cid_idx is not None:
        v[1 + cid_idx] = 1.0
    if 1 <= minion.tier <= MAX_TIER:
        v[11 + (minion.tier - 1)] = 1.0
    v[14] = minion.base_attack / _STAT_NORM
    v[15] = minion.base_health / _STAT_NORM
    v[16] = minion.bonus_attack / _STAT_NORM
    v[17] = minion.bonus_health / _STAT_NORM
    v[18] = 1.0 if Keyword.TAUNT in minion.keywords else 0.0
    v[19] = 1.0 if Keyword.SHIELD in minion.keywords else 0.0
    v[20] = 1.0 if minion.has_shield else 0.0
    triggers = {ab.trigger for ab in minion.abilities}
    v[21] = 1.0 if Trigger.ON_BUY in triggers else 0.0
    v[22] = 1.0 if Trigger.ON_DEATH in triggers else 0.0
    v[23] = 1.0 if Trigger.AURA in triggers else 0.0
    v[24] = 1.0 if Trigger.ON_TURN_END in triggers else 0.0
    return v


def encode_slots(
    minions: Sequence[Optional[Minion]], num_slots: int
) -> np.ndarray:
    out = np.zeros((num_slots, SLOT_DIM), dtype=np.float32)
    for i in range(min(num_slots, len(minions))):
        if minions[i] is not None:
            out[i] = encode_minion(minions[i])
    return out


def i_have_round_initiative(state: MiniBGState, player_idx: int) -> float:
    """Tie-breaker: who attacks first when boards have equal size."""
    is_odd_round = state.round_number % 2 == 1
    if is_odd_round:
        return 1.0 if state.initiative_player == player_idx else 0.0
    return 1.0 if state.initiative_player != player_idx else 0.0


def build_observation(
    state: MiniBGState,
    player_idx: int,
    last_battle_signed: float,
    enemy_last_seen_board: Optional[List[Minion]],
) -> np.ndarray:
    me = state.players[player_idx]
    enemy = state.players[1 - player_idx]

    actions_left = MAX_SHOP_ACTIONS - me.shop_actions_used

    globals_arr = np.array(
        [
            state.round_number / MAX_ROUNDS,
            me.health / STARTING_HEALTH,
            enemy.health / STARTING_HEALTH,
            me.gold / GOLD_AT_CAP,
            gold_for_round(state.round_number) / GOLD_AT_CAP,
            me.tavern_tier / MAX_TIER,
            enemy.tavern_tier / MAX_TIER,
            actions_left / MAX_SHOP_ACTIONS,
            len(me.board) / BOARD_SIZE,
            i_have_round_initiative(state, player_idx),
        ],
        dtype=np.float32,
    )

    own_board = encode_slots(list(me.board), BOARD_SIZE)
    shop = encode_slots(list(me.shop), SHOP_SIZE)
    enemy_board = encode_slots(
        list(enemy_last_seen_board) if enemy_last_seen_board else [],
        BOARD_SIZE,
    )
    last_battle = np.array([last_battle_signed], dtype=np.float32)

    return np.concatenate(
        [
            globals_arr,
            own_board.flatten(),
            shop.flatten(),
            enemy_board.flatten(),
            last_battle,
        ]
    )


__all__ = [
    "CARD_IDS_FOR_ENCODING",
    "NUM_CARD_IDS",
    "CARD_ID_TO_INDEX",
    "SLOT_DIM",
    "GLOBAL_DIM",
    "LAST_BATTLE_DIM",
    "OBS_DIM",
    "encode_minion",
    "encode_slots",
    "i_have_round_initiative",
    "build_observation",
]

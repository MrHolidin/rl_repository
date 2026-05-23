"""Self-centric observation for 8p lobby (minibg slot encoding, no enemy board)."""

from __future__ import annotations

from typing import List, Mapping, Optional

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.envs.minibg import obs as minibg_obs
from src.envs.minibg.obs import (
    GLOBAL_DIM,
    LAST_BATTLE_DIM,
    PENDING_CHOICE_DIM,
    PHASE_DIM,
    SLOT_DIM,
    encode_minion,
    encode_pending_choice,
)

from .actions import (
    BOARD_SIZE,
    GOLD_AT_CAP,
    HAND_SIZE,
    LEVEL_UP_COSTS,
    MAX_ROUNDS,
    MAX_SHOP_ACTIONS,
    MAX_SHOP_SLOTS,
    MAX_TIER,
    NUM_PLAYERS,
    STARTING_HEALTH,
    gold_for_round,
)
from .placement import is_seat_eliminated
from .state import BGLikeState

LEVEL_UP_COST_MAX = max(LEVEL_UP_COSTS.values())

# Re-export slot layout constants for nets/tests.
PRESENCE_OFFSET = minibg_obs.PRESENCE_OFFSET
SLOT_DIM = minibg_obs.SLOT_DIM
HAND_LEN = HAND_SIZE

OBS_DIM = (
    GLOBAL_DIM
    + BOARD_SIZE * SLOT_DIM
    + MAX_SHOP_SLOTS * SLOT_DIM
    + HAND_LEN * SLOT_DIM
    + LAST_BATTLE_DIM
    + PHASE_DIM
    + PENDING_CHOICE_DIM
)


def _count_non_golden_same_card_hand(
    player: PlayerState, card_id: str, *, exclude_hand_idx: Optional[int] = None
) -> int:
    n = 0
    for i, hm in enumerate(player.hand):
        if i == exclude_hand_idx or hm is None or hm.is_golden:
            continue
        if hm.card_id == card_id:
            n += 1
    return n


def _count_non_golden_same_card_board(
    player: PlayerState,
    card_id: str,
    *,
    exclude_board_idx: Optional[int] = None,
) -> int:
    n = 0
    for i, m in enumerate(player.board):
        if i == exclude_board_idx or m.is_golden:
            continue
        if m.card_id == card_id:
            n += 1
    return n


def _encode_own_board(
    player: PlayerState,
    *,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    out = np.zeros((BOARD_SIZE, SLOT_DIM), dtype=np.float32)
    for i, m in enumerate(player.board):
        if i >= BOARD_SIZE:
            break
        nh = _count_non_golden_same_card_hand(player, m.card_id)
        nb = _count_non_golden_same_card_board(
            player, m.card_id, exclude_board_idx=i
        )
        out[i] = encode_minion(
            m,
            same_non_golden_hand_elsewhere=nh,
            same_non_golden_board_elsewhere=nb,
            card_id_to_dense=card_id_to_dense,
        )
    return out


def _encode_hand(
    player: PlayerState,
    *,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    out = np.zeros((HAND_LEN, SLOT_DIM), dtype=np.float32)
    for i, hm in enumerate(player.hand):
        if i >= HAND_LEN or hm is None:
            continue
        nh = _count_non_golden_same_card_hand(
            player, hm.card_id, exclude_hand_idx=i
        )
        nb = _count_non_golden_same_card_board(player, hm.card_id)
        out[i] = encode_minion(
            hm,
            same_non_golden_hand_elsewhere=nh,
            same_non_golden_board_elsewhere=nb,
            card_id_to_dense=card_id_to_dense,
        )
    return out


def _encode_shop(
    player: PlayerState,
    *,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    out = np.zeros((MAX_SHOP_SLOTS, SLOT_DIM), dtype=np.float32)
    for i, m in enumerate(player.shop):
        if m is None or i >= MAX_SHOP_SLOTS:
            continue
        nh = _count_non_golden_same_card_hand(player, m.card_id)
        nb = _count_non_golden_same_card_board(player, m.card_id)
        out[i] = encode_minion(
            m,
            same_non_golden_hand_elsewhere=nh,
            same_non_golden_board_elsewhere=nb,
            card_id_to_dense=card_id_to_dense,
        )
    return out


def i_have_round_initiative(state: BGLikeState, seat: int) -> float:
    is_odd_round = state.round_number % 2 == 1
    if is_odd_round:
        return 1.0 if state.initiative_player == seat else 0.0
    return 1.0 if state.initiative_player != seat else 0.0


def build_observation(
    state: BGLikeState,
    seat: int,
    last_battle_signed: float,
    *,
    is_my_turn: bool,
    patch: PatchContext,
    rl_pending=None,
) -> np.ndarray:
    me = state.players[seat]
    card_id_to_dense = patch.card_id_to_dense
    meta = patch.meta
    actions_left = MAX_SHOP_ACTIONS - me.shop_actions_used
    tier_up_cost = (
        0.0 if me.tavern_tier >= MAX_TIER else float(me.next_tier_up_cost)
    )

    globals_core = np.array(
        [
            state.round_number / MAX_ROUNDS,
            me.health / STARTING_HEALTH,
            me.gold / GOLD_AT_CAP,
            gold_for_round(state.round_number) / GOLD_AT_CAP,
            me.tavern_tier / MAX_TIER,
            actions_left / MAX_SHOP_ACTIONS,
            len(me.board) / BOARD_SIZE,
            len(state.alive) / float(NUM_PLAYERS),
            i_have_round_initiative(state, seat),
            tier_up_cost / LEVEL_UP_COST_MAX,
            1.0 if is_my_turn else 0.0,
        ],
        dtype=np.float32,
    )
    globals_arr = np.concatenate(
        [
            globals_core,
            minibg_obs._encode_shop_rotation_globals(
                state.shop_excluded_race,
                rotation_tribes=meta.rotation_tribes,
                cnt_active_shop_tribes=meta.cnt_active_shop_tribes,
            ),
        ]
    )

    own_board = _encode_own_board(me, card_id_to_dense=card_id_to_dense)
    shop = _encode_shop(me, card_id_to_dense=card_id_to_dense)
    hand = _encode_hand(me, card_id_to_dense=card_id_to_dense)
    last_battle = np.array([float(last_battle_signed)], dtype=np.float32)
    phase_val = (
        1.0
        if me.phase == PlayerPhase.SHOP and me.shop_actions_used >= MAX_SHOP_ACTIONS
        else 0.0
    )
    phase_arr = np.array([phase_val], dtype=np.float32)
    pending_arr = encode_pending_choice(
        me, rl_pending=rl_pending, card_id_to_dense=card_id_to_dense
    )

    return np.concatenate(
        [
            globals_arr,
            own_board.flatten(),
            shop.flatten(),
            hand.flatten(),
            last_battle,
            phase_arr,
            pending_arr,
        ]
    )


__all__ = [
    "BOARD_SIZE",
    "HAND_LEN",
    "OBS_DIM",
    "SLOT_DIM",
    "build_observation",
    "i_have_round_initiative",
]

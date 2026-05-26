"""Self-centric observation for 8p lobby (minibg slot encoding, no enemy board)."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.bg_catalog.cards import normalize_shop_excluded_races
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion, Race
from src.bg_lobby.match_types import EliminatedSnapshot
from src.bg_lobby.pairing import peek_next_opponent
from src.bg_lobby.player import BATTLE_HISTORY_LEN, PlayerPhase, PlayerState
from src.envs.minibg import obs as minibg_obs
from src.envs.minibg.obs import (
    GLOBAL_DIM,
    LAST_BATTLE_DIM,
    PENDING_CHOICE_DIM,
    PHASE_DIM,
    RACE_ONEHOT_DIM,
    SLOT_DIM,
    _RACE_ORDER,
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
    ROLL_COST,
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

# Lobby panel: HP / alive flag / tribe-lock one-hot per opponent (excluding self).
MAX_OPPS = NUM_PLAYERS - 1
TRIBE_LOCK_THRESHOLD = 4  # ≥4 minions of a tribe → that tribe is shown (real-BG rule).
LOBBY_PANEL_DIM = (
    MAX_OPPS  # opp_hp_sorted (descending)
    + MAX_OPPS  # opp_alive_mask
    + MAX_OPPS * RACE_ONEHOT_DIM  # opp_tribe_lock one-hot (synchronised with hp_sorted)
)
# Last-N battle deltas for self + next opponent, each with a validity mask and a
# single bit marking whether the next opp is deterministically known.
BATTLE_HISTORY_OBS_DIM = (
    BATTLE_HISTORY_LEN  # self battles
    + BATTLE_HISTORY_LEN  # self mask
    + BATTLE_HISTORY_LEN  # next opp battles
    + BATTLE_HISTORY_LEN  # next opp mask
    + 1  # next opp known flag
)

# bglike has 3 extra economy globals not in minibg: effective roll cost,
# free roll charges, and elemental shop bonus.
BGLIKE_GLOBAL_CORE_DIM = minibg_obs.GLOBAL_CORE_DIM + 3  # 11 + 3 = 14
BGLIKE_GLOBAL_DIM = (
    BGLIKE_GLOBAL_CORE_DIM
    + minibg_obs.SHOP_ROTATION_OBS_DIM
    + LOBBY_PANEL_DIM
    + BATTLE_HISTORY_OBS_DIM
)

OBS_DIM = (
    BGLIKE_GLOBAL_DIM
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


def encode_board_minions(
    board: Sequence,
    *,
    card_id_to_dense: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    """Encode a bare list/tuple of ``Minion`` objects as a (BOARD_SIZE, SLOT_DIM)
    slot tensor, without the "same card in hand / on board elsewhere" counts.

    Used by the battle-prediction head to encode pre-combat board snapshots
    where we don't carry the surrounding player state (hand, shop) — these
    counts are zero for snapshot encoding. The main observation builder uses
    ``_encode_own_board`` instead because it has full PlayerState context.
    """
    out = np.zeros((BOARD_SIZE, SLOT_DIM), dtype=np.float32)
    for i, m in enumerate(board):
        if i >= BOARD_SIZE or m is None:
            break
        out[i] = encode_minion(
            m,
            same_non_golden_hand_elsewhere=0,
            same_non_golden_board_elsewhere=0,
            card_id_to_dense=card_id_to_dense,
        )
    return out


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
        frozen = bool(player.shop_frozen[i]) if i < len(player.shop_frozen) else False
        out[i] = encode_minion(
            m,
            same_non_golden_hand_elsewhere=nh,
            same_non_golden_board_elsewhere=nb,
            card_id_to_dense=card_id_to_dense,
            is_frozen=frozen,
        )
    return out


def _tribe_lock_one_hot(tribe_counts: Mapping[Race, int]) -> np.ndarray:
    """One-hot of the tribe with the highest count meeting ``TRIBE_LOCK_THRESHOLD``.

    Ties broken by ``_RACE_ORDER`` index (stable across runs). Returns all-zeros
    if no tribe is locked. ``Race.ALL`` contribution is already baked into
    ``last_round_tribe_counts`` upstream — this just picks the dominant tribe.
    """
    out = np.zeros(RACE_ONEHOT_DIM, dtype=np.float32)
    best_race: Optional[Race] = None
    best_count = 0
    best_order = 0
    for race, count in tribe_counts.items():
        if count < TRIBE_LOCK_THRESHOLD:
            continue
        try:
            order = _RACE_ORDER.index(race)
        except ValueError:
            continue
        if (count > best_count) or (count == best_count and order < best_order):
            best_race = race
            best_count = count
            best_order = order
    if best_race is not None:
        out[best_order] = 1.0
    return out


def _opponent_panel(
    state: BGLikeState, seat: int
) -> np.ndarray:
    """``MAX_OPPS`` rows of (HP-sorted desc, alive flag, tribe-lock one-hot).

    Eliminated players keep their last snapshot (``EliminatedSnapshot.last_board``
    drives tribe-lock; HP is 0 with alive_mask=0). Remaining slots (lobby smaller
    than ``NUM_PLAYERS``) are zero-padded.
    """
    elim_by_seat: Dict[int, EliminatedSnapshot] = {
        snap.seat: snap for snap in state.eliminated
    }
    rows: List[tuple] = []
    for i, p in enumerate(state.players):
        if i == seat:
            continue
        is_alive = i in state.alive
        if is_alive:
            hp = max(int(p.health), 0)
            tribe_counts = p.last_round_tribe_counts
        else:
            hp = 0
            snap = elim_by_seat.get(i)
            if snap is not None:
                tribe_counts = _count_board_tribes(snap.last_board)
            else:
                # Player never lived (lobby smaller than NUM_PLAYERS); skip.
                continue
        rows.append((hp, is_alive, tribe_counts))

    # Sort by HP descending; dead players (hp=0) fall to the tail naturally.
    rows.sort(key=lambda r: (-r[0], 0 if r[1] else 1))

    hp_block = np.zeros(MAX_OPPS, dtype=np.float32)
    alive_block = np.zeros(MAX_OPPS, dtype=np.float32)
    tribe_block = np.zeros((MAX_OPPS, RACE_ONEHOT_DIM), dtype=np.float32)
    for j, (hp, is_alive, tribe_counts) in enumerate(rows[:MAX_OPPS]):
        hp_block[j] = float(hp) / float(STARTING_HEALTH)
        alive_block[j] = 1.0 if is_alive else 0.0
        tribe_block[j] = _tribe_lock_one_hot(tribe_counts)
    return np.concatenate([hp_block, alive_block, tribe_block.flatten()])


def _count_board_tribes(board: Sequence[Minion]) -> Dict[Race, int]:
    """Re-implementation of ``eight_player._count_board_tribes`` for eliminated
    snapshots (whose tribe-counts aren't precomputed). ``Race.ALL`` counts toward
    every specific tribe; ``None`` is ignored."""
    counts: Dict[Race, int] = {}
    all_count = 0
    for m in board:
        r = m.race
        if r is None:
            continue
        if r == Race.ALL:
            all_count += 1
            continue
        counts[r] = counts.get(r, 0) + 1
    if all_count:
        for r in (
            Race.BEAST,
            Race.DEMON,
            Race.MECHANICAL,
            Race.MURLOC,
            Race.DRAGON,
            Race.PIRATE,
            Race.ELEMENTAL,
        ):
            counts[r] = counts.get(r, 0) + all_count
    return counts


def _encode_battle_history(history: Sequence[float]) -> tuple:
    """Pad/clip ``history`` (oldest-first, length ≤ BATTLE_HISTORY_LEN) into a
    fixed-length value vector + boolean mask. Mask is 1 where a real battle
    contributed, 0 where the slot is empty (e.g. early game)."""
    values = np.zeros(BATTLE_HISTORY_LEN, dtype=np.float32)
    mask = np.zeros(BATTLE_HISTORY_LEN, dtype=np.float32)
    n = min(len(history), BATTLE_HISTORY_LEN)
    if n > 0:
        # Most recent in the last slot — left-pad with zeros so position is stable
        # across rounds (slot[-1] always = most recent battle).
        values[BATTLE_HISTORY_LEN - n :] = np.asarray(history[-n:], dtype=np.float32)
        mask[BATTLE_HISTORY_LEN - n :] = 1.0
    return values, mask


def _battle_history_block(state: BGLikeState, seat: int) -> np.ndarray:
    self_vals, self_mask = _encode_battle_history(state.players[seat].battle_history)
    next_opp = peek_next_opponent(
        state.alive,
        state.recent_opponents,
        n_seats=len(state.players),
        full_lobby_cycle_round=state.full_lobby_cycle_round,
        seat=seat,
    )
    if next_opp is not None and 0 <= next_opp < len(state.players):
        opp_vals, opp_mask = _encode_battle_history(state.players[next_opp].battle_history)
        known = 1.0
    else:
        opp_vals = np.zeros(BATTLE_HISTORY_LEN, dtype=np.float32)
        opp_mask = np.zeros(BATTLE_HISTORY_LEN, dtype=np.float32)
        known = 0.0
    return np.concatenate(
        [self_vals, self_mask, opp_vals, opp_mask, np.array([known], dtype=np.float32)]
    )


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

    effective_roll_cost = float(
        me.next_roll_cost_override if me.next_roll_cost_override is not None else ROLL_COST
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
            # patch-74257 economy globals
            effective_roll_cost / float(ROLL_COST),
            min(float(me.free_roll_charges), 5.0) / 5.0,
            float(me.shop_elemental_bonus) / 4.0,
        ],
        dtype=np.float32,
    )
    globals_arr = np.concatenate(
        [
            globals_core,
            minibg_obs._encode_shop_rotation_globals(
                state.shop_excluded_race,
                rotation_tribes=meta.rotation_tribes,
                cnt_active_shop_tribes=max(
                    0,
                    len(meta.rotation_tribes)
                    - len(normalize_shop_excluded_races(state.shop_excluded_race)),
                ),
            ),
            _opponent_panel(state, seat),
            _battle_history_block(state, seat),
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
    "BATTLE_HISTORY_OBS_DIM",
    "BGLIKE_GLOBAL_CORE_DIM",
    "BGLIKE_GLOBAL_DIM",
    "BOARD_SIZE",
    "encode_board_minions",
    "HAND_LEN",
    "LOBBY_PANEL_DIM",
    "MAX_OPPS",
    "OBS_DIM",
    "SLOT_DIM",
    "TRIBE_LOCK_THRESHOLD",
    "build_observation",
    "i_have_round_initiative",
]

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..action_map import (
    A_BUY_BASE,
    A_SELECT_ORDER_BASE,
    A_SELL_BASE,
    NUM_ENV_ACTIONS,
    PERMUTATIONS_4,
    legal_order_indices,
)
from ..actions import BOARD_SIZE, BUY_COST, LEVEL_UP_COSTS, MAX_TIER, SHOP_SIZE
from ..effects import Keyword, Trigger
from ..state import Minion, PlayerState

BASE_PRIORITY: dict[str, int] = {
    "recruit": 10,
    "guard": 12,
    "buffer": 8,
    "bruiser": 20,
    "shield_bot": 22,
    "pack_rat": 21,
    "big_guy": 32,
    "commander": 30,
    "summoner": 31,
    "mentor": 34,
}

GOOD_BUFFER_TARGET_IDS = frozenset(
    {"recruit", "guard", "bruiser", "shield_bot", "pack_rat", "big_guy", "commander", "summoner"}
)

BUFFER_T2_GOOD_TARGET_IDS = frozenset(
    {"shield_bot", "pack_rat", "bruiser", "guard", "big_guy", "summoner"}
)

TOKEN_BUFFER_OK_IDS = frozenset(
    {"pack_rat", "summoner", "guard", "shield_bot"}
)


def count_non_buffer_minions(board: Sequence[Minion]) -> int:
    return sum(1 for m in board if m.card_id != "buffer")


def buffer_is_good(board: Sequence[Minion]) -> bool:
    return count_non_buffer_minions(board) >= 1


def has_other_minion(board: Sequence[Minion]) -> bool:
    return len(board) >= 1


def has_good_buffer_target(board: Sequence[Minion]) -> bool:
    return any(m.card_id in GOOD_BUFFER_TARGET_IDS for m in board)


def keyword_effect_bonus(m: Minion) -> int:
    b = 0
    if Keyword.TAUNT in m.keywords:
        b += 2
    if Keyword.SHIELD in m.keywords:
        b += 2
    for ab in m.abilities:
        if ab.trigger == Trigger.ON_DEATH:
            b += 2
        elif ab.trigger == Trigger.AURA:
            b += 2
        elif ab.trigger == Trigger.ON_BUY:
            b += 1
        elif ab.trigger == Trigger.ON_TURN_END:
            b += 2
    return b


def minion_effective_stats(m: Minion) -> int:
    return m.raw_attack + m.max_health


def board_value(board: Sequence[Minion]) -> int:
    s = 0
    for m in board:
        s += (
            m.raw_attack
            + m.max_health
            + m.tier * 2
            + keyword_effect_bonus(m)
        )
    return s


def shop_minions_with_slots(
    shop: Sequence[Optional[Minion]],
) -> List[Tuple[int, Minion]]:
    out: List[Tuple[int, Minion]] = []
    for i, m in enumerate(shop):
        if m is not None and i < SHOP_SIZE:
            out.append((i, m))
    return out


def legal_env_indices(mask: np.ndarray) -> List[int]:
    return [i for i in range(NUM_ENV_ACTIONS) if bool(mask[i])]


def can_afford_buy(p: PlayerState) -> bool:
    return p.gold >= BUY_COST


def can_afford_level(p: PlayerState) -> bool:
    if p.tavern_tier >= MAX_TIER:
        return False
    cost = LEVEL_UP_COSTS[p.tavern_tier]
    return p.gold >= cost


def board_full(p: PlayerState) -> bool:
    return len(p.board) >= BOARD_SIZE


def reset_shop_counters_if_needed(
    p: PlayerState, rolls_ref: list[int], last_used_ref: list[int]
) -> None:
    if p.shop_actions_used == 0:
        rolls_ref[0] = 0
    last_used_ref[0] = p.shop_actions_used


def score_shop_minion_base(card_id: str) -> int:
    return BASE_PRIORITY.get(card_id, 5)


def score_minion_on_board(m: Minion) -> float:
    cid = m.card_id
    v = float(
        m.raw_attack
        + m.max_health
        + m.tier * 2
        + keyword_effect_bonus(m)
    )
    if cid == "buffer":
        if m.bonus_attack + m.bonus_health >= 3:
            v += 6.0
        else:
            v -= 4.0
    if cid == "recruit":
        v -= 1.0
    if cid == "guard" and m.raw_attack + m.max_health <= 5:
        v -= 0.5
    return v


def worst_board_sell_index(p: PlayerState) -> Optional[int]:
    if not p.board:
        return None
    scores = [(i, score_minion_on_board(m)) for i, m in enumerate(p.board)]
    scores.sort(key=lambda t: (t[1], -t[0]))
    return scores[0][0]


def order_key_token(old_idx: int, board: Sequence[Minion]) -> Tuple[float, int]:
    m = board[old_idx]
    w = 0.0
    if Keyword.TAUNT in m.keywords:
        w -= 100.0
    if m.card_id == "commander":
        w += 50.0
    if m.card_id == "mentor":
        w += 45.0
    if any(ab.trigger == Trigger.ON_DEATH for ab in m.abilities):
        w -= 15.0
    if m.card_id == "buffer":
        w += 5.0
    return (w, old_idx)


def order_key_default(old_idx: int, board: Sequence[Minion]) -> Tuple[float, int]:
    m = board[old_idx]
    w = 0.0
    if Keyword.TAUNT in m.keywords:
        w -= 30.0
    if m.card_id in ("commander", "mentor"):
        w += 20.0
    return (w, old_idx)


def perm_from_desired_order(k: int, order: Sequence[int]) -> Tuple[int, int, int, int]:
    if len(order) != k:
        raise ValueError("order length mismatch")
    perm_list = list(order) + list(range(k, BOARD_SIZE))
    return (perm_list[0], perm_list[1], perm_list[2], perm_list[3])


def find_select_action_for_perm(
    k: int, desired_perm: Tuple[int, int, int, int], mask: np.ndarray
) -> Optional[int]:
    legal = legal_order_indices(k)
    for j in legal:
        if PERMUTATIONS_4[j] == desired_perm:
            a = A_SELECT_ORDER_BASE + j
            if a < NUM_ENV_ACTIONS and bool(mask[a]):
                return a
    for j in legal:
        a = A_SELECT_ORDER_BASE + j
        if bool(mask[a]):
            return a
    return None


def choose_final_order(
    board: List[Minion],
    mask: np.ndarray,
    key_fn: Callable[[int, Sequence[Minion]], Tuple[float, int]],
) -> int:
    k = len(board)
    if k == 0:
        return A_SELECT_ORDER_BASE
    order = sorted(range(k), key=lambda oi: key_fn(oi, board))
    desired = perm_from_desired_order(k, order)
    a = find_select_action_for_perm(k, desired, mask)
    if a is not None:
        return a
    return A_SELECT_ORDER_BASE


def best_buy_slot(
    p: PlayerState,
    mask: np.ndarray,
    score_fn: Callable[[Minion, PlayerState], float],
) -> Optional[int]:
    best_slot: Optional[int] = None
    best_score = -1e18
    for slot, m in shop_minions_with_slots(p.shop):
        a = A_BUY_BASE + slot
        if not bool(mask[a]):
            continue
        sc = score_fn(m, p)
        if sc > best_score:
            best_score = sc
            best_slot = slot
    return best_slot


def incoming_shop_minion(p: PlayerState, slot: int) -> Optional[Minion]:
    if 0 <= slot < len(p.shop):
        return p.shop[slot]
    return None

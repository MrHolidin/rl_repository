from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..action_map import (
    A_FINISH,
    A_FINISH_FREEZE_SHOP,
    A_SWAP_BOARD_0,
    NUM_ENV_ACTIONS,
    NUM_SWAP_ADJ,
)
from ..actions import BOARD_SIZE
from ..state import Minion

from .value_model import order_key_structured


def legal_env_indices(mask: np.ndarray) -> List[int]:
    return [i for i in range(NUM_ENV_ACTIONS) if bool(mask[i])]


def masked_finish(mask: np.ndarray) -> int:
    """FINISH / FINISH_FREEZE_SHOP only when set in ``mask``; else first legal action."""
    if bool(mask[A_FINISH]):
        return int(A_FINISH)
    if bool(mask[A_FINISH_FREEZE_SHOP]):
        return int(A_FINISH_FREEZE_SHOP)
    legal = legal_env_indices(mask)
    if not legal:
        raise RuntimeError("masked_finish: legal_actions_mask has no True entries")
    return int(legal[0])


def board_full(board_len: int) -> bool:
    return board_len >= BOARD_SIZE


def order_key_token(old_idx: int, board: Sequence[Minion | None]) -> Tuple[float, int]:
    return order_key_structured(old_idx, board)


def order_key_default(old_idx: int, board: Sequence[Minion | None]) -> Tuple[float, int]:
    m = board[old_idx]
    assert m is not None
    return (-float(m.raw_attack), old_idx)


def target_board_from_perm(
    board: List[Minion], perm: Tuple[int, ...]
) -> List[Minion]:
    k = len(board)
    return [board[p] for p in perm if p < k]


def perm_from_desired_order(k: int, order: Sequence[int]) -> Tuple[int, ...]:
    if len(order) != k:
        raise ValueError("order length mismatch")
    used = set(order)
    tail = [j for j in range(BOARD_SIZE) if j not in used]
    return tuple(list(order) + tail)


def choose_one_swap_toward_perm(
    board: List[Minion], mask: np.ndarray, perm: Tuple[int, ...]
) -> int:
    k = len(board)
    if k <= 1:
        return masked_finish(mask)
    target = target_board_from_perm(board, perm)
    for i in range(k):
        if board[i] is not target[i]:
            want = target[i]
            j = next((jj for jj in range(k) if board[jj] is want), None)
            if j is None:
                return masked_finish(mask)
            if j > i:
                si = j - 1
            else:
                si = j
            if (
                0 <= si < NUM_SWAP_ADJ
                and si + 1 < k
                and bool(mask[A_SWAP_BOARD_0 + si])
            ):
                return int(A_SWAP_BOARD_0 + si)
            return masked_finish(mask)
    return masked_finish(mask)


def choose_final_order(
    board: List[Minion],
    mask: np.ndarray,
    key_fn: Callable[[int, Sequence[Minion | None]], Tuple[float, int]],
) -> int:
    k = len(board)
    if k == 0:
        return masked_finish(mask)
    order = sorted(range(k), key=lambda oi: key_fn(oi, board))
    perm = perm_from_desired_order(k, order)
    return choose_one_swap_toward_perm(board, mask, perm)

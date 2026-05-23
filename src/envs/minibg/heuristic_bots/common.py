from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..action_map import (
    A_FINISH,
    A_FINISH_FREEZE_SHOP,
    A_SWAP_BOARD_0,
    NUM_ENV_ACTIONS,
    NUM_SWAP_ADJ,
    is_swap_board,
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
    non_swap = [a for a in legal if not is_swap_board(a)]
    if non_swap:
        return int(non_swap[0])
    return int(legal[0])


def board_full(board_len: int) -> bool:
    return board_len >= BOARD_SIZE


def order_key_token(
    old_idx: int, board: Sequence[Minion | None]
) -> Tuple[float, str, float, float]:
    return order_key_structured(old_idx, board)


def order_key_default(
    old_idx: int, board: Sequence[Minion | None]
) -> Tuple[float, str, float, float]:
    m = board[old_idx]
    assert m is not None
    return (-float(m.raw_attack), m.card_id, float(m.raw_attack), float(m.max_health))


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


def _perm_mismatch_count(board: List[Minion], target: List[Minion]) -> int:
    return sum(1 for i in range(len(board)) if board[i] is not target[i])


def _board_after_adj_swap(board: List[Minion], si: int) -> List[Minion]:
    out = list(board)
    out[si], out[si + 1] = out[si + 1], out[si]
    return out


def _canonical_board_order(
    board: List[Minion],
    key_fn: Callable[[int, Sequence[Minion | None]], tuple],
) -> List[Minion]:
    return [
        board[i]
        for i in sorted(
            range(len(board)),
            key=lambda oi: (key_fn(oi, board), id(board[oi])),
        )
    ]


def _board_matches_target(board: List[Minion], target: List[Minion]) -> bool:
    return all(board[i] is target[i] for i in range(len(board)))


def choose_one_swap_toward_target(
    board: List[Minion], mask: np.ndarray, target: List[Minion]
) -> int:
    k = len(board)
    if k <= 1:
        return masked_finish(mask)
    dist_before = _perm_mismatch_count(board, target)
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
                after = _board_after_adj_swap(board, si)
                dist_after = _perm_mismatch_count(after, target)
                if dist_after >= dist_before:
                    return masked_finish(mask)
                back = _board_after_adj_swap(after, si)
                if dist_after > 0 and all(back[i] is board[i] for i in range(k)):
                    return masked_finish(mask)
                return int(A_SWAP_BOARD_0 + si)
            return masked_finish(mask)
    return masked_finish(mask)


def choose_one_swap_toward_perm(
    board: List[Minion], mask: np.ndarray, perm: Tuple[int, ...]
) -> int:
    return choose_one_swap_toward_target(
        board, mask, target_board_from_perm(board, perm)
    )


def choose_final_order(
    board: List[Minion],
    mask: np.ndarray,
    key_fn: Callable[[int, Sequence[Minion | None]], Tuple[float, str, float, float]],
) -> int:
    k = len(board)
    if k == 0:
        return masked_finish(mask)
    target = _canonical_board_order(board, key_fn)
    if _board_matches_target(board, target):
        return masked_finish(mask)
    return choose_one_swap_toward_target(board, mask, target)

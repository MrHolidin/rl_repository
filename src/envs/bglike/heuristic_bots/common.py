"""Shared helpers for BGLike heuristic bots (env-level board swaps in shop)."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

from src.bg_lobby.player import Minion

from ..action_map import (
    A_FINISH,
    A_FINISH_FREEZE_SHOP,
    A_APPLY_EFFECT_SKIP,
    A_SWAP_BOARD_0,
    A_TARGET_BOARD_BASE,
    NUM_ENV_ACTIONS,
    NUM_SWAP_ADJ,
    is_apply_effect_skip,
    is_target_board,
)
from ..actions import BOARD_SIZE


def legal_env_indices(mask: np.ndarray) -> List[int]:
    return [i for i in range(NUM_ENV_ACTIONS) if bool(mask[i])]


def masked_finish(mask: np.ndarray) -> int:
    if bool(mask[A_FINISH]):
        return int(A_FINISH)
    if bool(mask[A_FINISH_FREEZE_SHOP]):
        return int(A_FINISH_FREEZE_SHOP)
    legal = legal_env_indices(mask)
    if not legal:
        raise RuntimeError("masked_finish: legal_actions_mask has no True entries")
    return int(legal[0])


def pick_rl_apply_action(env, mask: np.ndarray) -> int | None:
    """Pick TARGET_BOARD / SKIP while staged place is open (Argus-style)."""
    plan = getattr(env, "rl_pending", None)
    if plan is None:
        return None
    player = env.state.players[env.current_player()]
    eligible = plan.eligible_on_board_live(player.board)
    for i in eligible:
        a = A_TARGET_BOARD_BASE + i
        if bool(mask[a]):
            return int(a)
    if plan.can_skip_second_adjacent() and bool(mask[A_APPLY_EFFECT_SKIP]):
        return int(A_APPLY_EFFECT_SKIP)
    legal = legal_env_indices(mask)
    targets = [a for a in legal if is_target_board(a)]
    if targets:
        return int(targets[0])
    if any(is_apply_effect_skip(a) for a in legal):
        return int(A_APPLY_EFFECT_SKIP)
    return None


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
            si = j - 1 if j > i else j
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
    key_fn: Callable[[int, Sequence[Minion | None]], Tuple[float, int]] | None,
) -> int:
    k = len(board)
    if k == 0:
        return masked_finish(mask)
    if key_fn is None:
        from src.envs.minibg.heuristic_bots.common import order_key_default

        key_fn = order_key_default
    order = sorted(range(k), key=lambda oi: key_fn(oi, board))
    perm = perm_from_desired_order(k, order)
    return choose_one_swap_toward_perm(board, mask, perm)

"""Faceless Taverngoer: pick a shop offer to transform into."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_lobby.player import PendingChoice, PendingChoiceKind, PlayerState


def filled_shop_slot_indices(player: PlayerState) -> Tuple[int, ...]:
    return tuple(i for i, m in enumerate(player.shop) if m is not None)


def apply_transform_into_shop_minion(
    player: PlayerState,
    board_idx: int,
    shop_slot: int,
    *,
    patch: PatchContext,
) -> None:
    if not (0 <= board_idx < len(player.board)):
        raise ValueError(f"invalid transform board index: {board_idx}")
    offer = player.shop[shop_slot]
    if offer is None:
        raise ValueError(f"empty shop slot for transform: {shop_slot}")
    player.board[board_idx] = make_minion(offer.card_id, patch=patch)


def try_open_transform_shop_modal(
    player: PlayerState,
    board_idx: int,
    *,
    patch: PatchContext,
    rng: np.random.Generator,
) -> bool:
    """Apply transform or open a shop-slot modal. Returns True if modal opened."""
    slots = filled_shop_slot_indices(player)
    if not slots:
        return False
    if len(slots) == 1:
        apply_transform_into_shop_minion(
            player, board_idx, slots[0], patch=patch
        )
        return False
    opts: list[str] = ["", "", ""]
    for i, slot in enumerate(slots[:3]):
        offer = player.shop[slot]
        assert offer is not None
        opts[i] = offer.card_id
    player.pending_choice = PendingChoice(
        PendingChoiceKind.TRANSFORM_SHOP_MINION,
        (opts[0], opts[1], opts[2]),
        0,
        transform_board_idx=board_idx,
    )
    return True


def resolve_transform_shop_pick(
    player: PlayerState,
    shop_slot: int,
    *,
    patch: PatchContext,
    on_after_placed: Optional[Callable[[PlayerState, object], None]] = None,
) -> None:
    pc = player.pending_choice
    assert pc is not None
    assert pc.kind == PendingChoiceKind.TRANSFORM_SHOP_MINION
    assert pc.transform_board_idx is not None
    board_idx = pc.transform_board_idx
    apply_transform_into_shop_minion(player, board_idx, shop_slot, patch=patch)
    player.pending_choice = None
    if on_after_placed is not None and 0 <= board_idx < len(player.board):
        on_after_placed(player, player.board[board_idx])
    player.placed_minion_pending_after = None
    player.placed_minion_board_index = None


__all__ = [
    "apply_transform_into_shop_minion",
    "filled_shop_slot_indices",
    "resolve_transform_shop_pick",
    "try_open_transform_shop_modal",
]

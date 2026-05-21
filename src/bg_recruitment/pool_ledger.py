"""Bridge minion lifecycle events to ``SharedCardPool`` accounting."""

from __future__ import annotations

from typing import Optional, Tuple

from src.bg_core.minion import Minion
from src.bg_lobby.player import PendingChoice, PlayerState
from src.bg_lobby.shared_pool import SharedCardPool, copies_for_minion

__all__ = [
    "on_bought_from_shop",
    "on_discover_to_hand",
    "on_sell_minion",
    "on_eliminate_player",
    "release_discover_options",
    "reserve_discover_options",
]


def on_bought_from_shop(
    _pool: Optional[SharedCardPool],
    _minion: Minion,
) -> None:
    """Shop slot cleared on buy: offer was already reserved when displayed."""
    return


def reserve_discover_options(
    pool: SharedCardPool,
    options: Tuple[str, str, str],
) -> bool:
    """Reserve one lobby copy per discover option (shop-offer style)."""
    reserved: list[str] = []
    for cid in options:
        if not pool.try_reserve_offer(cid):
            for released in reserved:
                pool.release_offer(released, 1)
            return False
        reserved.append(cid)
    return True


def release_discover_options(
    pool: SharedCardPool,
    options: Tuple[str, str, str],
    *,
    keep_slot: Optional[int] = None,
) -> None:
    """Return reserved copies for unpicked discover slots."""
    for i, cid in enumerate(options):
        if keep_slot is not None and i == keep_slot:
            continue
        pool.release_offer(cid, 1)


def on_discover_to_hand(
    pool: Optional[SharedCardPool],
    minion: Minion,
) -> None:
    if pool is None:
        return
    n = copies_for_minion(minion)
    if n <= 0:
        return
    if not pool.acquire_new(minion.card_id, n):
        raise RuntimeError(
            f"shared pool: cannot acquire {n} of {minion.card_id!r} for discover"
        )


def release_pending_discover_pool(
    pool: Optional[SharedCardPool],
    pending: Optional[PendingChoice],
) -> None:
    if pool is None or pending is None or not pending.options_pool_reserved:
        return
    release_discover_options(pool, pending.options)


def on_sell_minion(pool: Optional[SharedCardPool], minion: Minion) -> None:
    if pool is None:
        return
    pool.release_minion(minion)


def on_eliminate_player(pool: Optional[SharedCardPool], player: PlayerState) -> None:
    if pool is None:
        return
    release_pending_discover_pool(pool, player.pending_choice)
    for m in player.board:
        pool.release_minion(m)
    for m in player.hand:
        if m is not None:
            pool.release_minion(m)
    for m in player.shop:
        if m is not None:
            pool.release_offer(m.card_id, copies_for_minion(m))

"""Board helpers shared by shop and combat effect resolution."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from copy import copy

from .minion import Minion, Race


def count_unique_tribes(
    board: Sequence[Minion],
    *,
    exclude: Optional[Minion] = None,
    exclude_self_card: bool = False,
) -> int:
    """Count distinct non-neutral tribes on ``board`` (``Race.ALL`` ignored).

    ``exclude``: omit this minion instance from the count (Amalgadon self).
    """
    tribes: set[Race] = set()
    for m in board:
        if exclude is not None and m is exclude:
            continue
        if exclude_self_card and exclude is not None and m.card_id == exclude.card_id:
            continue
        if m.race is None or m.race == Race.ALL:
            continue
        tribes.add(m.race)
    return len(tribes)


def minion_matches_tribe(minion: Minion, tribe: Any) -> bool:
    if minion.race is None:
        return False
    if tribe == Race.ALL or minion.race == Race.ALL:
        return True
    return minion.race == tribe


def count_friendly_tribe(
    board: Sequence[Minion],
    tribe: Any,
    *,
    exclude: Optional[Minion] = None,
) -> int:
    return sum(
        1
        for m in board
        if (exclude is None or m is not exclude) and minion_matches_tribe(m, tribe)
    )


def count_golden_friendlies(
    board: Sequence[Minion],
    *,
    exclude: Optional[Minion] = None,
) -> int:
    return sum(
        1 for m in board if (exclude is None or m is not exclude) and m.is_golden
    )


def snapshot_warband(board: Sequence[Minion]) -> Tuple[Minion, ...]:
    """Shallow-copy minions for ``PlayerState.last_opponent_board``."""
    return tuple(copy(m) for m in board)


__all__ = [
    "count_unique_tribes",
    "minion_matches_tribe",
    "count_friendly_tribe",
    "count_golden_friendlies",
    "snapshot_warband",
]

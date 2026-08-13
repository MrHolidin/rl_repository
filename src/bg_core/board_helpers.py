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


def count_for_source(
    source: "CountSource",
    board: Sequence[Minion],
    *,
    tribe: Any = None,
    exclude: Optional[Minion] = None,
) -> int:
    """Dispatch a :class:`CountSource` onto the matching board count."""
    from .effects import CountSource

    if source is CountSource.FRIENDLY_OF_TRIBE:
        return count_friendly_tribe(board, tribe, exclude=exclude)
    if source is CountSource.UNIQUE_TRIBES:
        return count_unique_tribes(board, exclude=exclude)
    if source is CountSource.GOLDEN_FRIENDLIES:
        return count_golden_friendlies(board, exclude=exclude)
    raise ValueError(f"unhandled CountSource {source!r}")


def apply_buff_self_per_count(
    effect: "BuffSelfPerCount",
    listener: Minion,
    board: Sequence[Minion],
) -> None:
    """Apply ``BuffSelfPerCount`` to ``listener`` (its own board is ``board``).

    Single implementation for what used to be three copies of this body, one
    per counting class.
    """
    n = count_for_source(
        effect.source,
        board,
        tribe=effect.tribe,
        exclude=listener if effect.exclude_self else None,
    )
    listener.bonus_attack += effect.attack_per * n
    listener.bonus_health += effect.health_per * n


def snapshot_warband(board: Sequence[Minion]) -> Tuple[Minion, ...]:
    """Shallow-copy minions for ``PlayerState.last_opponent_board``."""
    return tuple(copy(m) for m in board)


__all__ = [
    "apply_buff_self_per_count",
    "count_for_source",
    "count_unique_tribes",
    "minion_matches_tribe",
    "count_friendly_tribe",
    "count_golden_friendlies",
    "snapshot_warband",
]

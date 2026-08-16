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


def multiplier_for(minions, kind) -> int:
    """Product of the ``kind`` multiplier auras standing on a board.

    One scan for what used to be four near-identical copies: Brann in the shop,
    Baron and Khadgar in combat, and Khadgar again in the shop once the tavern
    half of his text was implemented. Takes plain ``Minion`` templates so both
    phases can call it -- combat passes ``bm.template``.
    """
    from .effects import Multiplier, Trigger

    p = 1
    for m in minions:
        for ab in m.abilities:
            if ab.trigger == Trigger.AURA and isinstance(ab.effect, Multiplier):
                if ab.effect.kind is kind:
                    p *= ab.effect.factor
    return p


def buff_matching_hits(
    effect: Any,
    candidate: Minion,
    source: Optional[Minion] = None,
    *,
    idx_candidate: Optional[int] = None,
    idx_source: Optional[int] = None,
) -> bool:
    """Does ``candidate`` match ``effect``'s target predicate?

    Operates on the *template* level so shop (``Minion``) and combat
    (``BattleMinion.template``) share one predicate. Aliveness and the
    combat rule "a deathrattle never buffs its own corpse" stay at the call
    sites, which is where they were before the merge.
    """
    from .effects import BuffTarget

    t = effect.target
    if t is BuffTarget.ALL_FRIENDLY:
        return True
    if t is BuffTarget.FRIENDLY_OF_TRIBE:
        return minion_matches_tribe(candidate, effect.tribe)
    if t is BuffTarget.OTHER_OF_TRIBE:
        if source is not None and candidate is source:
            return False
        return minion_matches_tribe(candidate, effect.tribe)
    if t is BuffTarget.FRIENDLY_WITH_KEYWORD:
        return effect.keyword in candidate.all_keywords
    if t is BuffTarget.ADJACENT:
        # Positional, so the caller has to supply where the two stand.
        if idx_candidate is None or idx_source is None:
            return False
        return idx_candidate in (idx_source - 1, idx_source + 1)
    raise ValueError(f"unhandled BuffTarget {t!r}")


def apply_buff_matching(effect, minions, source=None, *, repeats: int = 1) -> None:
    """Apply a ``BuffMatching`` to everyone on ``minions`` it reaches.

    One body for what the shop and combat each spelled out. Everything they
    differed by is a parameter now: which minions (a board or a battle side),
    who the source is, and how many times the trigger fires (Baron). Combat
    settles its auras afterwards at the call site, which is the one thing that
    genuinely has no shop equivalent.

    ``source`` is passed through to the predicate, which excludes it only for
    ``OTHER_OF_TRIBE`` -- the targets whose card text says "other". Combat used
    to exclude the source unconditionally, which is invisible for a real death
    (the body is in the graveyard) but wrong when Monstrous Macaw fires a
    living minion's deathrattle: Goldrinn is a Beast and "give your Beasts"
    includes it.
    """
    for _ in range(max(1, repeats)):
        for m in minions:
            if not buff_matching_hits(effect, m, source):
                continue
            m.bonus_attack += effect.attack
            m.bonus_health += effect.health


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
    "apply_buff_matching",
    "buff_matching_hits",
    "count_for_source",
    "multiplier_for",
    "count_unique_tribes",
    "minion_matches_tribe",
    "count_friendly_tribe",
    "count_golden_friendlies",
    "snapshot_warband",
]

"""Summon helpers: insert/append tokens, resolve summon target side."""
from __future__ import annotations

from copy import copy
from typing import Optional

from src.bg_core.minion import Minion

from .state import BattleMinion, BattleSide, _CombatRuntime
from .events import MinionSummoned
from .auras import _board_index, _mark_health_aura_dirty, _sync_health_all


def _summon_insert(
    rt: _CombatRuntime,
    side_idx: int,
    template: Minion,
    at_idx: Optional[int] = None,
) -> Optional[BattleMinion]:
    """Summon at list position ``at_idx`` (None / past-end → rightmost).

    Inserting at or before the side's scan cursor shifts the cursor right so
    the attack rotation neither skips nor repeats a minion; a token inserted
    behind the pointer waits for the next pass (real-BG behaviour).
    """
    side = rt.side(side_idx)
    if side.alive_count() >= rt.combat_board_max:
        return None
    bid = rt.alloc_id()
    bm = BattleMinion.from_minion(copy(template), bid)
    if at_idx is None or at_idx >= len(side.minions):
        side.minions.append(bm)
    else:
        # Inserting at or before the pointer shifts it, so the minion whose
        # turn it was keeps its turn and the newcomer waits for the next pass.
        # The exception is a token filling a slot its previous occupant just
        # vacated by dying: nobody is waiting on that slot, so the token
        # inherits its place in the rotation and swings this pass.
        #
        # Ask the graveyard directly rather than using "are we resolving a
        # death" as a proxy. The proxy is wrong in both directions: Monstrous
        # Macaw fires a *living* minion's deathrattle, and an on-damage summon
        # (Security Rover) can fire while an unrelated deathrattle is running.
        claims_slot = any(b.death_pos == at_idx for b in side.graveyard)
        side.minions.insert(at_idx, bm)
        side.shift_graveyard_slots(at_idx, +1)
        if at_idx <= side.cursor and not claims_slot:
            side.cursor += 1
    _mark_health_aura_dirty(rt, side_idx)
    rt.queue.append(MinionSummoned(side_idx, bid, template.card_id))
    _sync_health_all(rt)
    return bm


def _summon_append(
    rt: _CombatRuntime,
    side_idx: int,
    template: Minion,
) -> Optional[BattleMinion]:
    return _summon_insert(rt, side_idx, template, None)


def _insert_idx_after(side: BattleSide, anchor: Optional[BattleMinion]) -> Optional[int]:
    """Slot a summon from ``anchor`` should take; ``None`` anchor → append.

    A living anchor summons to its right. A dead one summons *into* the slot it
    vacated, which is what ``death_pos`` records — the body itself is no longer
    in ``side.minions`` to be indexed.
    """
    if anchor is None:
        return None
    idx = _board_index(side, anchor)
    if idx is not None:
        return idx + 1
    return anchor.death_pos if anchor.death_pos >= 0 else None


def _summon_target_side(dead_side_idx: int, for_opponent: bool) -> int:
    return (1 - dead_side_idx) if for_opponent else dead_side_idx

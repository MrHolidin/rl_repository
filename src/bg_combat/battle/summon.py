"""Summon helpers: insert/append tokens, resolve summon target side."""
from __future__ import annotations

from typing import Optional

from src.bg_core.minion import Minion

from .state import BattleMinion, BattleSide, _CombatRuntime, battle_copy
from .events import MinionSummoned
from src.bg_core.board_helpers import has_attack_threshold_ability, index_of

from .auras import _mark_health_aura_dirty, _sync_health_all


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
    bm = battle_copy(template, bid)
    if not rt.watch_attack_thresholds and has_attack_threshold_ability(template):
        rt.watch_attack_thresholds = True
    if at_idx is None or at_idx >= len(side.minions):
        side.minions.append(bm)
    else:
        # Inserting at or before the pointer shifts it, so the minion whose
        # turn it was keeps its turn and the newcomer waits for the next pass.
        # The exception is a token filling a slot its previous occupant just
        # vacated by dying: nobody is waiting on that slot, so the token
        # inherits its place in the rotation and swings this pass.
        #
        # Both halves are required. The slot has to be one a body vacated --
        # asked of the graveyard directly, because "are we resolving a death"
        # is a bad proxy in both directions (Monstrous Macaw fires a *living*
        # minion's deathrattle; an on-damage summon fires while an unrelated
        # deathrattle runs). And it has to be the slot the pointer is on: a
        # token dropped into a vacated slot to the *left* of the pointer is
        # still an insertion in front of whoever is waiting, so it shifts them
        # along. Dropping that half let a minion that had already swung this
        # pass take a second turn while the one waiting was skipped.
        claims_slot = at_idx == side.cursor and any(
            b.death_pos == at_idx for b in side.graveyard
        )
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
    idx = index_of(side.minions, anchor)
    if idx is not None:
        return idx + 1
    return anchor.death_pos if anchor.death_pos >= 0 else None


def _summon_target_side(dead_side_idx: int, for_opponent: bool) -> int:
    return (1 - dead_side_idx) if for_opponent else dead_side_idx

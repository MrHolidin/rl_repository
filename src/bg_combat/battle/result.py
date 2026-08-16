"""Battle result types and survivor / gold / hand-add output emission."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion

from .state import BattleSide, _CombatRuntime


def _winner_damage_raw(side: BattleSide, winner_tavern_tier: int) -> int:
    """Uncapped winner damage formula INCLUDING tavern tier (used for HP)."""
    return int(winner_tavern_tier) + sum(m.tier for m in side.minions)


def _winner_damage_board_only(side: BattleSide) -> int:
    """Board-only damage component: ``Σ alive_minion_tiers``, no tavern tier.

    Used as the auxiliary battle-prediction head's regression target. The
    tavern-tier contribution is excluded on purpose: it's deterministic from
    the pre-combat state (both seats' tavern tiers are known scalars) and
    would let the head trivially memorize it from ``state_emb`` if exposed.
    Keeping the target board-derived makes the prediction depend only on the
    minions that actually fought.
    """
    return sum(m.tier for m in side.minions)


def _winner_damage(side: BattleSide, winner_tavern_tier: int, damage_cap: int) -> int:
    return min(int(damage_cap), _winner_damage_raw(side, winner_tavern_tier))


@dataclass(frozen=True)
class RawBattleSnapshot:
    """Side-neutral snapshot of boards at a specific step of combat.

    ``step_index=0`` is the pre-combat state (boards as they entered
    ``simulate_battle``). Future mid-battle snapshots will have higher indices.
    """

    side0_board: Tuple[Minion, ...]
    side1_board: Tuple[Minion, ...]
    step_index: int = 0


@dataclass(frozen=True)
class BattleResult:
    """Structured combat result. Backward-compatible with the legacy ``(dmg_p0, dmg_p1)``
    tuple via iter/index protocols so existing call sites that do
    ``dmg_p0, dmg_p1 = simulate_battle(...)`` keep working.

    Fields beyond the legacy pair:
    - ``raw_damage_p0`` / ``raw_damage_p1``: **board-only** uncapped damage —
      just ``Σ alive_minion_tiers`` on the winning side (no tavern-tier term).
      Zero for the loser/draws. Target for the auxiliary battle-prediction
      head. Tavern tier is excluded so the head's prediction depends only on
      what minions fought, not on the trivially-known hero state.
    - ``attack_first_side``: which side struck first (0 or 1); 0 by default for
      degenerate cases (both empty / one empty pre-combat).
    - ``snapshots``: at least the initial pre-combat snapshot.
    """

    damage_p0: int
    damage_p1: int
    raw_damage_p0: int
    raw_damage_p1: int
    attack_first_side: int
    snapshots: Tuple[RawBattleSnapshot, ...]

    def __iter__(self):
        # Legacy callsites: ``dmg_p0, dmg_p1 = simulate_battle(...)``
        return iter((self.damage_p0, self.damage_p1))

    def __getitem__(self, idx: int) -> int:
        return (self.damage_p0, self.damage_p1)[idx]

    def __len__(self) -> int:
        return 2

    def __eq__(self, other) -> bool:
        # Allow direct comparisons with legacy ``(dmg_p0, dmg_p1)`` tuples so
        # callers using ``simulate_battle(...) == (0, 0)`` keep working.
        if isinstance(other, BattleResult):
            return (
                self.damage_p0 == other.damage_p0
                and self.damage_p1 == other.damage_p1
                and self.raw_damage_p0 == other.raw_damage_p0
                and self.raw_damage_p1 == other.raw_damage_p1
                and self.attack_first_side == other.attack_first_side
                and self.snapshots == other.snapshots
            )
        if isinstance(other, tuple) and len(other) == 2:
            return (self.damage_p0, self.damage_p1) == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.damage_p0, self.damage_p1, self.raw_damage_p0, self.raw_damage_p1))


def persist_shop_board_from_side(side: BattleSide, max_slots: int) -> List[Minion]:
    """Alive combat minions in scan order, shallow-copied to shop ``Minion`` (shields re-arm)."""
    out: List[Minion] = []
    for bm in side.minions:
        if not bm.alive:
            continue
        if len(out) >= max_slots:
            break
        m = copy(bm.template)
        if Keyword.SHIELD in m.all_keywords:
            m.has_shield = True
        out.append(m)
    return out


def _emit_survivor_outputs(
    side0: BattleSide,
    side1: BattleSide,
    *,
    p0_survivors_out: Optional[List[str]] = None,
    p1_survivors_out: Optional[List[str]] = None,
    p0_board_out: Optional[List[Minion]] = None,
    p1_board_out: Optional[List[Minion]] = None,
    max_board_slots: int,
) -> None:
    if p0_survivors_out is not None:
        p0_survivors_out.clear()
        p0_survivors_out.extend(m.template.card_id for m in side0.minions)
    if p1_survivors_out is not None:
        p1_survivors_out.clear()
        p1_survivors_out.extend(m.template.card_id for m in side1.minions)
    if p0_board_out is not None:
        p0_board_out.clear()
        p0_board_out.extend(persist_shop_board_from_side(side0, max_board_slots))
    if p1_board_out is not None:
        p1_board_out.clear()
        p1_board_out.extend(persist_shop_board_from_side(side1, max_board_slots))


def _emit_combat_hand_adds(
    rt: _CombatRuntime, combat_hand_adds_out: Optional[List[List[str]]]
) -> None:
    if combat_hand_adds_out is None:
        return
    if len(combat_hand_adds_out) >= 1:
        combat_hand_adds_out[0] = list(rt.combat_hand_adds[0])
    if len(combat_hand_adds_out) >= 2:
        combat_hand_adds_out[1] = list(rt.combat_hand_adds[1])


def _emit_combat_gold(
    rt: _CombatRuntime, combat_gold_out: Optional[List[int]]
) -> None:
    if combat_gold_out is None:
        return
    if len(combat_gold_out) >= 1:
        combat_gold_out[0] = rt.combat_gold[0]
    if len(combat_gold_out) >= 2:
        combat_gold_out[1] = rt.combat_gold[1]

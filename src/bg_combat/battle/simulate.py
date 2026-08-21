"""``simulate_battle``: top-level entry that wires sides, runs the loop, emits results."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_core.minion import Minion

from .seat import CombatSeat, RecordingSeat
from .state import BattleSide, _CombatRuntime
from .sides import _build_side
from .auras import _sync_health_all
from .engine import (
    _decide_first_side,
    _fire_start_of_combat,
    _next_attacker,
    _run_attacker_activation,
    _side_has_attackers,
)
from .result import (
    BattleResult,
    RawBattleSnapshot,
    _emit_combat_gold,
    _emit_combat_hand_adds,
    _emit_survivor_outputs,
    _winner_damage,
    _winner_damage_board_only,
)


def _keep_combat_gains(rt: "_CombatRuntime") -> None:
    """Write home what a body was promised to keep (Tarecgosa).

    Runs once the fighting is over, on survivors only: a card that keeps its
    gains has to be there at the end to keep them. Everything else a combat
    picks up still dies with the copy, which is what makes a board come out of
    a fight the way it went in.
    """
    from src.bg_core.board_helpers import minion_matches_tribe
    from src.bg_core.effects import KeepCombatGainsEffect, Trigger

    for side_idx in (0, 1):
        living = list(rt.side(side_idx).iter_living())
        keepers: dict = {}
        for i, bm in enumerate(living):
            for ability in bm.abilities:
                eff = ability.effect
                if ability.trigger is not Trigger.AURA or not isinstance(
                    eff, KeepCombatGainsEffect
                ):
                    continue
                multiple = max(1, int(getattr(eff, "factor", 1)))
                if not eff.adjacent:
                    keepers[bm.instance_id] = max(
                        multiple, keepers.get(bm.instance_id, 1)
                    )
                    continue
                # Granted to the neighbours instead: Persistent Poet keeps
                # nothing itself.
                for j in (i - 1, i + 1):
                    if 0 <= j < len(living) and (
                        eff.tribe is None
                        or minion_matches_tribe(living[j], eff.tribe)
                    ):
                        keepers[living[j].instance_id] = max(
                            multiple, keepers.get(living[j].instance_id, 1)
                        )
        for bm in living:
            if bm.instance_id not in keepers:
                continue
            # "keeps ... **double** stats gained" on the Golden printing. The
            # keywords are not a number and are kept once either way.
            multiple = keepers[bm.instance_id]
            gained_attack = (bm.bonus_attack - bm.start_bonus_attack) * multiple
            gained_health = (bm.bonus_health - bm.start_bonus_health) * multiple
            gained_keywords = bm.keywords - bm.start_keywords
            if gained_attack or gained_health or gained_keywords:
                # By origin id: the seat's board knows the body it sent, not
                # the combat-local id the copy fought under.
                rt.seats[side_idx].keep_combat_gains(
                    bm.origin_instance_id, gained_attack, gained_health, gained_keywords
                )


def simulate_battle(
    p0_board: List[Minion],
    p1_board: List[Minion],
    *,
    p0_has_initiative: bool,
    rng: np.random.Generator,
    combat_board_max: int,
    damage_cap: int,
    max_board_slots: int,
    max_attacks: int = 200,
    death_log: Optional[List[Tuple[int, str]]] = None,
    mech_death_log: Optional[List[Tuple[int, Minion]]] = None,
    p0_survivors_out: Optional[List[str]] = None,
    p1_survivors_out: Optional[List[str]] = None,
    p0_board_out: Optional[List[Minion]] = None,
    p1_board_out: Optional[List[Minion]] = None,
    p0_tavern_tier: int = 1,
    p1_tavern_tier: int = 1,
    patch: PatchContext,
    combat_gold_out: Optional[List[int]] = None,
    combat_hand_adds_out: Optional[List[List[str]]] = None,
    # The two seats this combat is fought on behalf of. Omitted — every test
    # and the pure-rules API — each side gets a RecordingSeat, which collects
    # what the fight hands out and applies none of it, exactly as the two
    # ``*_out`` lists above did on their own.
    seats: Optional[Tuple["CombatSeat", "CombatSeat"]] = None,
    p0_attack_aura_all: int = 0,
    p1_attack_aura_all: int = 0,
    p0_start_combat_keywords: frozenset = frozenset(),
    p1_start_combat_keywords: frozenset = frozenset(),
) -> "BattleResult":
    # Snapshot the input boards (deep-ish copy by tuple) BEFORE any combat
    # mutation. This is the initial (step=0) snapshot fed to the battle
    # prediction head. ``p0_board`` / ``p1_board`` are passed by reference and
    # ``_build_side`` copies each minion into the battle, so the caller's own
    # objects are never touched and a tuple of references suffices.
    _initial_snapshot = RawBattleSnapshot(
        side0_board=tuple(p0_board),
        side1_board=tuple(p1_board),
        step_index=0,
    )
    _snapshots: Tuple[RawBattleSnapshot, ...] = (_initial_snapshot,)
    ctx = require_patch(patch, where="battle.simulate_battle")
    rt = _CombatRuntime(
        sides=(BattleSide(), BattleSide()),
        rng=rng,
        combat_board_max=int(combat_board_max),
        damage_cap=int(damage_cap),
        patch=ctx,
        death_hook=(lambda si, cid: death_log.append((si, cid))) if death_log is not None else None,
        mech_hook=(lambda si, tpl: mech_death_log.append((si, tpl))) if mech_death_log is not None else None,
        seats=seats if seats is not None else (RecordingSeat(), RecordingSeat()),
    )
    if death_log is not None:
        death_log.clear()
    if mech_death_log is not None:
        mech_death_log.clear()

    rt.sides = (_build_side(p0_board, rt), _build_side(p1_board, rt))
    side0, side1 = rt.sides
    # Deathwing's +Attack aura buffs ALL minions in the combat → same flat bonus
    # on both sides (sum so two Deathwings stack correctly).
    combined_attack_aura = int(p0_attack_aura_all) + int(p1_attack_aura_all)
    side0.attack_aura_all = combined_attack_aura
    side1.attack_aura_all = combined_attack_aura
    side0.start_combat_keywords = frozenset(p0_start_combat_keywords)
    side1.start_combat_keywords = frozenset(p1_start_combat_keywords)
    _sync_health_all(rt)

    def _make_result(damage_p0: int, damage_p1: int, attack_first_side: int = 0) -> "BattleResult":
        # raw_damage_pX is the BOARD-ONLY uncapped winner-damage (no tavern tier).
        # Used as the auxiliary head's regression target; tier is deliberately
        # excluded so the head learns purely from board composition.
        if damage_p0 > 0 and damage_p1 == 0:
            raw_p0 = _winner_damage_board_only(side1)
            raw_p1 = 0
        elif damage_p1 > 0 and damage_p0 == 0:
            raw_p0 = 0
            raw_p1 = _winner_damage_board_only(side0)
        else:
            raw_p0 = 0
            raw_p1 = 0
        return BattleResult(
            damage_p0=int(damage_p0),
            damage_p1=int(damage_p1),
            raw_damage_p0=int(raw_p0),
            raw_damage_p1=int(raw_p1),
            attack_first_side=int(attack_first_side),
            snapshots=_snapshots,
        )

    if not side0.has_alive() and not side1.has_alive():
        _emit_survivor_outputs(
            side0,
            side1,
            p0_survivors_out=p0_survivors_out,
            p1_survivors_out=p1_survivors_out,
            p0_board_out=p0_board_out,
            p1_board_out=p1_board_out,
            max_board_slots=max_board_slots,
        )
        return _make_result(0, 0)
    if not side0.has_alive():
        _emit_survivor_outputs(
            side0,
            side1,
            p0_survivors_out=p0_survivors_out,
            p1_survivors_out=p1_survivors_out,
            p0_board_out=p0_board_out,
            p1_board_out=p1_board_out,
            max_board_slots=max_board_slots,
        )
        return _make_result(_winner_damage(side1, p1_tavern_tier, rt.damage_cap), 0, attack_first_side=1)
    if not side1.has_alive():
        _emit_survivor_outputs(
            side0,
            side1,
            p0_survivors_out=p0_survivors_out,
            p1_survivors_out=p1_survivors_out,
            p0_board_out=p0_board_out,
            p1_board_out=p1_board_out,
            max_board_slots=max_board_slots,
        )
        return _make_result(0, _winner_damage(side0, p0_tavern_tier, rt.damage_cap), attack_first_side=0)

    _fire_start_of_combat(rt)

    attacker_idx = _decide_first_side(side0, side1, p0_has_initiative)
    _first_side = attacker_idx
    sides = (side0, side1)

    attacks = 0
    while side0.has_alive() and side1.has_alive() and attacks < max_attacks:
        attacker_side = sides[attacker_idx]
        defender_side = sides[1 - attacker_idx]
        attacker_can_attack = _side_has_attackers(attacker_side, battle_field=sides)
        defender_can_attack = _side_has_attackers(defender_side, battle_field=sides)

        if not attacker_can_attack:
            if not defender_can_attack:
                break
            attacker_idx = 1 - attacker_idx
            continue

        attacker = _next_attacker(attacker_side, battle_field=sides)
        if attacker is None:
            if not defender_can_attack:
                break
            attacker_idx = 1 - attacker_idx
            continue

        if not defender_side.has_alive():
            break

        _run_attacker_activation(rt, attacker, attacker_idx, 1 - attacker_idx)

        attacker_idx = 1 - attacker_idx
        attacks += 1

    p0_alive = side0.has_alive()
    p1_alive = side1.has_alive()
    _keep_combat_gains(rt)
    _emit_survivor_outputs(
        side0,
        side1,
        p0_survivors_out=p0_survivors_out,
        p1_survivors_out=p1_survivors_out,
        p0_board_out=p0_board_out,
        p1_board_out=p1_board_out,
        max_board_slots=max_board_slots,
    )
    if p0_alive and not p1_alive:
        _emit_combat_gold(rt, combat_gold_out)
        _emit_combat_hand_adds(rt, combat_hand_adds_out)
        return _make_result(0, _winner_damage(side0, p0_tavern_tier, rt.damage_cap), attack_first_side=_first_side)
    if p1_alive and not p0_alive:
        _emit_combat_gold(rt, combat_gold_out)
        _emit_combat_hand_adds(rt, combat_hand_adds_out)
        return _make_result(_winner_damage(side1, p1_tavern_tier, rt.damage_cap), 0, attack_first_side=_first_side)
    _emit_combat_gold(rt, combat_gold_out)
    _emit_combat_hand_adds(rt, combat_hand_adds_out)
    return _make_result(0, 0, attack_first_side=_first_side)

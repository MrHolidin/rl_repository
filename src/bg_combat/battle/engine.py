"""Combat engine: event dispatch, swing resolution, attacker rotation,
start-of-combat firing, and first-side decision."""
from __future__ import annotations

from typing import Optional, Tuple

from src.bg_core.effects import Keyword, StartOfCombatDamagePerFriendlyTribe, Trigger

from .state import BattleMinion, BattleSide, _CombatRuntime
from .events import (
    AttackCompleted,
    BeginAttackExchange,
    DamageDealt,
    DamageStrike,
    MinionDied,
    MinionSummoned,
    Overkill,
    ShieldLost,
)
from .auras import attack_value, _grant_keyword, _sync_health_all
from .targeting import (
    _attacker_has_cleave,
    _cleave_victim_ids_at_swing_start,
    _pick_target,
)
from .effects import (
    _count_friendlies_of_tribe,
    _deal_random_enemy_minion_damage,
    _enqueue_strike_events,
    _fire_when_attacked,
    _handle_attack_completed,
    _handle_damage_dealt,
    _handle_minion_died,
    _handle_minion_summoned,
    _handle_overkill,
    _handle_shield_lost,
)


def _fire_start_of_combat(rt: _CombatRuntime) -> None:
    # Hero Start-of-Combat keyword grants to the left-most minion (Al'Akir:
    # Windfury + Divine Shield + Taunt) — applied before minion start-of-combat.
    for side_idx in (0, 1):
        side = rt.side(side_idx)
        if not side.start_combat_keywords:
            continue
        for bm in side.minions:
            if bm.alive:
                for kw in side.start_combat_keywords:
                    _grant_keyword(rt, side_idx, bm, kw)
                break
    for side_idx in (0, 1):
        side = rt.side(side_idx)
        enemy_idx = 1 - side_idx
        for bm in side.minions:
            if not bm.alive:
                continue
            for ab in bm.template.abilities:
                if ab.trigger != Trigger.ON_START_OF_COMBAT:
                    continue
                eff = ab.effect
                if isinstance(eff, StartOfCombatDamagePerFriendlyTribe):
                    count = _count_friendlies_of_tribe(side, eff.tribe)
                    if count <= 0:
                        continue
                    amount = count * eff.amount_per_match
                    for _ in range(max(1, eff.repeats)):
                        _deal_random_enemy_minion_damage(rt, side_idx, amount)
    _sync_health_all(rt)
    while rt.queue:
        ev = rt.queue.popleft()
        _dispatch(rt, ev)


def _dispatch(rt: _CombatRuntime, event: BattleEvent) -> None:
    if isinstance(event, BeginAttackExchange):
        return
    if isinstance(event, DamageStrike):
        _enqueue_strike_events(rt, event)
        return
    if isinstance(event, ShieldLost):
        _handle_shield_lost(rt, event)
        return
    if isinstance(event, DamageDealt):
        _handle_damage_dealt(rt, event)
        return
    if isinstance(event, Overkill):
        _handle_overkill(rt, event)
        return
    if isinstance(event, AttackCompleted):
        _handle_attack_completed(rt, event)
        return
    if isinstance(event, MinionDied):
        _handle_minion_died(rt, event)
        return
    if isinstance(event, MinionSummoned):
        _handle_minion_summoned(rt, event)
        return


def _run_single_swing(
    rt: _CombatRuntime,
    attacker: BattleMinion,
    target: BattleMinion,
    attacker_side_idx: int,
    defender_side_idx: int,
) -> None:
    atk_side = rt.side(attacker_side_idx)
    def_side = rt.side(defender_side_idx)
    rt.in_death_resolution = False
    if not attacker.alive or not target.alive:
        return
    _fire_when_attacked(rt, defender_side_idx, target)
    bf = (rt.side(0), rt.side(1))
    a_dmg = attack_value(attacker, atk_side, death_resolution=False, battle_field=bf)
    d_dmg = attack_value(target, def_side, death_resolution=False, battle_field=bf)

    rt.swing_damage_survivors.clear()
    rt.attacker_killed_this_swing = False
    rt.queue.append(BeginAttackExchange(attacker_side_idx, defender_side_idx))
    rt.queue.append(
        DamageStrike(
            attacker.instance_id,
            target.instance_id,
            defender_side_idx,
            a_dmg,
        )
    )
    if _attacker_has_cleave(attacker):
        for vid in _cleave_victim_ids_at_swing_start(def_side, target):
            rt.queue.append(
                DamageStrike(
                    attacker.instance_id,
                    vid,
                    defender_side_idx,
                    a_dmg,
                )
            )
    rt.queue.append(
        DamageStrike(
            target.instance_id,
            attacker.instance_id,
            attacker_side_idx,
            d_dmg,
        )
    )
    rt.queue.append(
        AttackCompleted(attacker_side_idx, attacker.instance_id)
    )
    while rt.queue:
        ev = rt.queue.popleft()
        _dispatch(rt, ev)


def _run_attacker_activation(
    rt: _CombatRuntime,
    attacker: BattleMinion,
    attacker_side_idx: int,
    defender_side_idx: int,
) -> None:
    """Resolve one board position's attack: Windfury may chain two swings before side swap."""
    attacker_side = rt.side(attacker_side_idx)
    battle_field = (rt.side(0), rt.side(1))
    if not _can_attack(attacker, attacker_side, battle_field=battle_field):
        return
    kws = attacker.template.all_keywords
    if Keyword.MEGA_WINDFURY in kws:
        n_swings = 4
    elif Keyword.WINDFURY in kws:
        n_swings = 2
    else:
        n_swings = 1
    defender_side = rt.side(defender_side_idx)
    for _ in range(n_swings):
        if (
            not _can_attack(attacker, attacker_side, battle_field=battle_field)
            or not defender_side.has_alive()
        ):
            break
        tgt = _pick_target(
            defender_side,
            rt.rng,
            attacker,
            battle_field=battle_field,
        )
        if tgt is None:
            break
        _run_single_swing(rt, attacker, tgt, attacker_side_idx, defender_side_idx)


def _can_attack(
    minion: BattleMinion,
    side: BattleSide,
    *,
    battle_field: Tuple[BattleSide, BattleSide],
) -> bool:
    return (
        minion.alive
        and attack_value(
            minion,
            side,
            death_resolution=False,
            battle_field=battle_field,
        )
        > 0
    )


def _side_has_attackers(
    side: BattleSide,
    *,
    battle_field: Tuple[BattleSide, BattleSide],
) -> bool:
    return any(_can_attack(m, side, battle_field=battle_field) for m in side.minions)


def _next_attacker(
    side: BattleSide,
    *,
    battle_field: Tuple[BattleSide, BattleSide],
) -> Optional[BattleMinion]:
    n = len(side.minions)
    if n == 0:
        return None
    start = side.cursor % n
    for offset in range(n):
        idx = (start + offset) % n
        if _can_attack(side.minions[idx], side, battle_field=battle_field):
            side.cursor = (idx + 1) % n
            return side.minions[idx]
    return None


def _decide_first_side(
    side0: BattleSide,
    side1: BattleSide,
    p0_has_initiative: bool,
) -> int:
    n0 = side0.alive_count()
    n1 = side1.alive_count()
    if n0 > n1:
        return 0
    if n1 > n0:
        return 1
    return 0 if p0_has_initiative else 1

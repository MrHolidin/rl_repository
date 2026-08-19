"""Combat engine: event dispatch, swing resolution, attacker rotation,
start-of-combat firing, and first-side decision."""
from __future__ import annotations

from typing import List, Optional, Tuple

from src.bg_core.effects import (
    BuffMatching,
    CastSpellAtEffect,
    DealDamageAllMinions,
    DevourNeighbourEffect,
    GainStatsFromHandEffect,
    Keyword,
    RepeatPerCountEffect,
    SummonBestFromHandEffect,
    StartOfCombatDamagePerFriendlyTribe,
    SummonSelfCopyFromHandEffect,
    Trigger,
)

from .state import BattleMinion, BattleSide, _CombatRuntime, battle_copy
from .summon import _summon_append
from .effects import (
    _deal_damage_to_battle_minion,
    _summon_best_from_hand,
    cast_spell_in_combat,
)
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
from src.bg_core.board_helpers import apply_buff_matching, minion_matches_tribe
from .targeting import (
    _attacker_has_cleave,
    _cleave_victim_ids_at_swing_start,
    _pick_target,
)
from .effects import (
    _count_friendlies_of_tribe,
    _deal_random_enemy_minion_damage,
    _devour_neighbour,
    _enqueue_strike_events,
    _fire_rally,
    _fire_when_attacked,
    _handle_attack_completed,
    _handle_damage_dealt,
    _handle_minion_died,
    _handle_minion_summoned,
    _handle_overkill,
    _handle_shield_lost,
)


def _queue_hand_start_of_combat(
    rt: _CombatRuntime,
    side_idx: int,
    pending: List[Tuple[BattleMinion, object]],
) -> None:
    """Queue the Start of Combat triggers of cards held in *hand*.

    Flighty Scout is the shape: "If this minion is in your hand, summon a copy
    of it". The card never joins the board, so it is materialised only to carry
    its ability into the queue below — and only for effects that say they fire
    from hand, since every other Start of Combat means the board.

    Queuing locks the card for this fight. It does not leave the hand; it is
    simply not available to be summoned out of again, which is the rule for
    every summon from hand.
    """
    for instance_id, card_id, _attack, _health in rt.seats[side_idx].hand_minions():
        template = rt.patch.templates.get(card_id)
        if template is None:
            continue
        for ab in template.abilities:
            if ab.trigger != Trigger.ON_START_OF_COMBAT:
                continue
            if not isinstance(ab.effect, SummonSelfCopyFromHandEffect):
                continue
            # The card is spent for this fight the moment it queues: it stays in
            # hand, but nothing else may summon it again — the same lock a Rally
            # that reaches into the hand takes.
            rt.hand_summoned[side_idx].add(instance_id)
            pending.append((battle_copy(template, rt.alloc_id()), ab.effect))


def _fire_start_of_combat(rt: _CombatRuntime) -> None:
    # Hero Start-of-Combat keyword grants to the left-most minion (Al'Akir:
    # Windfury + Divine Shield + Taunt) — applied before minion start-of-combat.
    for side_idx in (0, 1):
        side = rt.side(side_idx)
        if not side.start_combat_keywords:
            continue
        # Al'Akir grants to the left-most living minion only.
        for bm in side.iter_living():
            for kw in side.start_combat_keywords:
                _grant_keyword(rt, side_idx, bm, kw)
            break
    # Both sides' triggers in board order, left to right.
    pending: Tuple[List[Tuple[BattleMinion, object]], List[Tuple[BattleMinion, object]]] = ([], [])
    for side_idx in (0, 1):
        for bm in rt.side(side_idx).minions:
            for ab in bm.abilities:
                if ab.trigger == Trigger.ON_START_OF_COMBAT:
                    pending[side_idx].append((bm, ab.effect))
        _queue_hand_start_of_combat(rt, side_idx, pending[side_idx])

    # Real BG draws a dominant player at random, then the sides alternate one
    # trigger at a time, each taking its left-most untriggered minion, with
    # deaths (and the deathrattles they set off) resolved between triggers.
    # Firing side 0's whole board first handed the lower seat every start-of-
    # combat race — and the lower seat is always side 0, since pairings are
    # emitted with ``a < b``. The draw is only taken when both sides have
    # something to fire, so boards without a contest keep their RNG stream.
    if pending[0] and pending[1]:
        turn = int(rt.rng.integers(0, 2))
    else:
        turn = 0 if pending[0] else 1
    pos = [0, 0]
    while pos[0] < len(pending[0]) or pos[1] < len(pending[1]):
        if pos[turn] >= len(pending[turn]):
            turn = 1 - turn
            continue
        bm, eff = pending[turn][pos[turn]]
        pos[turn] += 1
        if bm.alive:
            _apply_start_of_combat_effect(rt, turn, bm, eff)
            _sync_health_all(rt)
            while rt.queue:
                _dispatch(rt, rt.queue.popleft())
        turn = 1 - turn
    _sync_health_all(rt)
    while rt.queue:
        ev = rt.queue.popleft()
        _dispatch(rt, ev)


def _apply_start_of_combat_effect(
    rt: _CombatRuntime, side_idx: int, source: BattleMinion, eff: object
) -> None:
    """One Start-of-Combat trigger. Counts are read now, not up front, so a
    minion killed by an earlier trigger no longer feeds this one's tally."""
    if isinstance(eff, StartOfCombatDamagePerFriendlyTribe):
        count = _count_friendlies_of_tribe(rt.side(side_idx), eff.tribe)
        if count <= 0:
            return
        amount = count * eff.amount_per_match
        for _ in range(max(1, eff.repeats)):
            _deal_random_enemy_minion_damage(rt, side_idx, amount)
    elif isinstance(eff, BuffMatching):
        # "Start of Combat: give your other Dragons +1/+1", "…your Beasts have
        # +1 Attack for the rest of this combat", "give your left-most Dragon
        # +1/+2 and Windfury" (that one is ``limit=1`` plus a granted keyword).
        # All the same write: it lands on the combat copies and dies with them.
        apply_buff_matching(
            eff,
            rt.side(side_idx).minions,
            source,
            grant=lambda m, kw: _grant_keyword(rt, side_idx, m, kw),
        )
        _sync_health_all(rt)
    elif isinstance(eff, DealDamageAllMinions):
        # "Start of Combat: deal 3 damage to all other minions" — every body in
        # the fight but the one that said so.
        for other_side in (0, 1):
            for bm in list(rt.side(other_side).iter_living()):
                if bm is source:
                    continue
                _deal_damage_to_battle_minion(rt, other_side, bm, eff.amount)
    elif isinstance(eff, RepeatPerCountEffect):
        # "Improves permanently after you cast a Tavern spell" — the tally is
        # the seat's, so the level is asked of it and the inner effect repeats.
        times = rt.seats[side_idx].improve_level(eff.counter, eff.per)
        for _ in range(max(1, times)):
            _apply_start_of_combat_effect(rt, side_idx, source, eff.effect)
    elif isinstance(eff, DevourNeighbourEffect):
        _devour_neighbour(rt, side_idx, source, eff)
    elif isinstance(eff, CastSpellAtEffect):
        cast_spell_in_combat(rt, side_idx, source, eff.card_id)
    elif isinstance(eff, GainStatsFromHandEffect):
        held = rt.seats[side_idx].hand_minions()
        if held:
            if eff.highest_attack_only:
                source.bonus_attack += max(row[2] for row in held)
            else:
                source.bonus_attack += sum(row[2] for row in held)
                source.bonus_health += sum(row[3] for row in held)
            _sync_health_all(rt)
    elif isinstance(eff, SummonBestFromHandEffect):
        # Same reach as the Rally that summons from hand, at a different moment:
        # the card stays put and a copy joins the fight.
        _summon_best_from_hand(rt, side_idx, source, eff)
    elif isinstance(eff, SummonSelfCopyFromHandEffect):
        # ``source`` is the card in hand, made only to carry this trigger; the
        # copy that joins the fight is built from its template like any summon.
        template = rt.patch.templates.get(source.card_id)
        if template is not None:
            _summon_append(rt, side_idx, template)
    else:
        raise NotImplementedError(
            f"Start of Combat effect {type(eff).__name__} has no combat handler "
            f"(minion {source.card_id})"
        )


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
    # Rally reads the board the attack was declared on: after the target is
    # locked in and the defender's own on-attacked triggers have run, but before
    # either side's damage is measured below.
    _fire_rally(rt, attacker, attacker_side_idx, target)
    # A Rally that deals damage can empty the slot it was aiming at, and the
    # deaths it caused are sitting in the queue: drain them here, or they are
    # announced late (or not at all, if this swing turns out to be the last).
    while rt.queue:
        _dispatch(rt, rt.queue.popleft())
    if not attacker.alive or not target.alive:
        return
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
    # Who is swinging, for the cards that are immune only while they are.
    previous_swinger = rt.swinging_instance_id
    rt.swinging_instance_id = attacker.instance_id
    try:
        _run_swings(rt, attacker, attacker_side_idx, defender_side_idx, attacker_side, battle_field)
    finally:
        rt.swinging_instance_id = previous_swinger


def _run_swings(
    rt: _CombatRuntime,
    attacker: BattleMinion,
    attacker_side_idx: int,
    defender_side_idx: int,
    attacker_side,
    battle_field,
) -> None:
    kws = attacker.all_keywords
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

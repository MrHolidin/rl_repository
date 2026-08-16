"""Trigger/effect interpreters and the event handlers they drive.

Deathrattles, on-attack / -death / -summon / -damage listeners, and the damage
helpers they use. ``_handle_*`` event handlers live here; the engine's
``_dispatch`` routes events to them. Some on-death / on-survive effects trigger
an immediate extra attack (engine behaviour) — reached via the local
``_run_attacker_activation`` forwarder below, which breaks the effects<->engine
import cycle.
"""
from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import Any, List, Optional, Tuple

from src.bg_catalog.cards import make_minion
from src.bg_core.effects import (
    AdjacentStatAura,
    AttackBonusPerOtherMurlocGlobal,
    AttackImmediatelyAfterSurvivingEffect,
    BuffMatching,
    BuffTarget,
    BuffAdjacentOnAttackedEffect,
    BuffAttackedMinionEffect,
    BuffAttackerOnFriendlyAttackEffect,
    BuffListenerIfSummonedMatches,
    BuffRandomOtherFriendlyCombat,
    AddRandomMinionToHandOnKillEffect,
    BuffSelf,
    BuffSummonedIfRace,
    BuffDeadMinionNeighborsEffect,
    CleaveOnAttack,
    DealDamageRandomEnemyMinion,
    DealDamageLeftmostEnemyMinion,
    DealDamageAllMinions,
    DealExcessDamageToAdjacentEffect,
    TransferAttackToRandomFriendlyEffect,
    SummonRandomAndCopyToHandEffect,
    GainGoldOnDeathEffect,
    GrantKeywordRandomFriendly,
    GrantKeywordAllFriendlyOfTribe,
    GrantListenerKeywordIfSummonedMatches,
    Keyword,
    KeywordStatAura,
    DeathrattleMultiplierAura,
    MultiplySelfAttackEffect,
    StatAura,
    StartOfCombatDamagePerFriendlyTribe,
    SummonEffect,
    SummonRandomMinionEffect,
    SummonFirstDeadFriendlyMechsThisCombat,
    SummonMultiplierAura,
    SummonOnSelfDamaged,
    SummonRandomOnSelfDamagedEffect,
    TriggerRandomFriendlyDeathrattleEffect,
    TribalOtherStatAura,
    Trigger,
    ZappTargeting,
)
from src.bg_core.minion import Minion

from .events import (
    AttackCompleted,
    BattleEvent,
    DamageDealt,
    DamageStrike,
    MinionDied,
    MinionSummoned,
    Overkill,
    ShieldLost,
)
from .state import BattleMinion, BattleSide, _CombatRuntime
from src.bg_core.board_helpers import buff_matching_hits
from .auras import (
    attack_value,
    _board_index,
    _deathrattle_multiplier,
    _grant_keyword,
    _mark_health_aura_dirty,
    _matches_tribe_for_aura,
    _summon_multiplier,
    _sync_health_all,
)
from .summon import _insert_idx_after, _summon_insert, _summon_target_side
from .sides import _is_mech_template


def _run_attacker_activation(rt, attacker, attacker_side_idx, defender_side_idx):
    # Forwarder breaking the effects<->engine import cycle (an extra immediate
    # attack from on-death / on-survive effects is engine behaviour).
    from .engine import _run_attacker_activation as _impl

    return _impl(rt, attacker, attacker_side_idx, defender_side_idx)


# Summon pool lives in ``src.envs.minibg`` which imports this package back;
# resolve it lazily so importing the battle package never pulls in src.envs.
def summon_pool_for(*args, **kwargs):
    from src.envs.minibg.summon_pool import summon_pool_for as _impl

    return _impl(*args, **kwargs)


def hs_race_string(*args, **kwargs):
    from src.envs.minibg.summon_pool import hs_race_string as _impl

    return _impl(*args, **kwargs)


def _enqueue_strike_events(rt: _CombatRuntime, strike: DamageStrike) -> None:
    vic = rt.find_minion(strike.victim_side_idx, strike.victim_instance_id)
    att = rt.find_minion(1 - strike.victim_side_idx, strike.attacker_instance_id)
    if vic is None or not vic.alive or strike.amount <= 0:
        return
    v_kw = vic.template.all_keywords
    if vic.shield_armed and Keyword.SHIELD in v_kw:
        vic.shield_armed = False
        rt.queue.appendleft(ShieldLost(strike.victim_side_idx, strike.victim_instance_id))
        return

    hp_before = vic.current_health
    poison = att is not None and Keyword.POISONOUS in att.template.all_keywords
    vic.current_health -= strike.amount
    if poison:
        vic.current_health = 0
    lost = max(0, hp_before - max(vic.current_health, 0))
    trailing: List[BattleEvent] = [
        DamageDealt(
            strike.victim_side_idx,
            strike.victim_instance_id,
            strike.attacker_instance_id,
            lost,
            poison,
        ),
    ]
    if strike.amount > hp_before and hp_before > 0:
        trailing.append(
            Overkill(
                strike.victim_side_idx,
                strike.victim_instance_id,
                1 - strike.victim_side_idx,
                strike.attacker_instance_id,
                strike.amount - hp_before,
            )
        )
    if vic.current_health <= 0 and att is not None:
        killer_side = 1 - strike.victim_side_idx
        rt.kill_attribution[(strike.victim_side_idx, strike.victim_instance_id)] = (
            killer_side,
            strike.attacker_instance_id,
        )
        rt.attacker_killed_this_swing = True
    if hp_before > 0 and vic.current_health <= 0:
        _mark_health_aura_dirty(rt, strike.victim_side_idx)
    for ev in reversed(trailing):
        rt.queue.appendleft(ev)
    _sync_health_all(rt)
    # Take the body off the board here, where it died, rather than leaving it
    # for the end of the swing: the DamageDealt / Overkill events queued just
    # above are dispatched with the board already closed up, which is the
    # board the game would show them.
    #
    # Both sides, always side 0 first: a trade kills attacker and defender at
    # once, and the death order is the tie-break the death log has always
    # used.
    for sidx in (0, 1):
        _reap_side(rt, sidx)


def _handle_attack_completed(rt: _CombatRuntime, e: AttackCompleted) -> None:
    attacker = rt.find_minion(e.attacker_side_idx, e.attacker_instance_id)
    if attacker is not None and attacker.alive:
        _fire_after_attack(rt, attacker, e.attacker_side_idx)
        _fire_friendly_attack_listeners(rt, attacker, e.attacker_side_idx)
    seen: set[Tuple[int, int]] = set()
    for side_idx, instance_id in rt.swing_damage_survivors:
        key = (side_idx, instance_id)
        if key in seen:
            continue
        seen.add(key)
        bm = rt.find_minion(side_idx, instance_id)
        if bm is not None and bm.alive:
            _fire_survived_attack_effects(rt, side_idx, bm)
    rt.swing_damage_survivors.clear()
    for sidx in (0, 1):
        _reap_side(rt, sidx)
    _announce_deaths(rt)


def _deal_random_enemy_minion_damage(
    rt: _CombatRuntime, from_side_idx: int, amount: int
) -> None:
    if amount <= 0:
        return
    enemy_side = 1 - from_side_idx
    es = rt.side(enemy_side)
    victims = [m for m in es.minions if m.alive]
    if not victims:
        return
    vic = victims[int(rt.rng.integers(0, len(victims)))]
    _deal_damage_to_battle_minion(rt, enemy_side, vic, amount)


def _deal_leftmost_enemy_minion_damage(
    rt: _CombatRuntime, from_side_idx: int, amount: int
) -> None:
    if amount <= 0:
        return
    enemy_side = 1 - from_side_idx
    es = rt.side(enemy_side)
    victims = [m for m in es.minions if m.alive]
    if not victims:
        return
    _deal_damage_to_battle_minion(rt, enemy_side, victims[0], amount)


def _deal_damage_all_minions(rt: _CombatRuntime, amount: int) -> None:
    if amount <= 0:
        return
    for side_idx in (0, 1):
        for m in list(rt.side(side_idx).minions):
            if m.alive:
                _deal_damage_to_battle_minion(rt, side_idx, m, amount)


def _buff_neighbors_of_dead(
    rt: _CombatRuntime,
    side_idx: int,
    dead: BattleMinion,
    *,
    attack: int,
    health: int,
) -> None:
    side = rt.side(side_idx)
    idx = _board_index(side, dead)
    if idx is not None:
        neighbours = (idx - 1, idx + 1)
    elif dead.death_pos >= 0:
        # The body is gone and the board closed up behind it: the minion on its
        # left kept its slot, the one on its right slid into the vacated one.
        neighbours = (dead.death_pos - 1, dead.death_pos)
    else:
        return
    for j in neighbours:
        if 0 <= j < len(side.minions):
            ally = side.minions[j]
            if not ally.alive:
                continue
            ally.template.bonus_attack += attack
            ally.template.bonus_health += health
            ally.current_health += health


def _queue_combat_hand_add_card(
    rt: _CombatRuntime, side_idx: int, card_id: str
) -> None:
    rt.combat_hand_adds[side_idx].append(card_id)


def _summon_attack_immediately_if_requested(
    rt: _CombatRuntime,
    bm: Optional[BattleMinion],
    side_idx: int,
) -> None:
    if bm is None or not bm.alive or rt.bonus_attack_depth > 0:
        return
    rt.bonus_attack_depth += 1
    try:
        _run_attacker_activation(rt, bm, side_idx, 1 - side_idx)
    finally:
        rt.bonus_attack_depth -= 1


def _fire_self_damaged(rt: _CombatRuntime, side_idx: int, bm: BattleMinion) -> None:
    if not bm.alive:
        return
    for ab in bm.template.abilities:
        if ab.trigger != Trigger.ON_SELF_DAMAGED:
            continue
        eff = ab.effect
        if isinstance(eff, SummonOnSelfDamaged):
            anchor: Optional[BattleMinion] = bm
            n_sum = _summon_multiplier(rt.side(side_idx))
            for _ in range(max(0, eff.count)):
                for __ in range(n_sum):
                    tok = make_minion(eff.token_id, patch=rt.patch)
                    summoned = _summon_insert(
                        rt,
                        side_idx,
                        tok,
                        _insert_idx_after(rt.side(side_idx), anchor),
                    )
                    if summoned is None:
                        return
                    anchor = summoned
        elif isinstance(eff, SummonRandomOnSelfDamagedEffect):
            race_hs = hs_race_string(eff.race_filter)
            pool = summon_pool_for(
                None,
                False,
                False,
                race_hs,
                None,
                patch=rt.patch,
            )
            if not pool:
                return
            anchor2: Optional[BattleMinion] = bm
            n_sum = _summon_multiplier(rt.side(side_idx))
            for _ in range(max(0, eff.count)):
                for __ in range(n_sum):
                    cid = pool[int(rt.rng.integers(0, len(pool)))]
                    tok = make_minion(cid, patch=rt.patch)
                    if eff.grant_taunt:
                        tok.keywords = frozenset(tok.keywords | {Keyword.TAUNT})
                    summoned = _summon_insert(
                        rt,
                        side_idx,
                        tok,
                        _insert_idx_after(rt.side(side_idx), anchor2),
                    )
                    if summoned is None:
                        return
                    anchor2 = summoned


def _handle_minion_summoned(rt: _CombatRuntime, e: MinionSummoned) -> None:
    side = rt.side(e.side_idx)
    summoned = rt.find_minion(e.side_idx, e.instance_id)
    if summoned is None or not summoned.alive:
        return
    for listener in list(side.minions):
        if not listener.alive or listener is summoned:
            continue
        for ab in listener.template.abilities:
            if ab.trigger != Trigger.ON_FRIENDLY_MINION_SUMMONED:
                continue
            eff = ab.effect
            if isinstance(eff, BuffSummonedIfRace):
                if _matches_tribe_for_aura(summoned.template, eff.tribe):
                    summoned.template.bonus_attack += eff.attack
                    summoned.template.bonus_health += eff.health
                    summoned.current_health += eff.health
            elif isinstance(eff, GrantListenerKeywordIfSummonedMatches):
                if _matches_tribe_for_aura(summoned.template, eff.tribe):
                    _grant_keyword(rt, e.side_idx, listener, eff.keyword)
            elif isinstance(eff, BuffListenerIfSummonedMatches):
                if _matches_tribe_for_aura(summoned.template, eff.tribe):
                    listener.template.bonus_attack += eff.attack
                    listener.template.bonus_health += eff.health
                    listener.current_health += eff.health
    _sync_health_all(rt)


def _fire_friendly_kill_listeners(
    rt: _CombatRuntime, killer_side_idx: int, killer_instance_id: int
) -> None:
    killer = rt.find_minion(killer_side_idx, killer_instance_id)
    if killer is None:
        return
    killer_tpl = killer.template
    side = rt.side(killer_side_idx)
    for listener in list(side.minions):
        if not listener.alive:
            continue
        for ab in listener.template.abilities:
            if ab.trigger != Trigger.ON_FRIENDLY_KILL:
                continue
            if ab.filter_race is not None and not _matches_tribe_for_aura(
                killer_tpl, ab.filter_race
            ):
                continue
            eff = ab.effect
            if isinstance(eff, BuffSelf):
                listener.template.bonus_attack += eff.attack
                listener.template.bonus_health += eff.health
                listener.current_health += eff.health
    _sync_health_all(rt)


def _queue_random_combat_hand_add(
    rt: _CombatRuntime, side_idx: int, tribe: Optional[Any]
) -> None:
    race_hs = hs_race_string(tribe)
    pool = summon_pool_for(None, False, False, race_hs, None, patch=rt.patch)
    if not pool:
        return
    cid = pool[int(rt.rng.integers(0, len(pool)))]
    rt.combat_hand_adds[side_idx].append(cid)


def _reap_all(rt: _CombatRuntime) -> None:
    """Sweep both sides. Losing a health aura kills without any damage being
    dealt, so the aura recompute is a death site like any other -- it was the
    one path that left a body on the board."""
    for side_idx in (0, 1):
        rt.side(side_idx).reap_dead()


def _reap_side(rt: _CombatRuntime, side_idx: int) -> None:
    """Take a side's dead off the board, without announcing them yet.

    Single choke point: a minion can die from a swing, from a spell-like
    effect, or from losing the aura that was holding its health up, and every
    one of those routes has to take the body off the board the same way.
    """
    rt.side(side_idx).reap_dead()


def _announce_deaths(rt: _CombatRuntime) -> None:
    """Raise MinionDied for bodies already off the board, side 0 first.

    Kept separate from the sweep so the board is correct the moment a minion
    dies while the *order* deathrattles resolve in stays what it was: a trade
    kills both minions in the same exchange, and side 0's death has always
    been announced first.
    """
    for side_idx in (0, 1):
        for bm in rt.side(side_idx).graveyard:
            if not bm.death_announced:
                bm.death_announced = True
                rt.queue.append(MinionDied(side_idx, bm.instance_id))


def _deal_damage_to_battle_minion(
    rt: _CombatRuntime, side_idx: int, bm: BattleMinion, amount: int
) -> None:
    if amount <= 0 or not bm.alive:
        return
    if bm.shield_armed and Keyword.SHIELD in bm.template.all_keywords:
        bm.shield_armed = False
        rt.queue.append(ShieldLost(side_idx, bm.instance_id))
        return
    bm.current_health -= amount
    if bm.current_health <= 0:
        bm.current_health = 0
        _mark_health_aura_dirty(rt, side_idx)
    _sync_health_all(rt)
    if not bm.alive:
        _reap_side(rt, side_idx)
        _announce_deaths(rt)
    elif amount > 0:
        rt.swing_damage_survivors.append((side_idx, bm.instance_id))
        # ON_SELF_DAMAGED fires on ANY damage taken while surviving (juggler /
        # Red Whelp / deathrattle damage included), matching real-BG triggers —
        # not just strike damage (which fires via the DamageDealt event).
        _fire_self_damaged(rt, side_idx, bm)


def _deal_excess_to_adjacent(
    rt: _CombatRuntime,
    victim_side_idx: int,
    victim_instance_id: int,
    amount: int,
    *,
    both_adjacent: bool = False,
) -> None:
    if amount <= 0:
        return
    side = rt.side(victim_side_idx)
    vic = rt.find_minion(victim_side_idx, victim_instance_id)
    if vic is None:
        return
    vi = _board_index(side, vic)
    if vi is not None:
        slots = (vi - 1, vi + 1)
    elif vic.death_pos >= 0:
        # Overkill resolves after the body has left the board, so read the
        # slot it vacated: the minion on its left kept its index, the one on
        # its right slid into the vacated one. Looking the body up in
        # ``minions`` and giving up when it is absent silently threw the
        # excess damage away.
        slots = (vic.death_pos - 1, vic.death_pos)
    else:
        return
    adj: List[BattleMinion] = []
    for j in slots:
        if 0 <= j < len(side.minions):
            m = side.minions[j]
            if m.alive:
                adj.append(m)
    if not adj:
        return
    if both_adjacent:
        for target in adj:
            _deal_damage_to_battle_minion(rt, victim_side_idx, target, amount)
    else:
        target = adj[int(rt.rng.integers(0, len(adj)))]
        _deal_damage_to_battle_minion(rt, victim_side_idx, target, amount)


def _fire_friendly_minion_died_listeners(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int
) -> None:
    side = rt.side(side_idx)
    for listener in list(side.minions):
        if not listener.alive or listener is dead:
            continue
        for ab in listener.template.abilities:
            if ab.trigger != Trigger.ON_FRIENDLY_MINION_DIED:
                continue
            if ab.filter_race is not None and not _matches_tribe_for_aura(
                dead.template, ab.filter_race
            ):
                continue
            if ab.filter_victim_keyword is not None:
                if ab.filter_victim_keyword not in dead.template.all_keywords:
                    continue
            eff = ab.effect
            if isinstance(eff, BuffSelf):
                listener.template.bonus_attack += eff.attack
                listener.template.bonus_health += eff.health
                listener.current_health += eff.health
            elif isinstance(eff, DealDamageRandomEnemyMinion):
                for _ in range(max(1, eff.repeats)):
                    _deal_random_enemy_minion_damage(rt, side_idx, eff.amount)
            elif isinstance(eff, BuffDeadMinionNeighborsEffect):
                _buff_neighbors_of_dead(
                    rt,
                    side_idx,
                    dead,
                    attack=eff.attack,
                    health=eff.health,
                )
    _sync_health_all(rt)


def _minion_has_deathrattle(bm: BattleMinion) -> bool:
    return any(ab.trigger == Trigger.ON_DEATH for ab in bm.template.abilities)


def _trigger_random_friendly_deathrattle(
    rt: _CombatRuntime,
    side_idx: int,
    exclude: Optional[BattleMinion],
    effect: TriggerRandomFriendlyDeathrattleEffect,
) -> None:
    side = rt.side(side_idx)
    pool = [
        m
        for m in side.minions
        if m.alive
        and (not effect.exclude_self or m is not exclude)
        and _minion_has_deathrattle(m)
    ]
    for _ in range(max(1, effect.repeats)):
        if not pool:
            return
        pick = pool[int(rt.rng.integers(0, len(pool)))]
        _fire_deathrattle(rt, pick, side_idx)


def _fire_after_attack(
    rt: _CombatRuntime, attacker: BattleMinion, side_idx: int
) -> None:
    side = rt.side(side_idx)
    bf = (rt.side(0), rt.side(1))
    for ab in attacker.template.abilities:
        if ab.trigger != Trigger.ON_AFTER_ATTACK:
            continue
        eff = ab.effect
        if isinstance(eff, TriggerRandomFriendlyDeathrattleEffect):
            _trigger_random_friendly_deathrattle(rt, side_idx, attacker, eff)
        elif isinstance(eff, MultiplySelfAttackEffect):
            cur = attack_value(
                attacker, side, death_resolution=False, battle_field=bf
            )
            attacker.template.bonus_attack += cur * max(0, eff.factor - 1)
        elif isinstance(eff, AddRandomMinionToHandOnKillEffect):
            if rt.attacker_killed_this_swing:
                for _ in range(max(1, eff.count)):
                    _queue_random_combat_hand_add(rt, side_idx, eff.tribe)
    _sync_health_all(rt)


def _fire_friendly_attack_listeners(
    rt: _CombatRuntime, attacker: BattleMinion, attacker_side_idx: int
) -> None:
    side = rt.side(attacker_side_idx)
    for listener in list(side.minions):
        if not listener.alive or listener is attacker:
            continue
        for ab in listener.template.abilities:
            if ab.trigger != Trigger.ON_FRIENDLY_ATTACK:
                continue
            if ab.filter_race is not None and not _matches_tribe_for_aura(
                attacker.template, ab.filter_race
            ):
                continue
            eff = ab.effect
            if isinstance(eff, BuffAttackerOnFriendlyAttackEffect):
                if not _matches_tribe_for_aura(attacker.template, eff.tribe):
                    continue
                attacker.template.bonus_attack += eff.attack
                attacker.template.bonus_health += eff.health
                attacker.current_health += eff.health
            # ALL_FRIENDLY only: before the merge just ``BuffAllFriendlyMinions``
            # reached this branch, so matching every BuffMatching variant here
            # would newly fire the tribe/keyword ones on this trigger.
            elif isinstance(eff, BuffMatching) and eff.target is BuffTarget.ALL_FRIENDLY:
                for ally in side.minions:
                    if not ally.alive:
                        continue
                    ally.template.bonus_attack += eff.attack
                    ally.template.bonus_health += eff.health
                    ally.current_health += eff.health
    _sync_health_all(rt)


def _fire_when_attacked(
    rt: _CombatRuntime,
    victim_side_idx: int,
    victim: BattleMinion,
) -> None:
    side = rt.side(victim_side_idx)
    idx_v = _board_index(side, victim)

    for ab in victim.template.abilities:
        if ab.trigger != Trigger.ON_WHEN_ATTACKED:
            continue
        eff = ab.effect
        if isinstance(eff, BuffAdjacentOnAttackedEffect) and idx_v is not None:
            for j in (idx_v - 1, idx_v + 1):
                if 0 <= j < len(side.minions):
                    ally = side.minions[j]
                    if not ally.alive:
                        continue
                    ally.template.bonus_attack += eff.attack
                    ally.template.bonus_health += eff.health
                    ally.current_health += eff.health

    for listener in list(side.minions):
        if not listener.alive or listener is victim:
            continue
        for ab in listener.template.abilities:
            if ab.trigger != Trigger.ON_FRIENDLY_WHEN_ATTACKED:
                continue
            if ab.filter_victim_keyword is not None:
                if ab.filter_victim_keyword not in victim.template.all_keywords:
                    continue
            eff = ab.effect
            if isinstance(eff, BuffSelf):
                listener.template.bonus_attack += eff.attack
                listener.template.bonus_health += eff.health
                listener.current_health += eff.health
            elif isinstance(eff, BuffAttackedMinionEffect):
                victim.template.bonus_attack += eff.attack
                victim.template.bonus_health += eff.health
                victim.current_health += eff.health
    _sync_health_all(rt)


def _fire_survived_attack_effects(
    rt: _CombatRuntime, side_idx: int, bm: BattleMinion
) -> None:
    if not bm.alive:
        return
    for ab in bm.template.abilities:
        if ab.trigger != Trigger.ON_SURVIVED_ATTACK:
            continue
        if isinstance(ab.effect, AttackImmediatelyAfterSurvivingEffect):
            if rt.bonus_attack_depth > 0:
                continue
            rt.bonus_attack_depth += 1
            try:
                _run_attacker_activation(rt, bm, side_idx, 1 - side_idx)
            finally:
                rt.bonus_attack_depth -= 1


def _fire_friendly_shield_lost_listeners(
    rt: _CombatRuntime, victim_side_idx: int, victim: BattleMinion
) -> None:
    side = rt.side(victim_side_idx)
    for listener in list(side.minions):
        if not listener.alive or listener is victim:
            continue
        for ab in listener.template.abilities:
            if ab.trigger != Trigger.ON_FRIENDLY_SHIELD_LOST:
                continue
            eff = ab.effect
            if isinstance(eff, BuffSelf):
                listener.template.bonus_attack += eff.attack
                listener.template.bonus_health += eff.health
                listener.current_health += eff.health
    _sync_health_all(rt)


def _handle_shield_lost(rt: _CombatRuntime, e: ShieldLost) -> None:
    bm = rt.find_minion(e.victim_side_idx, e.victim_instance_id)
    if bm is not None:
        _fire_self_damaged(rt, e.victim_side_idx, bm)
        _fire_friendly_shield_lost_listeners(rt, e.victim_side_idx, bm)


def _handle_damage_dealt(rt: _CombatRuntime, e: DamageDealt) -> None:
    bm = rt.find_minion(e.victim_side_idx, e.victim_instance_id)
    if bm is not None and bm.alive and e.hp_loss > 0:
        _fire_self_damaged(rt, e.victim_side_idx, bm)
        rt.swing_damage_survivors.append((e.victim_side_idx, e.victim_instance_id))


# --- Deathrattle (ON_DEATH) effect handlers ------------------------------
# One handler per effect type; the Baron-style _deathrattle_multiplier and
# Khadgar-style _summon_multiplier loops live inside each handler (they differ
# per effect). _fire_deathrattle iterates the dead minion's ON_DEATH abilities
# and dispatches by effect type via _DEATHRATTLE_HANDLERS. To add a card
# effect: write a _dr_* handler and register it below.


def _dr_summon(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: SummonEffect
) -> None:
    side = rt.side(side_idx)
    rt.in_death_resolution = False
    if effect.count_from_source_attack:
        bf = (rt.side(0), rt.side(1))
        base = max(
            0,
            attack_value(
                dead,
                side,
                death_resolution=False,
                battle_field=bf,
            ),
        )
    else:
        base = max(0, effect.count)
    rt.in_death_resolution = True
    target_side = _summon_target_side(side_idx, effect.for_opponent)
    anchor = dead if target_side == side_idx else None
    wave_cap = max(1, getattr(effect, "dr_wave_count", 1))
    rep = 0
    while rep < _deathrattle_multiplier(rt.side(side_idx)):
        rep += 1
        n_sum = _summon_multiplier(rt.side(side_idx))
        for _ in range(n_sum):
            for _wave in range(wave_cap):
                for __ in range(base):
                    tok = make_minion(effect.token_id, patch=rt.patch)
                    bm = _summon_insert(
                        rt,
                        target_side,
                        tok,
                        _insert_idx_after(rt.side(target_side), anchor),
                    )
                    if bm is not None and anchor is not None:
                        anchor = bm
                    if effect.attack_immediately:
                        _summon_attack_immediately_if_requested(
                            rt, bm, target_side
                        )
                    if bm is None:
                        break


def _dr_summon_random(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: SummonRandomMinionEffect
) -> None:
    side = rt.side(side_idx)
    race_hs = hs_race_string(effect.race_filter)
    pool = summon_pool_for(
        effect.exact_tier,
        effect.legendary_only,
        effect.require_deathrattle,
        race_hs,
        dead.template.card_id if effect.exclude_source else None,
        patch=rt.patch,
    )
    if not pool:
        return
    target_side = _summon_target_side(side_idx, effect.for_opponent)
    anchor = dead if target_side == side_idx else None
    rep = 0
    while rep < _deathrattle_multiplier(rt.side(side_idx)):
        rep += 1
        n_sum = _summon_multiplier(rt.side(side_idx))
        for _ in range(n_sum):
            for __ in range(effect.count):
                cid = pool[int(rt.rng.integers(0, len(pool)))]
                tok = make_minion(cid, patch=rt.patch)
                bm = _summon_insert(
                    rt,
                    target_side,
                    tok,
                    _insert_idx_after(rt.side(target_side), anchor),
                )
                if bm is None:
                    break
                if anchor is not None:
                    anchor = bm


def _dr_damage_random(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: DealDamageRandomEnemyMinion
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        # ``repeats`` is the effect's own count (golden Kaboom Bot fires twice);
        # the Baron loop above is orthogonal to it. The other two copies of this
        # effect (overkill, friendly-died) always read it — this one used to not.
        for _ in range(max(1, effect.repeats)):
            _deal_random_enemy_minion_damage(rt, side_idx, effect.amount)


def _dr_damage_leftmost(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: DealDamageLeftmostEnemyMinion
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        _deal_leftmost_enemy_minion_damage(rt, side_idx, effect.amount)


def _dr_damage_all(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: DealDamageAllMinions
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        _deal_damage_all_minions(rt, effect.amount)


def _dr_transfer_attack(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: TransferAttackToRandomFriendlyEffect
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        bf = (rt.side(0), rt.side(1))
        atk = attack_value(
            dead,
            side,
            death_resolution=False,
            battle_field=bf,
        )
        if atk <= 0:
            continue
        pool = [
            m
            for m in side.minions
            if m.alive and (not effect.exclude_self or m is not dead)
        ]
        if not pool:
            continue
        tgt = pool[int(rt.rng.integers(0, len(pool)))]
        tgt.template.bonus_attack += atk
    _sync_health_all(rt)


def _dr_summon_copy_hand(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: SummonRandomAndCopyToHandEffect
) -> None:
    side = rt.side(side_idx)
    race_hs = hs_race_string(effect.race_filter)
    pool = summon_pool_for(
        None,
        False,
        False,
        race_hs,
        dead.template.card_id if effect.exclude_source else None,
        patch=rt.patch,
    )
    if not pool:
        return
    target_side = side_idx
    anchor = dead
    rep = 0
    while rep < _deathrattle_multiplier(side):
        rep += 1
        n_sum = _summon_multiplier(side)
        for _ in range(n_sum):
            for __ in range(effect.count):
                cid = pool[int(rt.rng.integers(0, len(pool)))]
                tok = make_minion(cid, patch=rt.patch)
                bm = _summon_insert(
                    rt,
                    target_side,
                    tok,
                    _insert_idx_after(rt.side(target_side), anchor),
                )
                if bm is None:
                    break
                anchor = bm
                _queue_combat_hand_add_card(rt, side_idx, cid)


def _dr_buff_matching(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: BuffMatching
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        for m in side.minions:
            if (not m.alive) or m is dead:
                continue
            # Source exclusion is the caller's job here (``m is dead`` above),
            # so no ``source`` is passed. ``_matches_tribe_for_aura`` and
            # ``minion_matches_tribe`` are the same predicate, verified
            # line-for-line, so the shared helper is exact.
            if not buff_matching_hits(effect, m.template):
                continue
            m.template.bonus_attack += effect.attack
            m.template.bonus_health += effect.health
            m.current_health += effect.health
    _sync_health_all(rt)


def _dr_buff_random_other(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: BuffRandomOtherFriendlyCombat
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        pool = [m for m in side.minions if m.alive and m is not dead]
        if not pool:
            continue
        t = pool[int(rt.rng.integers(0, len(pool)))]
        t.template.bonus_attack += effect.attack
        t.template.bonus_health += effect.health
        t.current_health += effect.health
    _sync_health_all(rt)


def _dr_summon_dead_mechs(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: SummonFirstDeadFriendlyMechsThisCombat
) -> None:
    side = rt.side(side_idx)
    anchor = dead
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        templates = _dead_friendly_mech_templates_ordered(side, dead)
        take = templates[: max(0, effect.count)]
        n_sum = _summon_multiplier(side)
        for _k in range(n_sum):
            for tpl in take:
                bm = _summon_insert(
                    rt,
                    side_idx,
                    copy(tpl),
                    _insert_idx_after(side, anchor),
                )
                if bm is None:
                    break
                anchor = bm


def _dr_grant_kw_random(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: GrantKeywordRandomFriendly
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        for _kw in range(max(1, effect.repeats)):
            pool = []
            for m in side.minions:
                if not m.alive or m is dead:
                    continue
                if effect.filter_race is not None and not _matches_tribe_for_aura(
                    m.template, effect.filter_race
                ):
                    continue
                pool.append(m)
            if not pool:
                continue
            t = pool[int(rt.rng.integers(0, len(pool)))]
            _grant_keyword(rt, side_idx, t, effect.keyword)


def _dr_grant_kw_all_of_tribe(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: GrantKeywordAllFriendlyOfTribe
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        for m in side.minions:
            if (not m.alive) or m is dead:
                continue
            if not _matches_tribe_for_aura(m.template, effect.tribe):
                continue
            _grant_keyword(rt, side_idx, m, effect.keyword)


def _dr_gain_gold(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: GainGoldOnDeathEffect
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        rt.combat_gold[side_idx] += effect.amount


_DEATHRATTLE_HANDLERS = {
    SummonEffect: _dr_summon,
    SummonRandomMinionEffect: _dr_summon_random,
    DealDamageRandomEnemyMinion: _dr_damage_random,
    DealDamageLeftmostEnemyMinion: _dr_damage_leftmost,
    DealDamageAllMinions: _dr_damage_all,
    TransferAttackToRandomFriendlyEffect: _dr_transfer_attack,
    SummonRandomAndCopyToHandEffect: _dr_summon_copy_hand,
    BuffMatching: _dr_buff_matching,
    BuffRandomOtherFriendlyCombat: _dr_buff_random_other,
    SummonFirstDeadFriendlyMechsThisCombat: _dr_summon_dead_mechs,
    GrantKeywordRandomFriendly: _dr_grant_kw_random,
    GrantKeywordAllFriendlyOfTribe: _dr_grant_kw_all_of_tribe,
    GainGoldOnDeathEffect: _dr_gain_gold,
}


def _fire_deathrattle(rt: _CombatRuntime, dead: BattleMinion, side_idx: int) -> None:
    prev = rt.in_death_resolution
    rt.in_death_resolution = True
    try:
        for ab in dead.template.abilities:
            if ab.trigger != Trigger.ON_DEATH:
                continue
            handler = _DEATHRATTLE_HANDLERS.get(type(ab.effect))
            if handler is None:
                # Deliberately loud. This table is the entire contract for what
                # a deathrattle can do, so a miss means a card ships an
                # ON_DEATH ability that nothing implements — half a card that
                # silently does nothing. That is exactly how King Bagurgle's
                # deathrattle stayed broken for the life of the package: the
                # lookup returned None and the dispatch shrugged.
                # ``test_deathrattle_coverage`` pins the shipped patches so
                # this can never actually fire in a game or a training run.
                raise KeyError(
                    f"no deathrattle handler for {type(ab.effect).__name__} "
                    f"(card {dead.template.card_id!r}); register it in "
                    "_DEATHRATTLE_HANDLERS"
                )
            handler(rt, dead, side_idx, ab.effect)
    finally:
        rt.in_death_resolution = prev


def _dead_friendly_mech_templates_ordered(
    side: BattleSide, dead: BattleMinion
) -> List[Minion]:
    out: List[Minion] = []
    for m in side.minions:
        if m.alive or m is dead:
            continue
        if not _is_mech_template(m.template):
            continue
        out.append(copy(m.template))
    return out


def _handle_overkill(rt: _CombatRuntime, e: Overkill) -> None:
    att = rt.find_minion(e.attacker_side_idx, e.attacker_instance_id)
    if att is None or not att.alive or e.excess_damage <= 0:
        return
    for ab in att.template.abilities:
        if ab.trigger != Trigger.ON_OVERKILL:
            continue
        eff = ab.effect
        if isinstance(eff, SummonEffect):
            if eff.for_opponent or eff.count_from_source_attack:
                continue
            side = rt.side(e.attacker_side_idx)
            anchor: Optional[BattleMinion] = att
            n_sum = _summon_multiplier(side)
            for _ in range(max(0, eff.count)):
                for __ in range(n_sum):
                    tok = make_minion(eff.token_id, patch=rt.patch)
                    summoned = _summon_insert(
                        rt,
                        e.attacker_side_idx,
                        tok,
                        _insert_idx_after(side, anchor),
                    )
                    if summoned is None:
                        return
                    anchor = summoned
        elif isinstance(eff, DealDamageRandomEnemyMinion):
            for _ in range(max(1, eff.repeats)):
                _deal_random_enemy_minion_damage(rt, e.attacker_side_idx, eff.amount)
        elif isinstance(eff, DealDamageLeftmostEnemyMinion):
            _deal_leftmost_enemy_minion_damage(rt, e.attacker_side_idx, eff.amount)
        elif isinstance(eff, DealExcessDamageToAdjacentEffect):
            _deal_excess_to_adjacent(
                rt,
                e.victim_side_idx,
                e.victim_instance_id,
                e.excess_damage,
                both_adjacent=eff.both_adjacent,
            )
        # OTHER_OF_TRIBE only, for the same reason as the ALL_FRIENDLY branch
        # above: this used to be reachable by exactly one effect class.
        elif isinstance(eff, BuffMatching) and eff.target is BuffTarget.OTHER_OF_TRIBE:
            side = rt.side(e.attacker_side_idx)
            for m in side.minions:
                if not m.alive or m is att:
                    continue
                if not _matches_tribe_for_aura(m.template, eff.tribe):
                    continue
                m.template.bonus_attack += eff.attack
                m.template.bonus_health += eff.health
                m.current_health += eff.health
            _sync_health_all(rt)


def _handle_minion_died(rt: _CombatRuntime, e: MinionDied) -> None:
    bm = rt.find_minion(e.side_idx, e.instance_id)
    if bm is None or bm.alive or bm.deathrattle_fired:
        return
    bm.deathrattle_fired = True
    if rt.death_hook is not None:
        rt.death_hook(e.side_idx, bm.template.card_id)
    if rt.mech_hook is not None and _is_mech_template(bm.template):
        rt.mech_hook(e.side_idx, copy(bm.template))

    _fire_friendly_minion_died_listeners(rt, bm, e.side_idx)
    attr = rt.kill_attribution.pop((e.side_idx, e.instance_id), None)
    if attr is not None:
        killer_side, killer_id = attr
        _fire_friendly_kill_listeners(rt, killer_side, killer_id)
    _fire_deathrattle(rt, bm, e.side_idx)
    _try_reborn(rt, e.side_idx, bm)
    _sync_health_all(rt)


def _minion_has_reborn(bm: BattleMinion) -> bool:
    return Keyword.REBORN in bm.template.all_keywords and not bm.reborn_consumed


def _strip_reborn_keyword(bm: BattleMinion) -> None:
    kws = frozenset(k for k in bm.template.keywords if k != Keyword.REBORN)
    granted = frozenset(k for k in bm.template.granted_keywords if k != Keyword.REBORN)
    bm.template = replace(bm.template, keywords=kws, granted_keywords=granted)


def _try_reborn(rt: _CombatRuntime, side_idx: int, bm: BattleMinion) -> None:
    if not _minion_has_reborn(bm):
        return
    side = rt.side(side_idx)
    if len(side.minions) >= rt.combat_board_max:
        return  # no slot to come back to
    bm.reborn_consumed = True
    _strip_reborn_keyword(bm)
    bm.current_health = 1
    # The body left the board when it died, so Reborn has to put it back --
    # into the slot it vacated, ahead of whoever slid into it.
    if bm in side.graveyard:
        side.graveyard.remove(bm)
    at = bm.death_pos if 0 <= bm.death_pos <= len(side.minions) else len(side.minions)
    side.minions.insert(at, bm)
    bm.death_pos = -1
    if at <= side.cursor:
        side.cursor += 1
    _mark_health_aura_dirty(rt, side_idx)


def _count_friendlies_of_tribe(side: BattleSide, tribe: Any) -> int:
    return sum(
        1 for m in side.minions if m.alive and _matches_tribe_for_aura(m.template, tribe)
    )

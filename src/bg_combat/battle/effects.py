"""Trigger/effect interpreters and the event handlers they drive.

Deathrattles, on-attack / -death / -summon / -damage listeners, and the damage
helpers they use. ``_handle_*`` event handlers live here; the engine's
``_dispatch`` routes events to them. Some on-death / on-survive effects trigger
an immediate extra attack (engine behaviour) — reached via the local
``_run_attacker_activation`` forwarder below, which breaks the effects<->engine
import cycle.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from src.bg_catalog.cards import make_minion
from copy import copy

from src.bg_recruitment.game_counts import DIED, SUMMONED
from src.bg_core.effects import (
    AttackImmediatelyAfterSurvivingEffect,
    AvengeEffect,
    BuffMatching,
    BuffTarget,
    BuffAdjacentOnAttackedEffect,
    BuffAttackedMinionEffect,
    BuffAttackerOnFriendlyAttackEffect,
    BuffRandomOtherFriendlyCombat,
    AddRandomMinionToHandEffect,
    GainBloodGemsEffect,
    PlayBloodGemsOnAttackerEffect,
    BuffRandomHandMinionEffect,
    BuffSelfOnFriendlyDamageEffect,
    CastSpellAtEffect,
    ImmuneWhileAttackingEffect,
    DamageFromOwnAttackEffect,
    BuffShopOnEveryRefreshEffect,
    KeepCombatGainsEffect,
    IncreaseTavernSpellBonusEffect,
    RaiseStandingBonusEffect,
    RewardAtDamageDealtEffect,
    SelfBonusPerGameCount,
    SummonBestFromHandEffect,
    AddRandomMinionToHandOnKillEffect,
    BloodGemTarget,
    IncreaseBloodGemBonusEffect,
    PlayBloodGemsEffect,
    BuffSelf,
    BuffDeadMinionNeighborsEffect,
    DealDamageRandomEnemyMinion,
    DealDamageLeftmostEnemyMinion,
    DealDamageAllMinions,
    DealExcessDamageToAdjacentEffect,
    TransferAttackToRandomFriendlyEffect,
    SummonRandomAndCopyToHandEffect,
    GainGoldOnDeathEffect,
    GrantKeywordRandomFriendly,
    GrantKeywordAllFriendlyOfTribe,
    Keyword,
    MultiplySelfAttackEffect,
    SummonEffect,
    SummonRandomMinionEffect,
    SummonFirstDeadFriendlyMechsThisCombat,
    SummonOnSelfDamaged,
    SummonRandomOnSelfDamagedEffect,
    TriggerRandomFriendlyDeathrattleEffect,
    Trigger,
)
from src.bg_core.minion import Minion, Race

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
from src.bg_core.board_helpers import (
    apply_buff_matching,
    apply_summoned_listener,
    grant_keyword_random,
    index_of,
    apply_buff_matching,
    grant_keyword_random,
    minion_matches_tribe,
)
from .auras import (
    attack_value,
    _deathrattle_multiplier,
    _grant_keyword,
    _mark_health_aura_dirty,
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


def _is_immune_attacker(rt: _CombatRuntime, bm: BattleMinion) -> bool:
    """Whether this body is mid-swing and immune for the duration of it."""
    if rt.swinging_instance_id != bm.instance_id:
        return False
    return any(
        isinstance(ab.effect, ImmuneWhileAttackingEffect) for ab in bm.abilities
    )


def _enqueue_strike_events(rt: _CombatRuntime, strike: DamageStrike) -> None:
    vic = rt.find_minion(strike.victim_side_idx, strike.victim_instance_id)
    att = rt.find_minion(1 - strike.victim_side_idx, strike.attacker_instance_id)
    if vic is None or not vic.alive or strike.amount <= 0:
        return
    if _is_immune_attacker(rt, vic):
        # "Immune while attacking": the retaliation for its own swing lands on
        # nothing. Checked here rather than only in the generic damage helper,
        # because an attack exchange writes its damage itself.
        return
    v_kw = vic.all_keywords
    if vic.has_shield and Keyword.SHIELD in v_kw:
        vic.has_shield = False
        rt.queue.appendleft(ShieldLost(strike.victim_side_idx, strike.victim_instance_id))
        return

    hp_before = vic.current_health
    att_kw = att.all_keywords if att is not None else frozenset()
    # Venomous is Poisonous that the kill uses up. It is checked *after* the
    # Divine Shield branch above returned, so a hit the shield ate does not
    # spend it — the same rule that keeps Poisonous from being wasted on a
    # shielded body.
    poison = Keyword.POISONOUS in att_kw
    venom = (
        att is not None and not att.venom_spent and Keyword.VENOMOUS in att_kw
    )
    vic.damage_taken += strike.amount
    if poison or venom:
        vic.damage_taken = vic.max_health + vic.aura_health
        if venom:
            att.venom_spent = True
    poison = poison or venom
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
    victims = list(es.minions)
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
    victims = list(es.minions)
    if not victims:
        return
    _deal_damage_to_battle_minion(rt, enemy_side, victims[0], amount)


def _deal_damage_all_minions(rt: _CombatRuntime, amount: int) -> None:
    if amount <= 0:
        return
    for side_idx in (0, 1):
        for m in rt.side(side_idx).iter_living():
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
    idx = index_of(side.minions, dead)
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
            ally.bonus_attack += attack
            ally.bonus_health += health


def _queue_combat_hand_add_card(
    rt: _CombatRuntime, side_idx: int, card_id: str
) -> None:
    rt.seats[side_idx].add_card_to_hand(card_id)


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
    for ab in bm.abilities:
        if ab.trigger != Trigger.ON_SELF_DAMAGED:
            continue
        eff = ab.effect
        if isinstance(eff, BuffRandomHandMinionEffect):
            # "give a minion in your hand +2/+1" — the hand is the seat's, and
            # a combat copy has none.
            rt.seats[side_idx].buff_hand_minion(eff.attack, eff.health, rng=rt.rng)
        elif isinstance(eff, SummonOnSelfDamaged):
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
    _count_arrival(rt, e.side_idx, summoned)

    def grant(minion: BattleMinion, keyword: Keyword) -> None:
        _grant_keyword(rt, e.side_idx, minion, keyword)

    for listener, eff in side.listeners(
        Trigger.ON_FRIENDLY_MINION_SUMMONED, summoned
    ):
        apply_summoned_listener(eff, listener, summoned, grant_keyword=grant)
    _sync_health_all(rt)


def _fire_friendly_kill_listeners(
    rt: _CombatRuntime, killer_side_idx: int, killer_instance_id: int
) -> None:
    killer = rt.find_minion(killer_side_idx, killer_instance_id)
    if killer is None:
        return
    killer_tpl = killer
    side = rt.side(killer_side_idx)
    for listener, eff in side.listeners(
        Trigger.ON_FRIENDLY_KILL, killer_tpl, exclude_subject=False
    ):
        if isinstance(eff, BuffSelf):
            listener.bonus_attack += eff.attack
            listener.bonus_health += eff.health
    _sync_health_all(rt)


def _queue_random_combat_hand_add(
    rt: _CombatRuntime, side_idx: int, tribe: Optional[Any]
) -> None:
    race_hs = hs_race_string(tribe)
    pool = summon_pool_for(None, False, False, race_hs, None, patch=rt.patch)
    if not pool:
        return
    cid = pool[int(rt.rng.integers(0, len(pool)))]
    rt.seats[side_idx].add_card_to_hand(cid)


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
    if _is_immune_attacker(rt, bm):
        # "Immune while attacking": the swing lands, the answer does not.
        return
    if bm.has_shield and Keyword.SHIELD in bm.all_keywords:
        bm.has_shield = False
        rt.queue.append(ShieldLost(side_idx, bm.instance_id))
        return
    bm.damage_taken += amount
    if not bm.alive:
        # Cap the damage at lethal so health reads 0 rather than negative --
        # the same shape the absolute had -- and tell the auras this side lost
        # a body, which is what a dying aura source means for everyone else.
        bm.damage_taken = bm.max_health + bm.aura_health
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
    vi = index_of(side.minions, vic)
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


def _apply_friendly_death_effect(
    rt: _CombatRuntime, listener: BattleMinion, eff, dead: BattleMinion, side_idx: int
) -> None:
    if isinstance(eff, BuffSelf):
        listener.bonus_attack += eff.attack
        listener.bonus_health += eff.health
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
    else:
        raise NotImplementedError(
            f"friendly-death listener {listener.card_id} carries an effect this "
            f"trigger does not handle: {type(eff).__name__}"
        )


def _fire_friendly_minion_died_listeners(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int
) -> None:
    _count_death(rt, dead, side_idx)
    side = rt.side(side_idx)
    for listener, eff in side.listeners(Trigger.ON_FRIENDLY_MINION_DIED, dead):
        if isinstance(eff, AvengeEffect):
            # Count this death; fire and rearm only when the count is reached.
            listener.avenge_progress += 1
            if listener.avenge_progress < max(1, eff.count):
                continue
            listener.avenge_progress = 0
            _apply_friendly_death_effect(rt, listener, eff.effect, dead, side_idx)
            continue
        _apply_friendly_death_effect(rt, listener, eff, dead, side_idx)
    _sync_health_all(rt)


def _minion_has_deathrattle(bm: BattleMinion) -> bool:
    return any(ab.trigger == Trigger.ON_DEATH for ab in bm.abilities)


def _trigger_random_friendly_deathrattle(
    rt: _CombatRuntime,
    side_idx: int,
    exclude: Optional[BattleMinion],
    effect: TriggerRandomFriendlyDeathrattleEffect,
) -> None:
    side = rt.side(side_idx)
    pool = [
        m
        for m in side.iter_living()
        if (not effect.exclude_self or m is not exclude)
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
    for ab in attacker.abilities:
        if ab.trigger != Trigger.ON_AFTER_ATTACK:
            continue
        eff = ab.effect
        if isinstance(eff, TriggerRandomFriendlyDeathrattleEffect):
            _trigger_random_friendly_deathrattle(rt, side_idx, attacker, eff)
        elif isinstance(eff, MultiplySelfAttackEffect):
            cur = attack_value(
                attacker, side, death_resolution=False, battle_field=bf
            )
            attacker.bonus_attack += cur * max(0, eff.factor - 1)
        elif isinstance(eff, AddRandomMinionToHandOnKillEffect):
            if rt.attacker_killed_this_swing:
                for _ in range(max(1, eff.count)):
                    _queue_random_combat_hand_add(rt, side_idx, eff.tribe)
    _sync_health_all(rt)


def _fire_friendly_attack_listeners(
    rt: _CombatRuntime, attacker: BattleMinion, attacker_side_idx: int
) -> None:
    side = rt.side(attacker_side_idx)
    for listener, eff in side.listeners(Trigger.ON_FRIENDLY_ATTACK, attacker):
        if isinstance(eff, BuffAttackerOnFriendlyAttackEffect):
            if not minion_matches_tribe(attacker, eff.tribe):
                continue
            attacker.bonus_attack += eff.attack
            attacker.bonus_health += eff.health
        elif isinstance(eff, PlayBloodGemsOnAttackerEffect):
            _play_combat_blood_gems(
                rt,
                attacker,
                attacker_side_idx,
                PlayBloodGemsEffect(target=BloodGemTarget.SELF, count=eff.count),
            )
        elif isinstance(eff, BuffMatching):
            apply_buff_matching(eff, list(side.iter_living()), listener)
    _sync_health_all(rt)


def _summon_best_from_hand(
    rt: _CombatRuntime,
    side_idx: int,
    source: Optional[BattleMinion],
    effect: SummonBestFromHandEffect,
) -> None:
    """Summon the biggest matching card in hand, for this combat only.

    Two cards print this at different moments — a Rally and a Start of Combat —
    and it is the same reach either way: the card stays in hand and a copy of it
    joins the fight, carrying whatever the real card had gained.

    A card that has been summoned this fight is locked and this looks past it,
    so a second Rally reaches the next-biggest rather than the same card twice.
    """
    held = _hand_candidates(rt, side_idx, effect.filter_race)
    if not held:
        return
    instance_id, card_id, attack, health = max(held, key=lambda row: row[2])
    template = rt.patch.templates.get(card_id)
    if template is None:
        return
    body = copy(template)
    body.bonus_attack += max(0, attack - template.raw_attack)
    body.bonus_health += max(0, health - template.max_health)
    if _summon_beside(
        rt, side_idx, source, SummonEffect(token_id=card_id, count=1), template=body
    ):
        rt.hand_summoned[side_idx].add(instance_id)


def cast_spell_in_combat(
    rt: _CombatRuntime, side_idx: int, source: Optional[BattleMinion], card_id: str
) -> None:
    """Resolve a named spell against a combat side.

    Two Dragons cast one mid-fight — a Start of Combat and a Rally — and the
    spells they cast are board-wide buffs, so what a cast *means* here is the
    spell's own abilities applied to the living side. The spell is never in
    anyone's hand and nothing is spent.
    """
    spell = rt.patch.tavern_spells.get(card_id)
    if spell is None:
        return
    side = rt.side(side_idx)
    for ability in spell.abilities:
        eff = ability.effect
        if isinstance(eff, BuffMatching):
            apply_buff_matching(eff, list(side.iter_living()), source)
        else:
            raise NotImplementedError(
                f"spell {card_id} does {type(eff).__name__}, which no combat cast "
                f"knows how to resolve"
            )
    _sync_health_all(rt)


def _hand_candidates(rt: _CombatRuntime, side_idx: int, filter_race):
    """Hand minions a summon-from-hand may still choose.

    Filtered by tribe where the card names one, and by the lock either way: a
    body already summoned into this fight is not available again.
    """
    locked = rt.hand_summoned[side_idx]
    out = []
    for instance_id, card_id, attack, health in rt.seats[side_idx].hand_minions():
        if instance_id in locked:
            continue
        if filter_race is not None:
            template = rt.patch.templates.get(card_id)
            if template is None or not minion_matches_tribe(template, filter_race):
                continue
        out.append((instance_id, card_id, attack, health))
    return out


def _summon_beside(
    rt: _CombatRuntime,
    side_idx: int,
    source: Optional[BattleMinion],
    effect: SummonEffect,
    *,
    template: Optional[Minion] = None,
) -> bool:
    """Summon ``effect``'s tokens beside a *living* source, Khadgar included.

    The deathrattle path (``_dr_summon``) cannot serve here: it summons into the
    slot a dead body vacated and multiplies by the side's deathrattle count. A
    Rally or an Overkill fires while its minion is still standing, so the tokens
    land to its right and only the summon multiplier applies.

    Returns ``False`` once the board is full and the rest of the summon is lost.
    """
    side = rt.side(side_idx)
    anchor: Optional[BattleMinion] = source
    n_sum = _summon_multiplier(side)
    for _ in range(max(0, effect.count)):
        for __ in range(n_sum):
            tok = (
                copy(template)
                if template is not None
                else make_minion(effect.token_id, patch=rt.patch)
            )
            summoned = _summon_insert(rt, side_idx, tok, _insert_idx_after(side, anchor))
            if summoned is None:
                return False
            anchor = summoned
    return True


def _fire_rally(
    rt: _CombatRuntime,
    attacker: BattleMinion,
    attacker_side_idx: int,
    target: BattleMinion,
) -> None:
    """Rally: "Whenever this attacks", with the target already chosen.

    Fires before either combatant's damage is measured, so a Rally that buffs
    the attacker is felt by the swing that triggered it, and one that strips the
    target's keywords still finds the target alive. ``ON_AFTER_ATTACK`` is the
    other end of the same swing and stays where it is.

    Effects reaching here are the ones a Rally can be written from today; a card
    that needs "the target" specifically (removing its Reborn, splashing damage
    onto it and a neighbour) needs an effect that takes one, which is why the
    target is threaded in rather than left to the call site to forget later.
    """
    side = rt.side(attacker_side_idx)
    for ab in attacker.abilities:
        if ab.trigger != Trigger.ON_ATTACK:
            continue
        eff = ab.effect
        if isinstance(eff, BuffSelf):
            attacker.bonus_attack += eff.attack
            attacker.bonus_health += eff.health
        elif isinstance(eff, BuffMatching):
            # Through the shared applier, not a loop of its own: this branch
            # used to spell out ALL_FRIENDLY by hand and so ignored every field
            # BuffMatching has grown since — limit, grant_keyword, and the
            # exclude_source that "give your *other* minions" is made of.
            apply_buff_matching(eff, list(side.iter_living()), attacker)
        elif isinstance(eff, DealDamageRandomEnemyMinion):
            for _ in range(max(1, eff.repeats)):
                _deal_random_enemy_minion_damage(rt, attacker_side_idx, eff.amount)
        elif isinstance(eff, AddRandomMinionToHandEffect):
            # "Rally: Get a random Beast" — the card lands in hand after combat,
            # through the same queue a Deathrattle hand-add uses.
            _queue_random_combat_hand_add(rt, attacker_side_idx, eff.tribe)
        elif isinstance(eff, PlayBloodGemsEffect):
            _play_combat_blood_gems(rt, attacker, attacker_side_idx, eff)
        elif isinstance(eff, CastSpellAtEffect):
            cast_spell_in_combat(rt, attacker_side_idx, attacker, eff.card_id)
        elif isinstance(eff, DamageFromOwnAttackEffect):
            amount = attack_value(attacker, side, death_resolution=False)
            enemy_idx = 1 - attacker_side_idx
            enemy = rt.side(enemy_idx)
            living = list(enemy.iter_living())
            if target in living:
                hit = [living.index(target)]
                if eff.include_adjacent:
                    hit += [i for i in (hit[0] - 1, hit[0] + 1) if 0 <= i < len(living)]
                for i in sorted(set(hit)):
                    _deal_damage_to_battle_minion(rt, enemy_idx, living[i], amount)
        elif isinstance(eff, GrantKeywordRandomFriendly):
            grant_keyword_random(
                eff,
                list(side.iter_living()),
                attacker,
                rng=rt.rng,
                grant=lambda m, kw: _grant_keyword(rt, attacker_side_idx, m, kw),
            )
        elif isinstance(eff, GainBloodGemsEffect):
            # Into hand, not onto a body — the seat holds it until it is played.
            rt.seats[attacker_side_idx].gain_blood_gems(eff.count)
        elif isinstance(eff, RaiseStandingBonusEffect):
            # "Rally: your Undead have +2 Attack this game" — owed to the seat,
            # not to this copy, and to Undead it has not bought yet.
            rt.seats[attacker_side_idx].raise_standing_bonus(
                eff.scope_kind,
                eff.scope_key if eff.scope_key is not None else attacker.card_id,
                eff.attack,
                eff.health,
            )
        elif isinstance(eff, IncreaseTavernSpellBonusEffect):
            rt.seats[attacker_side_idx].raise_tavern_spell_bonus(eff.attack, eff.health)
        elif isinstance(eff, IncreaseBloodGemBonusEffect):
            # "Rally: Your Blood Gems give an extra +1/+1 this game" — raised
            # mid-combat, and a permanent Gem played after it is worth more.
            rt.seats[attacker_side_idx].raise_blood_gem_value(eff.attack, eff.health)
        elif isinstance(eff, SummonBestFromHandEffect):
            _summon_best_from_hand(rt, attacker_side_idx, attacker, eff)
        elif isinstance(eff, SummonEffect) and not (
            eff.for_opponent or eff.count_from_source_attack
        ):
            # "Rally: Summon a 1/1 Beast" — the token lands beside the attacker
            # mid-swing, in time to be attacked itself this combat.
            _summon_beside(rt, attacker_side_idx, attacker, eff)
        else:
            raise NotImplementedError(
                f"Rally effect {type(eff).__name__} has no combat handler "
                f"(minion {attacker.card_id})"
            )
    _sync_health_all(rt)


def _combat_blood_gem_targets(
    side: BattleSide, source: BattleMinion, target: BloodGemTarget
) -> List[BattleMinion]:
    """Who a "this plays a Blood Gem on ..." reaches on a combat board."""
    living = list(side.iter_living())
    if target is BloodGemTarget.SELF:
        return [source] if source.alive else []
    if target is BloodGemTarget.ALL_FRIENDLY:
        return living
    if target is BloodGemTarget.ALL_OTHER_FRIENDLY:
        return [m for m in living if m is not source]
    if target is BloodGemTarget.ALL_FRIENDLY_QUILBOAR:
        return [m for m in living if m.race is Race.QUILBOAR]
    if target is BloodGemTarget.ADJACENT:
        idx = index_of(side.minions, source)
        if idx is None:
            return []
        return [
            side.minions[j]
            for j in (idx - 1, idx + 1)
            if 0 <= j < len(side.minions)
        ]
    raise NotImplementedError(f"no combat target resolution for {target!r}")


def _play_combat_blood_gems(
    rt: _CombatRuntime,
    source: BattleMinion,
    side_idx: int,
    effect: PlayBloodGemsEffect,
) -> None:
    """A Gem played inside a fight, on the combat board and maybe beyond it.

    The stats always land on the copy, so the Gem counts for the rest of this
    combat. A Gem the card prints as *permanent* is additionally written
    through to the owner's real minion, found by ``origin_instance_id`` — the
    copy fights under a combat-local id, so the board minion's own identity has
    to ride along for anything to be able to point back at it.
    """
    seat = rt.seats[side_idx]
    attack, health = seat.blood_gem_value()
    for target in _combat_blood_gem_targets(rt.side(side_idx), source, effect.target):
        if target.cannot_gain_stats:
            continue
        for _ in range(max(0, int(effect.count))):
            target.bonus_attack += attack
            target.bonus_health += health
            target.blood_gem_attack += attack
            target.blood_gem_health += health
        if effect.permanent:
            seat.play_permanent_blood_gem(
                target.origin_instance_id, max(0, int(effect.count))
            )
    _sync_health_all(rt)


def _fire_when_attacked(
    rt: _CombatRuntime,
    victim_side_idx: int,
    victim: BattleMinion,
) -> None:
    side = rt.side(victim_side_idx)
    idx_v = index_of(side.minions, victim)

    for ab in victim.abilities:
        if ab.trigger != Trigger.ON_WHEN_ATTACKED:
            continue
        eff = ab.effect
        if isinstance(eff, BuffAdjacentOnAttackedEffect) and idx_v is not None:
            for j in (idx_v - 1, idx_v + 1):
                if 0 <= j < len(side.minions):
                    ally = side.minions[j]
                    if not ally.alive:
                        continue
                    ally.bonus_attack += eff.attack
                    ally.bonus_health += eff.health

    for listener, eff in side.listeners(
        Trigger.ON_FRIENDLY_WHEN_ATTACKED, victim
    ):
        if isinstance(eff, BuffSelf):
            listener.bonus_attack += eff.attack
            listener.bonus_health += eff.health
        elif isinstance(eff, BuffAttackedMinionEffect):
            victim.bonus_attack += eff.attack
            victim.bonus_health += eff.health
    _sync_health_all(rt)


def _fire_survived_attack_effects(
    rt: _CombatRuntime, side_idx: int, bm: BattleMinion
) -> None:
    if not bm.alive:
        return
    for ab in bm.abilities:
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
    for listener, eff in side.listeners(Trigger.ON_FRIENDLY_SHIELD_LOST, victim):
        if isinstance(eff, BuffSelf):
            listener.bonus_attack += eff.attack
            listener.bonus_health += eff.health
    _sync_health_all(rt)


def _handle_shield_lost(rt: _CombatRuntime, e: ShieldLost) -> None:
    bm = rt.find_minion(e.victim_side_idx, e.victim_instance_id)
    if bm is not None:
        _fire_self_damaged(rt, e.victim_side_idx, bm)
        _fire_friendly_shield_lost_listeners(rt, e.victim_side_idx, bm)


def _handle_damage_dealt(rt: _CombatRuntime, e: DamageDealt) -> None:
    _count_damage_dealt(rt, e)
    _fire_friendly_damage_listeners(rt, e)
    bm = rt.find_minion(e.victim_side_idx, e.victim_instance_id)
    if bm is not None and bm.alive and e.hp_loss > 0:
        _fire_self_damaged(rt, e.victim_side_idx, bm)
        rt.swing_damage_survivors.append((e.victim_side_idx, e.victim_instance_id))


def _fire_friendly_damage_listeners(rt: _CombatRuntime, e: DamageDealt) -> None:
    """"After another friendly Demon deals damage, gain +1/+2 permanently."

    Damage, not an attack: this fires for anything that hurts an enemy, which
    is what separates it from ON_FRIENDLY_ATTACK. Permanent gains are written
    back to the owner's minion as well as to the copy, by the same route
    Tarecgosa's take.
    """
    if e.hp_loss <= 0:
        return
    dealer_side = 1 - e.victim_side_idx
    dealer = rt.find_minion(dealer_side, e.source_instance_id)
    if dealer is None:
        return
    for listener in rt.side(dealer_side).iter_living():
        if listener is dealer:
            continue
        for ability in listener.abilities:
            eff = ability.effect
            if not isinstance(eff, BuffSelfOnFriendlyDamageEffect):
                continue
            if eff.filter_race is not None and not minion_matches_tribe(
                dealer, eff.filter_race
            ):
                continue
            listener.bonus_attack += eff.attack
            listener.bonus_health += eff.health
            if eff.permanent:
                rt.seats[dealer_side].keep_combat_gains(
                    listener.origin_instance_id, eff.attack, eff.health, frozenset()
                )
    _sync_health_all(rt)


def _count_damage_dealt(rt: _CombatRuntime, e: DamageDealt) -> None:
    """Tally damage for a body that is counting toward a reward.

    Damage dealt *by* it, so the source is looked up on the other side of the
    victim, and it is written through to the owner's real minion by origin id:
    the tally is that body's — two Treasure Parrots count separately — and the
    copy doing the swinging does not survive the fight.
    """
    if e.hp_loss <= 0:
        return
    dealer_side = 1 - e.victim_side_idx
    dealer = rt.find_minion(dealer_side, e.source_instance_id)
    if dealer is None:
        return
    for ability in dealer.abilities:
        if ability.trigger is not Trigger.AURA:
            continue
        eff = ability.effect
        if isinstance(eff, RewardAtDamageDealtEffect):
            rt.seats[dealer_side].record_damage_dealt(
                dealer.origin_instance_id, e.hp_loss, eff.threshold, eff.card_id
            )


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
    race_hs = hs_race_string(effect.race_filter)
    pool = summon_pool_for(
        effect.exact_tier,
        effect.legendary_only,
        effect.require_deathrattle,
        race_hs,
        dead.card_id if effect.exclude_source else None,
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
                if effect.set_attack or effect.set_health:
                    # "Set its stats to 6/6" — set, not added, so whatever it
                    # rolled is replaced rather than improved.
                    tok.base_attack = effect.set_attack
                    tok.base_health = effect.set_health
                    tok.bonus_attack = 0
                    tok.bonus_health = 0
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
            if not effect.exclude_self or m is not dead
        ]
        if not pool:
            continue
        tgt = pool[int(rt.rng.integers(0, len(pool)))]
        tgt.bonus_attack += atk
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
        dead.card_id if effect.exclude_source else None,
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
    apply_buff_matching(
        effect, side.minions, dead, repeats=_deathrattle_multiplier(side)
    )
    _sync_health_all(rt)


def _count_arrival(rt: _CombatRuntime, side_idx: int, arrived: BattleMinion) -> None:
    """Count a mid-combat summon on the owner's tally, if the card keeps one.

    Read off the newcomer, so one arrival counts once however many copies are
    already standing.
    """
    for ability in arrived.abilities:
        if ability.trigger is not Trigger.AURA:
            continue
        eff = ability.effect
        if isinstance(eff, SelfBonusPerGameCount) and eff.counter == SUMMONED:
            rt.seats[side_idx].bump_game_count(
                eff.counter, eff.subject or arrived.card_id
            )


def _count_death(rt: _CombatRuntime, dead: BattleMinion, side_idx: int) -> None:
    """Count a death on the owner's tally ("each Eternal Knight that died")."""
    for ability in dead.abilities:
        if ability.trigger is not Trigger.AURA:
            continue
        eff = ability.effect
        if isinstance(eff, SelfBonusPerGameCount) and eff.counter == DIED:
            rt.seats[side_idx].bump_game_count(
                eff.counter, eff.subject or dead.card_id
            )


def _dr_promise_refresh_buff(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: BuffShopOnEveryRefreshEffect
) -> None:
    """Waveling: from now on, every tavern roll buffs one minion in it."""
    rt.seats[side_idx].add_refresh_buff(effect.attack, effect.health)


def _dr_buff_hand_minion(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: BuffRandomHandMinionEffect
) -> None:
    """"Deathrattle: give a random minion in your hand +7/+7" — a seat write,
    because a combat copy has no hand to reach into."""
    rt.seats[side_idx].buff_hand_minion(effect.attack, effect.health, rng=rt.rng)


def _dr_buff_random_other(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: BuffRandomOtherFriendlyCombat
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        pool = [
            m
            for m in side.minions
            if m is not dead
            and (effect.filter_race is None or minion_matches_tribe(m, effect.filter_race))
        ]
        if not pool:
            continue
        t = pool[int(rt.rng.integers(0, len(pool)))]
        t.bonus_attack += effect.attack
        t.bonus_health += effect.health
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

    def grant(minion: BattleMinion, keyword: Keyword) -> None:
        _grant_keyword(rt, side_idx, minion, keyword)

    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        grant_keyword_random(
            effect, side.minions, dead, rng=rt.rng, grant=grant
        )


def _dr_grant_kw_all_of_tribe(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: GrantKeywordAllFriendlyOfTribe
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        for m in side.minions:
            if m is dead:
                continue
            if not minion_matches_tribe(m, effect.tribe):
                continue
            _grant_keyword(rt, side_idx, m, effect.keyword)


def _dr_gain_gold(
    rt: _CombatRuntime, dead: BattleMinion, side_idx: int, effect: GainGoldOnDeathEffect
) -> None:
    side = rt.side(side_idx)
    rep_dr = 0
    while rep_dr < _deathrattle_multiplier(side):
        rep_dr += 1
        rt.seats[side_idx].gain_gold(effect.amount)


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
    BuffRandomHandMinionEffect: _dr_buff_hand_minion,
    BuffShopOnEveryRefreshEffect: _dr_promise_refresh_buff,
}


def _fire_deathrattle(rt: _CombatRuntime, dead: BattleMinion, side_idx: int) -> None:
    prev = rt.in_death_resolution
    rt.in_death_resolution = True
    try:
        for ab in dead.abilities:
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
                    f"(card {dead.card_id!r}); register it in "
                    "_DEATHRATTLE_HANDLERS"
                )
            handler(rt, dead, side_idx, ab.effect)
    finally:
        rt.in_death_resolution = prev


def _dead_friendly_mech_templates_ordered(
    side: BattleSide, dead: BattleMinion
) -> List[Minion]:
    out: List[Minion] = []
    # The dead live in the graveyard, in death order -- which is exactly the
    # order this card asks for ("the first 2 friendly Mechs that died"). It
    # used to scan ``minions`` for bodies, and once bodies stopped being left
    # there it silently found none: Kangor's Apprentice resummoned nothing.
    for m in side.graveyard:
        if m is dead:
            continue
        if not _is_mech_template(m):
            continue
        out.append(copy(m))
    return out


def _handle_overkill(rt: _CombatRuntime, e: Overkill) -> None:
    att = rt.find_minion(e.attacker_side_idx, e.attacker_instance_id)
    if att is None or not att.alive or e.excess_damage <= 0:
        return
    for ab in att.abilities:
        if ab.trigger != Trigger.ON_OVERKILL:
            continue
        eff = ab.effect
        if isinstance(eff, SummonEffect):
            if eff.for_opponent or eff.count_from_source_attack:
                continue
            if not _summon_beside(rt, e.attacker_side_idx, att, eff):
                return
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
            apply_buff_matching(eff, rt.side(e.attacker_side_idx).minions, att)
            _sync_health_all(rt)


def _handle_minion_died(rt: _CombatRuntime, e: MinionDied) -> None:
    bm = rt.find_minion(e.side_idx, e.instance_id)
    if bm is None or bm.alive or bm.deathrattle_fired:
        return
    bm.deathrattle_fired = True
    if rt.death_hook is not None:
        rt.death_hook(e.side_idx, bm.card_id)
    if rt.mech_hook is not None and _is_mech_template(bm):
        rt.mech_hook(e.side_idx, copy(bm))

    _fire_friendly_minion_died_listeners(rt, bm, e.side_idx)
    attr = rt.kill_attribution.pop((e.side_idx, e.instance_id), None)
    if attr is not None:
        killer_side, killer_id = attr
        _fire_friendly_kill_listeners(rt, killer_side, killer_id)
    _fire_deathrattle(rt, bm, e.side_idx)
    _try_reborn(rt, e.side_idx, bm)
    _sync_health_all(rt)


def _minion_has_reborn(bm: BattleMinion) -> bool:
    return Keyword.REBORN in bm.all_keywords and not bm.reborn_consumed


def _strip_reborn_keyword(bm: BattleMinion) -> None:
    # In place: the minion *is* the entity now, so rebinding a local (which is
    # what the mechanical de-wrapping turned this into) would drop the change.
    bm.keywords = frozenset(k for k in bm.keywords if k != Keyword.REBORN)
    bm.granted_keywords = frozenset(
        k for k in bm.granted_keywords if k != Keyword.REBORN
    )


def _try_reborn(rt: _CombatRuntime, side_idx: int, bm: BattleMinion) -> None:
    if not _minion_has_reborn(bm):
        return
    side = rt.side(side_idx)
    if len(side.minions) >= rt.combat_board_max:
        return  # no slot to come back to
    bm.reborn_consumed = True
    _strip_reborn_keyword(bm)
    bm.damage_taken = bm.max_health + bm.aura_health - 1
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
        1 for m in side.minions if minion_matches_tribe(m, tribe)
    )

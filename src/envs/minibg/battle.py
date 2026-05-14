from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, List, Optional, Tuple, Union

import numpy as np

from .actions import BOARD_SIZE, DAMAGE_CAP

# Battlegrounds combat board limit (shop / placement still uses ``BOARD_SIZE`` in ``actions``).
COMBAT_BOARD_MAX = 7
from .cards import make_minion
from .effects import (
    AdjacentStatAura,
    AttackBonusPerOtherMurlocGlobal,
    BuffAllFriendlyMinions,
    BuffAllFriendlyOfTribe,
    BuffAllWithKeyword,
    BuffListenerIfSummonedMatches,
    BuffRandomOtherFriendlyCombat,
    BuffSelf,
    BuffSummonedIfRace,
    CleaveOnAttack,
    DealDamageRandomEnemyMinion,
    GrantKeywordRandomFriendly,
    GrantListenerKeywordIfSummonedMatches,
    Keyword,
    KeywordStatAura,
    DeathrattleMultiplierAura,
    KangorSummonCopy,
    StatAura,
    SummonEffect,
    SummonRandomMinionEffect,
    SummonMultiplierAura,
    SummonOnSelfDamaged,
    TribalOtherStatAura,
    Trigger,
    ZappTargeting,
)
from .state import Minion, Race
from .summon_pool import build_summon_pool, hs_race_string, minion_from_hsjson_card


@dataclass
class BattleMinion:
    template: Minion
    current_health: int
    shield_armed: bool
    deathrattle_fired: bool = False
    instance_id: int = 0
    health_aura_snapshot: int = 0

    @property
    def alive(self) -> bool:
        return self.current_health > 0

    @property
    def raw_attack(self) -> int:
        return self.template.raw_attack

    @property
    def tier(self) -> int:
        return self.template.tier

    @classmethod
    def from_minion(cls, minion: Minion, instance_id: int) -> "BattleMinion":
        armed = minion.has_shield and Keyword.SHIELD in minion.all_keywords
        return cls(
            template=minion,
            current_health=minion.max_health,
            shield_armed=armed,
            instance_id=instance_id,
        )


@dataclass
class BattleSide:
    minions: List[BattleMinion] = field(default_factory=list)
    cursor: int = 0

    def alive_minions(self) -> List[BattleMinion]:
        return [m for m in self.minions if m.alive]

    def alive_count(self) -> int:
        return sum(1 for m in self.minions if m.alive)

    def has_alive(self) -> bool:
        return any(m.alive for m in self.minions)


@dataclass(frozen=True)
class BeginAttackExchange:
    """Attacker and defender sides for this strike (board indices, not minion refs)."""

    attacker_side_idx: int
    defender_side_idx: int


@dataclass(frozen=True)
class ShieldLost:
    victim_side_idx: int
    victim_instance_id: int


@dataclass(frozen=True)
class DamageDealt:
    victim_side_idx: int
    victim_instance_id: int
    source_instance_id: int
    hp_loss: int
    set_hp_to_zero_from_poison: bool


@dataclass(frozen=True)
class Overkill:
    victim_side_idx: int
    victim_instance_id: int
    attacker_side_idx: int
    attacker_instance_id: int
    excess_damage: int


@dataclass(frozen=True)
class AttackCompleted:
    """Both combatants have applied their strike damage; enqueue death batch next."""


@dataclass(frozen=True)
class MinionDied:
    side_idx: int
    instance_id: int


@dataclass(frozen=True)
class MinionSummoned:
    side_idx: int
    instance_id: int
    template_card_id: str


@dataclass(frozen=True)
class DamageStrike:
    attacker_instance_id: int
    victim_instance_id: int
    victim_side_idx: int
    amount: int


BattleEvent = Union[
    BeginAttackExchange,
    ShieldLost,
    DamageDealt,
    Overkill,
    AttackCompleted,
    MinionDied,
    MinionSummoned,
    DamageStrike,
]


def _deathrattle_multiplier(side: BattleSide) -> int:
    """Product of Baron-style auras on living minions (re-read at DR execution time)."""
    p = 1
    for bm in side.minions:
        if not bm.alive:
            continue
        for ab in bm.template.abilities:
            if ab.trigger == Trigger.AURA and isinstance(
                ab.effect, DeathrattleMultiplierAura
            ):
                p *= ab.effect.factor
    return p


def _summon_multiplier(side: BattleSide) -> int:
    """Product of Khadgar-style auras on living minions."""
    p = 1
    for bm in side.minions:
        if not bm.alive:
            continue
        for ab in bm.template.abilities:
            if ab.trigger == Trigger.AURA and isinstance(
                ab.effect, SummonMultiplierAura
            ):
                p *= ab.effect.factor
    return p


def _board_index(side: BattleSide, bm: BattleMinion) -> Optional[int]:
    try:
        return side.minions.index(bm)
    except ValueError:
        return None


def _matches_tribe_for_aura(recipient_t: Minion, required: Any) -> bool:
    r = recipient_t.race
    if r is None:
        return False
    if required == Race.ALL or r == Race.ALL:
        return True
    return r == required


def _recipient_gets_stat_from_source(
    recipient: BattleMinion,
    source: BattleMinion,
    eff: object,
    *,
    idx_r: int,
    idx_s: int,
) -> Tuple[int, int]:
    atk, hp = 0, 0
    if isinstance(eff, StatAura):
        atk, hp = eff.attack, eff.health
    elif isinstance(eff, TribalOtherStatAura):
        if _matches_tribe_for_aura(recipient.template, eff.tribe):
            atk, hp = eff.attack, eff.health
    elif isinstance(eff, KeywordStatAura):
        if eff.keyword in recipient.template.all_keywords:
            atk, hp = eff.attack, eff.health
    elif isinstance(eff, AdjacentStatAura):
        if idx_r in (idx_s - 1, idx_s + 1):
            atk, hp = eff.attack, eff.health
    return atk, hp


def _iter_stat_aura_contributions(
    recipient: BattleMinion,
    source: BattleMinion,
    side: BattleSide,
) -> Tuple[int, int]:
    if source is recipient or not source.alive:
        return 0, 0
    idx_r = _board_index(side, recipient)
    idx_s = _board_index(side, source)
    if idx_r is None or idx_s is None:
        return 0, 0
    ta, th = 0, 0
    for ab in source.template.abilities:
        if ab.trigger != Trigger.AURA:
            continue
        a, h = _recipient_gets_stat_from_source(
            recipient, source, ab.effect, idx_r=idx_r, idx_s=idx_s
        )
        ta += a
        th += h
    return ta, th


def _self_aura_attack_bonus(
    minion: BattleMinion,
    battle_field: Optional[Tuple[BattleSide, BattleSide]],
    own_side: BattleSide,
) -> int:
    sides: Tuple[BattleSide, ...] = (
        battle_field if battle_field is not None else (own_side,)
    )
    bonus = 0
    for ab in minion.template.abilities:
        if ab.trigger != Trigger.AURA:
            continue
        eff = ab.effect
        if isinstance(eff, AttackBonusPerOtherMurlocGlobal):
            n = 0
            for s in sides:
                for m in s.minions:
                    if m.alive and m is not minion:
                        if m.template.race in (Race.MURLOC, Race.ALL):
                            n += 1
            bonus += eff.per_attack * n
    return bonus


def attack_value(
    minion: BattleMinion,
    side: BattleSide,
    *,
    death_resolution: bool,
    battle_field: Optional[Tuple[BattleSide, BattleSide]] = None,
) -> int:
    """During death-resolution windows stat auras do not apply (BG-style snapshot)."""
    if death_resolution:
        return minion.template.raw_attack
    bonus = 0
    for other in side.minions:
        a, _ = _iter_stat_aura_contributions(minion, other, side)
        bonus += a
    return minion.raw_attack + bonus + _self_aura_attack_bonus(
        minion, battle_field, side
    )


def health_aura_bonus(
    minion: BattleMinion,
    side: BattleSide,
    *,
    death_resolution: bool,
) -> int:
    if death_resolution:
        return 0
    bonus = 0
    for other in side.minions:
        _, h = _iter_stat_aura_contributions(minion, other, side)
        bonus += h
    return bonus


def _sync_health_aura_side(side: BattleSide, death_resolution: bool) -> None:
    for bm in side.minions:
        if not bm.alive:
            continue
        b = health_aura_bonus(bm, side, death_resolution=death_resolution)
        prev = bm.health_aura_snapshot
        delta = b - prev
        bm.health_aura_snapshot = b
        bm.current_health += delta
        emax = bm.template.max_health + b
        if bm.current_health > emax:
            bm.current_health = emax


def _sync_health_all(rt: _CombatRuntime) -> None:
    dr = rt.in_death_resolution
    _sync_health_aura_side(rt.side(0), dr)
    _sync_health_aura_side(rt.side(1), dr)


def attack_with_auras(minion: BattleMinion, side: BattleSide) -> int:
    """Attack during the combat strike phase (auras from living neighbors apply)."""
    return attack_value(minion, side, death_resolution=False)


@dataclass
class _CombatRuntime:
    sides: Tuple[BattleSide, BattleSide]
    rng: np.random.Generator
    queue: Deque[BattleEvent] = field(default_factory=deque)
    next_id: int = 1
    in_death_resolution: bool = False
    death_hook: Optional[Callable[[int, str], None]] = None
    mech_hook: Optional[Callable[[int, Minion], None]] = None

    def alloc_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    def side(self, idx: int) -> BattleSide:
        return self.sides[idx]

    def find_minion(self, side_idx: int, instance_id: int) -> Optional[BattleMinion]:
        for m in self.side(side_idx).minions:
            if m.instance_id == instance_id:
                return m
        return None


def _is_mech_template(m: Minion) -> bool:
    return m.race in (Race.MECHANICAL, Race.ALL)


def _build_side(board: List[Minion], rt: _CombatRuntime) -> BattleSide:
    out: List[BattleMinion] = []
    for m in board:
        bid = rt.alloc_id()
        out.append(BattleMinion.from_minion(copy(m), bid))
    return BattleSide(minions=out)


def build_battle_side(board: List[Minion]) -> BattleSide:
    """Build a battle line with fresh instance IDs (tests, tooling)."""
    rt = _CombatRuntime(sides=(BattleSide(), BattleSide()), rng=np.random.default_rng(0))
    return _build_side(board, rt)


def _attacker_has_zapp_targeting(attacker: BattleMinion) -> bool:
    for ab in attacker.template.abilities:
        if ab.trigger == Trigger.AURA and isinstance(ab.effect, ZappTargeting):
            return True
    return False


def _attacker_has_cleave(attacker: BattleMinion) -> bool:
    for ab in attacker.template.abilities:
        if ab.trigger == Trigger.AURA and isinstance(ab.effect, CleaveOnAttack):
            return True
    return False


def _cleave_victim_ids_at_swing_start(
    defender_side: BattleSide, primary: BattleMinion
) -> List[int]:
    try:
        idx = defender_side.minions.index(primary)
    except ValueError:
        return []
    out: List[int] = []
    if idx > 0:
        left = defender_side.minions[idx - 1]
        if left.alive:
            out.append(left.instance_id)
    if idx + 1 < len(defender_side.minions):
        right = defender_side.minions[idx + 1]
        if right.alive:
            out.append(right.instance_id)
    return out


def _pick_target(
    defender_side: BattleSide,
    rng: np.random.Generator,
    attacker: Optional[BattleMinion] = None,
    battle_field: Optional[Tuple[BattleSide, BattleSide]] = None,
) -> Optional[BattleMinion]:
    alive = defender_side.alive_minions()
    if not alive:
        return None
    taunts = [m for m in alive if Keyword.TAUNT in m.template.all_keywords]
    pool = taunts if taunts else alive
    if attacker is not None and _attacker_has_zapp_targeting(attacker):
        atk_vals = [
            attack_value(m, defender_side, death_resolution=False, battle_field=battle_field)
            for m in pool
        ]
        mna = min(atk_vals)
        pool = [m for m, av in zip(pool, atk_vals) if av == mna]
    idx = int(rng.integers(0, len(pool)))
    return pool[idx]


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
    for ev in reversed(trailing):
        rt.queue.appendleft(ev)
    _sync_health_all(rt)


def _handle_attack_completed(rt: _CombatRuntime) -> None:
    pending: List[Tuple[int, BattleMinion]] = []
    for sidx in (0, 1):
        side = rt.side(sidx)
        for m in side.minions:
            if (not m.alive) and (not m.deathrattle_fired):
                pending.append((sidx, m))
    for sidx, m in pending:
        rt.queue.append(MinionDied(sidx, m.instance_id))


def _summon_append(
    rt: _CombatRuntime,
    side_idx: int,
    template: Minion,
) -> Optional[BattleMinion]:
    side = rt.side(side_idx)
    if side.alive_count() >= COMBAT_BOARD_MAX:
        return None
    bid = rt.alloc_id()
    bm = BattleMinion.from_minion(copy(template), bid)
    side.minions.append(bm)
    rt.queue.append(MinionSummoned(side_idx, bid, template.card_id))
    _sync_health_all(rt)
    return bm


def _summon_target_side(dead_side_idx: int, for_opponent: bool) -> int:
    return (1 - dead_side_idx) if for_opponent else dead_side_idx


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
    if vic.shield_armed and Keyword.SHIELD in vic.template.all_keywords:
        vic.shield_armed = False
    else:
        vic.current_health -= amount
        if vic.current_health <= 0:
            vic.current_health = 0
    _sync_health_all(rt)
    if not vic.alive:
        rt.queue.append(MinionDied(enemy_side, vic.instance_id))


def _fire_self_damaged(rt: _CombatRuntime, side_idx: int, bm: BattleMinion) -> None:
    if not bm.alive:
        return
    for ab in bm.template.abilities:
        if ab.trigger != Trigger.ON_SELF_DAMAGED:
            continue
        eff = ab.effect
        if isinstance(eff, SummonOnSelfDamaged):
            n_sum = _summon_multiplier(rt.side(side_idx))
            for _ in range(max(0, eff.count)):
                for __ in range(n_sum):
                    tok = make_minion(eff.token_id)
                    if _summon_append(rt, side_idx, tok) is None:
                        return


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
                    listener.template.keywords = frozenset(
                        listener.template.keywords | {eff.keyword}
                    )
                    if eff.keyword == Keyword.SHIELD:
                        listener.shield_armed = True
            elif isinstance(eff, BuffListenerIfSummonedMatches):
                if _matches_tribe_for_aura(summoned.template, eff.tribe):
                    listener.template.bonus_attack += eff.attack
                    listener.template.bonus_health += eff.health
                    listener.current_health += eff.health
    _sync_health_all(rt)


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
            eff = ab.effect
            if isinstance(eff, BuffSelf):
                listener.template.bonus_attack += eff.attack
                listener.template.bonus_health += eff.health
                listener.current_health += eff.health
            elif isinstance(eff, DealDamageRandomEnemyMinion):
                _deal_random_enemy_minion_damage(rt, side_idx, eff.amount)
    _sync_health_all(rt)


def _handle_shield_lost(rt: _CombatRuntime, e: ShieldLost) -> None:
    bm = rt.find_minion(e.victim_side_idx, e.victim_instance_id)
    if bm is not None:
        _fire_self_damaged(rt, e.victim_side_idx, bm)


def _handle_damage_dealt(rt: _CombatRuntime, e: DamageDealt) -> None:
    bm = rt.find_minion(e.victim_side_idx, e.victim_instance_id)
    if bm is not None and bm.alive and e.hp_loss > 0:
        _fire_self_damaged(rt, e.victim_side_idx, bm)


def _fire_deathrattle(rt: _CombatRuntime, dead: BattleMinion, side_idx: int) -> None:
    side = rt.side(side_idx)
    prev = rt.in_death_resolution
    rt.in_death_resolution = True
    try:
        for ab in dead.template.abilities:
            if ab.trigger != Trigger.ON_DEATH:
                continue
            effect = ab.effect
            if isinstance(effect, SummonEffect):
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
                wave_cap = max(1, getattr(effect, "dr_wave_count", 1))
                rep = 0
                while rep < _deathrattle_multiplier(rt.side(side_idx)):
                    rep += 1
                    n_sum = _summon_multiplier(rt.side(side_idx))
                    for _ in range(n_sum):
                        for _wave in range(wave_cap):
                            for __ in range(base):
                                tok = make_minion(effect.token_id)
                                if _summon_append(rt, target_side, tok) is None:
                                    break
            elif isinstance(effect, SummonRandomMinionEffect):
                race_hs = hs_race_string(effect.race_filter)
                pool = build_summon_pool(
                    effect.exact_cost,
                    effect.legendary_only,
                    effect.require_deathrattle,
                    race_hs,
                    dead.template.card_id if effect.exclude_source else None,
                )
                if not pool:
                    continue
                target_side = _summon_target_side(side_idx, effect.for_opponent)
                rep = 0
                while rep < _deathrattle_multiplier(rt.side(side_idx)):
                    rep += 1
                    n_sum = _summon_multiplier(rt.side(side_idx))
                    for _ in range(n_sum):
                        for __ in range(effect.count):
                            row = pool[int(rt.rng.integers(0, len(pool)))]
                            tok = minion_from_hsjson_card(row)
                            if _summon_append(rt, target_side, tok) is None:
                                break
            elif isinstance(effect, DealDamageRandomEnemyMinion):
                rep_dr = 0
                while rep_dr < _deathrattle_multiplier(side):
                    rep_dr += 1
                    _deal_random_enemy_minion_damage(rt, side_idx, effect.amount)
            elif isinstance(effect, BuffAllFriendlyMinions):
                rep_dr = 0
                while rep_dr < _deathrattle_multiplier(side):
                    rep_dr += 1
                    for m in side.minions:
                        if (not m.alive) or m is dead:
                            continue
                        m.template.bonus_attack += effect.attack
                        m.template.bonus_health += effect.health
                        m.current_health += effect.health
                _sync_health_all(rt)
            elif isinstance(effect, BuffAllFriendlyOfTribe):
                rep_dr = 0
                while rep_dr < _deathrattle_multiplier(side):
                    rep_dr += 1
                    for m in side.minions:
                        if (not m.alive) or m is dead:
                            continue
                        if not _matches_tribe_for_aura(m.template, effect.tribe):
                            continue
                        m.template.bonus_attack += effect.attack
                        m.template.bonus_health += effect.health
                        m.current_health += effect.health
                _sync_health_all(rt)
            elif isinstance(effect, BuffAllWithKeyword):
                rep_dr = 0
                while rep_dr < _deathrattle_multiplier(side):
                    rep_dr += 1
                    for m in side.minions:
                        if (not m.alive) or m is dead:
                            continue
                        if effect.keyword not in m.template.all_keywords:
                            continue
                        m.template.bonus_attack += effect.attack
                        m.template.bonus_health += effect.health
                        m.current_health += effect.health
                _sync_health_all(rt)
            elif isinstance(effect, BuffRandomOtherFriendlyCombat):
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
            elif isinstance(effect, GrantKeywordRandomFriendly):
                rep_dr = 0
                while rep_dr < _deathrattle_multiplier(side):
                    rep_dr += 1
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
                    t.template.keywords = frozenset(
                        t.template.keywords | {effect.keyword}
                    )
                    if effect.keyword == Keyword.SHIELD:
                        t.shield_armed = True
                    if effect.keyword == Keyword.POISONOUS:
                        pass
    finally:
        rt.in_death_resolution = prev


def _fire_kangor_listeners(
    rt: _CombatRuntime,
    dead: BattleMinion,
    dead_side_idx: int,
) -> None:
    if not _is_mech_template(dead.template):
        return
    side = rt.side(dead_side_idx)
    prev = rt.in_death_resolution
    rt.in_death_resolution = True
    try:
        for listener in list(side.minions):
            if listener is dead or not listener.alive:
                continue
            for ab in listener.template.abilities:
                if ab.trigger != Trigger.ON_FRIENDLY_MECH_DIED:
                    continue
                if not isinstance(ab.effect, KangorSummonCopy):
                    continue
                tpl = copy(dead.template)
                copy_i = 0
                while copy_i < _summon_multiplier(rt.side(dead_side_idx)):
                    copy_i += 1
                    if _summon_append(rt, dead_side_idx, copy(tpl)) is None:
                        break
    finally:
        rt.in_death_resolution = prev


def _handle_overkill(rt: _CombatRuntime, e: Overkill) -> None:
    att = rt.find_minion(e.attacker_side_idx, e.attacker_instance_id)
    if att is None or not att.alive or e.excess_damage <= 0:
        return
    for ab in att.template.abilities:
        if ab.trigger != Trigger.ON_OVERKILL:
            continue
        eff = ab.effect
        if not isinstance(eff, SummonEffect):
            continue
        if eff.for_opponent or eff.count_from_source_attack:
            continue
        side = rt.side(e.attacker_side_idx)
        n_sum = _summon_multiplier(side)
        for _ in range(max(0, eff.count)):
            for __ in range(n_sum):
                tok = make_minion(eff.token_id)
                if _summon_append(rt, e.attacker_side_idx, tok) is None:
                    return


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
    _fire_deathrattle(rt, bm, e.side_idx)
    _fire_kangor_listeners(rt, bm, e.side_idx)
    _sync_health_all(rt)


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
        _handle_attack_completed(rt)
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
    bf = (rt.side(0), rt.side(1))
    a_dmg = attack_value(attacker, atk_side, death_resolution=False, battle_field=bf)
    d_dmg = attack_value(target, def_side, death_resolution=False, battle_field=bf)

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
    rt.queue.append(AttackCompleted())
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
    n_swings = 2 if Keyword.WINDFURY in attacker.template.all_keywords else 1
    defender_side = rt.side(defender_side_idx)
    for _ in range(n_swings):
        if not attacker.alive or not defender_side.has_alive():
            break
        tgt = _pick_target(
            defender_side,
            rt.rng,
            attacker,
            battle_field=(rt.side(0), rt.side(1)),
        )
        if tgt is None:
            break
        _run_single_swing(rt, attacker, tgt, attacker_side_idx, defender_side_idx)


def _next_attacker(side: BattleSide) -> Optional[BattleMinion]:
    n = len(side.minions)
    if n == 0:
        return None
    start = side.cursor % n
    for offset in range(n):
        idx = (start + offset) % n
        if side.minions[idx].alive:
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


def _winner_damage(side: BattleSide, winner_tavern_tier: int) -> int:
    raw = int(winner_tavern_tier) + sum(m.tier for m in side.minions if m.alive)
    return min(DAMAGE_CAP, raw)


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
        p0_survivors_out.extend(m.template.card_id for m in side0.minions if m.alive)
    if p1_survivors_out is not None:
        p1_survivors_out.clear()
        p1_survivors_out.extend(m.template.card_id for m in side1.minions if m.alive)
    if p0_board_out is not None:
        p0_board_out.clear()
        p0_board_out.extend(persist_shop_board_from_side(side0, max_board_slots))
    if p1_board_out is not None:
        p1_board_out.clear()
        p1_board_out.extend(persist_shop_board_from_side(side1, max_board_slots))


def simulate_battle(
    p0_board: List[Minion],
    p1_board: List[Minion],
    p0_has_initiative: bool,
    rng: np.random.Generator,
    max_attacks: int = 200,
    death_log: Optional[List[Tuple[int, str]]] = None,
    mech_death_log: Optional[List[Tuple[int, Minion]]] = None,
    p0_survivors_out: Optional[List[str]] = None,
    p1_survivors_out: Optional[List[str]] = None,
    p0_board_out: Optional[List[Minion]] = None,
    p1_board_out: Optional[List[Minion]] = None,
    max_board_slots: int = BOARD_SIZE,
    *,
    p0_tavern_tier: int = 1,
    p1_tavern_tier: int = 1,
) -> Tuple[int, int]:
    rt = _CombatRuntime(
        sides=(BattleSide(), BattleSide()),
        rng=rng,
        death_hook=(lambda si, cid: death_log.append((si, cid))) if death_log is not None else None,
        mech_hook=(lambda si, tpl: mech_death_log.append((si, tpl))) if mech_death_log is not None else None,
    )
    if death_log is not None:
        death_log.clear()
    if mech_death_log is not None:
        mech_death_log.clear()

    rt.sides = (_build_side(p0_board, rt), _build_side(p1_board, rt))
    side0, side1 = rt.sides
    _sync_health_all(rt)

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
        return 0, 0
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
        return _winner_damage(side1, p1_tavern_tier), 0
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
        return 0, _winner_damage(side0, p0_tavern_tier)

    attacker_idx = _decide_first_side(side0, side1, p0_has_initiative)
    sides = (side0, side1)

    attacks = 0
    while side0.has_alive() and side1.has_alive() and attacks < max_attacks:
        attacker_side = sides[attacker_idx]
        defender_side = sides[1 - attacker_idx]

        attacker = _next_attacker(attacker_side)
        if attacker is None:
            break

        if not defender_side.has_alive():
            break

        _run_attacker_activation(rt, attacker, attacker_idx, 1 - attacker_idx)

        attacker_idx = 1 - attacker_idx
        attacks += 1

    p0_alive = side0.has_alive()
    p1_alive = side1.has_alive()
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
        return 0, _winner_damage(side0, p0_tavern_tier)
    if p1_alive and not p0_alive:
        return _winner_damage(side1, p1_tavern_tier), 0
    return 0, 0


__all__ = [
    "BattleMinion",
    "BattleSide",
    "persist_shop_board_from_side",
    "BattleEvent",
    "BeginAttackExchange",
    "ShieldLost",
    "DamageDealt",
    "Overkill",
    "AttackCompleted",
    "MinionDied",
    "MinionSummoned",
    "DamageStrike",
    "COMBAT_BOARD_MAX",
    "attack_with_auras",
    "attack_value",
    "health_aura_bonus",
    "build_battle_side",
    "simulate_battle",
]

from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Deque, List, Optional, Tuple, Union

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_core.effects import (
    AdjacentStatAura,
    AttackBonusPerOtherMurlocGlobal,
    AttackImmediatelyAfterSurvivingEffect,
    BuffAllFriendlyMinions,
    BuffAllFriendlyOfTribe,
    BuffAllOtherOfTribe,
    BuffAllWithKeyword,
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
from src.bg_core.minion import Minion, Race
from src.envs.minibg.summon_pool import hs_race_string, summon_pool_for


@dataclass
class BattleMinion:
    template: Minion
    current_health: int
    shield_armed: bool
    deathrattle_fired: bool = False
    reborn_consumed: bool = False
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
    # Flat Attack added to every minion on this side (Deathwing's global +Attack
    # aura; set equal on both sides since it buffs all minions in the combat).
    attack_aura_all: int = 0
    # Keywords granted to this side's left-most minion at Start of Combat (Al'Akir).
    start_combat_keywords: frozenset = field(default_factory=frozenset)

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

    attacker_side_idx: int
    attacker_instance_id: int


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


def _mark_health_aura_dirty(rt: "_CombatRuntime", *side_indices: int) -> None:
    for side_idx in side_indices:
        rt.health_aura_dirty[side_idx] = True


def _grant_keyword(
    rt: "_CombatRuntime",
    side_idx: int,
    minion: BattleMinion,
    keyword: Keyword,
) -> None:
    if keyword not in minion.template.keywords:
        minion.template.keywords = frozenset(minion.template.keywords | {keyword})
        _mark_health_aura_dirty(rt, side_idx)
    if keyword == Keyword.SHIELD:
        minion.shield_armed = True


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
    return minion.raw_attack + bonus + side.attack_aura_all + _self_aura_attack_bonus(
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
    if rt.health_aura_dr_snapshot != dr:
        _mark_health_aura_dirty(rt, 0, 1)
        rt.health_aura_dr_snapshot = dr
    for side_idx in (0, 1):
        if not rt.health_aura_dirty[side_idx]:
            continue
        _sync_health_aura_side(rt.side(side_idx), dr)
        rt.health_aura_dirty[side_idx] = False


def attack_with_auras(minion: BattleMinion, side: BattleSide) -> int:
    """Attack during the combat strike phase (auras from living neighbors apply)."""
    return attack_value(minion, side, death_resolution=False)


@dataclass
class _CombatRuntime:
    sides: Tuple[BattleSide, BattleSide]
    rng: np.random.Generator
    combat_board_max: int
    damage_cap: int
    patch: PatchContext
    queue: Deque[BattleEvent] = field(default_factory=deque)
    next_id: int = 1
    in_death_resolution: bool = False
    death_hook: Optional[Callable[[int, str], None]] = None
    mech_hook: Optional[Callable[[int, Minion], None]] = None
    swing_damage_survivors: List[Tuple[int, int]] = field(default_factory=list)
    bonus_attack_depth: int = 0
    combat_gold: List[int] = field(default_factory=lambda: [0, 0])
    combat_hand_adds: List[List[str]] = field(default_factory=lambda: [[], []])
    kill_attribution: dict[Tuple[int, int], Tuple[int, int]] = field(
        default_factory=dict
    )
    attacker_killed_this_swing: bool = False
    health_aura_dirty: List[bool] = field(default_factory=lambda: [True, True])
    health_aura_dr_snapshot: Optional[bool] = None

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


def build_battle_side(board: List[Minion], *, patch: PatchContext) -> BattleSide:
    """Build a battle line with fresh instance IDs (tests, tooling)."""
    ctx = require_patch(patch, where="battle.build_battle_side")
    rt = _CombatRuntime(
        sides=(BattleSide(), BattleSide()),
        rng=np.random.default_rng(0),
        combat_board_max=10**9,
        damage_cap=10**9,
        patch=ctx,
    )
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
    pending: List[Tuple[int, BattleMinion]] = []
    for sidx in (0, 1):
        side = rt.side(sidx)
        for m in side.minions:
            if (not m.alive) and (not m.deathrattle_fired):
                pending.append((sidx, m))
    for sidx, m in pending:
        rt.queue.append(MinionDied(sidx, m.instance_id))


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
        side.minions.insert(at_idx, bm)
        if at_idx <= side.cursor:
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
    """List index right after ``anchor``. Dead bodies stay in the minion list,
    so a dead source is a valid anchor; ``None`` anchor → append at the end."""
    if anchor is None:
        return None
    idx = _board_index(side, anchor)
    return None if idx is None else idx + 1


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
    try:
        idx = side.minions.index(dead)
    except ValueError:
        return
    for j in (idx - 1, idx + 1):
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
        rt.queue.append(MinionDied(side_idx, bm.instance_id))
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
    try:
        vi = side.minions.index(vic)
    except ValueError:
        return
    adj: List[BattleMinion] = []
    for j in (vi - 1, vi + 1):
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
            elif isinstance(eff, BuffAllFriendlyMinions):
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
            elif isinstance(effect, SummonRandomMinionEffect):
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
                    continue
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
            elif isinstance(effect, DealDamageRandomEnemyMinion):
                rep_dr = 0
                while rep_dr < _deathrattle_multiplier(side):
                    rep_dr += 1
                    _deal_random_enemy_minion_damage(rt, side_idx, effect.amount)
            elif isinstance(effect, DealDamageLeftmostEnemyMinion):
                rep_dr = 0
                while rep_dr < _deathrattle_multiplier(side):
                    rep_dr += 1
                    _deal_leftmost_enemy_minion_damage(rt, side_idx, effect.amount)
            elif isinstance(effect, DealDamageAllMinions):
                rep_dr = 0
                while rep_dr < _deathrattle_multiplier(side):
                    rep_dr += 1
                    _deal_damage_all_minions(rt, effect.amount)
            elif isinstance(effect, TransferAttackToRandomFriendlyEffect):
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
            elif isinstance(effect, SummonRandomAndCopyToHandEffect):
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
                    continue
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
            elif isinstance(effect, SummonFirstDeadFriendlyMechsThisCombat):
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
            elif isinstance(effect, GrantKeywordRandomFriendly):
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
            elif isinstance(effect, GrantKeywordAllFriendlyOfTribe):
                rep_dr = 0
                while rep_dr < _deathrattle_multiplier(side):
                    rep_dr += 1
                    for m in side.minions:
                        if (not m.alive) or m is dead:
                            continue
                        if not _matches_tribe_for_aura(m.template, effect.tribe):
                            continue
                        _grant_keyword(rt, side_idx, m, effect.keyword)
            elif isinstance(effect, GainGoldOnDeathEffect):
                rep_dr = 0
                while rep_dr < _deathrattle_multiplier(side):
                    rep_dr += 1
                    rt.combat_gold[side_idx] += effect.amount
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
        elif isinstance(eff, BuffAllOtherOfTribe):
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
    bm.reborn_consumed = True
    _strip_reborn_keyword(bm)
    bm.current_health = 1
    _mark_health_aura_dirty(rt, side_idx)


def _count_friendlies_of_tribe(side: BattleSide, tribe: Any) -> int:
    return sum(
        1 for m in side.minions if m.alive and _matches_tribe_for_aura(m.template, tribe)
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


def _winner_damage_raw(side: BattleSide, winner_tavern_tier: int) -> int:
    """Uncapped winner damage formula INCLUDING tavern tier (used for HP)."""
    return int(winner_tavern_tier) + sum(m.tier for m in side.minions if m.alive)


def _winner_damage_board_only(side: BattleSide) -> int:
    """Board-only damage component: ``Σ alive_minion_tiers``, no tavern tier.

    Used as the auxiliary battle-prediction head's regression target. The
    tavern-tier contribution is excluded on purpose: it's deterministic from
    the pre-combat state (both seats' tavern tiers are known scalars) and
    would let the head trivially memorize it from ``state_emb`` if exposed.
    Keeping the target board-derived makes the prediction depend only on the
    minions that actually fought.
    """
    return sum(m.tier for m in side.minions if m.alive)


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
    p0_attack_aura_all: int = 0,
    p1_attack_aura_all: int = 0,
    p0_start_combat_keywords: frozenset = frozenset(),
    p1_start_combat_keywords: frozenset = frozenset(),
) -> "BattleResult":
    # Snapshot the input boards (deep-ish copy by tuple) BEFORE any combat
    # mutation. This is the initial (step=0) snapshot fed to the battle
    # prediction head. ``p0_board`` / ``p1_board`` are passed by reference and
    # ``_build_side`` constructs ``BattleSide`` objects from them — the original
    # Minion templates remain unchanged, so a tuple suffices.
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
    "attack_with_auras",
    "attack_value",
    "health_aura_bonus",
    "build_battle_side",
    "simulate_battle",
]

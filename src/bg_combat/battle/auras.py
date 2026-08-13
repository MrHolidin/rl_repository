"""Aura / stat system: attack & health bonuses, keyword grants, health sync."""
from __future__ import annotations

from typing import Any, Optional, Tuple

from src.bg_core.effects import (
    AdjacentStatAura,
    AttackBonusPerOtherMurlocGlobal,
    AttackImmediatelyAfterSurvivingEffect,
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

from .state import BattleMinion, BattleSide, _CombatRuntime


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

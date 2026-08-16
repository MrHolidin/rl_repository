"""Aura / stat system: attack & health bonuses, keyword grants, health sync."""
from __future__ import annotations

from typing import Any, Optional, Tuple

from src.bg_core.effects import (
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
    Trigger,
    ZappTargeting,
)
from src.bg_core.board_helpers import buff_matching_hits
from src.bg_core.minion import Minion, Race

from .state import BattleMinion, BattleSide, _CombatRuntime


def _deathrattle_multiplier(side: BattleSide) -> int:
    """Product of Baron-style auras on living minions (re-read at DR execution time)."""
    p = 1
    for bm in side.minions:
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
    if not isinstance(eff, StatAura):
        return 0, 0
    if not buff_matching_hits(
        eff, recipient.template, idx_candidate=idx_r, idx_source=idx_s
    ):
        return 0, 0
    return eff.attack, eff.health


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
                for m in s.iter_living():
                    if m is not minion:
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
    for bm in side.iter_living():
        b = health_aura_bonus(bm, side, death_resolution=death_resolution)
        prev = bm.health_aura_snapshot
        delta = b - prev
        bm.health_aura_snapshot = b
        bm.current_health += delta
        emax = bm.template.max_health + b
        if bm.current_health > emax:
            bm.current_health = emax
        if bm.current_health < 0:
            # Losing a health aura can be lethal (Mal'Ganis dies, a damaged
            # Demon goes with it) -- that is the rule. Stopping at 0 is what
            # damage already does; letting it run negative left the body in a
            # state no other death path produces.
            bm.current_health = 0


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
        # An aura that shrank may have just killed someone with no damage
        # dealt, so this is a death site too. Not while a death is being
        # resolved: the board is mid-flight there (a deathrattle is placing
        # tokens around the body it just left), and sweeping again would pull
        # bodies out from under it.
        if not dr:
            rt.side(side_idx).reap_dead()


def attack_with_auras(minion: BattleMinion, side: BattleSide) -> int:
    """Attack during the combat strike phase (auras from living neighbors apply)."""
    return attack_value(minion, side, death_resolution=False)

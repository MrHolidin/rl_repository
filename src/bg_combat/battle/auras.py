"""Aura / stat system: attack & health bonuses, keyword grants, health sync."""
from __future__ import annotations

from typing import Optional, Tuple

from src.bg_core.effects import (
    AttackBonusPerOtherMurlocGlobal,
    Keyword,
    MultiplierKind,
    Trigger,
)
from src.bg_core.board_helpers import multiplier_for, stat_aura_bonus
from src.bg_core.minion import Race

from .state import BattleMinion, BattleSide, _CombatRuntime


def _deathrattle_multiplier(side: BattleSide) -> int:
    """Product of Baron-style auras (re-read at DR execution time)."""
    return multiplier_for(
        (bm for bm in side.minions), MultiplierKind.DEATHRATTLE
    )


def _summon_multiplier(side: BattleSide) -> int:
    """Product of Khadgar-style auras."""
    return multiplier_for((bm for bm in side.minions), MultiplierKind.SUMMON)


def _mark_health_aura_dirty(rt: "_CombatRuntime", *side_indices: int) -> None:
    for side_idx in side_indices:
        rt.health_aura_dirty[side_idx] = True


def _grant_keyword(
    rt: "_CombatRuntime",
    side_idx: int,
    minion: BattleMinion,
    keyword: Keyword,
) -> None:
    if keyword not in minion.keywords:
        minion.keywords = frozenset(minion.keywords | {keyword})
        _mark_health_aura_dirty(rt, side_idx)
    if keyword == Keyword.SHIELD:
        minion.has_shield = True


def _self_aura_attack_bonus(
    minion: BattleMinion,
    battle_field: Optional[Tuple[BattleSide, BattleSide]],
    own_side: BattleSide,
) -> int:
    sides: Tuple[BattleSide, ...] = (
        battle_field if battle_field is not None else (own_side,)
    )
    bonus = 0
    for ab in minion.abilities:
        if ab.trigger != Trigger.AURA:
            continue
        eff = ab.effect
        if isinstance(eff, AttackBonusPerOtherMurlocGlobal):
            n = 0
            for s in sides:
                for m in s.iter_living():
                    if m is not minion:
                        if m.race in (Race.MURLOC, Race.ALL):
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
        return minion.raw_attack
    bonus, _ = stat_aura_bonus(side.minions, minion, live_only=True)
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
    _, bonus = stat_aura_bonus(side.minions, minion, live_only=True)
    return bonus


def _sync_health_aura_side(side: BattleSide, death_resolution: bool) -> None:
    for bm in side.iter_living():
        # Store the contribution, not a delta to patch into an absolute. A
        # shrinking aura lowers the derived health on its own -- lethally if
        # the minion was damaged (Mal'Ganis dies, a hurt Demon goes with it),
        # which is the rule -- so there is nothing to add, clamp or remember.
        bm.aura_health = health_aura_bonus(
            bm, side, death_resolution=death_resolution
        )


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

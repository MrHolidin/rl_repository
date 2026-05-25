"""Forged triple (golden) ability resolution — not always ``EFFECT[normal]`` ×3 concatenation."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from src.bg_core.effects import (
    Ability,
    BattlecryMultiplierAura,
    BuffSelfFromHeroDamageTaken,
    DealDamageRandomEnemyMinion,
    DealHeroDamage,
    DeathrattleMultiplierAura,
    Effect,
    MultiplySelfAttackEffect,
    StartOfCombatDamagePerFriendlyTribe,
    SummonMultiplierAura,
    ZappTargeting,
)
from src.bg_catalog.golden_catalog import golden_hints_for_card
from src.bg_catalog.patch_catalog import golden_upgrade_card_id

_GOLDEN_INT_FIELDS = frozenset(
    {
        "attack",
        "health",
        "amount",
        "count",
        "repeats",
        "attack_per",
        "health_per",
        "attack_each",
        "health_each",
        "amount_per_match",
        "gold_reward",
        "stat_multiplier",
        "per_attack",
        "dr_wave_count",
        "health_per_damage",
        "factor",
        "uses",
    }
)

_NO_GENERIC_SCALE: Tuple[Type[Effect], ...] = (
    ZappTargeting,
)


def _scale_factor(effect: Effect, value: int, hints: Dict[str, Any]) -> int:
    if isinstance(
        effect, (BattlecryMultiplierAura, DeathrattleMultiplierAura, SummonMultiplierAura)
    ):
        return 3 if value == 2 else value
    if isinstance(effect, MultiplySelfAttackEffect):
        if hints.get("triple_factor") and value == 2:
            return 3
        return value * 2 if value > 0 else value
    return value * 2 if value > 0 else value


def _should_skip_field(
    effect: Effect, field_name: str, value: int, hints: Dict[str, Any]
) -> bool:
    if field_name == "amount" and isinstance(effect, DealHeroDamage):
        return bool(hints.get("preserve_hero_damage_amount"))
    if hints.get("prefer_repeats"):
        if field_name == "amount" and isinstance(effect, DealDamageRandomEnemyMinion):
            return True
        if field_name == "repeats" and isinstance(effect, DealDamageRandomEnemyMinion):
            return True
        if field_name == "amount_per_match" and isinstance(
            effect, StartOfCombatDamagePerFriendlyTribe
        ):
            return True
    return False


def implicit_triple_golden_effect(
    e: Effect, hints: Optional[Dict[str, Any]] = None
) -> Effect:
    """Scale effect numerics for a forged golden when no authored ``TB_BaconUps_*`` row exists."""
    hints = hints or {}
    if type(e) in _NO_GENERIC_SCALE:
        return e

    updates: Dict[str, Any] = {}
    for f in fields(e):
        if f.name not in _GOLDEN_INT_FIELDS:
            continue
        val = getattr(e, f.name)
        if not isinstance(val, int) or val <= 0:
            continue
        if _should_skip_field(e, f.name, val, hints):
            continue
        if f.name == "factor":
            updates[f.name] = _scale_factor(e, val, hints)
        else:
            updates[f.name] = val * 2

    scaled = replace(e, **updates) if updates else e
    return _apply_prefer_repeats(scaled, hints)


def _apply_prefer_repeats(e: Effect, hints: Dict[str, Any]) -> Effect:
    if not hints.get("prefer_repeats"):
        return e
    if isinstance(e, DealDamageRandomEnemyMinion):
        return replace(e, repeats=max(1, e.repeats) * 2)
    return e


def resolve_triple_forged_abilities(
    normal_card_id: str,
    effects_table: Mapping[str, Tuple[Ability, ...]],
    *,
    catalog_path: Optional[Path] = None,
) -> Tuple[Ability, ...]:
    """Prefer HS golden ``TB_BaconUps_*`` row where authored; else catalog-aware implicit scale."""
    gid = golden_upgrade_card_id(normal_card_id, catalog_path)
    if gid is None:
        return tuple(effects_table.get(normal_card_id, ()))

    spe = effects_table.get(gid)
    if spe:
        return tuple(spe)

    cat_key = catalog_path
    hints = golden_hints_for_card(normal_card_id, catalog_path)
    base = effects_table.get(normal_card_id, ())
    return tuple(
        replace(ab, effect=implicit_triple_golden_effect(ab.effect, hints)) for ab in base
    )


__all__ = [
    "implicit_triple_golden_effect",
    "resolve_triple_forged_abilities",
]

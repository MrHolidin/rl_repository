"""Forged triple (golden) ability resolution — not always ``EFFECT[normal]`` ×3 concatenation."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from src.bg_core.effects import (
    Ability,
    AvengeEffect,
    BuffSelfFromHeroDamageTaken,
    DealDamageRandomEnemyMinion,
    DealHeroDamage,
    Effect,
    Multiplier,
    MultiplySelfAttackEffect,
    StartOfCombatDamagePerFriendlyTribe,
    SummonEffect,
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
        # Added with the golden printings that move them: a Lockbox opening
        # two turns sooner, a hand-stat grab taken twice, a keep-what-you-
        # gained that keeps double.
        "sooner",
        "times",
        "limit",
        "set_attack",
        "set_health",
        "attack_outside_combat",
        "health_outside_combat",
    }
)

_NO_GENERIC_SCALE: Tuple[Type[Effect], ...] = (
    ZappTargeting,
)


def _scale_factor(effect: Effect, value: int, hints: Dict[str, Any]) -> int:
    # "double it" → "triple it" is the one place a golden does not double the
    # number it prints, and the golden text says so. Read from the hint rather
    # than from the effect's type: Banana Slamma multiplies a summoned minion's
    # Attack and reads exactly the same way Rivendare's multiplier does.
    if hints.get("triple_factor") and value == 2:
        return 3
    if isinstance(effect, Multiplier):
        return 3 if value == 2 else value
    return value * 2 if value > 0 else value


def _scales_at_the_wrapper(hints: Dict[str, Any], effect: Effect) -> bool:
    """Whether the golden's "twice" lands here rather than on what this wraps.

    "give two friendly Pirates +4/+5 **twice**" keeps the payload's numbers and
    changes how often it resolves. It lands here when this effect is the one
    that counts resolutions — and when it is not, the recursion carries it
    inward: Glambot's "Magnetize a Satellite to it twice" is a repeat of the
    Magnetize, and the watcher wrapping it has nothing to double.
    """
    return bool(hints.get("prefer_repeats")) and _has_repeats(effect)


def _has_repeats(effect: Effect) -> bool:
    return _has_field(effect, "repeats")


def _has_field(effect: Effect, name: str) -> bool:
    return any(f.name == name for f in fields(effect))


def _scales_something_else(effect: Effect) -> bool:
    """Whether this effect has a number besides ``repeats`` for a golden to double."""
    return any(
        f.name in _GOLDEN_INT_FIELDS
        and f.name != "repeats"
        and isinstance(getattr(effect, f.name, None), int)
        and getattr(effect, f.name) > 0
        for f in fields(effect)
    )


def _is_effect(value: Any) -> bool:
    """A nested effect, as opposed to an int, a tribe, or a Minion reference."""
    return (
        is_dataclass(value)
        and not isinstance(value, type)
        and type(value).__module__ == "src.bg_core.effects"
    )


def _should_skip_field(
    effect: Effect, field_name: str, value: int, hints: Dict[str, Any]
) -> bool:
    if hints.get("prefer_repeats"):
        if field_name == "amount" and isinstance(effect, DealDamageRandomEnemyMinion):
            return True
        if field_name == "repeats" and isinstance(effect, DealDamageRandomEnemyMinion):
            # Doubled by ``_apply_prefer_repeats`` instead, so not here too.
            return True
        if field_name == "amount_per_match" and isinstance(
            effect, StartOfCombatDamagePerFriendlyTribe
        ):
            return True
        if _has_repeats(effect):
            # "give another random friendly Elemental +1/+1 **twice**": two
            # payouts of what is printed, not one payout of double.
            return field_name != "repeats"
    if field_name == "repeats" and _scales_something_else(effect):
        # A golden that says nothing about repeating doubles the numbers, not
        # how often they land. Houndmaster's golden is +4/+4 once, and doubling
        # both would be four times the card.
        return True
    if field_name == "count" and (
        int(getattr(effect, "set_attack", 0) or 0) > 0
        or int(getattr(effect, "set_health", 0) or 0) > 0
    ):
        # "Summon **a** random Beast. Set its stats to 6/6" — the Golden sets
        # them to 12/12 and still summons one.
        return True
    if field_name == "limit":
        # A cap on how many bodies an effect reaches. It moves only when the
        # Golden says so in words ("give **two** other friendly Dragons"), the
        # way `repeats` does — otherwise a Golden that just pays more would
        # quietly reach further as well.
        return not hints.get("more_targets")
    if hints.get("more_targets") and _has_field(effect, "limit"):
        # The Golden reaches one more body and pays each of them what the plain
        # printing pays: the count moves, the numbers do not.
        return field_name != "limit"
    if hints.get("repeats_only"):
        # Reached by recursion under "twice": the wrapper had no resolutions to
        # count, so the repeat belongs to what it wraps and nothing else there
        # moves.
        return field_name != "repeats"
    if hints.get("double_stats"):
        # "consume a minion to gain **double** its stats", "give it **double**
        # this minion's maximum stats": the numbers printed stay, and the
        # multiple is the only thing that moves.
        return field_name not in ("stat_multiplier", "factor")
    if field_name == "count" and isinstance(effect, AvengeEffect):
        # "Avenge (3)" is a countdown, not a payout: the Golden pays more per
        # trigger and still triggers every three deaths.
        return True
    if field_name == "count" and hints.get("golden_token"):
        # A golden that summons the Golden token summons the same number of
        # them, not twice as many plain ones.
        return isinstance(effect, SummonEffect)
    if field_name == "dr_wave_count":
        # The wave is Rat Pack's lever and only Rat Pack's: its count comes
        # from its Attack, so doubling the count is not available and the
        # golden summons the same wave twice instead. On a summon that prints
        # a fixed count, doubling *both* is doubling twice — a golden "summon
        # two Beetles" that puts four on the board.
        return not getattr(effect, "count_from_source_attack", False)
    if field_name == "amount" and isinstance(effect, DealHeroDamage):
        return bool(hints.get("preserve_hero_damage_amount"))
    return False


def implicit_triple_golden_effect(
    e: Effect,
    hints: Optional[Dict[str, Any]] = None,
    catalog_path: Optional[Path] = None,
) -> Effect:
    """Scale effect numerics for a forged golden when no authored ``TB_BaconUps_*`` row exists."""
    hints = hints or {}
    if type(e) in _NO_GENERIC_SCALE:
        return e

    # Decided before the loop, because the nested field can be visited before
    # ``repeats`` is: an effect whose own repeat count doubles must not also
    # double what it wraps, or "cast two Easterly Winds" casts four.
    repeats_scales = _has_repeats(e) and not _should_skip_field(
        e, "repeats", int(getattr(e, "repeats", 0) or 0), hints
    )
    updates: Dict[str, Any] = {}
    for f in fields(e):
        if f.name not in _GOLDEN_INT_FIELDS:
            # A wrapper scales by scaling what it wraps: Choose One doubles both
            # halves, and "repeat this for each X" doubles the thing repeated.
            # Without this the Golden Tasty Lobster still gave +1/+1.
            nested = getattr(e, f.name, None)
            if (
                _is_effect(nested)
                and not _scales_at_the_wrapper(hints, e)
                and not repeats_scales
            ):
                inner = hints
                if hints.get("prefer_repeats"):
                    # The "twice" has to land somewhere, and it is not here.
                    inner = {**hints, "prefer_repeats": False, "repeats_only": True}
                scaled_nested = implicit_triple_golden_effect(nested, inner)
                if scaled_nested is not nested:
                    updates[f.name] = scaled_nested
            continue
        val = getattr(e, f.name)
        if not isinstance(val, int) or val <= 0:
            continue
        if _should_skip_field(e, f.name, val, hints):
            continue
        if f.name in ("stat_multiplier", "factor") and hints.get("stat_multiple"):
            # The Golden names the multiple outright ("triple its stats").
            updates[f.name] = int(hints["stat_multiple"])
        elif f.name == "factor":
            updates[f.name] = _scale_factor(e, val, hints)
        else:
            updates[f.name] = val * 2

    scaled = replace(e, **updates) if updates else e
    scaled = _apply_golden_token(scaled, hints, catalog_path)
    return _apply_prefer_repeats(scaled, hints)


def _apply_golden_token(
    e: Effect, hints: Dict[str, Any], catalog_path: Optional[Path] = None
) -> Effect:
    """Point a golden's summon at the Golden printing of its token.

    Only where there is one. The id was built by appending ``_G``, which is a
    guess about the catalog rather than a reading of it — and a wrong guess is
    a token nothing can build: a Golden Eternal Summoner asked for
    ``BG25_008_G`` and its deathrattle raised ``KeyError`` in the middle of a
    fight. A token with no Golden printing summons the plain one.
    """
    if not hints.get("golden_token") or not isinstance(e, SummonEffect):
        return e
    if e.token_id.endswith("_G"):
        return e
    golden_id = golden_upgrade_card_id(e.token_id, catalog_path)
    if golden_id is None:
        return e
    return replace(e, token_id=golden_id)


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
        replace(
            ab, effect=implicit_triple_golden_effect(ab.effect, hints, catalog_path)
        )
        for ab in base
    )


__all__ = [
    "implicit_triple_golden_effect",
    "resolve_triple_forged_abilities",
]

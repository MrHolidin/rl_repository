"""Forged triple (golden) ability resolution — not always `EFFECT[normal]` ×3 concatenation."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Tuple

from src.bg_core.effects import (
    Ability,
    AdaptAllMurlocsEffect,
    AdjacentStatAura,
    AttackBonusPerOtherMurlocGlobal,
    BattlecryMultiplierAura,
    BuffAdjacentBattlecry,
    BuffAllFriendlyMinions,
    BuffAllFriendlyOfTribe,
    BuffAllOtherOfTribe,
    BuffAllWithKeyword,
    BuffListenerIfSummonedMatches,
    BuffRandomFriendly,
    BuffOnePerListedTribeFriendly,
    BuffRandomOtherFriendlyCombat,
    BuffSelf,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffSummonedIfRace,
    DealDamageRandomEnemyMinion,
    DealHeroDamage,
    DeathrattleMultiplierAura,
    DiscoverMurlocEffect,
    Effect,
    GrantKeywordRandomFriendly,
    KeywordStatAura,
    PogoHopperBattlecry,
    StatAura,
    SummonEffect,
    SummonFirstDeadFriendlyMechsThisCombat,
    SummonMultiplierAura,
    SummonOnSelfDamaged,
    SummonRandomMinionEffect,
    TribalOtherStatAura,
)
from src.bg_catalog.patch_catalog import golden_upgrade_card_id


def implicit_triple_golden_effect(e: Effect) -> Effect:
    """Scale a shop/combat ``Effect`` when the catalog golden card lacks an authored override.

    Multiplier auras (Brann-style) go 2→3 as in BG; summons / most numeric battlecries roughly double."""

    # Singleton golden tiers use x3 triggers; normal shop card uses x2 where applicable.
    if isinstance(e, (BattlecryMultiplierAura, DeathrattleMultiplierAura, SummonMultiplierAura)):
        if e.factor == 2:
            return replace(e, factor=3)
        return e

    if isinstance(e, SummonEffect):
        if e.count_from_source_attack:
            return replace(e, dr_wave_count=max(1, e.dr_wave_count * 2))
        return replace(e, count=max(1, e.count * 2))

    if isinstance(e, SummonRandomMinionEffect):
        return replace(e, count=e.count * 2)

    if isinstance(e, SummonOnSelfDamaged):
        return replace(e, count=e.count * 2)

    if isinstance(e, BuffRandomFriendly):
        return replace(
            e,
            attack=e.attack * 2,
            health=e.health * 2,
            repeats=max(1, e.repeats * 2),
        )

    if isinstance(e, BuffOnePerListedTribeFriendly):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, DealHeroDamage):
        return replace(e, amount=e.amount * 2)

    if isinstance(e, DealDamageRandomEnemyMinion):
        return replace(e, amount=e.amount * 2)

    if isinstance(e, DiscoverMurlocEffect):
        return replace(e, repeats=max(1, e.repeats * 2))

    if isinstance(e, AdaptAllMurlocsEffect):
        return replace(e, repeats=max(1, e.repeats * 2))

    if isinstance(e, PogoHopperBattlecry):
        return replace(e, attack_each=e.attack_each * 2, health_each=e.health_each * 2)

    if isinstance(e, BuffAdjacentBattlecry):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, BuffAllOtherOfTribe):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, BuffAllFriendlyOfTribe):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, BuffAllWithKeyword):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, BuffAllFriendlyMinions):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, BuffRandomOtherFriendlyCombat):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, BuffSummonedIfRace):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, BuffListenerIfSummonedMatches):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, BuffSelf):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, StatAura):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, TribalOtherStatAura):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, KeywordStatAura):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, AdjacentStatAura):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, SummonFirstDeadFriendlyMechsThisCombat):
        return replace(e, count=max(1, e.count * 2))

    if isinstance(e, GrantKeywordRandomFriendly):
        return replace(e, repeats=max(1, e.repeats * 2))

    if isinstance(e, BuffSelfWhenFriendlyBattlecryPlaced):
        return replace(e, attack=e.attack * 2, health=e.health * 2)

    if isinstance(e, AttackBonusPerOtherMurlocGlobal):
        return replace(e, per_attack=e.per_attack * 2)

    return e
def resolve_triple_forged_abilities(
    normal_card_id: str,
    effects_table: Mapping[str, Tuple[Ability, ...]],
) -> Tuple[Ability, ...]:
    """Prefer HS golden ``TB_BaconUps_*`` row where authored; otherwise scale normals once."""

    gid = golden_upgrade_card_id(normal_card_id)
    if gid is None:
        return tuple(effects_table.get(normal_card_id, ()))

    spe = effects_table.get(gid)
    if spe:
        return tuple(spe)

    base = effects_table.get(normal_card_id, ())
    return tuple(replace(ab, effect=implicit_triple_golden_effect(ab.effect)) for ab in base)


__all__ = [
    "implicit_triple_golden_effect",
    "resolve_triple_forged_abilities",
]

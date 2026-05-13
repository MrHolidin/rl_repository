"""Build ``CARD_TEMPLATES`` from BG patch catalog (15.6.2 / build 36393)."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, FrozenSet, Tuple

from .effects import (
    Ability,
    AdjacentStatAura,
    AdaptAllMurlocsEffect,
    AttackBonusPerOtherMurlocGlobal,
    BattlecryMultiplierAura,
    BuffAdjacentBattlecry,
    BuffAllFriendlyMinions,
    BuffAllFriendlyOfTribe,
    BuffAllOtherOfTribe,
    BuffAllWithKeyword,
    BuffRandomFriendly,
    BuffRandomOtherFriendlyCombat,
    BuffSelf,
    BuffSelfFromHeroDamageTaken,
    BuffSummonedIfRace,
    BuffListenerIfSummonedMatches,
    CleaveOnAttack,
    DealDamageRandomEnemyMinion,
    DealHeroDamage,
    DeathrattleMultiplierAura,
    DiscoverMurlocEffect,
    GrantKeywordRandomFriendly,
    GrantListenerKeywordIfSummonedMatches,
    HeroImmuneAura,
    KangorSummonCopy,
    Keyword,
    KeywordStatAura,
    SummonEffect,
    SummonMultiplierAura,
    SummonRandomMinionEffect,
    SummonOnSelfDamaged,
    Trigger,
    TribalOtherStatAura,
    ZappTargeting,
    PogoHopperBattlecry,
)
from .patch_catalog import load_tavern_minions, minion_by_id, minion_from_tavern_record
from .state import Minion, Race

# Golden rewards (triple) — not in tavern pool; separate ``card_id`` rows.
GOLDEN_REWARD_IDS: FrozenSet[str] = frozenset(
    {
        "TB_BaconUps_045",
        "TB_BaconUps_055",
        "TB_BaconUps_034",
        "TB_BaconUps_084",
        "TB_BaconUps_099",
    }
)

# Tokens summoned by implemented effects (catalog ids; not tavern offers).
TOKEN_IDS: FrozenSet[str] = frozenset(
    {
        "skele21",
        "CFM_316t",
        "BOT_312t",
        "EX1_finkle",
        "KAR_005a",
        "BRM_006t",
        "EX1_506a",
        "CFM_315t",
        "BOT_445t",
        "EX1_534t",
        "OG_216a",
        "BOT_537t",
        "TRL_232t",
        "BOT_218t",
    }
)

EFFECTS: Dict[str, Tuple[Ability, ...]] = {
    "EX1_162": (Ability(Trigger.AURA, AdjacentStatAura(attack=1, health=0)),),
    "EX1_507": (
        Ability(
            Trigger.AURA,
            TribalOtherStatAura(Race.MURLOC, attack=2, health=0),
        ),
    ),
    "ULD_179": (
        Ability(
            Trigger.AURA,
            KeywordStatAura(Keyword.TAUNT, attack=2, health=0),
        ),
    ),
    "EX1_185": (
        Ability(
            Trigger.AURA,
            TribalOtherStatAura(Race.DEMON, attack=1, health=0),
        ),
    ),
    "GVG_021": (
        Ability(Trigger.AURA, HeroImmuneAura()),
        Ability(
            Trigger.AURA,
            TribalOtherStatAura(Race.DEMON, attack=2, health=2),
        ),
    ),
    "BGS_027": (
        Ability(Trigger.ON_TURN_START, BuffSelf(attack=1, health=0)),
    ),
    "BGS_004": (
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            DealHeroDamage(1),
            filter_race=Race.DEMON,
        ),
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffSelf(attack=2, health=2),
            filter_race=Race.DEMON,
        ),
    ),
    "LOOT_013": (Ability(Trigger.ON_PLACE, DealHeroDamage(2)),),
    "UNG_073": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomFriendly(attack=1, health=1, exclude_self=True),
        ),
    ),
    "BGS_020": (Ability(Trigger.ON_PLACE, DiscoverMurlocEffect(repeats=1)),),
    "BGS_031": (Ability(Trigger.ON_PLACE, AdaptAllMurlocsEffect(repeats=1)),),
    "TB_BaconUps_084": (
        Ability(Trigger.ON_PLACE, AdaptAllMurlocsEffect(repeats=2)),
    ),
    "EX1_093": (
        Ability(
            Trigger.ON_PLACE,
            BuffAdjacentBattlecry(attack=1, health=1, grant_taunt=True),
        ),
    ),
    "BOT_312": (
        Ability(
            Trigger.ON_DEATH,
            SummonEffect(token_id="BOT_312t", count=3),
        ),
    ),
    # Combat engine models Kangor as mech-death listeners (not the printed DR).
    "BGS_012": (Ability(Trigger.ON_FRIENDLY_MECH_DIED, KangorSummonCopy()),),
    "FP1_031": (Ability(Trigger.AURA, DeathrattleMultiplierAura(factor=2)),),
    "LOE_077": (Ability(Trigger.AURA, BattlecryMultiplierAura(factor=2)),),
    "TB_BaconUps_045": (
        Ability(Trigger.AURA, BattlecryMultiplierAura(factor=3)),
    ),
    "TB_BaconUps_055": (
        Ability(Trigger.AURA, DeathrattleMultiplierAura(factor=3)),
    ),
    "DAL_575": (Ability(Trigger.AURA, SummonMultiplierAura(factor=2)),),
    "TB_BaconUps_034": (
        Ability(Trigger.AURA, SummonMultiplierAura(factor=3)),
    ),
    "BGS_022": (Ability(Trigger.AURA, ZappTargeting()),),
    "LOOT_078": (Ability(Trigger.AURA, CleaveOnAttack()),),
    "BGS_010": (Ability(Trigger.ON_PLACE, BuffSelfFromHeroDamageTaken()),),
    "CFM_316": (
        Ability(
            Trigger.ON_DEATH,
            SummonEffect(
                token_id="CFM_316t",
                count=1,
                count_from_source_attack=True,
            ),
        ),
    ),
    "EX1_556": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="skele21", count=1)),
    ),
    "KAR_005": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="KAR_005a", count=1)),
    ),
    "GVG_027": (
        Ability(
            Trigger.ON_TURN_END,
            BuffRandomFriendly(attack=2, health=2, exclude_self=True),
        ),
    ),
    "EX1_577": (
        Ability(
            Trigger.ON_DEATH,
            SummonEffect(token_id="EX1_finkle", for_opponent=True),
        ),
    ),
    "BOT_445": (Ability(Trigger.ON_DEATH, SummonEffect(token_id="BOT_445t", count=1)),),
    "CFM_315": (Ability(Trigger.ON_PLACE, SummonEffect(token_id="CFM_315t", count=1)),),
    "EX1_506": (Ability(Trigger.ON_PLACE, SummonEffect(token_id="EX1_506a", count=1)),),
    "EX1_062": (Ability(Trigger.AURA, AttackBonusPerOtherMurlocGlobal(per_attack=1)),),
    "BOT_606": (Ability(Trigger.ON_DEATH, DealDamageRandomEnemyMinion(amount=4)),),
    "BGS_001": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomFriendly(2, 2, exclude_self=True, filter_race=Race.DEMON),
        ),
    ),
    "BGS_025": (
        Ability(Trigger.ON_DEATH, SummonRandomMinionEffect(count=1, exact_cost=1)),
    ),
    "GVG_048": (
        Ability(
            Trigger.ON_PLACE,
            BuffAllOtherOfTribe(Race.MECHANICAL, attack=2, health=0),
        ),
    ),
    "KAR_095": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomFriendly(1, 1, exclude_self=True, repeats=3),
        ),
    ),
    "OG_256": (Ability(Trigger.ON_DEATH, BuffAllFriendlyMinions(attack=1, health=1)),),
    "OG_221": (
        Ability(
            Trigger.ON_DEATH,
            GrantKeywordRandomFriendly(Keyword.SHIELD, exclude_self=True),
        ),
    ),
    "BGS_023": (
        Ability(Trigger.ON_DEATH, SummonRandomMinionEffect(count=1, exact_cost=2)),
    ),
    "CFM_610": (
        Ability(
            Trigger.ON_PLACE,
            BuffAllFriendlyOfTribe(Race.DEMON, attack=1, health=1),
        ),
    ),
    "DS1_070": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomFriendly(
                2,
                2,
                exclude_self=True,
                filter_race=Race.BEAST,
                grant_taunt=True,
            ),
        ),
    ),
    "EX1_103": (
        Ability(
            Trigger.ON_PLACE,
            BuffAllOtherOfTribe(Race.MURLOC, attack=0, health=2),
        ),
    ),
    "GVG_055": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomFriendly(2, 2, exclude_self=True, filter_race=Race.MECHANICAL),
        ),
    ),
    "OG_216": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="OG_216a", count=2)),
    ),
    "UNG_037": (
        Ability(Trigger.ON_DEATH, BuffRandomOtherFriendlyCombat(attack=1, health=1)),
    ),
    "BGS_024": (
        Ability(Trigger.ON_DEATH, SummonRandomMinionEffect(count=1, exact_cost=4)),
    ),
    "CFM_816": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomFriendly(2, 2, exclude_self=True, filter_race=Race.BEAST),
        ),
    ),
    "DAL_077": (
        Ability(
            Trigger.ON_PLACE,
            GrantKeywordRandomFriendly(
                Keyword.POISONOUS, filter_race=Race.MURLOC, exclude_self=True
            ),
        ),
    ),
    "KAR_702": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomFriendly(2, 2, exclude_self=True, repeats=3),
        ),
    ),
    "BGS_009": (
        Ability(
            Trigger.ON_TURN_END,
            BuffRandomFriendly(2, 2, exclude_self=True, repeats=4),
        ),
    ),
    "BGS_018": (
        Ability(
            Trigger.ON_DEATH,
            BuffAllFriendlyOfTribe(Race.BEAST, attack=4, health=4),
        ),
    ),
    "BOT_537": (Ability(Trigger.ON_DEATH, SummonEffect(token_id="BOT_537t", count=1)),),
    "EX1_534": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="EX1_534t", count=2)),
    ),
    "ICC_807": (
        Ability(Trigger.ON_PLACE, BuffAllWithKeyword(Keyword.TAUNT, attack=2, health=2)),
    ),
    "TRL_232": (Ability(Trigger.ON_OVERKILL, SummonEffect(token_id="TRL_232t", count=1)),),
    "UNG_010": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="EX1_506a", count=3)),
    ),
    "BGS_006": (
        Ability(
            Trigger.ON_DEATH,
            SummonRandomMinionEffect(count=1, legendary_only=True),
        ),
    ),
    "BGS_008": (
        Ability(
            Trigger.ON_DEATH,
            SummonRandomMinionEffect(count=2, require_deathrattle=True),
        ),
    ),
    "GVG_113": (Ability(Trigger.AURA, CleaveOnAttack()),),
    "LOOT_368": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="CS2_065", count=3)),
    ),
    "BGS_017": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            BuffSummonedIfRace(Race.BEAST, attack=3, health=0),
        ),
    ),
    "BGS_021": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            BuffSummonedIfRace(Race.BEAST, attack=4, health=4),
        ),
    ),
    "GVG_062": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            GrantListenerKeywordIfSummonedMatches(Race.MECHANICAL, Keyword.SHIELD),
        ),
    ),
    "EX1_509": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            BuffListenerIfSummonedMatches(Race.MURLOC, attack=1, health=0),
        ),
    ),
    "BOT_218": (
        Ability(
            Trigger.ON_SELF_DAMAGED,
            SummonOnSelfDamaged(token_id="BOT_218t", count=1),
        ),
    ),
    "BRM_006": (
        Ability(
            Trigger.ON_SELF_DAMAGED,
            SummonOnSelfDamaged(token_id="BRM_006t", count=1),
        ),
    ),
    "EX1_531": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_DIED,
            BuffSelf(attack=2, health=1),
            filter_race=Race.BEAST,
        ),
    ),
    "BGS_002": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_DIED,
            DealDamageRandomEnemyMinion(amount=3),
            filter_race=Race.DEMON,
        ),
    ),
    "GVG_106": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_DIED,
            BuffSelf(attack=2, health=2),
            filter_race=Race.MECHANICAL,
        ),
    ),
    "BGS_028": (Ability(Trigger.ON_PLACE, PogoHopperBattlecry()),),
}


def _merge_template(rec_id: str, base: Minion) -> Minion:
    fx = EFFECTS.get(rec_id, ())
    if not fx:
        return base
    return replace(base, abilities=fx)


def build_card_templates() -> Dict[str, Minion]:
    rows = load_tavern_minions()
    by_id = minion_by_id()
    out: Dict[str, Minion] = {}

    for rec in rows:
        if rec.is_bacon_pool and not rec.is_golden:
            base = minion_from_tavern_record(rec)
            out[rec.id] = _merge_template(rec.id, base)

    for gid in GOLDEN_REWARD_IDS:
        rec = by_id[gid]
        base = minion_from_tavern_record(rec)
        out[gid] = _merge_template(gid, base)

    for tid in TOKEN_IDS:
        rec = by_id[tid]
        base = minion_from_tavern_record(rec)
        m = replace(base, is_token=True)
        out[tid] = _merge_template(tid, m)

    out["adapt_plant"] = Minion(
        card_id="adapt_plant",
        base_attack=2,
        base_health=2,
        tier=1,
        name="Plant",
        is_token=True,
    )
    return out


CARD_TEMPLATES: Dict[str, Minion] = build_card_templates()

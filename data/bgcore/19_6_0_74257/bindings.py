"""Card ability bindings for patch 19.6.0 (build 74257).

T-B0: copied from 36393 for overlapping pool ids; 74257-specific rows appended.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple



from src.bg_core.effects import (
    Ability,
    AdjacentStatAura,
    AdaptAllMurlocsEffect,
    AdaptSelfRandomEffect,
    AttackBonusPerOtherMurlocGlobal,
    AttackImmediatelyAfterSurvivingEffect,
    BattlecryMultiplierAura,
    BuffAdjacentBattlecry,
    BuffAllFriendlyMinions,
    BuffAllFriendlyOfTribe,
    BuffAllOtherOfTribe,
    BuffAllShopOffersEffect,
    BuffAllWithKeyword,
    BuffAttackerOnFriendlyAttackEffect,
    BuffListenerIfSummonedMatches,
    BuffOnePerListedTribeFriendly,
    BuffRandomFriendly,
    BuffRandomOtherFriendlyCombat,
    BuffRandomUniqueTribeFriendlies,
    BuffSelf,
    BuffSelfFromHeroDamageTaken,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffSummonedIfRace,
    BuffTargetFriendlyBattlecry,
    BuffTargetFromPiratesBoughtBattlecry,
    BuffSelfWhenFriendlyDeathrattlePlaced,
    BuffLeftmostRepeatedEffect,
    BuffRandomFriendlyFromPlacedTierEffect,
    DealExcessDamageToAdjacentEffect,
    AddRandomMinionToHandOnKillEffect,
    SummonRandomOnSelfDamagedEffect,
    CleaveOnAttack,
    Condition,
    ConditionKind,
    DealDamageRandomEnemyMinion,
    DealHeroDamage,
    DeathrattleMultiplierAura,
    DiscoverMurlocEffect,
    GrantKeywordRandomFriendly,
    GrantListenerKeywordIfSummonedMatches,
    HeroImmuneAura,
    Keyword,
    KeywordStatAura,
    MultiplySelfAttackEffect,
    PogoHopperBattlecry,
    SummonEffect,
    SummonFirstDeadFriendlyMechsThisCombat,
    SummonMultiplierAura,
    SummonOnSelfDamaged,
    SummonRandomMinionEffect,
    StartOfCombatDamagePerFriendlyTribe,
    GainGoldOnDeathEffect,
    GainGoldThisTurnEffect,
    AddRandomMinionToShopEffect,
    AddFromLastOpponentBoardEffect,
    AddRandomMinionToHandEffect,
    AddTokenToHandEffect,
    BuffAdjacentOnAttackedEffect,
    BuffAttackedMinionEffect,
    BuffSelfFromFriendlyTribeCount,
    BuffSelfFromGoldenFriendlyCount,
    BuffSelfFromUniqueTribeCount,
    ConsumeFriendlyBattlecry,
    GrantKeywordAllFriendlyOfTribe,
    TransformIntoShopMinionEffect,
    IncrementShopTribeBonusEffect,
    ReduceUpgradeCostEffect,
    SetNextRollCostEffect,
    Trigger,
    TriggerRandomFriendlyDeathrattleEffect,
    TribalOtherStatAura,
    ZappTargeting,
)
from src.bg_core.minion import Race

GOLDEN_REWARD_IDS: FrozenSet[str] = frozenset(
    {
        "TB_BaconUps_014",
        "TB_BaconUps_034",
        "TB_BaconUps_037",
        "TB_BaconUps_045",
        "TB_BaconUps_055",
        "TB_BaconUps_084",
        "TB_BaconUps_087",
        "TB_BaconUps_099",
    }
)

TOKEN_IDS: FrozenSet[str] = frozenset(
    {
        "BGS_115t",
        "BGS_046t",
        "BGS_061t",
        "BOT_218t",
        "BOT_312t",
        "BOT_445t",
        "BOT_537t",
        "BRM_006t",
        "CFM_315t",
        "CFM_316t",
        "CS2_065",
        "EX1_506a",
        "EX1_534t",
        "EX1_finkle",
        "KAR_005a",
        "OG_216a",
        "TRL_232t",
        "skele21",
    }
)

# Pool minions with no EFFECTS row: keywords / sell override from catalog only.
KEYWORD_ONLY_POOL_IDS: FrozenSet[str] = frozenset(
    {
        "BGS_034",
        "BGS_039",
        "BGS_049",
        "BGS_106",
        "BGS_119",
        "BGS_131",
    }
)

EFFECTS: Dict[str, Tuple[Ability, ...]] = {
    "BGS_001": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomFriendly(2, 2, exclude_self=True, filter_race=Race.DEMON),
        ),
    ),
    "BGS_002": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_DIED,
            DealDamageRandomEnemyMinion(amount=3),
            filter_race=Race.DEMON,
        ),
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
    "BGS_009": (
        Ability(
            Trigger.ON_TURN_END,
            BuffOnePerListedTribeFriendly(
                2,
                2,
                (Race.MECHANICAL, Race.MURLOC, Race.DEMON, Race.BEAST),
                exclude_self=True,
            ),
        ),
    ),
    "BGS_010": (Ability(Trigger.ON_PLACE, BuffSelfFromHeroDamageTaken()),),
    "BGS_012": (
        Ability(Trigger.ON_DEATH, SummonFirstDeadFriendlyMechsThisCombat(count=2)),
    ),
    "BGS_014": (Ability(Trigger.ON_DEATH, SummonEffect(token_id="CS2_065", count=1)),),
    "BGS_017": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            BuffSummonedIfRace(Race.BEAST, attack=3, health=0),
        ),
    ),
    "BGS_018": (
        Ability(
            Trigger.ON_DEATH,
            BuffAllFriendlyOfTribe(Race.BEAST, attack=4, health=4),
        ),
    ),
    "BGS_020": (
        Ability(
            Trigger.ON_PLACE,
            DiscoverMurlocEffect(repeats=1),
            condition=Condition(ConditionKind.OTHER_TRIBE_ON_BOARD, Race.MURLOC),
        ),
    ),
    "BGS_033": (
        Ability(
            Trigger.ON_TURN_START,
            BuffSelf(attack=2, health=2),
            condition=Condition(ConditionKind.LAST_COMBAT_WON),
        ),
    ),
    "BGS_037": (
        Ability(Trigger.ON_SELL, BuffAllShopOffersEffect(attack=1, health=1)),
    ),
    "BGS_021": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            BuffSummonedIfRace(Race.BEAST, attack=4, health=4),
        ),
    ),
    "BGS_022": (Ability(Trigger.AURA, ZappTargeting()),),
    "BGS_023": (
        Ability(Trigger.ON_DEATH, SummonRandomMinionEffect(count=1, exact_tier=2)),
    ),
    "BGS_027": (
        Ability(Trigger.ON_TURN_START, BuffSelf(attack=1, health=0)),
    ),
    "BGS_030": (
        Ability(Trigger.ON_PLACE, BuffAllOtherOfTribe(Race.MURLOC, attack=2, health=2)),
        Ability(Trigger.ON_DEATH, BuffAllOtherOfTribe(Race.MURLOC, attack=2, health=2)),
    ),
    "BGS_032": (Ability(Trigger.ON_OVERKILL, DealDamageRandomEnemyMinion(amount=3)),),
    "BGS_036": (
        Ability(
            Trigger.ON_TURN_END,
            BuffSelfFromFriendlyTribeCount(Race.DRAGON, attack_per=1, health_per=1),
        ),
    ),
    "BGS_038": (
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(attack=2, health=2, filter_race=Race.DRAGON),
        ),
    ),
    "BGS_040": (
        Ability(
            Trigger.ON_DEATH,
            GrantKeywordAllFriendlyOfTribe(Keyword.SHIELD, Race.DRAGON),
        ),
    ),
    "BOT_218": (
        Ability(
            Trigger.ON_SELF_DAMAGED,
            SummonOnSelfDamaged(token_id="BOT_218t", count=1),
        ),
    ),
    "BOT_312": (
        Ability(
            Trigger.ON_DEATH,
            SummonEffect(token_id="BOT_312t", count=3),
        ),
    ),
    "BOT_537": (Ability(Trigger.ON_DEATH, SummonEffect(token_id="BOT_537t", count=1)),),
    "BOT_606": (Ability(Trigger.ON_DEATH, DealDamageRandomEnemyMinion(amount=4)),),
    "BRM_006": (
        Ability(
            Trigger.ON_SELF_DAMAGED,
            SummonOnSelfDamaged(token_id="BRM_006t", count=1),
        ),
    ),
    "CFM_315": (Ability(Trigger.ON_PLACE, SummonEffect(token_id="CFM_315t", count=1)),),
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
    "CFM_610": (
        Ability(
            Trigger.ON_PLACE,
            BuffAllFriendlyOfTribe(Race.DEMON, attack=1, health=1),
        ),
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
    "DAL_575": (Ability(Trigger.AURA, SummonMultiplierAura(factor=2)),),
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
    "EX1_062": (Ability(Trigger.AURA, AttackBonusPerOtherMurlocGlobal(per_attack=1)),),
    "EX1_093": (
        Ability(
            Trigger.ON_PLACE,
            BuffAdjacentBattlecry(attack=1, health=1, grant_taunt=True),
        ),
    ),
    "EX1_103": (
        Ability(
            Trigger.ON_PLACE,
            BuffAllOtherOfTribe(Race.MURLOC, attack=0, health=2),
        ),
    ),
    "EX1_185": (
        Ability(
            Trigger.AURA,
            TribalOtherStatAura(Race.DEMON, attack=1, health=0),
        ),
    ),
    "EX1_506": (Ability(Trigger.ON_PLACE, SummonEffect(token_id="EX1_506a", count=1)),),
    "EX1_507": (
        Ability(
            Trigger.AURA,
            TribalOtherStatAura(Race.MURLOC, attack=2, health=0),
        ),
    ),
    "EX1_509": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            BuffListenerIfSummonedMatches(Race.MURLOC, attack=1, health=0),
        ),
    ),
    "EX1_531": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_DIED,
            BuffSelf(attack=2, health=1),
            filter_race=Race.BEAST,
        ),
    ),
    "EX1_534": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="EX1_534t", count=2)),
    ),
    "EX1_556": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="skele21", count=1)),
    ),
    "FP1_031": (Ability(Trigger.AURA, DeathrattleMultiplierAura(factor=2)),),
    "GVG_021": (
        Ability(Trigger.AURA, HeroImmuneAura()),
        Ability(
            Trigger.AURA,
            TribalOtherStatAura(Race.DEMON, attack=2, health=2),
        ),
    ),
    "GVG_027": (
        Ability(
            Trigger.ON_TURN_END,
            BuffOnePerListedTribeFriendly(
                2, 2, (Race.MECHANICAL,), exclude_self=True
            ),
        ),
    ),
    "GVG_048": (
        Ability(
            Trigger.ON_PLACE,
            BuffAllOtherOfTribe(Race.MECHANICAL, attack=2, health=0),
        ),
    ),
    "GVG_055": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomFriendly(2, 2, exclude_self=True, filter_race=Race.MECHANICAL),
        ),
    ),
    "GVG_106": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_DIED,
            BuffSelf(attack=2, health=2),
            filter_race=Race.MECHANICAL,
        ),
    ),
    "GVG_113": (Ability(Trigger.AURA, CleaveOnAttack()),),
    "ICC_029": (
        Ability(
            Trigger.ON_TURN_END,
            BuffRandomFriendly(attack=3, health=0, exclude_self=True),
        ),
    ),
    "ICC_807": (
        Ability(Trigger.ON_PLACE, BuffAllWithKeyword(Keyword.TAUNT, attack=2, health=2)),
    ),
    "ICC_858": (
        Ability(Trigger.ON_FRIENDLY_SHIELD_LOST, BuffSelf(attack=2, health=0)),
    ),
    "KAR_005": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="KAR_005a", count=1)),
    ),
    "LOE_077": (Ability(Trigger.AURA, BattlecryMultiplierAura(factor=2)),),
    "LOOT_013": (Ability(Trigger.ON_PLACE, DealHeroDamage(2)),),
    "LOOT_078": (Ability(Trigger.AURA, CleaveOnAttack()),),
    "LOOT_368": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="CS2_065", count=3)),
    ),
    "OG_216": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="OG_216a", count=2)),
    ),
    "OG_221": (
        Ability(
            Trigger.ON_DEATH,
            GrantKeywordRandomFriendly(Keyword.SHIELD, exclude_self=True, repeats=1),
        ),
    ),
    "OG_256": (Ability(Trigger.ON_DEATH, BuffAllFriendlyMinions(attack=1, health=1)),),
    "TRL_232": (Ability(Trigger.ON_OVERKILL, SummonEffect(token_id="TRL_232t", count=1)),),
    "ULD_217": (
        Ability(
            Trigger.ON_TURN_END,
            BuffRandomFriendly(attack=1, health=0, exclude_self=True),
        ),
    ),
    "UNG_073": (
        Ability(
            Trigger.ON_PLACE,
            BuffOnePerListedTribeFriendly(
                1, 1, (Race.MURLOC,), exclude_self=True
            ),
        ),
    ),
    "target_buffer": (
        Ability(Trigger.ON_PLACE, BuffTargetFriendlyBattlecry(attack=1, health=1)),
    ),
    "BGS_035": (
        Ability(
            Trigger.ON_FRIENDLY_KILL,
            BuffSelf(attack=2, health=2),
            filter_race=Race.DRAGON,
        ),
    ),
    "BGS_019": (
        Ability(
            Trigger.ON_START_OF_COMBAT,
            StartOfCombatDamagePerFriendlyTribe(Race.DRAGON, amount_per_match=1),
        ),
    ),
    "BGS_041": (
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffAllFriendlyOfTribe(Race.DRAGON, attack=1, health=1),
        ),
    ),
    "BGS_044": (
        Ability(
            Trigger.ON_SELF_DAMAGED,
            SummonRandomOnSelfDamagedEffect(
                race_filter=Race.DEMON, count=1, grant_taunt=True
            ),
        ),
    ),
    "BGS_046": (
        Ability(Trigger.ON_AFTER_ATTACK, AddRandomMinionToHandOnKillEffect()),
    ),
    "BGS_043": (
        Ability(Trigger.ON_PLACE, AddFromLastOpponentBoardEffect()),
    ),
    "BGS_055": (Ability(Trigger.ON_PLACE, ReduceUpgradeCostEffect(amount=1)),),
    "BGS_059": (
        Ability(
            Trigger.ON_PLACE,
            ConsumeFriendlyBattlecry(filter_race=Race.DEMON, gold_reward=3, stat_multiplier=1),
        ),
    ),
    "BGS_071": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            GrantListenerKeywordIfSummonedMatches(Race.MECHANICAL, Keyword.SHIELD),
        ),
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            BuffListenerIfSummonedMatches(Race.MECHANICAL, attack=1, health=0),
        ),
    ),
    "BGS_075": (
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffSelfWhenFriendlyDeathrattlePlaced(attack=1, health=2),
        ),
    ),
    "BGS_069": (
        Ability(
            Trigger.ON_PLACE,
            AdaptSelfRandomEffect(count_from_unique_other_tribes=True),
        ),
    ),
    "BGS_072": (
        Ability(
            Trigger.ON_FRIENDLY_BOUGHT, GainGoldThisTurnEffect(amount=1, filter_race=Race.PIRATE)),
    ),
    "BGS_100": (
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffRandomFriendlyFromPlacedTierEffect(),
            filter_race=Race.ELEMENTAL,
        ),
    ),
    "BGS_105": (
        Ability(
            Trigger.ON_TURN_END,
            BuffLeftmostRepeatedEffect(counter="elementals_played", attack=1, health=1),
        ),
    ),
    "BGS_104": (
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            IncrementShopTribeBonusEffect(Race.ELEMENTAL),
            filter_race=Race.ELEMENTAL,
        ),
    ),
    "BGS_115": (
        Ability(Trigger.ON_SELL, AddTokenToHandEffect(token_id="BGS_115t", count=1)),
    ),
    "BGS_112": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomUniqueTribeFriendlies(count=3, attack=2, health=2),
        ),
    ),
    "BGS_113": (
        Ability(Trigger.ON_PLACE, TransformIntoShopMinionEffect()),
    ),
    "BGS_116": (Ability(Trigger.ON_PLACE, SetNextRollCostEffect(cost=0)),),
    "BGS_122": (
        Ability(
            Trigger.ON_PLACE,
            AddRandomMinionToShopEffect(Race.ELEMENTAL, freeze_slot=True),
        ),
    ),
    "BGS_045": (
        Ability(Trigger.ON_AFTER_ATTACK, MultiplySelfAttackEffect(factor=2)),
    ),
    "BGS_056": (
        Ability(
            Trigger.ON_FRIENDLY_ATTACK,
            BuffAttackerOnFriendlyAttackEffect(Race.PIRATE, attack=2, health=2),
            filter_race=Race.PIRATE,
        ),
    ),
    "BGS_060": (
        Ability(Trigger.ON_SURVIVED_ATTACK, AttackImmediatelyAfterSurvivingEffect()),
    ),
    "BGS_067": (
        Ability(Trigger.ON_FRIENDLY_SHIELD_LOST, BuffSelf(attack=2, health=2)),
    ),
    "BGS_078": (
        Ability(
            Trigger.ON_AFTER_ATTACK,
            TriggerRandomFriendlyDeathrattleEffect(repeats=1),
        ),
    ),
    "BGS_082": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomUniqueTribeFriendlies(count=3, attack=1, health=1),
        ),
    ),
    "BGS_083": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomUniqueTribeFriendlies(count=3, attack=2, health=2),
        ),
    ),
    "BGS_110": (
        Ability(
            Trigger.ON_FRIENDLY_WHEN_ATTACKED,
            BuffAttackedMinionEffect(attack=2, health=0),
            filter_victim_keyword=Keyword.TAUNT,
        ),
    ),
    "BGS_111": (
        Ability(
            Trigger.ON_FRIENDLY_WHEN_ATTACKED,
            BuffSelf(attack=1, health=1),
            filter_victim_keyword=Keyword.TAUNT,
        ),
    ),
    "BGS_200": (Ability(Trigger.ON_DEATH, GainGoldOnDeathEffect(amount=1)),),
    "BGS_201": (
        Ability(
            Trigger.ON_WHEN_ATTACKED,
            BuffAdjacentOnAttackedEffect(attack=1, health=1),
        ),
    ),
    "BGS_047": (
        Ability(
            Trigger.ON_FRIENDLY_ATTACK,
            BuffAllFriendlyMinions(attack=2, health=1),
            filter_race=Race.PIRATE,
        ),
    ),
    "BGS_048": (
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFromPiratesBoughtBattlecry(
                attack_per=1, health_per=1, filter_race=Race.PIRATE
            ),
        ),
    ),
    "BGS_053": (
        Ability(Trigger.ON_PLACE, BuffAllOtherOfTribe(Race.PIRATE, attack=3, health=0)),
    ),
    "BGS_061": (Ability(Trigger.ON_DEATH, SummonEffect(token_id="BGS_061t", count=1)),),
    "BGS_066": (
        Ability(
            Trigger.ON_TURN_END,
            BuffSelfFromGoldenFriendlyCount(attack_per=2, health_per=2),
        ),
    ),
    "BGS_079": (
        Ability(
            Trigger.ON_DEATH,
            SummonRandomMinionEffect(count=3, race_filter=Race.PIRATE),
        ),
    ),
    "BGS_080": (
        Ability(
            Trigger.ON_OVERKILL,
            BuffAllOtherOfTribe(Race.PIRATE, attack=2, health=2),
        ),
    ),
    "BGS_081": (
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffSelf(attack=1, health=1),
            filter_race=Race.PIRATE,
        ),
    ),
    "BGS_120": (
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffRandomFriendly(
                attack=1, health=1, exclude_self=True, filter_race=Race.ELEMENTAL
            ),
            filter_race=Race.ELEMENTAL,
        ),
    ),
    "BGS_121": (
        Ability(
            Trigger.ON_DEATH,
            SummonRandomMinionEffect(count=1, race_filter=Race.ELEMENTAL),
        ),
    ),
    "BGS_123": (
        Ability(Trigger.ON_PLACE, AddRandomMinionToHandEffect(tribe=Race.ELEMENTAL)),
    ),
    "BGS_127": (
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffSelf(attack=0, health=1),
            filter_race=Race.ELEMENTAL,
        ),
    ),
    "BGS_128": (
        Ability(
            Trigger.ON_PLACE,
            BuffAllOtherOfTribe(Race.ELEMENTAL, attack=1, health=1),
        ),
    ),
    "BGS_124": (
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffSelfFromFriendlyTribeCount(
                Race.ELEMENTAL, attack_per=0, health_per=1, exclude_self=False
            ),
            filter_race=Race.ELEMENTAL,
        ),
    ),
    "BGS_126": (
        Ability(Trigger.ON_OVERKILL, DealExcessDamageToAdjacentEffect()),
    ),
    "BGS_202": (
        Ability(
            Trigger.ON_TURN_END,
            BuffSelfFromUniqueTribeCount(attack_per=1, health_per=2),
        ),
    ),
    "BGS_204": (
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            BuffListenerIfSummonedMatches(Race.DEMON, attack=1, health=1),
        ),
    ),
}


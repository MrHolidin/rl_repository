"""Card ability bindings for patch 36.2.0 (build 248348).

Filled in tier order — a tier-1 card is reachable in every game, a tier-7 card
in few — leaning on the engine mechanics already in place: Blood Gems, Rally,
Spellcraft with its "until next turn" buffs, Activate, Choose One, Venomous,
Avenge, Lockbox, Fishbait.

``scripts/check_patch_coverage.py data/bgcore/36_2_0_248348`` lists every pool
card whose text promises something no binding delivers. The list is the queue;
it shrinks as bindings land, and the checker fails the day a binding names a
card the catalog does not have.

Tier 1 is done except for seven cards that need engine mechanics that do not
exist yet; each is named in ``UNBOUND_NEEDS_ENGINE`` below with what it wants.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

from src.bg_core.effects import (
    Ability,
    BuffAttackerOnFriendlyAttackEffect,
    AddRandomCardToHandEffect,
    BuffOnSpellCastOnTribeEffect,
    BuffBoughtMinionEffect,
    AddRandomGoldenMinionEffect,
    BuffHandMinionsEffect,
    GainStatsFromHandEffect,
    GiveOwnStatsToHandEffect,
    AddRandomMinionOfCommonTribeEffect,
    BuffTargetPerGoldSpentEffect,
    BloodGemsOnEveryRefreshEffect,
    DiscoverHeroPowerEffect,
    SummonOnCombatSpaceEffect,
    PayInHealthEffect,
    StealNeighbourBloodGemsEffect,
    Condition,
    ConditionKind,
    GainNearestEnemyStatsEffect,
    PromiseNextTurnEffect,
    RefreshWithTavernSpellsEffect,
    RefreshWithTribeEffect,
    MakeFriendlyGoldenEffect,
    MultiplyFriendlyAttackEffect,
    SetEnemyHealthEffect,
    SellFriendlyForStatsEffect,
    TransformToHigherTierEffect,
    ConsumeTavernMinionEffect,
    DamageFromOwnAttackEffect,
    GrantCombinedChooseOneEffect,
    ImmuneWhileAttackingEffect,
    IncreaseBloodGemBonusEffect,
    RaiseGoldCapEffect,
    SpellsCastResponseEffect,
    SummonGemGolemEffect,
    KeepCombatGainsEffect,
    GoldSpentResponseEffect,
    IncreaseTribeGiftEffect,
    StatsFromNextBuyEffect,
    BuffSharedTribeEffect,
    BuffPerMagnetizationEffect,
    CastSpellAtEffect,
    DoubleNextMagnetizeEffect,
    EchoMagnetizeEffect,
    MagnetizeTokenEffect,
    MagnetizesToTribesEffect,
    AddRandomTavernSpellToHandEffect,
    BuffOnePerListedTribeFriendly,
    CleaveOnAttack,
    DealDamageAllMinions,
    BuffSelfOnFriendlyDamageEffect,
    BuffSelfOnFriendlySoldEffect,
    BuffShopOnEveryRefreshEffect,
    CastRandomTavernSpellEffect,
    DestroyFriendlyEffect,
    BuffListenerIfSummonedMatches,
    DealExcessDamageToAdjacentEffect,
    GrantKeywordRandomFriendly,
    GrantListenerKeywordIfSummonedMatches,
    IncrementShopTribeBonusEffect,
    SetNextRollCostEffect,
    AddCardToNextRefreshesEffect,
    AddRandomMinionToHandEffect,
    AddTavernSpellToHandEffect,
    AddTokenToHandEffect,
    BloodGemTarget,
    BuffAdjacentBattlecry,
    BuffAllShopOffersEffect,
    BuffMatching,
    BuffSummonedIfRace,
    BuffPlacedMinionEffect,
    BuffRandomHandMinionEffect,
    BumpSeatCounterEffect,
    BuffRandomOtherFriendlyCombat,
    BuffTarget,
    BuffSelf,
    BuffTargetFriendlyBattlecry,
    ChooseOneEffect,
    Condition,
    ConditionKind,
    DiscoverTribeEffect,
    ConsumeTavernMinionEffect,
    IncreaseBloodGemBonusEffect,
    CreateSpellcraftSpellEffect,
    DiscoverMinionAtTierEffect,
    DiscoverTavernSpellEffect,
    DealHeroDamage,
    DiscoverMinionAtTierEffect,
    GainBloodGemsEffect,
    GainGoldNextTurnEffect,
    GainGoldThisTurnEffect,
    CountSource,
    FirstSpellcraftIsPermanentEffect,
    GiveLockboxEffect,
    IncreaseTavernSpellBonusEffect,
    KeepCombatGainsEffect,
    GrantKeywordAtAttackThreshold,
    GrantTemporaryBuffEffect,
    Keyword,
    PlayBloodGemsEffect,
    PlaceFishbaitEffect,
    PlayBloodGemsOnAttackerEffect,
    RaiseStandingBonusEffect,
    RefreshesCostHealthEffect,
    RewardAtDamageDealtEffect,
    ScopeKind,
    SelfBonusPerGameCount,
    SummonBestFromHandEffect,
    ReduceTavernSpellCostEffect,
    HeroDamageResponseEffect,
    RepeatPerCountEffect,
    StealTavernMinionEffect,
    SummonBestFromHandEffect,
    GiveOwnStatsToSummonedEffect,
    MultiplySummonedAttackEffect,
    SummonEffect,
    SummonRandomMinionEffect,
    SummonSelfCopyFromHandEffect,
    BuffFromSubjectAttackEffect,
    DevourNeighbourEffect,
    SummonStashedEffect,
    AvengeEffect,
    Multiplier,
    MultiplierKind,
    GainTargetAttackEffect,
    StripKeywordsFromTargetEffect,
    DestroyKillerEffect,
    SummonFirstDeadFriendlyMechsThisCombat,
    CopyLastTavernSpellEffect,
    SetStatsEffect,
    SellValueEffect,
    DealHeroDamagePerTierEffect,
    RetriggerFriendlyAbilityEffect,
    ElementalsPlayedResponseEffect,
    GainStatsFromTavernEffect,
    CopyTargetingSpellEffect,
    CopyTavernMinionEffect,
    TriplesWithAnyOfTribeEffect,
    AddSharedTribeMinionEffect,
    SetArmorEffect,
    Trigger,
    TriggerLeftmostDeathrattleEffect,
)
from src.bg_core.minion import Race

#: Golden rewards ("Get a Discover of a higher tier") — none bound yet.
GOLDEN_REWARD_IDS: FrozenSet[str] = frozenset()

#: Tokens summoned by bound cards. Grows with the deathrattles that summon them.
TOKEN_IDS: FrozenSet[str] = frozenset(
    {
        "BG28_603t",  # Beetle 2/2 — Buzzing Vermin
        "BG_BOT_312t",  # Microbot 1/1 — Cord Puller
        "BG_ICC_026t",  # Skeleton 1/1 — Harmless Bonehead
        "BG36_200t",  # Foraging Bat 1/1 — Flittering Bat
        "BGS_115t",  # Water Droplet 2/2 — Sellemental
        "BG35_150t",  # Demon Fodder — Laboratory Assistant
        "BG25_010t",  # Helping Hand 2/1 Reborn — Handless Forsaken
        "BG31_171t",  # Satellite 3/3 — welded on by Spark Snapper
        "BG19_010",  # Sewer Rat 3/2 — Sewer Lord, and it summons one too
        "BG19_010_G",  # Golden Sewer Rat, which the Golden Lord summons
        "BG19_010t",  # Half-Shell 2/3 Taunt — the Rat's deathrattle
        "BG19_010_Gt",  # Golden Half-Shell 4/6
        "BG30_MagicItem_442t",  # Blood Golem 1/1 — Jailbird Juggernaut's Rally
        # Golden printings a Golden card summons by name rather than by count:
        # "Summon a Golden Tasty Lobster", "Summon a Golden Ancestral Automaton".
        "BG36_202_G",
        "BG_TTN_401_G",
        # The five Chromadrakes: out of the tavern pool this patch, still handed
        # over by Hired Mount, so the package has to carry them.
        "BG34_634t",
        "BG34_635t",
        "BG34_636t",
        "BG34_637t",
        "BG34_638t",
    }
)

#: The spells this package calls Bounties. A family the card data does not
#: mark, and one card reads it ("your Bounties cast twice").
BOUNTY_IDS: FrozenSet[str] = frozenset(
    {
        "BG33_811",  # Healthy
        "BG33_812",  # Hostile
        "BG33_813",  # Selfish
        "BG33_814",  # Friendly
        "BG33_815",  # Wealthy
        "BG31_886",  # Forest's Bounty
    }
)

#: Tavern spells the tavern only offers when a tribe is in the lobby. Nothing
#: in the card data marks a spell's tribe, so the package says it, the same way
#: it names its Bounties above. Only the three families with a printed source
#: are here: the Bounties are Pirate-lobby spells, Spitescale Special makes
#: Spellcraft spells that only Naga mint, and Temperature Shift hands over two
#: Elementals -- which is the one that matters beyond flavour, because it put a
#: tribe on the board that the rotation had excluded.
SPELL_TRIBE_GATES: Dict[str, Race] = {
    "BG33_811": Race.PIRATE,  # Healthy Bounty
    "BG33_812": Race.PIRATE,  # Hostile Bounty
    "BG33_813": Race.PIRATE,  # Selfish Bounty
    "BG33_814": Race.PIRATE,  # Friendly Bounty
    "BG33_815": Race.PIRATE,  # Wealthy Bounty
    "BG31_886": Race.PIRATE,  # Forest's Bounty
    "BG28_606": Race.NAGA,  # Spitescale Special
    "BG31_819": Race.ELEMENTAL,  # Temperature Shift
}

#: Cards that come out of the tavern already golden. The flag is the whole
#: rule: the copy renders golden, and the triple resolver only merges non-golden
#: copies, so three of them never combine into a Triple Reward.
ALWAYS_GOLDEN_POOL_IDS: FrozenSet[str] = frozenset(
    {
        "BG32_236",  # Aureate Laureate
    }
)

#: Pool cards whose whole text is keywords the catalog already carries
#: (a plain Taunt/Divine Shield body), so they need no binding to be correct.
KEYWORD_ONLY_POOL_IDS: FrozenSet[str] = frozenset(
    {
        "BGS_119",  # Crackling Cyclone — Divine Shield, Windfury
        "BG25_001",  # Risen Rider — Taunt, Reborn
        "BG_BOT_911",  # Annoy-o-Module — Magnetic, Divine Shield, Taunt
        "BGS_131",  # Deadly Spore — Venomous
    }
)

#: Pool cards left unbound because the mechanic they need does not exist in the
#: engine yet. Kept here rather than in a commit message so the next pass over a
#: tier can read what is missing without re-deriving it from card text.
#:
#: Empty through tier 1. (Duos-only cards are not listed: they never reach a
#: solo pool — see ``is_duos_only_card_id``.)
UNBOUND_NEEDS_ENGINE: Dict[str, str] = {
    "BG33_891": "Magicfin Mycologist: 'get a 1/1 Murloc and teach it that spell' "
    "needs a token whose abilities are lifted from a spell the seat bought, and "
    "nothing in the engine builds a minion out of a spell. Left out on purpose "
    "rather than pending — the machinery is the card's alone.",
    "BG32_842": "Glowing Cinder: 'your Elementals give an extra +2 Health this game' "
    "modifies what the Elemental-played trigger hands out, and shop_elemental_bonus "
    "is one int used as both the count and the value — it cannot carry a Health-only "
    "bonus. Needs that field split before this card can be said correctly.",
}

EFFECTS: Dict[str, Tuple[Ability, ...]] = {
    # ------------------------------------------------------------------ tier 1
    "BGS_004": (  # Wrath Weaver — after you play a Demon, 1 damage to hero, +2/+2
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
    "BGS_127": (  # Molten Rock — after you play an Elemental, gain +1 Health
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffSelf(attack=0, health=1),
            filter_race=Race.ELEMENTAL,
        ),
    ),
    "BG31_803": (  # Buzzing Vermin — Deathrattle: summon a 2/2 Beetle
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG28_603t", count=1)),
    ),
    "BG29_611": (  # Cord Puller — Deathrattle: summon a 1/1 Microbot
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG_BOT_312t", count=1)),
    ),
    "BG28_300": (  # Harmless Bonehead — Deathrattle: summon two 1/1 Skeletons
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG_ICC_026t", count=2)),
    ),
    "BG26_146": (  # Lullabot — Magnetic; at the end of your turn, gain +1 Health
        Ability(Trigger.ON_TURN_END, BuffSelf(attack=0, health=1)),
    ),
    "BG20_100": (  # Razorfen Geomancer — Battlecry: get 2 Blood Gems
        Ability(Trigger.ON_PLACE, GainBloodGemsEffect(count=2)),
    ),
    "BG23_000": (  # Mini-Myrmidon — Spellcraft: give a minion +2 Attack until next turn
        Ability(
            Trigger.ON_TURN_START,
            CreateSpellcraftSpellEffect(
                buff=GrantTemporaryBuffEffect(attack=2),
                card_id="BG23_000t",
                name="Myrmidon's Might",
            ),
        ),
    ),
    "BG36_200": (  # Flittering Bat — Rally: summon a 1/1 Beast
        Ability(Trigger.ON_ATTACK, SummonEffect(token_id="BG36_200t", count=1)),
    ),
    "BG29_888": (  # Glim Guardian — Rally: gain +2 Attack
        Ability(Trigger.ON_ATTACK, BuffSelf(attack=2, health=0)),
    ),
    "BG33_886": (  # Tusked Camper — Rally: this plays a Blood Gem on itself
        Ability(
            Trigger.ON_ATTACK,
            PlayBloodGemsEffect(target=BloodGemTarget.SELF, count=1),
        ),
    ),
    "BG25_013": (  # Rot Hide Gnoll — +1 Attack per friendly dead this combat
        # Written as a listener rather than an aura: deaths only ever accumulate
        # within a combat, so counting them as they happen gives the same number
        # the card's "for each" would, and combat copies are thrown away after,
        # which is what keeps it from carrying into the next fight.
        Ability(Trigger.ON_FRIENDLY_MINION_DIED, BuffSelf(attack=1, health=0)),
    ),
    "BG33_140": (  # River Skipper — when you sell this, get a random Tier 1 minion
        Ability(Trigger.ON_SELL, AddRandomMinionToHandEffect(tier=1)),
    ),
    "BG26_135": (  # Southsea Busker — Battlecry: gain 1 Gold next turn
        Ability(Trigger.ON_PLACE, GainGoldNextTurnEffect(amount=1)),
    ),
    "BG36_345": (  # Suspicious Prisonguard — Activate (1): give another minion +3/+3
        Ability(
            Trigger.ON_ACTIVATE,
            BuffTargetFriendlyBattlecry(attack=3, health=3, exclude_self=True),
            activate_cost=1,
        ),
    ),
    "BG36_921": (  # Fleeing Fugitive — whenever you cast a spell on this, +1 Health
        Ability(Trigger.ON_TARGETED_BY_SPELL, BuffSelf(attack=0, health=1)),
    ),
    "BG35_814": (  # Scarlet Survivor — once this reaches 6 Attack, gain Divine Shield
        Ability(
            Trigger.AURA,
            GrantKeywordAtAttackThreshold(threshold=6, keyword=Keyword.SHIELD),
        ),
    ),
    "BG32_330": (  # Flighty Scout — Start of Combat: if in hand, summon a copy
        Ability(Trigger.ON_START_OF_COMBAT, SummonSelfCopyFromHandEffect()),
    ),
    "BG31_330": (  # Ominous Seer — Battlecry: next Tavern spell costs (1) less
        Ability(Trigger.ON_PLACE, ReduceTavernSpellCostEffect(amount=1)),
    ),
    # ------------------------------------------------------------------ tier 2
    "BGS_115": (  # Sellemental — when you sell this, get a 3/3 Elemental
        Ability(Trigger.ON_SELL, AddTokenToHandEffect(token_id="BGS_115t")),
    ),
    "BG25_022": (  # Scarlet Skull — Reborn; Deathrattle: a friendly Undead +1/+2
        Ability(
            Trigger.ON_DEATH,
            BuffRandomOtherFriendlyCombat(attack=1, health=2, filter_race=Race.UNDEAD),
        ),
    ),
    "BG20_101": (  # Roadboar — Rally: get a Blood Gem
        Ability(Trigger.ON_ATTACK, GainBloodGemsEffect(count=1)),
    ),
    "BG36_520": (  # Bilgewater Breakout — Battlecry: get a Lockbox (or hurry one)
        Ability(Trigger.ON_PLACE, GiveLockboxEffect(sooner=1)),
    ),
    "BG23_002": (  # Shell Collector — Battlecry: get a Tavern Coin
        Ability(Trigger.ON_PLACE, AddTavernSpellToHandEffect(card_id="BG28_810")),
    ),
    "BG26_963": (  # Electric Synthesizer — Battlecry *and* Start of Combat:
        # give your other Dragons +1/+1. Two triggers, one effect, exactly as
        # the card prints it.
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(BuffTarget.OTHER_OF_TRIBE, tribe=Race.DRAGON, attack=1, health=1),
        ),
        Ability(
            Trigger.ON_START_OF_COMBAT,
            BuffMatching(BuffTarget.OTHER_OF_TRIBE, tribe=Race.DRAGON, attack=1, health=1),
        ),
    ),
    "BG26_805": (  # Humming Bird — Start of Combat: your Beasts have +1 Attack
        Ability(
            Trigger.ON_START_OF_COMBAT,
            BuffMatching(
                BuffTarget.FRIENDLY_OF_TRIBE,
                tribe=Race.BEAST,
                attack=1,
                health=0,
                lasting=True,  # "For the rest of this combat"
            ),
        ),
    ),
    "BG36_342": (  # Clever Castaway — Activate (2): Discover a Tavern spell
        Ability(Trigger.ON_ACTIVATE, DiscoverTavernSpellEffect(), activate_cost=2),
    ),
    "BG32_237": (  # Intrepid Botanist — Choose One: your Tavern spells give
        # an extra +1 Attack this game; or +1 Health
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=IncreaseTavernSpellBonusEffect(attack=1, health=0),
                second=IncreaseTavernSpellBonusEffect(attack=0, health=1),
            ),
        ),
    ),
    # "Improves" — a seat tally the card multiplies itself by. The level starts
    # at one, so an unimproved card is worth exactly what it prints, and the
    # bump comes *after* the effect it improves, because the cards say "future".
    "BG32_170": (  # Metallic Hunter — Deathrattle: get a Pointy Arrow
        Ability(Trigger.ON_DEATH, AddTavernSpellToHandEffect(card_id="EBG_Spell_014")),
    ),
    "BG27_002": (  # Oozeling Gladiator — Battlecry: get two Slimy Shields
        Ability(
            Trigger.ON_PLACE,
            AddTavernSpellToHandEffect(card_id="BG27_002t", count=2),
        ),
    ),
    "BG31_320": (  # Crater Miner — Choose One: 2 Blood Gems; or a Gem Day
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=GainBloodGemsEffect(count=2),
                second=AddTavernSpellToHandEffect(card_id="BG31_893"),
            ),
        ),
    ),
    # The Gem Day card itself: its whole text is the Choose One it offers.
    "BG31_816": (  # Fire Baller — sell: your minions +1 Attack, and Ballers improve
        Ability(
            Trigger.ON_SELL,
            RepeatPerCountEffect(
                counter="ballers_sold",
                effect=BuffMatching(BuffTarget.ALL_FRIENDLY, attack=1, health=0),
            ),
        ),
        Ability(Trigger.ON_SELL, BumpSeatCounterEffect(counter="ballers_sold")),
    ),
    "BG31_818": (  # Snow Baller — the same, in Health, off the same tally
        Ability(
            Trigger.ON_SELL,
            RepeatPerCountEffect(
                counter="ballers_sold",
                effect=BuffMatching(BuffTarget.ALL_FRIENDLY, attack=0, health=1),
            ),
        ),
        Ability(Trigger.ON_SELL, BumpSeatCounterEffect(counter="ballers_sold")),
    ),
    "BG24_715": (  # Patient Scout — sell: Discover a Tier 1 minion, improving each turn
        Ability(
            Trigger.ON_SELL,
            DiscoverMinionAtTierEffect(tier=1, counter="patient_scout_turns"),
        ),
        Ability(Trigger.ON_TURN_END, BumpSeatCounterEffect(counter="patient_scout_turns")),
    ),
    "BG31_924": (  # Thaumaturgist — Spellcraft +1/+1, improved every 4 spells cast
        Ability(
            Trigger.ON_TURN_START,
            CreateSpellcraftSpellEffect(
                buff=GrantTemporaryBuffEffect(attack=1, health=1),
                card_id="BG31_924t",
                name="Thaumaturgy",
                counter="spells_cast:*",
                per=4,
            ),
        ),
    ),
    "BG34_140": (  # Expert Aviator — Rally: summon the best card in hand for this fight
        Ability(Trigger.ON_ATTACK, SummonBestFromHandEffect()),
    ),
    "BG35_150": (  # Laboratory Assistant — a Fodder in each of the next 3 Refreshes
        Ability(
            Trigger.ON_PLACE,
            AddCardToNextRefreshesEffect(card_id="BG35_150t", refreshes=3),
        ),
    ),
    "BG23_009": (  # Lava Lurker — the first Spellcraft spell on this each turn sticks
        Ability(Trigger.AURA, FirstSpellcraftIsPermanentEffect()),
    ),
    "BG23_357": (  # Mind Muck — a friendly Demon eats a tavern minion for its stats
        Ability(
            Trigger.ON_PLACE,
            ConsumeTavernMinionEffect(filter_race=Race.DEMON, count=1),
        ),
    ),
    "BG26_174": (  # Soul Rewinder — hero damage undone, and this grows
        Ability(
            Trigger.ON_HERO_DAMAGE,
            HeroDamageResponseEffect(rewind=True, effect=BuffSelf(attack=0, health=1)),
        ),
    ),
    "BG21_015": (  # Tarecgosa — keeps the stats and keywords it gains in combat
        Ability(Trigger.AURA, KeepCombatGainsEffect()),
    ),
    "BG29_300": (  # Very Hungry Winterfinner — damaged: a minion in hand +2/+1
        Ability(
            Trigger.ON_SELF_DAMAGED,
            BuffRandomHandMinionEffect(attack=2, health=1),
        ),
    ),
    "BG_TTN_401": (  # Ancestral Automaton — +3/+2 per *other* one summoned this game
        Ability(
            Trigger.AURA,
            SelfBonusPerGameCount(counter="summoned", attack_per=3, health_per=2),
        ),
    ),
    "BG25_008": (  # Eternal Knight — +4/+2 per friendly Eternal Knight that died
        # count_self: a Knight that died is one of the deaths its living copies
        # count, and it is not around to double-count itself.
        Ability(
            Trigger.AURA,
            SelfBonusPerGameCount(
                counter="died", attack_per=4, health_per=2, count_self=True
            ),
        ),
    ),
    "BG25_011": (  # Nerubian Deathswarmer — your Undead have +1 Attack this game
        Ability(
            Trigger.ON_PLACE,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.TRIBE, scope_key=Race.UNDEAD, attack=1, health=0
            ),
        ),
    ),
    "BG31_801": (  # Forest Rover — your Beetles have +2/+1 this game; DR: a Beetle
        Ability(
            Trigger.ON_PLACE,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.CARD, scope_key="BG28_603t", attack=2, health=1
            ),
        ),
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG28_603t", count=1)),
    ),
    "BG31_177": (  # Mechagnome Interpreter — play or Magnetize a Mech, give it +3/+1
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffPlacedMinionEffect(attack=3, health=1),
            filter_race=Race.MECHANICAL,
        ),
    ),
    "BG33_430": (  # Prodigious Tusker — another friendly attacks, gem it
        Ability(
            Trigger.ON_FRIENDLY_ATTACK,
            PlayBloodGemsOnAttackerEffect(count=1),
            excludes_self=True,  # "whenever **another** friendly minion"
        ),
    ),
    "BG36_354": (  # Decoy Conjurer — Activate (2): steal the biggest tavern minion
        Ability(
            Trigger.ON_ACTIVATE,
            StealTavernMinionEffect(highest_attack=True),
            activate_cost=2,
        ),
    ),
    "BG36_201": (  # Lurking Lionfish — Activate (2): bait a tavern card
        Ability(Trigger.ON_ACTIVATE, PlaceFishbaitEffect(), activate_cost=2),
    ),
    "BG32_235": (  # Surfing Sylvar — end of turn: adjacent +1 Attack, again per Golden
        Ability(
            Trigger.ON_TURN_END,
            RepeatPerCountEffect(
                source=CountSource.GOLDEN_FRIENDLIES,
                # BuffAdjacentBattlecry is "buff my neighbours" whatever fires
                # it; the name is the printing it was written for and renaming
                # it would move an encoded effect id.
                effect=BuffAdjacentBattlecry(attack=1, health=0),
            ),
        ),
    ),
    "BG29_810": (  # Thousandth Paper Drake — SoC: left-most Dragon +1/+2, Windfury
        Ability(
            Trigger.ON_START_OF_COMBAT,
            BuffMatching(
                BuffTarget.FRIENDLY_OF_TRIBE,
                tribe=Race.DRAGON,
                attack=1,
                health=2,
                limit=1,
                leftmost=True,  # "your **left-most** Dragon"
                grant_keyword=Keyword.WINDFURY,
            ),
        ),
    ),
    # ------------------------------------------------------------------ tier 3
    "BG26_147": (  # Accord-o-Tron — Magnetic; start of turn, gain 1 Gold
        Ability(Trigger.ON_TURN_START, GainGoldThisTurnEffect(amount=1)),
    ),
    "BG24_500": (  # Amber Guardian — SoC: another Dragon +2/+2 and Divine Shield
        Ability(
            Trigger.ON_START_OF_COMBAT,
            BuffMatching(
                BuffTarget.OTHER_OF_TRIBE,
                tribe=Race.DRAGON,
                attack=2,
                health=2,
                limit=1,
                grant_keyword=Keyword.SHIELD,
            ),
        ),
    ),
    "BG33_830": (  # Azsharan Cutlassier — your Tavern spells give +1 Attack
        Ability(Trigger.ON_PLACE, IncreaseTavernSpellBonusEffect(attack=1, health=0)),
    ),
    "BG33_924": (  # Blue Whelp — Rally: your Tavern spells give +1 Health
        Ability(Trigger.ON_ATTACK, IncreaseTavernSpellBonusEffect(attack=0, health=1)),
    ),
    "BG36_507": (  # Breakout Mastermind — Activate (2): get a random Murloc
        Ability(
            Trigger.ON_ACTIVATE,
            AddRandomMinionToHandEffect(tribe=Race.MURLOC),
            activate_cost=2,
        ),
    ),
    "BG34_683": (  # Briarback Drummer — Battlecry: get a Blood Gem Barrage
        Ability(Trigger.ON_PLACE, AddTavernSpellToHandEffect(card_id="BG34_689")),
    ),
    "BG30_125": (  # Cadaver Caretaker — Deathrattle: three 1/1 Skeletons
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG_ICC_026t", count=3)),
    ),
    "BG36_508": (  # Cagey Conjurer — Activate (1): cast a random Tavern spell
        Ability(Trigger.ON_ACTIVATE, CastRandomTavernSpellEffect(), activate_cost=1),
    ),
    "BG23_004": (  # Deep-Sea Angler — Spellcraft: +2/+6 and Taunt until next turn
        Ability(
            Trigger.ON_TURN_START,
            CreateSpellcraftSpellEffect(
                buff=GrantTemporaryBuffEffect(attack=2, health=6, keyword=Keyword.TAUNT),
                card_id="BG23_004t",
                name="Anglerfish",
            ),
        ),
    ),
    "BGS_071": (  # Deflect-o-Bot — a Mech summoned in combat: +2 Attack, Divine Shield
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            GrantListenerKeywordIfSummonedMatches(Race.MECHANICAL, Keyword.SHIELD),
            combat_only=True,
        ),
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            BuffListenerIfSummonedMatches(Race.MECHANICAL, attack=2, health=0),
            combat_only=True,
        ),
    ),
    "BG33_323": (  # Dustbone Devastator — Rally: your Undead +2 Attack this game
        Ability(
            Trigger.ON_ATTACK,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.TRIBE, scope_key=Race.UNDEAD, attack=2, health=0
            ),
        ),
    ),
    "BG36_346": (  # Fruit Vendor — Activate (1): get 2 Tavern Dish Bananas
        Ability(
            Trigger.ON_ACTIVATE,
            AddTavernSpellToHandEffect(card_id="BG28_897", count=2),
            activate_cost=1,
        ),
    ),
    "BG31_326": (  # Gem Rat — at the end of your turn, get a Gem Day
        Ability(Trigger.ON_TURN_END, AddTavernSpellToHandEffect(card_id="BG31_893")),
    ),
    "BG25_010": (  # Handless Forsaken — Deathrattle: a 2/1 Helping Hand with Reborn
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG25_010t", count=1)),
    ),
    "BG36_521": (  # Locked-up Mutineer — Deathrattle: a Lockbox, or hurry one
        Ability(Trigger.ON_DEATH, GiveLockboxEffect(sooner=1)),
    ),
    "BG35_140": (  # Mama Mrrglton — other Murlocs +2 Attack, improved per Mrrglton
        Ability(
            Trigger.ON_PLACE,
            RepeatPerCountEffect(
                counter="mrrgltons_played",
                effect=BuffMatching(
                    BuffTarget.OTHER_OF_TRIBE, tribe=Race.MURLOC, attack=2, health=0
                ),
            ),
        ),
        Ability(Trigger.ON_PLACE, BumpSeatCounterEffect(counter="mrrgltons_played")),
    ),
    "BG35_141": (  # Papa Mrrglton — the same in Health, off the same tally
        Ability(
            Trigger.ON_PLACE,
            RepeatPerCountEffect(
                counter="mrrgltons_played",
                effect=BuffMatching(
                    BuffTarget.OTHER_OF_TRIBE, tribe=Race.MURLOC, attack=0, health=2
                ),
            ),
        ),
        Ability(Trigger.ON_PLACE, BumpSeatCounterEffect(counter="mrrgltons_played")),
    ),
    "BG28_309": (  # Mummifier — Deathrattle: a different friendly Undead gets Reborn
        Ability(
            Trigger.ON_DEATH,
            GrantKeywordRandomFriendly(Keyword.REBORN, filter_race=Race.UNDEAD),
        ),
    ),
    "BG36_509": (  # Private Investigator — Activate (1): 2 Gold next turn
        Ability(Trigger.ON_ACTIVATE, GainGoldNextTurnEffect(amount=2), activate_cost=1),
    ),
    "BG36_854": (  # Rescue Bot — Taunt; Deathrattle: get a Repair Job
        Ability(Trigger.ON_DEATH, AddTavernSpellToHandEffect(card_id="BG36_624")),
    ),
    "BG29_816": (  # Roaring Recruiter — another Dragon attacks: give it +3/+1
        Ability(
            Trigger.ON_FRIENDLY_ATTACK,
            BuffAttackerOnFriendlyAttackEffect(Race.DRAGON, attack=3, health=1),
            excludes_self=True,  # "whenever **another** friendly Dragon"
        ),
    ),
    "BG32_841": (  # Sand Swirler — your Elementals give an extra +1 Attack
        Ability(
            Trigger.ON_PLACE,
            IncreaseTribeGiftEffect(tribe=Race.ELEMENTAL, attack=1, health=0),
        ),
    ),
    "BG26_360": (  # Scourfin — Deathrattle: a random minion in hand +7/+7
        Ability(Trigger.ON_DEATH, BuffRandomHandMinionEffect(attack=7, health=7)),
    ),
    "BG36_330": (  # Sly Infiltrator — Choose One: 2 free Refreshes; or 3 Blood Gems
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=SetNextRollCostEffect(cost=0, uses=2),
                second=GainBloodGemsEffect(count=3),
            ),
        ),
    ),
    "BG27_084": (  # Sprightly Scarab — Choose One, on a Beast
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=BuffTargetFriendlyBattlecry(
                    attack=1,
                    health=1,
                    filter_race=Race.BEAST,
                    grant_keyword=Keyword.REBORN,
                ),
                second=BuffTargetFriendlyBattlecry(
                    attack=4,
                    health=0,
                    filter_race=Race.BEAST,
                    grant_keyword=Keyword.WINDFURY,
                ),
            ),
        ),
    ),
    "BG36_202": (  # Tasty Lobster — Deathrattle: two Beasts +1/+1, improving
        Ability(
            Trigger.ON_DEATH,
            RepeatPerCountEffect(
                counter="tasty_lobsters",
                effect=BuffMatching(
                    BuffTarget.OTHER_OF_TRIBE,
                    tribe=Race.BEAST,
                    attack=1,
                    health=1,
                    limit=2,
                ),
            ),
        ),
        Ability(Trigger.ON_DEATH, BumpSeatCounterEffect(counter="tasty_lobsters")),
    ),
    "BG27_005": (  # Timecap'n Hooktail — a Tavern spell cast: your minions +1 Attack
        Ability(
            Trigger.ON_TAVERN_SPELL_CAST,
            BuffMatching(BuffTarget.ALL_FRIENDLY, attack=1, health=0),
        ),
    ),
    "BG36_730": (  # Trapped Clapper — Deathrattle: a Fodder in the next 3 Refreshes
        Ability(
            Trigger.ON_DEATH,
            AddCardToNextRefreshesEffect(card_id="BG35_150t", refreshes=3),
        ),
    ),
    "BG23_007": (  # Waverider — Spellcraft: +2/+2, and Windfury to a Naga
        Ability(
            Trigger.ON_TURN_START,
            CreateSpellcraftSpellEffect(
                buff=GrantTemporaryBuffEffect(
                    attack=2,
                    health=2,
                    keyword=Keyword.WINDFURY,
                    keyword_if_race=Race.NAGA,
                ),
                card_id="BG23_007t",
                name="Waverider's Wave",
            ),
        ),
    ),
    "BGS_126": (  # Wildfire Elemental — overkill splashes onto a neighbour
        Ability(Trigger.ON_OVERKILL, DealExcessDamageToAdjacentEffect()),
    ),
    "BG33_155": (  # Devout Hellcaller — another friendly Demon deals damage: +1/+2
        Ability(
            Trigger.AURA,
            BuffSelfOnFriendlyDamageEffect(
                attack=1, health=2, filter_race=Race.DEMON, permanent=True
            ),
        ),
    ),
    "BG27_556": (  # Diremuck Forager — SoC: the best Murloc in hand, for this fight
        Ability(
            Trigger.ON_START_OF_COMBAT,
            SummonBestFromHandEffect(filter_race=Race.MURLOC),
        ),
    ),
    "BG36_240": (  # Hired Mount — Activate (2): get a random Chromadrake
        Ability(
            Trigger.ON_ACTIVATE,
            AddRandomCardToHandEffect(
                card_ids=(
                    "BG34_634t",  # Blue
                    "BG34_635t",  # Black
                    "BG34_636t",  # Green
                    "BG34_637t",  # Bronze
                    "BG34_638t",  # Red
                )
            ),
            activate_cost=2,
        ),
    ),
    "BG31_843": (  # Meteorite Crasher — after you sell an Elemental, gain +2/+2
        Ability(
            Trigger.ON_SELL,
            BuffSelfOnFriendlySoldEffect(attack=2, health=2, filter_race=Race.ELEMENTAL),
        ),
    ),
    "BG25_806": (  # Sly Raptor — Deathrattle: a random Beast, set to 6/6
        Ability(
            Trigger.ON_DEATH,
            SummonRandomMinionEffect(
                count=1, race_filter=Race.BEAST, set_attack=6, set_health=6
            ),
        ),
    ),
    "BG34_856": (  # Waveling — Deathrattle: every roll from now on buffs the tavern
        Ability(Trigger.ON_DEATH, BuffShopOnEveryRefreshEffect(attack=3, health=3)),
    ),
    "BG36_207": (  # Wolf Pup — Rally: give your *other* minions +4/+2
        Ability(
            Trigger.ON_ATTACK,
            BuffMatching(
                BuffTarget.ALL_FRIENDLY, attack=4, health=2, exclude_source=True
            ),
        ),
    ),
    "BG28_303": (  # Disguised Graverobber — destroy a friendly Undead for a plain copy
        Ability(
            Trigger.ON_PLACE,
            DestroyFriendlyEffect(filter_race=Race.UNDEAD),
        ),
    ),
    "BG26_524": (  # Malchezaar — two Refreshes a turn cost Health instead of Gold
        Ability(Trigger.AURA, RefreshesCostHealthEffect(amount=1, uses=2)),
    ),
    "BG36_763": (  # Treasure Parrot — once this has dealt 40 damage, a Golden Touch
        Ability(
            Trigger.AURA,
            RewardAtDamageDealtEffect(threshold=40, card_id="BG28_830"),
        ),
    ),
    # ------------------------------------------------------------------ tier 4
    "BG32_172": (  # Auto Assembler — Magnetic; Deathrattle: an Ancestral Automaton
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG_TTN_401", count=1)),
    ),
    "BG33_822": (  # Bigwig Bandit — Rally: get a random Bounty
        Ability(
            Trigger.ON_ATTACK,
            AddRandomCardToHandEffect(
                card_ids=(
                    "BG33_811",  # Healthy
                    "BG33_812",  # Hostile
                    "BG33_813",  # Selfish
                    "BG33_814",  # Friendly
                    "BG33_815",  # Wealthy
                )
            ),
        ),
    ),
    "BG26_817": (  # Blade Collector — the swing also hits whoever stands beside
        Ability(Trigger.AURA, CleaveOnAttack()),
    ),
    "BG20_104": (  # Bonker — Windfury; Rally: a Blood Gem on all your others
        Ability(
            Trigger.ON_ATTACK,
            PlayBloodGemsEffect(target=BloodGemTarget.ALL_OTHER_FRIENDLY, count=1),
        ),
    ),
    "BG36_620": (  # Boom-in-a-Box — Taunt; Start of Combat: 3 to all other minions
        Ability(Trigger.ON_START_OF_COMBAT, DealDamageAllMinions(amount=3)),
    ),
    "BG36_242": (  # Bronze Timewalker — Rally: get a random Chromadrake
        Ability(
            Trigger.ON_ATTACK,
            AddRandomCardToHandEffect(
                card_ids=(
                    "BG34_634t",
                    "BG34_635t",
                    "BG34_636t",
                    "BG34_637t",
                    "BG34_638t",
                )
            ),
        ),
    ),
    "BG36_211": (  # Cage Gnawer — a friendly Beast's swing pays your Beasts
        Ability(
            Trigger.ON_FRIENDLY_ATTACK,
            BuffMatching(
                target=BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.BEAST, attack=2, health=1
            ),
            filter_race=Race.BEAST,
        ),
    ),
    "BG36_760": (  # Captain Cookie — Deathrattle: get a Chef's Choice
        Ability(Trigger.ON_DEATH, AddTavernSpellToHandEffect(card_id="BG28_518")),
    ),
    "BG35_143": (  # Deepwater Chieftain — Battlecry *and* Deathrattle: a Deepwater Clan
        Ability(Trigger.ON_PLACE, AddTavernSpellToHandEffect(card_id="BG35_149")),
        Ability(Trigger.ON_DEATH, AddTavernSpellToHandEffect(card_id="BG35_149")),
    ),
    "BG36_762": (  # Devilish Distractor — a spell on this buffs the tavern for good
        Ability(
            Trigger.ON_TARGETED_BY_SPELL,
            RaiseStandingBonusEffect(scope_kind=ScopeKind.SHOP, attack=2, health=2),
        ),
    ),
    "BG34_865": (  # En-Djinn Blazer — every roll from now on buffs the tavern
        Ability(Trigger.ON_PLACE, BuffShopOnEveryRefreshEffect(attack=7, health=7)),
    ),
    "BG30_123": (  # Fearless Foodie — Choose One: better Gems, or four of them
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=IncreaseBloodGemBonusEffect(attack=1, health=1),
                second=GainBloodGemsEffect(count=4),
            ),
        ),
    ),
    "BG32_880": (  # Friendly Geist — Deathrattle: Tavern spells give +1 Attack
        Ability(Trigger.ON_DEATH, IncreaseTavernSpellBonusEffect(attack=1, health=0)),
    ),
    "BG36_764": (  # Gearfin — end of turn: two 1-Cost Tavern spells
        Ability(
            Trigger.ON_TURN_END,
            AddRandomTavernSpellToHandEffect(count=2, max_cost=1),
        ),
    ),
    "BG36_204": (  # Headhunter Gryphon — Rally: get a random Beast
        Ability(Trigger.ON_ATTACK, AddRandomMinionToHandEffect(tribe=Race.BEAST)),
    ),
    "BG36_524": (  # Maritime Extortionist — +8/+8 per Golden minion played this game
        Ability(
            Trigger.AURA,
            SelfBonusPerGameCount(
                counter="golden_played", subject="*", attack_per=8, health_per=8,
                count_self=True,
            ),
        ),
    ),
    "BG27_080": (  # Motley Phalanx — Deathrattle: one of each type +2/+2, for keeps
        Ability(
            Trigger.ON_DEATH,
            BuffOnePerListedTribeFriendly(attack=2, health=2, permanent=True),
        ),
    ),
    "BG34_682": (  # Razorfen Flapper — Deathrattle: get a Blood Gem Barrage
        Ability(Trigger.ON_DEATH, AddTavernSpellToHandEffect(card_id="BG34_689")),
    ),
    "BGS_116": (  # Refreshing Anomaly — Battlecry: two free Refreshes
        Ability(Trigger.ON_PLACE, SetNextRollCostEffect(cost=0, uses=2)),
    ),
    "BGS_123": (  # Tavern Tempest — Battlecry: get a random Elemental
        Ability(Trigger.ON_PLACE, AddRandomMinionToHandEffect(tribe=Race.ELEMENTAL)),
    ),
    "BG34_684": (  # Trench Fighter — end of turn: get a Gem Confiscation
        Ability(Trigger.ON_TURN_END, AddTavernSpellToHandEffect(card_id="BG28_698")),
    ),
    # ------------------------------------------------------- the Mech family
    "BG_DEEP_015": (  # Prosthetic Hand — Magnetic, Reborn; welds to Undead too
        Ability(
            Trigger.AURA,
            MagnetizesToTribesEffect(tribes=(Race.MECHANICAL, Race.UNDEAD)),
        ),
    ),
    "BG36_851": (  # Spark Snapper — a Mech played gets a 3/3 Satellite, improving
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            MagnetizeTokenEffect(
                token_id="BG31_171t",  # Satellite, printed 6/6
                attack=3,
                health=3,
                improves="spark_snappers",
            ),
            filter_race=Race.MECHANICAL,
        ),
    ),
    "BG36_506": (  # Drone Duplicator — Activate (1): the next weld here is doubled
        Ability(Trigger.ON_ACTIVATE, DoubleNextMagnetizeEffect(), activate_cost=1),
    ),
    "BG26_152": (  # Utility Drone — end of turn: +4/+4 per Magnetization carried
        Ability(Trigger.ON_TURN_END, BuffPerMagnetizationEffect(attack=4, health=4)),
    ),
    "BG26_149": (  # Polarizing Beatboxer — a weld elsewhere also lands here
        Ability(Trigger.AURA, EchoMagnetizeEffect()),
    ),
    # ------------------------------------------------------- the Naga family
    "BG23_008": (  # Glowscale — Spellcraft: Divine Shield until next turn
        Ability(
            Trigger.ON_TURN_START,
            CreateSpellcraftSpellEffect(
                buff=GrantTemporaryBuffEffect(keyword=Keyword.SHIELD),
                card_id="BG23_008t",
                name="Glowscale's Ward",
            ),
        ),
    ),
    "BG33_319": (  # Rimescale Priestess — Spellcraft: a Tavern spell that gives stats
        Ability(
            Trigger.ON_TURN_START,
            CreateSpellcraftSpellEffect(
                buff=AddRandomTavernSpellToHandEffect(count=1, gives_stats=True),
                card_id="BG33_319t",
                name="Rimescale Rites",
            ),
        ),
    ),
    "BG32_835": (  # Tranquil Meditative — Spellcraft: your Tavern spells give +1/+1
        Ability(
            Trigger.ON_TURN_START,
            CreateSpellcraftSpellEffect(
                buff=IncreaseTavernSpellBonusEffect(attack=1, health=1),
                card_id="BG32_835t",
                name="Tranquil Tide",
            ),
        ),
    ),
    "BG31_920": (  # Darkcrest Strategist — Spellcraft: a Tier 1 Naga, improving
        Ability(
            Trigger.ON_TURN_START,
            CreateSpellcraftSpellEffect(
                buff=AddRandomMinionToHandEffect(tribe=Race.NAGA, tier=1),
                card_id="BG31_920t",
                name="Darkcrest Call",
            ),
        ),
    ),
    "BG35_921": (  # Abyssal Bruiser — +2/+1 per Tavern spell cast this game
        Ability(
            Trigger.AURA,
            SelfBonusPerGameCount(
                counter="tavern_spells_cast", subject="*",
                attack_per=2, health_per=1, count_self=True,
            ),
        ),
    ),
    "BG31_925": (  # Showy Cyclist — Deathrattle: your Naga +2/+2, improving
        Ability(
            Trigger.ON_DEATH,
            RepeatPerCountEffect(
                counter="spells_cast:*",
                per=4,
                effect=BuffMatching(
                    BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.NAGA, attack=2, health=2
                ),
            ),
        ),
    ),
    "BG31_035": (  # Groundbreaker — after you play a Naga, gain +1/+1, improving
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            RepeatPerCountEffect(
                counter="spells_cast:*",
                per=4,
                effect=BuffSelf(attack=1, health=1),
            ),
            filter_race=Race.NAGA,
        ),
    ),
    "BG36_622": (  # Torrential Ruiner — a spell on a Naga: your minions +3/+3
        Ability(
            Trigger.AURA,
            BuffOnSpellCastOnTribeEffect(tribe=Race.NAGA, attack=3, health=3),
        ),
    ),
    "BG32_837": (  # Fauna Whisperer — end of turn: Natural Blessing on the neighbours
        Ability(
            Trigger.ON_TURN_END,
            CastSpellAtEffect(card_id="BG28_845", adjacent=True),
        ),
    ),
    "BG34_925": (  # Seafloor Recruiter — Rally: Chef's Choice on the minion right
        Ability(
            Trigger.ON_ATTACK,
            CastSpellAtEffect(card_id="BG28_518", to_the_right=True),
        ),
    ),
    # ----------------------------------------------------- the Demon family
    "BG34_500": (  # Flaming Enforcer — end of turn: eat the biggest tavern minion
        Ability(
            Trigger.ON_TURN_END,
            ConsumeTavernMinionEffect(highest_health=True, eater_is_source=True),
        ),
    ),
    "BG36_503": (  # Soulkeeping Jailer — Activate (2): your Demons each eat one
        Ability(
            Trigger.ON_ACTIVATE,
            ConsumeTavernMinionEffect(filter_race=Race.DEMON, each=True),
            activate_cost=2,
        ),
    ),
    "BG21_004": (  # Insatiable Ur'zul — Taunt; a Demon played, and it eats
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            ConsumeTavernMinionEffect(eater_is_source=True),
            filter_race=Race.DEMON,
        ),
    ),
    "BG32_873": (  # Ashen Corruptor — hero damage undone, and the tavern grows
        Ability(
            Trigger.ON_HERO_DAMAGE,
            HeroDamageResponseEffect(
                rewind=True, effect=BuffAllShopOffersEffect(attack=1, health=1)
            ),
        ),
    ),
    "BG26_523": (  # Tichondrius — hero damage taken: your Demons +3/+2
        Ability(
            Trigger.ON_HERO_DAMAGE,
            HeroDamageResponseEffect(
                effect=BuffMatching(
                    BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.DEMON, attack=3, health=2
                )
            ),
        ),
    ),
    "BG36_733": (  # Eredar Escapist — every 4 hero damage, a Tavern spell
        Ability(
            Trigger.ON_HERO_DAMAGE,
            HeroDamageResponseEffect(
                threshold=4,
                effect=AddRandomTavernSpellToHandEffect(count=1, gives_stats=True),
            ),
        ),
    ),
    "BG35_152": (  # Void Pup Trainer — the tavern's small minions +3/+3 this game
        Ability(
            Trigger.ON_PLACE,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.SHOP, attack=3, health=3, scope_max_tier=3
            ),
        ),
    ),
    "BG27_016": (  # Champion of Sargeras — Battlecry *and* Deathrattle
        Ability(
            Trigger.ON_PLACE,
            RaiseStandingBonusEffect(scope_kind=ScopeKind.SHOP, attack=5, health=5),
        ),
        Ability(
            Trigger.ON_DEATH,
            RaiseStandingBonusEffect(scope_kind=ScopeKind.SHOP, attack=5, health=5),
        ),
    ),
    "BG35_155": (  # Twisted Wrathguard — a sale leaves a Fodder in the next roll
        # A watcher of *other* sales, not something its own sale does — so the
        # card it leaves behind rides on the watcher rather than on ON_SELL,
        # which would only fire when the Wrathguard itself was sold.
        Ability(
            Trigger.ON_SELL,
            BuffSelfOnFriendlySoldEffect(
                effect=AddCardToNextRefreshesEffect(card_id="BG35_150t", refreshes=1)
            ),
        ),
    ),
    "BG36_731": (  # Imp-lusionist — Deathrattle: two Methodical Madness
        Ability(
            Trigger.ON_DEATH,
            AddTavernSpellToHandEffect(card_id="BG36_880", count=2),
        ),
    ),
    # ------------------------------------------------- the Elemental family
    "BG32_842": (  # Glowing Cinder — your Elementals give an extra +2 Health
        Ability(
            Trigger.ON_DEATH,
            IncreaseTribeGiftEffect(tribe=Race.ELEMENTAL, attack=0, health=2),
        ),
    ),
    "BG36_351": (  # Moat Custodian — Rally: your Elementals give an extra +1/+2
        Ability(
            Trigger.ON_ATTACK,
            IncreaseTribeGiftEffect(tribe=Race.ELEMENTAL, attack=1, health=2),
        ),
    ),
    "BG36_181": (  # Air Baller — sell: your minions +2/+2, and Ballers improve
        Ability(
            Trigger.ON_SELL,
            RepeatPerCountEffect(
                counter="ballers_sold",
                effect=BuffMatching(BuffTarget.ALL_FRIENDLY, attack=2, health=2),
            ),
        ),
        Ability(Trigger.ON_SELL, BumpSeatCounterEffect(counter="ballers_sold")),
    ),
    "BG26_162": (  # Dancing Barnstormer — Battlecry *and* Deathrattle
        Ability(
            Trigger.ON_PLACE,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.SHOP, scope_key=Race.ELEMENTAL, attack=8, health=8
            ),
        ),
        Ability(
            Trigger.ON_DEATH,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.SHOP, scope_key=Race.ELEMENTAL, attack=8, health=8
            ),
        ),
    ),
    "BG26_537": (  # Flourishing Frostling — +2/+1 per Elemental played this game
        Ability(
            Trigger.AURA,
            SelfBonusPerGameCount(
                counter="elementals_played", subject="*",
                attack_per=2, health_per=1, count_self=True,
            ),
        ),
    ),
    "BG32_846": (  # Unleashed Mana Surge — an Elemental played: your Elementals +4/+4
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffMatching(
                BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.ELEMENTAL, attack=4, health=4
            ),
            filter_race=Race.ELEMENTAL,
        ),
    ),
    "BG34_858": (  # Air Revenant — every 7 Gold spent, cast Easterly Winds
        Ability(
            Trigger.AURA,
            GoldSpentResponseEffect(
                threshold=7,
                effect=CastSpellAtEffect(card_id="BG34_444"),
            ),
        ),
    ),
    "BG36_180": (  # Living Prison — Activate (1): take the next buy's stats
        Ability(Trigger.ON_ACTIVATE, StatsFromNextBuyEffect(), activate_cost=1),
    ),
    "BG34_950": (  # Stone Age Slab — a minion bought gets +10/+10 and doubles
        Ability(
            Trigger.AURA,
            BuffBoughtMinionEffect(
                attack=10, health=10, stat_multiplier=2, once_per_turn=True
            ),
        ),
    ),
    # ---------------------------------------------------- the Pirate family
    "BG36_523": (  # Enterprising Escapee — every 5 Gold, a Lockbox
        Ability(
            Trigger.AURA,
            GoldSpentResponseEffect(threshold=5, effect=GiveLockboxEffect(sooner=1)),
        ),
    ),
    "BG26_810": (  # Gunpowder Courier — every 5 Gold, your Pirates +2 Attack
        Ability(
            Trigger.AURA,
            GoldSpentResponseEffect(
                threshold=5,
                effect=BuffMatching(
                    BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.PIRATE, attack=2, health=0
                ),
            ),
        ),
    ),
    "BG31_824": (  # Dual-Wield Corsair — every 5 Gold, two Pirates +4/+5
        Ability(
            Trigger.AURA,
            GoldSpentResponseEffect(
                threshold=5,
                effect=BuffMatching(
                    BuffTarget.FRIENDLY_OF_TRIBE,
                    tribe=Race.PIRATE,
                    attack=4,
                    health=5,
                    limit=2,
                ),
            ),
        ),
    ),
    "BG33_823": (  # Sky Admiral Rogers — every 9 Gold, a random Bounty
        Ability(
            Trigger.AURA,
            GoldSpentResponseEffect(
                threshold=9,
                effect=AddRandomCardToHandEffect(
                    card_ids=(
                    "BG33_811",  # Healthy
                    "BG33_812",  # Hostile
                    "BG33_813",  # Selfish
                    "BG33_814",  # Friendly
                    "BG33_815",  # Wealthy
                    )
                ),
            ),
        ),
    ),
    "BG33_821": (  # Shipwrecked Rascal — Battlecry *and* Deathrattle: a Bounty
        Ability(
            Trigger.ON_PLACE,
            AddRandomCardToHandEffect(
                card_ids=(
                    "BG33_811",  # Healthy
                    "BG33_812",  # Hostile
                    "BG33_813",  # Selfish
                    "BG33_814",  # Friendly
                    "BG33_815",  # Wealthy
                )
            ),
        ),
        Ability(
            Trigger.ON_DEATH,
            AddRandomCardToHandEffect(
                card_ids=(
                    "BG33_811",  # Healthy
                    "BG33_812",  # Hostile
                    "BG33_813",  # Selfish
                    "BG33_814",  # Friendly
                    "BG33_815",  # Wealthy
                )
            ),
        ),
    ),
    "BG33_825": (  # Proud Privateer — your Bounties cast twice
        Ability(Trigger.AURA, Multiplier(MultiplierKind.BOUNTY, factor=2)),
    ),
    "BG36_343": (  # Silent Deliverer — a random Golden Tier 4, owing no Triple
        Ability(Trigger.ON_PLACE, AddRandomGoldenMinionEffect(tier=4)),
    ),
    "BG36_344": (  # Hooktusk — a Discover pays your other Pirates, improving
        Ability(
            Trigger.ON_DISCOVERED,
            RepeatPerCountEffect(
                counter="golden_played:*",
                effect=BuffMatching(
                    BuffTarget.FRIENDLY_OF_TRIBE,
                    tribe=Race.PIRATE,
                    attack=1,
                    health=1,
                    exclude_source=True,
                ),
            ),
        ),
    ),
    "BG26_814": (  # Lovesick Balladist — a Pirate +1 Health per Gold spent this turn
        Ability(
            Trigger.ON_PLACE,
            BuffTargetPerGoldSpentEffect(attack=0, health=1, filter_race=Race.PIRATE),
        ),
    ),
    "BG25_034": (  # Captain Sanders — make a friendly from Tier 6 or below Golden
        Ability(Trigger.ON_PLACE, MakeFriendlyGoldenEffect(max_tier=6)),
    ),
    # ---------------------------------------------------- the Murloc family
    "BG26_137": (  # Bream Counter — while in *hand*, a Murloc played pays it
        Ability(
            Trigger.WHILE_IN_HAND,
            BuffSelf(attack=6, health=6),
            filter_race=Race.MURLOC,
        ),
    ),
    "BG36_703": (  # Twilight Tidehunter — a spell on this pays the first card in hand
        Ability(
            Trigger.ON_TARGETED_BY_SPELL,
            BuffHandMinionsEffect(attack=6, health=6, leftmost=True),
        ),
    ),
    "BG36_704": (  # Shamanic Tidecaller — a spell on a Murloc pays them all
        # A board watcher, not a card watching spells cast at itself: the spell
        # may land on any Murloc. The payload reaches the hand, which flat stats
        # on the watcher cannot.
        Ability(
            Trigger.AURA,
            BuffOnSpellCastOnTribeEffect(
                tribe=Race.MURLOC,
                effect=BuffHandMinionsEffect(
                    attack=3, health=3, tribe=Race.MURLOC, also_board=True
                ),
            ),
        ),
    ),
    "BG33_318": (  # Bile Spitter — Venomous; Rally: another friendly Murloc too
        Ability(
            Trigger.ON_ATTACK,
            GrantKeywordRandomFriendly(Keyword.VENOMOUS, filter_race=Race.MURLOC),
        ),
    ),
    "BG34_142": (  # Costume Enthusiast — SoC: the biggest Attack waiting in hand
        Ability(Trigger.ON_START_OF_COMBAT, GainStatsFromHandEffect(highest_attack_only=True)),
    ),
    "BG26_354": (  # Choral Mrrrglr — SoC: everything waiting in hand
        Ability(Trigger.ON_START_OF_COMBAT, GainStatsFromHandEffect()),
    ),
    "BG34_145": (  # Futurefin — end of turn: its stats to the first card in hand
        Ability(Trigger.ON_TURN_END, GiveOwnStatsToHandEffect()),
    ),
    "BG35_142": (  # Cousin Errgl — end of turn: a Mrrglton, either parent
        Ability(
            Trigger.ON_TURN_END,
            AddRandomCardToHandEffect(card_ids=("BG35_140", "BG35_141")),
        ),
    ),
    "BG33_893": (  # Primitive Painter — a small card played pays your Murlocs
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffMatching(
                BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.MURLOC, attack=2, health=2
            ),
            filter_max_tier=3,
        ),
    ),
    "BGS_020": (  # Primalfin Lookout — with another Murloc out, Discover one
        Ability(
            Trigger.ON_PLACE,
            DiscoverTribeEffect(tribe=Race.MURLOC, repeats=1),
            condition=Condition(ConditionKind.OTHER_TRIBE_ON_BOARD, Race.MURLOC),
        ),
    ),
    # ---------------------------------------------------- the Dragon family
    "BG29_813": (  # Persistent Poet — its neighbours keep what the fight gave them
        Ability(
            Trigger.AURA,
            KeepCombatGainsEffect(adjacent=True, tribe=Race.DRAGON),
        ),
    ),
    "BG36_245": (  # Runic Arcanist — Start of Combat: cast Shiny Ring
        Ability(Trigger.ON_START_OF_COMBAT, CastSpellAtEffect(card_id="BG28_168")),
    ),
    "BG36_241": (  # Crimson Vindicator — Rally: cast Mighty Dragonbreath
        Ability(Trigger.ON_ATTACK, CastSpellAtEffect(card_id="BG36_246")),
    ),
    "BG34_633": (  # Draconic Warden — Battlecry *and* Deathrattle: a Chromadrake
        Ability(
            Trigger.ON_PLACE,
            AddRandomCardToHandEffect(
                card_ids=("BG34_634t", "BG34_635t", "BG34_636t", "BG34_637t", "BG34_638t")
            ),
        ),
        Ability(
            Trigger.ON_DEATH,
            AddRandomCardToHandEffect(
                card_ids=("BG34_634t", "BG34_635t", "BG34_636t", "BG34_637t", "BG34_638t")
            ),
        ),
    ),
    "BG32_820": (  # Firescale Hoarder — Battlecry *and* Deathrattle: a Shiny Ring
        Ability(Trigger.ON_PLACE, AddTavernSpellToHandEffect(card_id="BG28_168")),
        Ability(Trigger.ON_DEATH, AddTavernSpellToHandEffect(card_id="BG28_168")),
    ),
    "BGS_041": (  # Kalecgos — a Battlecry triggered pays your Dragons
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            BuffMatching(
                BuffTarget.FRIENDLY_OF_TRIBE,
                tribe=Race.DRAGON,
                attack=2,
                health=2,
                requires_placed_battlecry=True,
            ),
        ),
    ),
    "BG32_822": (  # Fire-forged Evoker — SoC: your Dragons +2/+1, improving
        Ability(
            Trigger.ON_START_OF_COMBAT,
            RepeatPerCountEffect(
                counter="tavern_spells_cast:*",
                effect=BuffMatching(
                    BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.DRAGON, attack=2, health=1
                ),
            ),
        ),
    ),
    "BG28_595": (  # Ignition Specialist — end of turn: two random Tavern spells
        Ability(Trigger.ON_TURN_END, AddRandomTavernSpellToHandEffect(count=2)),
    ),
    "BG24_004": (  # Warpwing — takes nothing back from what it swings at
        Ability(Trigger.AURA, ImmuneWhileAttackingEffect()),
    ),
    "BG27_017": (  # Obsidian Ravager — Rally: its Attack to the target and beside
        Ability(Trigger.ON_ATTACK, DamageFromOwnAttackEffect(include_adjacent=True)),
    ),
    # The Golden reaches "its **neighbors**" where the plain printing reaches
    # "**an** adjacent minion" — a word, not a number, so nothing derives it.
    "BG27_017_G": (
        Ability(
            Trigger.ON_ATTACK,
            DamageFromOwnAttackEffect(include_adjacent=True, adjacent_count=2),
        ),
    ),
    # -------------------------------------------------- the Quilboar family
    "BG33_883": (  # Razorfen Vineweaver — Rally: three Gems on itself, for keeps
        Ability(
            Trigger.ON_ATTACK,
            PlayBloodGemsEffect(
                target=BloodGemTarget.SELF, count=3, permanent=True
            ),
        ),
    ),
    "BG36_510": (  # Vigilant Bristlemane — a spell on this gems its neighbours
        Ability(
            Trigger.ON_TARGETED_BY_SPELL,
            PlayBloodGemsEffect(target=BloodGemTarget.ADJACENT, count=1),
        ),
    ),
    "BG33_885": (  # Sanguine Refiner — Rally: your Blood Gems give an extra +1/+1
        Ability(Trigger.ON_ATTACK, IncreaseBloodGemBonusEffect(attack=1, health=1)),
    ),
    "BG23_017": (  # Sanguine Champion — Battlecry *and* Deathrattle, the same
        Ability(Trigger.ON_PLACE, IncreaseBloodGemBonusEffect(attack=1, health=1)),
        Ability(Trigger.ON_DEATH, IncreaseBloodGemBonusEffect(attack=1, health=1)),
    ),
    "BG31_323": (  # Turbo Hogrider — a Choose One played gems your other Quilboar
        Ability(
            Trigger.ON_CHOOSE_ONE_PLAYED,
            PlayBloodGemsEffect(
                target=BloodGemTarget.ALL_FRIENDLY_QUILBOAR, count=1
            ),
        ),
    ),
    "BG31_327": (  # Thorned Trailblazer — one Choose One each turn takes both halves
        # Each turn, not once: the card prints "(1 left!)", which is a charge
        # that refills rather than a battlecry.
        Ability(Trigger.ON_TURN_START, GrantCombinedChooseOneEffect(count=1)),
    ),
    "BG36_332": (  # Snare Trapper — Choose One: a Quilboar, or a bigger purse
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=AddRandomMinionToHandEffect(tribe=Race.QUILBOAR),
                second=RaiseGoldCapEffect(amount=1),
            ),
        ),
    ),
    "BG28_633": (  # Felboar — every three spells, it eats one off the counter
        Ability(
            Trigger.AURA,
            SpellsCastResponseEffect(
                threshold=3,
                effect=ConsumeTavernMinionEffect(eater_is_source=True),
            ),
        ),
    ),
    "BG36_333": (  # Jailbird Juggernaut — Rally: a Golem made of its own Gems
        Ability(
            Trigger.ON_ATTACK,
            SummonGemGolemEffect(
                # A Blood Golem token, not a copy of the Juggernaut: pointing
                # at its own card id put a second tier-5 Quilboar on the board,
                # which every "all your Quilboar" effect then counted.
                token_id="BG30_MagicItem_442t"
            ),
        ),
    ),
    "BG36_341": (  # Veteran Brigand — Choose One: Gems everywhere, or Barrages
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=PlayBloodGemsEffect(
                    target=BloodGemTarget.ALL_FRIENDLY, count=3
                ),
                second=AddTavernSpellToHandEffect(card_id="BG34_689", count=3),
            ),
        ),
    ),
    "BG36_331": (  # Bramble Tunneler — Rally: a random Choose One card
        Ability(
            Trigger.ON_ATTACK,
            AddRandomCardToHandEffect(
                card_ids=(
                    "BG31_893",  # Gem Day
                    "BG31_880",  # Alliance Flag
                    "BG31_881",  # Time Management
                    "BG31_890",  # Boundless Potential
                    "BG31_886",  # Forest's Bounty
                    "BG31_884",  # The Road Less Traveled
                )
            ),
        ),
    ),
    # ------------------------------------------------------ the Beast family
    # A tribe that fights by arriving: most of these pay when a minion is
    # summoned, which is why the shared summon listener grew rather than five
    # cards each growing their own.
    "BG26_802": (  # Banana Slamma — a Beast summoned in combat has its Attack doubled
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            MultiplySummonedAttackEffect(tribe=Race.BEAST, factor=2),
        ),
    ),
    "BG34_322": (  # Stalwart Kodo — a summon gets this minion's maximum stats, 3x
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            GiveOwnStatsToSummonedEffect(charges=3, factor=1),
        ),
    ),
    "BG35_602": (  # Lurking Leviathan — +2 Attack to a summoned Beast, improving
        Ability(
            Trigger.ON_FRIENDLY_MINION_SUMMONED,
            BuffSummonedIfRace(tribe=Race.BEAST, attack=2, improves=True),
        ),
    ),
    "BG36_206": (  # Snarky Shark — sold, it refreshes the tavern with a Fishbait
        Ability(
            Trigger.ON_SELL,
            PlaceFishbaitEffect(refresh=True, auto_attack=True),
        ),
    ),
    "BG36_210": (  # Hoarding Hyena — Rally: summon a Tasty Lobster
        Ability(Trigger.ON_ATTACK, SummonEffect(token_id="BG36_202", count=1)),
    ),
    "BG35_604": (  # Sewer Lord — Deathrattle: two Sewer Rats, which leave Half-Shells
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG19_010", count=2)),
    ),
    "BG19_010": (  # Sewer Rat (token) — Deathrattle: a 2/3 Taunt Half-Shell
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG19_010t", count=1)),
    ),
    # The Golden Rat is written out because its own golden text names the
    # Half-Shell by stats rather than as Golden, so nothing derives it: without
    # this the implicit rule would give it two plain Half-Shells.
    "BG19_010_G": (
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG19_010_Gt", count=1)),
    ),
    "BG31_809": (  # Turquoise Skitterer — Deathrattle: Beetles +5/+5 this game, and one
        Ability(
            Trigger.ON_DEATH,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.CARD, scope_key="BG28_603t", attack=5, health=5
            ),
        ),
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG28_603t", count=1)),
    ),
    "BG36_209": (  # Ravaging Scorpid — every friendly attack raises the Beetles
        Ability(
            Trigger.ON_FRIENDLY_ATTACK,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.CARD, scope_key="BG28_603t", attack=3, health=3
            ),
        ),
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG28_603t", count=1)),
    ),
    "BG36_208": (  # Deathstrider — a Rally minion's swing fires your left-most Deathrattle
        Ability(
            Trigger.ON_FRIENDLY_ATTACK,
            TriggerLeftmostDeathrattleEffect(repeats=1),
            filter_subject_rally=True,
        ),
    ),
    "BGS_018": (  # Goldrinn, the Great Wolf — Deathrattle: your Beasts +8/+8
        Ability(
            Trigger.ON_DEATH,
            BuffMatching(
                target=BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.BEAST, attack=8, health=8
            ),
        ),
    ),
    # ----------------------------------------------------- the Undead family
    # Two threads: bodies traded away for something (a Discover, stats, a
    # tribe-wide bonus) and bodies that come back. The destroy is one effect
    # with the payout as a field, and Reborn finally has a trigger.
    "BG32_340": (  # Maw Caster — destroy a friendly Undead to Discover an Undead
        Ability(
            Trigger.ON_PLACE,
            DestroyFriendlyEffect(
                filter_race=Race.UNDEAD,
                get_copy=False,
                exclude_self=True,
                then=DiscoverTribeEffect(tribe=Race.UNDEAD, repeats=1),
            ),
        ),
    ),
    "BG36_511": (  # Dead Bellringer — Reborn onto an Undead, then eat it for +4/+4
        Ability(
            Trigger.ON_ACTIVATE,
            DestroyFriendlyEffect(
                filter_race=Race.UNDEAD,
                get_copy=False,
                exclude_self=True,
                grant_keyword=Keyword.REBORN,
                then=BuffSelf(attack=4, health=4),
            ),
            activate_cost=1,
        ),
    ),
    "BG34_690": (  # Plaguerunner — Undead +2 Attack this game, +4 out of combat
        Ability(
            Trigger.ON_DEATH,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.TRIBE,
                scope_key=Race.UNDEAD,
                attack=2,
                attack_outside_combat=4,
            ),
        ),
    ),
    "BG34_692": (  # Forsaken Weaver — a Tavern spell raises your Undead for good
        Ability(
            Trigger.ON_TAVERN_SPELL_CAST,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.TRIBE, scope_key=Race.UNDEAD, attack=2
            ),
        ),
    ),
    "BG36_514": (  # Barrier Banshee — a friendly Reborn pays it a shield and +7/+7
        Ability(
            Trigger.ON_FRIENDLY_REBORN,
            BuffSelf(attack=7, health=7, keyword=Keyword.SHIELD),
        ),
    ),
    "BG36_515": (  # Snazzy Phantom — the reborn minion's Attack, onto your right-most Undead
        Ability(
            Trigger.ON_FRIENDLY_REBORN,
            BuffFromSubjectAttackEffect(tribe=Race.UNDEAD, rightmost=True, factor=1),
        ),
    ),
    "BG32_324": (  # Drustfallen Butcher — Avenge (3): get a Butchering
        Ability(
            Trigger.ON_FRIENDLY_MINION_DIED,
            AvengeEffect(count=3, effect=AddTavernSpellToHandEffect(card_id="BG28_604")),
        ),
    ),
    "BG31_835": (  # Deathly Striker — Avenge (4): get an Undead; give it back on death
        Ability(
            Trigger.ON_FRIENDLY_MINION_DIED,
            AvengeEffect(count=4, effect=AddRandomMinionToHandEffect(tribe=Race.UNDEAD)),
        ),
        # "Summon *it*" — the card it fetched. Nothing links a hand card back to
        # the body that fetched it, so this summons the best Undead in hand,
        # which is that card whenever the seat is not hoarding others.
        Ability(Trigger.ON_DEATH, SummonBestFromHandEffect(filter_race=Race.UNDEAD)),
    ),
    "BG25_009": (  # Eternal Summoner — Reborn; Deathrattle: an Eternal Knight
        Ability(Trigger.ON_DEATH, SummonEffect(token_id="BG25_008", count=1)),
    ),
    "BG31_999": (  # Stitched Salvager — eats its left neighbour, gives it back on death
        Ability(
            Trigger.ON_START_OF_COMBAT,
            DevourNeighbourEffect(adjacent=False, exclude_same_card=True),
        ),
        Ability(Trigger.ON_DEATH, SummonStashedEffect()),
    ),
    # The Golden eats *both* neighbours, which is a flag rather than a bigger
    # number, so nothing derives it.
    "BG31_999_G": (
        Ability(
            Trigger.ON_START_OF_COMBAT,
            DevourNeighbourEffect(adjacent=True, exclude_same_card=True),
        ),
        Ability(Trigger.ON_DEATH, SummonStashedEffect()),
    ),
    # -------------------------------------------------- the tribeless cards
    # No tribe to hold them together, so what they share is shape: four say
    # "your X happen more than once", two read the minion a Rally is swinging
    # at, and the rest are one of a kind.
    "BG_LOE_077": (  # Brann Bronzebeard — your Battlecries trigger twice
        Ability(Trigger.AURA, Multiplier(MultiplierKind.BATTLECRY, factor=2)),
    ),
    "BG25_354": (  # Titus Rivendare — your Deathrattles trigger an extra time
        Ability(Trigger.AURA, Multiplier(MultiplierKind.DEATHRATTLE, factor=2)),
    ),
    "BG26_ICC_901": (  # Drakkari Enchanter — your end of turn effects trigger twice
        Ability(Trigger.AURA, Multiplier(MultiplierKind.END_OF_TURN, factor=2)),
    ),
    "BG35_883": (  # Balinda Stonehearth — spells aimed at a friendly cast twice
        Ability(Trigger.AURA, Multiplier(MultiplierKind.TARGETED_SPELL, factor=2)),
    ),
    "BG34_604": (  # Heroic Underdog — Rally: gain the target's Attack
        Ability(Trigger.ON_ATTACK, GainTargetAttackEffect(factor=1)),
    ),
    "BG25_016": (  # Sin'dorei Straight Shot — Rally: strip the target's Reborn and Taunt
        Ability(
            Trigger.ON_ATTACK,
            StripKeywordsFromTargetEffect(keywords=(Keyword.REBORN, Keyword.TAUNT)),
        ),
    ),
    "BG23_318": (  # Leeroy the Reckless — Deathrattle: destroy whatever killed this
        Ability(Trigger.ON_DEATH, DestroyKillerEffect()),
    ),
    "BGS_012": (  # Kangor's Apprentice — Deathrattle: plain copies of your first 2 dead Mechs
        Ability(Trigger.ON_DEATH, SummonFirstDeadFriendlyMechsThisCombat(count=2)),
    ),
    "BGS_104": (  # Nomi, Kitchen Nightmare — the tavern's Elementals, for good
        Ability(
            Trigger.AFTER_FRIENDLY_MINION_PLACED,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.SHOP, scope_key=Race.ELEMENTAL, attack=4, health=4
            ),
            filter_race=Race.ELEMENTAL,
        ),
    ),
    "BG32_341": (  # Humon'gozz — while it stands, your Tavern spells give an extra +1/+2
        Ability(Trigger.AURA, IncreaseTavernSpellBonusEffect(attack=1, health=2)),
    ),
    "BG35_123": (  # Cataclysmic Harbinger — end of turn, a copy of your last spell
        Ability(Trigger.ON_TURN_END, CopyLastTavernSpellEffect()),
    ),
    "BG28_550": (  # Rodeo Performer — Battlecry: Discover a Tavern spell
        Ability(Trigger.ON_PLACE, DiscoverTavernSpellEffect()),
    ),
    "BG34_319": (  # Highkeeper Ra — Battlecry, Deathrattle *and* Rally: a Tier 6 minion
        Ability(Trigger.ON_PLACE, AddRandomMinionToHandEffect(tier=6)),
        Ability(Trigger.ON_DEATH, AddRandomMinionToHandEffect(tier=6)),
        Ability(Trigger.ON_ATTACK, AddRandomMinionToHandEffect(tier=6)),
    ),
    "BG36_356": (  # Tyrael — Activate (2): set another minion's stats to 40/40
        Ability(
            Trigger.ON_ACTIVATE,
            SetStatsEffect(attack=40, health=40, exclude_self=True),
            activate_cost=2,
        ),
    ),
    "BG24_018": (  # Tortollan Blue Shell — worth 5 Gold only after a loss
        Ability(
            Trigger.AURA,
            SellValueEffect(amount=5),
            condition=Condition(ConditionKind.LAST_COMBAT_WON, negate=True),
        ),
    ),
    # ------------------------------------------------- the rest of the Mechs
    # Two of them are the same sentence other tribes already say — a Tavern
    # spell bonus, a spell cast on a friendly — and the other four are the
    # Magnetize machinery reached from somewhere new.
    "BG35_341": (  # Enchanted Sentinel — Magnetic; your Tavern spells give +1/+1
        Ability(Trigger.AURA, IncreaseTavernSpellBonusEffect(attack=1, health=1)),
    ),
    "BG28_741": (  # Charging Czarina — a Tavern spell pays your Divine Shields
        Ability(
            Trigger.ON_TAVERN_SPELL_CAST,
            BuffMatching(
                target=BuffTarget.FRIENDLY_WITH_KEYWORD,
                keyword=Keyword.SHIELD,
                attack=4,
            ),
        ),
    ),
    "BG36_853": (  # Glambot — a spell cast on a Mech welds a Satellite to it
        Ability(
            Trigger.AURA,
            BuffOnSpellCastOnTribeEffect(
                tribe=Race.MECHANICAL,
                effect=MagnetizeTokenEffect(token_id="BG31_171t", repeats=1),
            ),
        ),
    ),
    "BG26_148": (  # Scrap Scraper — Deathrattle: a random *Magnetic* Mech
        Ability(
            Trigger.ON_DEATH,
            AddRandomMinionToHandEffect(
                tribe=Race.MECHANICAL, keyword=Keyword.MAGNETIC, count=1
            ),
        ),
    ),
    "BG29_503": (  # Clunker Junker — Discover a Mech and weld it to a friendly one
        Ability(
            Trigger.ON_PLACE,
            DiscoverTribeEffect(tribe=Race.MECHANICAL, magnetize_onto_target=True),
        ),
    ),
    "BG35_342": (  # Falling Sky Golem — +4/+2 per Deathrattle triggered this game
        Ability(
            Trigger.AURA,
            SelfBonusPerGameCount(
                # A seat-wide tally rather than one this card keeps about
                # itself, which is what the "*" subject says.
                counter="deathrattles_triggered",
                subject="*",
                attack_per=4,
                health_per=2,
                count_self=True,
            ),
        ),
    ),
    # ---------------------------------------------------- the Amalgam family
    # All three say the same sentence as each other or as a spell: "a friendly
    # minion of each type", which is now what an empty tribe list means.
    "BG34_320": (  # The Last One Standing — Rally: one of each type +12/+12 for keeps
        Ability(
            Trigger.ON_ATTACK,
            BuffOnePerListedTribeFriendly(attack=12, health=12, permanent=True),
        ),
    ),
    "BG32_111": (  # Nightmare Par-tea Guest — Battlecry *and* Deathrattle: a Tea Set
        Ability(Trigger.ON_PLACE, AddTavernSpellToHandEffect(card_id="BG28_888")),
        Ability(Trigger.ON_DEATH, AddTavernSpellToHandEffect(card_id="BG28_888")),
    ),
    "BG36_640": (  # Gatekeeper Amalgam — a spell cast on it makes it cast a Tea Set
        Ability(
            Trigger.ON_TARGETED_BY_SPELL,
            CastSpellAtEffect(card_id="BG28_888", untargeted=True),
        ),
    ),
    "BG27_514": (  # Sea Witch Zar'jira — Spellcraft: copy a minion off the counter
        Ability(
            # ON_TURN_START like every other Spellcraft Naga: the spell comes
            # back each turn, and ON_PLACE handed it over exactly once ever.
            Trigger.ON_TURN_START,
            CreateSpellcraftSpellEffect(
                buff=CopyTavernMinionEffect(count=1, exclude_card_id="BG27_514"),
                card_id="BG27_514t",
                name="Siren's Song",
            ),
        ),
    ),
    "BG26_175": (  # Elemental of Surprise — completes a pair of any Elemental
        Ability(Trigger.AURA, TriplesWithAnyOfTribeEffect(tribe=Race.ELEMENTAL)),
    ),
    # ------------------------------------------- the last of the deferred six
    # Each of these was deferred for one missing piece, and the piece has since
    # been built for something else: the tribe Discover for Maw Caster, the
    # tavern Rally for the Fishbait, the refilling countdown for Felboar.
    "BG26_525": (  # Imposing Percussionist — Discover a Demon, and pay its Tier
        Ability(Trigger.ON_PLACE, DiscoverTribeEffect(tribe=Race.DEMON, repeats=1)),
        Ability(Trigger.ON_DISCOVERED, DealHeroDamagePerTierEffect()),
    ),
    "BG36_621": (  # Deft Deserter — the whole tavern +8/+8 and a keyword each
        Ability(
            Trigger.ON_ACTIVATE,
            BuffAllShopOffersEffect(
                attack=8,
                health=8,
                keyword_choices=(Keyword.TAUNT, Keyword.SHIELD, Keyword.WINDFURY),
            ),
            activate_cost=1,
        ),
    ),
    "BG36_243": (  # Sky-hatch Runaway — Activate: trigger a friendly's Rally
        Ability(
            Trigger.ON_ACTIVATE,
            RetriggerFriendlyAbilityEffect(trigger=Trigger.ON_ATTACK),
            activate_cost=1,
        ),
    ),
    "BG36_701": (  # Kelp Keeper — Activate: trigger a friendly's Battlecry
        Ability(
            Trigger.ON_ACTIVATE,
            RetriggerFriendlyAbilityEffect(trigger=Trigger.ON_PLACE),
            activate_cost=1,
        ),
    ),
    "BG36_352": (  # Unbound Tempest — every 3 Elementals, take the tavern's best
        Ability(
            Trigger.AURA,
            ElementalsPlayedResponseEffect(
                threshold=3, effect=GainStatsFromTavernEffect(highest_health=True)
            ),
        ),
    ),
    "BG26_505": (  # Zesty Shaker — a Spellcraft spell cast on it comes back
        Ability(
            Trigger.ON_TARGETED_BY_SPELL,
            CopyTargetingSpellEffect(count=1, once_per_turn=True),
        ),
    ),
}


#: Tavern spells, bound the same way minions are. A spell's whole text is its
#: battlecry, so every ability here hangs off ``Trigger.ON_PLACE`` — it fires
#: when the card is cast, which for a spell is the only thing it ever does.
SPELL_EFFECTS: Dict[str, Tuple[Ability, ...]] = {
    "BG28_168": (  # Shiny Ring — give your minions +1/+1
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(BuffTarget.ALL_FRIENDLY, attack=1, health=1),
        ),
    ),
    "BG36_246": (  # Mighty Dragonbreath — everyone, then Dragons, then shields
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(BuffTarget.ALL_FRIENDLY, attack=1, health=1),
        ),
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(
                BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.DRAGON, attack=1, health=1
            ),
        ),
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(
                BuffTarget.FRIENDLY_WITH_KEYWORD, keyword=Keyword.SHIELD,
                attack=1, health=1,
            ),
        ),
    ),

    # ------------------------------------------------------------- Bounties
    "BG33_811": (  # Healthy Bounty — four friendly minions +4 Health
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(BuffTarget.ALL_FRIENDLY, attack=0, health=4, limit=4),
        ),
    ),
    "BG33_812": (  # Hostile Bounty — four friendly minions +4 Attack
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(BuffTarget.ALL_FRIENDLY, attack=4, health=0, limit=4),
        ),
    ),
    "BG33_813": (  # Selfish Bounty — your left-most minion +6/+6
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(
                BuffTarget.ALL_FRIENDLY,
                attack=6,
                health=6,
                limit=1,
                leftmost=True,  # "your **left-most** minion"
            ),
        ),
    ),
    "BG33_815": (  # Wealthy Bounty — gain 2 Gold
        Ability(Trigger.ON_PLACE, GainGoldThisTurnEffect(amount=2)),
    ),
    "BG33_814": (  # Friendly Bounty — a random minion of your most common type
        Ability(Trigger.ON_PLACE, AddRandomMinionOfCommonTribeEffect()),
    ),
    "BG31_886": (  # Forest's Bounty — Choose One
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=BuffTargetFriendlyBattlecry(
                    attack=12, health=12, exclude_self=False
                ),
                second=BuffMatching(BuffTarget.ALL_FRIENDLY, attack=2, health=2),
            ),
        ),
    ),

    "BG34_444": (  # Easterly Winds — every roll from now on buffs the tavern
        Ability(Trigger.ON_PLACE, BuffShopOnEveryRefreshEffect(attack=6, health=6)),
    ),

    "BG28_845": (  # Natural Blessing — everyone sharing the target's type +3/+3
        Ability(Trigger.ON_PLACE, BuffSharedTribeEffect(attack=3, health=3)),
    ),

    # ---------------------------------------------------- spells handed out
    # Not sold in the tavern; these arrive from a minion that names them.
    "EBG_Spell_014": (  # Pointy Arrow — give a minion +4 Attack
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(attack=4, health=0, exclude_self=False),
        ),
    ),
    "BG27_002t": (  # Slimy Shield — give a minion +1/+1 and Taunt
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(
                attack=1, health=1, exclude_self=False, grant_keyword=Keyword.TAUNT
            ),
        ),
    ),
    "BG31_893": (  # Gem Day — Choose One: your Blood Gems give +1 Attack; or +1 Health
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=IncreaseBloodGemBonusEffect(attack=1, health=0),
                second=IncreaseBloodGemBonusEffect(attack=0, health=1),
            ),
        ),
    ),

    # ------------------------------------------------------------------ tier 1
    "BG28_810": (  # Tavern Coin — Gain 1 Gold
        Ability(Trigger.ON_PLACE, GainGoldThisTurnEffect(amount=1)),
    ),
    # ------------------------------------ tavern spells the tavern can offer
    # Every one of these was offerable and inert. They are bindings, not new
    # mechanics: the effect each needs was built for a minion that says the
    # same sentence.
    "BG28_882": (  # Contracted Corpse — Discover a Deathrattle minion
        Ability(Trigger.ON_PLACE, DiscoverTribeEffect(require_deathrattle=True)),
    ),
    "BG28_GIL_836": (  # Hired Headhunter — Discover a Battlecry minion
        Ability(Trigger.ON_PLACE, DiscoverTribeEffect(require_battlecry=True)),
    ),
    "BG28_521": (  # Planar Telescope — Discover a minion of your most common type
        Ability(Trigger.ON_PLACE, DiscoverTribeEffect(most_common_tribe=True)),
    ),
    "BG34_330": (  # Search Through Time — a minion of *your* Tier, held a turn
        Ability(
            Trigger.ON_PLACE,
            DiscoverTribeEffect(exact_tier=True, lock_turns=1),
        ),
    ),
    "BG28_500": (  # Armor Stash — set your Armor to 5
        Ability(Trigger.ON_PLACE, SetArmorEffect(amount=5)),
    ),
    "BG28_825": (  # Defender's Rites — give a minion +7/+7 and Taunt
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(
                attack=7, health=7, exclude_self=False, grant_keyword=Keyword.TAUNT
            ),
        ),
    ),
    "BG28_507": (  # Sacred Gift — give a minion Divine Shield
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(
                attack=0, health=0, exclude_self=False, grant_keyword=Keyword.SHIELD
            ),
        ),
    ),
    "BG28_838": (  # Perfect Vision — set a minion's stats to 20/20
        Ability(Trigger.ON_PLACE, SetStatsEffect(attack=20, health=20, exclude_self=False)),
    ),
    "BG35_951": (  # Might of Stormwind — four random friendly minions +1/+2
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(target=BuffTarget.ALL_FRIENDLY, attack=1, health=2, limit=4),
        ),
    ),
    "BG33_817": (  # Sanctify — your Divine Shields +6 Attack
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(
                target=BuffTarget.FRIENDLY_WITH_KEYWORD,
                keyword=Keyword.SHIELD,
                attack=6,
            ),
        ),
    ),
    "BG35_922": (  # Queen's Command — your minions +2/+2, and Naga another +2/+2
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(target=BuffTarget.ALL_FRIENDLY, attack=2, health=2),
        ),
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(
                target=BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.NAGA, attack=2, health=2
            ),
        ),
    ),
    "BG28_169": (  # Azerite Empowerment — your minions +2/+2, twice
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(target=BuffTarget.ALL_FRIENDLY, attack=2, health=2),
        ),
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(target=BuffTarget.ALL_FRIENDLY, attack=2, health=2),
        ),
    ),
    "BG28_805": (  # Strike Oil — raise your maximum Gold by 1
        Ability(Trigger.ON_PLACE, RaiseGoldCapEffect(amount=1)),
    ),
    "BG28_827": (  # Leaf Through the Pages — two free Refreshes
        Ability(Trigger.ON_PLACE, SetNextRollCostEffect(cost=0, uses=2)),
    ),
    "BG28_800": (  # Careful Investment — 2 Gold next turn
        Ability(Trigger.ON_PLACE, GainGoldNextTurnEffect(amount=2)),
    ),
    "BG28_886": (  # Staff of Enrichment — the tavern's minions +2/+2 this game
        Ability(
            Trigger.ON_PLACE,
            RaiseStandingBonusEffect(scope_kind=ScopeKind.SHOP, attack=2, health=2),
        ),
    ),
    "BG36_884": (  # Weapons Forge — three Pointy Arrows
        Ability(
            Trigger.ON_PLACE,
            AddTavernSpellToHandEffect(card_id="EBG_Spell_014", count=3),
        ),
    ),
    "BG31_896": (  # Hallowed Ritual — Discover a Tier 7 minion
        Ability(Trigger.ON_PLACE, DiscoverMinionAtTierEffect(tier=7)),
    ),
    "BG34_888": (  # Tomb Turning — an Undead, dead if you play it this turn
        Ability(
            Trigger.ON_PLACE,
            DiscoverTribeEffect(
                tribe=Race.UNDEAD, dies_if_played_this_turn=True
            ),
        ),
    ),
    "BG28_607": (  # Corrupted Cupcakes — a Demon eats three off the counter
        Ability(
            Trigger.ON_PLACE,
            ConsumeTavernMinionEffect(
                filter_race=Race.DEMON, count=3, eater_is_source=True
            ),
        ),
    ),
    "BG36_624": (  # Repair Job — give a minion +4/+8
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(attack=4, health=8, exclude_self=False),
        ),
    ),
    "BG35_149": (  # Deepwater Clan — a minion +2/+2, and your Murlocs +2/+2
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(attack=2, health=2, exclude_self=False),
        ),
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(
                target=BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.MURLOC, attack=2, health=2
            ),
        ),
    ),
    "BG28_604": (  # Butchering — eat an Undead, and the rest gain for the game
        Ability(
            Trigger.ON_PLACE,
            DestroyFriendlyEffect(
                filter_race=Race.UNDEAD,
                get_copy=False,
                then=RaiseStandingBonusEffect(
                    scope_kind=ScopeKind.TRIBE, scope_key=Race.UNDEAD, attack=5
                ),
            ),
        ),
    ),
    "BG36_880": (  # Methodical Madness — a Demon eats two, stats and keywords
        Ability(
            Trigger.ON_PLACE,
            ConsumeTavernMinionEffect(
                filter_race=Race.DEMON,
                count=2,
                eater_is_source=True,
                gain_keywords=True,
            ),
        ),
    ),
    "BG28_518": (  # Chef's Choice — a different minion of the target's own type
        Ability(Trigger.ON_PLACE, AddSharedTribeMinionEffect(count=1)),
    ),
    "BG28_503": (  # Fortify — give a minion +3 Health and Taunt
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(
                attack=0, health=3, exclude_self=False, grant_keyword=Keyword.TAUNT
            ),
        ),
    ),
    "BG28_897": (  # Tavern Dish Banana — give a minion +2/+2
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(attack=2, health=2, exclude_self=False),
        ),
    ),
    "BG28_888": (  # Misplaced Tea Set — a friendly minion of each type +2/+2
        Ability(Trigger.ON_PLACE, BuffOnePerListedTribeFriendly(attack=2, health=2)),
    ),
    "BG28_966": (  # Them Apples — give minions in the Tavern +1/+2
        Ability(Trigger.ON_PLACE, BuffAllShopOffersEffect(attack=1, health=2)),
    ),
    "BG28_504": (  # Recruit a Trainee — get a random Tier 1 minion
        Ability(Trigger.ON_PLACE, AddRandomMinionToHandEffect(tier=1)),
    ),
    "BG28_512": (  # Enchanted Lasso — steal a random minion from the Tavern
        Ability(Trigger.ON_PLACE, StealTavernMinionEffect()),
    ),
    "BG33_101": (  # A New Sprout — Discover a Tier 1 minion
        Ability(Trigger.ON_PLACE, DiscoverMinionAtTierEffect(tier=1)),
    ),
    "BG31_880": (  # Alliance Flag — Choose One: give a minion +3/+1; or +1/+3
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=BuffTargetFriendlyBattlecry(attack=3, health=1, exclude_self=False),
                second=BuffTargetFriendlyBattlecry(attack=1, health=3, exclude_self=False),
            ),
        ),
    ),
    "BG28_520": (  # Tricky Trousers — +1/+2 and Taunt, or Taunt off if it has it
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(
                attack=1,
                health=2,
                exclude_self=False,
                grant_keyword=Keyword.TAUNT,
                toggle_keyword=True,
            ),
        ),
    ),
    "BG32_815": (  # Shifting Tide — +1/+1 twice, and twice again on a Naga
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(
                attack=1,
                health=1,
                exclude_self=False,
                times=2,
                repeat_if_tribe=Race.NAGA,
            ),
        ),
    ),
    "BG35_912": (  # Eonar's Favor — the Tavern's minions of its type +3/+3 this game
        Ability(
            Trigger.ON_PLACE,
            RaiseStandingBonusEffect(
                scope_kind=ScopeKind.SHOP,
                scope_key_from_target=True,
                attack=3,
                health=3,
            ),
        ),
    ),
    "BG34_990": (  # Wave of Gold — your minions +3/+2, Golden ones another +3/+2
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(target=BuffTarget.ALL_FRIENDLY, attack=3, health=2),
        ),
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(target=BuffTarget.FRIENDLY_GOLDEN, attack=3, health=2),
        ),
    ),
    "BG34_272": (  # Menagerie Tableware — +3/+3 once per friendly minion type
        Ability(
            Trigger.ON_PLACE,
            BuffMatching(
                target=BuffTarget.ALL_FRIENDLY,
                attack=3,
                health=3,
                repeat_per_tribe_kind=True,
            ),
        ),
    ),
    "BG28_601": (  # Cloning Conch — a random Murloc, and a copy of that one
        Ability(
            Trigger.ON_PLACE,
            AddRandomMinionToHandEffect(tribe=Race.MURLOC, count=2, same_card=True),
        ),
    ),
    # ------------------------------------------------ the last few
    "EBG_Spell_037": (  # Unmasked Identity — Discover a new Hero Power
        Ability(Trigger.ON_PLACE, DiscoverHeroPowerEffect()),
    ),
    "BG28_603": (  # Boon of Beetles — two Taunt Beetles, twice, when there is room
        Ability(
            Trigger.ON_PLACE,
            SummonOnCombatSpaceEffect(
                token_id="BG28_603t", count=2, charges=2, grant_keyword=Keyword.TAUNT
            ),
        ),
    ),
    "BG31_819": (  # Temperature Shift — a Fire Baller and a Snow Baller
        Ability(Trigger.ON_PLACE, AddTokenToHandEffect(token_id="BG31_816")),
        Ability(Trigger.ON_PLACE, AddTokenToHandEffect(token_id="BG31_818")),
    ),
    "BG31_890": (  # Boundless Potential — Choose One: a minion, or a spell
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=DiscoverTribeEffect(exact_tier=True),
                second=DiscoverTavernSpellEffect(exact_tier=True),
            ),
        ),
    ),
    "BG28_698": (  # Gem Confiscation — two Gems, and its neighbours' as well
        Ability(Trigger.ON_PLACE, StealNeighbourBloodGemsEffect(gems=2)),
    ),
    "BG31_892": (  # Fandral's Fortune — a Choose One card, with both halves
        Ability(Trigger.ON_PLACE, DiscoverTribeEffect(require_choose_one=True)),
        Ability(Trigger.ON_PLACE, GrantCombinedChooseOneEffect(count=1)),
    ),
    "BG28_571": (  # Hasty Excavation — 1 Gold, bought with Health
        Ability(Trigger.AURA, PayInHealthEffect()),
        Ability(Trigger.ON_PLACE, GainGoldThisTurnEffect(amount=1)),
    ),

    # ------------------------------------- rolling the Tavern sideways
    "BG28_849": (  # Saloon's Finest — the counter shows Tavern spells
        Ability(Trigger.ON_PLACE, RefreshWithTavernSpellsEffect()),
    ),
    "EBG_Spell_038": (  # Lost Staff of Hamuul — a counter of one chosen type
        Ability(Trigger.ON_PLACE, RefreshWithTribeEffect()),
    ),
    "BG34_689": (  # Blood Gem Barrage — every later roll, Gems on the counter
        Ability(Trigger.ON_PLACE, BloodGemsOnEveryRefreshEffect(count=1)),
    ),

    # ------------------------------ promises that come due next turn
    "BG28_884": (  # Overconfidence — 3 Gold on a win, 1 on a tie
        Ability(
            Trigger.ON_PLACE,
            PromiseNextTurnEffect(
                effect=GainGoldThisTurnEffect(amount=3),
                condition=Condition(ConditionKind.LAST_COMBAT_WON),
            ),
        ),
        Ability(
            Trigger.ON_PLACE,
            PromiseNextTurnEffect(
                effect=GainGoldThisTurnEffect(amount=1),
                condition=Condition(ConditionKind.LAST_COMBAT_WON, tie=True),
            ),
        ),
    ),
    "BG36_883": (  # Winner's Bread — +2/+3 now, and +4/+6 next turn if you win
        Ability(
            Trigger.ON_PLACE,
            BuffTargetFriendlyBattlecry(attack=2, health=3, exclude_self=False),
        ),
        Ability(
            Trigger.ON_PLACE,
            PromiseNextTurnEffect(
                effect=BuffTargetFriendlyBattlecry(
                    attack=4, health=6, exclude_self=False
                ),
                condition=Condition(ConditionKind.LAST_COMBAT_WON),
            ),
        ),
    ),
    "BG31_881": (  # Time Management — Choose One: +2/+2 now, or twice next turn
        Ability(
            Trigger.ON_PLACE,
            ChooseOneEffect(
                first=BuffMatching(target=BuffTarget.ALL_FRIENDLY, attack=2, health=2),
                second=PromiseNextTurnEffect(
                    effect=BuffMatching(
                        target=BuffTarget.ALL_FRIENDLY, attack=2, health=2
                    ),
                    repeats=2,
                ),
            ),
        ),
    ),

    # --------------------------- Start of Combat, bought a turn early
    # A spell with no body of its own: the seat holds the promise from the
    # cast until the next fight reads it.
    "BG28_573": (  # Upper Hand — a random enemy minion drops to 1 Health
        Ability(Trigger.ON_START_OF_COMBAT, SetEnemyHealthEffect(health=1)),
    ),
    "BG34_889": (  # Brood of Nozdormu — your left-most minion's Attack, doubled
        Ability(Trigger.ON_START_OF_COMBAT, MultiplyFriendlyAttackEffect(factor=2)),
    ),
    "BG31_889": (  # Sharing is Caring — your left-most takes the nearest foe's stats
        Ability(Trigger.ON_START_OF_COMBAT, GainNearestEnemyStatsEffect()),
    ),

    # ------------------------------- a body traded for what it becomes
    "BG28_830": (  # Golden Touch — a random minion on the counter goes Golden
        Ability(Trigger.ON_PLACE, MakeFriendlyGoldenEffect(in_tavern=True)),
    ),
    "EBG_Spell_017": (  # Eyes of the Earth Mother — a friendly Tier 4 or below
        Ability(Trigger.ON_PLACE, MakeFriendlyGoldenEffect(max_tier=4)),
    ),
    "BG30_804": (  # Robust Evolution — a Tier higher, keeping its stats
        Ability(Trigger.ON_PLACE, TransformToHigherTierEffect()),
    ),
    "BG33_899": (  # Mounting Avalanche — sold, and your left-most Elemental grows
        Ability(
            Trigger.ON_PLACE,
            SellFriendlyForStatsEffect(recipient_tribe=Race.ELEMENTAL, leftmost=True),
        ),
    ),
    "EBG_Spell_032": (  # Channel the Devourer — sold, and a random friendly grows
        Ability(Trigger.ON_PLACE, SellFriendlyForStatsEffect()),
    ),
    "BG28_606": (  # Spitescale Special — three random Spellcraft spells
        Ability(
            Trigger.ON_PLACE,
            AddRandomTavernSpellToHandEffect(count=3, spellcraft=True),
        ),
    ),
}

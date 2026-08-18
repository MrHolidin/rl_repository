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
    ConsumeTavernMinionEffect,
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
    DestroyFriendlyForCopyEffect,
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
    BuffPlacedMinionEffect,
    BuffRandomHandMinionEffect,
    BumpSeatCounterEffect,
    BuffRandomOtherFriendlyCombat,
    BuffTarget,
    BuffSelf,
    BuffTargetFriendlyBattlecry,
    ChooseOneEffect,
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
    SummonEffect,
    SummonRandomMinionEffect,
    SummonSelfCopyFromHandEffect,
    Trigger,
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
        # The five Chromadrakes: out of the tavern pool this patch, still handed
        # over by Hired Mount, so the package has to carry them.
        "BG34_634t",
        "BG34_635t",
        "BG34_636t",
        "BG34_637t",
        "BG34_638t",
    }
)

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
            BuffMatching(BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.BEAST, attack=1, health=0),
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
        Ability(Trigger.ON_FRIENDLY_ATTACK, PlayBloodGemsOnAttackerEffect(count=1)),
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
                limit=1,  # "your left-most Dragon"
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
            DestroyFriendlyForCopyEffect(filter_race=Race.UNDEAD),
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
    "BG36_211": (  # Cage Gnawer — a friendly Beast attacks: your Beasts +2/+1
        Ability(
            Trigger.ON_FRIENDLY_ATTACK,
            BuffMatching(
                BuffTarget.FRIENDLY_OF_TRIBE, tribe=Race.BEAST, attack=2, health=1
            ),
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
    "BG27_080": (  # Motley Phalanx — Deathrattle: one friendly of each type +2/+2
        Ability(
            Trigger.ON_DEATH,
            BuffOnePerListedTribeFriendly(
                attack=2,
                health=2,
                tribes=(
                    Race.BEAST,
                    Race.DEMON,
                    Race.DRAGON,
                    Race.ELEMENTAL,
                    Race.MECHANICAL,
                    Race.MURLOC,
                    Race.NAGA,
                    Race.PIRATE,
                    Race.QUILBOAR,
                    Race.UNDEAD,
                ),
            ),
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
                attack=10, health=10, double_stats=True, once_per_turn=True
            ),
        ),
    ),
}


#: Tavern spells, bound the same way minions are. A spell's whole text is its
#: battlecry, so every ability here hangs off ``Trigger.ON_PLACE`` — it fires
#: when the card is cast, which for a spell is the only thing it ever does.
SPELL_EFFECTS: Dict[str, Tuple[Ability, ...]] = {
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
}

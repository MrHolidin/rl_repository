"""Hero catalog for HS Battlegrounds patch 36.2.0 (build 248348).

Loaded by :class:`src.bg_catalog.patch_context.PatchContext` the way
``bindings.py`` is. ``HEROES`` maps ``hero_id`` → :class:`Hero`;
``HERO_POOL_IDS`` is the assignable pool when the env runs with
``with_heroes=True``.

**Passive powers only, and only the ones that need no new cards.** The patch
ships 121 heroes: 56 whose power is passive and 65 the seat clicks. An active
power needs somewhere in the action space to be pressed, and that space is
frozen — so the active half waits on a decision this file does not make.

Of the 56 passives, these seventeen are the ones that need no card the package
does not already carry. What is left out is the rest: The Curator's Amalgam,
N'Zoth's Fish, Onyxia's Whelp, Sneed's Shredder, Jim Raynor's Battlecruiser,
the Timewarps, the trinket heroes, and the ones whose power is itself a
Discover of new Hero Powers.

Nine of the seventeen are the descriptors the engine already had; the other
eight brought a descriptor of their own, which is engine work rather than new
cards.

Keyed by card id rather than by a short slug, unlike the 2021 package: every
other id in this package is a card id, and it is what the catalog's
``heroes`` section can be joined on.

Health and armor are the catalog's own. Patchwerk's power reads "start the
game with 30 extra Health", which the data has already applied — its ``health``
is 60, not 30 plus something. Its armor field is absent, which the builder
records as 0 rather than null: absent armor is none of it.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

from src.bg_core.effects import Keyword
from src.bg_core.hero import (
    AttackOnKill,
    BuffCombatSummons,
    CastRandomSpellEachTurn,
    CombatAttackAuraAll,
    DiscoverAtTierOnGoldSpent,
    DiscoverHeroPowerOnTurn,
    EveryNthTavernSpellFree,
    FewerShopSlots,
    FreeBuyEachTurnAfterAttacks,
    FreezeShopEachTurn,
    ExtraShopDragon,
    FlatBuyCost,
    FlatRefreshCost,
    FreeFirstRefreshEachTurn,
    GoldNextTurnOnSell,
    GoldOnBuyTribe,
    GoldOnUpgrade,
    Hero,
    OnAttacksAddCardToHand,
    OnNthBuyAddCardToHand,
    OnNthDeathAddRaceToHand,
    OnNthSellAddRaceToHand,
    OnRefreshCopyHighestTier,
    OnRefreshGrantBonusKeyword,
    OnTiersBoughtAddCardToHand,
    ShopStatBuffPerBuys,
    SkipTurnsThenDiscover,
    StartOfCombatBuffEnds,
    StartOfCombatBuffOnePerTribe,
    TavernSpellBonusPerTurns,
    StartOfCombatGrantLeftmost,
    SummonCopyWhenSpace,
    UpgradeCostSurcharge,
    UpgradeDiscountPerElementals,
)
from src.bg_core.minion import Race

#: Cards these heroes hand over. All three are in the package already — a hero
#: that would need a new one is not in this file.
BRANN = "BG_LOE_077"
TAVERN_COIN = "BG28_810"
TRIPLE_REWARD = "triple_reward_discover"

HEROES: Dict[str, Hero] = {
    # I'm ready to rumble! — "Start the game with 30 extra Health."
    "TB_BaconShop_HERO_34": Hero(
        "TB_BaconShop_HERO_34",
        "Patchwerk",
        start_health=60,
        start_armor=0,
    ),
    # Hand of Time — "At the start of your turn, gain a free Refresh."
    "TB_BaconShop_HERO_57": Hero(
        "TB_BaconShop_HERO_57",
        "Nozdormu",
        start_armor=13,
        passives=(FreeFirstRefreshEachTurn(),),
    ),
    # Shopping Spree — "Minions and Refreshes cost 2 Gold. Upgrading the
    # Tavern costs (1) more."
    "TB_BaconShop_HERO_49": Hero(
        "TB_BaconShop_HERO_49",
        "Millhouse Manastorm",
        start_armor=16,
        passives=(FlatBuyCost(2), FlatRefreshCost(2), UpgradeCostSurcharge(1)),
    ),
    # Everbloom — "After you upgrade the Tavern, gain 2 Gold."
    "TB_BaconShop_HERO_74": Hero(
        "TB_BaconShop_HERO_74",
        "Forest Warden Omu",
        start_armor=6,
        passives=(GoldOnUpgrade(2),),
    ),
    # Avalanche — "After you play 3 Elementals, reduce the Cost of upgrading
    # the Tavern by (3)."
    "TB_BaconShop_HERO_78": Hero(
        "TB_BaconShop_HERO_78",
        "Chenvaala",
        start_armor=15,
        passives=(UpgradeDiscountPerElementals(per=3, reduction=3),),
    ),
    # ALL Will Burn! — "Start of Combat: Give ALL minions +2 Attack
    # permanently." Two on this patch, three on the 2021 one.
    "TB_BaconShop_HERO_52": Hero(
        "TB_BaconShop_HERO_52",
        "Deathwing",
        start_armor=18,
        passives=(CombatAttackAuraAll(2),),
    ),
    # Dream Portal — "The Tavern offers an extra Dragon whenever it is
    # Refreshed."
    "TB_BaconShop_HERO_53": Hero(
        "TB_BaconShop_HERO_53",
        "Ysera",
        start_armor=17,
        passives=(ExtraShopDragon(),),
    ),
    # Swatting Insects — "Start of Combat: Give your left-most minion Windfury,
    # Divine Shield, and Taunt."
    "TB_BaconShop_HERO_76": Hero(
        "TB_BaconShop_HERO_76",
        "Al'Akir",
        start_armor=15,
        passives=(
            StartOfCombatGrantLeftmost(
                (Keyword.WINDFURY, Keyword.SHIELD, Keyword.TAUNT)
            ),
        ),
    ),
    # Whatever You Want — "After you buy a Pirate, gain 1 Gold."
    "BG26_HERO_101": Hero(
        "BG26_HERO_101",
        "Cap'n Hoggarr",
        start_armor=12,
        passives=(GoldOnBuyTribe(Race.PIRATE, 1),),
    ),
    # Smart Savings — "After you sell a minion, gain 1 Gold next turn."
    "TB_BaconShop_HERO_10": Hero(
        "TB_BaconShop_HERO_10",
        "Trade Prince Gallywix",
        start_armor=5,
        passives=(GoldNextTurnOnSell(1),),
    ),
    # I'll Take That! — "After you sell 5 minions, get a random Murloc."
    # The 2021 printing put it in the Tavern; this one hands it over.
    "TB_BaconShop_HERO_55": Hero(
        "TB_BaconShop_HERO_55",
        "Fungalmancer Flurgl",
        start_armor=12,
        passives=(OnNthSellAddRaceToHand(n=5, race=Race.MURLOC),),
    ),
    # Repair Mode — "After 9 friendly minions die, get a random Mech."
    "BG22_HERO_200": Hero(
        "BG22_HERO_200",
        "Ini Stormcoil",
        start_armor=15,
        passives=(OnNthDeathAddRaceToHand(n=9, race=Race.MECHANICAL),),
    ),
    # Relic Vendor — "Every third Tavern spell you buy costs (0)."
    "BG28_HERO_800": Hero(
        "BG28_HERO_800",
        "Tae'thelan Bloodwatcher",
        start_armor=18,
        passives=(EveryNthTavernSpellFree(3),),
    ),
    # Grasp of Nature — "Give +1/+2 and Taunt to minions you summon during
    # combat."
    "TB_BaconShop_HERO_95": Hero(
        "TB_BaconShop_HERO_95",
        "Greybough",
        start_armor=16,
        passives=(
            BuffCombatSummons(attack=1, health=2, keywords=(Keyword.TAUNT,)),
        ),
    ),
    # Blademaster — "After a friendly minion kills an enemy, give it +1 Attack
    # permanently."
    "BG20_HERO_100": Hero(
        "BG20_HERO_100",
        "Rokara",
        start_armor=18,
        passives=(AttackOnKill(1),),
    ),
    # Frostwolf Banner — "When you have space in combat, summon a copy of your
    # highest-Attack minion. (Unlocks on Turn 7.)"
    "BG22_HERO_002": Hero(
        "BG22_HERO_002",
        "Drek'Thar",
        start_armor=10,
        passives=(SummonCopyWhenSpace(unlocks_on_turn=7),),
    ),
    # Stormpike Banner — the same, read for Health.
    "BG22_HERO_003": Hero(
        "BG22_HERO_003",
        "Vanndar Stormpike",
        start_armor=12,
        passives=(SummonCopyWhenSpace(by_health=True, unlocks_on_turn=7),),
    ),
    # Sharpen Blades — "After 14 friendly minions attack, the first minion you
    # buy each turn is free."
    "TB_BaconShop_HERO_59": Hero(
        "TB_BaconShop_HERO_59",
        "Aranna Starseeker",
        start_armor=12,
        passives=(FreeBuyEachTurnAfterAttacks(attacks=14),),
    ),
    # Blood Gems — "Minions in the Tavern have +1/+1. Improves after you buy
    # 3 minions."
    "BG20_HERO_102": Hero(
        "BG20_HERO_102",
        "Overlord Saurfang",
        start_armor=15,
        passives=(ShopStatBuffPerBuys(attack=1, health=1, per=3),),
    ),
    # Verdant Spheres — "After you buy 3 minions, get a Tavern Coin."
    "TB_BaconShop_HERO_60": Hero(
        "TB_BaconShop_HERO_60",
        "Kael'thas Sunstrider",
        start_armor=16,
        passives=(OnNthBuyAddCardToHand(n=3, card_id=TAVERN_COIN),),
    ),
    # Bring It On! — "After you buy 4 Battlecry minions, get a Brann
    # Bronzebeard. (Once per game.)"
    "TB_BaconShop_HERO_43": Hero(
        "TB_BaconShop_HERO_43",
        "Dinotamer Brann",
        start_armor=18,
        passives=(
            OnNthBuyAddCardToHand(
                n=4, card_id=BRANN, require_battlecry=True, once=True
            ),
        ),
    ),
    # Nature's Ally — "After you buy 20 Tiers' worth of cards, get a Triple
    # Reward."
    "BG20_HERO_242": Hero(
        "BG20_HERO_242",
        "Guff Runetotem",
        start_armor=12,
        passives=(OnTiersBoughtAddCardToHand(n=20, card_id=TRIPLE_REWARD),),
    ),
    # Living Legend — "After 15 friendly minions attack, get a Triple Reward."
    "BG33_HERO_001": Hero(
        "BG33_HERO_001",
        "Loh, the Living Legend",
        start_armor=17,
        passives=(OnAttacksAddCardToHand(n=15, card_id=TRIPLE_REWARD),),
    ),
    # Wax Warband — "Your Tavern spells give an extra +1/+1. At the start of
    # every 3 turns, improve this."
    "TB_BaconShop_HERO_75": Hero(
        "TB_BaconShop_HERO_75",
        "Rakanishu",
        start_armor=10,
        passives=(TavernSpellBonusPerTurns(attack=1, health=1, per_turns=3),),
    ),
    # Puzzle Box — "At the start of your turn, cast a random Tavern spell.
    # (Unlocks on Turn 3.)"
    "TB_BaconShop_HERO_35": Hero(
        "TB_BaconShop_HERO_35",
        "Yogg-Saron, Hope's End",
        start_armor=10,
        passives=(CastRandomSpellEachTurn(unlocks_on_turn=3),),
    ),
    # Twice as Nice — "After the Tavern is Refreshed, copy its highest-Tier
    # minion and Freeze them both."
    "BG22_HERO_004": Hero(
        "BG22_HERO_004",
        "Varden Dawngrasp",
        start_armor=18,
        passives=(OnRefreshCopyHighestTier(),),
    ),
    # Enhance-o Mechano — "After the Tavern is Refreshed, give a random minion
    # in it a random Bonus Keyword, twice."
    "BG24_HERO_204": Hero(
        "BG24_HERO_204",
        "Enhance-o Mechano",
        start_armor=14,
        passives=(OnRefreshGrantBonusKeyword(repeats=2),),
    ),
    # Procrastinate — "Skip your first two turns, then Discover a minion from
    # Tier 3 and Tier 4."
    "TB_BaconShop_HERO_16": Hero(
        "TB_BaconShop_HERO_16",
        "A. F. Kay",
        start_armor=15,
        passives=(SkipTurnsThenDiscover(rounds=(1, 2), tiers=(3, 4)),),
    ),
    # Deep Dive — "Skip your first turn. Discover minions from Tiers 6, 4 and
    # 2 to get at those Tiers."
    "BG22_HERO_201": Hero(
        "BG22_HERO_201",
        "Ambassador Faelin",
        start_armor=14,
        passives=(SkipTurnsThenDiscover(rounds=(1,), tiers=(6, 4, 2)),),
    ),
    # Stormlord's Boon — "At the start of the game, Discover a Tier 7 minion
    # to get after you spend 60 Gold."
    "BG27_HERO_801": Hero(
        "BG27_HERO_801",
        "Thorim, Stormlord",
        start_armor=18,
        passives=(DiscoverAtTierOnGoldSpent(tier=7, gold=60),),
    ),
    # Wax Rager — "On Turn 4, Discover two Hero Powers to replace this."
    "BG35_HERO_001": Hero(
        "BG35_HERO_001",
        "Genn, Worgen King",
        start_armor=7,
        passives=(DiscoverHeroPowerOnTurn(on_turn=4, options=2),),
    ),
    # Pandaren Mystic — "At the start of every turn, choose from 2 new Hero
    # Powers."
    "BG20_HERO_202": Hero(
        "BG20_HERO_202",
        "Master Nguyen",
        start_armor=10,
        passives=(DiscoverHeroPowerOnTurn(every_turn=True, options=2),),
    ),
    # Demon Hunter — "Start of Combat: Your left- and right-most minions gain
    # +2/+1 and attack immediately."
    "TB_BaconShop_HERO_08": Hero(
        "TB_BaconShop_HERO_08",
        "Illidan Stormrage",
        start_armor=18,
        passives=(
            StartOfCombatBuffEnds(attack=2, health=1, attack_immediately=True),
        ),
    ),
    # Wax Warband — "Start of Combat: Give a friendly minion of each type
    # +1/+1. (Improves after you spend 10 Gold.)"
    "TB_BaconShop_HERO_14": Hero(
        "TB_BaconShop_HERO_14",
        "Queen Wagtoggle",
        start_armor=14,
        passives=(StartOfCombatBuffOnePerTribe(attack=1, health=1, per_gold=10),),
    ),
    # Frost Shards — "Minions cost (2). The Tavern offers one fewer minion and
    # Freezes at the end of each turn."
    "TB_BaconShop_HERO_27": Hero(
        "TB_BaconShop_HERO_27",
        "Sindragosa",
        start_armor=7,
        passives=(FlatBuyCost(2), FewerShopSlots(1), FreezeShopEachTurn()),
    ),
}


HERO_POOL_IDS: FrozenSet[str] = frozenset(HEROES.keys())

#: No hero here starts the seat with a token, so the package needs no extra
#: card index entries. The 2021 package appends The Curator's Amalgam here;
#: that hero waits for its token to be carried.
HERO_TOKEN_IDS: FrozenSet[str] = frozenset()

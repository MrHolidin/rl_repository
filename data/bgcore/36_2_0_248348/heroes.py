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
    CombatAttackAuraAll,
    EveryNthTavernSpellFree,
    ExtraShopDragon,
    FlatBuyCost,
    FlatRefreshCost,
    FreeFirstRefreshEachTurn,
    GoldNextTurnOnSell,
    GoldOnBuyTribe,
    GoldOnUpgrade,
    Hero,
    OnNthDeathAddRaceToHand,
    OnNthSellAddRaceToHand,
    StartOfCombatGrantLeftmost,
    SummonCopyWhenSpace,
    UpgradeCostSurcharge,
    UpgradeDiscountPerElementals,
)
from src.bg_core.minion import Race

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
}


HERO_POOL_IDS: FrozenSet[str] = frozenset(HEROES.keys())

#: No hero here starts the seat with a token, so the package needs no extra
#: card index entries. The 2021 package appends The Curator's Amalgam here;
#: that hero waits for its token to be carried.
HERO_TOKEN_IDS: FrozenSet[str] = frozenset()

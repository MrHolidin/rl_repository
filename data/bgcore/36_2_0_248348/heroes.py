"""Hero catalog for HS Battlegrounds patch 36.2.0 (build 248348).

Loaded by :class:`src.bg_catalog.patch_context.PatchContext` the way
``bindings.py`` is. ``HEROES`` maps ``hero_id`` → :class:`Hero`;
``HERO_POOL_IDS`` is the assignable pool when the env runs with
``with_heroes=True``.

**Passive powers only, and only the ones that need no new cards.** The patch
ships 121 heroes: 56 whose power is passive and 65 the seat clicks. An active
power needs somewhere in the action space to be pressed, and that space is
frozen — so the active half waits on a decision this file does not make.

Of the 56 passives, these eight are the ones expressible with the passive
descriptors the engine already has and with cards this package already holds.
The rest of the 56 want either a descriptor that does not exist yet (Cap'n
Hoggarr's gold per Pirate bought, Rokara's Attack on a kill) or a card the
package does not carry (The Curator's Amalgam, N'Zoth's Fish, Onyxia's Whelp),
and the second kind is deliberately out of scope here.

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
    CombatAttackAuraAll,
    ExtraShopDragon,
    FlatBuyCost,
    FlatRefreshCost,
    FreeFirstRefreshEachTurn,
    GoldOnUpgrade,
    Hero,
    StartOfCombatGrantLeftmost,
    UpgradeCostSurcharge,
    UpgradeDiscountPerElementals,
)

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
}


HERO_POOL_IDS: FrozenSet[str] = frozenset(HEROES.keys())

#: No hero here starts the seat with a token, so the package needs no extra
#: card index entries. The 2021 package appends The Curator's Amalgam here;
#: that hero waits for its token to be carried.
HERO_TOKEN_IDS: FrozenSet[str] = frozenset()

"""Hero catalog for HS Battlegrounds patch 19.6.0.74257 (Jan 2021).

Loaded by :class:`src.bg_catalog.patch_context.PatchContext` (mirrors how
``bindings.py`` is loaded). ``HEROES`` maps ``hero_id`` → :class:`Hero`;
``HERO_POOL_IDS`` is the assignable pool when the env runs with
``with_heroes=True``.

Only **simple passive** heroes are included for now. Powers and numeric values
are pinned to this patch (verified against patch history): e.g. Deathwing was
+3 Attack here (nerfed to +2 in 20.8, July 2021); The Rat King granted +2/+2
(held Nov 2020 → July 2021); Chenvaala's discount was already -3 (raised from -2
in 18.6, Oct 2020).

Deferred (need shop-resizing past 6 slots or attack-ordering): Aranna
(7-minion shop), Illidan (edge minions attack first).
"""

from __future__ import annotations

from typing import Dict, FrozenSet

from src.bg_core.effects import Keyword
from src.bg_core.hero import (
    Hero,
    CombatAttackAuraAll,
    EveryNthBuyBuff,
    ExtraShopDragon,
    FlatBuyCost,
    FlatRefreshCost,
    FreeFirstRefreshEachTurn,
    GoldOnUpgrade,
    OnSellBuffRandomShop,
    OnSellRaceAddToShop,
    RotatingBuyTribeBuff,
    ShopTribeStatBuff,
    StartHandToken,
    StartOfCombatGrantLeftmost,
    StartTierMinions,
    UpgradeCostSurcharge,
    UpgradeDiscountPerElementals,
    ZeroGoldForRounds,
)
from src.bg_core.minion import Race

# Curator's Amalgam token (1/2, race ALL) — present in this patch's catalog as a
# non-pool minion; registered into TOKEN_IDS (bindings.py) so make_minion works.
AMALGAM_TOKEN_ID = "TB_BaconShop_HP_033t"


HEROES: Dict[str, Hero] = {
    # I'm ready to rumble! — start with 55 Health.
    "patchwerk": Hero("patchwerk", "Patchwerk", start_health=55),
    # Hand of Time — the first Refresh each turn costs (0).
    "nozdormu": Hero("nozdormu", "Nozdormu", passives=(FreeFirstRefreshEachTurn(),)),
    # Shopping Spree — Minions cost (2); Refresh costs (2); Tavern upgrades cost (1) more.
    "millhouse": Hero(
        "millhouse",
        "Millhouse Manastorm",
        passives=(FlatBuyCost(2), FlatRefreshCost(2), UpgradeCostSurcharge(1)),
    ),
    # Everbloom — after you upgrade Bob's Tavern, gain 2 Gold.
    "omu": Hero("omu", "Forest Warden Omu", passives=(GoldOnUpgrade(2),)),
    # Avalanche — after you play 3 Elementals, the next Tavern upgrade costs (3) less.
    "chenvaala": Hero(
        "chenvaala",
        "Chenvaala",
        passives=(UpgradeDiscountPerElementals(per=3, reduction=3),),
    ),
    # ALL Will Burn! — ALL minions have +3 Attack (both sides, in combat).
    "deathwing": Hero("deathwing", "Deathwing", passives=(CombatAttackAuraAll(3),)),
    # Tinker — Mechs in Bob's Tavern have +1/+1.
    "millificent": Hero(
        "millificent",
        "Millificent Manastorm",
        passives=(ShopTribeStatBuff(Race.MECHANICAL, attack=1, health=1),),
    ),
    # Dream Portal — Bob's Tavern always has an extra Dragon.
    "ysera": Hero("ysera", "Ysera", passives=(ExtraShopDragon(),)),
    # Verdant Spheres — every third minion you buy gains +2/+2.
    "kaelthas": Hero(
        "kaelthas",
        "Kael'thas Sunstrider",
        passives=(EveryNthBuyBuff(n=3, attack=2, health=2),),
    ),
    # A Tale of Kings — buy a minion of the rotating type → +2/+2; type swaps each turn.
    "rat_king": Hero(
        "rat_king",
        "The Rat King",
        passives=(RotatingBuyTribeBuff(attack=2, health=2),),
    ),
    # Cleaning Up — after you sell a minion, give 2 random Tavern minions +1/+1.
    "deryl": Hero(
        "deryl",
        "Dancin' Deryl",
        passives=(OnSellBuffRandomShop(count=2, attack=1, health=1),),
    ),
    # I'll take that! — after you sell a Murloc, add a random Murloc to Bob's Tavern.
    "flurgl": Hero(
        "flurgl",
        "Fungalmancer Flurgl",
        passives=(OnSellRaceAddToShop(Race.MURLOC),),
    ),
    # Swatting Insects — Start of Combat: give your left-most minion Windfury,
    # Divine Shield, and Taunt.
    "alakir": Hero(
        "alakir",
        "Al'Akir the Windlord",
        passives=(
            StartOfCombatGrantLeftmost(
                (Keyword.WINDFURY, Keyword.SHIELD, Keyword.TAUNT)
            ),
        ),
    ),
    # Menagerist — Start of Game: add a 1/2 Amalgam (all minion types) to your hand.
    "curator": Hero(
        "curator",
        "The Curator",
        passives=(StartHandToken(AMALGAM_TOKEN_ID),),
    ),
    # Procrastinate — skip your first two turns; start with two Tavern-Tier-3 minions.
    "afkay": Hero(
        "afkay",
        "A.F. Kay",
        passives=(ZeroGoldForRounds((1, 2)), StartTierMinions(count=2, tier=3)),
    ),
}


HERO_POOL_IDS: FrozenSet[str] = frozenset(HEROES.keys())

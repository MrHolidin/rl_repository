"""Battlegrounds heroes with passive powers.

A :class:`Hero` is assigned to a seat at game start (only when the env runs with
``with_heroes=True``). Heroes carry a tuple of typed **passive descriptors**; the
dispatch that applies them at each game event lives in
:mod:`src.bg_recruitment.hero_passives`.

This module is a leaf (it imports only :mod:`src.bg_core.minion` /
:mod:`src.bg_core.effects`) so economy/shop/combat code can read passive-derived
values straight off ``player.hero`` without import cycles.

Powers are pinned to Hearthstone Battlegrounds patch **19.6.0.74257** (Jan 2021).
Numeric values that changed across patches are documented at their use sites in
``data/bgcore/19_6_0_74257/heroes.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

from src.bg_core.effects import Keyword
from src.bg_core.minion import Race

__all__ = [
    "Hero",
    "HeroPassive",
    "StartHandToken",
    "StartTierMinions",
    "ZeroGoldForRounds",
    "FreeFirstRefreshEachTurn",
    "FlatRefreshCost",
    "FlatBuyCost",
    "UpgradeCostSurcharge",
    "GoldOnUpgrade",
    "UpgradeDiscountPerElementals",
    "CombatAttackAuraAll",
    "ShopTribeStatBuff",
    "ExtraShopDragon",
    "EveryNthBuyBuff",
    "RotatingBuyTribeBuff",
    "OnSellBuffRandomShop",
    "OnSellRaceAddToShop",
    "StartOfCombatGrantLeftmost",
]


# --------------------------------------------------------------------------- #
# Passive descriptors (one small frozen dataclass per distinct mechanic).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StartHandToken:
    """Add a token to the player's hand at game start (Curator → Amalgam)."""

    card_id: str


@dataclass(frozen=True)
class StartTierMinions:
    """Add ``count`` random minions of exactly ``tier`` to hand at game start (A.F. Kay)."""

    count: int
    tier: int


@dataclass(frozen=True)
class ZeroGoldForRounds:
    """Force gold to 0 on the listed round numbers (A.F. Kay 'skips' rounds 1-2)."""

    rounds: Tuple[int, ...]


@dataclass(frozen=True)
class FreeFirstRefreshEachTurn:
    """The first Refresh each turn costs 0 (Nozdormu)."""


@dataclass(frozen=True)
class FlatRefreshCost:
    """Every Refresh costs a fixed amount (Millhouse → 2)."""

    cost: int


@dataclass(frozen=True)
class FlatBuyCost:
    """Buying a minion costs a fixed amount (Millhouse → 2)."""

    cost: int


@dataclass(frozen=True)
class UpgradeCostSurcharge:
    """Tavern upgrades cost this much more, persistently (Millhouse → +1)."""

    amount: int


@dataclass(frozen=True)
class GoldOnUpgrade:
    """Gain this much gold after upgrading the Tavern (Omu → +2)."""

    amount: int


@dataclass(frozen=True)
class UpgradeDiscountPerElementals:
    """After every ``per`` Elementals played, reduce the next upgrade cost by
    ``reduction`` (Chenvaala → 3 Elementals, -3)."""

    per: int
    reduction: int


@dataclass(frozen=True)
class CombatAttackAuraAll:
    """All minions on BOTH sides get +``amount`` Attack in combat (Deathwing → +3
    at patch 19.6; nerfed to +2 in patch 20.8, July 2021)."""

    amount: int


@dataclass(frozen=True)
class ShopTribeStatBuff:
    """Minions of ``race`` in Bob's Tavern get +atk/+hp while offered
    (Millificent → Mechs +1/+1)."""

    race: Race
    attack: int
    health: int


@dataclass(frozen=True)
class ExtraShopDragon:
    """Bob's Tavern always offers one extra slot that is a Dragon (Ysera).

    The extra slot is capped at the max visible shop size, so at Tavern Tier 6
    (already 6 offers) there is no room for the extra — a minor deviation.
    """


@dataclass(frozen=True)
class EveryNthBuyBuff:
    """Every ``n``-th minion bought gains +atk/+hp (Kael'thas → every 3rd, +2/+2)."""

    n: int
    attack: int
    health: int


@dataclass(frozen=True)
class RotatingBuyTribeBuff:
    """Buying a minion of the current rotating tribe grants +atk/+hp; the tribe
    swaps each turn (The Rat King → +2/+2 at patch 19.6)."""

    attack: int
    health: int


@dataclass(frozen=True)
class OnSellBuffRandomShop:
    """After selling a minion, give ``count`` random Tavern minions +atk/+hp
    (Dancin' Deryl → 2 minions +1/+1)."""

    count: int
    attack: int
    health: int


@dataclass(frozen=True)
class OnSellRaceAddToShop:
    """After selling a minion of ``race``, add a random minion of ``race`` to the
    Tavern (Fungalmancer Flurgl → Murloc)."""

    race: Race


@dataclass(frozen=True)
class StartOfCombatGrantLeftmost:
    """Start of Combat: grant ``keywords`` to your left-most minion
    (Al'Akir → Windfury, Divine Shield, Taunt)."""

    keywords: Tuple[Keyword, ...]


HeroPassive = Union[
    StartHandToken,
    StartTierMinions,
    ZeroGoldForRounds,
    FreeFirstRefreshEachTurn,
    FlatRefreshCost,
    FlatBuyCost,
    UpgradeCostSurcharge,
    GoldOnUpgrade,
    UpgradeDiscountPerElementals,
    CombatAttackAuraAll,
    ShopTribeStatBuff,
    ExtraShopDragon,
    EveryNthBuyBuff,
    RotatingBuyTribeBuff,
    OnSellBuffRandomShop,
    OnSellRaceAddToShop,
    StartOfCombatGrantLeftmost,
]


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Hero:
    hero_id: str
    name: str
    # None ⇒ inherit the ruleset's default starting health; only heroes whose
    # power sets a custom pool (Patchwerk) override it.
    start_health: Optional[int] = None
    passives: Tuple[HeroPassive, ...] = field(default_factory=tuple)

    # -- passive-derived reads (cheap scans; called from economy/shop/combat) --

    def flat_buy_cost(self) -> Optional[int]:
        for p in self.passives:
            if isinstance(p, FlatBuyCost):
                return p.cost
        return None

    def flat_refresh_cost(self) -> Optional[int]:
        for p in self.passives:
            if isinstance(p, FlatRefreshCost):
                return p.cost
        return None

    def upgrade_cost_surcharge(self) -> int:
        return sum(p.amount for p in self.passives if isinstance(p, UpgradeCostSurcharge))

    def extra_shop_slots(self) -> int:
        return sum(1 for p in self.passives if isinstance(p, ExtraShopDragon))

    def shop_tribe_buff(self) -> Optional[ShopTribeStatBuff]:
        for p in self.passives:
            if isinstance(p, ShopTribeStatBuff):
                return p
        return None

    def combat_attack_aura(self) -> int:
        return sum(p.amount for p in self.passives if isinstance(p, CombatAttackAuraAll))

    def start_combat_leftmost_keywords(self) -> frozenset:
        kws: set = set()
        for p in self.passives:
            if isinstance(p, StartOfCombatGrantLeftmost):
                kws.update(p.keywords)
        return frozenset(kws)

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
from typing import Any, Optional, Tuple, Union

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
    "GoldOnBuyTribe",
    "GoldNextTurnOnSell",
    "OnNthSellAddRaceToHand",
    "OnNthDeathAddRaceToHand",
    "EveryNthTavernSpellFree",
    "BuffCombatSummons",
    "AttackOnKill",
    "SummonCopyWhenSpace",
    "PowerCostGrowsPerUse",
    "OnNthBuyAddCardToHand",
    "OnTiersBoughtAddCardToHand",
    "OnAttacksAddCardToHand",
    "FreeBuyEachTurnAfterAttacks",
    "ShopStatBuffPerBuys",
    "TavernSpellBonusPerTurns",
    "CastRandomSpellEachTurn",
    "OnRefreshCopyHighestTier",
    "OnRefreshGrantBonusKeyword",
    "SkipTurnsThenDiscover",
    "DiscoverAtTierOnGoldSpent",
    "DiscoverHeroPowerOnTurn",
    "FewerShopSlots",
    "FreezeShopEachTurn",
    "StartOfCombatBuffEnds",
    "StartOfCombatBuffOnePerTribe",
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
class GoldOnBuyTribe:
    """After you buy a minion of ``race``, gain gold (Cap'n Hoggarr)."""

    race: Race
    amount: int = 1


@dataclass(frozen=True)
class GoldNextTurnOnSell:
    """After you sell a minion, gain gold *next* turn (Trade Prince Gallywix).

    Next turn, not this one, which is the whole shape of the card: the gold is
    banked and paid at the start of the turn after.
    """

    amount: int = 1


@dataclass(frozen=True)
class OnNthSellAddRaceToHand:
    """Every ``n`` sales, a random minion of ``race`` to hand (Flurgl).

    To hand rather than to the counter — the 2021 printing put it in the
    Tavern, this one hands it over.
    """

    n: int
    race: Race


@dataclass(frozen=True)
class OnNthDeathAddRaceToHand:
    """Every ``n`` friendly deaths, a random minion of ``race`` to hand (Ini).

    Counted from the seat's own game-long death tally and paid at its next turn
    start: the deaths happen inside a fight, and a fight hands what it owes to
    the seat rather than reaching into the hand mid-combat.
    """

    n: int
    race: Race


@dataclass(frozen=True)
class EveryNthTavernSpellFree:
    """Every ``n``th Tavern spell bought costs nothing (Tae'thelan)."""

    n: int = 3


@dataclass(frozen=True)
class BuffCombatSummons:
    """Minions summoned during combat arrive bigger (Greybough)."""

    attack: int = 0
    health: int = 0
    keywords: Tuple[Keyword, ...] = ()


@dataclass(frozen=True)
class AttackOnKill:
    """After a friendly kills an enemy, it keeps +``amount`` Attack (Rokara).

    Permanently: the gain outlives the combat copy that earned it, so it goes
    back to the seat the way every other kept gain does.
    """

    amount: int = 1


@dataclass(frozen=True)
class SummonCopyWhenSpace:
    """While there is room in combat, copy your biggest minion (Drek'Thar).

    ``by_health`` picks the highest-Health one instead (Vanndar Stormpike), and
    ``unlocks_on_turn`` is the turn the power starts working.
    """

    by_health: bool = False
    unlocks_on_turn: int = 1


@dataclass(frozen=True)
class PowerCostGrowsPerUse:
    """"Costs (1) more after each use" — Elise's price climbs as she is used."""

    amount: int = 1


@dataclass(frozen=True)
class OnNthBuyAddCardToHand:
    """Every ``n`` cards bought, a named card to hand (Kael'thas' Tavern Coin).

    ``require_battlecry`` counts only the minions that have one (Dinotamer
    Brann), and ``once`` stops after the first payout.
    """

    n: int
    card_id: str
    require_battlecry: bool = False
    once: bool = False


@dataclass(frozen=True)
class OnTiersBoughtAddCardToHand:
    """Every ``n`` Tiers' worth of cards bought, a named card (Guff)."""

    n: int
    card_id: str


@dataclass(frozen=True)
class OnAttacksAddCardToHand:
    """Every ``n`` friendly attacks, a named card to hand (Loh)."""

    n: int
    card_id: str


@dataclass(frozen=True)
class FreeBuyEachTurnAfterAttacks:
    """After ``attacks`` friendly attacks, the first buy each turn is free."""

    attacks: int


@dataclass(frozen=True)
class ShopStatBuffPerBuys:
    """Minions in the Tavern have +N/+N, growing every ``per`` buys."""

    attack: int = 1
    health: int = 1
    per: int = 3


@dataclass(frozen=True)
class TavernSpellBonusPerTurns:
    """"Your Tavern spells give an extra +1/+1", improving every N turns."""

    attack: int = 1
    health: int = 1
    per_turns: int = 3


@dataclass(frozen=True)
class CastRandomSpellEachTurn:
    """At the start of your turn, cast a random Tavern spell (Yogg-Saron)."""

    unlocks_on_turn: int = 1


@dataclass(frozen=True)
class OnRefreshCopyHighestTier:
    """After a Refresh, copy the counter's best minion and freeze both."""


@dataclass(frozen=True)
class OnRefreshGrantBonusKeyword:
    """After a Refresh, hand a random Tavern minion a random Bonus Keyword."""

    repeats: int = 1


@dataclass(frozen=True)
class SkipTurnsThenDiscover:
    """Skip the opening turns, then Discover at each named tier in turn."""

    rounds: Tuple[int, ...]
    tiers: Tuple[int, ...]


@dataclass(frozen=True)
class DiscoverAtTierOnGoldSpent:
    """Discover a minion at ``tier`` now; it arrives after ``gold`` is spent."""

    tier: int
    gold: int


@dataclass(frozen=True)
class DiscoverHeroPowerOnTurn:
    """Swap this power for one of a Discover, on a turn or every turn."""

    on_turn: int = 0
    every_turn: bool = False
    options: int = 2


@dataclass(frozen=True)
class FewerShopSlots:
    """The Tavern shows this many fewer minions (Sindragosa)."""

    amount: int = 1


@dataclass(frozen=True)
class FreezeShopEachTurn:
    """The Tavern Freezes at the end of each turn (Sindragosa)."""


@dataclass(frozen=True)
class StartOfCombatBuffEnds:
    """Start of Combat: the end minions gain stats and swing at once."""

    attack: int = 0
    health: int = 0
    attack_immediately: bool = False


@dataclass(frozen=True)
class StartOfCombatBuffOnePerTribe:
    """Start of Combat: a friendly of each type gains stats (Wagtoggle).

    ``per_gold`` is the improvement the card prints, read off the gold the seat
    has spent all game.
    """

    attack: int = 1
    health: int = 1
    per_gold: int = 0


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
    GoldOnBuyTribe,
    GoldNextTurnOnSell,
    OnNthSellAddRaceToHand,
    OnNthDeathAddRaceToHand,
    EveryNthTavernSpellFree,
    BuffCombatSummons,
    AttackOnKill,
    SummonCopyWhenSpace,
    PowerCostGrowsPerUse,
    OnNthBuyAddCardToHand,
    OnTiersBoughtAddCardToHand,
    OnAttacksAddCardToHand,
    FreeBuyEachTurnAfterAttacks,
    ShopStatBuffPerBuys,
    TavernSpellBonusPerTurns,
    CastRandomSpellEachTurn,
    OnRefreshCopyHighestTier,
    OnRefreshGrantBonusKeyword,
    SkipTurnsThenDiscover,
    DiscoverAtTierOnGoldSpent,
    DiscoverHeroPowerOnTurn,
    FewerShopSlots,
    FreezeShopEachTurn,
    StartOfCombatBuffEnds,
    StartOfCombatBuffOnePerTribe,
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
    # Flat armor granted at game start (absorbs damage before health; modern
    # per-hero balance lever, 0 on classic/no-armor patches).
    start_armor: int = 0
    passives: Tuple[HeroPassive, ...] = field(default_factory=tuple)
    #: What the seat presses. A tuple of abilities resolved by the same
    #: dispatcher a Tavern spell goes through, because a hero power is the
    #: same kind of thing: an effect with no body behind it.
    power: Tuple[Any, ...] = field(default_factory=tuple)
    #: The price on the card, and how often it may be pressed. The pool prints
    #: four different limits and they compose rather than exclude:
    #:
    #: * ``power_uses`` — presses per turn. One unless the card says otherwise
    #:   ("Twice per turn"), which two of the 65 do.
    #: * ``power_charges`` — presses for the whole game, 0 for no limit.
    #:   "Once per game" is one charge; Captain Eudora's "(4 Digs left.)",
    #:   Putricide's "(3 Creations left!)" and Zephrys' "(3 Wishes left!)" are
    #:   the rest. Not to be confused with the seven cards that print "(N
    #:   left!)" as a countdown to a *payout* rather than a use limit.
    #: * ``power_cooldown_turns`` — turns it sleeps after a press. Snake Eyes
    #:   alone, and the number is the die roll rather than a constant, so the
    #:   effect sets it.
    power_cost: int = 0
    power_uses: int = 1
    power_charges: int = 0
    power_cooldown_turns: int = 0
    #: "Improves after you buy 4 cards" — the power's numbers are multiplied by
    #: how many times the seat has bought that many. A level rather than a
    #: rewrite, so the card still prints what one use is worth.
    power_improve_per_buys: int = 0
    #: Powers that only wake up later ("Unlocks at Tier 4", "on Turn 3").
    power_unlocks_at_tier: int = 0
    power_unlocks_on_turn: int = 0

    def with_power_of(self, other: "Hero") -> "Hero":
        """This hero, playing ``other``'s power.

        "Discover a new Hero Power" takes the power and leaves the hero: the
        seat keeps its name, its armor and every passive it is still playing —
        including the passive that opened the Discover, which is how Master
        Nguyen's power can change *every* turn rather than once.
        """
        from dataclasses import replace

        return replace(
            self,
            power=other.power,
            power_cost=other.power_cost,
            power_uses=other.power_uses,
            power_charges=other.power_charges,
            power_cooldown_turns=other.power_cooldown_turns,
            power_improve_per_buys=other.power_improve_per_buys,
            power_unlocks_at_tier=other.power_unlocks_at_tier,
            power_unlocks_on_turn=other.power_unlocks_on_turn,
        )

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

    def has_power(self) -> bool:
        return bool(self.power)

    def fewer_shop_slots(self) -> int:
        return sum(p.amount for p in self.passives if isinstance(p, FewerShopSlots))

    def shop_stat_buff_per_buys(self) -> Optional["ShopStatBuffPerBuys"]:
        for p in self.passives:
            if isinstance(p, ShopStatBuffPerBuys):
                return p
        return None

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

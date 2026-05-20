from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Tuple, Union


class Keyword(Enum):
    TAUNT = auto()
    SHIELD = auto()  # Divine Shield (printed or granted)
    WINDFURY = auto()
    POISONOUS = auto()
    CHARGE = auto()
    MAGNETIC = auto()


class Trigger(Enum):
    """Shop-phase triggers use ``PlayerState`` context; ON_DEATH/AURA use combat."""

    ON_BUY = auto()
    ON_PLACE = auto()
    AFTER_FRIENDLY_MINION_PLACED = auto()
    ON_DEATH = auto()
    ON_TURN_END = auto()
    AURA = auto()
    ON_FRIENDLY_MECH_DIED = auto()  # legacy trigger flag (obs); unused in rules
    ON_TURN_START = auto()  # shop: after round increment, before shop reroll (board then hand)
    ON_OVERKILL = auto()  # combat-only: excess kill damage on defender
    ON_FRIENDLY_MINION_SUMMONED = auto()  # shop + combat: another friendly hit the board
    ON_SELF_DAMAGED = auto()  # combat-only: this minion lost divine shield or took HP damage
    ON_FRIENDLY_MINION_DIED = (
        auto()
    )  # combat: another friendly died (``Ability.filter_race`` = dead minion's tribe)


@dataclass(frozen=True)
class SummonEffect:
    """Summon ``count`` copies of a fixed token (``CARD_TEMPLATES``), or one token per attack if flagged."""

    token_id: str
    count: int = 1
    count_from_source_attack: bool = False
    for_opponent: bool = False
    # Golden Rat Pack: same DR resolves multiple sweeps before Baron/Duplicator multipliers.
    dr_wave_count: int = 1


@dataclass(frozen=True)
class SummonRandomMinionEffect:
    """Deathrattle: summon ``count`` random BG tavern minions (tier filter / optional legendary or DR)."""

    count: int = 1
    exact_tier: Optional[int] = None
    legendary_only: bool = False
    require_deathrattle: bool = False
    race_filter: Optional[Any] = None
    exclude_source: bool = True
    for_opponent: bool = False


@dataclass(frozen=True)
class BuffRandomFriendly:
    attack: int
    health: int
    exclude_self: bool = True
    filter_race: Optional[Any] = None
    grant_taunt: bool = False
    repeats: int = 1


@dataclass(frozen=True)
class BuffOnePerListedTribeFriendly:
    """Shop: for each entry in ``tribes``, pick uniformly among matching friendlies (if any).

    Dragon and other HS tribes omitted from ``Race`` are skipped at card definition time.
    """

    attack: int
    health: int
    tribes: Tuple[Any, ...]
    exclude_self: bool = True


@dataclass(frozen=True)
class BuffAllOtherOfTribe:
    """One-shot shop battlecry: buff every other friendly minion of ``tribe`` (``Race.ALL`` matches any)."""

    tribe: Any
    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class BuffAllFriendlyOfTribe:
    """One-shot: all friendly minions matching ``tribe`` (including self if it matches)."""

    tribe: Any
    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class BuffAllWithKeyword:
    """All friendly minions that have ``keyword`` (shop battlecry or combat deathrattle)."""

    keyword: Keyword
    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class GrantKeywordRandomFriendly:
    """Random eligible friendly gains a keyword (shop battlecry or combat deathrattle)."""

    keyword: Keyword
    filter_race: Optional[Any] = None
    exclude_self: bool = True
    repeats: int = 1


@dataclass(frozen=True)
class BuffSelfWhenFriendlyBattlecryPlaced:
    """Shop: source gains stats after another friendly with an ``ON_PLACE`` ability is placed."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class BuffAllFriendlyMinions:
    """Combat deathrattle: buff every surviving friendly (e.g. Spawn of N'Zoth)."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class BuffRandomOtherFriendlyCombat:
    """Combat deathrattle: one random other friendly (Tortollan Shellraiser)."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class DealDamageRandomEnemyMinion:
    """Combat deathrattle: deal ``amount`` to one random enemy minion (Kaboom Bot)."""

    amount: int


@dataclass(frozen=True)
class AttackBonusPerOtherMurlocGlobal:
    """Combat: +``per_attack`` Attack per other Murloc (or ALL) anywhere on the battlefield (Old Murk-Eye)."""

    per_attack: int = 1


@dataclass(frozen=True)
class BuffSummonedIfRace:
    """When a friendly minion is summoned, buff it if it matches ``tribe`` (Pack Leader, Mama Bear)."""

    tribe: Any
    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class GrantListenerKeywordIfSummonedMatches:
    """Listener gains ``keyword`` when a summoned friendly matches ``tribe`` (Cobalt Guardian)."""

    tribe: Any
    keyword: Keyword


@dataclass(frozen=True)
class BuffListenerIfSummonedMatches:
    """Listener buffs itself when a summoned friendly matches ``tribe`` (Murloc Tidecaller)."""

    tribe: Any
    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class SummonOnSelfDamaged:
    """Combat: after this minion takes damage (incl. divine shield pop), summon token(s)."""

    token_id: str
    count: int = 1


@dataclass(frozen=True)
class PogoHopperBattlecry:
    """Shop: +attack_each/+health_each for each other Pogo-Hopper already played; then increment counter."""

    attack_each: int = 2
    health_each: int = 2


@dataclass(frozen=True)
class StatAura:
    """Raid-leader style: every **other** living friendly (not `self`) gains these stats."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class TribalOtherStatAura:
    """BG 'your other {tribe}' buff; recipient must match ``tribe`` (``Race.ALL`` matches any)."""

    tribe: Any
    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class KeywordStatAura:
    """Grants stats only to minions that have ``keyword`` (e.g. Phalanx Commander + Taunt)."""

    keyword: Keyword
    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class AdjacentStatAura:
    """Dire Wolf–style: living minions in the immediate board slots left/right of the source."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class BuffAdjacentBattlecry:
    """Defender of Argus–style: buff minions in adjacent board slots on play (shop only)."""

    attack: int = 0
    health: int = 0
    grant_taunt: bool = False


@dataclass(frozen=True)
class BuffTargetFriendlyBattlecry:
    """Shop battlecry: player picks another friendly on board (modal if 2+ eligible)."""

    attack: int = 1
    health: int = 1
    exclude_self: bool = True
    filter_race: Optional[Any] = None


@dataclass(frozen=True)
class HeroImmuneAura:
    """While this aura source is alive on your board, ``_damage_hero`` is blocked (BG Mal'Ganis)."""


@dataclass(frozen=True)
class DealHeroDamage:
    amount: int


@dataclass(frozen=True)
class BuffSelf:
    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class BuffSelfFromHeroDamageTaken:
    """+0/+X where X = total hero damage taken this game (Annihilan battlecry)."""


@dataclass(frozen=True)
class SummonFirstDeadFriendlyMechsThisCombat:
    """Deathrattle: summon shallow copies of the first ``count`` dead friendly Mech corpses (board order)."""

    count: int = 2


@dataclass(frozen=True)
class BattlecryMultiplierAura:
    """BG Brann-style: product of factors on board multiplies ON_PLACE (battlecry) executions."""

    factor: int


@dataclass(frozen=True)
class DeathrattleMultiplierAura:
    """BG Baron-style: product multiplies each ON_DEATH execution count in combat."""

    factor: int


@dataclass(frozen=True)
class SummonMultiplierAura:
    """BG Khadgar-style: product multiplies each summon iteration from summon effects."""

    factor: int


@dataclass(frozen=True)
class ZappTargeting:
    """Combat-only: choose defender with minimum attack among legal taunt pool (BG Zapp Slywick)."""


@dataclass(frozen=True)
class CleaveOnAttack:
    """Combat-only: primary attack also deals the same swing damage to adjacent defender indices."""


@dataclass(frozen=True)
class DiscoverMurlocEffect:
    """Battlecry: Discover a Murloc (tier-weighted pool). ``repeats`` stacks with Brann (product)."""

    repeats: int = 1


@dataclass(frozen=True)
class AdaptAllMurlocsEffect:
    """Battlecry: Adapt your Murlocs — pick 3 of 10, apply to all friendly Murlocs."""

    repeats: int = 1


Effect = Union[
    SummonEffect,
    SummonRandomMinionEffect,
    BuffRandomFriendly,
    BuffOnePerListedTribeFriendly,
    BuffAllOtherOfTribe,
    BuffAllFriendlyOfTribe,
    BuffAllWithKeyword,
    GrantKeywordRandomFriendly,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffAllFriendlyMinions,
    BuffRandomOtherFriendlyCombat,
    DealDamageRandomEnemyMinion,
    AttackBonusPerOtherMurlocGlobal,
    BuffSummonedIfRace,
    GrantListenerKeywordIfSummonedMatches,
    BuffListenerIfSummonedMatches,
    SummonOnSelfDamaged,
    PogoHopperBattlecry,
    StatAura,
    TribalOtherStatAura,
    KeywordStatAura,
    AdjacentStatAura,
    BuffAdjacentBattlecry,
    BuffTargetFriendlyBattlecry,
    HeroImmuneAura,
    DealHeroDamage,
    BuffSelf,
    BuffSelfFromHeroDamageTaken,
    SummonFirstDeadFriendlyMechsThisCombat,
    BattlecryMultiplierAura,
    DeathrattleMultiplierAura,
    SummonMultiplierAura,
    ZappTargeting,
    CleaveOnAttack,
    DiscoverMurlocEffect,
    AdaptAllMurlocsEffect,
]


@dataclass(frozen=True)
class Ability:
    trigger: Trigger
    effect: Effect
    """If set: ``AFTER_FRIENDLY_MINION_PLACED`` / ``ON_FRIENDLY_MINION_DIED`` filter on placed/dead minion race."""

    filter_race: Optional[Any] = None


__all__ = [
    "Keyword",
    "Trigger",
    "SummonEffect",
    "SummonRandomMinionEffect",
    "BuffRandomFriendly",
    "BuffOnePerListedTribeFriendly",
    "BuffAllOtherOfTribe",
    "BuffAllFriendlyOfTribe",
    "BuffAllWithKeyword",
    "GrantKeywordRandomFriendly",
    "BuffSelfWhenFriendlyBattlecryPlaced",
    "BuffAllFriendlyMinions",
    "BuffRandomOtherFriendlyCombat",
    "DealDamageRandomEnemyMinion",
    "AttackBonusPerOtherMurlocGlobal",
    "BuffSummonedIfRace",
    "GrantListenerKeywordIfSummonedMatches",
    "BuffListenerIfSummonedMatches",
    "SummonOnSelfDamaged",
    "PogoHopperBattlecry",
    "StatAura",
    "TribalOtherStatAura",
    "KeywordStatAura",
    "AdjacentStatAura",
    "BuffAdjacentBattlecry",
    "BuffTargetFriendlyBattlecry",
    "HeroImmuneAura",
    "DealHeroDamage",
    "BuffSelf",
    "BuffSelfFromHeroDamageTaken",
    "SummonFirstDeadFriendlyMechsThisCombat",
    "BattlecryMultiplierAura",
    "DeathrattleMultiplierAura",
    "SummonMultiplierAura",
    "ZappTargeting",
    "CleaveOnAttack",
    "DiscoverMurlocEffect",
    "AdaptAllMurlocsEffect",
    "Effect",
    "Ability",
]

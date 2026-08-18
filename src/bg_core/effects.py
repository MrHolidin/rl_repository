from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Tuple, Union


class Keyword(Enum):
    TAUNT = auto()
    SHIELD = auto()  # Divine Shield (printed or granted)
    WINDFURY = auto()
    MEGA_WINDFURY = auto()
    POISONOUS = auto()
    CHARGE = auto()
    MAGNETIC = auto()
    REBORN = auto()
    #: Poisonous that is used up by the kill it makes (modern builds print this
    #: where older ones printed Poisonous). Appended, never inserted — see Race.
    VENOMOUS = auto()


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
    ON_START_OF_COMBAT = auto()  # combat-only: after setup, before first attack
    ON_SELL = auto()  # shop: when sold from board, before removal
    ON_FRIENDLY_BOUGHT = auto()  # shop: board listener when another minion is bought
    ON_AFTER_ATTACK = auto()  # combat: after this minion completes an attack swing
    ON_FRIENDLY_ATTACK = auto()  # combat: board listener when another friendly attacks
    ON_SURVIVED_ATTACK = auto()  # combat: this minion took damage and survived the swing
    ON_FRIENDLY_SHIELD_LOST = auto()  # combat: another friendly lost Divine Shield
    ON_WHEN_ATTACKED = auto()  # combat: this minion is targeted by an attack swing
    ON_FRIENDLY_WHEN_ATTACKED = auto()  # combat: another friendly is targeted by an attack
    ON_FRIENDLY_KILL = auto()  # combat: a friendly minion killed an enemy minion
    #: combat: this minion is attacking, target chosen, before any damage —
    #: the modern **Rally** keyword ("Whenever this attacks"). Distinct from
    #: ON_AFTER_ATTACK, which lands once the swing is over: a Rally that strips
    #: the target's Reborn has to run while the target is still standing.
    ON_ATTACK = auto()
    #: shop: a spell was cast on this minion — the Spellcraft spells and the
    #: Blood Gems, which are the two things the engine lets a seat cast at a
    #: body ("Whenever you cast a spell on this, gain +1 Health").
    ON_TARGETED_BY_SPELL = auto()
    #: shop: the seat played a card with Choose One, after the option resolved
    #: (Turbo Hogrider: "After you play Choose One card, this plays a Blood Gem
    #: on all your other Quilboar").
    ON_CHOOSE_ONE_PLAYED = auto()
    #: shop: the **Activate** keyword — the seat spends gold to fire this
    #: minion's ability, once per turn. Alone among the triggers it is not an
    #: event the engine raises but a move the player makes, so nothing fires it
    #: on its own; see ``src/bg_recruitment/activate.py``. The gold it costs
    #: lives on the ability (``Ability.activate_cost``).
    ON_ACTIVATE = auto()


class ConditionKind(Enum):
    OTHER_TRIBE_ON_BOARD = auto()
    LAST_COMBAT_WON = auto()


@dataclass(frozen=True)
class Condition:
    kind: ConditionKind
    tribe: Optional[Any] = None


@dataclass(frozen=True)
class SummonEffect:
    """Summon ``count`` copies of a fixed token (``CARD_TEMPLATES``), or one token per attack if flagged."""

    token_id: str
    count: int = 1
    count_from_source_attack: bool = False
    for_opponent: bool = False
    attack_immediately: bool = False
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


class BuffTarget(Enum):
    """Which friendlies a :class:`BuffMatching` hits.

    Each member is part of the effect's observation identity (see
    ``_EFFECT_SIGNATURES`` in ``minibg.obs``), so renaming one is an obs
    change — add, don't rename.
    """

    #: every friendly (no filter); the source is included where the call site
    #: includes it — combat deathrattles always skip the dead source.
    ALL_FRIENDLY = auto()
    #: friendlies matching ``tribe``, source included if it matches
    FRIENDLY_OF_TRIBE = auto()
    #: friendlies matching ``tribe``, source always excluded
    OTHER_OF_TRIBE = auto()
    #: friendlies carrying ``keyword``
    FRIENDLY_WITH_KEYWORD = auto()
    #: the board slots either side of the source
    ADJACENT = auto()


@dataclass(frozen=True)
class BuffMatching:
    """+``attack``/+``health`` to every friendly matching ``target``.

    Composed replacement for four classes that shared this body and differed
    only in the predicate: ``BuffAllFriendlyMinions``, ``BuffAllFriendlyOfTribe``,
    ``BuffAllOtherOfTribe`` and ``BuffAllWithKeyword``. ``tribe`` is read only
    for the two ``*_OF_TRIBE`` targets, ``keyword`` only for
    ``FRIENDLY_WITH_KEYWORD``.

    Field names are load-bearing beyond this module: ``attack`` / ``health``
    are read by name for golden doubling (``triple_effects._GOLDEN_INT_FIELDS``)
    and the v12 static table (``card_static.NUMBER_FIELDS``); ``tribe`` and
    ``keyword`` are probed by name by the v5 ability-token encoder.
    """

    target: BuffTarget
    attack: int = 0
    health: int = 0
    tribe: Any = None
    keyword: Optional[Keyword] = None
    #: Stop after this many matches, in board order — 0 means "everyone who
    #: matches". ``limit=1`` is how a card says "your left-most Dragon"
    #: (Thousandth Paper Drake). Deliberately outside ``_GOLDEN_INT_FIELDS``:
    #: a golden printing buffs its one target twice as hard, it does not find
    #: a second target.
    limit: int = 0
    #: A keyword handed to everyone the buff reaches. Distinct from ``keyword``
    #: above, which *matches* on one (``FRIENDLY_WITH_KEYWORD``) — the same
    #: split ``BuffTargetFriendlyBattlecry`` already draws.
    grant_keyword: Optional[Keyword] = None


@dataclass(frozen=True)
class GrantKeywordRandomFriendly:
    """Random eligible friendly gains a keyword (shop battlecry or combat deathrattle)."""

    keyword: Keyword
    filter_race: Optional[Any] = None
    exclude_self: bool = True
    repeats: int = 1


@dataclass(frozen=True)
class BuffSelfWhenFriendlyDeathrattlePlaced:
    """Shop: after a friendly with Deathrattle is played, buff this minion."""

    attack: int = 1
    health: int = 2


@dataclass(frozen=True)
class BuffSelfWhenFriendlyBattlecryPlaced:
    """Shop: source gains stats after another friendly with an ``ON_PLACE`` ability is placed."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class BuffRandomOtherFriendlyCombat:
    """Combat deathrattle: one random other friendly (Tortollan Shellraiser).

    ``filter_race`` narrows it to one tribe ("Deathrattle: give a friendly
    Undead +1/+2"). Not a golden-doubled field — a golden printing buffs one
    minion twice as hard, not two minions.
    """

    attack: int = 0
    health: int = 0
    filter_race: Optional[Any] = None


@dataclass(frozen=True)
class DealDamageRandomEnemyMinion:
    """Combat deathrattle: deal ``amount`` to one random enemy minion (Kaboom Bot)."""

    amount: int
    repeats: int = 1


@dataclass(frozen=True)
class DealDamageLeftmostEnemyMinion:
    """Combat overkill/deathrattle: deal ``amount`` to the leftmost alive enemy minion."""

    amount: int


@dataclass(frozen=True)
class DealDamageAllMinions:
    """Combat deathrattle: deal ``amount`` to every alive minion on both sides."""

    amount: int


@dataclass(frozen=True)
class BuffDeadMinionNeighborsEffect:
    """Combat: when a filtered friendly dies, buff its immediate board neighbors."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class TransferAttackToRandomFriendlyEffect:
    """Combat deathrattle: give this minion's Attack to a random other friendly."""

    exclude_self: bool = True


@dataclass(frozen=True)
class SummonRandomAndCopyToHandEffect:
    """Combat deathrattle: summon random ``race_filter`` minion and queue a hand copy."""

    race_filter: Optional[Any] = None
    count: int = 1
    exclude_source: bool = True


@dataclass(frozen=True)
class StartOfCombatDamagePerFriendlyTribe:
    """Start of Combat: deal ``amount_per_match`` × friendly ``tribe`` count to one random enemy."""

    tribe: Any
    amount_per_match: int = 1
    repeats: int = 1


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
class SummonRandomOnSelfDamagedEffect:
    """Combat: after this minion takes damage, summon random minion(s) of ``race_filter``."""

    race_filter: Optional[Any] = None
    count: int = 1
    grant_taunt: bool = False


@dataclass(frozen=True)
class PogoHopperBattlecry:
    """Shop: +attack_each/+health_each for each other Pogo-Hopper already played; then increment counter."""

    attack_each: int = 2
    health_each: int = 2


@dataclass(frozen=True)
class StatAura:
    """Continuous +stats to every other living friendly matching ``target``.

    One class for what used to be four that differed only in the predicate --
    plain (Raid Leader), by tribe ('your other Murlocs'), by keyword (Phalanx
    Commander), and by board adjacency (Dire Wolf Alpha). The source never
    buffs itself; that is the aura machinery's rule, not the target's.

    Shares ``BuffTarget`` with :class:`BuffMatching` on purpose: "who does this
    hit" is one vocabulary, whether the answer is applied once or recomputed
    every step.
    """

    target: BuffTarget = BuffTarget.ALL_FRIENDLY
    attack: int = 0
    health: int = 0
    tribe: Any = None
    keyword: Optional[Keyword] = None


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
    # Houndmaster hands out Taunt with the stats; Toxfin is keyword-only (0/0).
    grant_keyword: Optional[Keyword] = None


@dataclass(frozen=True)
class BuffTargetFromPiratesBoughtBattlecry:
    """Shop battlecry: buff target by +stats per pirate bought this turn."""

    attack_per: int = 1
    health_per: int = 1
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
    """+0/+X where X = ``health_per_damage`` × total hero damage taken (Annihilan battlecry)."""

    health_per_damage: int = 1


@dataclass(frozen=True)
class SummonFirstDeadFriendlyMechsThisCombat:
    """Deathrattle: summon shallow copies of the first ``count`` dead friendly Mech corpses (board order)."""

    count: int = 2


class MultiplierKind(Enum):
    """What a :class:`Multiplier` scales.

    Each kind has a scope that follows from the card text, not from the code:
    battlecries only fire on play, deathrattles only in combat, and Khadgar's
    "your cards that summon minions" has no phase clause at all -- which is
    why the summon multiplier applies in both phases and the other two do not.
    """

    #: Brann: ON_PLACE (battlecry) executions, shop only
    BATTLECRY = auto()
    #: Baron: ON_DEATH execution count, combat only
    DEATHRATTLE = auto()
    #: Khadgar: summon iterations, both phases
    SUMMON = auto()


@dataclass(frozen=True)
class Multiplier:
    """Aura that multiplies how many times something resolves.

    One class for what used to be three that differed only in which event they
    scaled, each with its own near-identical board scan.
    """

    kind: MultiplierKind
    factor: int = 1


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
class SetNextRollCostEffect:
    """Shop battlecry: next ``uses`` manual refreshes cost ``cost`` gold (then clears)."""

    cost: int = 0
    uses: int = 1


@dataclass(frozen=True)
class ReduceUpgradeCostEffect:
    """Shop battlecry: ``next_tier_up_cost`` reduced by ``amount`` until next level-up."""

    amount: int = 1


@dataclass(frozen=True)
class SummonSelfCopyFromHandEffect:
    """Start of Combat, fired by a card sitting in *hand*: summon a copy of it.

    The only effect in the engine whose source is not on the board, which is why
    it is its own type rather than a flag: the start-of-combat scan has to be
    told to look somewhere else for it.
    """


@dataclass(frozen=True)
class GrantKeywordAtAttackThreshold:
    """Latch: the first time this minion's Attack reaches ``threshold``, it keeps
    ``keyword`` for good ("Once this reaches 6 Attack, gain Divine Shield").

    A latch and not an aura, which is the whole difficulty: the keyword does not
    come off when the Attack later drops, and — for Divine Shield — a popped
    shield must not silently re-arm on the next recount. Both fall out of
    checking ``keyword not in minion.keywords`` before granting, since the
    keyword stays on the minion while ``has_shield`` is the flag that pops.
    """

    threshold: int
    keyword: Keyword


@dataclass(frozen=True)
class GainGoldThisTurnEffect:
    """Shop: grant ``amount`` gold when trigger fires (this turn only)."""

    amount: int = 1
    filter_race: Optional[Any] = None


@dataclass(frozen=True)
class BuffPlacedMinionEffect:
    """Buff the minion that was just played, not the listener.

    "Whenever you play or Magnetize a Mech, give it +3/+1" — the *it* is the
    newcomer, which is what separates this from every other listener on
    AFTER_FRIENDLY_MINION_PLACED: those pay the watcher.
    """

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class PlayBloodGemsOnAttackerEffect:
    """Combat: Gems onto whichever friendly just attacked.

    Sibling of :class:`BuffAttackerOnFriendlyAttackEffect` and separate from
    :class:`PlayBloodGemsEffect` for the same reason: the attacker is context
    the trigger carries, not a position on the board that ``BloodGemTarget``
    could name.
    """

    count: int = 1


@dataclass(frozen=True)
class RepeatPerCountEffect:
    """Fire ``effect`` once, then once more per thing counted on the board.

    "At the end of your turn, give adjacent minions +1 Attack. Repeat for each
    friendly Golden minion." A wrapper rather than a field on the inner effect,
    the way :class:`AvengeEffect` puts a counter in front of one: the repeat
    rule is the same whatever is being repeated.
    """

    source: Any
    effect: Any
    tribe: Any = None
    base_repeats: int = 1


@dataclass(frozen=True)
class PlaceFishbaitEffect:
    """Replace a tavern card with a Fishbait for the left-most Beast to attack."""


class ScopeKind(Enum):
    """What class of card a standing "this game" bonus reaches.

    Lives here with the other effect vocabularies (``BuffTarget``,
    ``BloodGemTarget``, ``CountSource``) because a package's bindings name it,
    and a package may only import ``bg_core``.
    """

    #: One printed card, by id ("your Beetles", "each Eternal Knight").
    CARD = auto()
    #: A tribe, wherever its members are ("your Undead have +1 Attack").
    TRIBE = auto()
    #: Whatever is on the tavern counter, now and after every reroll.
    SHOP = auto()


@dataclass(frozen=True)
class RaiseStandingBonusEffect:
    """Open or raise a "this game" bonus on the seat.

    ``scope_kind``/``scope_key`` name what it reaches: a tribe ("your Undead
    have +1 Attack this game"), a printed card ("your Beetles have +2/+1"), or
    the tavern counter ("minions in the Tavern have +5/+5"). ``scope_key`` of
    ``None`` on a CARD scope means *this card* — the shape "has +X for each
    other <me> you've summoned" takes.

    The bonus follows the class of card, not the board: something bought or
    summoned afterwards arrives already carrying it.
    """

    scope_kind: Any
    attack: int = 0
    health: int = 0
    scope_key: Any = None


@dataclass(frozen=True)
class GiveLockboxEffect:
    """Hand the seat a Lockbox, or hurry the one it already has along.

    Both halves of "Get a Lockbox. If you already have one, it opens 1 turn
    sooner" — which is one rule, since a seat only ever holds one.
    """

    sooner: int = 1


@dataclass(frozen=True)
class AddTavernSpellToHandEffect:
    """Put a named Tavern spell in hand ("Battlecry: Get a Tavern Coin").

    Names the card rather than rolling one: these battlecries print which spell
    they hand over. A package that does not carry it hands over nothing.
    """

    card_id: str


@dataclass(frozen=True)
class ReduceTavernSpellCostEffect:
    """Shop: the next Tavern spell bought costs ``amount`` less (Ominous Seer).

    Sibling of :class:`ReduceUpgradeCostEffect`, and separate for the same
    reason the two prices are separate: a discount banked against the tavern
    upgrade must not follow the seat onto the spell counter.
    """

    amount: int = 1


@dataclass(frozen=True)
class StealTavernMinionEffect:
    """Take a minion off the tavern counter into hand, free (Enchanted Lasso,
    Decoy Conjurer). Not a purchase: no gold changes hands and the slot empties.

    ``highest_attack`` picks the biggest instead of one at random, which is the
    difference between the two cards printing this.
    """

    highest_attack: bool = False


@dataclass(frozen=True)
class DiscoverMinionAtTierEffect:
    """Discover a minion of exactly ``tier`` ("Discover a Tier 1 minion").

    Distinct from the triple-reward Discover, which reads its tier off the
    seat's tavern tier; this one is printed on the card and never moves.
    """

    tier: int = 1


@dataclass(frozen=True)
class GainGoldNextTurnEffect:
    """Shop: bank ``amount`` gold for the *next* turn (Southsea Busker).

    A separate effect rather than a flag on :class:`GainGoldThisTurnEffect`,
    because the two are spent from different places: this turn's gold is added
    to a seat that may already have spent some, and next turn's is added to a
    coin count that has not been set yet.
    """

    amount: int = 1


@dataclass(frozen=True)
class AddTokenToHandEffect:
    """Shop ON_SELL / battlecry: add ``token_id`` to first free hand slot."""

    token_id: str
    count: int = 1


@dataclass(frozen=True)
class IncrementShopTribeBonusEffect:
    """After playing a tribe: permanent +stats for that tribe in shop (Nomi)."""

    tribe: Any
    attack: int = 1
    health: int = 1


@dataclass(frozen=True)
class AdaptAllMurlocsEffect:
    """Battlecry: Adapt your Murlocs — pick 3 of 10, apply to all friendly Murlocs."""

    repeats: int = 1


@dataclass(frozen=True)
class AdaptSelfRandomEffect:
    """Battlecry: apply random adapts to self (Amalgadon — no modal)."""

    repeats: int = 1
    count_from_unique_other_tribes: bool = False


@dataclass(frozen=True)
class TriggerRandomFriendlyDeathrattleEffect:
    """Combat after-attack: trigger a random living friendly minion's deathrattle."""

    repeats: int = 1
    exclude_self: bool = True


@dataclass(frozen=True)
class MultiplySelfAttackEffect:
    """Combat after-attack: multiply this minion's current Attack (Glyph Guardian)."""

    factor: int = 2


@dataclass(frozen=True)
class BuffAttackerOnFriendlyAttackEffect:
    """Combat: when another friendly attacks, buff the attacker if tribe matches."""

    tribe: Any
    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class AttackImmediatelyAfterSurvivingEffect:
    """Combat: after surviving an attack, this minion attacks immediately (Yo-Ho-Ogre)."""


@dataclass(frozen=True)
class BuffRandomUniqueTribeFriendlies:
    """Shop battlecry: buff up to ``count`` random friendlies with distinct tribes."""

    count: int = 3
    attack: int = 1
    health: int = 1
    exclude_self: bool = True


@dataclass(frozen=True)
class BuffAllShopOffersEffect:
    """Shop ON_SELL / battlecry: buff every minion currently in the tavern offers."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class AddRandomMinionToShopEffect:
    """Shop battlecry: add a random ``tribe`` minion to an empty offer slot."""

    tribe: Any
    freeze_slot: bool = False


@dataclass(frozen=True)
class ConsumeFriendlyBattlecry:
    """Shop battlecry: remove a friendly minion to gain its stats and gold."""

    filter_race: Optional[Any] = None
    exclude_self: bool = True
    gold_reward: int = 3
    stat_multiplier: int = 1


@dataclass(frozen=True)
class AddFromLastOpponentBoardEffect:
    """Shop battlecry: add a random minion from ``last_opponent_board`` to hand."""

    make_golden: bool = False


@dataclass(frozen=True)
class TransformIntoShopMinionEffect:
    """Shop battlecry: transform source into a plain copy of a random shop offer."""

    copy_golden: bool = False


@dataclass(frozen=True)
class GrantKeywordAllFriendlyOfTribe:
    """Grant ``keyword`` to every friendly minion of ``tribe`` (combat deathrattle)."""

    keyword: Keyword
    tribe: Any


class BloodGemTarget(Enum):
    """Who a card's own Blood Gems land on.

    Every shape printed on a live Quilboar card: "on itself", "on adjacent
    minions", "on all your other minions", "on all your Quilboar". Members are
    part of the effect's identity — add, never rename.
    """

    SELF = auto()
    ADJACENT = auto()
    ALL_FRIENDLY = auto()
    ALL_OTHER_FRIENDLY = auto()
    ALL_FRIENDLY_QUILBOAR = auto()


@dataclass(frozen=True)
class GainBloodGemsEffect:
    """Put ``count`` Blood Gems in hand ("Battlecry: Get 2 Blood Gems").

    ``quilboar_keyword`` covers the printings that hand out a Gem which also
    gives a Quilboar Taunt / Reborn / Divine Shield.
    """

    count: int = 1
    quilboar_keyword: Optional[Keyword] = None


@dataclass(frozen=True)
class PlayBloodGemsEffect:
    """The source plays ``count`` Gems itself, on ``target``.

    Distinct from :class:`GainBloodGemsEffect`: nothing reaches hand and the
    seat makes no choice — the card names who gets them.

    ``permanent`` is printed on the card and only means anything in combat: a
    Gem played mid-fight normally dies with the combat copy, and "permanent"
    is what sends it through to the owner's real board (Skulking Bristlemane's
    deathrattle, Razorfen Vineweaver's and Timewarped Bonker's Rally).
    """

    target: BloodGemTarget
    count: int = 1
    permanent: bool = False


@dataclass(frozen=True)
class IncreaseBloodGemBonusEffect:
    """"Your Blood Gems give an extra +1/+1 this game."

    Raises the value of every *future* Gem for this seat; Gems already played
    keep the stats they gave. Attack and Health move independently because Gem
    Day grants one or the other.
    """

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class ChooseOneEffect:
    """Two effects printed on one card; the player takes one.

    Always exactly two on every live printing ("Choose One - Get 2 Blood Gems;
    or Get a Gem Day"), so the pair is named rather than a list of unknown
    length. Both halves are ordinary effects, which is what lets Thorned
    Trailblazer and Fandral's Fortune hand out *both* without a second code
    path: the resolver applies one option, or two.
    """

    first: Any
    second: Any


@dataclass(frozen=True)
class GrantTemporaryBuffEffect:
    """Stats and/or a keyword on one minion, gone by the owner's next turn.

    What every Spellcraft spell does ("Give a minion +2/+6 and Taunt until next
    turn"). ``keyword_if_race`` covers Waverider, whose spell gives +2/+2 to
    anyone but Windfury only to a Naga.
    """

    attack: int = 0
    health: int = 0
    keyword: Optional[Keyword] = None
    keyword_if_race: Optional[Any] = None


@dataclass(frozen=True)
class CreateSpellcraftSpellEffect:
    """Put this minion's Spellcraft spell in hand.

    The spell is described here rather than looked up, because that is how the
    card reads: the Naga's text *is* the spell's text. It is handed out when
    the minion is played and again at the start of every turn, and it is
    discarded unused at end of turn. A golden Naga makes a golden spell with
    double the effect, which is the ``buff``'s doubled stats.
    """

    buff: GrantTemporaryBuffEffect
    card_id: str = ""
    name: str = ""


class CountSource(Enum):
    """What :class:`BuffSelfPerCount` counts on the owner's board.

    Ordering is irrelevant to gameplay but each member is part of the effect's
    observation identity (see ``_EFFECT_SIGNATURES`` in ``minibg.obs``), so
    renaming a member is an obs change — add, don't rename.
    """

    FRIENDLY_OF_TRIBE = auto()
    UNIQUE_TRIBES = auto()
    GOLDEN_FRIENDLIES = auto()


@dataclass(frozen=True)
class BuffSelfPerCount:
    """+stats on the listener, once per thing counted on its own board.

    Composed replacement for what used to be three near-identical classes
    (``BuffSelfFrom{FriendlyTribe,UniqueTribe,GoldenFriendly}Count``): they
    shared this exact body and differed only in ``source``. ``tribe`` is read
    only when ``source is FRIENDLY_OF_TRIBE``.

    Field names are load-bearing beyond this module: ``attack_per`` /
    ``health_per`` are read by name for golden doubling
    (``triple_effects._GOLDEN_INT_FIELDS``) and for the v12 static card table
    (``card_static.NUMBER_FIELDS``). Renaming them silently changes both.
    """

    source: CountSource
    tribe: Any = None
    attack_per: int = 1
    health_per: int = 1
    exclude_self: bool = True


@dataclass(frozen=True)
class BuffLeftmostRepeatedEffect:
    """Shop turn end: buff leftmost minion, repeat from a ``PlayerState`` counter field."""

    counter: str
    attack: int = 1
    health: int = 1


@dataclass(frozen=True)
class BuffRandomFriendlyFromPlacedTierEffect:
    """Shop: after a filtered friendly is played, buff a random friendly by its tier."""

    attack_per_tier: bool = True
    health_per_tier: bool = True
    exclude_self: bool = False


@dataclass(frozen=True)
class DealExcessDamageToAdjacentEffect:
    """Combat ON_OVERKILL: deal excess kill damage to adjacent enemy minion(s)."""

    both_adjacent: bool = False


@dataclass(frozen=True)
class AddRandomMinionToHandOnKillEffect:
    """Combat ON_AFTER_ATTACK: if this minion killed an enemy, queue a random minion for hand."""

    tribe: Optional[Any] = None
    count: int = 1


@dataclass(frozen=True)
class AddRandomMinionToHandEffect:
    """Shop battlecry: add a random ``tribe`` minion to hand.

    ``tier`` pins the draw to one tavern tier ("get a random Tier 1 minion",
    River Skipper). Left ``None`` the draw is the usual one — anything up to the
    seat's own tavern tier — which is what every card without a printed tier
    means. It is deliberately not in ``_GOLDEN_INT_FIELDS``: a golden printing
    hands out two cards of the same tier, never one of a higher tier.
    """

    tribe: Optional[Any] = None
    tier: Optional[int] = None


@dataclass(frozen=True)
class BuffAttackedMinionEffect:
    """Combat listener: buff the friendly minion that was attacked."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class BuffAdjacentOnAttackedEffect:
    """Combat: when this minion is attacked, buff adjacent friendlies."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class GainGoldOnDeathEffect:
    """Combat deathrattle stub for Gold Coin (grants gold after battle)."""

    amount: int = 1


Effect = Union[
    SummonEffect,
    SummonRandomMinionEffect,
    BuffRandomFriendly,
    BuffOnePerListedTribeFriendly,
    BuffMatching,
    GrantKeywordRandomFriendly,
    BuffSelfWhenFriendlyDeathrattlePlaced,
    BuffSelfWhenFriendlyBattlecryPlaced,
    BuffTargetFromPiratesBoughtBattlecry,
    SummonRandomOnSelfDamagedEffect,
    BuffLeftmostRepeatedEffect,
    BuffRandomFriendlyFromPlacedTierEffect,
    DealExcessDamageToAdjacentEffect,
    AddRandomMinionToHandOnKillEffect,
    BuffRandomOtherFriendlyCombat,
    DealDamageRandomEnemyMinion,
    DealDamageLeftmostEnemyMinion,
    DealDamageAllMinions,
    BuffDeadMinionNeighborsEffect,
    TransferAttackToRandomFriendlyEffect,
    SummonRandomAndCopyToHandEffect,
    StartOfCombatDamagePerFriendlyTribe,
    AttackBonusPerOtherMurlocGlobal,
    BuffSummonedIfRace,
    GrantListenerKeywordIfSummonedMatches,
    BuffListenerIfSummonedMatches,
    SummonOnSelfDamaged,
    PogoHopperBattlecry,
    StatAura,
    BuffAdjacentBattlecry,
    BuffTargetFriendlyBattlecry,
    HeroImmuneAura,
    DealHeroDamage,
    BuffSelf,
    BuffSelfFromHeroDamageTaken,
    SummonFirstDeadFriendlyMechsThisCombat,
    ZappTargeting,
    CleaveOnAttack,
    DiscoverMurlocEffect,
    AdaptAllMurlocsEffect,
    AdaptSelfRandomEffect,
    TriggerRandomFriendlyDeathrattleEffect,
    MultiplySelfAttackEffect,
    BuffAttackerOnFriendlyAttackEffect,
    AttackImmediatelyAfterSurvivingEffect,
    BuffRandomUniqueTribeFriendlies,
    BuffAllShopOffersEffect,
    AddRandomMinionToShopEffect,
    ConsumeFriendlyBattlecry,
    AddFromLastOpponentBoardEffect,
    TransformIntoShopMinionEffect,
    GrantKeywordAllFriendlyOfTribe,
    BuffSelfPerCount,
    AddRandomMinionToHandEffect,
    BuffAttackedMinionEffect,
    BuffAdjacentOnAttackedEffect,
    GainGoldOnDeathEffect,
    SetNextRollCostEffect,
    ReduceUpgradeCostEffect,
    GainGoldThisTurnEffect,
    GainGoldNextTurnEffect,
    BuffPlacedMinionEffect,
    RaiseStandingBonusEffect,
    PlayBloodGemsOnAttackerEffect,
    RepeatPerCountEffect,
    PlaceFishbaitEffect,
    GiveLockboxEffect,
    AddTavernSpellToHandEffect,
    ReduceTavernSpellCostEffect,
    StealTavernMinionEffect,
    DiscoverMinionAtTierEffect,
    GrantKeywordAtAttackThreshold,
    SummonSelfCopyFromHandEffect,
    AddTokenToHandEffect,
    IncrementShopTribeBonusEffect,
]


@dataclass(frozen=True)
class AvengeEffect:
    """Avenge (N): fire ``effect`` once every ``count`` friendly deaths.

    Carried by an ``Ability`` on ``Trigger.ON_FRIENDLY_MINION_DIED`` rather than
    a trigger of its own: Avenge *is* that trigger with a counter in front of
    it, and a new trigger id would resize the ability vocabulary every trained
    network embeds (see ``NUM_TRIGGER_IDS``).

    The count is per minion and lives on the combat copy, so it resets between
    combats — a board that loses two minions each fight never accumulates its
    way to an Avenge (3).
    """

    count: int
    effect: Any


@dataclass(frozen=True)
class Ability:
    trigger: Trigger
    effect: Effect
    """If set: filter placed/dead minion race, or killer race for ``ON_FRIENDLY_KILL``."""

    filter_race: Optional[Any] = None
    condition: Optional[Condition] = None
    filter_victim_keyword: Optional[Keyword] = None
    combat_only: bool = False
    #: Gold an ``ON_ACTIVATE`` ability costs to fire. Every printing charges 1
    #: or 2; on any other trigger it is meaningless and stays 0.
    activate_cost: int = 0


__all__ = [
    "Keyword",
    "Trigger",
    "ConditionKind",
    "Condition",
    "AvengeEffect",
    "SummonEffect",
    "SummonRandomMinionEffect",
    "BuffRandomFriendly",
    "BuffOnePerListedTribeFriendly",
    "BuffMatching",
    "BuffTarget",
    "Multiplier",
    "MultiplierKind",
    "GrantKeywordRandomFriendly",
    "BuffSelfWhenFriendlyBattlecryPlaced",
    "BuffRandomOtherFriendlyCombat",
    "DealDamageRandomEnemyMinion",
    "DealDamageLeftmostEnemyMinion",
    "DealDamageAllMinions",
    "BuffDeadMinionNeighborsEffect",
    "TransferAttackToRandomFriendlyEffect",
    "SummonRandomAndCopyToHandEffect",
    "StartOfCombatDamagePerFriendlyTribe",
    "AttackBonusPerOtherMurlocGlobal",
    "BuffSummonedIfRace",
    "GrantListenerKeywordIfSummonedMatches",
    "BuffListenerIfSummonedMatches",
    "SummonOnSelfDamaged",
    "PogoHopperBattlecry",
    "StatAura",
    "BuffAdjacentBattlecry",
    "BuffTargetFriendlyBattlecry",
    "BuffTargetFromPiratesBoughtBattlecry",
    "BuffSelfWhenFriendlyDeathrattlePlaced",
    "SummonRandomOnSelfDamagedEffect",
    "BuffLeftmostRepeatedEffect",
    "BuffRandomFriendlyFromPlacedTierEffect",
    "DealExcessDamageToAdjacentEffect",
    "AddRandomMinionToHandOnKillEffect",
    "HeroImmuneAura",
    "DealHeroDamage",
    "BuffSelf",
    "BuffSelfFromHeroDamageTaken",
    "SummonFirstDeadFriendlyMechsThisCombat",
    "ZappTargeting",
    "CleaveOnAttack",
    "DiscoverMurlocEffect",
    "AdaptAllMurlocsEffect",
    "AdaptSelfRandomEffect",
    "TriggerRandomFriendlyDeathrattleEffect",
    "MultiplySelfAttackEffect",
    "BuffAttackerOnFriendlyAttackEffect",
    "AttackImmediatelyAfterSurvivingEffect",
    "BuffRandomUniqueTribeFriendlies",
    "BuffAllShopOffersEffect",
    "AddRandomMinionToShopEffect",
    "ConsumeFriendlyBattlecry",
    "AddFromLastOpponentBoardEffect",
    "TransformIntoShopMinionEffect",
    "GrantKeywordAllFriendlyOfTribe",
    "BloodGemTarget",
    "GainBloodGemsEffect",
    "PlayBloodGemsEffect",
    "IncreaseBloodGemBonusEffect",
    "BuffSelfPerCount",
    "CountSource",
    "AddRandomMinionToHandEffect",
    "BuffAttackedMinionEffect",
    "BuffAdjacentOnAttackedEffect",
    "GainGoldOnDeathEffect",
    "SetNextRollCostEffect",
    "ReduceUpgradeCostEffect",
    "GainGoldThisTurnEffect",
    "GainGoldNextTurnEffect",
    "BuffPlacedMinionEffect",
    "ScopeKind",
    "RaiseStandingBonusEffect",
    "PlayBloodGemsOnAttackerEffect",
    "RepeatPerCountEffect",
    "PlaceFishbaitEffect",
    "GiveLockboxEffect",
    "AddTavernSpellToHandEffect",
    "ReduceTavernSpellCostEffect",
    "StealTavernMinionEffect",
    "DiscoverMinionAtTierEffect",
    "GrantKeywordAtAttackThreshold",
    "SummonSelfCopyFromHandEffect",
    "AddTokenToHandEffect",
    "IncrementShopTribeBonusEffect",
    "Effect",
    "Ability",
]

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
    #: shop: this card is in *hand* and something happened on the board
    #: ("while this is in your hand, after you play a Murloc, gain +6/+6"). The
    #: only listener in the game that is not on the board.
    WHILE_IN_HAND = auto()
    #: shop: the seat kept a card out of a Discover, whatever offered it.
    ON_DISCOVERED = auto()
    #: shop: the seat's hero just took combat damage. Listeners may undo it —
    #: which is why they run *before* the health is written, not after.
    ON_HERO_DAMAGE = auto()
    #: shop: the seat cast a Tavern spell, after it resolved. Narrower than
    #: ON_TARGETED_BY_SPELL on purpose — that one is "a spell hit this minion",
    #: this one is "the seat cast one at all", and Blood Gems are not Tavern
    #: spells (see ``SpellCard``).
    ON_TAVERN_SPELL_CAST = auto()
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
    #: A friendly minion came back from Reborn. Combat-only: nothing dies in a
    #: tavern, so nothing is reborn there either.
    ON_FRIENDLY_REBORN = auto()


class ConditionKind(Enum):
    OTHER_TRIBE_ON_BOARD = auto()
    LAST_COMBAT_WON = auto()


@dataclass(frozen=True)
class Condition:
    """A precondition on an ability.

    ``negate`` reads the same test the other way ("if you **lost** your last
    combat"). A field rather than a second ConditionKind: that enum sizes an
    embedding table every trained network carries, and an inverse is not a new
    question.
    """

    kind: ConditionKind
    tribe: Optional[Any] = None
    negate: bool = False


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
    #: Land the summon on these stats instead of its printed ones ("summon a
    #: random Beast. Set its stats to 6/6"). Set, not added: the card says what
    #: the body ends up as, whatever it was.
    set_attack: int = 0
    set_health: int = 0


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
    #: Leave the source out ("give your **other** minions +4/+2"). A flag rather
    #: than a fifth target, because "other" is a question about the source and
    #: not about who matches — which is why ``OTHER_OF_TRIBE`` exists at all,
    #: and why it needed a tribe it did not otherwise care about.
    #: ``BloodGemTarget`` has carried ALL_OTHER_FRIENDLY all along; this is the
    #: same idea, said once and composable with every target.
    exclude_source: bool = False
    #: Fire only when the minion just played has a Battlecry (Kalecgos). A field
    #: rather than a ``ConditionKind`` because that vocabulary sizes an
    #: embedding table every trained network carries, and rather than a gate on
    #: the target — it is one card's rule, and gating the whole
    #: FRIENDLY_OF_TRIBE variant on it silently broke the next card to use it.
    requires_placed_battlecry: bool = False


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
    """When a friendly minion is summoned, buff it if it matches ``tribe`` (Pack Leader, Mama Bear).

    ``improves`` is Lurking Leviathan's "and improve this permanently": the
    payout is multiplied by how many times this body has already paid, and the
    tally is the body's own rather than the seat's.
    """

    tribe: Any
    attack: int = 0
    health: int = 0
    improves: bool = False


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
    """Stats onto the card that owns the ability, and optionally a keyword.

    ``keyword`` is Barrier Banshee's "gain Divine Shield and +7/+7" — one clause
    on the card, so one effect here rather than a second ability whose ordering
    against this one would be nobody's decision.
    """

    attack: int = 0
    health: int = 0
    keyword: Optional[Keyword] = None


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
    #: Drakkari Enchanter: ON_TURN_END executions, shop only
    END_OF_TURN = auto()
    #: Balinda Stonehearth: casts of a spell aimed at a friendly minion, shop
    #: only — a spell cast from inside a fight has no seat to aim from.
    TARGETED_SPELL = auto()


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
class DiscoverTribeEffect:
    """Discover a minion of one tribe (tier-weighted pool).

    Printed by four cards on four different tribes — Primalfin Lookout wants a
    Murloc, Maw Caster an Undead, Imposing Percussionist a Demon, Clunker
    Junker a Mech — and they differ by nothing else. ``repeats`` stacks with
    Brann (product).
    """

    tribe: Any = None
    repeats: int = 1
    #: "Discover a Mech **to Magnetize to it**" — the pick is welded onto a
    #: friendly the seat named rather than landing in hand, so this Discover
    #: needs no hand slot and is opened by the targeted-battlecry path.
    magnetize_onto_target: bool = False


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

    source: Any = None
    effect: Any = None
    tribe: Any = None
    base_repeats: int = 1
    #: Read a seat tally instead of the board ("improved by every 4 spells
    #: you've cast this game"). The level starts at ``base_repeats`` — one, so
    #: an unimproved card is worth exactly what it prints — and rises by one per
    #: ``per`` events counted.
    counter: str = ""
    per: int = 1


@dataclass(frozen=True)
class PlaceFishbaitEffect:
    """Replace a tavern card with a Fishbait for the left-most Beast to attack.

    Two printings. Lurking Lionfish names the slot and stops there. Snarky
    Shark names none: ``refresh`` rerolls the tavern first, and ``auto_attack``
    sends the Beast without waiting to be told to.
    """

    refresh: bool = False
    auto_attack: bool = False


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
    #: Whatever is on the tavern counter, now and after every reroll. Takes
    #: the optional filters the printed cards use: a tribe ("give Elementals in
    #: the Tavern +8/+8") and a tier cap ("minions in the Tavern from Tier 3
    #: and below"). What it hands out is *kept* when the minion is bought.
    SHOP = auto()


@dataclass(frozen=True)
class IncreaseTavernSpellBonusEffect:
    """"Your Tavern spells give an extra +1 Attack this game."

    Sibling of :class:`IncreaseBloodGemBonusEffect`, and separate because they
    are separate buffs: a card that raises what a Gem is worth says nothing
    about what a Tavern spell is worth.
    """

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class AddRandomTavernSpellToHandEffect:
    """"Get a random Tavern spell", with the filters the cards print.

    ``max_cost`` is "get two **1-Cost** Tavern spells"; ``gives_stats`` is "a
    random Tavern spell **that gives stats**", which is asked of the spell's
    own bindings rather than its text.
    """

    count: int = 1
    max_cost: int = 0
    gives_stats: bool = False


@dataclass(frozen=True)
class DiscoverTavernSpellEffect:
    """"Discover a Tavern spell" — three offered, the seat keeps one.

    ``repeats`` is the Golden that opens the modal twice.
    """

    repeats: int = 1


@dataclass(frozen=True)
class CastRandomTavernSpellEffect:
    """"Cast a random Tavern spell (targets this if possible)."

    Cast, not acquired: it never reaches hand and costs nothing. ``self_target``
    is the parenthetical — a spell that needs a minion takes the caster.
    """

    self_target: bool = True


@dataclass(frozen=True)
class CopyLastTavernSpellEffect:
    """"Get a copy of the last Tavern spell you cast." Nothing cast, nothing
    copied — the seat remembers which one, not that one was cast.

    ``count`` is the Golden that hands over two of them.
    """

    count: int = 1


@dataclass(frozen=True)
class BumpSeatCounterEffect:
    """Count one event on a named seat tally.

    The other half of "improves": the card that says it improves also says what
    improves it, and this is that event. Ordered after the effect it improves,
    because the cards say *future* ("Improve your future Ballers").
    """

    counter: str


@dataclass(frozen=True)
class BuffOnSpellCastOnTribeEffect:
    """Watch every spell the seat casts at a friendly of ``tribe``.

    Distinct from ``ON_TARGETED_BY_SPELL``, which is a card watching spells cast
    at *itself*: Torrential Ruiner is watching the board.

    ``effect`` is for the watchers that pay somewhere the flat stats cannot
    reach — Shamanic Tidecaller pays Murlocs in hand as well as on the board.
    """

    tribe: Any = None
    attack: int = 0
    health: int = 0
    effect: Any = None


@dataclass(frozen=True)
class BuffHandMinionsEffect:
    """Stats onto minions in the owner's *hand*.

    ``leftmost`` pays only the first ("give the left-most minion in your hand
    +6/+6"); ``tribe`` narrows it; neither set pays every minion there. The hand
    is the seat's, so a combat reaching it goes through the seat.
    """

    attack: int = 0
    health: int = 0
    leftmost: bool = False
    tribe: Any = None
    also_board: bool = False


@dataclass(frozen=True)
class GainStatsFromHandEffect:
    """Take stats off the cards waiting in hand.

    Two printings: the biggest Attack in hand (Costume Enthusiast) and the sum
    of everything there (Choral Mrrrglr). Both are Start of Combat reads, and
    the hand is reached through the seat.
    """

    highest_attack_only: bool = False


@dataclass(frozen=True)
class GiveOwnStatsToHandEffect:
    """Hand this body's stats to the left-most minion in hand (Futurefin)."""


@dataclass(frozen=True)
class DoubleBountiesEffect:
    """Proud Privateer: this seat's Bounties resolve twice.

    A standing property of the seat rather than a change to the cards: the
    Bounty in hand is the same card, and casting it is what happens twice.
    """


@dataclass(frozen=True)
class AddRandomGoldenMinionEffect:
    """Get a random Golden minion of one tier, owing no Triple Reward.

    Made rather than forged, so nothing is owed — the same distinction
    :class:`MakeFriendlyGoldenEffect` draws.
    """

    tier: int = 1


@dataclass(frozen=True)
class AddRandomMinionOfCommonTribeEffect:
    """Get a random minion of the tribe the seat has most of.

    Ties go to nobody in particular; a board with no tribe at all fetches
    nothing, because there is no most-common type to name.
    """


@dataclass(frozen=True)
class MakeFriendlyGoldenEffect:
    """Turn a friendly into its Golden printing.

    ``max_tier`` is the cap the card prints ("from Tier 6 or below"). The copy
    gives no Triple Reward, because nothing was tripled — it was made.
    """

    max_tier: int = 0


@dataclass(frozen=True)
class BuffTargetPerGoldSpentEffect:
    """A targeted buff scaled by the gold spent *this turn*.

    Lovesick Balladist: "Give a Pirate +1 Health. (Improved by each Gold you
    spent this turn!)" — a different tally from the "(5 Gold left!)" cards,
    which count without ever resetting.
    """

    attack: int = 0
    health: int = 0
    filter_race: Optional[Any] = None


@dataclass(frozen=True)
class BuffBoughtMinionEffect:
    """Pay the minion the seat just bought.

    ``double_stats`` is Stone Age Slab's second half, applied after the flat
    stats. ``once_per_turn`` is printed on it too, and the flag is spent by the
    buy that used it.
    """

    attack: int = 0
    health: int = 0
    double_stats: bool = False
    once_per_turn: bool = False


@dataclass(frozen=True)
class StatsFromNextBuyEffect:
    """Living Prison: take the stats of the next minion bought this turn.

    A promise rather than an effect that resolves now — the minion it reads has
    not been bought yet — so it is remembered on the body and spent by that buy.
    """


@dataclass(frozen=True)
class GoldSpentResponseEffect:
    """Answer every ``threshold`` Gold the seat spends.

    "(5 Gold left!)" is a countdown that refills, so the running total lives on
    the body and the payload fires once per full threshold.
    """

    threshold: int
    effect: Any = None
    #: How many times the payload resolves per full threshold. The Golden
    #: printings say "twice" rather than printing bigger numbers, so this is
    #: what their doubling has to land on.
    repeats: int = 1


@dataclass(frozen=True)
class IncreaseTribeGiftEffect:
    """"Your Elementals give an extra +1 Attack this game."

    Raises what *each* Elemental-played grant hands out, not the running total —
    a modifier on a modifier, the same relationship the Blood Gem bonus has to a
    Gem. Sibling of :class:`IncreaseBloodGemBonusEffect` and
    :class:`IncreaseTavernSpellBonusEffect`, and separate for the same reason:
    they are different buffs.
    """

    tribe: Any = None
    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class BuffSharedTribeEffect:
    """Buff every friendly sharing a tribe with the minion this was cast at.

    "Choose a minion. Give all minions that share a type with it +3/+3" — the
    tribe is read off the target rather than named, which is what no fixed
    ``BuffMatching`` can say.
    """

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class CastSpellAtEffect:
    """Cast a named spell at a minion the position names, not the seat.

    "Cast Chef's Choice on the minion to the right", "cast Natural Blessing on
    adjacent minions" — the card picks the target, so nobody is asked.
    """

    card_id: str
    to_the_right: bool = False
    adjacent: bool = False


@dataclass(frozen=True)
class MagnetizeTokenEffect:
    """Magnetize a token onto a Mech, without it ever being in hand.

    "Whenever you play a Mech, Magnetize a 3/3 Satellite to it" — the Satellite
    is made on the spot and merged in, so everything that watches a
    Magnetization sees this one too.

    ``attack``/``health`` override the token's printed stats where the card
    names them: there is one Satellite card, printed 6/6, and Spark Snapper
    welds a 3/3 of it while Glambot welds it as printed.
    """

    token_id: str
    improves: str = ""
    attack: int = 0
    health: int = 0
    #: How many are welded per trigger. The Golden Glambot welds "twice".
    repeats: int = 1


@dataclass(frozen=True)
class MagnetizesToTribesEffect:
    """This magnet may attach to more than Mechs (Prosthetic Hand: "Can
    Magnetize to Mechs or Undead")."""

    tribes: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class DoubleNextMagnetizeEffect:
    """The next Magnetization onto this minion this turn lands twice."""


@dataclass(frozen=True)
class BuffPerMagnetizationEffect:
    """+stats to every friendly, once per Magnetization it carries."""

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class EchoMagnetizeEffect:
    """When a Magnetization lands elsewhere, it lands on this body too."""


@dataclass(frozen=True)
class AddRandomCardToHandEffect:
    """One of a named set of cards, at random ("get a random Chromadrake").

    The set is written into the binding because the card names a family the
    catalog has no flag for — five Chromadrakes, and nothing marks them as such.
    """

    card_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BuffSelfOnFriendlyDamageEffect:
    """Grow when another friendly deals damage in combat.

    ``permanent`` is the difference between a body that is bigger for this fight
    and one that is bigger afterwards — Devout Hellcaller says "permanently", so
    the gain is written back to the owner's minion as well as to the copy.
    """

    attack: int = 0
    health: int = 0
    filter_race: Optional[Any] = None
    permanent: bool = True


@dataclass(frozen=True)
class BuffSelfOnFriendlySoldEffect:
    """Grow when the seat sells a friendly ("after you sell an Elemental").

    Sits on ``Trigger.ON_SELL`` like the sold minion's own effects, and the type
    is what tells the two apart: this one belongs to a *watcher* on the board,
    not to the card leaving it.

    ``effect`` is for the watchers that do something other than grow — Twisted
    Wrathguard leaves a card in the next roll — and stats and effect are not
    exclusive, since a card could print both.
    """

    attack: int = 0
    health: int = 0
    filter_race: Optional[Any] = None
    effect: Any = None


@dataclass(frozen=True)
class BuffShopOnEveryRefreshEffect:
    """From now on, every tavern roll buffs one random minion in it.

    "After the Tavern is Refreshed this game" — a standing promise on the seat
    rather than a one-off, so it outlives the body that made it.
    """

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class RewardAtDamageDealtEffect:
    """Once this body has dealt ``threshold`` damage, hand the owner a card.

    The tally is damage dealt **in combat**, and it carries across fights —
    "(40 left!)" counts down over the whole game, not over one battle. The card
    lands on the seat, so it survives the copy that earned it.
    """

    threshold: int
    card_id: str


@dataclass(frozen=True)
class RefreshesCostHealthEffect:
    """The first ``uses`` refreshes each turn are paid in Health, not Gold.

    The payment is hero damage and goes through ``apply_hero_damage``, which is
    what makes it interact with everything else that reads hero damage — armor
    absorbs it, and a card that undoes hero damage undoes this too.
    """

    amount: int = 1
    uses: int = 2


@dataclass(frozen=True)
class DestroyFriendlyEffect:
    """Destroy a friendly the seat picks, and pay for it.

    Four cards trade a body for something and differ only in what: a plain copy
    of it in hand (Disguised Graverobber), a Discover (Maw Caster), stats on the
    card that ate it (Dead Bellringer), a bonus on the whole tribe (Butchering).
    So the destruction is the effect and the payout is a field.

    ``get_copy`` is the Graverobber's half: *plain* means the printed card, so
    whatever the body had gained — buffs, granted keywords, Blood Gems — does
    not come with it, which is the whole cost of that trade. ``then`` is any
    other payout, applied with the destroyer as its source. ``grant_keyword``
    is handed to the victim before it goes.

    A destroyed minion in the tavern is a death, and is counted as one (Eternal
    Knight reads that tally). It does not fire deathrattles and Reborn does not
    return it: those are combat rules, and this happens in the recruit phase.
    """

    filter_race: Optional[Any] = None
    get_copy: bool = True
    grant_keyword: Optional[Keyword] = None
    then: Any = None
    #: "a **different** friendly Undead" — the destroyer may not eat itself.
    exclude_self: bool = False


@dataclass(frozen=True)
class SummonBestFromHandEffect:
    """Summon the highest-Attack minion in hand, for this combat only.

    The card stays in hand — what joins the fight is a copy, and it dies with
    the combat like any other. Reads the hand through the seat, the same way
    ``SummonSelfCopyFromHandEffect`` does. ``filter_race`` narrows it to one
    tribe ("the highest-Attack Murloc from your hand"), and ``count`` is the
    Golden that summons two of them.
    """

    filter_race: Optional[Any] = None
    count: int = 1


@dataclass(frozen=True)
class BuffRandomHandMinionEffect:
    """Stats onto a random minion in the owner's *hand*.

    Fired from combat ("whenever this takes damage, give a minion in your hand
    +2/+1"), so it goes through the seat: a hand is not something a combat copy
    has.
    """

    attack: int = 0
    health: int = 0


@dataclass(frozen=True)
class KeepCombatGainsEffect:
    """Tarecgosa: what this body gained in the fight, it keeps.

    Everything a combat normally throws away with the copy — stats bought by
    buffs, keywords granted mid-fight — is written back to the owner's real
    minion when the fight ends.

    ``adjacent`` grants it to the neighbours instead of to this body (Persistent
    Poet), and ``tribe`` narrows who among them.
    """

    adjacent: bool = False
    tribe: Any = None


@dataclass(frozen=True)
class GrantCombinedChooseOneEffect:
    """The next ``count`` Choose One cards take both halves.

    Thorned Trailblazer prints "(1 left!)", so the charge is handed out each
    turn rather than once when the card is played.
    """

    count: int = 1


@dataclass(frozen=True)
class MultiplySummonedAttackEffect:
    """Multiply the Attack of a friendly summoned in combat (Banana Slamma)."""

    tribe: Any = None
    factor: int = 2


@dataclass(frozen=True)
class GiveOwnStatsToSummonedEffect:
    """Hand a summoned friendly this minion's own stats.

    ``charges`` is what the card prints ("3 times per combat"), spent per fight
    because the count lives on the combat copy. Deliberately not named ``uses``:
    that name is on the golden scaler's doubling list, and the Golden printing
    still gets three — it doubles the stats instead, which is ``factor``.
    """

    charges: int = 0
    factor: int = 1


@dataclass(frozen=True)
class TriggerLeftmostDeathrattleEffect:
    """Fire the deathrattle of the left-most friendly that has one, without
    killing it (Deathstrider). ``repeats`` is the Golden's "twice"."""

    repeats: int = 1


@dataclass(frozen=True)
class DevourNeighbourEffect:
    """Start of Combat: destroy a neighbour and keep it to give back on death.

    Stitched Salvager's two clauses are one mechanism: the destroy stashes an
    exact copy on this body, and ``SummonStashedEffect`` on its deathrattle
    hands the stash back. ``adjacent`` is the Golden's "destroy adjacent
    minions" — the same reach, both sides.

    ``exclude_same_card`` is the "(Except Stitched Salvager.)" on the card: a
    second copy of it standing to the left is not food.
    """

    adjacent: bool = False
    exclude_same_card: bool = True


@dataclass(frozen=True)
class SummonStashedEffect:
    """Deathrattle: summon back whatever this body stashed (Stitched Salvager)."""


@dataclass(frozen=True)
class SellValueEffect:
    """What this minion sells for, when the card prints a price of its own.

    Freedealing Gambler prints one flat; Tortollan Blue Shell prints one behind
    a condition, which is why this is an ability rather than the plain
    ``Minion.sell_value`` field the catalog text fills in.
    """

    amount: int = 1


@dataclass(frozen=True)
class SetStatsEffect:
    """Set a friendly's stats outright ("set another minion's stats to 40/40").

    Not a buff: what the body was is discarded, which is the whole point on a
    1/1 and the whole cost on a 60/60.
    """

    attack: int = 0
    health: int = 0
    exclude_self: bool = True


@dataclass(frozen=True)
class GainTargetAttackEffect:
    """Rally: take the Attack of whoever this is swinging at (Heroic Underdog).

    ``factor`` is the Golden's "double the target's Attack".
    """

    factor: int = 1


@dataclass(frozen=True)
class StripKeywordsFromTargetEffect:
    """Rally: take keywords off the minion this is swinging at.

    Sin'dorei Straight Shot removes Reborn and Taunt, which is a removal and
    not a grant: the target keeps everything else it had.
    """

    keywords: tuple = ()


@dataclass(frozen=True)
class DestroyKillerEffect:
    """Deathrattle: destroy whatever killed this (Leeroy the Reckless)."""


@dataclass(frozen=True)
class BuffFromSubjectAttackEffect:
    """Hand a friendly stats equal to the event subject's Attack.

    Snazzy Phantom reads the Attack of the minion that was just Reborn and pays
    its right-most Undead with it; ``factor`` is the Golden's "double its
    Attack".
    """

    tribe: Any = None
    rightmost: bool = True
    factor: int = 1


@dataclass(frozen=True)
class RaiseGoldCapEffect:
    """Raise the most gold a turn can hand this seat ("increase your maximum
    Gold by 1")."""

    amount: int = 1


@dataclass(frozen=True)
class SpellsCastResponseEffect:
    """Answer every ``threshold`` spells the seat casts.

    "(3 left!)" is a countdown that refills, so the running total is the body's
    own — the same shape the gold and hero-damage watchers take.
    """

    threshold: int
    effect: Any = None


@dataclass(frozen=True)
class SummonGemGolemEffect:
    """Summon a body whose stats are the Blood Gems this minion is carrying.

    Reads ``blood_gem_attack``/``blood_gem_health``, which the Gems already
    record on whoever they were played on, so nothing new is counted.
    """

    token_id: str
    attack_immediately: bool = True


@dataclass(frozen=True)
class ImmuneWhileAttackingEffect:
    """Warpwing: takes no damage from the minion it is swinging at.

    Not a keyword: the keyword channels size an observation every checkpoint
    reads, and this is one card.
    """


@dataclass(frozen=True)
class DamageFromOwnAttackEffect:
    """Deal damage equal to this minion's Attack, to the target and its
    neighbours (Obsidian Ravager's Rally)."""

    include_adjacent: bool = True


@dataclass(frozen=True)
class HeroDamageResponseEffect:
    """What a minion does when its owner's hero takes damage.

    Four cards, three axes. ``rewind`` undoes the damage ("rewind it") and is
    what separates Soul Rewinder from Tichondrius, who only watches. ``effect``
    is the payload, which is an ordinary effect — a self buff, a tribe buff, a
    card fetched. ``threshold`` makes it a counter rather than a trigger:
    "after your hero takes 4 damage" fires once per four, and the running total
    is the body's own.
    """

    effect: Any = None
    rewind: bool = False
    threshold: int = 0


@dataclass(frozen=True)
class AddCardToNextRefreshesEffect:
    """Slip a card into each of the next ``refreshes`` tavern rolls.

    "Add a Fodder to your next 3 Refreshes" — a promise held by the seat and
    spent one roll at a time, not a card handed over now.
    """

    card_id: str
    refreshes: int = 1


@dataclass(frozen=True)
class FirstSpellcraftIsPermanentEffect:
    """Lava Lurker: the first Spellcraft spell cast on this each turn sticks.

    A Spellcraft buff normally expires at the owner's next turn; this makes one
    of them permanent instead. Carried on ``Trigger.AURA`` because it is a
    standing property of the body, not something that fires.
    """


@dataclass(frozen=True)
class ConsumeTavernMinionEffect:
    """A friendly eats a minion off the tavern counter and takes its stats.

    ``filter_race`` is who may do the eating ("choose a friendly Demon"). The
    eaten minion leaves the tavern the way a bought one does.
    """

    filter_race: Optional[Any] = None
    count: int = 1
    #: Take the biggest rather than one at random ("consume the highest-Health
    #: minion in the Tavern").
    highest_health: bool = False
    #: Every friendly of ``filter_race`` eats, not one of them ("your Demons
    #: *each* consume a random minion in the Tavern").
    each: bool = False
    #: What multiple of the eaten minion's stats the eater takes. The Golden
    #: printing gains *double* the stats of one minion rather than eating two.
    stat_multiplier: int = 1
    #: The eater is the card carrying this, not one the seat picks — which is
    #: what separates Insatiable Ur'zul from Mind Muck.
    eater_is_source: bool = False


@dataclass(frozen=True)
class SelfBonusPerGameCount:
    """Stats scaled by a game-long tally the seat keeps.

    "Has +3/+2 for each other Ancestral Automaton you've summoned this game",
    "+4/+2 for each friendly Eternal Knight that died this game". Carried on
    ``Trigger.AURA`` because it is a continuous read, not an event: the card
    shows the number wherever it is, and the seat's tally is the only state.

    ``subject`` names whose tally to read and defaults to the card's own id —
    which is what every printing of this shape wants. ``count_self`` is the
    difference between "for each" and "for each *other*": left False, a copy
    whose own arrival was counted leaves itself out.
    """

    counter: str
    attack_per: int = 0
    health_per: int = 0
    subject: str = ""
    count_self: bool = False


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
    #: SHOP only: cap the tier it reaches ("minions in the Tavern from Tier 3
    #: and below"). 0 means every tier.
    scope_max_tier: int = 0
    #: What it raises instead when it fires outside a fight ("+2 Attack this
    #: game. (+4 if triggered outside combat!)"). 0 means the same either way —
    #: Plaguerunner is the only card that pays two prices.
    attack_outside_combat: int = 0
    health_outside_combat: int = 0


@dataclass(frozen=True)
class GiveLockboxEffect:
    """Hand the seat a Lockbox, or hurry the one it already has along.

    Both halves of "Get a Lockbox. If you already have one, it opens 1 turn
    sooner" — which is one rule, since a seat only ever holds one.
    """

    sooner: int = 1


@dataclass(frozen=True)
class AddTavernSpellToHandEffect:
    """Put a named spell in hand ("Battlecry: Get a Tavern Coin").

    Names the card rather than rolling one: these battlecries print which spell
    they hand over. Not only Tavern spells — Slimy Shield and Gem Day are plain
    spells handed over the same way — so the package must carry the card, and
    one that does not hands over nothing.
    """

    card_id: str
    count: int = 1


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
    seat's tavern tier; this one is printed on the card — and moves only if the
    card says it improves, in which case ``counter`` multiplies it.
    """

    tier: int = 1
    counter: str = ""
    per: int = 1


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

    #: What the spell does. Usually a buff that expires — that is what the
    #: keyword was built around — but not always: some Nagas hand out a spell
    #: that fetches a card or raises a seat bonus, and those are ordinary
    #: effects wearing a Spellcraft spell as a wrapper.
    buff: Any
    card_id: str = ""
    name: str = ""
    #: "Improved by every 4 spells you've cast this game" — the buff is
    #: multiplied by the level when the spell is made.
    counter: str = ""
    per: int = 1


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
    hands out two cards of the same tier, never one of a higher tier. ``count``
    is the field that carries that "two" — and it *is* on the scaler's list,
    which is the whole difference between the pair.
    """

    tribe: Optional[Any] = None
    tier: Optional[int] = None
    count: int = 1
    #: Narrow the draw to cards carrying a keyword ("a random **Magnetic**
    #: Mech"). A property of the card, so it reads the same in either phase.
    keyword: Optional[Keyword] = None


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
    DiscoverTribeEffect,
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
    SummonBestFromHandEffect,
    BuffRandomHandMinionEffect,
    GrantCombinedChooseOneEffect,
    MultiplySummonedAttackEffect,
    GiveOwnStatsToSummonedEffect,
    TriggerLeftmostDeathrattleEffect,
    BuffFromSubjectAttackEffect,
    DestroyKillerEffect,
    StripKeywordsFromTargetEffect,
    GainTargetAttackEffect,
    SetStatsEffect,
    SellValueEffect,
    SummonStashedEffect,
    DevourNeighbourEffect,
    RaiseGoldCapEffect,
    SpellsCastResponseEffect,
    SummonGemGolemEffect,
    ImmuneWhileAttackingEffect,
    DamageFromOwnAttackEffect,
    KeepCombatGainsEffect,
    HeroDamageResponseEffect,
    AddCardToNextRefreshesEffect,
    FirstSpellcraftIsPermanentEffect,
    ConsumeTavernMinionEffect,
    SelfBonusPerGameCount,
    IncreaseTavernSpellBonusEffect,
    AddRandomTavernSpellToHandEffect,
    DiscoverTavernSpellEffect,
    CastRandomTavernSpellEffect,
    CopyLastTavernSpellEffect,
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
    #: Fire only for a placed card of this tier or below ("after you play a card
    #: from Tier 3 or below"). A trigger filter like ``filter_race`` beside it,
    #: and a field rather than a ConditionKind for the same reason the Battlecry
    #: requirement is: that vocabulary sizes an embedding table.
    filter_max_tier: int = 0
    #: Fire only when the subject of the event has a Rally ("whenever a friendly
    #: **Rally** minion attacks"). A property of the minion the event happened
    #: to, like ``filter_race`` beside it, so it is checked where every other
    #: subject filter is rather than inside three effect handlers.
    filter_subject_rally: bool = False
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
    "DiscoverTribeEffect",
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
    "BumpSeatCounterEffect",
    "DestroyFriendlyEffect",
    "RefreshesCostHealthEffect",
    "BuffOnSpellCastOnTribeEffect",
    "BuffSharedTribeEffect",
    "BuffHandMinionsEffect",
    "GainStatsFromHandEffect",
    "GiveOwnStatsToHandEffect",
    "DoubleBountiesEffect",
    "AddRandomGoldenMinionEffect",
    "AddRandomMinionOfCommonTribeEffect",
    "MakeFriendlyGoldenEffect",
    "BuffTargetPerGoldSpentEffect",
    "BuffBoughtMinionEffect",
    "StatsFromNextBuyEffect",
    "GoldSpentResponseEffect",
    "IncreaseTribeGiftEffect",
    "CastSpellAtEffect",
    "MagnetizeTokenEffect",
    "MagnetizesToTribesEffect",
    "DoubleNextMagnetizeEffect",
    "BuffPerMagnetizationEffect",
    "EchoMagnetizeEffect",
    "AddRandomCardToHandEffect",
    "BuffSelfOnFriendlyDamageEffect",
    "BuffSelfOnFriendlySoldEffect",
    "BuffShopOnEveryRefreshEffect",
    "RewardAtDamageDealtEffect",
    "SummonBestFromHandEffect",
    "BuffRandomHandMinionEffect",
    "GrantCombinedChooseOneEffect",
    "MultiplySummonedAttackEffect",
    "GiveOwnStatsToSummonedEffect",
    "TriggerLeftmostDeathrattleEffect",
    "BuffFromSubjectAttackEffect",
    "DestroyKillerEffect",
    "StripKeywordsFromTargetEffect",
    "GainTargetAttackEffect",
    "SetStatsEffect",
    "SellValueEffect",
    "SummonStashedEffect",
    "DevourNeighbourEffect",
    "RaiseGoldCapEffect",
    "SpellsCastResponseEffect",
    "SummonGemGolemEffect",
    "ImmuneWhileAttackingEffect",
    "DamageFromOwnAttackEffect",
    "KeepCombatGainsEffect",
    "HeroDamageResponseEffect",
    "AddCardToNextRefreshesEffect",
    "FirstSpellcraftIsPermanentEffect",
    "ConsumeTavernMinionEffect",
    "SelfBonusPerGameCount",
    "IncreaseTavernSpellBonusEffect",
    "AddRandomTavernSpellToHandEffect",
    "DiscoverTavernSpellEffect",
    "CastRandomTavernSpellEffect",
    "CopyLastTavernSpellEffect",
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

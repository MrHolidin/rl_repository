from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Tuple

from itertools import count as _count

from .effects import Ability, Keyword

# Entity identity, handed out wherever a minion comes into existence. Events
# queued against a minion outlive the board moving under it, and object
# identity cannot answer "is this one still on the board" -- an id can, via
# ``_CombatRuntime.find_minion``. Process-wide and never serialised: no
# observation, replay or golden-trace digest reads it.
_INSTANCE_IDS = _count(1)


def next_instance_id() -> int:
    return next(_INSTANCE_IDS)


class Race(Enum):
    """Tribes, in the order they entered the game.

    Members are appended, never inserted: ``auto()`` values are what a patch
    package's layout and every serialised tribe reference resolve to, so
    renumbering the existing ones would silently re-tag old data. Which of these
    a given patch actually uses is the package's business (``meta.json`` rotation
    and layout), not this enum's — a build that predates Quilboar simply never
    names it.
    """

    BEAST = auto()
    DEMON = auto()
    MECHANICAL = auto()
    MURLOC = auto()
    DRAGON = auto()
    PIRATE = auto()
    ELEMENTAL = auto()
    ALL = auto()
    # Post-19.6 tribes (Quilboar 20.0, Naga 23.2, Undead 25.0).
    QUILBOAR = auto()
    NAGA = auto()
    UNDEAD = auto()


@dataclass
class Minion:
    card_id: str
    base_attack: int
    base_health: int
    tier: int
    # Catalog / card display name — for replays, logs, heuristics (not used in obs).
    name: str = ""
    bonus_attack: int = 0
    bonus_health: int = 0
    race: Optional[Race] = None
    keywords: frozenset[Keyword] = field(default_factory=frozenset)
    granted_keywords: frozenset[Keyword] = field(default_factory=frozenset)
    abilities: Tuple[Ability, ...] = ()
    has_shield: bool = False
    is_token: bool = False
    is_golden: bool = False
    """Set when forged from three non-golden copies (golden minion only, not the reward spell)."""
    from_triple_merge: bool = False
    dbf_id: Optional[int] = None
    sell_value: Optional[int] = None
    # Stats this minion has taken from Blood Gems, kept alongside the buff they
    # already applied to ``bonus_*``. Recorded rather than merged because cards
    # read it back: Jailbird Juggernaut summons a Golem "with stats equal to
    # this minion's Blood Gems", and Gem Confiscation "steals all Blood Gems
    # from its neighbors" — neither can be expressed if a Gem is just +1/+1
    # dissolved into the total.
    blood_gem_attack: int = 0
    blood_gem_health: int = 0
    #: An Activate ability is spent for the turn once fired. Board state, not
    #: combat state: it is cleared at the start of the seat's turn.
    activate_used_this_turn: bool = False
    #: Whether this body has already had its one permanent Spellcraft spell
    #: this turn (Lava Lurker). Reset with the Activate flag at turn start.
    spellcraft_kept_this_turn: bool = False

    #: entity identity; see ``next_instance_id``
    instance_id: int = field(default_factory=next_instance_id)
    #: On a combat copy: the ``instance_id`` of the board minion it was made
    #: from, so an effect printed as permanent can name the body it means on
    #: the owner's real board. A copy gets a fresh ``instance_id`` of its own
    #: (the combat addresses minions by it), which is why the origin has to be
    #: carried rather than inherited. 0 on a board minion and on anything
    #: summoned mid-combat, which has no counterpart to write back to.
    origin_instance_id: int = 0

    # --- combat lifecycle -------------------------------------------------
    # Only meaningful while this minion is in Zone.COMBAT, which is always a
    # *copy* of the board: a battle never writes back, so damage taken and
    # shields popped die with the copy and the next battle starts from the
    # untouched original. Outside combat these are simply not maintained.
    #: Damage this minion has taken. Health is derived, not stored: a buff
    #: raises ``bonus_health`` and the current health follows on its own, which
    #: is why no effect has to remember to raise two numbers. Fourteen of the
    #: twenty-three writes to the old absolute existed only to do that.
    damage_taken: int = 0
    #: Attack and health contributed by continuous auras right now, as values
    #: rather than as deltas patched into an absolute. Both are recomputed from
    #: zero when the board changes, which is how the reference battlegrounds
    #: simulators do it too -- there is no delta to get out of step.
    #:
    #: ``aura_attack`` covers the per-board auras only. Old Murk-Eye counts
    #: murlocs on *both* boards and a hero can grant a side-wide bonus, and
    #: neither can live on the minion without re-syncing it whenever the other
    #: board moves; ``attack_value`` adds those two on top.
    aura_attack: int = 0
    aura_health: int = 0
    deathrattle_fired: bool = False
    reborn_consumed: bool = False
    #: Venomous has made its kill and is used up for the rest of this combat.
    #: Like a popped Divine Shield, it lives on the combat copy only, so the
    #: minion comes back venomous next combat.
    venom_spent: bool = False
    #: Friendly deaths seen since this minion's Avenge last fired (combat only).
    avenge_progress: int = 0
    #: "This can't gain stats" — Fishbait, the 0/1 the tavern puts up to be
    #: killed. Honoured by the buff paths that can reach a card sitting in the
    #: tavern; a general funnel for every buff site is the right home for it
    #: once a second card needs the flag.
    cannot_gain_stats: bool = False
    #: What this card has already absorbed from each of the owner's standing
    #: "this game" bonuses, as ``(scope, attack, health)`` rows. Per scope and
    #: not one total, because a bonus is never taken back off: a minion buffed
    #: on the tavern counter keeps those stats after it is bought, and must
    #: still be able to take a later raise of a *different* scope.
    #:
    #: A tuple rather than a dict to keep every field of this class immutable,
    #: which is what makes ``__copy__`` a safe shallow clone.
    standing_absorbed: Tuple[Tuple[Any, int, int], ...] = ()
    #: Stats currently granted by a game-long tally, so a recompute can apply
    #: the difference instead of stacking (see ``game_counts``).
    count_bonus_granted: Tuple[int, int] = (0, 0)
    #: A name a promise can call this body by, later. ``instance_id`` cannot:
    #: ``__copy__`` re-issues it, and the seat's state is copied once per
    #: action, so an id noted this turn belongs to nobody by the next one. This
    #: is stamped once, carried by every copy, and means only "the body that
    #: was promised something".
    promise_tag: int = 0
    #: How many minions have been Magnetized onto this body. Read by the cards
    #: that pay per Magnetization, and by nothing else — the stats themselves
    #: are merged in, not derived from this.
    magnetized_count: int = 0
    #: What the Magnetizations onto this body contributed: stats folded into
    #: ``base_*`` and the abilities appended to ``abilities``. Kept separately
    #: so a triple can carry them over -- the merge rebuilds ``base_*`` from
    #: the printed card, which silently dropped every part a host was wearing.
    #: Patch 29.2.2.198608 fixed the same bug in the real game.
    magnet_attack: int = 0
    magnet_health: int = 0
    magnet_abilities: Tuple[Any, ...] = ()
    #: The next Magnetization onto this body lands twice (Drone Duplicator).
    #: Spent by that Magnetization, and cleared at the start of the turn.
    magnet_doubles_next: bool = False
    #: Waiting to take the stats of the next minion the seat buys this turn.
    wants_next_buy_stats: bool = False
    #: Turns this card stays locked in hand ("Lock it in your hand for 1
    #: turn"). A locked card is inert: it cannot be played, sold or magnetized,
    #: it does not count toward a triple, and nothing that reads the hand can
    #: see it. It still occupies its slot, which is the cost of holding one.
    #: Counted down at the seat's turn start, so "1 turn" is one of its own.
    locked_turns: int = 0
    #: How many times a body that improves *itself* has improved ("give it +2
    #: Attack and improve this permanently"). Per body, not per printing: two
    #: Leviathans improve separately, and a golden built from three starts over.
    self_improves: int = 0
    #: Bodies this minion ate at the start of the fight and owes back on death
    #: (Stitched Salvager). Copies, so what they had gained comes back with
    #: them — "an exact copy" and not the printed card.
    stashed_bodies: tuple = ()
    #: Charges left on a per-combat "N times" ability, spent as they are used.
    #: Lives on the combat copy, so it refills with every fight.
    combat_uses_left: int = -1
    #: Spells the seat has cast while this body watched, for the cards that
    #: answer every Nth rather than every one.
    spells_seen: int = 0
    #: Elementals the seat has played while this body watched, for the cards
    #: that answer every Nth rather than every one.
    elementals_seen: int = 0
    #: Spent by the Spellcraft spell it answered, for the watchers printed
    #: "once per turn".
    spell_answered_this_turn: bool = False
    #: Gold the seat has spent while this body watched, for the cards that
    #: answer every Nth coin rather than every purchase.
    gold_spent_seen: int = 0
    #: Spent by the buy it paid, for the watchers printed "once per turn".
    buy_answered_this_turn: bool = False
    #: Hero damage this body has watched, for the cards that answer every Nth
    #: point rather than every hit ("after your hero takes 4 damage").
    hero_damage_seen: int = 0
    #: Damage this body has dealt in combat, across every fight it has been in
    #: ("once this deals 40 damage…"). Per body and not per printing: two
    #: copies count separately, and a golden made from three starts over,
    #: because ``merge_three_non_golden_into_golden`` builds a new card.
    damage_dealt_total: int = 0
    #: Whether the reward for that total has already been paid, so a body pays
    #: once however much further it swings.
    damage_reward_paid: bool = False
    #: Whether this card's own arrival is one of the events its tally counted —
    #: the single subtraction behind "for each **other**".
    self_counted: bool = False
    #: What this body was worth when the fight began, so a card that keeps its
    #: combat gains (Tarecgosa) can tell what it gained. Set by ``battle_copy``
    #: and meaningless on a shop minion.
    start_bonus_attack: int = 0
    start_bonus_health: int = 0
    start_keywords: frozenset = frozenset()
    # --- "until next turn" ------------------------------------------------
    #: Stats and keywords that expire at the start of the owner's next recruit
    #: phase (Spellcraft: "Give a minion +2/+6 and Taunt until next turn").
    #: Separate from ``bonus_*`` because they are removed again, and separate
    #: from the combat-only fields above because they must survive the combat
    #: they were cast for: cast in recruit N, felt in combat N, gone by recruit
    #: N+1. ``expire_temporary_buffs`` is the only thing that clears them.
    temp_attack: int = 0
    temp_health: int = 0
    temp_keywords: frozenset[Keyword] = field(default_factory=frozenset)
    #: board slot this minion vacated when it died, so its deathrattle can
    #: summon there and Reborn can return there
    death_pos: int = -1
    #: MinionDied has been queued for this body
    death_announced: bool = False

    @property
    def current_health(self) -> int:
        """Health right now: printed + buffs + auras, less damage taken."""
        return self.max_health + self.aura_health - self.damage_taken

    @property
    def alive(self) -> bool:
        """Combat-only: outside a battle nothing maintains damage or auras."""
        return self.current_health > 0

    def __copy__(self) -> "Minion":
        # Fast shallow clone: identical to copy.copy(self) but skips the generic
        # __reduce_ex__/_reconstruct machinery. All fields are immutable
        # (ints/str/None) or already-immutable containers (frozenset/tuple), so
        # sharing them in a shallow copy is safe — matches prior copy.copy use.
        new = object.__new__(Minion)
        new.__dict__ = self.__dict__.copy()
        new.instance_id = next_instance_id()
        return new

    @property
    def all_keywords(self) -> frozenset[Keyword]:
        return self.keywords | self.granted_keywords | self.temp_keywords

    @property
    def max_health(self) -> int:
        return self.base_health + self.bonus_health + self.temp_health

    @property
    def raw_attack(self) -> int:
        return self.base_attack + self.bonus_attack + self.temp_attack


#: Every tribe a minion can be, which is what "of each type" means. ALL is the
#: Amalgam marker rather than a type of its own, so it is not one of them.
ALL_TRIBES = tuple(r for r in Race if r is not Race.ALL)

def is_locked(card) -> bool:
    """Whether a hand card is held shut and must be treated as not there.

    One predicate rather than a check per reader: "does not interact with
    anything" is only true if every place that walks the hand asks the same
    question, and there are a dozen of them.
    """
    return int(getattr(card, "locked_turns", 0) or 0) > 0


__all__ = ["ALL_TRIBES", "Race", "Minion", "is_locked"]

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

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

    #: entity identity; see ``next_instance_id``
    instance_id: int = field(default_factory=next_instance_id)

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
        return self.keywords | self.granted_keywords

    @property
    def max_health(self) -> int:
        return self.base_health + self.bonus_health

    @property
    def raw_attack(self) -> int:
        return self.base_attack + self.bonus_attack


__all__ = ["Race", "Minion"]

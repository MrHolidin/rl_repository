from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

from .effects import Ability, Keyword


class Race(Enum):
    BEAST = auto()
    DEMON = auto()
    MECHANICAL = auto()
    MURLOC = auto()
    DRAGON = auto()
    PIRATE = auto()
    ELEMENTAL = auto()
    ALL = auto()


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

    # Identity. Assigned wherever a minion is created, so an event queued
    # against a minion can still find it after the board has moved under it --
    # object identity cannot answer "is it still on the board".
    instance_id: int = 0

    # --- combat lifecycle -------------------------------------------------
    # Only meaningful while this minion is in Zone.COMBAT, which is always a
    # *copy* of the board: a battle never writes back, so damage taken and
    # shields popped die with the copy and the next battle starts from the
    # untouched original. Outside combat these are simply not maintained.
    current_health: int = 0
    deathrattle_fired: bool = False
    reborn_consumed: bool = False
    health_aura_snapshot: int = 0
    #: board slot this minion vacated when it died, so its deathrattle can
    #: summon there and Reborn can return there
    death_pos: int = -1
    #: MinionDied has been queued for this body
    death_announced: bool = False

    @property
    def alive(self) -> bool:
        """Combat-only: a minion outside a battle has no current health."""
        return self.current_health > 0

    def __copy__(self) -> "Minion":
        # Fast shallow clone: identical to copy.copy(self) but skips the generic
        # __reduce_ex__/_reconstruct machinery. All fields are immutable
        # (ints/str/None) or already-immutable containers (frozenset/tuple), so
        # sharing them in a shallow copy is safe — matches prior copy.copy use.
        new = object.__new__(Minion)
        new.__dict__ = self.__dict__.copy()
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

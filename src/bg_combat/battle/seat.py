"""What a combat is allowed to do to the seat that owns a board.

Combat runs on *copies*: damage taken, shields popped and Venomous spent die
with the copy, which is what makes a board come out of a fight the way it went
in. That stays. What this adds is the other half — the things a card genuinely
hands its owner from inside a fight, which the copy cannot carry:

* gold ("Deathrattle: gain 2 Gold"),
* cards ("Rally: get a random Beast"),
* **permanent** stat changes on a specific board minion (the Blood Gems printed
  as permanent, and the "+X/+X this game" buffs),
* game-long modifiers on the seat itself ("your Blood Gems give an extra +1/+1
  this game", raised mid-combat by a Rally).

Those used to leave as two ad-hoc out-parameters that the caller re-applied,
duplicated across the two lobby types. They are operations now, for a reason
beyond tidiness: a permanent Gem has to *read* the seat's current Gem value at
the moment it is played, and another Rally in the same combat can raise that
value first. Collecting requests and pricing them afterwards gets the order
wrong.

Two implementations:

* :class:`RecordingSeat` — the default. Collects, applies nothing, and is what
  keeps ``simulate_battle`` callable with two lists of minions and no player at
  all (every combat test, and the pure-rules API).
* the lobby's seat adapter, which writes straight through to a ``PlayerState``.

A minion is addressed by ``instance_id``, which survives the copy, so an effect
can name the body it means on the owner's real board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, Tuple


class CombatSeat(Protocol):
    """The seat behind one side of a combat."""

    def gain_gold(self, amount: int) -> None: ...

    def add_card_to_hand(self, card_id: str) -> None: ...

    def blood_gem_value(self) -> Tuple[int, int]:
        """What one Blood Gem is worth to this seat right now."""

    def play_permanent_blood_gem(self, instance_id: int, count: int = 1) -> None:
        """Put ``count`` Gems on the owner's real minion, for keeps."""

    def raise_blood_gem_value(self, attack: int, health: int) -> None:
        """"Your Blood Gems give an extra +1/+1 this game", raised in combat."""


@dataclass
class PermanentGemGrant:
    instance_id: int
    count: int


@dataclass
class RecordingSeat:
    """Collects what a combat hands out, applies none of it.

    The seatless default. Its fields are exactly the old out-parameters, which
    is why a caller that never heard of seats keeps working.
    """

    gold: int = 0
    hand_adds: List[str] = field(default_factory=list)
    permanent_gems: List[PermanentGemGrant] = field(default_factory=list)
    gem_value_raise: Tuple[int, int] = (0, 0)
    #: Gem value a recording seat reports: the printed +1/+1, since it has no
    #: player to read a bonus off.
    base_gem_value: Tuple[int, int] = (1, 1)

    def gain_gold(self, amount: int) -> None:
        self.gold += int(amount)

    def add_card_to_hand(self, card_id: str) -> None:
        self.hand_adds.append(card_id)

    def blood_gem_value(self) -> Tuple[int, int]:
        base_attack, base_health = self.base_gem_value
        extra_attack, extra_health = self.gem_value_raise
        return (base_attack + extra_attack, base_health + extra_health)

    def play_permanent_blood_gem(self, instance_id: int, count: int = 1) -> None:
        self.permanent_gems.append(PermanentGemGrant(int(instance_id), int(count)))

    def raise_blood_gem_value(self, attack: int, health: int) -> None:
        current_attack, current_health = self.gem_value_raise
        self.gem_value_raise = (current_attack + int(attack), current_health + int(health))


__all__ = ["CombatSeat", "PermanentGemGrant", "RecordingSeat"]

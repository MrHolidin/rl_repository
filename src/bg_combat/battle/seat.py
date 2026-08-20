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

    def hand_minions(self) -> Tuple[Tuple[int, str, int, int], ...]:
        """``(instance_id, card_id, attack, health)`` for each minion in hand.

        The instance id is what lets a combat tell two copies of one card apart,
        which it must: a card summoned from hand is locked for the rest of the
        fight, and locking by card id would lock the wrong one — or both.
        """

    def gain_blood_gems(self, count: int) -> None:
        """Gems into the owner's hand ("Rally: Get a Blood Gem").

        Not ``add_card_to_hand``: that queue is card ids the lobby turns into
        minions after the fight, and a Gem is a spell.
        """

    def raise_standing_bonus(
        self, scope_kind: object, scope_key: object, attack: int, health: int
    ) -> None:
        """Raise a "this game" bonus on the owner, from inside the fight."""

    def start_combat_promises(self) -> Tuple[object, ...]:
        """Start-of-Combat abilities the owner bought with a spell.

        Read, not consumed: the seat lets go of them at its own next turn
        start, so a combat replayed against the same seat sees the same board.
        """

    def settle_standing_bonus(self, minion: object) -> None:
        """Pay a body whatever the owner's "this game" table already owes it.

        Read rather than write, and the mirror of the one above. A token
        summoned mid-fight is a minion the seat owns, and every card that
        raises one of these prints *wherever they are* — a Beetle that arrives
        by deathrattle carries Forest Rover's +2/+1 the same as one bought off
        the counter. Idempotent, because each body records what it has already
        absorbed, so a summoned *copy* of a paid minion is not paid twice.
        """

    def raise_tavern_spell_bonus(self, attack: int, health: int) -> None:
        """"Rally: your Tavern spells give an extra +1 Health this game"."""

    def add_refresh_buff(self, attack: int, health: int) -> None:
        """"After the Tavern is Refreshed this game, give a random minion in it
        +3/+3" — a standing promise, so it outlives the body that made it."""

    def record_damage_dealt(
        self, instance_id: int, amount: int, threshold: int, reward_card_id: str
    ) -> None:
        """Count damage a body dealt, and pay its reward when the total lands.

        Named by ``instance_id`` because the tally is that body's own: two
        copies of the same card count separately, and the copy that swings does
        not survive the fight to keep it.
        """

    def improve_level(self, counter: str, per: int) -> int:
        """How many times over a card that improves is worth its printed value.

        Read from inside a fight by the Start of Combat effects that improve —
        the tally is the owner's, and a combat copy has none.
        """

    def bump_game_count(self, family: str, subject: str) -> None:
        """Count one event on the owner's tally, from inside the fight.

        "Has +4/+2 for each friendly Eternal Knight that **died** this game" is
        counted by a death, and the deaths happen here. It has to reach the seat
        rather than the combat copy: the copy is thrown away, and the tally is
        read by Knights in hand, in the tavern, and in every fight after this.
        """

    def buff_hand_minion(self, attack: int, health: int, *, rng) -> None:
        """Stats onto a random minion in the owner's hand."""

    def promise_refresh_card(self, card_id: str, refreshes: int) -> None:
        """"Add a Fodder to your next 3 Refreshes" — a promise, not a card."""

    def give_lockbox(self, sooner: int) -> None:
        """A Lockbox onto the seat, or a turn off the one it already holds."""

    def raise_tribe_gift(self, tribe: object, attack: int, health: int) -> None:
        """"Your Elementals give an extra +1 Attack this game" — what each
        grant hands out, not the running total."""

    def improve_body(self, instance_id: int) -> None:
        """Count one improve on a body that improves *itself* permanently.

        Named by ``instance_id`` for the same reason the damage tally is: the
        improve belongs to that body, and the copy that earned it inside the
        fight is thrown away when the fight ends.
        """

    def keep_combat_gains(self, instance_id: int, attack: int, health: int, keywords) -> None:
        """Write a body's combat gains through to the owner's real minion.

        The one thing a combat normally never does. Tarecgosa is the card that
        asks for it, and it asks for exactly this: the stats and keywords the
        copy picked up, kept.
        """


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
    #: Gems a combat handed the owner, waiting for a seat that can hold them.
    blood_gems: int = 0
    #: "This game" bonuses a combat raised, for a seat that has a table.
    standing_raises: List[Tuple[object, object, int, int]] = field(default_factory=list)
    #: Standing tavern-roll buffs a combat promised the owner.
    refresh_buffs: List[Tuple[int, int]] = field(default_factory=list)
    #: Damage a counting body dealt: (instance_id, amount, threshold, reward).
    damage_dealt: List[Tuple[int, int, int, str]] = field(default_factory=list)
    #: What a combat raised the owner's Tavern-spell bonus by.
    tavern_spell_raise: Tuple[int, int] = (0, 0)
    #: Events a combat counted, for a seat that keeps tallies.
    count_bumps: List[Tuple[str, str]] = field(default_factory=list)
    #: Buffs a combat aimed at the owner's hand.
    hand_buffs: List[Tuple[int, int]] = field(default_factory=list)
    #: Combat gains a body asked to keep (instance_id, attack, health, keywords).
    kept_gains: List[Tuple[int, int, int, frozenset]] = field(default_factory=list)
    body_improves: List[int] = field(default_factory=list)
    refresh_card_promises: List[Tuple[str, int]] = field(default_factory=list)
    lockboxes: List[int] = field(default_factory=list)
    tribe_gifts: List[Tuple[object, int, int]] = field(default_factory=list)
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

    def hand_minions(self) -> Tuple[Tuple[int, str, int, int], ...]:
        """Empty: a seatless combat has two lists of minions and no hand."""
        return ()

    def gain_blood_gems(self, count: int) -> None:
        self.blood_gems += int(count)

    def raise_standing_bonus(
        self, scope_kind: object, scope_key: object, attack: int, health: int
    ) -> None:
        self.standing_raises.append((scope_kind, scope_key, int(attack), int(health)))

    def start_combat_promises(self) -> Tuple[object, ...]:
        """None: a seatless combat has two lists of minions and no hand."""
        return ()

    def settle_standing_bonus(self, minion: object) -> None:
        """Nothing: a seatless combat has no table to owe anything from."""

    def add_refresh_buff(self, attack: int, health: int) -> None:
        self.refresh_buffs.append((int(attack), int(health)))

    def raise_tavern_spell_bonus(self, attack: int, health: int) -> None:
        current_attack, current_health = self.tavern_spell_raise
        self.tavern_spell_raise = (current_attack + int(attack), current_health + int(health))

    def record_damage_dealt(
        self, instance_id: int, amount: int, threshold: int, reward_card_id: str
    ) -> None:
        self.damage_dealt.append(
            (int(instance_id), int(amount), int(threshold), reward_card_id)
        )

    def improve_level(self, counter: str, per: int) -> int:
        """One: a seatless combat keeps no tallies, so nothing has improved."""
        return 1

    def bump_game_count(self, family: str, subject: str) -> None:
        self.count_bumps.append((family, subject))

    def buff_hand_minion(self, attack: int, health: int, *, rng) -> None:
        self.hand_buffs.append((int(attack), int(health)))

    def promise_refresh_card(self, card_id: str, refreshes: int) -> None:
        self.refresh_card_promises.append((card_id, int(refreshes)))

    def give_lockbox(self, sooner: int) -> None:
        self.lockboxes.append(int(sooner))

    def raise_tribe_gift(self, tribe: object, attack: int, health: int) -> None:
        self.tribe_gifts.append((tribe, int(attack), int(health)))

    def improve_body(self, instance_id: int) -> None:
        self.body_improves.append(int(instance_id))

    def keep_combat_gains(self, instance_id: int, attack: int, health: int, keywords) -> None:
        self.kept_gains.append((int(instance_id), int(attack), int(health), frozenset(keywords)))


__all__ = ["CombatSeat", "PermanentGemGrant", "RecordingSeat"]

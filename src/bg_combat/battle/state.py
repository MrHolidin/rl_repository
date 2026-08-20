"""Mutable combat state: minions, sides, and the per-battle runtime."""
from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass, field
from typing import Callable, Deque, Iterator, List, Optional, Tuple

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword, Trigger
from src.bg_core.minion import Minion

from .events import BattleEvent
from .seat import CombatSeat, RecordingSeat


# A minion in a battle *is* a Minion -- there is no wrapper. Combat runs on a
# copy of the board (see ``battle_copy``), so damage taken and shields popped
# die with that copy and never reach the player's board. The alias is kept so
# call sites and annotations still read as "this is a combat-side minion".
BattleMinion = Minion


def battle_copy(minion: Minion, instance_id: int) -> Minion:
    """A minion prepared to fight: a copy, at full health, shield re-armed.

    The copy is the isolation mechanism. Re-arming from the keyword is what
    makes divine shields refresh between combats without anyone writing back.
    """
    bm = copy(minion)
    # The copy fights under a combat-local id; the board minion's own id rides
    # along so a permanent effect can find the body it came from afterwards.
    bm.origin_instance_id = minion.instance_id
    bm.instance_id = instance_id
    # What it came in worth, for the one card that keeps what it gains.
    bm.start_bonus_attack = minion.bonus_attack
    bm.start_bonus_health = minion.bonus_health
    bm.start_keywords = minion.keywords
    bm.damage_taken = 0
    bm.has_shield = minion.has_shield and Keyword.SHIELD in minion.all_keywords
    bm.deathrattle_fired = False
    bm.reborn_consumed = False
    bm.venom_spent = False
    bm.avenge_progress = 0
    bm.aura_health = 0
    bm.death_pos = -1
    bm.death_announced = False
    return bm


@dataclass
class BattleSide:
    minions: List[BattleMinion] = field(default_factory=list)
    # Bodies that have left the board, in death order. They are kept because
    # events already in flight reference them by ``instance_id`` and because a
    # deathrattle resolves after the body is gone; ``Minion.death_pos``
    # carries the slot it vacated.
    graveyard: List[BattleMinion] = field(default_factory=list)
    cursor: int = 0
    # Flat Attack added to every minion on this side (Deathwing's global +Attack
    # aura; set equal on both sides since it buffs all minions in the combat).
    attack_aura_all: int = 0
    # Keywords granted to this side's left-most minion at Start of Combat (Al'Akir).
    start_combat_keywords: frozenset = field(default_factory=frozenset)

    def reap_dead(self) -> List[BattleMinion]:
        """Take dead bodies off the board; return them in board order.

        The board closes up behind them, which is what makes adjacency correct:
        two survivors that had a corpse between them become neighbours, exactly
        as they do in the game. Each body records the slot it vacated in
        ``death_pos`` so its deathrattle can summon there and Reborn can return
        there, and moves to ``graveyard`` so in-flight events can still resolve
        it by ``instance_id``.

        ``cursor`` (the attack rotation pointer) is an index into ``minions``,
        so it shifts with the removals: a body left of the pointer pulls it
        left, and a body *at* the pointer leaves it pointing at whoever slid
        into that slot — the next minion in rotation, not the one after it.
        """
        taken: List[BattleMinion] = []
        i = 0
        while i < len(self.minions):
            m = self.minions[i]
            if m.alive:
                i += 1
                continue
            m.death_pos = i
            self.minions.pop(i)
            # Bodies already waiting to summon sit at recorded slots; taking a
            # body out from their left shifts those slots down. Without this
            # the recorded position goes stale as soon as two minions die in
            # the same exchange and are swept at different moments.
            self.shift_graveyard_slots(i, -1)
            self.graveyard.append(m)
            taken.append(m)
            if self.cursor > i:
                self.cursor -= 1
        if self.minions:
            self.cursor %= len(self.minions)
        else:
            self.cursor = 0
        self.assert_no_corpses()
        return taken

    def shift_graveyard_slots(self, at: int, delta: int) -> None:
        """Keep recorded death slots valid as the board around them moves."""
        # Strictly to the right: the slot ``at`` itself belongs to the body
        # being removed from it, or to the body whose token is filling it, and
        # in neither case does that body's own recorded slot move.
        for bm in self.graveyard:
            if bm.death_pos > at:
                bm.death_pos += delta

    def listeners(
        self,
        trigger: Trigger,
        subject: Optional[BattleMinion] = None,
        *,
        exclude_subject: bool = True,
    ) -> Iterator[Tuple[BattleMinion, object]]:
        """Every (listener, effect) on this side that ``trigger`` reaches.

        ``subject`` is whom the event happened to: the minion that died, that
        attacked, that lost its shield. An ability can narrow to the subject's
        tribe (Scavenging Hyena wants a Beast) or to its keywords (Bolvar wants
        a popped Divine Shield), and those are the same two ability fields
        wherever this loop is written out. Six copies of it disagreed on which
        of the two they bothered to check -- harmlessly, because no card in
        either catalog pairs a filter with a trigger whose site ignored it, but
        the next one to do so would have been silently ignored.

        Whether a listener hears an event about *itself* is the card's own
        business, and the catalog says which: "whenever a friendly minion
        attacks" includes the watcher, "whenever **another** friendly Dragon
        attacks" does not. The triggers whose plain reading is inclusive pass
        ``exclude_subject=False`` and let ``Ability.excludes_self`` mark the
        exceptions; the rest exclude the subject outright, because a minion
        cannot avenge its own death or answer its own arrival.
        """
        from src.bg_core.board_helpers import minion_matches_tribe

        for listener in self.iter_living():
            for ab in listener.abilities:
                if ab.trigger != trigger:
                    continue
                if listener is subject and (exclude_subject or ab.excludes_self):
                    continue
                if subject is not None:
                    if ab.filter_race is not None and not minion_matches_tribe(
                        subject, ab.filter_race
                    ):
                        continue
                    if (
                        ab.filter_victim_keyword is not None
                        and ab.filter_victim_keyword not in subject.all_keywords
                    ):
                        continue
                    if ab.filter_subject_rally and not any(
                        sub_ab.trigger is Trigger.ON_ATTACK
                        for sub_ab in subject.abilities
                    ):
                        continue
                yield listener, ab.effect

    def sync_auras(self, *, death_resolution: bool = False) -> None:
        """Recompute every minion's aura contribution from this side's board.

        Attack now reads a stored ``aura_attack`` instead of re-deriving it on
        every swing, so it has the same requirement health already had: after
        the board changes, someone must sync. In a battle the runtime does it
        (a board change marks the side dirty, ``_sync_health_all`` runs before
        anything swings); a side assembled by hand has to say so.
        """
        from .auras import _sync_health_aura_side

        _sync_health_aura_side(self, death_resolution)

    def iter_living(self) -> Iterator[BattleMinion]:
        """Walk the board, skipping anyone who dies while the walk is running.

        ``minions`` only ever holds the living, but an effect fired inside one
        of these loops can kill a minion further along it, and that death takes
        the body out of the list mid-iteration. Snapshotting the board and
        re-checking each entry as it comes up is what every such loop used to
        spell out by hand -- seventeen copies of the same two lines, each one a
        chance to forget. This is the reference simulator's ``for_each_alive``.
        """
        for m in list(self.minions):
            if m.alive:
                yield m

    def assert_no_corpses(self) -> None:
        """``minions`` holds the living. Called from ``reap_dead`` so the
        guarantee is checked rather than merely intended -- a death path that
        forgets to sweep fails here instead of quietly buffing a corpse.

        Runs under ``-O`` as a no-op, so the hot loop pays nothing in
        production runs while tests and dev runs keep the check.

        Combat sides only. ``alive`` reads ``current_health``, which is
        maintained inside a battle and nowhere else, so every minion on a shop
        board reads as dead and this would fire on all of them.
        """
        dead = [m.card_id for m in self.minions if not m.alive]
        assert not dead, f"dead minions left on the board: {dead}"

    def alive_minions(self) -> List[BattleMinion]:
        return list(self.minions)

    def alive_count(self) -> int:
        return len(self.minions)

    def has_alive(self) -> bool:
        return bool(self.minions)


@dataclass
class _CombatRuntime:
    sides: Tuple[BattleSide, BattleSide]
    rng: np.random.Generator
    combat_board_max: int
    damage_cap: int
    patch: PatchContext
    queue: Deque[BattleEvent] = field(default_factory=deque)
    next_id: int = 1
    in_death_resolution: bool = False
    death_hook: Optional[Callable[[int, str], None]] = None
    mech_hook: Optional[Callable[[int, Minion], None]] = None
    swing_damage_survivors: List[Tuple[int, int]] = field(default_factory=list)
    bonus_attack_depth: int = 0
    #: One per side: what this combat hands its owner. A seatless combat gets
    #: RecordingSeats, which collect and apply nothing — see battle/seat.py.
    seats: Tuple["CombatSeat", "CombatSeat"] = field(
        default_factory=lambda: (RecordingSeat(), RecordingSeat())
    )
    kill_attribution: dict[Tuple[int, int], Tuple[int, int]] = field(
        default_factory=dict
    )
    attacker_killed_this_swing: bool = False
    #: Whether any minion in this combat watches its own Attack for a keyword
    #: latch (Scarlet Survivor). Almost always False, and _sync_health_all runs
    #: on every board change, so the flag keeps that hot path free of a scan.
    #: Hand cards already summoned into this fight, per side, by instance id.
    #: A card summoned from hand is locked for the rest of the battle — it does
    #: not leave the hand, and it cannot be summoned a second time — so a second
    #: Rally has to reach past it to the next one.
    hand_summoned: Tuple[set, set] = field(default_factory=lambda: (set(), set()))
    #: Buffs installed for the rest of the fight ("For the rest of this combat,
    #: your Beasts have +1 Attack"). Applied to the board when they land and to
    #: everything summoned afterwards, which is what makes them last.
    lasting_buffs: Tuple[list, list] = field(default_factory=lambda: ([], []))
    #: Buffs installed for the rest of the fight ("For the rest of this combat,
    #: your Beasts have +1 Attack"). Applied to the board when they land and to
    #: everything summoned afterwards, which is what makes them last.
    lasting_buffs: Tuple[list, list] = field(default_factory=lambda: ([], []))
    #: The body currently taking a swing, for the cards that are immune only
    #: while attacking. Set around the exchange and cleared after it.
    swinging_instance_id: int = -1
    watch_attack_thresholds: bool = False
    health_aura_dirty: List[bool] = field(default_factory=lambda: [True, True])
    health_aura_dr_snapshot: Optional[bool] = None

    def alloc_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    def side(self, idx: int) -> BattleSide:
        return self.sides[idx]

    def find_minion(self, side_idx: int, instance_id: int) -> Optional[BattleMinion]:
        side = self.side(side_idx)
        for m in side.minions:
            if m.instance_id == instance_id:
                return m
        # Events outlive the body: a MinionDied queued when health hit 0 is
        # dispatched after the corpse has left the board, and the handler still
        # needs it to fire the deathrattle.
        for m in side.graveyard:
            if m.instance_id == instance_id:
                return m
        return None

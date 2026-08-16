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
    bm.instance_id = instance_id
    bm.damage_taken = 0
    bm.has_shield = minion.has_shield and Keyword.SHIELD in minion.all_keywords
    bm.deathrattle_fired = False
    bm.reborn_consumed = False
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

        A listener does not hear an event about itself, except for
        ON_FRIENDLY_KILL: the killer is a friendly minion and hears its own
        kill. That is the one caller passing ``exclude_subject=False``.
        """
        from src.bg_core.board_helpers import minion_matches_tribe

        for listener in self.iter_living():
            if exclude_subject and listener is subject:
                continue
            for ab in listener.abilities:
                if ab.trigger != trigger:
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
    combat_gold: List[int] = field(default_factory=lambda: [0, 0])
    combat_hand_adds: List[List[str]] = field(default_factory=lambda: [[], []])
    kill_attribution: dict[Tuple[int, int], Tuple[int, int]] = field(
        default_factory=dict
    )
    attacker_killed_this_swing: bool = False
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

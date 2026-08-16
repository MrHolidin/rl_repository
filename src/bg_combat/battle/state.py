"""Mutable combat state: minions, sides, and the per-battle runtime."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion

from .events import BattleEvent


@dataclass
class BattleMinion:
    template: Minion
    current_health: int
    shield_armed: bool
    deathrattle_fired: bool = False
    reborn_consumed: bool = False
    instance_id: int = 0
    health_aura_snapshot: int = 0
    # Board slot this minion occupied when it died, recorded at death so a
    # deathrattle can summon into it and Reborn can come back there once the
    # body itself is out of ``BattleSide.minions``. -1 while alive.
    death_pos: int = -1
    # MinionDied has been queued for this body. Removal from the board and the
    # announcement are separate: the body leaves where it dies, the event is
    # raised at the end of the exchange in a fixed side order.
    death_announced: bool = False

    @property
    def alive(self) -> bool:
        return self.current_health > 0

    @property
    def raw_attack(self) -> int:
        return self.template.raw_attack

    @property
    def tier(self) -> int:
        return self.template.tier

    @classmethod
    def from_minion(cls, minion: Minion, instance_id: int) -> "BattleMinion":
        armed = minion.has_shield and Keyword.SHIELD in minion.all_keywords
        return cls(
            template=minion,
            current_health=minion.max_health,
            shield_armed=armed,
            instance_id=instance_id,
        )


@dataclass
class BattleSide:
    minions: List[BattleMinion] = field(default_factory=list)
    # Bodies that have left the board, in death order. They are kept because
    # events already in flight reference them by ``instance_id`` and because a
    # deathrattle resolves after the body is gone; ``BattleMinion.death_pos``
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
        return taken

    def shift_graveyard_slots(self, at: int, delta: int) -> None:
        """Keep recorded death slots valid as the board around them moves."""
        # Strictly to the right: the slot ``at`` itself belongs to the body
        # being removed from it, or to the body whose token is filling it, and
        # in neither case does that body's own recorded slot move.
        for bm in self.graveyard:
            if bm.death_pos > at:
                bm.death_pos += delta

    def assert_no_corpses(self) -> None:
        """The invariant everything below leans on: ``minions`` is the living.

        Dead bodies are swept into ``graveyard`` at every death site. Without
        this check the guarantee would be an observation rather than a rule,
        and the next death path that forgets to sweep would go back to being
        silently wrong instead of loudly.
        """
        dead = [m.template.card_id for m in self.minions if not m.alive]
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

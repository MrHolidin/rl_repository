"""The seat a real combat writes to — the live half of ``battle/seat.py``.

Two kinds of thing come out of a fight, and they are not the same kind:

**Applied on the spot**, because the effect reads or writes seat state at the
moment it happens. A permanent Blood Gem is worth the seat's *current* Gem
value, and another Rally in the same combat can raise that value first; collect
the request and price it afterwards and the order is wrong.

**Batched to after the fight**, because a rule says so. Cards handed to a full
hand, and the triple they may complete, resolve once when the combat is over —
not one at a time mid-fight. Gold likewise: nothing in a combat reads the
seat's gold, and applying it in one go keeps the lobby's existing order.

So this inherits the recording seat and overrides only the live operations.
What it does *not* touch is the board's combat state: damage, popped shields
and spent Venomous still die with the copy. The writes here are the explicit,
printed ones — "permanent", "this game".
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.bg_combat.battle.seat import RecordingSeat
from src.bg_core.minion import Minion
from src.bg_lobby.player import PlayerState

from .blood_gems import blood_gem_value, give_blood_gems, play_blood_gem_on
from .standing_bonuses import BonusScope, raise_standing_bonus

__all__ = ["PlayerCombatSeat"]


class PlayerCombatSeat(RecordingSeat):
    """A combat seat bound to a real ``PlayerState``."""

    def __init__(self, player: PlayerState) -> None:
        super().__init__()
        self.player = player

    # --- live: read and write the seat as the combat runs ------------------

    def blood_gem_value(self) -> Tuple[int, int]:
        return blood_gem_value(self.player)

    def raise_blood_gem_value(self, attack: int, health: int) -> None:
        self.player.blood_gem_bonus_attack += int(attack)
        self.player.blood_gem_bonus_health += int(health)

    def play_permanent_blood_gem(self, instance_id: int, count: int = 1) -> None:
        target = self._board_minion(instance_id)
        if target is None:
            # The body died before the Gem landed, or it was a summoned token
            # with no counterpart on the shop board. Either way there is nothing
            # permanent to write to, and the combat copy is about to be thrown
            # away regardless.
            return
        play_blood_gem_on(self.player, target, count=count)

    def gain_blood_gems(self, count: int) -> None:
        give_blood_gems(self.player, int(count))

    def raise_standing_bonus(
        self, scope_kind: object, scope_key: object, attack: int, health: int
    ) -> None:
        raise_standing_bonus(
            self.player, BonusScope(scope_kind, scope_key), attack, health
        )

    def hand_card_ids(self) -> Tuple[str, ...]:
        """The minions in hand, by card id. Spells are not minions and a Start
        of Combat that summons a copy of itself has nothing to say about them."""
        return tuple(
            card.card_id
            for card in self.player.hand
            if isinstance(card, Minion)
        )

    def _board_minion(self, instance_id: int) -> Optional[Minion]:
        """The owner's real minion behind a combat copy, by identity."""
        for minion in self.player.board:
            if minion.instance_id == instance_id:
                return minion
        return None

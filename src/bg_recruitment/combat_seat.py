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
from src.bg_core.minion import Minion, is_locked
from src.bg_lobby.player import PlayerState

from .blood_gems import blood_gem_value, give_blood_gems, play_blood_gem_on
from .game_counts import bump_game_count
from .hand_slots import first_free_hand_slot
from .standing_bonuses import (
    BonusScope,
    raise_standing_bonus,
    settle_one_standing_bonus,
)

__all__ = ["PlayerCombatSeat"]


class PlayerCombatSeat(RecordingSeat):
    """A combat seat bound to a real ``PlayerState``."""

    def __init__(self, player: PlayerState, *, patch=None) -> None:
        super().__init__()
        self.player = player
        # Only the rewards need it — a card handed over mid-fight has to be
        # built from somewhere — so a seat without one still fights fine and
        # simply cannot pay those out.
        self.patch = patch

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

    def settle_standing_bonus(self, minion: object) -> None:
        settle_one_standing_bonus(self.player, minion)

    def add_refresh_buff(self, attack: int, health: int) -> None:
        self.player.refresh_buffs = self.player.refresh_buffs + (
            (int(attack), int(health)),
        )

    def raise_tavern_spell_bonus(self, attack: int, health: int) -> None:
        self.player.tavern_spell_bonus_attack += int(attack)
        self.player.tavern_spell_bonus_health += int(health)

    def record_damage_dealt(
        self, instance_id: int, amount: int, threshold: int, reward_card_id: str
    ) -> None:
        body = self._board_minion(instance_id)
        if body is None or body.damage_reward_paid:
            return
        body.damage_dealt_total += int(amount)
        if body.damage_dealt_total < threshold:
            return
        body.damage_reward_paid = True
        slot = first_free_hand_slot(self.player)
        spell = (
            self.patch.tavern_spells.get(reward_card_id)
            if self.patch is not None
            else None
        )
        if slot is not None and spell is not None:
            self.player.hand[slot] = spell

    def promise_refresh_card(self, card_id: str, refreshes: int) -> None:
        have = self.player.refresh_promises.get(card_id, 0)
        self.player.refresh_promises[card_id] = have + int(refreshes)

    def give_lockbox(self, sooner: int) -> None:
        from .lockbox import give_lockbox

        give_lockbox(self.player, sooner=int(sooner))

    def raise_tribe_gift(self, tribe: object, attack: int, health: int) -> None:
        self.player.elemental_gift_attack += int(attack)
        self.player.elemental_gift_health += int(health)

    def improve_body(self, instance_id: int) -> None:
        body = self._board_minion(instance_id)
        if body is not None:
            body.self_improves += 1

    def improve_level(self, counter: str, per: int) -> int:
        from .game_counts import improve_level

        return improve_level(self.player, counter, per)

    def bump_game_count(self, family: str, subject: str) -> None:
        bump_game_count(self.player, family, subject)

    def hand_minions(self) -> Tuple[Tuple[int, str, int, int], ...]:
        return tuple(
            (card.instance_id, card.card_id, card.raw_attack, card.max_health)
            for card in self.player.hand
            if isinstance(card, Minion) and not is_locked(card)
        )

    def buff_hand_minion(self, attack: int, health: int, *, rng) -> None:
        held = [
            card
            for card in self.player.hand
            if isinstance(card, Minion) and not is_locked(card)
        ]
        if not held:
            return
        target = held[int(rng.integers(0, len(held)))]
        target.bonus_attack += int(attack)
        target.bonus_health += int(health)

    def keep_combat_gains(self, instance_id: int, attack: int, health: int, keywords) -> None:
        target = self._board_minion(instance_id)
        if target is None:
            return
        target.bonus_attack += int(attack)
        target.bonus_health += int(health)
        if keywords:
            target.granted_keywords = target.granted_keywords | frozenset(keywords)

    def _board_minion(self, instance_id: int) -> Optional[Minion]:
        """The owner's real minion behind a combat copy, by identity."""
        for minion in self.player.board:
            if minion.instance_id == instance_id:
                return minion
        return None

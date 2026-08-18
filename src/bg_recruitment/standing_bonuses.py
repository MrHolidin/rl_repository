"""Standing bonuses — the "this game" modifiers a seat accumulates.

Thirty-seven of the pool's unbound cards say *this game*, and under the wording
they are one mechanic: the seat acquires a modifier scoped to a class of card,
and every card in that class carries it — the ones it already owns, the ones it
buys later, and the ones summoned mid-combat.

    "Your Undead have +1 Attack this game (wherever they are)"   — a tribe
    "Your Beetles have +2/+1 this game"                          — a card id
    "Has +4/+2 for each Eternal Knight that died this game"      — its own id,
                                                                   raised on an
                                                                   event
    "Minions in the Tavern have +5/+5 this game"                 — a zone

"Wherever they are" is what makes this a seat property rather than a buff: a
Beetle bought three turns after Forest Rover died still has the +2/+1. So the
seat holds the table and the cards catch up to it.

**Catching up, not hooking.** A minion enters an owned zone in eight different
modules — bought, summoned, discovered, handed over by a deathrattle, rolled
into the shop, opened out of a Lockbox — and a hook in each is a hook someone
will forget. Instead every minion remembers what it has already absorbed
(``standing_attack`` / ``standing_health``) and ``settle`` hands out the
difference. That makes settling idempotent, so it is safe to call from anywhere
and cheap on a seat with no bonuses at all.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from src.bg_core.board_helpers import minion_matches_tribe
from src.bg_core.effects import ScopeKind
from src.bg_core.minion import Minion
from src.bg_lobby.player import PlayerState

__all__ = [
    "BonusScope",
    "ScopeKind",
    "raise_standing_bonus",
    "standing_bonus_for",
    "settle_standing_bonuses",
]


class BonusScope(tuple):
    """A ``(kind, key)`` pair, hashable so the seat can key a dict by it.

    A tuple subclass rather than a frozen dataclass because it is used as a
    dict key on a hot-ish path and it is exactly a pair.
    """

    __slots__ = ()

    def __new__(cls, kind: ScopeKind, key: Any = None) -> "BonusScope":
        return super().__new__(cls, (kind, key))

    @property
    def kind(self) -> ScopeKind:
        return self[0]

    @property
    def key(self) -> Any:
        return self[1]


def _scope_hits(scope: BonusScope, minion: Minion, *, in_shop: bool) -> bool:
    kind = scope.kind
    if kind is ScopeKind.CARD:
        return minion.card_id == scope.key
    if kind is ScopeKind.TRIBE:
        return minion_matches_tribe(minion, scope.key)
    if kind is ScopeKind.SHOP:
        return in_shop
    raise ValueError(f"unhandled ScopeKind {kind!r}")


def standing_bonus_for(
    player: PlayerState, minion: Minion, *, in_shop: bool = False
) -> Tuple[int, int]:
    """Everything the seat's table owes ``minion``, summed."""
    attack = health = 0
    for scope, (bonus_attack, bonus_health) in player.standing_bonuses.items():
        if _scope_hits(scope, minion, in_shop=in_shop):
            attack += bonus_attack
            health += bonus_health
    return attack, health


def _settle_one(player: PlayerState, minion: Minion, *, in_shop: bool) -> None:
    owed_attack, owed_health = standing_bonus_for(player, minion, in_shop=in_shop)
    delta_attack = owed_attack - minion.standing_attack
    delta_health = owed_health - minion.standing_health
    if not delta_attack and not delta_health:
        return
    minion.bonus_attack += delta_attack
    minion.bonus_health += delta_health
    minion.standing_attack = owed_attack
    minion.standing_health = owed_health


def settle_standing_bonuses(player: PlayerState) -> None:
    """Hand every card the seat owns whatever the table owes it and it lacks.

    Idempotent: a card that is already square with the table is untouched, so
    this is safe to call after anything and costs one attribute read per card
    on a seat that has no standing bonuses — which is most seats, most games.
    """
    if not player.standing_bonuses:
        return
    for minion in player.board:
        _settle_one(player, minion, in_shop=False)
    for card in player.hand:
        if isinstance(card, Minion):
            _settle_one(player, card, in_shop=False)
    for card in player.shop:
        if isinstance(card, Minion):
            _settle_one(player, card, in_shop=True)


def raise_standing_bonus(
    player: PlayerState,
    scope: BonusScope,
    attack: int,
    health: int,
) -> None:
    """Raise (or open) a standing bonus, and let everyone catch up to it."""
    have_attack, have_health = player.standing_bonuses.get(scope, (0, 0))
    player.standing_bonuses[scope] = (have_attack + int(attack), have_health + int(health))
    settle_standing_bonuses(player)

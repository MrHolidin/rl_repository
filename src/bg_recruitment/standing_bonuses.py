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
from src.bg_core.minion import Minion, is_locked
from src.bg_lobby.player import PlayerState

__all__ = [
    "BonusScope",
    "ScopeKind",
    "raise_standing_bonus",
    "standing_bonus_for",
    "settle_one_standing_bonus",
    "settle_standing_bonuses",
]


class BonusScope(tuple):
    """A ``(kind, key, max_tier)`` triple, hashable so the seat can key a dict.

    ``key`` is the card id for CARD, the tribe for TRIBE, and an *optional*
    tribe for SHOP; ``max_tier`` is a SHOP-only cap. A tuple subclass rather
    than a frozen dataclass because it is exactly that and it is a dict key.
    """

    __slots__ = ()

    def __new__(cls, kind: ScopeKind, key: Any = None, max_tier: int = 0) -> "BonusScope":
        return super().__new__(cls, (kind, key, int(max_tier)))

    @property
    def kind(self) -> ScopeKind:
        return self[0]

    @property
    def key(self) -> Any:
        return self[1]

    @property
    def max_tier(self) -> int:
        return self[2]


def _scope_hits(scope: BonusScope, minion: Minion, *, in_shop: bool) -> bool:
    kind = scope.kind
    if kind is ScopeKind.CARD:
        return minion.card_id == scope.key
    if kind is ScopeKind.TRIBE:
        return minion_matches_tribe(minion, scope.key)
    if kind is ScopeKind.SHOP:
        if not in_shop:
            return False
        if scope.key is not None and not minion_matches_tribe(minion, scope.key):
            return False
        return not scope.max_tier or minion.tier <= scope.max_tier
    raise ValueError(f"unhandled ScopeKind {kind!r}")


def standing_bonus_for(
    player: PlayerState, minion: Minion, *, in_shop: bool = False
) -> Tuple[int, int]:
    """Everything the seat's table currently offers ``minion``, summed."""
    attack = health = 0
    for scope, (bonus_attack, bonus_health) in player.standing_bonuses.items():
        if _scope_hits(scope, minion, in_shop=in_shop):
            attack += bonus_attack
            health += bonus_health
    return attack, health


def _settle_one(player: PlayerState, minion: Minion, *, in_shop: bool) -> None:
    """Pay ``minion`` whatever the table offers it and it has not had yet.

    Only ever pays. A bonus already absorbed is never reclaimed, which is the
    rule for the tavern ones: a minion buffed on the counter keeps the stats
    after it is bought, so once its slot stops being "in the shop" the scope
    simply stops offering more.
    """
    absorbed = dict((row[0], (row[1], row[2])) for row in minion.standing_absorbed)
    changed = False
    for scope, (offer_attack, offer_health) in player.standing_bonuses.items():
        if not _scope_hits(scope, minion, in_shop=in_shop):
            continue
        had_attack, had_health = absorbed.get(scope, (0, 0))
        delta_attack = max(0, offer_attack - had_attack)
        delta_health = max(0, offer_health - had_health)
        if not delta_attack and not delta_health:
            continue
        minion.bonus_attack += delta_attack
        minion.bonus_health += delta_health
        absorbed[scope] = (had_attack + delta_attack, had_health + delta_health)
        changed = True
    if changed:
        minion.standing_absorbed = tuple(
            (scope, a, h) for scope, (a, h) in absorbed.items()
        )


def settle_one_standing_bonus(player: PlayerState, minion: Minion) -> None:
    """Pay one body, wherever it is standing — including inside a fight.

    The zone-scoped bonuses ("minions **in the Tavern**") do not reach it, and
    nothing else needs to know where it is. Public because combat needs it: a
    token summoned mid-fight is a minion the seat owns, and "wherever they are"
    is printed on the cards that raise these.
    """
    if not player.standing_bonuses:
        return
    _settle_one(player, minion, in_shop=False)


def settle_standing_bonuses(player: PlayerState) -> None:
    """Hand every card the seat owns whatever the table owes it and it lacks.

    Idempotent: a card already square with the table is untouched, so this is
    safe to call after anything and costs one attribute read per card on a seat
    with no standing bonuses — which is most seats, most games.
    """
    if not player.standing_bonuses:
        return
    for minion in player.board:
        _settle_one(player, minion, in_shop=False)
    for card in player.hand:
        if isinstance(card, Minion) and not is_locked(card):
            _settle_one(player, card, in_shop=False)
    for card in player.shop:
        if isinstance(card, Minion):
            _settle_one(player, card, in_shop=True)


def _mark_absorbed(player: PlayerState, minion: Minion, scope: BonusScope) -> None:
    """Record ``minion`` as square with ``scope`` without paying it anything."""
    offered = player.standing_bonuses.get(scope, (0, 0))
    rows = [row for row in minion.standing_absorbed if row[0] != scope]
    rows.append((scope, offered[0], offered[1]))
    minion.standing_absorbed = tuple(rows)


def raise_standing_bonus(
    player: PlayerState,
    scope: BonusScope,
    attack: int,
    health: int,
    *,
    exclude: Optional[Minion] = None,
) -> None:
    """Raise (or open) a standing bonus, and let everyone catch up to it.

    ``exclude`` is the card whose own arrival caused the raise, and it is what
    makes "for each **other** Ancestral Automaton you've summoned" come out
    right: the newcomer banks every raise before its own and skips that one, so
    with N copies each carries exactly N-1 raises.
    """
    have_attack, have_health = player.standing_bonuses.get(scope, (0, 0))
    player.standing_bonuses[scope] = (have_attack + int(attack), have_health + int(health))
    if exclude is not None:
        _mark_absorbed(player, exclude, scope)
    settle_standing_bonuses(player)

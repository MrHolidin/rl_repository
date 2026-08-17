"""A spell in hand — a card that isn't a minion.

Kept as its own type — not a ``Minion`` subclass or union-of-fields hack —
because a spell genuinely has no attack/health/race/tier-as-a-minion.

**"Spell" and "Tavern spell" are different things**, and Battlegrounds cards
depend on the difference. A Tavern spell is bought off the tavern counter
(``BATTLEGROUND_SPELL``, spell school TAVERN, its own tier and pool); a Blood
Gem is a plain spell that goes straight to hand and never appears in the
tavern. Cards read as "Whenever you cast a **spell** on this" fire for both,
while "Whenever you cast a **Tavern spell**" fires only for the former — and
Timewarped Bloodbinder ("get 5 Blood Gems. They also count as Tavern spells")
only makes sense because the two are separate by default. Hence
``is_tavern_spell`` as a flag on one type rather than two types or, worse, a
type named for the narrower meaning and used for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .effects import Ability, Keyword


@dataclass(frozen=True)
class SpellCard:
    card_id: str
    name: str = ""
    cost: int = 0
    tier: int = 0
    abilities: Tuple[Ability, ...] = field(default_factory=tuple)
    dbf_id: Optional[int] = None
    #: Bought off the tavern counter, and counted by "Tavern spell" listeners.
    #: False for a Blood Gem and for the triple-reward discover reward, neither
    #: of which is ever offered in the tavern.
    is_tavern_spell: bool = False
    #: Only consumed by the triple-reward-discover spell today (which tavern
    #: tier its Discover offers). Real spell content is expected to leave this
    #: at 0 and drive behavior through ``abilities`` instead.
    triple_discover_tier: int = 0
    #: Made by a Naga's Spellcraft rather than bought. Its own school in the
    #: card data (``spellSchool=SPELLCRAFT``, next to TAVERN), and cards read
    #: it back: "the first Spellcraft spell played on this each turn is
    #: permanent", "when a Spellcraft spell is played on this, get a copy".
    #: Such a spell is discarded at end of turn if unused.
    is_spellcraft: bool = False
    #: A Blood Gem: +1/+1 (plus the seat's accumulated Gem bonus) onto one
    #: friendly minion. Its own flag rather than a card_id comparison because
    #: four printings exist — the plain Gem and three that also hand a Quilboar
    #: a keyword.
    is_blood_gem: bool = False
    #: Blood Gem printings that also grant a keyword, but only to a Quilboar.
    blood_gem_quilboar_keyword: Optional[Keyword] = None
    #: A Lockbox: unplayable, counts down a turn at a time and opens into a
    #: random Golden minion at zero. 0 on every other card.
    turns_until_open: int = 0


__all__ = ["SpellCard"]

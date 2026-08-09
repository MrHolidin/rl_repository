"""Battleground tavern spell — a hand card that isn't a minion.

First (and, until real spell content lands, only) instance: the triple-reward
discover spell, migrated off the ``Minion.is_triple_reward_spell`` hack it
used to be faked as (see ``src/bg_recruitment/triples.py``). Kept as its own
type — not a ``Minion`` subclass or union-of-fields hack — because a spell
genuinely has no attack/health/race/tier-as-a-minion; forcing it into
``Minion``'s shape is exactly the thing this migration retires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .effects import Ability


@dataclass(frozen=True)
class TavernSpell:
    card_id: str
    name: str = ""
    cost: int = 0
    tier: int = 0
    abilities: Tuple[Ability, ...] = field(default_factory=tuple)
    dbf_id: Optional[int] = None
    # Only consumed by the triple-reward-discover spell today (which tavern
    # tier its Discover offers). Real spell content is expected to leave this
    # at 0 and drive behavior through ``abilities`` instead.
    triple_discover_tier: int = 0


__all__ = ["TavernSpell"]

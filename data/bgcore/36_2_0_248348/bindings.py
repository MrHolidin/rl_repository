"""Card ability bindings for patch 36.2.0 (build 248348).

Empty on purpose. The catalog is complete — 274 pool minions, 75 tavern spells,
390 trinkets, 121 heroes — and nothing here is bound yet, so every card is a
vanilla body with its printed stats and keywords.

That is the honest starting state for a package this size, and it is what makes
the work visible: ``scripts/check_patch_coverage.py data/bgcore/36_2_0_248348``
lists every pool card whose text promises something no binding delivers. The
list is the queue; it shrinks as bindings land, and the checker fails the day a
binding names a card the catalog does not have.

Bind in tier order — a tier-1 card is reachable in every game, a tier-7 card in
few — and lean on the engine mechanics already in place: Blood Gems, Rally,
Spellcraft with its "until next turn" buffs, Activate, Choose One, Venomous,
Avenge, Lockbox, Fishbait.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

from src.bg_core.effects import Ability

#: Golden rewards ("Get a Discover of a higher tier") — none bound yet.
GOLDEN_REWARD_IDS: FrozenSet[str] = frozenset()

#: Tokens summoned by bound cards. Grows with the deathrattles that summon them.
TOKEN_IDS: FrozenSet[str] = frozenset()

#: Pool cards whose whole text is keywords the catalog already carries
#: (a plain Taunt/Divine Shield body), so they need no binding to be correct.
KEYWORD_ONLY_POOL_IDS: FrozenSet[str] = frozenset()

EFFECTS: Dict[str, Tuple[Ability, ...]] = {}

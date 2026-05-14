"""BG-style discover pools (tier-weighted) and Adapt option sets."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .actions import MAX_TIER
from .cards import CARD_TEMPLATES
from .effects import Ability, Keyword, SummonEffect, Trigger
from .state import Minion, Race

# Keys for Gentle Megasaur–style Adapt (HS Journey to Un'Goro set).
ADAPT_KEYS_ALL: Tuple[str, ...] = (
    "adapt_volcanic_might",  # +1/+1
    "adapt_crackling_shield",  # Divine Shield
    "adapt_flaming_claws",  # +3 Attack
    "adapt_living_spore",  # Deathrattle: summon two 2/2 tokens
    "adapt_lightning_speed",  # Windfury
    "adapt_razor_claws",  # +1 Attack
    "adapt_rocky_carapace",  # +3 Health
    "adapt_rockshell_armadillo",  # +1/+3 Taunt
    "adapt_massive",  # +3/+3
    "adapt_molten_blade",  # +1/+2
)

assert len(ADAPT_KEYS_ALL) == 10

# Approximate BG discover: offered tiers cluster around tavern tier (patch-era heuristic).
# For tavern T, eligible discover tiers are 1..min(T+1, MAX_TIER) with higher weight at T.
def _tier_weights(tavern_tier: int) -> Dict[int, float]:
    hi = min(MAX_TIER, tavern_tier + 1)
    w: Dict[int, float] = {}
    for t in range(1, hi + 1):
        dist = abs(t - tavern_tier)
        w[t] = 1.0 / (1.0 + float(dist) * float(dist))
    return w


def murloc_discover_card_ids() -> List[str]:
    return [
        cid
        for cid, m in CARD_TEMPLATES.items()
        if not m.is_token and m.race == Race.MURLOC
    ]


def roll_discover_murloc_triple(
    rng: np.random.Generator,
    tavern_tier: int,
    shop_excluded_race: Optional[Race] = None,
) -> Tuple[str, str, str]:
    """Three distinct discover options; tier-weighted by current tavern tier (BG-style)."""
    cap = min(MAX_TIER, tavern_tier + 1)
    if shop_excluded_race == Race.MURLOC:
        eligible: List[str] = []
    else:
        eligible = [
            cid
            for cid in murloc_discover_card_ids()
            if CARD_TEMPLATES[cid].tier <= cap
        ]
    if len(eligible) < 3:
        raise RuntimeError(
            f"need at least 3 murlocs for discover (tavern {tavern_tier}), got {len(eligible)}"
        )
    wmap = _tier_weights(tavern_tier)
    pool = list(eligible)
    picks: List[str] = []
    for _ in range(3):
        w = np.array([wmap.get(CARD_TEMPLATES[cid].tier, 0.1) for cid in pool], dtype=np.float64)
        w = w / w.sum()
        j = int(rng.choice(len(pool), p=w))
        picks.append(pool.pop(j))
    return (picks[0], picks[1], picks[2])


def roll_adapt_triple(rng: np.random.Generator) -> Tuple[str, str, str]:
    """Three distinct random options from the ten Adapt choices."""
    idx = rng.choice(len(ADAPT_KEYS_ALL), size=3, replace=False)
    keys = [ADAPT_KEYS_ALL[int(i)] for i in idx]
    return (keys[0], keys[1], keys[2])


def is_murloc_board_minion(m: Minion) -> bool:
    return m.race in (Race.MURLOC, Race.ALL)


def apply_adapt_key_to_minion(m: Minion, key: str) -> None:
    """Apply one Adapt choice to a single minion (Gentle Megasaur battlecry on each Murloc)."""
    if key == "adapt_volcanic_might":
        m.bonus_attack += 1
        m.bonus_health += 1
    elif key == "adapt_crackling_shield":
        m.has_shield = True
        m.keywords = frozenset(m.keywords | {Keyword.SHIELD})
    elif key == "adapt_flaming_claws":
        m.bonus_attack += 3
    elif key == "adapt_living_spore":
        m.abilities = m.abilities + (
            Ability(
                Trigger.ON_DEATH,
                SummonEffect(token_id="adapt_plant", count=2),
            ),
        )
    elif key == "adapt_lightning_speed":
        m.keywords = frozenset(m.keywords | {Keyword.WINDFURY})
    elif key == "adapt_razor_claws":
        m.bonus_attack += 1
    elif key == "adapt_rocky_carapace":
        m.bonus_health += 3
    elif key == "adapt_rockshell_armadillo":
        m.bonus_attack += 1
        m.bonus_health += 3
        m.keywords = frozenset(m.keywords | {Keyword.TAUNT})
    elif key == "adapt_massive":
        m.bonus_attack += 3
        m.bonus_health += 3
    elif key == "adapt_molten_blade":
        m.bonus_attack += 1
        m.bonus_health += 2
    else:
        raise ValueError(f"unknown adapt key {key!r}")


__all__ = [
    "ADAPT_KEYS_ALL",
    "murloc_discover_card_ids",
    "roll_discover_murloc_triple",
    "roll_adapt_triple",
    "is_murloc_board_minion",
    "apply_adapt_key_to_minion",
]

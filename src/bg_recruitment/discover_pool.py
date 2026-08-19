"""BG-style discover pools (tier-weighted) and Adapt option sets."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.bg_catalog.cards import (
    normalize_shop_excluded_races,
    shop_minion_allowed_with_exclusion,
    shop_pool_for_tier,
    templates,
)
from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_catalog.ruleset import DEFAULT_RULESET
from src.bg_core.effects import Ability, Keyword, SummonEffect, Trigger
from src.bg_core.minion import Minion, Race
from src.bg_lobby.shared_pool import SharedCardPool

# Keys for Gentle Megasaur–style Adapt (HS Journey to Un'Goro set).
ADAPT_KEYS_ALL: Tuple[str, ...] = (
    "adapt_volcanic_might",
    "adapt_crackling_shield",
    "adapt_flaming_claws",
    "adapt_living_spore",
    "adapt_lightning_speed",
    "adapt_razor_claws",
    "adapt_rocky_carapace",
    "adapt_rockshell_armadillo",
    "adapt_massive",
    "adapt_molten_blade",
)

assert len(ADAPT_KEYS_ALL) == 10


def _tier_weights(tavern_tier: int, max_tier: int) -> Dict[int, float]:
    hi = min(max_tier, tavern_tier + 1)
    w: Dict[int, float] = {}
    for t in range(1, hi + 1):
        dist = abs(t - tavern_tier)
        w[t] = 1.0 / (1.0 + float(dist) * float(dist))
    return w


def tribe_discover_card_ids(tribe: Race, *, patch: PatchContext) -> List[str]:
    tpl = templates(patch=require_patch(patch, where="discover_pool.tribe_discover_card_ids"))
    return [
        cid
        for cid, m in tpl.items()
        if not m.is_token and m.race == tribe
    ]


def roll_discover_tribe_triple(
    rng: np.random.Generator,
    tavern_tier: int,
    shop_excluded_race: Optional[Race] = None,
    *,
    tribe: Race,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> Optional[Tuple[str, str, str]]:
    """Three options of one tribe, weighted toward the seat's own tavern tier.

    A tribe left out of the lobby offers nothing, which is the same answer the
    tavern gives: there are no Undead to Discover in a game without Undead.
    """
    ctx = require_patch(patch, where="discover_pool.roll_discover_tribe_triple")
    tpl = ctx.templates
    max_tier = ctx.meta.ruleset.max_tier
    cap = min(max_tier, tavern_tier + 1)
    if tribe in normalize_shop_excluded_races(shop_excluded_race):
        eligible: List[str] = []
    else:
        eligible = [
            cid
            for cid in tribe_discover_card_ids(tribe, patch=ctx)
            if tpl[cid].tier <= cap
        ]
    if shared_pool is not None:
        eligible = [cid for cid in eligible if shared_pool.remaining_copies(cid) > 0]
    if len(eligible) < 3:
        if shared_pool is not None:
            return None
        raise RuntimeError(
            f"need at least 3 {tribe} for discover (tavern {tavern_tier}), "
            f"got {len(eligible)}"
        )
    wmap = _tier_weights(tavern_tier, max_tier)
    pool = list(eligible)
    picks: List[str] = []
    for _ in range(3):
        w = np.array([wmap.get(tpl[cid].tier, 0.1) for cid in pool], dtype=np.float64)
        w = w / w.sum()
        j = int(rng.choice(len(pool), p=w))
        picks.append(pool.pop(j))
    return (picks[0], picks[1], picks[2])


def roll_adapt_triple(rng: np.random.Generator) -> Tuple[str, str, str]:
    idx = rng.choice(len(ADAPT_KEYS_ALL), size=3, replace=False)
    keys = [ADAPT_KEYS_ALL[int(i)] for i in idx]
    return (keys[0], keys[1], keys[2])


def triple_reward_discover_tier(
    tavern_tier: int, *, patch: Optional[PatchContext] = None
) -> int:
    max_tier = patch.meta.ruleset.max_tier if patch is not None else DEFAULT_RULESET.max_tier
    return min(max_tier, int(tavern_tier) + 1)


def roll_triple_reward_discover_at_target_tier(
    rng: np.random.Generator,
    target_tier: int,
    shop_excluded_race: Optional[Race] = None,
    *,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> Optional[Tuple[str, str, str]]:
    ctx = require_patch(patch, where="discover_pool.roll_triple_reward_discover_at_target_tier")
    tpl = ctx.templates
    tgt = min(ctx.meta.ruleset.max_tier, max(1, int(target_tier)))
    eligible_exact = [
        cid
        for cid in shop_pool_for_tier(
            tgt, shop_excluded_race=shop_excluded_race, patch=ctx
        )
        if tpl[cid].tier == tgt
    ]
    eligible = eligible_exact
    if len(eligible) < 3:
        eligible = list(
            shop_pool_for_tier(tgt, shop_excluded_race=shop_excluded_race, patch=ctx)
        )
    if len(eligible) < 3:
        eligible = [
            cid
            for cid, m in tpl.items()
            if not m.is_token
            and not m.is_golden
            and m.tier <= tgt
            and shop_minion_allowed_with_exclusion(m, shop_excluded_race)
        ]
    if shared_pool is not None:
        eligible = [cid for cid in eligible if shared_pool.remaining_copies(cid) > 0]
    if len(eligible) < 3:
        if shared_pool is not None:
            return None
        raise RuntimeError(
            f"need at least 3 cards for triple-reward discover (tier {tgt}), got {len(eligible)}"
        )
    pool = list(eligible)
    picks: List[str] = []
    for _ in range(3):
        j = int(rng.integers(0, len(pool)))
        picks.append(pool.pop(j))
    return (picks[0], picks[1], picks[2])


def roll_triple_reward_discover_triple(
    rng: np.random.Generator,
    tavern_tier: int,
    shop_excluded_race: Optional[Race] = None,
    *,
    shared_pool: Optional[SharedCardPool] = None,
    patch: PatchContext,
) -> Optional[Tuple[str, str, str]]:
    return roll_triple_reward_discover_at_target_tier(
        rng,
        triple_reward_discover_tier(tavern_tier, patch=patch),
        shop_excluded_race,
        shared_pool=shared_pool,
        patch=patch,
    )


def is_murloc_board_minion(m: Minion) -> bool:
    return m.race in (Race.MURLOC, Race.ALL)


def apply_adapt_key_to_minion(m: Minion, key: str) -> None:
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
    "apply_adapt_key_to_minion",
    "is_murloc_board_minion",
    "tribe_discover_card_ids",
    "roll_adapt_triple",
    "roll_discover_tribe_triple",
    "roll_triple_reward_discover_at_target_tier",
    "roll_triple_reward_discover_triple",
    "triple_reward_discover_tier",
]

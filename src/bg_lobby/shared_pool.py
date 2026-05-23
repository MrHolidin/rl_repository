"""Lobby-wide tavern minion copy pool (retail-style shared pool)."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Mapping, Optional

import numpy as np

from src.bg_catalog.cards import shop_minion_allowed_with_exclusion
from src.bg_catalog.patch_context import DEFAULT_PATCH_DIR, PatchContext, load_patch_context, require_patch
from src.bg_core.minion import Minion, Race

_DEFAULT_PATCH = load_patch_context(str(DEFAULT_PATCH_DIR))
POOL_SIZE_BY_TIER: Dict[int, int] = dict(_DEFAULT_PATCH.meta.pool_copies_by_tier)

__all__ = [
    "POOL_SIZE_BY_TIER",
    "SharedCardPool",
    "build_initial_shared_pool",
    "copies_for_minion",
    "eligible_card_ids_for_tier",
]


def eligible_card_ids_for_tier(
    tavern_tier: int,
    shop_excluded_race: Optional[Race],
    *,
    templates: Mapping[str, Minion],
) -> List[str]:
    return [
        cid
        for cid, t in templates.items()
        if not t.is_token
        and not t.is_golden
        and not t.is_triple_reward_spell
        and t.tier <= tavern_tier
        and shop_minion_allowed_with_exclusion(t, shop_excluded_race)
    ]


def build_initial_shared_pool(
    shop_excluded_race: Optional[Race] = None,
    *,
    pool_copies_by_tier: Optional[Mapping[int, int]] = None,
    patch: PatchContext,
) -> SharedCardPool:
    ctx = require_patch(patch, where="shared_pool.build_initial_shared_pool")
    copies = (
        dict(pool_copies_by_tier)
        if pool_copies_by_tier is not None
        else dict(ctx.meta.pool_copies_by_tier)
    )
    tpl = ctx.templates
    remaining: Dict[str, int] = {}
    for cid, t in tpl.items():
        if t.is_token or t.is_golden or t.is_triple_reward_spell:
            continue
        if not shop_minion_allowed_with_exclusion(t, shop_excluded_race):
            continue
        cap = copies.get(t.tier)
        if cap is None:
            continue
        remaining[cid] = cap
    return SharedCardPool(
        remaining=remaining,
        initial=deepcopy(remaining),
        templates=tpl,
        patch=ctx,
    )


def copies_for_minion(m: Minion) -> int:
    """Copies one minion instance represents in the shared pool."""
    if m.is_triple_reward_spell:
        return 0
    if m.is_golden:
        return 3
    return 1


class SharedCardPool:
    """Tracks remaining tavern copies for the lobby.

    Variant B (retail-like offers):
    * ``reserve_offer`` when a minion is rolled into a shop slot (−1).
    * ``release_offer`` when a shop slot is cleared without purchase (+1).
    * ``acquire_new`` when a minion enters hand/board without coming from a shop slot
      (Discover, spell, etc.) (−n).
    * ``release_minion`` when sold or eliminated (+n).
    """

    __slots__ = ("remaining", "_initial", "_templates", "patch")

    def __init__(
        self,
        *,
        remaining: Dict[str, int],
        patch: PatchContext,
        initial: Optional[Dict[str, int]] = None,
        templates: Mapping[str, Minion],
    ) -> None:
        self.patch = require_patch(patch, where="SharedCardPool.__init__")
        self.remaining = remaining
        self._initial = initial if initial is not None else dict(remaining)
        self._templates = templates

    def remaining_copies(self, card_id: str) -> int:
        return int(self.remaining.get(card_id, 0))

    def total_remaining_eligible(
        self,
        tavern_tier: int,
        shop_excluded_race: Optional[Race],
    ) -> int:
        return sum(
            self.remaining_copies(cid)
            for cid in eligible_card_ids_for_tier(
                tavern_tier,
                shop_excluded_race,
                templates=self._templates,
            )
            if self.remaining_copies(cid) > 0
        )

    def try_reserve_offer(self, card_id: str) -> bool:
        n = self.remaining_copies(card_id)
        if n <= 0:
            return False
        self.remaining[card_id] = n - 1
        return True

    def release_offer(self, card_id: str, count: int = 1) -> None:
        if count <= 0:
            return
        cap = self._initial.get(card_id, 0)
        cur = self.remaining_copies(card_id)
        self.remaining[card_id] = min(cap, cur + count)

    def acquire_new(self, card_id: str, count: int = 1) -> bool:
        """Take copies that were not reserved via a shop offer (Discover, etc.)."""
        for _ in range(count):
            if not self.try_reserve_offer(card_id):
                return False
        return True

    def release_minion(self, m: Minion) -> None:
        n = copies_for_minion(m)
        if n > 0:
            self.release_offer(m.card_id, n)

    def roll_card_id(
        self,
        tavern_tier: int,
        shop_excluded_race: Optional[Race],
        rng: np.random.Generator,
    ) -> Optional[str]:
        """Weighted roll: remaining copies / sum remaining eligible."""
        ids: List[str] = []
        weights: List[float] = []
        for cid in eligible_card_ids_for_tier(
            tavern_tier,
            shop_excluded_race,
            templates=self._templates,
        ):
            w = self.remaining_copies(cid)
            if w > 0:
                ids.append(cid)
                weights.append(float(w))
        if not ids:
            return None
        w_arr = np.array(weights, dtype=np.float64)
        w_arr /= w_arr.sum()
        idx = int(rng.choice(len(ids), p=w_arr))
        return ids[idx]

    def copy(self) -> SharedCardPool:
        return SharedCardPool(
            remaining=dict(self.remaining),
            initial=dict(self._initial),
            templates=self._templates,
            patch=self.patch,
        )

    def roll_and_reserve_offer(
        self,
        tavern_tier: int,
        shop_excluded_race: Optional[Race],
        rng: np.random.Generator,
        *,
        max_attempts: int = 64,
    ) -> Optional[str]:
        for _ in range(max_attempts):
            cid = self.roll_card_id(tavern_tier, shop_excluded_race, rng)
            if cid is None:
                return None
            if self.try_reserve_offer(cid):
                return cid
        return None

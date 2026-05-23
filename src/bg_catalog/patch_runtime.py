"""Patch-scoped lobby rules derived from ``PatchContext.meta``."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_core.minion import Race

__all__ = [
    "active_shop_tribe_count",
    "pool_copies_by_tier",
    "pool_size_for_tier",
    "rotation_shop_tribes",
]


def rotation_shop_tribes(*, patch: PatchContext) -> tuple[Race, ...]:
    return require_patch(patch, where="patch_runtime.rotation_shop_tribes").meta.rotation_tribes


def active_shop_tribe_count(*, patch: PatchContext) -> int:
    return require_patch(patch, where="patch_runtime.active_shop_tribe_count").meta.cnt_active_shop_tribes


def pool_copies_by_tier(*, patch: PatchContext) -> Dict[int, int]:
    return dict(
        require_patch(patch, where="patch_runtime.pool_copies_by_tier").meta.pool_copies_by_tier
    )


def pool_size_for_tier(
    tier: int,
    *,
    pool_copies: Optional[Mapping[int, int]] = None,
    patch: PatchContext,
) -> Optional[int]:
    table = dict(pool_copies) if pool_copies is not None else pool_copies_by_tier(patch=patch)
    return table.get(tier)

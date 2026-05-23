"""BG tavern minion pools for random deathrattle summons (tier filter, not HS mana/cost)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_core.effects import Trigger
from .state import Race


def hs_race_string(race: Any) -> Optional[str]:
    if race is None or race is Race.ALL:
        return None
    rev = {
        Race.BEAST: "BEAST",
        Race.DEMON: "DEMON",
        Race.MECHANICAL: "MECHANICAL",
        Race.MURLOC: "MURLOC",
        Race.DRAGON: "DRAGON",
        Race.PIRATE: "PIRATE",
        Race.ELEMENTAL: "ELEMENTAL",
    }
    return rev.get(race)


def _record_has_deathrattle(
    rec_id: str,
    mechanics: frozenset,
    effects: dict,
) -> bool:
    if "DEATHRATTLE" in mechanics:
        return True
    return any(ab.trigger == Trigger.ON_DEATH for ab in effects.get(rec_id, ()))


@lru_cache(maxsize=256)
def build_summon_pool(
    exact_tier: Optional[int],
    legendary_only: bool,
    require_deathrattle: bool,
    race_hs: Optional[str],
    exclude_card_id: Optional[str],
    patch_dir: str,
) -> tuple[str, ...]:
    from src.bg_catalog.patch_catalog import load_tavern_minions

    ctx = PatchContext.load(Path(patch_dir))
    catalog = ctx.patch_dir / "catalog.json"
    pool: List[str] = []
    for rec in load_tavern_minions(catalog):
        cid = rec.id
        if not rec.is_bacon_pool or rec.is_golden:
            continue
        if cid in ctx.token_ids or cid in ctx.golden_reward_ids:
            continue
        if exclude_card_id is not None and cid == exclude_card_id:
            continue
        if exact_tier is not None and rec.tier != exact_tier:
            continue
        if legendary_only:
            if rec.rarity != "LEGENDARY":
                continue
        if require_deathrattle and not _record_has_deathrattle(
            cid, rec.mechanics, dict(ctx.effects)
        ):
            continue
        if race_hs is not None and rec.race != race_hs:
            continue
        pool.append(cid)
    return tuple(pool)


def summon_pool_for(
    exact_tier: Optional[int],
    legendary_only: bool,
    require_deathrattle: bool,
    race_hs: Optional[str],
    exclude_card_id: Optional[str],
    *,
    patch: PatchContext,
) -> tuple[str, ...]:
    ctx = require_patch(patch, where="summon_pool.summon_pool_for")
    return build_summon_pool(
        exact_tier,
        legendary_only,
        require_deathrattle,
        race_hs,
        exclude_card_id,
        str(ctx.patch_dir),
    )


__all__ = ["build_summon_pool", "hs_race_string", "summon_pool_for"]

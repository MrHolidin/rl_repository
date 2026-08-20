"""BG tavern minion pools for random deathrattle summons (tier filter, not HS mana/cost)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.bg_catalog.patch_context import PatchContext, require_patch
from src.bg_core.effects import Trigger
from .state import Race


def hs_race_string(race: Any) -> Optional[str]:
    """HS race string for a tribe, or ``None`` for "no single tribe".

    ``Race.ALL`` is an Amalgam, not a tribe, so it maps to ``None`` like a
    raceless minion. Every other member spells its HS string exactly as its
    enum name — derived rather than listed, so a tribe added to ``Race`` cannot
    silently read as raceless here.
    """
    if race is None or race is Race.ALL:
        return None
    return race.name if isinstance(race, Race) else None


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
    keyword_name: Optional[str] = None,
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
        if keyword_name is not None and not _record_has_keyword(rec, keyword_name):
            continue
        pool.append(cid)
    return tuple(pool)


def _record_has_keyword(rec, keyword_name: str) -> bool:
    """Whether a catalog row carries a keyword ("a random **Magnetic** Mech").

    Read off the record rather than the built template so this stays a pure
    function of the catalog, the way every other filter here is.
    """
    from src.bg_catalog.patch_catalog import keywords_for_tavern_record

    return any(k.name == keyword_name for k in keywords_for_tavern_record(rec))


def summon_pool_for(
    exact_tier: Optional[int],
    legendary_only: bool,
    require_deathrattle: bool,
    race_hs: Optional[str],
    exclude_card_id: Optional[str],
    *,
    patch: PatchContext,
    keyword=None,
) -> tuple[str, ...]:
    ctx = require_patch(patch, where="summon_pool.summon_pool_for")
    return build_summon_pool(
        exact_tier,
        legendary_only,
        require_deathrattle,
        race_hs,
        exclude_card_id,
        str(ctx.patch_dir),
        keyword.name if keyword is not None else None,
    )


__all__ = ["build_summon_pool", "hs_race_string", "summon_pool_for"]

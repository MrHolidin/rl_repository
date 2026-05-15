"""BG tavern minion pools for random deathrattle summons (tier filter, not HS mana/cost)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Optional

from .card_pool import EFFECTS, GOLDEN_REWARD_IDS, TOKEN_IDS
from .effects import Trigger
from .patch_catalog import load_tavern_minions
from .state import Race


def hs_race_string(race: Any) -> Optional[str]:
    if race is None or race is Race.ALL:
        return None
    rev = {
        Race.BEAST: "BEAST",
        Race.DEMON: "DEMON",
        Race.MECHANICAL: "MECHANICAL",
        Race.MURLOC: "MURLOC",
    }
    return rev.get(race)


def _record_has_deathrattle(rec_id: str, mechanics: frozenset) -> bool:
    if "DEATHRATTLE" in mechanics:
        return True
    return any(ab.trigger == Trigger.ON_DEATH for ab in EFFECTS.get(rec_id, ()))


@lru_cache(maxsize=256)
def build_summon_pool(
    exact_tier: Optional[int],
    legendary_only: bool,
    require_deathrattle: bool,
    race_hs: Optional[str],
    exclude_card_id: Optional[str],
) -> tuple[str, ...]:
    pool: List[str] = []
    for rec in load_tavern_minions():
        cid = rec.id
        if not rec.is_bacon_pool or rec.is_golden:
            continue
        if cid in TOKEN_IDS or cid in GOLDEN_REWARD_IDS:
            continue
        if exclude_card_id is not None and cid == exclude_card_id:
            continue
        if exact_tier is not None and rec.tier != exact_tier:
            continue
        if legendary_only:
            if rec.rarity != "LEGENDARY":
                continue
        if require_deathrattle and not _record_has_deathrattle(cid, rec.mechanics):
            continue
        if race_hs is not None and rec.race != race_hs:
            continue
        pool.append(cid)
    return tuple(pool)


__all__ = ["build_summon_pool", "hs_race_string"]

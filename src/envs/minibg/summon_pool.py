"""HS collectible minion pools for BG-style random deathrattle summons."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .effects import Keyword
from .state import Minion, Race

_HSJSON_PATH = Path(__file__).resolve().parents[3] / "data" / "minibg" / "cards_36393_raw.json"

_MECHANIC_TO_KW: dict[str, Keyword] = {
    "TAUNT": Keyword.TAUNT,
    "DIVINE_SHIELD": Keyword.SHIELD,
    "WINDFURY": Keyword.WINDFURY,
    "CHARGE": Keyword.CHARGE,
    "POISONOUS": Keyword.POISONOUS,
}


def _keywords_from_mechanics(mechanics: List[str], referenced: List[str]) -> frozenset[Keyword]:
    out: set[Keyword] = set()
    for tag in mechanics or []:
        k = _MECHANIC_TO_KW.get(tag)
        if k is not None:
            out.add(k)
    for tag in referenced or []:
        k = _MECHANIC_TO_KW.get(tag)
        if k is not None:
            out.add(k)
    return frozenset(out)


def _race_from_hs(value: Optional[str]) -> Optional[Race]:
    if value is None:
        return None
    mapping = {
        "BEAST": Race.BEAST,
        "DEMON": Race.DEMON,
        "MECHANICAL": Race.MECHANICAL,
        "MURLOC": Race.MURLOC,
        "ALL": Race.ALL,
        "PIRATE": None,
        "DRAGON": None,
        "ELEMENTAL": None,
        "QUILBOAR": None,
        "NAGA": None,
        "UNDEAD": None,
    }
    return mapping.get(value)


@lru_cache(maxsize=1)
def _collectible_minion_rows() -> tuple[Dict[str, Any], ...]:
    with _HSJSON_PATH.open(encoding="utf-8") as f:
        cards = json.load(f)
    out: List[Dict[str, Any]] = []
    for c in cards:
        if c.get("type") != "MINION":
            continue
        if not c.get("collectible"):
            continue
        cid = str(c.get("id", ""))
        if cid.startswith("TB_BaconUps_"):
            continue
        out.append(c)
    return tuple(out)


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


@lru_cache(maxsize=256)
def build_summon_pool(
    exact_cost: Optional[int],
    legendary_only: bool,
    require_deathrattle: bool,
    race_hs: Optional[str],
    exclude_card_id: Optional[str],
) -> tuple[Dict[str, Any], ...]:
    pool: List[Dict[str, Any]] = []
    for c in _collectible_minion_rows():
        if exact_cost is not None and c.get("cost") != exact_cost:
            continue
        if legendary_only and c.get("rarity") != "LEGENDARY":
            continue
        if require_deathrattle and "DEATHRATTLE" not in (c.get("mechanics") or []):
            continue
        if race_hs is not None and c.get("race") != race_hs:
            continue
        if exclude_card_id is not None and c.get("id") == exclude_card_id:
            continue
        pool.append(c)
    return tuple(pool)


def minion_from_hsjson_card(c: Dict[str, Any]) -> Minion:
    kws = _keywords_from_mechanics(
        list(c.get("mechanics") or []),
        list(c.get("referencedTags") or []),
    )
    race = _race_from_hs(c.get("race"))
    cost = int(c.get("cost") or 1)
    tier = max(1, min(cost, 6))
    atk = int(c.get("attack") or 0)
    hp = int(c.get("health") or 1)
    mid = str(c["id"])
    return Minion(
        card_id=mid,
        base_attack=atk,
        base_health=hp,
        tier=tier,
        name=str(c.get("name") or ""),
        race=race,
        keywords=kws,
        abilities=(),
        has_shield=Keyword.SHIELD in kws,
        is_token=False,
        dbf_id=int(c["dbfId"]) if c.get("dbfId") is not None else None,
    )


__all__ = [
    "build_summon_pool",
    "hs_race_string",
    "minion_from_hsjson_card",
]

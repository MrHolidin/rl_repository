from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

from .effects import Keyword

_CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "minibg" / "bg_patch_15_6_2_36393_catalog.json"

_MECHANIC_KEYWORDS: dict[str, Keyword] = {
    "TAUNT": Keyword.TAUNT,
    "DIVINE_SHIELD": Keyword.SHIELD,
    "WINDFURY": Keyword.WINDFURY,
    "CHARGE": Keyword.CHARGE,
    "POISONOUS": Keyword.POISONOUS,
}


@dataclass(frozen=True)
class TavernMinionRecord:
    dbf_id: int
    id: str
    name: str
    tier: int
    attack: int
    health: int
    cost: int
    race: Optional[str]
    set: str
    rarity: Optional[str]
    text: Optional[str]
    mechanics: FrozenSet[str]
    referenced_tags: FrozenSet[str]
    is_bacon_pool: bool
    is_golden: bool
    golden_dbf_id: Optional[int]

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "TavernMinionRecord":
        return cls(
            dbf_id=int(row["dbfId"]),
            id=str(row["id"]),
            name=str(row["name"]),
            tier=int(row["tier"]),
            attack=int(row["attack"])
            if row.get("attack") is not None
            else 0,
            health=int(row["health"])
            if row.get("health") is not None
            else 0,
            cost=int(row["cost"]) if row.get("cost") is not None else 0,
            race=row.get("race"),
            set=str(row.get("set", "")),
            rarity=row.get("rarity"),
            text=row.get("text"),
            mechanics=frozenset(row.get("mechanics") or []),
            referenced_tags=frozenset(row.get("referencedTags") or []),
            is_bacon_pool=bool(row.get("isBaconPoolMinion")),
            is_golden=bool(row.get("isGolden")),
            golden_dbf_id=(
                int(row["goldenDbfId"]) if row.get("goldenDbfId") is not None else None
            ),
        )


def catalog_path() -> Path:
    return _CATALOG_PATH


@lru_cache(maxsize=1)
def load_patch_catalog(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _CATALOG_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def patch_build() -> int:
    return int(load_patch_catalog()["build"])


def patch_version() -> str:
    return str(load_patch_catalog()["patch"])


def load_tavern_minions(path: Optional[Path] = None) -> List[TavernMinionRecord]:
    data = load_patch_catalog(path)
    return [TavernMinionRecord.from_row(r) for r in data["minions"]]


def tier_by_dbf_id(path: Optional[Path] = None) -> Dict[int, int]:
    return {m.dbf_id: m.tier for m in load_tavern_minions(path)}


def minion_by_dbf_id(path: Optional[Path] = None) -> Dict[int, TavernMinionRecord]:
    return {m.dbf_id: m for m in load_tavern_minions(path)}


def minion_by_id(path: Optional[Path] = None) -> Dict[str, TavernMinionRecord]:
    return {m.id: m for m in load_tavern_minions(path)}


def keywords_for_tavern_record(rec: TavernMinionRecord) -> FrozenSet[Keyword]:
    out: set[Keyword] = set()
    for tag in rec.mechanics:
        if tag == "MODULAR":
            out.add(Keyword.MAGNETIC)
            continue
        k = _MECHANIC_KEYWORDS.get(tag)
        if k is not None:
            out.add(k)
    for tag in rec.referenced_tags:
        k = _MECHANIC_KEYWORDS.get(tag)
        if k is not None:
            out.add(k)
    return frozenset(out)


def race_from_hs_string(value: Optional[str]):
    from .state import Race

    if value is None:
        return None
    mapping = {
        "BEAST": Race.BEAST,
        "DEMON": Race.DEMON,
        "MECHANICAL": Race.MECHANICAL,
        "MURLOC": Race.MURLOC,
        "ALL": Race.ALL,
    }
    if value not in mapping:
        raise KeyError(f"Unknown HS race {value!r}")
    return mapping[value]


def minion_from_tavern_record(rec: TavernMinionRecord):
    from .state import Minion

    kws = keywords_for_tavern_record(rec)
    race = race_from_hs_string(rec.race) if rec.race is not None else None
    return Minion(
        card_id=rec.id,
        base_attack=rec.attack,
        base_health=rec.health,
        tier=rec.tier,
        name=rec.name,
        race=race,
        keywords=kws,
        abilities=(),
        has_shield=Keyword.SHIELD in kws,
        is_token=False,
        is_golden=rec.is_golden,
        dbf_id=rec.dbf_id,
    )


__all__ = [
    "TavernMinionRecord",
    "catalog_path",
    "keywords_for_tavern_record",
    "load_patch_catalog",
    "load_tavern_minions",
    "minion_by_dbf_id",
    "minion_by_id",
    "minion_from_tavern_record",
    "patch_build",
    "patch_version",
    "race_from_hs_string",
    "tier_by_dbf_id",
]

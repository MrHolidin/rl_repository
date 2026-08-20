from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

from src.bg_core.effects import Keyword

# ``src/bg_catalog`` → repo root is two parents up.
_CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "bgcore"
    / "15_6_2_36393"
    / "catalog.json"
)

_MECHANIC_KEYWORDS: dict[str, Keyword] = {
    "TAUNT": Keyword.TAUNT,
    "DIVINE_SHIELD": Keyword.SHIELD,
    "WINDFURY": Keyword.WINDFURY,
    "CHARGE": Keyword.CHARGE,
    "POISONOUS": Keyword.POISONOUS,
    "REBORN": Keyword.REBORN,
    "VENOMOUS": Keyword.VENOMOUS,
    # The 2021 dumps tag Magnetic as MODULAR (special-cased in
    # keywords_for_tavern_record); modern builds print MAGNETIC on the card and
    # tag it that way too. Both names, one keyword.
    "MAGNETIC": Keyword.MAGNETIC,
}

# How each mechanic tag is printed in card text, for the self-grant check below.
_KEYWORD_TEXT_NAMES: dict[str, str] = {
    "TAUNT": "Taunt",
    "DIVINE_SHIELD": "Divine Shield",
    "WINDFURY": "Windfury",
    "CHARGE": "Charge",
    "POISONOUS": "Poisonous",
    "REBORN": "Reborn",
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


@dataclass(frozen=True)
class SpellRecord:
    """One row of the catalog's ``spells`` section.

    Its own record and not a :class:`TavernMinionRecord` with empty stats: a
    spell has a tier and a cost and nothing else a minion has, and the cost is
    load-bearing here in a way a minion's never is — a minion always costs the
    ruleset's buy price, a spell costs what is printed on it.

    Two flags, because the cards ask both questions and they are not the same
    one. ``is_tavern_spell`` is what "whenever you cast a Tavern spell" counts,
    and it is a property of the card; ``in_pool`` is whether the tavern ever
    offers it. Pointy Arrow is a Tavern spell that only ever arrives from a
    deathrattle, and a Blood Gem is neither.
    """

    dbf_id: int
    id: str
    name: str
    tier: int
    cost: int
    text: Optional[str]
    mechanics: FrozenSet[str]
    referenced_tags: FrozenSet[str]
    school: Optional[str]
    in_pool: bool

    @property
    def is_tavern_spell(self) -> bool:
        return self.school == "TAVERN"

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "SpellRecord":
        return cls(
            dbf_id=int(row["dbfId"]),
            id=str(row["id"]),
            name=str(row["name"]),
            tier=int(row.get("tier") or 0),
            cost=int(row.get("cost") or 0),
            text=row.get("text"),
            mechanics=frozenset(row.get("mechanics") or ()),
            referenced_tags=frozenset(row.get("referencedTags") or ()),
            school=row.get("spellSchool"),
            in_pool=bool(row.get("isPoolSpell")),
        )


def is_duos_only_card_id(card_id: str) -> bool:
    """Whether ``card_id`` belongs to the Duos-only set.

    Modern builds flag these as pool minions — they *are* pool minions, in the
    two-player mode — and their text says so ("the first time your team Passes",
    "your partner"). A solo lobby has no partner, so they are not this engine's
    cards. The set prefix is the whole rule: the 2021 packages predate Duos and
    contain no such id, so the filter is inert for them.
    """
    return card_id.startswith("BGDUO")


def load_spells(path: Optional[Path] = None) -> List[SpellRecord]:
    """Every spell the package carries, or nothing on a package with none.

    The 2021 packages predate Battlegrounds spells entirely and have no such
    section, which is not an error — it is what that patch was.
    """
    data = load_patch_catalog(path)
    return [SpellRecord.from_row(r) for r in data.get("spells", ())]


def tier_by_dbf_id(path: Optional[Path] = None) -> Dict[int, int]:
    return {m.dbf_id: m.tier for m in load_tavern_minions(path)}


def minion_by_dbf_id(path: Optional[Path] = None) -> Dict[int, TavernMinionRecord]:
    return {m.dbf_id: m for m in load_tavern_minions(path)}


@lru_cache(maxsize=1)
def normal_to_golden_card_id_map(path: Optional[Path] = None) -> Dict[str, str]:
    """Non-golden tavern ``card_id`` → catalog triple-upgrade golden ``card_id`` (usually ``TB_BaconUps_*``)."""

    by_dbf = minion_by_dbf_id(path)
    out: Dict[str, str] = {}
    for rec in load_tavern_minions(path):
        if rec.is_golden or rec.golden_dbf_id is None:
            continue
        g = by_dbf.get(rec.golden_dbf_id)
        if g is not None:
            out[rec.id] = g.id
    return out


def golden_upgrade_card_id(normal_card_id: str, path: Optional[Path] = None) -> Optional[str]:
    return normal_to_golden_card_id_map(path).get(normal_card_id)


def minion_by_id(path: Optional[Path] = None) -> Dict[str, TavernMinionRecord]:
    return {m.id: m for m in load_tavern_minions(path)}


def _text_has_mega_windfury(text: Optional[str]) -> bool:
    if not text:
        return False
    cleaned = re.sub(r"<[^>]+>", "", text).lower()
    return "mega-windfury" in cleaned


def _text_grants_keyword_to_self(text: Optional[str], tag: str) -> bool:
    """True when the card body opens with ``tag`` as a standalone keyword line.

    ``referencedTags`` says only that the card *mentions* a keyword, which is just
    as true of Houndmaster ("give a friendly Beast +2/+2 and Taunt") as of a card
    that actually has it. Taking the tag at face value handed Taunt to nine cards
    that hand it out, and Divine Shield to two that grant it on death. But the tag
    cannot simply be dropped either: hearthstonejson leaves ``mechanics`` empty for
    Zapp Slywick, Yo-Ho-Ogre and Seabreaker Goliath, whose keyword is real and only
    visible in the text. A keyword the minion owns is printed first, on its own,
    before any Battlecry/Deathrattle/trigger clause — that is what we match.
    """
    if not text:
        return False
    name = _KEYWORD_TEXT_NAMES.get(tag)
    if name is None:
        return False
    body = re.sub(r"\[x\]", "", text).replace("\n", " ").strip()
    return re.match(rf"^(<b>)?{name}(</b>)?\s*\.?(\s|$)", body, re.IGNORECASE) is not None


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
        if k is not None and _text_grants_keyword_to_self(rec.text, tag):
            out.add(k)
    if _text_has_mega_windfury(rec.text):
        out.add(Keyword.MEGA_WINDFURY)
    return frozenset(out)


def race_from_hs_string(value: Optional[str]):
    from src.bg_core.minion import Race

    if value is None:
        return None
    mapping = {
        "BEAST": Race.BEAST,
        "DEMON": Race.DEMON,
        "MECHANICAL": Race.MECHANICAL,
        "MURLOC": Race.MURLOC,
        "DRAGON": Race.DRAGON,
        "PIRATE": Race.PIRATE,
        "ELEMENTAL": Race.ELEMENTAL,
        "ALL": Race.ALL,
        # HSJSON spells these exactly like this on modern builds.
        "QUILBOAR": Race.QUILBOAR,
        "NAGA": Race.NAGA,
        "UNDEAD": Race.UNDEAD,
    }
    if value not in mapping:
        raise KeyError(f"Unknown HS race {value!r}")
    return mapping[value]


_SELL_FOR_GOLD_RE = re.compile(r"sells?\s+for\s+(\d+)\s+Gold", re.IGNORECASE)
#: "**If you lost your last combat**, this minion sells for 5 Gold" — a price
#: behind a condition, which a flat field cannot hold. Those come from a
#: ``SellValueEffect`` binding instead.
_CONDITIONAL_SELL_RE = re.compile(r"\bif\b[^.]*sells?\s+for\s+\d+\s+Gold", re.IGNORECASE)


def sell_value_from_catalog_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    if _CONDITIONAL_SELL_RE.search(text):
        return None
    m = _SELL_FOR_GOLD_RE.search(text)
    if m is None:
        return None
    return int(m.group(1))


def minion_from_tavern_record(rec: TavernMinionRecord):
    from src.bg_core.minion import Minion

    kws = keywords_for_tavern_record(rec)
    race = race_from_hs_string(rec.race) if rec.race is not None else None
    sell_value = sell_value_from_catalog_text(rec.text)
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
        sell_value=sell_value,
    )


__all__ = [
    "TavernMinionRecord",
    "catalog_path",
    "golden_upgrade_card_id",
    "keywords_for_tavern_record",
    "load_patch_catalog",
    "load_tavern_minions",
    "minion_by_dbf_id",
    "minion_by_id",
    "minion_from_tavern_record",
    "normal_to_golden_card_id_map",
    "patch_build",
    "patch_version",
    "race_from_hs_string",
    "sell_value_from_catalog_text",
    "tier_by_dbf_id",
]

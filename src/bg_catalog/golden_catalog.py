"""Catalog text helpers for triple-forged (golden) ability scaling."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from src.bg_core.effects import Keyword

from src.bg_catalog.patch_catalog import (
    golden_upgrade_card_id,
    keywords_for_tavern_record,
    minion_by_id,
)


def strip_catalog_text(text: Optional[str]) -> str:
    if not text:
        return ""
    out = re.sub(r"<[^>]+>", "", text)
    return out.replace("\xa0", " ").strip()


def catalog_golden_hints(normal_text: Optional[str], golden_text: Optional[str]) -> Dict[str, Any]:
    """Heuristics from HS card text: ``twice`` vs numeric doubling."""
    hints: Dict[str, Any] = {}
    if not normal_text or not golden_text:
        return hints

    n = strip_catalog_text(normal_text).lower()
    g = strip_catalog_text(golden_text).lower()

    if "twice" in g:
        hints["prefer_repeats"] = True
    if "triple" in g and "attack" in g:
        hints["triple_factor"] = True
    if "double" in g and "stats" in g:
        hints["double_stats"] = True

    if re.search(r"deal 1 damage to your hero", n) and re.search(
        r"deal 1 damage to your hero", g
    ):
        hints["preserve_hero_damage_amount"] = True

    n_deal = re.findall(r"deal (\d+) damage", n)
    g_deal = re.findall(r"deal (\d+) damage", g)
    if n_deal and g_deal and n_deal[0] == g_deal[0]:
        hints["preserve_hero_damage_amount"] = True

    return hints


def golden_hints_for_card(
    normal_card_id: str, catalog_path: Optional[Path] = None
) -> Dict[str, Any]:
    gid = golden_upgrade_card_id(normal_card_id, catalog_path)
    if gid is None:
        return {}
    by_id = minion_by_id(catalog_path)
    normal = by_id.get(normal_card_id)
    golden = by_id.get(gid)
    if normal is None or golden is None:
        return {}
    return catalog_golden_hints(normal.text, golden.text)


def forged_golden_keywords(
    normal_card_id: str,
    base_keywords: frozenset[Keyword],
    catalog_path: Optional[Path] = None,
) -> frozenset[Keyword]:
    """Static keywords on a forged golden (e.g. Windfury → Mega-Windfury)."""
    gid = golden_upgrade_card_id(normal_card_id, catalog_path)
    if gid is None:
        return base_keywords
    golden = minion_by_id(catalog_path).get(gid)
    if golden is None:
        return base_keywords
    golden_kws = keywords_for_tavern_record(golden)
    if Keyword.MEGA_WINDFURY in golden_kws:
        out = set(base_keywords) | {Keyword.MEGA_WINDFURY}
        for k in golden_kws:
            if k in (
                Keyword.SHIELD,
                Keyword.TAUNT,
                Keyword.POISONOUS,
                Keyword.REBORN,
                Keyword.CHARGE,
                Keyword.MAGNETIC,
            ):
                out.add(k)
        return frozenset(out)
    return base_keywords


__all__ = [
    "catalog_golden_hints",
    "forged_golden_keywords",
    "golden_hints_for_card",
    "strip_catalog_text",
]

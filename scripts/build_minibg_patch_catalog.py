#!/usr/bin/env python3
"""Build battlegrounds tavern-tier catalog for a pinned client build.

Reads:
  - HearthstoneJSON ``cards.json`` for locale strings, mechanics, stats, etc.
  - HearthSim ``CardDefs.xml`` at commit matching the patch for TECH_LEVEL,
    IS_BACON_POOL_MINION, and BACON triple-upgrade linkage (enumID 1429).

Example (after ``git -C ~/hsdata checkout e6cdbc5``)::

  python scripts/build_minibg_patch_catalog.py \\
    --card-defs ~/hsdata/CardDefs.xml \\
    --hsjson data/minibg/cards_36393_raw.json \\
    --out data/minibg/bg_patch_15_6_2_36393_catalog.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BACON_TRIPLE_ENUM_ID = "1429"


def _iter_entities(path: Path):
    for event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "Entity":
            yield elem
            elem.clear()


def parse_card_defs(path: Path) -> dict[int, dict]:
    """dbfId -> { card_id, tier, is_bacon_pool, golden_dbf_id }.

    Only entities carrying TECH_LEVEL (tavern tier) are returned.
    """
    out: dict[int, dict] = {}
    for elem in _iter_entities(path):
        card_id = elem.get("CardID")
        if not card_id:
            continue
        dbf_s = elem.get("ID")
        if dbf_s is None:
            continue
        dbf_id = int(dbf_s)
        tier = None
        is_bacon = False
        golden_dbf_id: int | None = None
        for child in elem:
            if child.tag != "Tag":
                continue
            name = child.get("name")
            eid = child.get("enumID")
            if name == "TECH_LEVEL":
                tier = int(child.get("value", 0))
            elif name == "IS_BACON_POOL_MINION":
                is_bacon = int(child.get("value", 0)) == 1
            elif eid == BACON_TRIPLE_ENUM_ID:
                v = child.get("value")
                if v is not None:
                    golden_dbf_id = int(v)
        if tier is None:
            continue
        out[dbf_id] = {
            "card_id": card_id,
            "tier": tier,
            "is_bacon_pool": is_bacon,
            "golden_dbf_id": golden_dbf_id,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--card-defs", type=Path, required=True)
    p.add_argument("--hsjson", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--build", type=int, default=36393)
    p.add_argument("--patch", type=str, default="15.6.2")
    args = p.parse_args()

    defs = parse_card_defs(args.card_defs)
    with args.hsjson.open(encoding="utf-8") as f:
        cards = json.load(f)
    by_dbf = {c["dbfId"]: c for c in cards if "dbfId" in c}

    missing_json = [d for d in defs if d not in by_dbf]
    if missing_json:
        raise SystemExit(f"{len(missing_json)} dbfIds in CardDefs but not in HSJSON")

    minions: list[dict] = []
    for dbf_id in sorted(defs):
        d = defs[dbf_id]
        c = by_dbf[dbf_id]
        if c.get("type") != "MINION":
            raise SystemExit(f"dbfId {dbf_id} ({d['card_id']}) is not MINION in HSJSON")
        card_id = c.get("id", d["card_id"])
        row = {
            "dbfId": dbf_id,
            "id": card_id,
            "name": c.get("name"),
            "tier": d["tier"],
            "attack": c.get("attack"),
            "health": c.get("health"),
            "cost": c.get("cost"),
            "race": c.get("race"),
            "set": c.get("set"),
            "rarity": c.get("rarity"),
            "text": c.get("text"),
            "mechanics": c.get("mechanics") or [],
            "referencedTags": c.get("referencedTags") or [],
            "isBaconPoolMinion": d["is_bacon_pool"],
            "isGolden": card_id.startswith("TB_BaconUps_"),
            "goldenDbfId": d["golden_dbf_id"],
        }
        minions.append(row)

    payload = {
        "build": args.build,
        "patch": args.patch,
        "locale": "enUS",
        "sources": {
            "hearthstonejson": f"https://api.hearthstonejson.com/v1/{args.build}/enUS/cards.json",
            "hsdata": "HearthSim/hsdata commit e6cdbc5 (Update to patch 15.6.2.36393) CardDefs.xml",
        },
        "tavernMinionCount": len(minions),
        "minions": minions,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(minions)} minions to {args.out}")


if __name__ == "__main__":
    main()

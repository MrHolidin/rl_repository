#!/usr/bin/env python3
"""Build battlegrounds tavern-tier catalog for a pinned client build.

Reads:
  - HearthstoneJSON ``cards.json`` for locale strings, mechanics, stats, etc.
    Default source: ``https://api.hearthstonejson.com/v1/{build}/{locale}/cards.json``
    (override with ``--hsjson`` for a local file or another HTTPS URL on the same host).
  - HearthSim ``CardDefs.xml`` at commit matching the patch for TECH_LEVEL,
    IS_BACON_POOL_MINION, and BACON triple-upgrade linkage (enumID 1429).

Examples::

  # Fetch card data for build 36393 from HearthstoneJSON (needs network)
  python scripts/build_minibg_patch_catalog.py \\
    --card-defs ~/hsdata/CardDefs.xml \\
    --build 36393 --patch 15.6.2 \\
    --out data/bgcore/15_6_2_36393/catalog.json

  # Offline: use a saved cards.json
  python scripts/build_minibg_patch_catalog.py \\
    --card-defs ~/hsdata/CardDefs.xml \\
    --hsjson data/minibg/cards_36393_raw.json \\
    --out data/bgcore/15_6_2_36393/catalog.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

_HSJSON_ALLOWED_NETLOC = "api.hearthstonejson.com"

BACON_TRIPLE_ENUM_ID = "1429"


def _iter_entities(path: Path):
    for event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "Entity":
            yield elem
            elem.clear()


def load_hsjson_cards(source: str) -> list:
    """Load HearthstoneJSON ``cards`` array from a local path or API URL.

    HTTPS URLs are restricted to ``api.hearthstonejson.com`` (returns 403 without
    a ``User-Agent``; we set a small project UA).
    """
    src = str(source).strip()
    if src.startswith(("http://", "https://")):
        from urllib.parse import urlparse

        p = urlparse(src)
        if p.netloc != _HSJSON_ALLOWED_NETLOC:
            raise SystemExit(
                f"Only {_HSJSON_ALLOWED_NETLOC!r} JSON URLs are allowed (got {p.netloc!r})"
            )
        req = urllib.request.Request(
            src,
            headers={"User-Agent": "RL-minibg-catalog/1.0 (build script)"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
    else:
        path = Path(src)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("HearthstoneJSON cards payload must be a JSON array")
    return data


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


def patch_package_dir(patch: str, build: int) -> Path:
    slug = patch.strip().replace(".", "_")
    return Path("data") / "bgcore" / f"{slug}_{build}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--card-defs", type=Path, required=True)
    p.add_argument(
        "--hsjson",
        type=str,
        default=None,
        help=(
            "Path to local cards.json, or "
            f"https://{_HSJSON_ALLOWED_NETLOC}/v1/BUILD/locale/cards.json . "
            "If omitted, cards are downloaded for --build and --locale."
        ),
    )
    p.add_argument(
        "--locale",
        type=str,
        default="enUS",
        help="Locale segment in api.hearthstonejson.com URL (default: enUS)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output catalog.json path (default: data/bgcore/{patch}_{build}/catalog.json)",
    )
    p.add_argument("--build", type=int, default=36393)
    p.add_argument("--patch", type=str, default="15.6.2")
    args = p.parse_args()
    if args.out is None:
        args.out = patch_package_dir(args.patch, args.build) / "catalog.json"

    defs = parse_card_defs(args.card_defs)
    hs_src = args.hsjson
    if hs_src is None:
        hs_src = f"https://{_HSJSON_ALLOWED_NETLOC}/v1/{args.build}/{args.locale}/cards.json"
    cards = load_hsjson_cards(hs_src)
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
        "locale": args.locale,
        "sources": {
            "hearthstonejson": hs_src
            if str(hs_src).startswith("http")
            else str(Path(hs_src).resolve()),
            "hsdata": "HearthSim/hsdata CardDefs.xml (commit must match this client build)",
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

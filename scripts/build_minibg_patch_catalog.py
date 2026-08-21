#!/usr/bin/env python3
"""Build battlegrounds tavern-tier catalog for a pinned client build.

Reads:
  - HearthstoneJSON ``cards.json`` for locale strings, mechanics, stats, etc.
    Default source: ``https://api.hearthstonejson.com/v1/{build}/{locale}/cards.json``
    (override with ``--hsjson`` for a local file or another HTTPS URL on the same host).
  - HearthSim ``CardDefs.xml`` at commit matching the patch for TECH_LEVEL,
    IS_BACON_POOL_MINION, and BACON triple-upgrade linkage (enumID 1429).
    **Optional.** Modern HearthstoneJSON carries all three itself as
    ``techLevel`` / ``isBattlegroundsPoolMinion`` / ``battlegroundsPremiumDbfId``,
    so a build from 2022 on needs no XML at all; ``--card-defs`` stays for the
    older builds whose JSON predates those fields.

A catalog also carries the non-minion cards a modern build offers, each in its
own section: tavern spells, trinkets, heroes and Dark Gifts. They are listed
whether or not the engine can bind them yet — a package that omits them cannot
even be checked for coverage, and what is missing is the point of the file.

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


def defs_from_hsjson(cards: list) -> dict[int, dict]:
    """The same table ``parse_card_defs`` builds, read off HearthstoneJSON.

    Modern JSON exposes the three tags the XML was needed for. ``techLevel`` is
    the tavern tier, and carrying one is what makes a card a tavern minion — the
    same rule the XML path applies.
    """
    out: dict[int, dict] = {}
    for c in cards:
        if c.get("type") != "MINION" or c.get("techLevel") is None:
            continue
        dbf_id = c.get("dbfId")
        card_id = c.get("id")
        if dbf_id is None or not card_id:
            continue
        out[int(dbf_id)] = {
            "card_id": card_id,
            "tier": int(c["techLevel"]),
            "is_bacon_pool": bool(c.get("isBattlegroundsPoolMinion")),
            "golden_dbf_id": c.get("battlegroundsPremiumDbfId"),
        }
    return out


def _row(c: dict, **extra) -> dict:
    """Shared card row: identity, text and tags, plus per-section fields."""
    row = {
        "dbfId": c.get("dbfId"),
        "id": c.get("id"),
        "name": c.get("name"),
        "cost": c.get("cost"),
        "set": c.get("set"),
        "text": c.get("text"),
        "mechanics": c.get("mechanics") or [],
        "referencedTags": c.get("referencedTags") or [],
    }
    row.update(extra)
    return row


def collect_spells(cards: list) -> list[dict]:
    """Every Battlegrounds spell, the way ``minions`` carries every minion.

    Three kinds live here and the flags tell them apart, because the cards do:

    * **offered on the counter** — ``isPoolSpell``, the 75 a seat can buy;
    * **Tavern spells that are handed out rather than sold** — school TAVERN
      without the pool flag (Pointy Arrow arrives from a deathrattle). They
      still count for "whenever you cast a Tavern spell";
    * **plain spells** — a Blood Gem, a Spellcraft spell, Slimy Shield, Gem Day.
      No school, never in the tavern, and not a Tavern spell to any listener.

    Emitting only the pool ones is what left three tier-2 cards unbindable: a
    binding can only name a card the catalog carries, and the tokens these
    cards hand over were not in it.
    """
    out = [
        _row(
            c,
            tier=c.get("techLevel"),
            spellSchool=c.get("spellSchool"),
            isPoolSpell=bool(c.get("isBattlegroundsPoolSpell")),
        )
        for c in cards
        if c.get("set") == "BATTLEGROUNDS"
        and c.get("type") in ("SPELL", "BATTLEGROUND_SPELL")
    ]
    return sorted(out, key=lambda r: (r["tier"] or 0, r["id"] or ""))


def collect_trinkets(cards: list) -> list[dict]:
    """Trinkets are a card *type*, not a flag — there is no isBattlegrounds
    equivalent for them the way there is for pool minions and spells."""
    out = [_row(c) for c in cards if c.get("type") == "BATTLEGROUND_TRINKET"]
    return sorted(out, key=lambda r: r["id"] or "")


def collect_heroes(cards: list) -> list[dict]:
    """Every Battlegrounds hero, carrying the power it plays with.

    A hero card is a portrait: a name, a health pool and some armor, and no
    rules text at all. What the hero *does* is a second card, which the hero
    points at by ``heroPowerDbfId`` — so a heroes section without it can name
    121 heroes and describe none of them.

    ``power_passive`` is the client's own ``hideCost``: a power the seat cannot
    click has no cost to show. It is the split that matters most to an engine,
    because a passive needs no action and an active one needs somewhere in the
    action space to be pressed.
    """
    by_dbf = {c.get("dbfId"): c for c in cards if c.get("dbfId") is not None}
    out = []
    for c in cards:
        if not c.get("battlegroundsHero"):
            continue
        power = by_dbf.get(c.get("heroPowerDbfId")) or {}
        out.append(
            _row(
                c,
                armor=c.get("armor"),
                health=c.get("health"),
                powerId=power.get("id"),
                powerName=power.get("name"),
                powerText=power.get("text"),
                powerCost=power.get("cost"),
                powerMechanics=power.get("mechanics") or [],
                powerPassive=bool(power.get("hideCost")),
            )
        )
    return sorted(out, key=lambda r: r["id"] or "")


def collect_dark_gifts(cards: list) -> list[dict]:
    out = [_row(c) for c in cards if c.get("isBattlegroundsDarkGift")]
    return sorted(out, key=lambda r: r["id"] or "")


def patch_package_dir(patch: str, build: int) -> Path:
    slug = patch.strip().replace(".", "_")
    return Path("data") / "bgcore" / f"{slug}_{build}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--card-defs",
        type=Path,
        default=None,
        help=(
            "HearthSim CardDefs.xml. Only needed for builds whose HearthstoneJSON "
            "predates techLevel / isBattlegroundsPoolMinion."
        ),
    )
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

    hs_src = args.hsjson
    if hs_src is None:
        hs_src = f"https://{_HSJSON_ALLOWED_NETLOC}/v1/{args.build}/{args.locale}/cards.json"
    cards = load_hsjson_cards(hs_src)
    by_dbf = {c["dbfId"]: c for c in cards if "dbfId" in c}

    if args.card_defs is not None:
        defs = parse_card_defs(args.card_defs)
        missing_json = [d for d in defs if d not in by_dbf]
        if missing_json:
            raise SystemExit(f"{len(missing_json)} dbfIds in CardDefs but not in HSJSON")
    else:
        defs = defs_from_hsjson(cards)
        if not defs:
            raise SystemExit(
                "no tavern minions found in HearthstoneJSON (no techLevel field): "
                "this build needs --card-defs"
            )

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

    # Non-minion cards, each kind in its own section. Empty on old builds,
    # which is itself the honest answer: 15.6.2 had none of them.
    for key, rows in (
        ("spells", collect_spells(cards)),
        ("trinkets", collect_trinkets(cards)),
        ("heroes", collect_heroes(cards)),
        ("darkGifts", collect_dark_gifts(cards)),
    ):
        if rows:
            payload[key] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    counts = ", ".join(
        f"{len(payload[k])} {k}"
        for k in ("spells", "trinkets", "heroes", "darkGifts")
        if k in payload
    )
    print(f"Wrote {len(minions)} minions{', ' + counts if counts else ''} to {args.out}")


if __name__ == "__main__":
    main()

"""Every card's triple-forged golden: into a fight, and against its own text.

Goldens are the least-tested surface on the modern package because they are
*derived* rather than authored -- only 104 of the 246 pool cards ship a
``TB_BaconUps_*`` row, so the rest get their abilities from
``triple_merge_golden_abilities`` scaling the normal card's numbers. Two things
can go wrong with that and neither shows up in a normal game: the scaled effect
can reach a dispatcher that has no branch for its shape, and the scaling can
land on the wrong number.

So this does both. ``--mode battle`` puts each golden on a board and fights it
(the combat fuzzer only ever sets ``is_golden = True`` on a *normal* body, so
golden abilities have never been through a battle). ``--mode text`` compares the
derived abilities against the numbers printed on the golden's own card text in
catalog.json.

    python scripts/fuzz_bglike_goldens.py --mode battle [--patch <dir>] [--reps 6]
    python scripts/fuzz_bglike_goldens.py --mode text  [--patch <dir>]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import traceback
from collections import Counter
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import src.envs  # noqa: F401  (import for side effect: module init order)
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion
from src.bg_combat.battle import simulate_battle
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment.combat_seat import PlayerCombatSeat
from src.bg_recruitment.triples import make_forged_golden_minion

BOARD_SIZE = 7


def _seat(patch: PatchContext, board: List[Minion]) -> PlayerState:
    return PlayerState(
        health=30,
        gold=10,
        tavern_tier=6,
        board=list(board),
        shop=[None] * 7,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )


def _filler(patch: PatchContext, rng: random.Random, n: int) -> List[Minion]:
    pool = sorted(patch.pool_ids)
    out = []
    for _ in range(n):
        m = patch.make_minion(rng.choice(pool))
        m.bonus_attack += rng.randint(0, 4)
        m.bonus_health += rng.randint(0, 4)
        out.append(m)
    return out


def battle_with_golden(patch: PatchContext, card_id: str, seed: int) -> None:
    rng = random.Random(seed)
    golden = make_forged_golden_minion(card_id, patch=patch)
    mine = _filler(patch, rng, rng.randint(0, BOARD_SIZE - 1))
    mine.insert(rng.randint(0, len(mine)), golden)
    theirs = _filler(patch, rng, rng.randint(1, BOARD_SIZE))
    seats = [_seat(patch, mine), _seat(patch, theirs)]
    for seat in seats:
        for slot in range(rng.randint(0, 3)):
            seat.hand[slot] = patch.make_minion(rng.choice(sorted(patch.pool_ids)))
    survivors: List[Minion] = []
    deaths: List[Tuple[int, str]] = []
    hand_adds: List[List[str]] = [[], []]
    simulate_battle(
        mine,
        theirs,
        p0_has_initiative=bool(rng.getrandbits(1)),
        rng=np.random.default_rng(seed),
        patch=patch,
        combat_board_max=BOARD_SIZE,
        damage_cap=15,
        max_board_slots=BOARD_SIZE,
        p0_board_out=survivors,
        death_log=deaths,
        combat_hand_adds_out=hand_adds,
        seats=(
            PlayerCombatSeat(seats[0], patch=patch),
            PlayerCombatSeat(seats[1], patch=patch),
        ),
    )


# ------------------------------------------------------------- text check

def _numbers(text: Optional[str]) -> List[int]:
    if not text:
        return []
    clean = re.sub(r"<[^>]+>", " ", text)
    return [int(x) for x in re.findall(r"\d+", clean)]


def _effect_numbers(obj: Any, out: Optional[List[int]] = None) -> List[int]:
    if out is None:
        out = []
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            _effect_numbers(getattr(obj, f.name), out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, int):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _effect_numbers(x, out)
    return out


def text_check(patch: PatchContext, patch_dir: Path) -> int:
    cat = json.loads((patch_dir / "catalog.json").read_text())
    by_dbf = {m["dbfId"]: m for m in cat["minions"]}
    by_id = {m["id"]: m for m in cat["minions"]}

    stat_bad: List[str] = []
    scale_missing: List[str] = []
    scale_unexpected: List[str] = []
    authored = 0
    derived = 0

    for card_id in sorted(patch.pool_ids):
        tpl = patch.templates[card_id]
        row = by_id.get(card_id)
        if row is None:
            continue
        gid = row.get("goldenDbfId")
        grow = by_dbf.get(gid) if gid else None
        if grow is None:
            continue
        forged = make_forged_golden_minion(card_id, patch=patch)
        # 1. printed stats
        if (forged.base_attack, forged.base_health) != (grow["attack"], grow["health"]):
            stat_bad.append(
                f"{card_id} {tpl.name}: forged {forged.base_attack}/{forged.base_health} "
                f"vs printed _G {grow['attack']}/{grow['health']}"
            )
        # 2. effect numbers vs the golden's own text
        if grow["id"] in patch.effects:
            authored += 1
            continue
        derived += 1
        n_norm = _numbers(row.get("text"))
        n_gold = _numbers(grow.get("text"))
        eff_norm = _effect_numbers(list(tpl.abilities))
        eff_gold = _effect_numbers(list(forged.abilities))
        text_scaled = n_norm != n_gold
        eff_scaled = eff_norm != eff_gold
        if text_scaled and not eff_scaled and eff_norm:
            scale_missing.append(
                f"{card_id} {tpl.name}: _G text {n_gold} vs normal {n_norm}, "
                f"but abilities unchanged {eff_norm}"
            )
        elif (not text_scaled) and eff_scaled and n_norm:
            scale_unexpected.append(
                f"{card_id} {tpl.name}: _G text same numbers {n_gold}, "
                f"but abilities {eff_norm} -> {eff_gold}"
            )

    print(f"{authored} goldens authored as TB_BaconUps rows, {derived} derived by scaling")
    for title, rows in (
        ("golden printed stats != 2x normal", stat_bad),
        ("golden text scales a number the derived abilities do not", scale_missing),
        ("derived abilities scale a number the golden text does not", scale_unexpected),
    ):
        print(f"\n{title}: {len(rows)}")
        for r in rows:
            print(f"  {r}")
    return 1 if (stat_bad or scale_missing or scale_unexpected) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="data/bgcore/36_2_0_248348")
    ap.add_argument("--mode", choices=["battle", "text"], default="battle")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traceback", action="store_true")
    ap.add_argument("--max-report", type=int, default=30)
    args = ap.parse_args()

    patch_dir = Path(args.patch)
    patch = PatchContext.load(patch_dir)
    if args.mode == "text":
        return text_check(patch, patch_dir)

    failures: Counter = Counter()
    first: Dict[str, Tuple[str, int, str]] = {}
    built = 0
    fought = 0
    for card_id in sorted(patch.pool_ids):
        try:
            make_forged_golden_minion(card_id, patch=patch)
            built += 1
        except Exception as exc:  # noqa: BLE001
            key = f"build_golden: {type(exc).__name__}: {str(exc)[:160]}"
            failures[key] += 1
            first.setdefault(key, (card_id, -1, traceback.format_exc()))
            continue
        for r in range(args.reps):
            seed = args.seed + r
            try:
                battle_with_golden(patch, card_id, seed)
                fought += 1
            except Exception as exc:  # noqa: BLE001
                key = f"battle: {type(exc).__name__}: {str(exc)[:160]}"
                failures[key] += 1
                first.setdefault(key, (card_id, seed, traceback.format_exc()))

    print(f"{built}/{len(patch.pool_ids)} goldens built, {fought} battles run")
    if not failures:
        print("no crashes")
        return 0
    for key, count in failures.most_common(args.max_report):
        card, seed, tb = first[key]
        print(f"\n{count:>4}x  (first card {card}, seed {seed})\n      {key}")
        if args.traceback:
            print(tb)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

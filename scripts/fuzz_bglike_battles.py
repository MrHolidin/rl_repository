"""Fire random battles at the combat engine and report what breaks.

Every dispatcher in the engine is deliberately loud — an effect nobody handles
raises rather than doing nothing quietly — which makes a fuzz run a real test
of coverage: a card whose ability has no handler takes the run down the first
time it is drawn into a fight. Boards are drawn from the package's own pool, so
what gets exercised is what a game would actually deal.

    python scripts/fuzz_bglike_battles.py --battles 200 [--patch <dir>] [--seed N]
"""

from __future__ import annotations

import argparse
import random
import traceback
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np

# The env package is the entry point the rest of the engine expects to be
# imported through; reaching for bg_combat first hits a partially initialised
# module. Same reason every test module starts from it.
import src.envs  # noqa: F401  (import for side effect: module init order)
from src.bg_catalog.patch_context import PatchContext
from src.bg_combat.battle import simulate_battle
from src.bg_core.minion import Minion
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment.combat_seat import PlayerCombatSeat

BOARD_SIZE = 7


def _seat(patch: PatchContext, board: List[Minion]) -> PlayerState:
    """A seat real enough for the effects that write to one."""
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


def _random_board(patch: PatchContext, rng: random.Random) -> List[Minion]:
    pool = sorted(patch.pool_ids)
    board: List[Minion] = []
    for _ in range(rng.randint(1, BOARD_SIZE)):
        minion = patch.make_minion(rng.choice(pool))
        # Bodies in a real fight are rarely printed-stat: they have been bought
        # into, buffed, and sometimes tripled.
        minion.bonus_attack += rng.randint(0, 8)
        minion.bonus_health += rng.randint(0, 8)
        if rng.random() < 0.12:
            minion.is_golden = True
        board.append(minion)
    return board


def _one_battle(patch: PatchContext, seed: int) -> None:
    rng = random.Random(seed)
    boards = [_random_board(patch, rng), _random_board(patch, rng)]
    seats = [_seat(patch, b) for b in boards]
    # Hands hold cards too: the effects that summon or fetch from one need it.
    for seat in seats:
        for slot in range(rng.randint(0, 3)):
            seat.hand[slot] = patch.make_minion(rng.choice(sorted(patch.pool_ids)))
    survivors: List[Minion] = []
    deaths: List[Tuple[int, str]] = []
    combat_hand_adds: List[List[str]] = [[], []]
    simulate_battle(
        boards[0],
        boards[1],
        p0_has_initiative=bool(rng.getrandbits(1)),
        rng=np.random.default_rng(seed),
        patch=patch,
        combat_board_max=BOARD_SIZE,
        damage_cap=15,
        max_board_slots=BOARD_SIZE,
        p0_board_out=survivors,
        death_log=deaths,
        combat_hand_adds_out=combat_hand_adds,
        seats=(
            PlayerCombatSeat(seats[0], patch=patch),
            PlayerCombatSeat(seats[1], patch=patch),
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=200)
    ap.add_argument("--patch", default="data/bgcore/36_2_0_248348")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traceback", action="store_true", help="print the first one")
    args = ap.parse_args()

    patch = PatchContext.load(Path(args.patch))
    failures: Counter = Counter()
    first: dict = {}
    for i in range(args.battles):
        seed = args.seed + i
        try:
            _one_battle(patch, seed)
        except Exception as exc:  # noqa: BLE001 — the point is to catch everything
            key = f"{type(exc).__name__}: {exc}"
            failures[key] += 1
            first.setdefault(key, (seed, traceback.format_exc()))

    ran = args.battles
    print(f"{ran - sum(failures.values())}/{ran} battles completed")
    if not failures:
        print("no crashes")
        return 0
    for key, count in failures.most_common():
        seed, tb = first[key]
        print(f"\n{count:>4}x  (first at seed {seed})\n      {key}")
        if args.traceback:
            print(tb)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

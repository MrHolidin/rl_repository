"""Play whole 8-player lobbies with random legal actions and report what breaks.

The combat fuzzer (``fuzz_bglike_battles.py``) only ever builds a board and
fires it at ``simulate_battle``. This one drives the *other* half: the shop
turn loop, buys/sells/rolls/freezes, plays, magnetise, targeted battlecries,
Discover modals, and the between-round combat -- every seat picked uniformly
from the legal mask, so the run is a coverage test of the recruit-phase
dispatchers the same way the battle fuzzer is one of the combat dispatchers.

    python scripts/fuzz_bglike_games.py --games 20 [--patch <dir>] [--seed N]
"""

from __future__ import annotations

import argparse
import random
import traceback
from collections import Counter
from typing import List

import numpy as np

import src.envs  # noqa: F401  (import for side effect: module init order)
from src.agents.random_agent import RandomAgent
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import SeatConfig, SeatKind
NUM_PLAYERS = 8
MAX_STEPS = 20_000


def _env(seed: int, patch_dir: str, *, heroes: bool, high_mode: bool) -> BGLobbyEnv:
    configs = tuple(
        SeatConfig(SeatKind.LEARNED, RandomAgent(seed=seed * 100 + s))
        for s in range(NUM_PLAYERS)
    )
    return BGLobbyEnv(
        configs,
        learned_seats=tuple(range(NUM_PLAYERS)),
        seed=seed,
        patch_dir=patch_dir,
        with_heroes=heroes,
        high_mode=high_mode,
    )


def _one_game(
    seed: int,
    patch_dir: str,
    *,
    heroes: bool,
    high_mode: bool,
    stats: Counter,
) -> None:
    rng = random.Random(seed)
    env = _env(seed, patch_dir, heroes=heroes, high_mode=high_mode)
    env.reset(seed=seed)
    steps = 0
    while not env.lobby_done and steps < MAX_STEPS:
        seat = env.current_seat()
        if not env._seat_can_act(seat):
            raise RuntimeError(
                f"stall: seat {seat} cannot act, lobby not done "
                f"(round={env.state.round_number}, phase={env.state.players[seat].phase.name})"
            )
        mask = env.legal_mask_for_seat(seat)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError(f"empty legal mask for acting seat {seat}")
        action = int(rng.choice(list(legal)))
        stats[f"action_{action}"] += 1
        env.step_action(seat, action)
        steps += 1
    stats["_steps"] += steps
    if not env.lobby_done:
        raise RuntimeError(f"game did not finish in {MAX_STEPS} steps")
    stats["_games"] += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--patch", default="data/bgcore/36_2_0_248348")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heroes", action="store_true")
    ap.add_argument("--high-mode", action="store_true")
    ap.add_argument("--traceback", action="store_true")
    ap.add_argument("--max-report", type=int, default=12)
    args = ap.parse_args()

    failures: Counter = Counter()
    first: dict = {}
    stats: Counter = Counter()
    for i in range(args.games):
        seed = args.seed + i
        try:
            _one_game(
                seed,
                args.patch,
                heroes=args.heroes,
                high_mode=args.high_mode,
                stats=stats,
            )
        except Exception as exc:  # noqa: BLE001 -- the point is to catch everything
            tb = traceback.format_exc()
            key = f"{type(exc).__name__}: {str(exc)[:200]}"
            failures[key] += 1
            first.setdefault(key, (seed, tb))

    ok = args.games - sum(failures.values())
    print(
        f"{ok}/{args.games} lobbies completed "
        f"({stats['_steps']} shop actions over {stats['_games']} finished games)"
    )
    for k, v in sorted(stats.items()):
        if k.startswith("known_"):
            print(f"  stepped over {v}x {k}")
    if not failures:
        print("no crashes")
        return 0
    for key, count in failures.most_common(args.max_report):
        seed, tb = first[key]
        print(f"\n{count:>4}x  (first at seed {seed})\n      {key}")
        if args.traceback:
            print(tb)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

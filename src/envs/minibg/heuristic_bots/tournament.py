from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..env import MiniBGEnv
from .bots import HeuristicBot, default_bot_constructors

GameOutcome = str  # "bot0" | "bot1" | "draw" | "timeout"


def make_bot(name: str, run_seed: int) -> HeuristicBot:
    ctors = default_bot_constructors()
    if name not in ctors:
        raise KeyError(f"Unknown bot {name!r}; known: {sorted(ctors)}")
    cls = ctors[name]
    return cls(seed=run_seed)  # type: ignore[misc]


def play_game(
    bot0: HeuristicBot,
    bot1: HeuristicBot,
    seed: int,
    max_steps: int = 20_000,
    *,
    timeout_sec: Optional[float] = None,
) -> GameOutcome:
    """Play one game. ``seed`` fixes env RNG (reproducible). ``timeout_sec`` caps wall time."""
    env = MiniBGEnv(seed=seed, patch_dir="data/bgcore/15_6_2_36393")
    deadline = (
        time.perf_counter() + timeout_sec if timeout_sec is not None else None
    )
    steps = 0
    while not env.done and steps < max_steps:
        if deadline is not None and time.perf_counter() >= deadline:
            return "timeout"
        idx = env.current_player()
        bot = bot0 if idx == 0 else bot1
        action = bot.choose_action(env)
        env.step(action)
        steps += 1
    if not env.done:
        return "draw"
    w = env.winner
    if w == 1:
        return "bot0"
    if w == -1:
        return "bot1"
    return "draw"


def play_pair_symmetric(
    name_a: str,
    name_b: str,
    games: int,
    base_seed: int,
    *,
    timeout_sec: Optional[float] = None,
) -> Tuple[int, int, int, int]:
    """Return (wins_a, wins_b, draws, timeouts) over `games` seeds × 2 seatings each."""
    a_wins = b_wins = draws = timeouts = 0
    for g in range(games):
        s = base_seed + g * 1009 + 13
        r1 = play_game(
            make_bot(name_a, s + 1),
            make_bot(name_b, s + 2),
            s,
            timeout_sec=timeout_sec,
        )
        if r1 == "bot0":
            a_wins += 1
        elif r1 == "bot1":
            b_wins += 1
        elif r1 == "timeout":
            draws += 1
            timeouts += 1
        else:
            draws += 1
        r2 = play_game(
            make_bot(name_b, s + 101),
            make_bot(name_a, s + 102),
            s + 100,
            timeout_sec=timeout_sec,
        )
        if r2 == "bot0":
            b_wins += 1
        elif r2 == "bot1":
            a_wins += 1
        elif r2 == "timeout":
            draws += 1
            timeouts += 1
        else:
            draws += 1
    return a_wins, b_wins, draws, timeouts


def run_tournament(
    names: Sequence[str],
    games_per_pair: int,
    base_seed: int = 0,
    *,
    timeout_sec: Optional[float] = None,
) -> Dict[Tuple[str, str], Tuple[int, int, int, int]]:
    out: Dict[Tuple[str, str], Tuple[int, int, int, int]] = {}
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i >= j:
                continue
            seed = base_seed + i * 10_007 + j * 9001
            out[(na, nb)] = play_pair_symmetric(
                na, nb, games_per_pair, seed, timeout_sec=timeout_sec
            )
    return out


def print_results(
    names: Sequence[str],
    results: Dict[Tuple[str, str], Tuple[int, int, int, int]],
) -> None:
    print(
        "Pairwise (wins A / wins B / draws), each entry = 2*games_per_pair games "
        "(seat swap). Timeouts count as draws.\n"
    )
    cell_w = max(12, max(len(n) for n in names) + 2)
    header = f"{'':<{cell_w}}" + "".join(f"{nb:>{cell_w}}" for nb in names)
    print(header)
    for na in names:
        row = f"{na:<{cell_w}}"
        for nb in names:
            if na == nb:
                row += f"{'—':>{cell_w}}"
            elif (na, nb) in results:
                wa, wb, d, _to = results[(na, nb)]
                row += f"{wa}-{wb}-{d}".rjust(cell_w)
            else:
                wa, wb, d, _to = results[(nb, na)]
                row += f"{wb}-{wa}-{d}".rjust(cell_w)
        print(row)

    totals = {n: 0 for n in names}
    played = {n: 0 for n in names}
    total_timeouts = 0
    for (na, nb), (wa, wb, d, to) in results.items():
        totals[na] += wa
        totals[nb] += wb
        nplays = wa + wb + d
        played[na] += nplays
        played[nb] += nplays
        total_timeouts += to
    print("\nWin rate (wins / games):")
    for n in names:
        if played[n]:
            print(f"  {n}: {totals[n]}/{played[n]} = {100.0 * totals[n] / played[n]:.1f}%")
        else:
            print(f"  {n}: n/a")
    if total_timeouts:
        print(f"\nTimeouts (counted as draws): {total_timeouts}")


def main() -> None:
    p = argparse.ArgumentParser(description="Mini BG heuristic bot tournament")
    p.add_argument(
        "--games",
        type=int,
        default=15,
        help="Seeds per ordered pair (each seed = 2 games with seat swap)",
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Wall-clock limit per game (unfinished → timeout/draw)",
    )
    p.add_argument(
        "--bots",
        type=str,
        default="",
        help="Comma-separated bot names (default: all)",
    )
    args = p.parse_args()
    all_names = sorted(default_bot_constructors().keys())
    names = [x.strip() for x in args.bots.split(",") if x.strip()] if args.bots else all_names
    for n in names:
        if n not in all_names:
            raise SystemExit(f"Unknown bot {n!r}. Choices: {all_names}")
    res = run_tournament(
        names, args.games, args.seed, timeout_sec=args.timeout
    )
    if args.timeout is not None:
        print(f"game_timeout_sec={args.timeout} base_seed={args.seed} games_per_pair={args.games}")
    print_results(names, res)


if __name__ == "__main__":
    main()

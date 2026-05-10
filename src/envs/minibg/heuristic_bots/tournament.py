from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..env import MiniBGEnv
from .bots import HeuristicBot, default_bot_constructors
from .common import legal_env_indices


def make_bot(name: str, run_seed: int) -> HeuristicBot:
    ctors = default_bot_constructors()
    if name not in ctors:
        raise KeyError(f"Unknown bot {name!r}; known: {sorted(ctors)}")
    cls = ctors[name]
    if name == "random":
        return cls(seed=run_seed)  # type: ignore[misc]
    return cls()


def play_game(
    bot0: HeuristicBot,
    bot1: HeuristicBot,
    seed: int,
    max_steps: int = 20_000,
) -> str:
    env = MiniBGEnv(seed=seed)
    rng = np.random.default_rng(seed + 17)
    steps = 0
    while not env.done and steps < max_steps:
        idx = env.current_player()
        bot = bot0 if idx == 0 else bot1
        mask = env.legal_actions_mask
        action = bot.choose_action(env)
        if not bool(mask[action]):
            legal = legal_env_indices(mask)
            if not legal:
                break
            action = int(rng.choice(legal))
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
) -> Tuple[int, int, int]:
    """Return (wins_a, wins_b, draws) over `games` seeds × 2 seatings each."""
    a_wins = b_wins = draws = 0
    for g in range(games):
        s = base_seed + g * 1009 + 13
        r1 = play_game(make_bot(name_a, s + 1), make_bot(name_b, s + 2), s)
        if r1 == "bot0":
            a_wins += 1
        elif r1 == "bot1":
            b_wins += 1
        else:
            draws += 1
        r2 = play_game(make_bot(name_b, s + 101), make_bot(name_a, s + 102), s + 100)
        if r2 == "bot0":
            b_wins += 1
        elif r2 == "bot1":
            a_wins += 1
        else:
            draws += 1
    return a_wins, b_wins, draws


def run_tournament(
    names: Sequence[str],
    games_per_pair: int,
    base_seed: int = 0,
) -> Dict[Tuple[str, str], Tuple[int, int, int]]:
    out: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i >= j:
                continue
            seed = base_seed + i * 10_007 + j * 9001
            out[(na, nb)] = play_pair_symmetric(na, nb, games_per_pair, seed)
    return out


def print_results(
    names: Sequence[str],
    results: Dict[Tuple[str, str], Tuple[int, int, int]],
) -> None:
    print("Pairwise (wins A / wins B / draws), each entry = 2*games_per_pair games (seat swap).\n")
    cell_w = max(12, max(len(n) for n in names) + 2)
    header = f"{'':<{cell_w}}" + "".join(f"{nb:>{cell_w}}" for nb in names)
    print(header)
    for na in names:
        row = f"{na:<{cell_w}}"
        for nb in names:
            if na == nb:
                row += f"{'—':>{cell_w}}"
            elif (na, nb) in results:
                wa, wb, d = results[(na, nb)]
                row += f"{wa}-{wb}-{d}".rjust(cell_w)
            else:
                wa, wb, d = results[(nb, na)]
                row += f"{wb}-{wa}-{d}".rjust(cell_w)
        print(row)

    totals = {n: 0 for n in names}
    played = {n: 0 for n in names}
    for (na, nb), (wa, wb, d) in results.items():
        totals[na] += wa
        totals[nb] += wb
        nplays = wa + wb + d
        played[na] += nplays
        played[nb] += nplays
    print("\nWin rate (wins / games):")
    for n in names:
        if played[n]:
            print(f"  {n}: {totals[n]}/{played[n]} = {100.0 * totals[n] / played[n]:.1f}%")
        else:
            print(f"  {n}: n/a")


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
    res = run_tournament(names, args.games, args.seed)
    print_results(names, res)


if __name__ == "__main__":
    main()

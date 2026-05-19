#!/usr/bin/env python3
"""Round-robin MiniBG checkpoint tournament (symmetric seating, two blocks per pair).

Either build the checkpoint list from ``--min-step`` / ``--max-step`` / ``--step-multiple``,
or pass an explicit ``--steps`` comma list (must have ``.pt`` files under ``run-dir/checkpoints``).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.envs import RewardConfig
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint
from src.registry import make_game
from src.training.trainer import StartPolicy
from src.utils.match import play_single_game


def _parse_steps(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _elo_from_pairwise(
    steps: Sequence[int],
    pairwise: List[Tuple[int, int, int, int, int]],
    *,
    base: float = 1500.0,
    k: float = 20.0,
    iters: int = 60,
) -> Dict[int, float]:
    """
    pairwise: (step_lo, step_hi, wins_lo, wins_hi, draws) with step_lo < step_hi.
    One update per pair per iteration (score = fraction of points for lo in that mini-match).
    """
    r = {int(s): float(base) for s in steps}
    for _ in range(iters):
        for slo, shi, wl, wh, dr in pairwise:
            n = wl + wh + dr
            if n <= 0:
                continue
            score_lo = (wl + 0.5 * dr) / n
            exp_lo = 1.0 / (1.0 + 10 ** ((r[shi] - r[slo]) / 400.0))
            delta = k * (score_lo - exp_lo)
            r[slo] += delta
            r[shi] -= delta
    mean_r = sum(r.values()) / max(len(r), 1)
    return {s: r[s] - mean_r + base for s in r}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="minibg_ppo_structured")
    ap.add_argument("--min-step", type=int, default=100_000)
    ap.add_argument("--max-step", type=int, default=9_999_999)
    ap.add_argument("--step-multiple", type=int, default=100_000)
    ap.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated training steps (overrides min/max/multiple). E.g. 100000,250000,1000000",
    )
    ap.add_argument(
        "--per-side",
        type=int,
        default=12,
        help="Games per seating; total games per pair = 2 × this.",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=55_019)
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Standings: step,wins,games,winrate",
    )
    ap.add_argument(
        "--out-pairwise",
        type=Path,
        default=None,
        help="Pairwise: step_lo,step_hi,wins_lo,wins_hi,draws,games",
    )
    ap.add_argument(
        "--out-elo",
        type=Path,
        default=None,
        help="Elo-style ratings: step,elo (same order as standings sort by elo desc)",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    prefix = args.prefix

    if args.steps:
        raw = _parse_steps(args.steps)
        steps = []
        for s in raw:
            p = run_dir / "checkpoints" / f"{prefix}_{s}.pt"
            if p.is_file():
                steps.append(s)
            else:
                print(f"skip missing checkpoint step {s}", file=sys.stderr, flush=True)
        steps = sorted(set(steps))
    else:
        mult = int(args.step_multiple)
        lo = int(args.min_step)
        hi = int(args.max_step)
        steps = []
        for s in range((lo + mult - 1) // mult * mult, hi + 1, mult):
            p = run_dir / "checkpoints" / f"{prefix}_{s}.pt"
            if p.is_file():
                steps.append(s)

    if len(steps) < 2:
        raise SystemExit("Need at least 2 checkpoints")

    paths = {s: run_dir / "checkpoints" / f"{prefix}_{s}.pt" for s in steps}
    n_side = int(args.per_side)
    cache: dict[int, object] = {}

    def get_agent(step: int) -> object:
        if step not in cache:
            cache[step] = load_training_agent_checkpoint(
                paths[step], device=args.device, seed=args.seed + (step % 10_007)
            )
            a = cache[step]
            if hasattr(a, "eval"):
                a.eval()
        return cache[step]

    def symmetric_match(a_step: int, b_step: int, seed_base: int) -> tuple[int, int, int]:
        agent_a = get_agent(a_step)
        agent_b = get_agent(b_step)
        wa = wb = dr = 0

        for g in range(n_side):
            env = make_game(
                "minibg",
                reward_config=RewardConfig(),
                battle_damage_shaping=0.06,
            )
            for p in (agent_a, agent_b):
                if hasattr(p, "set_env"):
                    p.set_env(env)
            res = play_single_game(
                env,
                agent_a,
                agent_b,
                start_policy=StartPolicy.RANDOM,
                random_opening_config=None,
                deterministic_agent=True,
                deterministic_opponent=True,
                seed=seed_base + g,
            )
            r = res["reward"]
            if r == 1:
                wa += 1
            elif r == -1:
                wb += 1
            else:
                dr += 1

        for g in range(n_side):
            env = make_game(
                "minibg",
                reward_config=RewardConfig(),
                battle_damage_shaping=0.06,
            )
            for p in (agent_a, agent_b):
                if hasattr(p, "set_env"):
                    p.set_env(env)
            res = play_single_game(
                env,
                agent_b,
                agent_a,
                start_policy=StartPolicy.RANDOM,
                random_opening_config=None,
                deterministic_agent=True,
                deterministic_opponent=True,
                seed=seed_base + n_side + g,
            )
            r = res["reward"]
            if r == 1:
                wb += 1
            elif r == -1:
                wa += 1
            else:
                dr += 1

        return wa, wb, dr

    wins = {s: 0 for s in steps}
    games = {s: 0 for s in steps}
    pair_i = 0
    n = len(steps)
    pairwise: List[Tuple[int, int, int, int, int]] = []

    g_per_pair = 2 * n_side
    print(
        f"Round-robin: {n} checkpoints, steps {steps[0]} … {steps[-1]}",
        flush=True,
    )
    print(
        f"Per pairing: {n_side} games each seat order → {g_per_pair} games total",
        flush=True,
    )

    for i in range(n):
        for j in range(i + 1, n):
            sa, sb = steps[i], steps[j]
            seed_base = args.seed + pair_i * 10_000 + (sa % 997) * 17 + (sb % 997) * 23
            pair_i += 1
            wa, wb, dr = symmetric_match(sa, sb, seed_base)
            assert wa + wb + dr == g_per_pair
            if sa < sb:
                pairwise.append((sa, sb, wa, wb, dr))
            else:
                pairwise.append((sb, sa, wb, wa, dr))
            wins[sa] += wa
            wins[sb] += wb
            games[sa] += g_per_pair
            games[sb] += g_per_pair
            print(f"  {sa} vs {sb}: {wa}-{wb} (draws {dr})", flush=True)

    print("\n=== Standings (win rate) ===", flush=True)
    standings = [(s, wins[s], games[s], wins[s] / games[s] if games[s] else 0.0) for s in steps]
    standings.sort(key=lambda x: (-x[3], -x[0]))
    print(f"{'step':>10}  {'W':>5}  {'G':>5}  {'win%':>7}", flush=True)
    for s, w, g, rate in standings:
        print(f"{s:>10}  {w:>5}  {g:>5}  {100.0 * rate:>6.2f}", flush=True)

    elo = _elo_from_pairwise(steps, pairwise)
    elo_rows = sorted(steps, key=lambda s: (-elo[s], -s))
    print("\n=== Elo (approx, centered mean 1500) ===", flush=True)
    for s in elo_rows:
        print(f"  {s:>10}  {elo[s]:>8.1f}", flush=True)

    out_csv = args.out_csv
    if out_csv is not None:
        out_csv = out_csv.resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["step", "wins", "games", "winrate"])
            for s, w, g, rate in sorted(standings, key=lambda x: x[0]):
                wtr.writerow([s, w, g, f"{rate:.6f}"])
        print(f"\nWrote {out_csv}", flush=True)

    if args.out_pairwise is not None:
        pth = args.out_pairwise.resolve()
        pth.parent.mkdir(parents=True, exist_ok=True)
        with pth.open("w", encoding="utf-8", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["step_lo", "step_hi", "wins_lo", "wins_hi", "draws", "games"])
            for slo, shi, wl, wh, dr in pairwise:
                wtr.writerow([slo, shi, wl, wh, dr, wl + wh + dr])
        print(f"Wrote {pth}", flush=True)

    if args.out_elo is not None:
        pth = args.out_elo.resolve()
        pth.parent.mkdir(parents=True, exist_ok=True)
        with pth.open("w", encoding="utf-8", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["step", "elo"])
            for s in elo_rows:
                wtr.writerow([s, f"{elo[s]:.4f}"])
        print(f"Wrote {pth}", flush=True)


if __name__ == "__main__":
    main()

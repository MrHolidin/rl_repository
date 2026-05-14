#!/usr/bin/env python3
"""Append 1.6M vs each 100k-multiple checkpoint (10 games symmetric), refit Bradley–Terry."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from src.envs import RewardConfig
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint
from src.registry import make_game
from src.training.trainer import StartPolicy
from src.utils.match import play_single_game


def parse_log_pairs(path: Path) -> list[tuple[int, int, int, int]]:
    """Lines 'a vs b: wa-wb (draws d)' with a<b; returns (a,b,wa,n)."""
    text = path.read_text(encoding="utf-8")
    pat = re.compile(r"^\s*(\d+)\s+vs\s+(\d+):\s+(\d+)-(\d+)\s+\(draws\s+(\d+)\)", re.M)
    out = []
    for m in pat.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        wa, wb, dr = int(m.group(3)), int(m.group(4)), int(m.group(5))
        n = wa + wb + dr
        out.append((a, b, wa, n))
    return out


def symmetric_match(
    agent_a: object,
    agent_b: object,
    *,
    n_side: int,
    seed_base: int,
) -> tuple[int, int, int]:
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


def neg_loglik_bt(theta_free: np.ndarray, W: np.ndarray, N: np.ndarray, m: int) -> float:
    theta = np.zeros(m)
    theta[:-1] = theta_free
    ll = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            nij = int(N[i, j])
            if nij == 0:
                continue
            wij = int(W[i, j])
            t = theta[i] - theta[j]

            def log_sigmoid(x: float) -> float:
                if x >= 0:
                    return -math.log1p(math.exp(-x))
                return x - math.log1p(math.exp(x))

            ll += wij * log_sigmoid(t) + (nij - wij) * log_sigmoid(-t)
    return -ll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--tournament-log", type=Path, required=True)
    ap.add_argument("--new-step", type=int, default=1_600_000)
    ap.add_argument("--games-per-pair", type=int, default=10, help="Total games; split symmetric half/half")
    ap.add_argument("--prefix", type=str, default="minibg_ppo_structured")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=88_220)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    pairs_log = parse_log_pairs(args.tournament_log.resolve())
    if not pairs_log:
        raise SystemExit("No pairs parsed from log")

    steps_set = set()
    for a, b, _, _ in pairs_log:
        steps_set.add(a)
        steps_set.add(b)
    base_steps = sorted(steps_set)
    new_step = int(args.new_step)
    if new_step in base_steps:
        raise SystemExit(f"{new_step} already in tournament log")
    all_steps = sorted(base_steps + [new_step])
    idx = {s: k for k, s in enumerate(all_steps)}
    m = len(all_steps)

    W = np.zeros((m, m), dtype=np.int64)
    N = np.zeros((m, m), dtype=np.int64)
    for a, b, wa, n in pairs_log:
        i, j = idx[a], idx[b]
        W[i, j] = wa
        W[j, i] = n - wa
        N[i, j] = n
        N[j, i] = n

    n_side = max(1, int(args.games_per_pair) // 2)
    paths = {s: run_dir / "checkpoints" / f"{args.prefix}_{s}.pt" for s in all_steps}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)

    cache: dict[int, object] = {}

    def get_agent(s: int) -> object:
        if s not in cache:
            cache[s] = load_training_agent_checkpoint(
                paths[s], device=args.device, seed=args.seed + (s % 10_007)
            )
            if hasattr(cache[s], "eval"):
                cache[s].eval()
        return cache[s]

    print(
        f"New matchups: {new_step} vs each of {len(base_steps)} checkpoints, "
        f"{2 * n_side} games each (symmetric)\n",
        flush=True,
    )
    row_lines = ["| opp | W_1.6M | W_opp | draws | 1.6M win % |", "|---|---:|---:|---:|---:|"]
    pair_k = 0
    for opp in base_steps:
        a_new = get_agent(new_step)
        a_opp = get_agent(opp)
        seed_b = args.seed + pair_k * 9973 + (opp % 1000) * 101
        pair_k += 1
        wa, wb, dr = symmetric_match(a_new, a_opp, n_side=n_side, seed_base=seed_b)
        assert wa + wb + dr == 2 * n_side
        # wa = wins new_step, wb = wins opp (symmetric_match(new, opp))
        lo, hi = idx[opp], idx[new_step]
        W[lo, hi] = wb
        W[hi, lo] = wa
        N[lo, hi] = N[hi, lo] = 2 * n_side
        tot = wa + wb
        pct = 100.0 * wa / tot if tot else 0.0
        row_lines.append(f"| {opp} | {wa} | {wb} | {dr} | {pct:.1f} |")
        print(f"  {new_step} vs {opp}: {wa}-{wb} (draws {dr})", flush=True)

    report_md = run_dir / "tournament_1600k_extension.md"
    report_md.write_text("\n".join(row_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {report_md}", flush=True)

    try:
        import scipy.optimize as opt

        x0 = np.zeros(m - 1)
        res = opt.minimize(
            lambda x: neg_loglik_bt(x, W, N, m),
            x0,
            method="L-BFGS-B",
            options={"maxiter": 2000},
        )
        theta = np.zeros(m)
        theta[:-1] = res.x
    except ImportError:
        theta_f = np.zeros(m - 1)
        for _ in range(25000):
            theta = np.zeros(m)
            theta[:-1] = theta_f
            g = np.zeros(m - 1)
            for i in range(m):
                for j in range(i + 1, m):
                    nij = int(N[i, j])
                    if nij == 0:
                        continue
                    wij = int(W[i, j])
                    tie = theta[i] - theta[j]
                    sig = 1.0 / (1.0 + math.exp(-tie))
                    err = wij - nij * sig
                    if i < m - 1:
                        g[i] += err
                    if j < m - 1:
                        g[j] -= err
            theta_f += 0.0025 * g
        theta = np.zeros(m)
        theta[:-1] = theta_f

    # Anchor newest at 0 → pseudo-Elo 1500 for 1.6M
    anchor_i = idx[new_step]
    shift = theta[anchor_i].copy()
    theta -= shift

    scale = 400 / math.log(10)
    R = 1500 + scale * theta

    order = np.argsort(-R)
    print("\n=== Bradley–Terry → pseudo-Elo (anchor 1.6M = 1500) ===\n")
    print(f"{'step':>10}  {'θ':>8}  {'pseudo-Elo':>10}")
    stand = []
    for o in order:
        s = all_steps[o]
        stand.append(f"| {s} | {theta[o]:.4f} | {R[o]:.1f} |")
        print(f"{s:>10}  {theta[o]:>8.4f}  {R[o]:>10.1f}")

    full_md = run_dir / "tournament_1600k_extension_standings.md"
    full_md.write_text(
        "\n".join(
            row_lines
            + [
                "",
                "### Standings (BT → pseudo-Elo, 1.6M = 1500)",
                "",
                "| step | θ | pseudo-Elo |",
                "|---|---:|---:|",
            ]
            + stand
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {full_md}", flush=True)


if __name__ == "__main__":
    main()

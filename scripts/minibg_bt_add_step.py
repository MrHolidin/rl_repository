#!/usr/bin/env python3
"""Build BT matrix: full 100k circle from log + prior extension MD + new step vs listed opponents."""

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
    text = path.read_text(encoding="utf-8")
    pat = re.compile(r"^\s*(\d+)\s+vs\s+(\d+):\s+(\d+)-(\d+)\s+\(draws\s+(\d+)\)", re.M)
    out = []
    for m in pat.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        wa, wb, dr = int(m.group(3)), int(m.group(4)), int(m.group(5))
        n = wa + wb + dr
        out.append((a, b, wa, n))
    return out


def parse_extension_md(path: Path) -> list[tuple[int, int, int]]:
    """Rows | opp | W_new | W_opp | ... → (opp, wins_new, wins_opp)."""
    rows = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line.strip())
        if not m:
            continue
        opp, wn, wo = int(m.group(1)), int(m.group(2)), int(m.group(3))
        rows.append((opp, wn, wo))
    return rows


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

    def log_sigmoid(x: float) -> float:
        if x >= 0:
            return -math.log1p(math.exp(-x))
        return x - math.log1p(math.exp(x))

    for i in range(m):
        for j in range(i + 1, m):
            nij = int(N[i, j])
            if nij == 0:
                continue
            wij = int(W[i, j])
            t = theta[i] - theta[j]
            ll += wij * log_sigmoid(t) + (nij - wij) * log_sigmoid(-t)
    return -ll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--tournament-log", type=Path, required=True)
    ap.add_argument(
        "--extension-md",
        type=Path,
        required=True,
        help="Markdown with | opp | W_ext | W_opp | rows (e.g. 1.6M extension)",
    )
    ap.add_argument("--extension-step", type=int, default=1_600_000)
    ap.add_argument("--new-step", type=int, required=True)
    ap.add_argument("--games-per-pair", type=int, default=10)
    ap.add_argument("--prefix", type=str, default="minibg_ppo_structured")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=91_055)
    ap.add_argument(
        "--anchor-step",
        type=int,
        default=None,
        help="BT anchor θ=0 → pseudo-Elo 1500 (default: --new-step)",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    ext_step = int(args.extension_step)
    new_step = int(args.new_step)
    anchor = int(args.anchor_step) if args.anchor_step is not None else new_step

    pairs_log = parse_log_pairs(args.tournament_log.resolve())
    if not pairs_log:
        raise SystemExit("No pairs in tournament log")

    base_set = set()
    for a, b, _, _ in pairs_log:
        base_set.add(a)
        base_set.add(b)
    base_steps = sorted(base_set)

    ext_rows = parse_extension_md(args.extension_md.resolve())
    ex_opps = sorted({o for o, _, _ in ext_rows})
    if ex_opps != base_steps:
        raise SystemExit(
            f"Extension opps {ex_opps} != base_steps {base_steps} "
            "(need same 100k-multiples as circle)"
        )

    mid_steps = sorted(set(base_steps + [ext_step]))
    if new_step in mid_steps:
        raise SystemExit(f"{new_step} already in base+extension")
    all_steps = sorted(mid_steps + [new_step])
    idx = {s: k for k, s in enumerate(all_steps)}
    m = len(all_steps)

    W = np.zeros((m, m), dtype=np.int64)
    N = np.zeros((m, m), dtype=np.int64)

    for a, b, wa, n in pairs_log:
        i, j = idx[a], idx[b]
        W[i, j] = wa
        W[j, i] = n - wa
        N[i, j] = N[j, i] = n

    for opp, w_ext, w_opp in ext_rows:
        tot = w_ext + w_opp
        lo, hi = idx[opp], idx[ext_step]
        W[lo, hi] = w_opp
        W[hi, lo] = w_ext
        N[lo, hi] = N[hi, lo] = tot

    n_side = max(1, int(args.games_per_pair) // 2)
    opp_list = base_steps + [ext_step]

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

    tag = f"{new_step // 1000}k"
    head = f"| opp | W_{tag} | W_opp | draws | {tag} win % |"
    row_lines = [head, "|---|---:|---:|---:|---:|"]
    print(f"{new_step} vs {len(opp_list)} opponents × {2 * n_side} games\n", flush=True)
    pair_k = 0
    for opp in opp_list:
        a_new = get_agent(new_step)
        a_opp = get_agent(opp)
        seed_b = args.seed + pair_k * 9973 + (opp % 1000) * 101
        pair_k += 1
        wa, wb, dr = symmetric_match(a_new, a_opp, n_side=n_side, seed_base=seed_b)
        assert wa + wb + dr == 2 * n_side
        lo, hi = idx[opp], idx[new_step]
        W[lo, hi] = wb
        W[hi, lo] = wa
        N[lo, hi] = N[hi, lo] = 2 * n_side
        pct = 100.0 * wa / (wa + wb)
        row_lines.append(f"| {opp} | {wa} | {wb} | {dr} | {pct:.1f} |")
        print(f"  {new_step} vs {opp}: {wa}-{wb} (draws {dr})", flush=True)

    out_md = run_dir / f"tournament_{tag}_extension.md"
    out_md.write_text("\n".join(row_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_md}", flush=True)

    try:
        import scipy.optimize as opt

        res = opt.minimize(
            lambda x: neg_loglik_bt(x, W, N, m),
            np.zeros(m - 1),
            method="L-BFGS-B",
            options={"maxiter": 2000},
        )
        theta = np.zeros(m)
        theta[:-1] = res.x
    except ImportError:
        theta_f = np.zeros(m - 1)
        for _ in range(30000):
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

    anchor_i = idx[anchor]
    theta -= theta[anchor_i]

    scale = 400 / math.log(10)
    R = 1500 + scale * theta

    order = np.argsort(-R)
    anchor_label = f"{anchor // 1000}k" if anchor >= 100000 else str(anchor)
    print(f"\n=== Bradley–Terry → pseudo-Elo (anchor {anchor} = 1500) ===\n")
    print(f"{'step':>10}  {'θ':>8}  {'pseudo-Elo':>10}")
    stand = []
    for o in order:
        s = all_steps[o]
        stand.append(f"| {s} | {theta[o]:.4f} | {R[o]:.1f} |")
        print(f"{s:>10}  {theta[o]:>8.4f}  {R[o]:>10.1f}")

    full_md = run_dir / f"tournament_bt_standings_{tag}_full.md"
    full_md.write_text(
        "\n".join(
            row_lines
            + [
                "",
                f"### Standings (BT → pseudo-Elo, anchor {anchor_label} = 1500)",
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

#!/usr/bin/env python3
"""Round-robin tournament: checkpoint steps divisible by 100k, symmetric seating."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.envs import RewardConfig
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint
from src.registry import make_game
from src.training.trainer import StartPolicy
from src.utils.match import play_single_game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="minibg_ppo_structured")
    ap.add_argument("--min-step", type=int, default=100_000)
    ap.add_argument("--max-step", type=int, default=9_999_999)
    ap.add_argument("--step-multiple", type=int, default=100_000)
    ap.add_argument("--per-side", type=int, default=12, help="Games as agent + games as opp each = 2x this total per pair per player")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=55_019)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    mult = int(args.step_multiple)
    lo = int(args.min_step)
    hi = int(args.max_step)
    steps = []
    for s in range((lo + mult - 1) // mult * mult, hi + 1, mult):
        p = run_dir / "checkpoints" / f"{args.prefix}_{s}.pt"
        if p.is_file():
            steps.append(s)
    if len(steps) < 2:
        raise SystemExit("Need at least 2 checkpoints")

    paths = {s: run_dir / "checkpoints" / f"{args.prefix}_{s}.pt" for s in steps}
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

    print(
        f"Round-robin: {n} checkpoints (steps {steps[0]}..{steps[-1]}, multiple of {mult})",
        flush=True,
    )
    print(f"Per pairing: {n_side} games each seat order → {2 * n_side} games total", flush=True)

    for i in range(n):
        for j in range(i + 1, n):
            sa, sb = steps[i], steps[j]
            seed_base = args.seed + pair_i * 10_000 + (sa % 997) * 17 + (sb % 997) * 23
            pair_i += 1
            wa, wb, dr = symmetric_match(sa, sb, seed_base)
            assert wa + wb + dr == 2 * n_side
            wins[sa] += wa
            wins[sb] += wb
            games[sa] += 2 * n_side
            games[sb] += 2 * n_side
            print(f"  {sa} vs {sb}: {wa}-{wb} (draws {dr})", flush=True)

    print("\n=== Standings (win rate) ===", flush=True)
    rows = [(s, wins[s], games[s], wins[s] / games[s] if games[s] else 0.0) for s in steps]
    rows.sort(key=lambda x: (-x[3], -x[0]))
    print(f"{'step':>10}  {'W':>5}  {'G':>5}  {'win%':>7}", flush=True)
    for s, w, g, rate in rows:
        print(f"{s:>10}  {w:>5}  {g:>5}  {100.0 * rate:>6.2f}", flush=True)


if __name__ == "__main__":
    main()

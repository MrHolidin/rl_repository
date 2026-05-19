#!/usr/bin/env python3
"""MiniBG: two training checkpoints head-to-head (symmetric 50/50 seating, 100 games total by default)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.envs  # noqa: F401

from src.envs import RewardConfig
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint
from src.registry import make_game
from src.training.trainer import StartPolicy
from src.utils.match import play_single_game


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-a", type=Path, required=True, help="First agent (.pt)")
    ap.add_argument("--ckpt-b", type=Path, required=True, help="Second agent (.pt)")
    ap.add_argument("--label-a", type=str, default="A")
    ap.add_argument("--label-b", type=str, default="B")
    ap.add_argument(
        "--per-side",
        type=int,
        default=50,
        help="Games with A opening seat order + this many with B opening → total 2×",
    )
    ap.add_argument("--battle-damage-shaping", type=float, default=0.06)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    ck_a = args.ckpt_a.resolve()
    ck_b = args.ckpt_b.resolve()
    n_side = int(args.per_side)
    seed = int(args.seed)

    agent_a = load_training_agent_checkpoint(ck_a, device=args.device, seed=seed)
    agent_b = load_training_agent_checkpoint(ck_b, device=args.device, seed=seed + 101)
    for a in (agent_a, agent_b):
        if hasattr(a, "eval"):
            a.eval()
        if hasattr(a, "epsilon"):
            a.epsilon = 0.0

    wins_a = wins_b = draws = 0
    mg = {"battle_damage_shaping": float(args.battle_damage_shaping)}

    for g in range(n_side):
        env = make_game("minibg", reward_config=RewardConfig(), **mg)
        res = play_single_game(
            env,
            agent_a,
            agent_b,
            start_policy=StartPolicy.RANDOM,
            random_opening_config=None,
            deterministic_agent=True,
            deterministic_opponent=True,
            seed=seed + g,
        )
        r = res["reward"]
        if r == 1:
            wins_a += 1
        elif r == -1:
            wins_b += 1
        else:
            draws += 1

    for g in range(n_side):
        env = make_game("minibg", reward_config=RewardConfig(), **mg)
        res = play_single_game(
            env,
            agent_b,
            agent_a,
            start_policy=StartPolicy.RANDOM,
            random_opening_config=None,
            deterministic_agent=True,
            deterministic_opponent=True,
            seed=seed + n_side + g,
        )
        r = res["reward"]
        if r == 1:
            wins_b += 1
        elif r == -1:
            wins_a += 1
        else:
            draws += 1

    total = wins_a + wins_b + draws
    print(
        f"{args.label_a} vs {args.label_b}: "
        f"{wins_a}-{wins_b} (draws {draws}) over {total} games — "
        f"{args.label_a} winrate {100.0 * wins_a / total:.2f}%",
        flush=True,
    )

    if args.out_csv:
        p = args.out_csv.resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ckpt_a",
                    "ckpt_b",
                    "label_a",
                    "label_b",
                    "games",
                    "wins_a",
                    "wins_b",
                    "draws",
                    "winrate_a",
                ]
            )
            w.writerow(
                [
                    str(ck_a),
                    str(ck_b),
                    args.label_a,
                    args.label_b,
                    total,
                    wins_a,
                    wins_b,
                    draws,
                    f"{wins_a / total:.6f}",
                ]
            )
        print(f"Wrote {p}", flush=True)


if __name__ == "__main__":
    main()

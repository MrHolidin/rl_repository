"""CLI: evaluate checkpoints vs random/heuristic, plot progress."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.eval_checkpoints import (
    eval_checkpoints_vs_opponents,
    find_checkpoints,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints vs random/heuristic opponents, plot win rate vs step.",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory containing checkpoints/ subdir (e.g. runs/connect4_main)",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=50,
        help="Games per checkpoint per opponent (default: 50)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for plot (default: run_dir/progress.png)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for DQN (cuda/cpu, default: auto)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Checkpoint prefix filter (e.g. dqn to match dqn_2000.pt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoints_dir = run_dir / "checkpoints"

    if not checkpoints_dir.exists():
        print(f"Checkpoints dir not found: {checkpoints_dir}", file=sys.stderr)
        sys.exit(1)

    found = find_checkpoints(checkpoints_dir, prefix=args.prefix)
    if not found:
        print(f"No checkpoints found in {checkpoints_dir}", file=sys.stderr)
        sys.exit(1)

    paths = [p for p, _ in found]
    print(f"Found {len(paths)} checkpoints: {[p.name for p in paths]}")

    df = eval_checkpoints_vs_opponents(
        paths,
        num_games=args.num_games,
        device=args.device,
        seed=args.seed,
        randomize_first_player=True,
    )

    if df.empty:
        print("No valid checkpoints evaluated.", file=sys.stderr)
        sys.exit(1)

    win_cols = [c for c in df.columns if c.startswith("win_rate_")]
    if not win_cols:
        print("No win rate columns.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or (run_dir / "progress.png")
    out_path = out_path.resolve()

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in win_cols:
        opp_name = col.replace("win_rate_", "")
        ax.plot(df["step"], df[col], "o-", label=f"vs {opp_name}", markersize=6)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Win rate")
    ax.set_title(f"Checkpoint progress: {run_dir.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")

    # Print summary
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

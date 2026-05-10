"""CLI: evaluate checkpoints vs random/heuristic, plot progress."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.envs import RewardConfig
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
        "--batch_size",
        type=int,
        default=None,
        help="Parallel envs for batched eval (default: num_games). Set 0 for sequential.",
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
        help="Checkpoint prefix filter (e.g. dqn to match dqn_2000.pt). "
        "For game=minibg defaults to minibg_dqn if omitted.",
    )
    parser.add_argument(
        "--checkpoint-stems",
        nargs="*",
        default=None,
        metavar="STEM",
        help="If set, only these stems (e.g. minibg_dqn_300000). Order preserved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--start_policy",
        type=str,
        choices=["random", "agent_first", "opponent_first"],
        default="random",
        help="Who goes first: random, agent_first, or opponent_first (default: random)",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["random", "heuristic"],
        metavar="OPP",
        help="Opponents: random, heuristic, smart_heuristic, othello_heuristic, or minimax_N e.g. minimax_2, minimax_4 (default: random heuristic)",
    )
    parser.add_argument(
        "--game",
        type=str,
        choices=["connect4", "othello", "minibg"],
        default="connect4",
        help="Game type (default: connect4)",
    )
    parser.add_argument(
        "--minibg-battle-damage-shaping",
        type=float,
        default=0.06,
        help="Forward to make_game(minibg, ...) when game=minibg (default: 0.06)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Eval table CSV (default: run_dir/eval_results.csv)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip progress.png (eval CSV only)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoints_dir = run_dir / "checkpoints"

    if not checkpoints_dir.exists():
        print(f"Checkpoints dir not found: {checkpoints_dir}", file=sys.stderr)
        sys.exit(1)

    prefix = args.prefix
    if args.game == "minibg" and prefix is None:
        prefix = "minibg_dqn"

    found = find_checkpoints(checkpoints_dir, prefix=prefix)
    if not found:
        print(f"No checkpoints found in {checkpoints_dir}", file=sys.stderr)
        sys.exit(1)

    by_stem = {p.stem: p for p, _ in found}
    if args.checkpoint_stems:
        paths = []
        for stem in args.checkpoint_stems:
            p = by_stem.get(stem)
            if p is None:
                avail = ", ".join(sorted(by_stem))
                print(f"Checkpoint stem not found: {stem!r}. Available: {avail}", file=sys.stderr)
                sys.exit(1)
            paths.append(p)
    else:
        paths = [p for p, _ in found]

    print(f"Evaluating {len(paths)} checkpoints: {[p.name for p in paths]}")

    batch_size = args.num_games if args.batch_size is None else args.batch_size

    minibg_params: Optional[dict] = None
    reward_config: Optional[RewardConfig] = None
    if args.game == "minibg":
        minibg_params = {"battle_damage_shaping": float(args.minibg_battle_damage_shaping)}
        reward_config = RewardConfig()

    out_csv = args.out_csv.resolve() if args.out_csv else (run_dir / "eval_results.csv")

    df = eval_checkpoints_vs_opponents(
        paths,
        opponent_names=args.opponents,
        num_games=args.num_games,
        batch_size=batch_size,
        device=args.device,
        seed=args.seed,
        reward_config=reward_config,
        start_policy=args.start_policy,
        game_id=args.game,
        out_csv=out_csv,
        minibg_params=minibg_params,
    )

    if df.empty:
        print("No valid checkpoints evaluated.", file=sys.stderr)
        sys.exit(1)

    win_cols = [c for c in df.columns if c.startswith("win_rate_")]
    if not win_cols:
        print("No win rate columns.", file=sys.stderr)
        sys.exit(1)

    if not args.no_plot:
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

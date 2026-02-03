"""Evaluate heuristic minimax (depth 2,3,4) vs DQN checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs import Connect4Env
from src.envs.connect4 import Connect4Game
from src.agents.dqn.agent import DQNAgent
from src.search.connect4 import (
    make_connect4_heuristic_minimax_policy,
    Connect4MinimaxEnvAdapter,
)
from src.evaluation.eval_checkpoints import find_checkpoints, load_dqn_checkpoint
from src.utils.match import play_match


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimax vs DQN checkpoint evaluation",
    )
    parser.add_argument(
        "checkpoint_or_run",
        type=str,
        help="Path to .pt checkpoint or to run dir (e.g. runs/connect4_self_play_dist_v1); "
        "if dir, uses latest in run_dir/checkpoints/",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Games per (depth, side) pair (default 100)",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Minimax depths (default 2 3 4)",
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        choices=["trivial", "smart"],
        default="trivial",
        help="Heuristic type: 'trivial' (terminal only) or 'smart' (pattern-based)",
    )
    parser.add_argument(
        "--no-alpha-beta",
        action="store_true",
        help="Disable alpha-beta pruning (slower, same result)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.checkpoint_or_run)
    if path.is_dir():
        checkpoints_dir = path / "checkpoints"
        if not checkpoints_dir.exists():
            print(f"No checkpoints dir: {checkpoints_dir}", file=sys.stderr)
            sys.exit(1)
        found = find_checkpoints(checkpoints_dir, prefix="dqn")
        if not found:
            print(f"No dqn checkpoints in {checkpoints_dir}", file=sys.stderr)
            sys.exit(1)
        checkpoint_path = found[-1][0]
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        if not path.exists():
            print(f"Checkpoint not found: {path}", file=sys.stderr)
            sys.exit(1)
        checkpoint_path = path

    dqn_agent = load_dqn_checkpoint(
        checkpoint_path,
        device=args.device,
        seed=args.seed,
    )
    game = Connect4Game(rows=6, cols=7)
    env = Connect4Env(rows=6, cols=7)

    use_ab = not args.no_alpha_beta
    print(f"Heuristic: {args.heuristic}, alpha-beta: {use_ab}")
    print("Depth | DQN wins | Draws | Minimax wins | DQN win%")
    print("-" * 55)

    for depth in args.depths:
        policy = make_connect4_heuristic_minimax_policy(
            game, depth=depth, heuristic=args.heuristic, use_alpha_beta=use_ab
        )
        adapter = Connect4MinimaxEnvAdapter(
            game,
            policy,
            get_state=lambda: env.get_state(),
        )
        w1, draws, w2 = play_match(
            dqn_agent,
            adapter,
            num_games=args.num_games,
            seed=args.seed + depth,
            randomize_first_player=True,
            env=env,
        )
        dqn_wins = w1
        minimax_wins = w2
        total = dqn_wins + draws + minimax_wins
        dqn_pct = 100.0 * dqn_wins / total if total else 0.0
        print(f"  {depth}  |    {dqn_wins:4}   |  {draws:4} |     {minimax_wins:4}     |  {dqn_pct:.1f}%")


if __name__ == "__main__":
    main()

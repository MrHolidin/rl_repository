"""Round-robin tournament: auto (latest per run) or manual (runs + steps) + Heuristic + SmartHeuristic."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import HeuristicAgent, SmartHeuristicAgent
from src.envs import Connect4Env, RewardConfig
from src.evaluation.eval_checkpoints import find_checkpoints, load_dqn_checkpoint
from src.utils.match import play_match_batched


def _step_from_name(name: str) -> int | None:
    m = re.search(r"_(\d+)\.pt$", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def collect_agents(runs_dir: Path, device: str | None = None, seed: int = 42):
    """Auto: latest checkpoint per run + heuristic + smart_heuristic."""
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        return [], []

    names: list[str] = []
    agents: list = []

    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue
        ckpt_dir = run_path / "checkpoints"
        if not ckpt_dir.is_dir():
            continue
        found = find_checkpoints(ckpt_dir, prefix=None, sort_by_step=True)
        if not found:
            continue
        path, step = found[-1]
        try:
            agent = load_dqn_checkpoint(path, device=device, seed=seed)
            name = f"{run_path.name}/{path.stem}"
            names.append(name)
            agents.append(agent)
        except Exception:
            continue

    _add_heuristics(names, agents, seed)
    return names, agents


def collect_agents_manual(
    runs_dir: Path,
    run_names: list[str],
    steps: list[int],
    *,
    device: str | None = None,
    seed: int = 42,
    prefix: str = "dqn",
) -> tuple[list[str], list]:
    """Manual: for each run, load checkpoints at given steps; add heuristic + smart_heuristic."""
    runs_dir = runs_dir.resolve()
    names: list[str] = []
    agents: list = []

    for run_name in run_names:
        ckpt_dir = runs_dir / run_name / "checkpoints"
        if not ckpt_dir.is_dir():
            continue
        found = find_checkpoints(ckpt_dir, prefix=prefix, sort_by_step=True)
        step_to_path = {step: path for path, step in found}
        for step in steps:
            path = step_to_path.get(step)
            if path is None:
                continue
            try:
                agent = load_dqn_checkpoint(path, device=device, seed=seed)
                name = f"{run_name}/{path.stem}"
                names.append(name)
                agents.append(agent)
            except Exception:
                continue

    _add_heuristics(names, agents, seed)
    return names, agents


def _add_heuristics(names: list[str], agents: list, seed: int) -> None:
    h = HeuristicAgent(seed=seed + 100)
    s = SmartHeuristicAgent(seed=seed + 101)
    for a in (h, s):
        a.eval()
    names.extend(["heuristic", "smart_heuristic"])
    agents.extend([h, s])


def main():
    p = argparse.ArgumentParser(
        description="Round-robin tournament: auto (latest per run) or manual (--manual --runs --steps) + heuristic + smart_heuristic",
    )
    p.add_argument("--runs_dir", type=Path, default=Path("runs"), help="Runs directory")
    p.add_argument("--num_games", type=int, default=100, help="Games per pair")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for eval")
    p.add_argument("--seed", type=int, default=12345, help="Tournament seed")
    p.add_argument("--device", type=str, default=None, help="Device for DQN (cuda/cpu)")
    p.add_argument("--out", type=Path, default=None, help="Winrate matrix output (default: runs/tournament_winrate.csv or runs/tournament_manual.csv)")
    p.add_argument("--manual", action="store_true", help="Manual mode: use --runs and --steps instead of auto-scan")
    p.add_argument("--runs", nargs="+", type=str, default=[], help="Run names (e.g. connect4_only_self_play connect4_only_self_play_per); used with --manual")
    p.add_argument("--steps", nargs="+", type=int, default=[], help="Checkpoint steps (e.g. 100000 200000 300000); used with --manual")
    p.add_argument("--prefix", type=str, default="dqn", help="Checkpoint prefix (default: dqn)")
    p.add_argument("--epsilon", type=float, default=0.0, help="DQN epsilon for eval (default: 0.0)")
    args = p.parse_args()

    if args.manual:
        if not args.runs or not args.steps:
            p.error("--manual requires --runs and --steps")
        names, agents = collect_agents_manual(
            args.runs_dir,
            args.runs,
            args.steps,
            device=args.device,
            seed=args.seed,
            prefix=args.prefix,
        )
    else:
        names, agents = collect_agents(args.runs_dir, device=args.device, seed=args.seed)
    for a in agents:
        if hasattr(a, "eval"):
            a.eval()
        if hasattr(a, "epsilon"):
            setattr(a, "epsilon", args.epsilon)

    n = len(agents)
    if n < 2:
        print("Need at least 2 agents.", file=sys.stderr)
        sys.exit(1)

    reward_config = RewardConfig()
    winrate = np.full((n, n), np.nan)
    total_wins = [0] * n
    total_games = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            match_seed = args.seed + (i * 997 + j) % (2**20)
            w1, draws, w2 = play_match_batched(
                agents[i], agents[j],
                num_games=args.num_games,
                batch_size=min(args.batch_size, args.num_games),
                seed=match_seed,
                randomize_first_player=True,
                reward_config=reward_config,
                deterministic=args.epsilon == 0,
            )
            total = w1 + draws + w2
            assert total == args.num_games
            winrate[i, j] = w1 / args.num_games
            winrate[j, i] = w2 / args.num_games
            total_wins[i] += w1
            total_wins[j] += w2
            total_games[i] += args.num_games
            total_games[j] += args.num_games
            print(f"  {names[i]} vs {names[j]}: {w1}-{draws}-{w2}  (wr {w1/args.num_games:.2%} / {w2/args.num_games:.2%})")

    avg_wr = [total_wins[i] / total_games[i] if total_games[i] else 0 for i in range(n)]
    order = sorted(range(n), key=lambda i: -avg_wr[i])

    print("\n--- Top (by avg winrate) ---")
    for r, i in enumerate(order, 1):
        games = total_games[i]
        wins = total_wins[i]
        wr = avg_wr[i]
        print(f"  {r}. {names[i]}: wins={wins}/{games}  avg_wr={wr:.2%}")

    out_path = args.out or (args.runs_dir.resolve() / ("tournament_manual.csv" if args.manual else "tournament_winrate.csv"))
    out_path = out_path.resolve()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row\\col"] + names)
        for i in range(n):
            row = [f"{winrate[i, j]:.4f}" if not np.isnan(winrate[i, j]) else "" for j in range(n)]
            w.writerow([names[i]] + row)
    print(f"\nWinrate matrix saved to {out_path}")


if __name__ == "__main__":
    main()

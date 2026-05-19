#!/usr/bin/env python3
"""Benchmark MiniBG self-play: 1 process × N games vs W workers × (N/W) games (no replays)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.envs  # noqa: F401

from src.envs import RewardConfig
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint
from src.registry import make_game
from src.training.trainer import StartPolicy
from src.utils.match import play_single_game, resolve_opening_agent_token


def _play_selfplay_batch(
    ck: Path,
    n: int,
    seed: int,
    mg: dict,
    device: str,
) -> int:
    agent = load_training_agent_checkpoint(ck, device=device, seed=seed)
    opponent = load_training_agent_checkpoint(ck, device=device, seed=seed + 913_123)
    for a in (agent, opponent):
        if hasattr(a, "eval"):
            a.eval()
        if hasattr(a, "epsilon"):
            a.epsilon = 0.0

    stem = ck.stem
    match_seed = seed + hash(stem) % (2**16) + hash("self") % (2**16)
    mg_base = dict(mg)

    for g in range(n):
        game_seed = match_seed + g
        inner_start = StartPolicy.RANDOM
        agent_token = resolve_opening_agent_token(inner_start, seed=game_seed)
        env_g = make_game("minibg", reward_config=RewardConfig(), **mg_base)
        for participant in (agent, opponent):
            if hasattr(participant, "set_env"):
                participant.set_env(env_g)
        play_single_game(
            env_g,
            agent,
            opponent,
            start_policy=inner_start,
            random_opening_config=None,
            deterministic_agent=True,
            deterministic_opponent=True,
            seed=game_seed,
        )
    return n


def _worker(args: tuple) -> int:
    import torch

    torch.set_num_threads(1)
    return _play_selfplay_batch(*args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--total-games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--battle-damage-shaping", type=float, default=0.06)
    ap.add_argument(
        "--serial-only",
        action="store_true",
        help="Only run single-process benchmark (default PyTorch thread count).",
    )
    ap.add_argument(
        "--parallel-only",
        action="store_true",
        help="Only run multi-process benchmark (torch.set_num_threads(1) per worker).",
    )
    ap.add_argument(
        "--games-per-worker",
        type=int,
        default=None,
        help="If set, total-games = workers * games-per-worker (overrides --total-games).",
    )
    args = ap.parse_args()

    ck = args.checkpoint.resolve()
    if not ck.is_file():
        raise SystemExit(f"checkpoint not found: {ck}")
    workers = int(args.workers)
    if args.games_per_worker is not None:
        per_worker = int(args.games_per_worker)
        total = workers * per_worker
    else:
        total = int(args.total_games)
        per_worker = total // workers
    if total <= 0 or workers <= 0:
        raise SystemExit("total-games and workers must be positive")
    if args.games_per_worker is None and total % workers != 0:
        raise SystemExit(f"total-games ({total}) must be divisible by workers ({workers})")
    if args.serial_only and args.parallel_only:
        raise SystemExit("use at most one of --serial-only / --parallel-only")

    mg = {"battle_damage_shaping": float(args.battle_damage_shaping)}
    batch_args = (ck, total, args.seed, mg, args.device)
    worker_args = [
        (ck, per_worker, args.seed + w * 10_000, mg, args.device) for w in range(workers)
    ]

    import torch

    n_threads = torch.get_num_threads()
    print(
        f"checkpoint={ck.name}  device={args.device}  workers={workers}  "
        f"games/worker={per_worker}  total={total}  torch_threads={n_threads}  (no replays)",
        flush=True,
    )

    serial_s = None
    if not args.parallel_only:
        t0 = time.perf_counter()
        _play_selfplay_batch(*batch_args)
        serial_s = time.perf_counter() - t0
        print()
        print("=== serial (default torch threads) ===")
        print(
            f"{total} games in 1 process -> {serial_s:.2f}s  "
            f"({serial_s / total:.3f}s/game, {total / serial_s:.2f} games/s)"
        )
        if args.serial_only:
            return

    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        counts = pool.map(_worker, worker_args)
    parallel_s = time.perf_counter() - t0

    print()
    print("=== parallel (torch.set_num_threads(1) per worker) ===")
    print(
        f"{sum(counts)} games in {workers} processes ({per_worker} each) -> "
        f"{parallel_s:.2f}s  ({parallel_s / total:.3f}s/game, {total / parallel_s:.2f} games/s)"
    )
    if serial_s is not None:
        print(f"speedup (serial/parallel): {serial_s / parallel_s:.2f}x", flush=True)


if __name__ == "__main__":
    main()

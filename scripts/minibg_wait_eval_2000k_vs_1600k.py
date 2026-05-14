#!/usr/bin/env python3
"""Ждёт чекпоинт 2M, затем 500 игр 2000k vs 1600k (случайный первый ход), пишет винрейт в run-dir."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.envs  # noqa: F401

from src.envs import RewardConfig
from src.registry import make_game
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint
from src.training.trainer import StartPolicy
from src.utils.match import play_single_game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="minibg_ppo_structured")
    ap.add_argument("--opp-step", type=int, default=1_600_000)
    ap.add_argument("--num-games", type=int, default=500)
    ap.add_argument("--poll-sec", type=float, default=20.0)
    ap.add_argument("--wait-max-sec", type=float, default=7200.0)
    ap.add_argument("--seed", type=int, default=204_816)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--battle-damage-shaping", type=float, default=0.06)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    ckpt_dir = run_dir / "checkpoints"
    opp_path = ckpt_dir / f"{args.prefix}_{args.opp_step}.pt"
    if not opp_path.is_file():
        print(f"Missing opponent: {opp_path}", file=sys.stderr)
        raise SystemExit(1)

    p200 = ckpt_dir / f"{args.prefix}_2000000.pt"
    pfinal = ckpt_dir / f"{args.prefix}_final.pt"
    st_path = run_dir / "status.json"
    t0 = time.monotonic()
    new_path: Path | None = None
    while time.monotonic() - t0 < args.wait_max_sec:
        if p200.is_file():
            new_path = p200
            break
        if st_path.is_file():
            try:
                meta = json.loads(st_path.read_text(encoding="utf-8"))
                step = int(meta.get("step", 0))
                running = meta.get("status") == "running"
                if not running and step >= 2_000_000 and pfinal.is_file():
                    new_path = pfinal
                    break
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass
        print(
            f"[wait {int(time.monotonic() - t0)}s] жду {p200.name} …",
            flush=True,
        )
        time.sleep(args.poll_sec)

    if new_path is None:
        if p200.is_file():
            new_path = p200
        elif pfinal.is_file():
            new_path = pfinal
    if new_path is None or not new_path.is_file():
        print("Таймаут: нет 2M и нет подходящего final после остановки.", file=sys.stderr)
        raise SystemExit(2)

    print(f"Новый чекпоинт: {new_path.name}", flush=True)

    agent_new = load_training_agent_checkpoint(new_path, device=args.device, seed=args.seed)
    agent_opp = load_training_agent_checkpoint(opp_path, device=args.device, seed=args.seed + 1)

    w_new = w_opp = dr = 0
    for g in range(int(args.num_games)):
        env = make_game(
            "minibg",
            reward_config=RewardConfig(),
            battle_damage_shaping=float(args.battle_damage_shaping),
        )
        for p in (agent_new, agent_opp):
            if hasattr(p, "set_env"):
                p.set_env(env)
        res = play_single_game(
            env,
            agent_new,
            agent_opp,
            start_policy=StartPolicy.RANDOM,
            random_opening_config=None,
            deterministic_agent=True,
            deterministic_opponent=True,
            seed=int(args.seed) + g,
        )
        r = res["reward"]
        if r == 1:
            w_new += 1
        elif r == -1:
            w_opp += 1
        else:
            dr += 1
        if (g + 1) % 50 == 0:
            print(f"  … {g+1}/{args.num_games}", flush=True)

    n = int(args.num_games)
    wr = 100.0 * w_new / n if n else 0.0
    out = {
        "new_checkpoint": new_path.name,
        "opponent_checkpoint": opp_path.name,
        "num_games": n,
        "wins_new": w_new,
        "wins_opponent": w_opp,
        "draws": dr,
        "winrate_new_pct": round(wr, 3),
        "winrate_new": w_new / n if n else 0.0,
    }
    jpath = run_dir / "eval_2000k_vs_1600k_500.json"
    jpath.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = run_dir / "eval_2000k_vs_1600k_500.md"
    md.write_text(
        f"# 500 игр: {new_path.stem} vs {opp_path.stem}\n\n"
        f"- Побед **новой** политики: **{w_new}** / {n} (**{wr:.2f}%**)\n"
        f"- Побед оппонента: {w_opp}\n"
        f"- Ничьих: {dr}\n\n"
        f"Параметры: случайный первый ход, deterministic act, "
        f"battle_damage_shaping={args.battle_damage_shaping}, seed base {args.seed}.\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))
    print(f"Wrote {jpath}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()

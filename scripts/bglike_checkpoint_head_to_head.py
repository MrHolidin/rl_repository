#!/usr/bin/env python3
"""BGLike 4v4 checkpoint head-to-head: average placement by team."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.placement import placement_for_seat
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.evaluation.eval_checkpoints import find_checkpoints, load_training_agent_checkpoint

_T975 = {2: 4.302653, 5: 2.570582, 10: 2.228139, 20: 2.0860, 40: 2.0211, 80: 1.9944}


def t_crit_975(n: int) -> float:
    if n < 2:
        return float("nan")
    return _T975.get(n - 1, 1.96)


def _resolve_ckpt(checkpoint_dir: Path, prefix: str, step: int | None, path: Path | None) -> Tuple[Path, int]:
    if path is not None:
        p = path.resolve()
        from src.evaluation.eval_checkpoints import _step_from_filename

        s = _step_from_filename(p.name)
        return p, int(s if s is not None else -1)
    found = find_checkpoints(checkpoint_dir.resolve(), prefix=prefix)
    if not found:
        raise SystemExit(f"No checkpoints in {checkpoint_dir}")
    if step is not None:
        ck, st = min(found, key=lambda x: abs(x[1] - step))
        return ck, st
    return found[-1]


def run_head_to_head(
    *,
    ck_a: Path,
    ck_b: Path,
    seats_a: Sequence[int],
    seats_b: Sequence[int],
    num_games: int,
    seed: int,
    device: str,
) -> List[dict]:
    agent_a = load_training_agent_checkpoint(ck_a, device=device, seed=seed)
    agent_b = load_training_agent_checkpoint(ck_b, device=device, seed=seed + 1)
    for ag in (agent_a, agent_b):
        if hasattr(ag, "eval"):
            ag.eval()
    agents = {}
    for s in seats_a:
        agents[s] = agent_a
    for s in seats_b:
        agents[s] = agent_b
    learned = tuple(sorted(set(seats_a) | set(seats_b)))
    configs = lobby_from_learned_seats(learned, agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=learned,
        training_seats=learned,
        seed=seed,
    )
    games: List[dict] = []
    for g in range(num_games):
        env.reset(seed=seed + g)
        env.drain_until_lobby_done(deterministic=True)
        st = env.state
        placements = {s: placement_for_seat(st, s) for s in range(8)}
        games.append(
            {
                "game": g,
                "seed": seed + g,
                "winner": st.winner,
                "placements": placements,
                "team_a_placements": [placements[s] for s in seats_a],
                "team_b_placements": [placements[s] for s in seats_b],
            }
        )
    return games


def summarize_placements(values: List[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci95_half": float("nan")}
    mu = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mu, "ci95_half": float("nan")}
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    half = t_crit_975(n) * math.sqrt(var / n)
    return {"n": n, "mean": mu, "ci95_half": half}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "runs/bglike/dist_ppo_005/checkpoints",
    )
    ap.add_argument("--prefix", type=str, default="dist_bglike_ppo")
    ap.add_argument("--ckpt-a", type=Path, default=None, help="Team A checkpoint path")
    ap.add_argument("--ckpt-b", type=Path, default=None, help="Team B checkpoint path")
    ap.add_argument("--step-a", type=int, default=5_000_000, help="Nearest step for team A")
    ap.add_argument("--step-b", type=int, default=2_500_000, help="Nearest step for team B")
    ap.add_argument("--seats-a", type=str, default="0,1,2,3", help="Seats for team A")
    ap.add_argument("--seats-b", type=str, default="4,5,6,7", help="Seats for team B")
    ap.add_argument("--label-a", type=str, default="5000k")
    ap.add_argument("--label-b", type=str, default="2500k")
    ap.add_argument("--num-games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    seats_a = tuple(int(x.strip()) for x in args.seats_a.split(",") if x.strip())
    seats_b = tuple(int(x.strip()) for x in args.seats_b.split(",") if x.strip())
    if set(seats_a) & set(seats_b):
        raise SystemExit("seat sets must not overlap")
    if len(seats_a) != 4 or len(seats_b) != 4:
        raise SystemExit("expected 4 seats per team")

    ck_a, step_a = _resolve_ckpt(args.checkpoint_dir, args.prefix, args.step_a, args.ckpt_a)
    ck_b, step_b = _resolve_ckpt(args.checkpoint_dir, args.prefix, args.step_b, args.ckpt_b)

    games = run_head_to_head(
        ck_a=ck_a,
        ck_b=ck_b,
        seats_a=seats_a,
        seats_b=seats_b,
        num_games=args.num_games,
        seed=args.seed,
        device=args.device,
    )
    all_a = [float(p) for g in games for p in g["team_a_placements"]]
    all_b = [float(p) for g in games for p in g["team_b_placements"]]
    sum_a = summarize_placements(all_a)
    sum_b = summarize_placements(all_b)
    wins_a = sum(1 for g in games if g["winner"] in seats_a)
    wins_b = sum(1 for g in games if g["winner"] in seats_b)

    print(f"team A ({args.label_a}): {ck_a.name}  seats={seats_a}")
    print(f"team B ({args.label_b}): {ck_b.name}  seats={seats_b}")
    print(f"games: {args.num_games}  seed: {args.seed}")
    print(
        f"mean placement {args.label_a}: {sum_a['mean']:.3f} "
        f"(95% CI ±{sum_a['ci95_half']:.3f}, n={sum_a['n']})  "
        f"[1=best, 8=worst]"
    )
    print(
        f"mean placement {args.label_b}: {sum_b['mean']:.3f} "
        f"(95% CI ±{sum_b['ci95_half']:.3f}, n={sum_b['n']})"
    )
    print(f"lobby wins: {args.label_a} {wins_a}/{args.num_games}, {args.label_b} {wins_b}/{args.num_games}")

    payload = {
        "team_a": {"label": args.label_a, "checkpoint": ck_a.name, "step": step_a, "seats": seats_a},
        "team_b": {"label": args.label_b, "checkpoint": ck_b.name, "step": step_b, "seats": seats_b},
        "num_games": args.num_games,
        "seed": args.seed,
        "summary": {
            "mean_placement_a": sum_a,
            "mean_placement_b": sum_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
        },
        "games": games,
    }
    out = args.out_json
    if out is None:
        out = (
            args.checkpoint_dir.parent
            / f"head_to_head_{args.label_a}_vs_{args.label_b}_{args.num_games}g.json"
        )
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run N BGLike self-play lobbies from a checkpoint; report end-board tier/stats/races."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401
from src.bg_core.minion import Minion, Race
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.bglike.state import BGLikeState
from src.evaluation.eval_checkpoints import find_checkpoints, load_training_agent_checkpoint

_T975 = {2: 4.302653, 5: 2.570582, 10: 2.228139, 20: 2.0860, 30: 2.0423}


def t_crit_975(n: int) -> float:
    if n < 2:
        return float("nan")
    return _T975.get(n - 1, 1.96)


def _race_label(m: Minion) -> str:
    if m.race is None:
        return "NONE"
    return m.race.name


def _board_stat_sum(board: Tuple[Minion, ...]) -> int:
    return sum(int(m.raw_attack) + int(m.max_health) for m in board)


def final_boards_by_seat(state: BGLikeState) -> Dict[int, Tuple[Tuple[Minion, ...], int]]:
    out: Dict[int, Tuple[Tuple[Minion, ...], int]] = {}
    for snap in state.eliminated:
        out[snap.seat] = (snap.last_board, snap.tavern_tier)
    for seat in state.alive:
        p = state.players[seat]
        out[seat] = (tuple(p.board), p.tavern_tier)
    return out


def run_games(
    *,
    checkpoint: Path,
    num_games: int,
    seed: int,
    device: str,
) -> List[dict]:
    agent = load_training_agent_checkpoint(checkpoint, device=device, seed=seed)
    if hasattr(agent, "eval"):
        agent.eval()
    agents = {s: agent for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=seed,
    )
    games: List[dict] = []
    for g in range(num_games):
        env.reset(seed=seed + g)
        env.drain_until_lobby_done(deterministic=True)
        st = env.state
        boards = final_boards_by_seat(st)
        per_seat = []
        for seat in range(8):
            board, tier = boards[seat]
            races = Counter(_race_label(m) for m in board)
            per_seat.append(
                {
                    "seat": seat,
                    "tier": tier,
                    "board_size": len(board),
                    "stat_sum": _board_stat_sum(board),
                    "races": dict(races),
                }
            )
        games.append(
            {
                "game": g,
                "seed": seed + g,
                "winner": st.winner,
                "round": st.round_number,
                "combat_round": st.combat_round,
                "seats": per_seat,
            }
        )
    return games


def summarize(games: List[dict]) -> dict:
    tiers: List[float] = []
    stat_sums: List[float] = []
    race_counts: Counter[str] = Counter()
    n_minions = 0
    for game in games:
        for seat in game["seats"]:
            tiers.append(float(seat["tier"]))
            stat_sums.append(float(seat["stat_sum"]))
            for race, c in seat["races"].items():
                race_counts[race] += int(c)
                n_minions += int(c)

    def mean_ci(vals: List[float]) -> dict:
        n = len(vals)
        if n == 0:
            return {"n": 0, "mean": float("nan"), "ci95_half": float("nan")}
        mu = sum(vals) / n
        if n < 2:
            return {"n": n, "mean": mu, "ci95_half": float("nan")}
        var = sum((x - mu) ** 2 for x in vals) / (n - 1)
        half = t_crit_975(n) * math.sqrt(var / n)
        return {"n": n, "mean": mu, "ci95_half": half}

    tier_s = mean_ci(tiers)
    stat_s = mean_ci(stat_sums)
    race_total = sum(race_counts.values()) or 1
    race_pct = {k: 100.0 * v / race_total for k, v in sorted(race_counts.items())}
    return {
        "num_games": len(games),
        "player_samples": len(tiers),
        "mean_tavern_tier": tier_s,
        "mean_board_stat_sum": stat_s,
        "race_minion_counts": dict(sorted(race_counts.items())),
        "race_minion_pct": race_pct,
        "total_minions_on_final_boards": n_minions,
    }


def _print_report(checkpoint: Path, summary: dict) -> None:
    print(f"checkpoint: {checkpoint.name}")
    print(f"games: {summary['num_games']}  player-samples: {summary['player_samples']}")
    mt = summary["mean_tavern_tier"]
    ms = summary["mean_board_stat_sum"]
    print(
        f"mean tavern tier: {mt['mean']:.3f} "
        f"(95% CI ±{mt['ci95_half']:.3f}, n={mt['n']})"
    )
    print(
        f"mean board stat sum (atk+hp): {ms['mean']:.1f} "
        f"(95% CI ±{ms['ci95_half']:.1f}, n={ms['n']})"
    )
    print(f"minions on final boards: {summary['total_minions_on_final_boards']}")
    print("race distribution (minions on final boards):")
    for race, pct in summary["race_minion_pct"].items():
        cnt = summary["race_minion_counts"][race]
        print(f"  {race:12s}  {cnt:4d}  ({pct:5.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "runs/bglike/dist_ppo_005/checkpoints",
    )
    ap.add_argument("--checkpoint", type=Path, default=None, help="Explicit .pt (else latest)")
    ap.add_argument("--prefix", type=str, default="dist_bglike_ppo")
    ap.add_argument("--num-games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write per-game + summary JSON (default: run_dir/board_stats_<stem>.json)",
    )
    args = ap.parse_args()

    if args.checkpoint is not None:
        ck_path = args.checkpoint.resolve()
    else:
        found = find_checkpoints(args.checkpoint_dir.resolve(), prefix=args.prefix)
        if not found:
            raise SystemExit(f"No checkpoints in {args.checkpoint_dir}")
        ck_path = found[-1][0]

    games = run_games(
        checkpoint=ck_path,
        num_games=args.num_games,
        seed=args.seed,
        device=args.device,
    )
    summary = summarize(games)
    payload = {
        "checkpoint": ck_path.name,
        "seed": args.seed,
        "summary": summary,
        "games": games,
    }
    out = args.out_json
    if out is None:
        out = ck_path.parent.parent / f"board_stats_{ck_path.stem}.json"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_report(ck_path, summary)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()

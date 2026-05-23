#!/usr/bin/env python3
"""8p FFA lobby: 2 seats per checkpoint milestone, aggregate stats."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401
import src.envs.bglike.lobby_env as lobby_mod
from src.bg_core.minion import Minion
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.placement import placement_for_seat
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.bglike.state import BGLikeState
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint

_T975 = {2: 4.302653, 5: 2.570582, 10: 2.228139, 20: 2.0860, 40: 2.0211, 80: 1.9949, 160: 1.9745}


def t_crit_975(n: int) -> float:
    if n < 2:
        return float("nan")
    return _T975.get(n - 1, 1.96)


def mean_ci(vals: Sequence[float]) -> dict:
    n = len(vals)
    if not n:
        return {"n": 0, "mean": float("nan"), "ci95_half": float("nan")}
    mu = sum(vals) / n
    if n < 2:
        return {"n": n, "mean": mu, "ci95_half": float("nan")}
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    return {"n": n, "mean": mu, "ci95_half": t_crit_975(n) * math.sqrt(var / n)}


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


def placements_for_state(state: BGLikeState) -> Dict[int, int]:
    try:
        return {s: placement_for_seat(state, s) for s in range(8)}
    except ValueError:
        pass
    if not state.done:
        raise ValueError("lobby not finished")
    out: Dict[int, int] = {}
    if state.winner is not None:
        out[int(state.winner)] = 1
    for i, snap in enumerate(state.eliminated):
        if snap.seat not in out:
            out[int(snap.seat)] = 8 - i
    alive_rest = [s for s in state.alive if s != state.winner]
    alive_rest.sort(key=lambda s: (-state.players[s].health, s))
    next_place = 2
    for seat in alive_rest:
        if seat not in out:
            out[int(seat)] = next_place
            next_place += 1
    if len(out) != 8:
        raise ValueError(f"unresolved placements: missing {set(range(8)) - set(out)}")
    return out


def run_eval(
    milestones: Sequence[Tuple[str, Path, int, Tuple[int, ...]]],
    *,
    num_games: int,
    seed: int,
    device: str,
    drain_steps: int,
) -> List[dict]:
    agents_by_label: Dict[str, object] = {}
    seat_to_label: Dict[int, str] = {}
    for i, (label, ckpt, step, seats) in enumerate(milestones):
        agent = load_training_agent_checkpoint(ckpt, device=device, seed=seed + i * 17)
        if hasattr(agent, "eval"):
            agent.eval()
        agents_by_label[label] = agent
        for s in seats:
            seat_to_label[int(s)] = label

    agents: Dict[int, object] = {}
    for label, _, _, seats in milestones:
        for s in seats:
            agents[int(s)] = agents_by_label[label]

    learned = tuple(sorted(agents))
    configs = lobby_from_learned_seats(learned, agent_by_seat=agents)
    env = BGLobbyEnv(configs, learned_seats=learned, training_seats=learned, seed=seed)

    old_cap = lobby_mod.MAX_DRAIN_STEPS
    lobby_mod.MAX_DRAIN_STEPS = int(drain_steps)
    games: List[dict] = []
    try:
        for g in range(num_games):
            game_seed = seed + g
            env.reset(seed=game_seed)
            env.drain_until_lobby_done(deterministic=True)
            st = env.state
            if not env.lobby_done:
                raise RuntimeError(f"lobby not done: alive={st.alive}")
            boards = final_boards_by_seat(st)
            placements = placements_for_state(st)
            winner_label = seat_to_label[int(st.winner)] if st.winner is not None else None
            seat_rows = []
            for seat in range(8):
                board, tier = boards[seat]
                races = Counter(_race_label(m) for m in board)
                seat_rows.append(
                    {
                        "seat": seat,
                        "cohort": seat_to_label[seat],
                        "placement": placements[seat],
                        "tier": tier,
                        "stat_sum": _board_stat_sum(board),
                        "races": dict(races),
                    }
                )
            games.append(
                {
                    "game": g,
                    "seed": game_seed,
                    "winner": st.winner,
                    "winner_cohort": winner_label,
                    "seats": seat_rows,
                }
            )
            if (g + 1) % 10 == 0 or g == 0:
                print(
                    f"  game {g + 1}/{num_games} winner=P{st.winner} ({winner_label})",
                    flush=True,
                )
    finally:
        lobby_mod.MAX_DRAIN_STEPS = old_cap
    return games


def summarize_cohort(games: List[dict], cohort: str) -> dict:
    stats: List[float] = []
    places: List[float] = []
    tiers: List[float] = []
    top4 = 0
    races: Counter[str] = Counter()
    wins = 0
    for game in games:
        if game["winner_cohort"] == cohort:
            wins += 1
        for seat in game["seats"]:
            if seat["cohort"] != cohort:
                continue
            stats.append(float(seat["stat_sum"]))
            places.append(float(seat["placement"]))
            tiers.append(float(seat["tier"]))
            if seat["placement"] <= 4:
                top4 += 1
            races.update(seat.get("races") or {})
    tot = sum(races.values()) or 1
    return {
        "cohort": cohort,
        "lobby_wins": wins,
        "num_games": len(games),
        "top4_rate": top4 / max(len(places), 1),
        "mean_board_stat_sum": mean_ci(stats),
        "mean_placement": mean_ci(places),
        "mean_tavern_tier": mean_ci(tiers),
        "race_minion_counts": dict(sorted(races.items())),
        "race_minion_pct": {k: 100.0 * v / tot for k, v in sorted(races.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="dist_bglike_ppo")
    ap.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1_000_000, 2_000_000, 3_000_000, 5_000_000],
    )
    ap.add_argument("--seats-per-cohort", type=int, default=2)
    ap.add_argument("--num-games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--drain-steps", type=int, default=20_000)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    ckpt_dir = args.run_dir.resolve() / "checkpoints"
    ckpts = {
        int(p.stem.rsplit("_", 1)[-1]): p
        for p in ckpt_dir.glob(f"{args.prefix}_*.pt")
        if p.stem.rsplit("_", 1)[-1].isdigit()
    }
    if not ckpts:
        raise SystemExit(f"no checkpoints in {ckpt_dir}")

    milestones: List[Tuple[str, Path, int, Tuple[int, ...]]] = []
    seat = 0
    for target in args.steps:
        step = min(ckpts, key=lambda s: abs(s - target))
        label = f"{target // 1_000_000}M"
        seats = tuple(range(seat, seat + args.seats_per_cohort))
        seat += args.seats_per_cohort
        milestones.append((label, ckpts[step], step, seats))
        print(f"  {label}: step {step} seats {list(seats)}", flush=True)

    out_dir = (args.out_dir or REPO_ROOT / "runs/bglike" / f"milestone_{args.run_dir.name}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== milestone lobby eval  {args.num_games} games ===", flush=True)
    games = run_eval(
        milestones,
        num_games=args.num_games,
        seed=args.seed,
        device=args.device,
        drain_steps=args.drain_steps,
    )

    summaries = {label: summarize_cohort(games, label) for label, _, _, _ in milestones}
    payload = {
        "run_dir": str(args.run_dir.resolve()),
        "num_games": args.num_games,
        "seed": args.seed,
        "milestones": [
            {
                "label": label,
                "step": step,
                "checkpoint": ckpt.name,
                "seats": list(seats),
            }
            for label, ckpt, step, seats in milestones
        ],
        "summaries": summaries,
        "games": games,
    }
    raw = out_dir / "raw.json"
    raw.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    hdr = ["cohort", "step", "lobby_wins", "top4_rate", "mean_stat", "stat_ci", "mean_place", "place_ci", "mean_tier", "tier_ci"]
    lines = [",".join(hdr)]
    for label, _, step, _ in milestones:
        sm = summaries[label]
        ms, mp, mt = sm["mean_board_stat_sum"], sm["mean_placement"], sm["mean_tavern_tier"]
        lines.append(
            ",".join(
                str(x)
                for x in (
                    label,
                    step,
                    sm["lobby_wins"],
                    f"{sm['top4_rate']:.4f}",
                    f"{ms['mean']:.2f}",
                    f"{ms['ci95_half']:.2f}",
                    f"{mp['mean']:.2f}",
                    f"{mp['ci95_half']:.2f}",
                    f"{mt['mean']:.2f}",
                    f"{mt['ci95_half']:.2f}",
                )
            )
        )
    csv_path = out_dir / "summary.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {csv_path}  and {raw}", flush=True)
    for label, _, step, _ in milestones:
        sm = summaries[label]
        ms, mp, mt = sm["mean_board_stat_sum"], sm["mean_placement"], sm["mean_tavern_tier"]
        print(
            f"  {label}@{step}: wins={sm['lobby_wins']}/{args.num_games}  "
            f"top4={sm['top4_rate']:.1%}  stat={ms['mean']:.1f}±{ms['ci95_half']:.1f}  "
            f"place={mp['mean']:.2f}±{mp['ci95_half']:.2f}  tier={mt['mean']:.2f}±{mt['ci95_half']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

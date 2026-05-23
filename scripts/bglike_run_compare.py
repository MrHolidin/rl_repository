#!/usr/bin/env python3
"""4v4 head-to-head: matched 023 vs 024 checkpoints, replays + board/placement stats."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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
from src.envs.bglike.replay import attach_replay, close_replay
from src.envs.bglike.replay_render import render_jsonl_file
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.bglike.state import BGLikeState
from src.evaluation.eval_checkpoints import find_checkpoints, load_training_agent_checkpoint

_T975 = {2: 4.302653, 5: 2.570582, 10: 2.228139, 20: 2.0860, 40: 2.0211, 160: 1.9745}


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


def _checkpoint_maps(run23: Path, run24: Path, prefix: str) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    ck23 = {s: p for p, s in find_checkpoints(run23 / "checkpoints", prefix=prefix)}
    ck24 = {s: p for p, s in find_checkpoints(run24 / "checkpoints", prefix=prefix)}
    if not ck23 or not ck24:
        raise SystemExit("missing checkpoints")
    return ck23, ck24


def resolve_pair(run23: Path, run24: Path, target_step: int, prefix: str) -> Tuple[Path, int, Path, int]:
    ck23, ck24 = _checkpoint_maps(run23, run24, prefix)
    s23 = min(ck23, key=lambda s: abs(s - target_step))
    s24 = min(ck24, key=lambda s: abs(s - target_step))
    return ck23[s23], s23, ck24[s24], s24


def latest_pairs(run23: Path, run24: Path, n: int, prefix: str) -> List[Tuple[Path, int, Path, int]]:
    """N newest 024 checkpoints, each paired with nearest 023 step."""
    ck23, ck24 = _checkpoint_maps(run23, run24, prefix)
    steps24 = sorted(ck24)[-n:]
    pairs: List[Tuple[Path, int, Path, int]] = []
    for s24 in steps24:
        s23 = min(ck23, key=lambda s: abs(s - s24))
        pairs.append((ck23[s23], s23, ck24[s24], s24))
    return pairs


def _step_from_ckpt(path: Path) -> int:
    tail = path.stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 10**18


def run_pair(
    *,
    ck_a: Path,
    step_a: int,
    ck_b: Path,
    step_b: int,
    label_a: str,
    label_b: str,
    seats_a: Sequence[int],
    seats_b: Sequence[int],
    num_games: int,
    seed: int,
    device: str,
    replay_dir: Path,
    drain_steps: int,
) -> List[dict]:
    replay_dir.mkdir(parents=True, exist_ok=True)
    agent_a = load_training_agent_checkpoint(ck_a, device=device, seed=seed)
    agent_b = load_training_agent_checkpoint(ck_b, device=device, seed=seed + 1)
    for ag in (agent_a, agent_b):
        if hasattr(ag, "eval"):
            ag.eval()
    agents: Dict[int, object] = {}
    for s in seats_a:
        agents[s] = agent_a
    for s in seats_b:
        agents[s] = agent_b
    learned = tuple(sorted(set(seats_a) | set(seats_b)))
    configs = lobby_from_learned_seats(learned, agent_by_seat=agents)
    env = BGLobbyEnv(configs, learned_seats=learned, training_seats=learned, seed=seed)

    old_cap = lobby_mod.MAX_DRAIN_STEPS
    lobby_mod.MAX_DRAIN_STEPS = int(drain_steps)
    games: List[dict] = []
    tag = f"{label_a}_{step_a}__vs__{label_b}_{step_b}"
    try:
        for g in range(num_games):
            game_seed = seed + g
            jpath = replay_dir / f"{tag}__{g:04d}.jsonl"
            meta = {
                "mode": "4v4_head_to_head",
                f"{label_a}_checkpoint": ck_a.name,
                f"{label_a}_step": step_a,
                f"{label_b}_checkpoint": ck_b.name,
                f"{label_b}_step": step_b,
                f"seats_{label_a}": list(seats_a),
                f"seats_{label_b}": list(seats_b),
                "game_index": g,
                "seed": game_seed,
            }
            attach_replay(env, jpath, meta)
            try:
                env.reset(seed=game_seed)
                env.drain_until_lobby_done(deterministic=True)
                st = env.state
                if not env.lobby_done:
                    raise RuntimeError(f"lobby not done: alive={st.alive}")
                boards = final_boards_by_seat(st)
                placements = placements_for_state(st)
                seat_rows = []
                for seat in range(8):
                    board, tier = boards[seat]
                    races = Counter(_race_label(m) for m in board)
                    team = label_a if seat in seats_a else label_b
                    seat_rows.append(
                        {
                            "seat": seat,
                            "team": team,
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
                        "replay": str(jpath),
                        "winner": st.winner,
                        "winner_team": label_a if st.winner in seats_a else label_b,
                        "seats": seat_rows,
                    }
                )
            finally:
                close_replay(env)
            jpath.with_suffix(".txt").write_text(render_jsonl_file(jpath), encoding="utf-8")
            print(
                f"  {tag} game {g + 1}/{num_games} winner=P{st.winner} ({games[-1]['winner_team']})",
                flush=True,
            )
    finally:
        lobby_mod.MAX_DRAIN_STEPS = old_cap
    return games


def summarize_team(games: List[dict], team: str) -> dict:
    stats: List[float] = []
    places: List[float] = []
    races: Counter[str] = Counter()
    wins = 0
    for game in games:
        if game["winner_team"] == team:
            wins += 1
        for seat in game["seats"]:
            if seat["team"] != team:
                continue
            stats.append(float(seat["stat_sum"]))
            places.append(float(seat["placement"]))
            races.update(seat["races"])

    def mean_ci(vals: List[float]) -> dict:
        n = len(vals)
        if n == 0:
            return {"n": 0, "mean": float("nan"), "ci95_half": float("nan")}
        mu = sum(vals) / n
        if n < 2:
            return {"n": n, "mean": mu, "ci95_half": float("nan")}
        var = sum((x - mu) ** 2 for x in vals) / (n - 1)
        return {"n": n, "mean": mu, "ci95_half": t_crit_975(n) * math.sqrt(var / n)}

    stat_s = mean_ci(stats)
    place_s = mean_ci(places)
    tot = sum(races.values()) or 1
    return {
        "team": team,
        "wins": wins,
        "num_games": len(games),
        "mean_board_stat_sum": stat_s,
        "mean_placement": place_s,
        "race_minion_counts": dict(sorted(races.items())),
        "race_minion_pct": {k: 100.0 * v / tot for k, v in sorted(races.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-023", type=Path, default=REPO_ROOT / "runs/bglike/dist_ppo_023")
    ap.add_argument("--run-024", type=Path, default=REPO_ROOT / "runs/bglike/dist_ppo_024")
    ap.add_argument("--prefix", type=str, default="dist_bglike_ppo")
    ap.add_argument(
        "--targets",
        type=int,
        nargs="*",
        default=None,
        help="Explicit target steps (nearest ckpt in each run). Ignored if --latest is set.",
    )
    ap.add_argument(
        "--latest",
        type=int,
        default=1,
        help="Use N newest 024 checkpoints paired with nearest 023 (default 1 = single latest match).",
    )
    ap.add_argument("--num-games", type=int, default=40, help="4v4 lobby games (total replays)")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--drain-steps", type=int, default=20_000)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs/bglike/compare_023_vs_024")
    ap.add_argument("--ckpt-a", type=Path, default=None, help="Explicit team-A checkpoint (.pt)")
    ap.add_argument("--ckpt-b", type=Path, default=None, help="Explicit team-B checkpoint (.pt)")
    ap.add_argument("--label-a", type=str, default="023", help="Team A label (seats 0-3)")
    ap.add_argument("--label-b", type=str, default="024", help="Team B label (seats 4-7)")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seats_a = (0, 1, 2, 3)
    seats_b = (4, 5, 6, 7)

    pair_specs: List[Tuple[Path, int, Path, int, int, str, str]] = []
    if args.ckpt_a is not None or args.ckpt_b is not None:
        if args.ckpt_a is None or args.ckpt_b is None:
            raise SystemExit("--ckpt-a and --ckpt-b must be given together")
        ck_a = args.ckpt_a.resolve()
        ck_b = args.ckpt_b.resolve()
        if not ck_a.is_file() or not ck_b.is_file():
            raise SystemExit(f"checkpoint missing: {ck_a} or {ck_b}")
        pair_specs.append(
            (
                ck_a,
                _step_from_ckpt(ck_a),
                ck_b,
                _step_from_ckpt(ck_b),
                _step_from_ckpt(ck_b),
                args.label_a,
                args.label_b,
            )
        )
    elif args.targets:
        for t in args.targets:
            ck_a, s_a, ck_b, s_b = resolve_pair(args.run_023, args.run_024, t, args.prefix)
            pair_specs.append((ck_a, s_a, ck_b, s_b, t, args.label_a, args.label_b))
    else:
        for ck_a, s_a, ck_b, s_b in latest_pairs(
            args.run_023, args.run_024, int(args.latest), args.prefix
        ):
            pair_specs.append((ck_a, s_a, ck_b, s_b, s_b, args.label_a, args.label_b))

    all_rows: List[dict] = []
    for i, (ck_a, step_a, ck_b, step_b, target, label_a, label_b) in enumerate(pair_specs):
        pair_seed = args.seed + i * 10_000
        replay_dir = out_dir / "replays" / f"{label_a}_{step_a}_vs_{label_b}_{step_b}"
        print(
            f"\n=== pair {i + 1}/{len(pair_specs)} {label_a}@{step_a} vs {label_b}@{step_b} "
            f"(Δ{step_a - step_b:+d}) games={args.num_games}",
            flush=True,
        )
        games = run_pair(
            ck_a=ck_a,
            step_a=step_a,
            ck_b=ck_b,
            step_b=step_b,
            label_a=label_a,
            label_b=label_b,
            seats_a=seats_a,
            seats_b=seats_b,
            num_games=args.num_games,
            seed=pair_seed,
            device=args.device,
            replay_dir=replay_dir,
            drain_steps=args.drain_steps,
        )
        sum_a = summarize_team(games, label_a)
        sum_b = summarize_team(games, label_b)
        payload = {
            "target_step": target,
            label_a: {"step": step_a, "checkpoint": ck_a.name, "summary": sum_a},
            label_b: {"step": step_b, "checkpoint": ck_b.name, "summary": sum_b},
            "games": games,
        }
        raw = out_dir / "raw" / f"{label_a}_{step_a}_vs_{label_b}_{step_b}.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        for side, step, ck, sm, opp_step in (
            (label_a, step_a, ck_a.stem, sum_a, step_b),
            (label_b, step_b, ck_b.stem, sum_b, step_a),
        ):
            ms, mp = sm["mean_board_stat_sum"], sm["mean_placement"]
            row = {
                "run": side,
                "target_step": target,
                "step": step,
                "checkpoint": ck,
                "opponent_step": opp_step,
                "mean_stat_sum": ms["mean"],
                "stat_ci_half": ms["ci95_half"],
                "mean_placement": mp["mean"],
                "place_ci_half": mp["ci95_half"],
                "lobby_wins": sm["wins"],
                "n_games": sm["num_games"],
            }
            for race, pct in sm["race_minion_pct"].items():
                row[f"pct_{race}"] = pct
            all_rows.append(row)
            print(
                f"  {side}@{step}: Σ(atk+hp)={ms['mean']:.1f}±{ms['ci95_half']:.1f}  "
                f"place={mp['mean']:.2f}±{mp['ci95_half']:.2f}  wins={sm['wins']}/{args.num_games}",
                flush=True,
            )

    races = sorted({k for r in all_rows for k in r if k.startswith("pct_")})
    hdr = [
        "run", "target_step", "step", "checkpoint", "opponent_step",
        "mean_stat_sum", "stat_ci_half", "mean_placement", "place_ci_half",
        "lobby_wins", "n_games", *races,
    ]
    lines = [",".join(hdr)]
    for row in all_rows:
        lines.append(",".join(
            str(row.get(c, "")) if not isinstance(row.get(c), float) else f"{row.get(c):.4f}"
            for c in hdr
        ))
    csv_path = out_dir / "summary.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"\nWrote {csv_path}  ({len(pair_specs)} pairs × {args.num_games} games = "
        f"{len(pair_specs) * args.num_games} lobbies)",
        flush=True,
    )


if __name__ == "__main__":
    main()

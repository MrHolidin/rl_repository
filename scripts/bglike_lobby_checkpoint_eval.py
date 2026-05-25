#!/usr/bin/env python3
"""Run BGLike 8p self-play lobbies per checkpoint; save replays + aggregate stats."""

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

_T975 = {2: 4.302653, 5: 2.570582, 10: 2.228139, 20: 2.0860, 40: 2.0211, 80: 1.9944, 320: 1.96}


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


def placements_for_state(state: BGLikeState) -> Dict[int, int]:
    """All seat placements; handles max_rounds termination with multiple alive seats."""
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
        raise ValueError(f"could not resolve placements for seats: {sorted(set(range(8)) - set(out))}")
    return out


def final_boards_by_seat(state: BGLikeState) -> Dict[int, Tuple[Tuple[Minion, ...], int]]:
    out: Dict[int, Tuple[Tuple[Minion, ...], int]] = {}
    for snap in state.eliminated:
        out[snap.seat] = (snap.last_board, snap.tavern_tier)
    for seat in state.alive:
        p = state.players[seat]
        out[seat] = (tuple(p.board), p.tavern_tier)
    return out


def step_from_stem(path: Path) -> int:
    tail = path.stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 10**18


def run_checkpoint_games(
    *,
    checkpoint: Path,
    run_label: str,
    num_games: int,
    seed: int,
    device: str,
    replay_dir: Path,
    drain_steps: int,
) -> List[dict]:
    replay_dir.mkdir(parents=True, exist_ok=True)
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

    old_cap = lobby_mod.MAX_DRAIN_STEPS
    lobby_mod.MAX_DRAIN_STEPS = int(drain_steps)
    games: List[dict] = []
    try:
        for g in range(num_games):
            game_seed = seed + g
            jpath = replay_dir / f"{checkpoint.stem}__{g:04d}.jsonl"
            meta = {
                "game": "bglike",
                "run": run_label,
                "checkpoint": checkpoint.name,
                "step": step_from_stem(checkpoint),
                "mode": "self_play_8p",
                "game_index": g,
                "seed": game_seed,
            }
            attach_replay(env, jpath, meta)
            try:
                env.reset(seed=game_seed)
                env.drain_until_lobby_done(deterministic=True)
                st = env.state
                if not env.lobby_done:
                    raise RuntimeError(
                        f"lobby did not finish (done={st.done}, alive={st.alive}, "
                        f"round={st.round_number})"
                    )
                boards = final_boards_by_seat(st)
                placements = placements_for_state(st)
                per_seat = []
                for seat in range(8):
                    board, tier = boards[seat]
                    races = Counter(_race_label(m) for m in board)
                    per_seat.append(
                        {
                            "seat": seat,
                            "placement": placements[seat],
                            "tier": tier,
                            "board_size": len(board),
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
                        "round": st.round_number,
                        "combat_round": st.combat_round,
                        "seats": per_seat,
                    }
                )
            finally:
                close_replay(env)
            jpath.with_suffix(".txt").write_text(
                render_jsonl_file(jpath), encoding="utf-8"
            )
            print(
                f"  [{run_label}] {checkpoint.stem} game {g + 1}/{num_games} "
                f"winner=P{st.winner} rounds={st.round_number}",
                flush=True,
            )
    finally:
        lobby_mod.MAX_DRAIN_STEPS = old_cap
    return games


def summarize_games(games: List[dict]) -> dict:
    stat_sums: List[float] = []
    placements: List[float] = []
    race_counts: Counter[str] = Counter()
    n_minions = 0
    for game in games:
        for seat in game["seats"]:
            stat_sums.append(float(seat["stat_sum"]))
            placements.append(float(seat["placement"]))
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

    stat_s = mean_ci(stat_sums)
    place_s = mean_ci(placements)
    race_total = sum(race_counts.values()) or 1
    race_pct = {k: 100.0 * v / race_total for k, v in sorted(race_counts.items())}
    return {
        "num_games": len(games),
        "player_samples": len(stat_sums),
        "mean_board_stat_sum": stat_s,
        "mean_placement": place_s,
        "race_minion_counts": dict(sorted(race_counts.items())),
        "race_minion_pct": race_pct,
        "total_minions_on_final_boards": n_minions,
    }


def parse_checkpoint_specs(raw: Sequence[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Expected label=path, got {item!r}")
        label, path = item.split("=", 1)
        out.append((label.strip(), Path(path.strip()).resolve()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run label and checkpoint path, e.g. 024_52k=runs/.../dist_bglike_ppo_52146.pt",
    )
    ap.add_argument("--num-games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=77)
    # For this BGLike lobby replay workload, inference is faster on CPU than GPU.
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--drain-steps", type=int, default=20_000)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs/bglike/lobby_eval",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replays").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    specs = parse_checkpoint_specs(args.checkpoint)

    summary_rows: List[dict] = []
    all_payload: dict = {
        "num_games": args.num_games,
        "seed": args.seed,
        "device": args.device,
        "checkpoints": [],
    }

    for label, ck_path in specs:
        if not ck_path.is_file():
            raise SystemExit(f"Missing checkpoint: {ck_path}")
        step = step_from_stem(ck_path)
        replay_dir = out_dir / "replays" / label / ck_path.stem
        print(
            f"\n=== {label} step={step} games={args.num_games} -> {replay_dir}",
            flush=True,
        )
        games = run_checkpoint_games(
            checkpoint=ck_path,
            run_label=label,
            num_games=args.num_games,
            seed=args.seed + hash(label) % 10_000,
            device=args.device,
            replay_dir=replay_dir,
            drain_steps=args.drain_steps,
        )
        summary = summarize_games(games)
        payload_ck = {
            "label": label,
            "checkpoint": ck_path.name,
            "step": step,
            "summary": summary,
            "games": games,
        }
        ck_json = out_dir / "raw" / f"{label}__{ck_path.stem}.json"
        ck_json.parent.mkdir(parents=True, exist_ok=True)
        ck_json.write_text(json.dumps(payload_ck, indent=2, ensure_ascii=False), encoding="utf-8")
        all_payload["checkpoints"].append(payload_ck)

        ms = summary["mean_board_stat_sum"]
        mp = summary["mean_placement"]
        row = {
            "label": label,
            "step": step,
            "checkpoint": ck_path.stem,
            "mean_stat_sum": ms["mean"],
            "stat_ci_half": ms["ci95_half"],
            "mean_placement": mp["mean"],
            "place_ci_half": mp["ci95_half"],
            "n_player_samples": summary["player_samples"],
            "n_games": summary["num_games"],
        }
        for race, pct in summary["race_minion_pct"].items():
            row[f"pct_{race}"] = pct
        summary_rows.append(row)

        print(
            f"  mean Σ(atk+hp)={ms['mean']:.1f} ±{ms['ci95_half']:.1f}  "
            f"mean place={mp['mean']:.2f} ±{mp['ci95_half']:.2f}",
            flush=True,
        )
        print("  races:", flush=True)
        for race, pct in summary["race_minion_pct"].items():
            cnt = summary["race_minion_counts"][race]
            print(f"    {race:12s} {cnt:4d} ({pct:5.1f}%)", flush=True)

    all_races = sorted({k for row in summary_rows for k in row if k.startswith("pct_")})
    csv_lines = [
        ",".join(
            [
                "label",
                "step",
                "checkpoint",
                "mean_stat_sum",
                "stat_ci_half",
                "mean_placement",
                "place_ci_half",
                "n_games",
                "n_player_samples",
                *all_races,
            ]
        )
    ]
    for row in summary_rows:
        cells = [
            str(row["label"]),
            str(row["step"]),
            str(row["checkpoint"]),
            f"{row['mean_stat_sum']:.4f}",
            f"{row['stat_ci_half']:.4f}",
            f"{row['mean_placement']:.4f}",
            f"{row['place_ci_half']:.4f}",
            str(row["n_games"]),
            str(row["n_player_samples"]),
        ]
        for race in all_races:
            cells.append(f"{row.get(f'pct_{race}', 0.0):.4f}")
        csv_lines.append(",".join(cells))

    csv_path = out_dir / "summary.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# BGLike lobby checkpoint eval",
        "",
        f"- games per checkpoint: {args.num_games}",
        f"- seed base: {args.seed}",
        f"- device: {args.device}",
        "",
        "| label | step | mean Σ(atk+hp) | mean place | n games |",
        "|-------|------|----------------|------------|---------|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['label']} | {row['step']} | "
            f"{row['mean_stat_sum']:.1f} ± {row['stat_ci_half']:.1f} | "
            f"{row['mean_placement']:.2f} ± {row['place_ci_half']:.2f} | "
            f"{row['n_games']} |"
        )
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nWrote {csv_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()

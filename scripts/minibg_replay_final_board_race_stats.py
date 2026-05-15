#!/usr/bin/env python3
"""Per checkpoint: N MiniBG replays; final learned board → avg Σhp, Σ(atk+hp), race pool %."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Counter as CounterT, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.envs  # noqa: F401

from src.envs import RewardConfig
from src.evaluation.eval_checkpoints import (
    build_opponents_from_names,
    evaluate_agent_vs_opponents_metrics,
    find_checkpoints,
    load_training_agent_checkpoint,
)
from src.registry import make_game
from src.training.trainer import StartPolicy
from src.utils.match import play_single_game, resolve_opening_agent_token


def step_from_name(path: Path) -> int:
    stem = path.stem
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 10**18


def terminal_learned_board_snapshot(
    path: Path,
) -> Tuple[int, int, CounterT[str]]:
    """
    Last done frame: learned player's board → (sum_hp, sum_atk_hp, race counts for that board).
    """
    learned_idx: Optional[int] = None
    last_hp = 0
    last_stats = 0
    last_race: CounterT[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("type") == "header":
            learned_idx = int(rec.get("learned_player_index", -1))
            continue
        if rec.get("type") != "frame" or learned_idx not in (0, 1):
            continue
        st = rec.get("state") or {}
        if not st.get("done"):
            continue
        pk = f"p{learned_idx}"
        player = st.get(pk) or {}
        board = player.get("board") or []
        sh = 0
        ss = 0
        rc: CounterT[str] = Counter()
        for m in board:
            if not isinstance(m, dict):
                continue
            atk = int(m.get("atk", 0))
            hp = int(m.get("hp", 0))
            sh += hp
            ss += atk + hp
            r = m.get("race")
            rc[str(r) if r is not None else "NEUTRAL"] += 1
        last_hp, last_stats, last_race = sh, ss, rc
    return last_hp, last_stats, last_race


def run_self_play_replays(
    ck: Path,
    *,
    stem: str,
    rdir: Path,
    n: int,
    seed: int,
    mg: dict,
    device: str,
) -> None:
    agent = load_training_agent_checkpoint(ck, device=device, seed=seed)
    opponent_agent = load_training_agent_checkpoint(
        ck, device=device, seed=seed + 913_123
    )
    for a in (agent, opponent_agent):
        if hasattr(a, "eval"):
            a.eval()
        if hasattr(a, "epsilon"):
            a.epsilon = 0.0

    match_seed = seed + hash(stem) % (2**16) + hash("self") % (2**16)
    inner_start = StartPolicy.RANDOM
    randomize_first = True
    agent_first_in_call = True
    learned_kind = type(agent).__name__
    mg_base = dict(mg)

    for g in range(n):
        rpath = rdir / f"{stem}__self__{g:04d}.jsonl"
        game_seed = (match_seed + g) if match_seed is not None else None
        inner_start = StartPolicy.RANDOM if randomize_first else StartPolicy.AGENT_FIRST
        agent_token = resolve_opening_agent_token(inner_start, seed=game_seed)
        learned_player_index = 0 if agent_token == 1 else 1
        scripted_player_index = 1 - learned_player_index
        meta = {
            "format": 2,
            "game": "minibg",
            "checkpoint": stem,
            "opponent": "self_play_same_ckpt",
            "game_index": g,
            "match_seed": match_seed,
            "game_seed": game_seed,
            "opening_policy": inner_start.name,
            "agent_token": agent_token,
            "learned_player_index": learned_player_index,
            "scripted_player_index": scripted_player_index,
            "learned_agent_kind": learned_kind,
            "scripted_opponent": "same_checkpoint_second_load",
            "roles_note": (
                f"self-play: P{learned_player_index}=checkpoint A; "
                f"P{scripted_player_index}=checkpoint B (identical weights)"
            ),
        }
        env_g = make_game(
            "minibg",
            reward_config=RewardConfig(),
            replay_path=rpath,
            replay_meta=meta,
            **mg_base,
        )
        for participant in (agent, opponent_agent):
            if hasattr(participant, "set_env"):
                participant.set_env(env_g)
        a1x, a2x = (
            (agent, opponent_agent)
            if agent_first_in_call
            else (opponent_agent, agent)
        )
        play_single_game(
            env_g,
            a1x,
            a2x,
            start_policy=inner_start,
            random_opening_config=None,
            deterministic_agent=True,
            deterministic_opponent=True,
            seed=game_seed,
        )
        if hasattr(env_g, "close_replay"):
            env_g.close_replay()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="minibg_ppo_structured")
    ap.add_argument("--num-games", type=int, default=50)
    ap.add_argument("--opponent", type=str, default="t_up_random")
    ap.add_argument("--self-play", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--battle-damage-shaping", type=float, default=0.06)
    ap.add_argument(
        "--replay-subdir",
        type=str,
        default=None,
        help="Under run-dir (default: replay_race_dist or replay_selfplay_race)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Long-form race CSV (optional if only wide summary needed)",
    )
    ap.add_argument(
        "--wide-csv",
        type=Path,
        default=None,
        help="One row per checkpoint: step, stem, avg_hp, avg_stats, pct_* (default when --self-play)",
    )
    ap.add_argument(
        "--checkpoint-stems",
        nargs="*",
        default=None,
        metavar="STEM",
        help="Only these stems (e.g. minibg_ppo_structured_350000). Default: all under prefix.",
    )
    ap.add_argument(
        "--from-disk",
        action="store_true",
        help="Skip playing games; aggregate only from existing JSONL under replay-subdir.",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    ckpts = [p for p, _ in find_checkpoints(run_dir / "checkpoints", prefix=args.prefix)]
    ckpts.sort(key=step_from_name)
    if not ckpts:
        raise SystemExit(f"No checkpoints with prefix {args.prefix!r} in {run_dir / 'checkpoints'}")
    if args.checkpoint_stems:
        by_stem = {p.stem: p for p in ckpts}
        missing = [s for s in args.checkpoint_stems if s not in by_stem]
        if missing:
            raise SystemExit(
                f"checkpoint stem(s) not found: {missing!r}. "
                f"Available: {sorted(by_stem)}"
            )
        ckpts = [by_stem[s] for s in args.checkpoint_stems]

    n = int(args.num_games)
    opponent = str(args.opponent)
    mg = {"battle_damage_shaping": float(args.battle_damage_shaping)}
    subdir = args.replay_subdir or (
        "replay_selfplay_race" if args.self_play else "replay_race_dist"
    )
    replay_root = run_dir / subdir

    csv_lines: list[str] = []
    csv_lines.append("step,stem,opponent,race,count,pct_within_ckpt,total_minions,games")

    mode = "self-play" if args.self_play else opponent
    print(
        f"run={run_dir.name}  mode={mode}  n={n}  ckpts={len(ckpts)}"
        f"{'  from_disk' if args.from_disk else ''}",
        flush=True,
    )

    import shutil

    wide_rows: List[dict] = []
    all_races: set[str] = set()

    for ck in ckpts:
        stem = ck.stem
        rdir = replay_root / stem
        file_suffix = "self" if args.self_play else opponent

        if args.from_disk:
            if not rdir.is_dir():
                print(f"  {stem}: skip (no replay dir)", flush=True)
                continue
            jsonl_paths = sorted(rdir.glob(f"{stem}__{file_suffix}__*.jsonl"))
            if not jsonl_paths:
                print(f"  {stem}: skip (no {stem}__{file_suffix}__*.jsonl)", flush=True)
                continue
        else:
            if rdir.exists():
                shutil.rmtree(rdir)
            rdir.mkdir(parents=True, exist_ok=True)

            if args.self_play:
                run_self_play_replays(
                    ck,
                    stem=stem,
                    rdir=rdir,
                    n=n,
                    seed=args.seed,
                    mg=mg,
                    device=args.device,
                )
            else:
                agent = load_training_agent_checkpoint(
                    ck, device=args.device, seed=args.seed
                )
                opps = build_opponents_from_names(
                    [opponent], seed=args.seed, game_id="minibg"
                )
                evaluate_agent_vs_opponents_metrics(
                    agent,
                    opponents=opps,
                    num_games=n,
                    batch_size=n,
                    game_id="minibg",
                    seed=args.seed,
                    reward_config=RewardConfig(),
                    start_policy="random",
                    minibg_params=mg,
                    replay_dir=rdir,
                    match_identity=stem,
                    deterministic=True,
                )
            jsonl_paths = [rdir / f"{stem}__{file_suffix}__{g:04d}.jsonl" for g in range(n)]

        pool: CounterT[str] = Counter()
        hp_vals: List[int] = []
        stat_vals: List[int] = []
        for jp in jsonl_paths:
            if not jp.exists():
                continue
            sh, ss, cr = terminal_learned_board_snapshot(jp)
            hp_vals.append(sh)
            stat_vals.append(ss)
            pool.update(cr)

        ng = len(hp_vals)
        tot = sum(pool.values())
        st = step_from_name(ck)
        st_s = str(st) if st < 10**17 else stem
        if tot == 0:
            print(f"  {stem}: no minions on terminal boards", flush=True)
            csv_lines.append(f"{st_s},{stem},{mode},,0,,0,{ng}")
            row = {
                "step": st_s,
                "stem": stem,
                "avg_sum_hp": float("nan"),
                "avg_sum_atk_hp": float("nan"),
                "n_games": ng,
            }
            wide_rows.append(row)
            continue

        avg_hp = sum(hp_vals) / max(len(hp_vals), 1)
        avg_st = sum(stat_vals) / max(len(stat_vals), 1)
        print(
            f"  {stem}: replays={ng} total_minions={tot} avg_Σhp={avg_hp:.2f} avg_Σ(atk+hp)={avg_st:.2f}",
            flush=True,
        )

        row = {
            "step": st_s,
            "stem": stem,
            "avg_sum_hp": round(avg_hp, 4),
            "avg_sum_atk_hp": round(avg_st, 4),
            "n_games": ng,
        }
        all_races.update(pool.keys())
        for race in sorted(pool.keys()):
            cnt = pool[race]
            pct = 100.0 * cnt / tot
            print(f"      {race}: {cnt} ({pct:.1f}%)", flush=True)
            csv_lines.append(
                f"{st_s},{stem},{mode},{race},{cnt},{pct:.4f},{tot},{ng}"
            )
            row[f"pct_{race}"] = round(pct, 4)
        wide_rows.append(row)

    long_out = args.out_csv
    if long_out is None:
        long_out = (
            run_dir / "replay_selfplay_race_long.csv"
            if args.self_play
            else run_dir / "replay_race_stats.csv"
        )
    else:
        long_out = long_out.resolve()
    if long_out:
        long_out.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        print(f"Wrote {long_out}", flush=True)

    wide_path = args.wide_csv or (
        run_dir / "selfplay_replay_table.csv" if args.self_play else None
    )
    if wide_path:
        wide_path = wide_path.resolve()
        races_sorted = sorted(all_races)
        hdr = ["step", "stem", "avg_sum_hp", "avg_sum_atk_hp", "n_games"] + [
            f"pct_{r}" for r in races_sorted
        ]
        lines_w = [",".join(hdr)]
        for row in wide_rows:
            cells = [
                str(row.get("step", "")),
                str(row.get("stem", "")),
                str(row.get("avg_sum_hp", "")),
                str(row.get("avg_sum_atk_hp", "")),
                str(row.get("n_games", n)),
            ]
            for r in races_sorted:
                v = row.get(f"pct_{r}", 0)
                cells.append(str(v))
            lines_w.append(",".join(cells))
        wide_path.write_text("\n".join(lines_w) + "\n", encoding="utf-8")
        print(f"Wrote {wide_path}", flush=True)


if __name__ == "__main__":
    main()

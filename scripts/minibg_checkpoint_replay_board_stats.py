#!/usr/bin/env python3
"""Per checkpoint: N MiniBG replays; mean sum(atk+hp) on learned board @ game end + 95% CI.

``--opponent <bot_id>``: vs heuristic. ``--self-play``: same checkpoint loaded twice
(two agents, identical weights, independent PPO caches).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.envs  # noqa: F401 — register minibg

from src.envs import RewardConfig
from src.envs.minibg.replay_render import render_jsonl_file
from src.evaluation.eval_checkpoints import (
    build_opponents_from_names,
    evaluate_agent_vs_opponents_metrics,
    find_checkpoints,
    load_training_agent_checkpoint,
)
from src.registry import make_game
from src.training.trainer import StartPolicy
from src.utils.match import play_single_game, resolve_opening_agent_token

_T975 = {
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
}


def t_crit_975(n: int) -> float:
    if n < 2:
        return float("nan")
    return _T975.get(n - 1, 1.96)


def terminal_learned_board_stat_sum(path: Path) -> int:
    learned_idx: Optional[int] = None
    last_sum: Optional[int] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        t = rec.get("type")
        if t == "header":
            learned_idx = int(rec.get("learned_player_index", -1))
            continue
        if t != "frame" or learned_idx not in (0, 1):
            continue
        st = rec.get("state") or {}
        if not st.get("done"):
            continue
        player = (st.get(f"p{learned_idx}") or {}) if learned_idx is not None else {}
        board = player.get("board") or []
        s = 0
        for m in board:
            if not isinstance(m, dict):
                continue
            s += int(m.get("atk", 0)) + int(m.get("hp", 0))
        last_sum = s
    return int(last_sum if last_sum is not None else -1)


def step_from_name(path: Path) -> int:
    stem = path.stem
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 10**18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="minibg_ppo_structured")
    ap.add_argument("--num-games", type=int, default=10)
    ap.add_argument(
        "--opponent",
        type=str,
        default="t_up_random",
        help="Heuristic bot id (ignored if --self-play).",
    )
    ap.add_argument(
        "--self-play",
        action="store_true",
        help="Same checkpoint loaded twice (two agents); replays tagged learned vs copy.",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--battle-damage-shaping", type=float, default=0.06)
    ap.add_argument("--exclude-final", action="store_true")
    ap.add_argument("--replay-subdir", type=str, default="replay_board_stats")
    ap.add_argument(
        "--checkpoint-stems",
        nargs="*",
        default=None,
        metavar="STEM",
        help="Only these stems (e.g. minibg_ppo_structured_750000). Default: all.",
    )
    ap.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="Report markdown under run-dir (default: replay_board_stat_report*.md).",
    )
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()

    ckpts = [p for p, _ in find_checkpoints(run_dir / "checkpoints", prefix=args.prefix)]
    ckpts.sort(key=step_from_name)
    if args.exclude_final:
        ckpts = [p for p in ckpts if not p.name.endswith("_final.pt")]
    if args.checkpoint_stems:
        want = frozenset(args.checkpoint_stems)
        ckpts = [p for p in ckpts if p.stem in want]
        found_stems = frozenset(p.stem for p in ckpts)
        missing = want - found_stems
        if missing:
            print(f"Missing checkpoints: {sorted(missing)}", file=sys.stderr)
            raise SystemExit(1)

    replay_root = run_dir / args.replay_subdir
    replay_root.mkdir(parents=True, exist_ok=True)
    minibg_params = {"battle_damage_shaping": float(args.battle_damage_shaping)}
    n = int(args.num_games)
    opponent = str(args.opponent)
    opp_label = "self" if args.self_play else opponent

    print(
        f"run={run_dir.name}  matchup={'self-play (two loads)' if args.self_play else opponent}  n={n}  "
        f"metric=sum(atk+hp) learned board @ done",
        flush=True,
    )

    rows: List[str] = []
    rows.append(
        "| step | checkpoint | mean Σ(atk+hp) | 95% CI | n | invalid |"
    )
    rows.append("|------|------------|---------------|--------|---|---------|")

    for ck in ckpts:
        stem = ck.stem
        rdir = replay_root / stem
        if rdir.exists():
            import shutil

            shutil.rmtree(rdir)
        rdir.mkdir(parents=True, exist_ok=True)

        agent = load_training_agent_checkpoint(ck, device=args.device, seed=args.seed)
        if args.self_play:
            # Second copy: same weights, separate buffers/cache (safe alternating turns).
            opponent_agent = load_training_agent_checkpoint(
                ck, device=args.device, seed=args.seed + 913_123
            )
            for a in (agent, opponent_agent):
                if hasattr(a, "eval"):
                    a.eval()
            match_seed = (
                args.seed
                + hash(stem) % (2**16)
                + hash("self") % (2**16)
            )
            sp = "random"
            randomize_first = sp == "random"
            agent_first_in_call = sp != "opponent_first"
            learned_kind = type(agent).__name__
            mg_base = dict(minibg_params)
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
                result = play_single_game(
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
                rpath.with_suffix(".txt").write_text(
                    render_jsonl_file(rpath), encoding="utf-8"
                )
                del result  # unused
        else:
            opps = build_opponents_from_names([opponent], seed=args.seed, game_id="minibg")
            evaluate_agent_vs_opponents_metrics(
                agent,
                opponents=opps,
                num_games=n,
                batch_size=n,
                game_id="minibg",
                seed=args.seed,
                reward_config=RewardConfig(),
                start_policy="random",
                minibg_params=minibg_params,
                replay_dir=rdir,
                match_identity=stem,
                deterministic=True,
            )

        vals: List[float] = []
        invalid = 0
        for g in range(n):
            jp = rdir / f"{stem}__{opp_label}__{g:04d}.jsonl"
            s = terminal_learned_board_stat_sum(jp)
            if s < 0:
                invalid += 1
                continue
            vals.append(float(s))

        if len(vals) < 2:
            mean_v = float(sum(vals) / len(vals)) if vals else float("nan")
            half = float("nan")
        else:
            mean_v = sum(vals) / len(vals)
            # sample std (ddof=1)
            m = mean_v
            var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
            sd = math.sqrt(var)
            half = t_crit_975(len(vals)) * sd / math.sqrt(len(vals))

        st = step_from_name(ck)
        st_display = str(st) if st < 10**17 else stem
        rows.append(
            f"| {st_display} | `{stem}` | {mean_v:.1f} | [{mean_v - half:.1f}, {mean_v + half:.1f}] | {len(vals)} | {invalid} |"
        )
        print(f"  {stem}: mean={mean_v:.1f} ci~[{mean_v - half:.1f},{mean_v + half:.1f}] valid={len(vals)}/{n}", flush=True)

    if args.report_name:
        report = run_dir / args.report_name
    else:
        report = run_dir / (
            "replay_board_stat_report_selfplay.md"
            if args.self_play
            else "replay_board_stat_report.md"
        )
    report.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {report}", flush=True)
    print("\n" + "\n".join(rows))


if __name__ == "__main__":
    main()

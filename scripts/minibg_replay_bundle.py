#!/usr/bin/env python3
"""Generate JSONL replays vs several opponents and write human-readable .txt next to them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401
from src.envs import RewardConfig
from src.envs.minibg.replay_render import render_jsonl_file
from src.evaluation.eval_checkpoints import build_opponents_from_names, find_checkpoints, load_dqn_checkpoint
from src.registry import make_game
from src.training.trainer import StartPolicy
from src.utils.match import play_single_game


DEFAULT_OPPONENTS = [
    "random",
    "tempo",
    "buffer_t2",
    "balanced",
    "wide_t1",
    "fast_t3",
    "token",
    "buffer_t1",
    "delayed_t2_pressure",
    "early_t2_pressure",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=REPO_ROOT / "runs/minibg/dqn_long_mixed/checkpoints",
        help="Directory containing minibg_dqn_*.pt",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "runs/minibg/replay_bundle",
        help="Output directory for .jsonl and .txt",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument(
        "--opponents",
        nargs="*",
        default=DEFAULT_OPPONENTS,
        help="Opponent bot keys (default: 10 mixed)",
    )
    args = ap.parse_args()

    found = find_checkpoints(args.checkpoint_dir.resolve())
    if not found:
        raise SystemExit(f"No checkpoints in {args.checkpoint_dir}")
    latest_path = found[-1][0]
    print("Using checkpoint:", latest_path.name, flush=True)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = load_dqn_checkpoint(latest_path, device=args.device, seed=args.seed)
    if hasattr(agent, "eval"):
        agent.eval()
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    names = list(args.opponents)
    opps = build_opponents_from_names(names, seed=args.seed + 1, game_id="minibg")

    for i, name in enumerate(names):
        opp = opps[name]
        stem = f"{latest_path.stem}__vs_{name}"
        jpath = out_dir / f"{stem}.jsonl"
        tpath = out_dir / f"{stem}.txt"
        meta = {
            "checkpoint": latest_path.stem,
            "opponent": name,
            "game_index": i,
            "seed_base": args.seed,
        }
        env = make_game(
            "minibg",
            reward_config=RewardConfig(),
            battle_damage_shaping=0.06,
            replay_path=jpath,
            replay_meta=meta,
        )
        opp.set_env(env)
        game_seed = args.seed + 1000 * i
        play_single_game(
            env,
            agent,
            opp,
            start_policy=StartPolicy.AGENT_FIRST,
            random_opening_config=None,
            deterministic_agent=True,
            deterministic_opponent=True,
            seed=game_seed,
        )
        env.close_replay()
        text = render_jsonl_file(jpath)
        tpath.write_text(text, encoding="utf-8")
        print(f"Wrote {jpath.name} + {tpath.name}", flush=True)


if __name__ == "__main__":
    main()

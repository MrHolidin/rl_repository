#!/usr/bin/env python3
"""Full JSONL replays for heuristic-bot lobbies (per-seat bot names).

``bglike_replay.py generate`` drives seats through ``BGLikeHeuristicAgent``,
whose ``set_env`` rejects the replay bridge, so it cannot record heuristic-bot
games. This drives the lobby directly (same view as
``bglike_elemental_vs_structured.py``) with the shared replay recorder attached.

Usage:
    python3 scripts/bglike_bot_replays.py --bots planner,structured,... \
        --episodes 10 --out runs/replays/mix.jsonl --render-txt runs/replays/mix.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.random_agent import RandomAgent
from src.envs.bglike.heuristic_bots.bots import make_bot
from src.bg_core.minion import Race
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.replay import attach_replay, close_replay
from src.envs.bglike.replay_render import render_jsonl_file
from src.envs.bglike.seat_config import SeatConfig, SeatKind

from bglike_elemental_vs_structured import DirectBotView  # noqa: E402

NUM_PLAYERS = 8
DEFAULT_PATCH = "data/bgcore/19_6_0_74257"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--render-txt", type=Path, default=None)
    ap.add_argument("--bots", type=str, required=True, help="8 comma-separated bot names")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--patch-dir", type=str, default=DEFAULT_PATCH)
    ap.add_argument(
        "--excluded-tribe",
        type=str,
        default=None,
        help="Pin the tribe the lobby leaves out (e.g. MURLOC), so the tribe "
             "under study is guaranteed to be in the pool",
    )
    ap.add_argument("--extended", action="store_true", default=True)
    args = ap.parse_args()

    names = [n.strip() for n in args.bots.split(",")]
    if len(names) != NUM_PLAYERS:
        raise SystemExit(f"--bots needs {NUM_PLAYERS} names, got {len(names)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = BGLobbyEnv(
        [SeatConfig(SeatKind.LEARNED, RandomAgent(seed=args.seed))]
        + [SeatConfig(SeatKind.RANDOM)] * (NUM_PLAYERS - 1),
        learned_seats=[0],
        seed=args.seed,
        patch_dir=args.patch_dir,
        shop_excluded_race=(
            None if args.excluded_tribe is None
            else Race[args.excluded_tribe.strip().upper()]
        ),
    )
    attach_replay(
        env,
        args.out,
        {"bots": names, "seed": args.seed, "episodes": args.episodes},
        record_seats=None,
        sparse=False,
    )
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            bots = [make_bot(n, seed=s + seed * 100) for s, n in enumerate(names)]
            views = [DirectBotView(env, s) for s in range(NUM_PLAYERS)]
            env.reset(seed=seed)
            steps = 0
            while not env.lobby_done and steps < 20_000:
                steps += 1
                cur = env.current_seat()
                view = views[cur]
                view.set_mask_override(env.legal_mask_for_seat(cur))
                try:
                    action = bots[cur].choose_action(view)
                finally:
                    view.set_mask_override(None)
                env.step_action(cur, action)
            placements = env.finalize_placements()
            by_place = sorted(placements.items(), key=lambda kv: kv[1])
            print(
                f"ep{ep} seed={seed} steps={steps} "
                + " ".join(f"{pl}:S{seat}({names[seat]})" for seat, pl in by_place)
            )
    finally:
        close_replay(env)
    print(f"Wrote {args.out}")

    if args.render_txt:
        txt = render_jsonl_file(args.out.resolve(), extended=args.extended)
        args.render_txt.parent.mkdir(parents=True, exist_ok=True)
        args.render_txt.write_text(txt, encoding="utf-8")
        print(f"Wrote {args.render_txt}")


if __name__ == "__main__":
    main()

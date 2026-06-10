#!/usr/bin/env python3
"""Generate ONE rich JSONL replay for a DvD checkpoint (2 seats/identity).

Routes the DvD per-identity seating through the SHARED replay recorder
(``attach_replay``, ``sparse=False`` → every action frame with full board /
shop / hand cards + tier), so the resulting JSONL answers every downstream
question (purchases, final boards, stat sums, tavern tiers, commitment timing)
WITHOUT re-simulating. Render with ``scripts/bglike_replay.py render``.

(The stock ``bglike_replay.py checkpoints`` command can't be used here: its
loader dispatches DvD nets to the base structured agent and drops the identity
API, and it has no 2-seats-per-identity seating.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401
from src.agents.ppo_dvd_agent import PPODvDAgent, SiblingOpponent
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.replay import attach_replay, close_replay
from src.envs.bglike.replay_render import render_jsonl_file
from src.envs.bglike.seat_config import lobby_from_learned_seats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="Output .jsonl path")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument("--identity-tribes", default="DRAGON,ELEMENTAL,BEAST,PIRATE")
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    ap.add_argument("--render-txt", type=Path, default=None)
    args = ap.parse_args()

    tribes = [t.strip().upper() for t in args.identity_tribes.split(",")]
    N = len(tribes)
    base = PPODvDAgent.load(str(args.checkpoint), device="cpu", seed=args.seed)
    base.eval(); base.training = False
    seat_identity = {s: (s // 2) % N for s in range(8)}  # 2 seats per identity
    agents = {s: SiblingOpponent(base, identity=seat_identity[s], greedy=True)
              for s in range(8)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(configs, learned_seats=tuple(range(8)), training_seats=(0,),
                     seed=args.seed, patch_dir=args.patch_dir, obs_kind="bglike_v5")
    attach_replay(
        env, args.out,
        {"checkpoint": args.checkpoint.name, "games": args.games,
         "seed": args.seed, "identity_tribes": tribes,
         "seat_identity": seat_identity},
        record_seats=None,   # all 8 seats
        sparse=False,        # every action frame → purchases recoverable
    )
    try:
        for g in range(args.games):
            env.reset(seed=args.seed + g)
            env.drain_until_lobby_done(deterministic=True)
    finally:
        close_replay(env)
    print(f"Wrote {args.out} ({args.games} games, 2 seats/identity, full frames)")

    if args.render_txt:
        args.render_txt.write_text(render_jsonl_file(args.out, extended=True), encoding="utf-8")
        print(f"Wrote {args.render_txt}")


if __name__ == "__main__":
    main()

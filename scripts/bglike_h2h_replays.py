#!/usr/bin/env python3
"""4v4 two checkpoints with JSONL replay capture, then summarise from the replays.

The card-usage diff counts buys and final boards straight out of the live loop.
That answers "which minions", not "how the game went": when a card was bought,
what was sold, when the tavern went up, what the board looked like round by
round. Those live only in a replay.

Writes one JSONL per lobby plus a per-seat index of which side played it, so any
later analysis can split the files by team without replaying anything. Frames are
milestone-only by default -- the end-of-turn board, which is the one that fights;
``--full-frames`` keeps every shop frame at ~8x the size (33 MB per lobby).

    python -m scripts.bglike_h2h_replays \
        --ckpt-a .../copyfix_final.pt --ckpt-b .../base_final.pt \
        --label-a copyfix --label-b base --games 30 --out-dir runs/bglike/replays_copyfix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

torch.set_num_threads(1)

import src.envs  # noqa: F401
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.placement import placement_for_seat
from src.envs.bglike.replay import attach_replay, close_replay
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint


class _SeatView:
    __slots__ = ("state",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-a", type=Path, required=True)
    ap.add_argument("--ckpt-b", type=Path, required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5150)
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    ap.add_argument("--out-dir", type=Path, required=True)
    # Full frames are ~33 MB per lobby; the milestone frame (end of each
    # seat's turn) is the board that actually fights, so it is the default.
    ap.add_argument("--full-frames", action="store_true", default=False)
    args = ap.parse_args()

    A = load_training_agent_checkpoint(str(args.ckpt_a), device="cpu", seed=args.seed)
    B = load_training_agent_checkpoint(str(args.ckpt_b), device="cpu", seed=args.seed + 1)
    A.eval(); B.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = []

    for g in range(args.games):
        # Alternate which seats each side holds so seat order cannot bias the set.
        a_seats = set(range(0, 8, 2)) if g % 2 == 0 else set(range(1, 8, 2))
        side = {s: (args.label_a if s in a_seats else args.label_b) for s in range(8)}
        agents = {s: (A if side[s] == args.label_a else B) for s in range(8)}

        env = BGLobbyEnv(
            lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents),
            learned_seats=tuple(range(8)), training_seats=(0,),
            seed=args.seed + 977 * g, patch_dir=args.patch_dir,
            obs_kind="bglike_v6_heroes", with_heroes=True,
        )
        env.reset(seed=args.seed + 977 * g)
        path = args.out_dir / f"lobby_{g:03d}.jsonl"
        attach_replay(
            env, path,
            {"lobby": g, "seed": args.seed + 977 * g,
             "side_by_seat": {str(s): side[s] for s in range(8)},
             "ckpt_a": str(args.ckpt_a), "ckpt_b": str(args.ckpt_b)},
            sparse=not args.full_frames,
        )

        steps = 0
        while not env.lobby_done and steps < 6000:
            seat = env.current_seat()
            if not env._seat_can_act(seat):
                break
            obs = env.obs_for_seat(seat)
            legal = env.legal_structured_actions_for_seat(seat)
            view = _SeatView(); view.state = env.state
            act, perm, _ = agents[seat].act_structured(obs, legal, view, deterministic=True)
            env.step_structured_for_seat(seat, act, board_perm=perm)
            steps += 1

        st = env.state
        index.append({
            "file": path.name, "lobby": g, "steps": steps,
            "side_by_seat": {str(s): side[s] for s in range(8)},
            "place_by_seat": {str(s): placement_for_seat(st, s) for s in range(8)},
        })
        close_replay(env)
        if (g + 1) % 5 == 0:
            print(f"  ...{g+1}/{args.games}", flush=True)

    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    total = sum(p.stat().st_size for p in args.out_dir.glob("lobby_*.jsonl"))
    print(f"\nwrote {len(index)} replays to {args.out_dir} ({total/1e6:.1f} MB)")
    print(f"index: {args.out_dir/'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""4v4 two checkpoints, then diff *which cards* each side actually plays.

The head-to-head script answers who wins and reports tribe/tier shares. That is
too coarse to see a behavioural change: two policies can hold the same tribe mix
and still be buying different cards inside it. This records the card behind every
BUY and the composition of every final board, then ranks cards by how differently
the two sides use them.

Both sides play under the *current* engine, which is the only coherent choice —
but note it when one side was trained under a different one: that side is being
scored slightly off its training distribution.

    python -m scripts.bglike_card_usage_diff \
        --ckpt-a runs/.../fix_3828197.pt --ckpt-b runs/.../base_3802331.pt \
        --label-a fix --label-b base --games 60
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

torch.set_num_threads(1)

import src.envs  # noqa: F401
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.placement import placement_for_seat
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.minibg.structured_actions import StructActionType
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint


class _SeatView:
    __slots__ = ("state",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-a", type=Path, required=True)
    ap.add_argument("--ckpt-b", type=Path, required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()

    A = load_training_agent_checkpoint(str(args.ckpt_a), device="cpu", seed=args.seed)
    B = load_training_agent_checkpoint(str(args.ckpt_b), device="cpu", seed=args.seed + 1)
    A.eval(); B.eval()
    print(f"A={args.label_a}: {args.ckpt_a.name}")
    print(f"B={args.label_b}: {args.ckpt_b.name}")
    print(f"{args.games} lobbies, 4v4\n")

    bought = {"A": Counter(), "B": Counter()}
    final = {"A": Counter(), "B": Counter()}
    tribes = {"A": Counter(), "B": Counter()}
    places = {"A": [], "B": []}
    seats_played = {"A": 0, "B": 0}

    for g in range(args.games):
        a_seats = set(range(0, 8, 2)) if g % 2 == 0 else set(range(1, 8, 2))
        side = {s: ("A" if s in a_seats else "B") for s in range(8)}
        agents = {s: (A if side[s] == "A" else B) for s in range(8)}
        env = BGLobbyEnv(
            lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents),
            learned_seats=tuple(range(8)), training_seats=(0,),
            seed=args.seed + 977 * g, patch_dir=args.patch_dir,
            obs_kind="bglike_v6_heroes", with_heroes=True,
        )
        env.reset(seed=args.seed + 977 * g)
        steps = 0
        while not env.lobby_done and steps < 6000:
            seat = env.current_seat()
            if not env._seat_can_act(seat):
                break
            st = env.state
            obs = env.obs_for_seat(seat)
            legal = env.legal_structured_actions_for_seat(seat)
            view = _SeatView(); view.state = st
            act, perm, _ = agents[seat].act_structured(obs, legal, view, deterministic=True)
            if act.type == StructActionType.BUY and act.args:
                slot = int(act.args[0])
                shop = st.players[seat].shop
                if 0 <= slot < len(shop) and shop[slot] is not None:
                    bought[side[seat]][shop[slot].name] += 1
            env.step_structured_for_seat(seat, act, board_perm=perm)
            steps += 1

        st = env.state
        for s in range(8):
            k = side[s]
            seats_played[k] += 1
            places[k].append(placement_for_seat(st, s))
            for m in st.players[s].board:
                if m is None:
                    continue
                final[k][m.name] += 1
                tribes[k][getattr(m.race, "name", "NONE")] += 1
        if (g + 1) % 10 == 0:
            print(f"  ...{g+1}/{args.games}")

    print(f"\nmean place  {args.label_a} {np.mean(places['A']):.3f}   "
          f"{args.label_b} {np.mean(places['B']):.3f}")

    def report(title, counts, per):
        print(f"\n=== {title} (на сиденье) ===")
        keys = set(counts["A"]) | set(counts["B"])
        rows = []
        for k in keys:
            a = counts["A"][k] / per["A"]
            b = counts["B"][k] / per["B"]
            rows.append((a - b, k, a, b))
        rows.sort(reverse=True)
        print(f"  {'карта':<28}{args.label_a:>9}{args.label_b:>9}{'разница':>10}")
        for d, k, a, b in rows[: args.top]:
            print(f"  {k:<28}{a:>9.3f}{b:>9.3f}{d:>+10.3f}")
        print(f"  {'...':<28}")
        for d, k, a, b in rows[-args.top:]:
            print(f"  {k:<28}{a:>9.3f}{b:>9.3f}{d:>+10.3f}")

    report("куплено карт", bought, seats_played)
    report("на финальной доске", final, seats_played)

    print("\n=== трибы на финальной доске (%) ===")
    keys = sorted(set(tribes["A"]) | set(tribes["B"]))
    ta, tb = sum(tribes["A"].values()) or 1, sum(tribes["B"].values()) or 1
    print(f"  {'триба':<20}{args.label_a:>9}{args.label_b:>9}{'разница':>10}")
    for k in keys:
        a, b = 100 * tribes["A"][k] / ta, 100 * tribes["B"][k] / tb
        print(f"  {k:<20}{a:>9.1f}{b:>9.1f}{a-b:>+10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Head-to-head: checkpoint A vs checkpoint B in one lobby, paired by identity.

Seats: even -> A, odd -> B; identity = seat//2. So each identity has exactly one
A seat and one B seat in the same lobby (A-dragon vs B-dragon, etc.). Reports
mean placement (1=best..8=worst) for A vs B overall and per identity, plus the
head-to-head win rate (how often A's seat outplaced B's seat of the same tribe).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import src.envs  # noqa: F401
from src.agents.ppo_dvd_agent import PPODvDAgent, SiblingOpponent
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.placement import placement_for_seat
from src.envs.bglike.seat_config import lobby_from_learned_seats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a", type=Path, help="checkpoint A (e.g. 088)")
    ap.add_argument("ckpt_b", type=Path, help="checkpoint B (e.g. 087)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--identity-tribes", default="DRAGON,ELEMENTAL,BEAST,PIRATE")
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    args = ap.parse_args()

    tribes = [t.strip().upper() for t in args.identity_tribes.split(",")]
    N = len(tribes)
    A = PPODvDAgent.load(str(args.ckpt_a), device="cpu", seed=args.seed); A.eval(); A.training = False
    B = PPODvDAgent.load(str(args.ckpt_b), device="cpu", seed=args.seed); B.eval(); B.training = False

    # even seats -> A, odd -> B; identity = seat//2
    seat_base = {s: (A if s % 2 == 0 else B) for s in range(8)}
    seat_ident = {s: s // 2 for s in range(8)}
    seat_side = {s: ("A" if s % 2 == 0 else "B") for s in range(8)}

    print(f"A = {args.ckpt_a.name}")
    print(f"B = {args.ckpt_b.name}")
    print(f"{args.games} games, paired by identity (A even seats vs B odd seats)\n")

    places = {"A": defaultdict(list), "B": defaultdict(list)}  # side -> identity -> [places]
    h2h = defaultdict(lambda: [0, 0, 0])  # identity -> [A_wins, B_wins, ties]

    for g in range(args.games):
        agents = {s: SiblingOpponent(seat_base[s], identity=seat_ident[s], greedy=True)
                  for s in range(8)}
        configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
        env = BGLobbyEnv(configs, learned_seats=tuple(range(8)), training_seats=(0,),
                         seed=args.seed + g, patch_dir=args.patch_dir, obs_kind="bglike_v5")
        env.reset(seed=args.seed + g)
        env.drain_until_lobby_done(deterministic=True)
        pl = {s: placement_for_seat(env.state, s) for s in range(8)}
        for s in range(8):
            places[seat_side[s]][seat_ident[s]].append(pl[s])
        for i in range(N):
            a_seat, b_seat = 2 * i, 2 * i + 1
            if pl[a_seat] < pl[b_seat]:
                h2h[i][0] += 1
            elif pl[b_seat] < pl[a_seat]:
                h2h[i][1] += 1
            else:
                h2h[i][2] += 1

    def allp(side):
        return [p for i in range(N) for p in places[side][i]]

    print("=== overall mean placement (lower = better) ===")
    print(f"  A: {np.mean(allp('A')):.2f}")
    print(f"  B: {np.mean(allp('B')):.2f}")
    aw = sum(h2h[i][0] for i in range(N)); bw = sum(h2h[i][1] for i in range(N)); tw = sum(h2h[i][2] for i in range(N))
    print(f"  head-to-head (same-tribe pairs): A wins {aw}, B wins {bw}, ties {tw} "
          f"-> A winrate {aw/(aw+bw+tw):.0%}")

    print("\n=== per identity: mean place A vs B, and A's same-tribe winrate ===")
    print(f"{'identity':<14}{'A place':>9}{'B place':>9}{'A win':>8}{'B win':>8}{'tie':>6}{'A wr':>7}")
    for i in range(N):
        a_m = np.mean(places["A"][i]); b_m = np.mean(places["B"][i])
        w, l, t = h2h[i]
        print(f"id{i} {tribes[i]:<9}{a_m:>9.2f}{b_m:>9.2f}{w:>8}{l:>8}{t:>6}{w/(w+l+t):>6.0%}")


if __name__ == "__main__":
    main()

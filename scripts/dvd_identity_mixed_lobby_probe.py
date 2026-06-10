#!/usr/bin/env python3
"""Mixed-lobby DvD probe: all identities in ONE lobby, 2 seats each.

Unlike dvd_identity_board_probe (which pins a whole lobby to one identity at a
time), this stacks all 4 identities together — seat s plays identity s//2 — so
they share the pool and face each other, exactly the training setting. Reports
the per-identity race distribution of final boards over N lobbies. If
conditioning works, each identity's assigned tribe is over-represented vs the
others; if it's a no-op, the four rows look identical.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import src.envs  # noqa: F401
from src.bg_core.minion import Minion
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.minibg.obs import _RACE_ORDER
from src.models.ppo_policy_factory import restore_ppo_actor_critic
from src.agents.ppo_dvd_agent import PPODvDAgent, SiblingOpponent


def _race_label(m: Minion) -> str:
    return "NONE" if m.race is None else m.race.name


def _final_boards_by_seat(state):
    out = {}
    for snap in state.eliminated:
        out[snap.seat] = snap.last_board
    for seat in state.alive:
        out[seat] = tuple(state.players[seat].board)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--identity-tribes", default="DEMON,ELEMENTAL,BEAST,PIRATE")
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    args = ap.parse_args()

    ck = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    net = restore_ppo_actor_critic(
        ck["ppo_network_type"], tuple(ck["observation_shape"]),
        int(ck["num_actions"]), ck["ppo_network_kwargs"],
    )
    missing, _ = net.load_state_dict(ck["policy_state_dict"], strict=False)
    net.eval()
    N = int(ck["ppo_network_kwargs"]["num_identities"])
    tribes = [t.strip().upper() for t in args.identity_tribes.split(",")]

    print(f"checkpoint: {args.checkpoint.name}")
    print(f"num_identities={N}  untrained(missing) params={missing}")
    print(f"identity_tribes = {dict(enumerate(tribes))}")
    print(f"identity_proj      norms = "
          f"{[round(float(net.identity_proj.weight[:,i].norm()),3) for i in range(N)]}")
    if hasattr(net, "identity_slot_proj"):
        print(f"identity_slot_proj norms = "
              f"{[round(float(net.identity_slot_proj.weight[:,i].norm()),3) for i in range(N)]}")

    base = PPODvDAgent(
        network=net, num_actions=int(ck["num_actions"]),
        observation_shape=tuple(ck["observation_shape"]), observation_type="vector",
        ppo_network_type=ck["ppo_network_type"], ppo_network_kwargs=ck["ppo_network_kwargs"],
        num_identities=N, identity_tribes=tribes, device="cpu",
    )
    base.policy_net = net
    base.training = False

    seat_identity = {s: (s // 2) % N for s in range(8)}
    agents = {s: SiblingOpponent(base, identity=seat_identity[s]) for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs, learned_seats=tuple(range(8)), training_seats=(0,),
        seed=args.seed, patch_dir=args.patch_dir, obs_kind="bglike_v5",
    )

    per_id_race = {i: Counter() for i in range(N)}
    per_id_assigned_frac = defaultdict(list)

    for g in range(args.games):
        env.reset(seed=args.seed + g)
        env.drain_until_lobby_done(deterministic=True)
        boards = _final_boards_by_seat(env.state)
        for seat in range(8):
            ident = seat_identity[seat]
            board = boards[seat]
            n = len(board)
            on = 0
            for m in board:
                lab = _race_label(m)
                per_id_race[ident][lab] += 1
                if lab == tribes[ident]:
                    on += 1
            if n > 0:
                per_id_assigned_frac[ident].append(on / n)

    all_races = [r.name if r is not None else "NONE" for r in _RACE_ORDER]
    print(f"\n=== {args.games} mixed lobbies — race % of each identity's final-board minions ===")
    print("identity            " + "".join(f"{r[:5]:>7}" for r in all_races) + "   own_frac")
    for i in range(N):
        tot = sum(per_id_race[i].values()) or 1
        row = f"id{i} {tribes[i][:10]:>10} "
        for r in all_races:
            row += f"{100.0*per_id_race[i].get(r,0)/tot:>7.1f}"
        af = per_id_assigned_frac[i]
        row += f"   {(sum(af)/len(af) if af else 0.0):>8.3f}"
        print(row)

    print("\n=== assigned-tribe fraction: own vs others' avg ===")
    n_ok = 0
    for i in range(N):
        af = per_id_assigned_frac[i]
        mine = sum(af) / len(af) if af else 0.0
        others = []
        for j in range(N):
            if j == i:
                continue
            tot = sum(per_id_race[j].values()) or 1
            others.append(per_id_race[j].get(tribes[i], 0) / tot)
        om = sum(others) / len(others) if others else 0.0
        ok = mine > om + 0.05
        n_ok += ok
        print(f"id{i} {tribes[i]:>9}: own={mine:.3f}  others_avg={om:.3f}  "
              f"{'OK' if ok else '** NO DIFFERENCE **'}")
    print(f"\nidentities showing their assigned tribe above others: {n_ok}/{N}")


if __name__ == "__main__":
    main()

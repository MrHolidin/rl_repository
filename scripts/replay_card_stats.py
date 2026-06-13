#!/usr/bin/env python3
"""Self-play replays for one checkpoint: per-card bought / final-board counts.

Drives N 8-seat self-play lobbies decision-by-decision (deterministic eval,
identity rotated per game), counting for every card: how many copies were
BOUGHT from the shop and how many sit on final boards (eliminated seats use
their elimination snapshot). Writes a CSV sorted by bought count.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

torch.set_num_threads(4)

import src.envs  # noqa: F401
from src.agents.ppo_dvd_agent import PPODvDAgent
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.minibg.structured_actions import StructActionType


class _StateView:
    def __init__(self, lobby: BGLobbyEnv) -> None:
        self.state = lobby.state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=606)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    ap.add_argument("--num-identities", type=int, default=4)
    args = ap.parse_args()

    agent = PPODvDAgent.load(str(args.checkpoint), device="cpu", seed=args.seed)
    agent.eval()
    agent.training = False
    agents = {s: agent for s in range(8)}
    cfgs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        cfgs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=args.seed,
        patch_dir=args.patch_dir,
        obs_kind="bglike_v5",
    )

    bought = Counter()
    final = Counter()
    meta = {}  # name -> (tier, race)

    def note(m):
        meta[m.name or m.card_id] = (m.tier, m.race.name if m.race else "NONE")

    for g in range(args.games):
        agent.set_episode_identity(g % args.num_identities)
        env.reset(seed=args.seed + 211 * g)
        steps = 0
        while not env.lobby_done and steps < 25000:
            steps += 1
            cur = env.current_seat()
            if not env._seat_can_act(cur):
                if env.state.done:
                    break
                raise RuntimeError("stall")
            obs = env.obs_for_seat(cur)
            legal = env.legal_structured_actions_for_seat(cur)
            sv = _StateView(env)
            act, perm, _ = agent.act_structured(obs, legal, sv, deterministic=True)
            if act.type == StructActionType.BUY:
                m = env.state.players[cur].shop[act.args[0]]
                if m is not None:
                    bought[m.name or m.card_id] += 1
                    note(m)
            env.step_structured_for_seat(cur, act, board_perm=perm)
        st = env.state
        boards = {snap.seat: list(snap.last_board) for snap in st.eliminated}
        for s in st.alive:
            boards[s] = list(st.players[s].board)
        for s in range(8):
            for m in boards.get(s, []):
                final[m.name or m.card_id] += 1
                note(m)
        print(f"game {g + 1}/{args.games} done", flush=True)

    names = sorted(set(bought) | set(final), key=lambda n: -bought.get(n, 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["card", "tier", "race", "bought", "bought_per_game", "final_board", "final_per_game"]
        )
        for n in names:
            t, r = meta.get(n, (0, "?"))
            w.writerow(
                [n, t, r, bought.get(n, 0), round(bought.get(n, 0) / args.games, 3),
                 final.get(n, 0), round(final.get(n, 0) / args.games, 3)]
            )
    print(f"wrote {args.out}  cards={len(names)}  total_bought={sum(bought.values())} total_final={sum(final.values())}")


if __name__ == "__main__":
    main()

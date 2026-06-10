#!/usr/bin/env python3
"""Which non-tribe minions does each DvD identity lean on?

2 seats/identity, N games, deterministic. For every identity, tally the minions
that are NOT of its assigned tribe, separately for:
  * PURCHASES  — what it actually buys from the shop during the game;
  * FINAL TABLE — what sits on its board at the end.
Reports the top non-tribe cards per identity (count + % of that identity's
non-tribe mass), plus the overall on-tribe vs off-tribe purchase split.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401
from src.agents.ppo_dvd_agent import PPODvDAgent
from src.bg_core.minion import Minion
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.placement import placement_for_seat
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.minibg.structured_actions import StructActionType


def _race_label(m: Minion) -> str:
    return "NONE" if m.race is None else m.race.name


def _name(m: Minion) -> str:
    base = m.name or m.card_id
    return ("golden " if m.is_golden else "") + base


class BuyRecordingSibling:
    def __init__(self, base: PPODvDAgent, identity: int, seat: int, tribe: str,
                 buys: Dict[int, Counter]):
        self._base = base
        self.identity = int(identity)
        self.seat = int(seat)
        self.tribe = tribe
        self._buys = buys

    def act_structured(self, obs, legal_list, env, *, deterministic: bool = False):
        st = env.state
        seat = int(st.current_player_index)
        shop = list(getattr(st.players[seat], "shop", []) or [])
        with self._base._forced_identity_ctx(self.identity):
            chosen, perm, idx = self._base.act_structured(
                obs, legal_list, env, deterministic=True)
        if chosen.type == StructActionType.BUY and chosen.args:
            slot = int(chosen.args[0])
            if 0 <= slot < len(shop) and shop[slot] is not None:
                m = shop[slot]
                self._buys[self.identity][(_race_label(m), _name(m))] += 1
        return chosen, perm, idx

    @property
    def training(self) -> bool:
        return self._base.training

    @training.setter
    def training(self, value: bool) -> None:
        self._base.training = value

    def eval(self):
        return None


def _final_boards_by_seat(state):
    out = {}
    for snap in state.eliminated:
        out[snap.seat] = tuple(snap.last_board)
    for seat in state.alive:
        out[seat] = tuple(state.players[seat].board)
    return out


def _top_nontribe(counter: Counter, tribe: str, k: int = 6):
    off = [(name, c) for (race, name), c in counter.items() if race != tribe]
    total_off = sum(c for _, c in off)
    off.sort(key=lambda x: -x[1])
    return off[:k], total_off


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument("--identity-tribes", default="DRAGON,ELEMENTAL,BEAST,PIRATE")
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    args = ap.parse_args()

    tribes = [t.strip().upper() for t in args.identity_tribes.split(",")]
    N = len(tribes)
    base = PPODvDAgent.load(str(args.checkpoint), device="cpu", seed=args.seed)
    base.eval(); base.training = False
    seat_identity = {s: (s // 2) % N for s in range(8)}

    buys: Dict[int, Counter] = {i: Counter() for i in range(N)}      # (race,name)->count
    finals: Dict[int, Counter] = {i: Counter() for i in range(N)}
    on_off_buy = {i: [0, 0] for i in range(N)}  # [on_tribe, off_tribe]

    print(f"checkpoint: {args.checkpoint.name}")
    print(f"identity_tribes = {dict(enumerate(tribes))}  | {args.games} games, 2 seats/id\n")

    for g in range(args.games):
        agents = {s: BuyRecordingSibling(base, seat_identity[s], s, tribes[seat_identity[s]], buys)
                  for s in range(8)}
        configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
        env = BGLobbyEnv(configs, learned_seats=tuple(range(8)), training_seats=(0,),
                         seed=args.seed + g, patch_dir=args.patch_dir, obs_kind="bglike_v5")
        env.reset(seed=args.seed + g)
        env.drain_until_lobby_done(deterministic=True)
        boards = _final_boards_by_seat(env.state)
        for seat in range(8):
            i = seat_identity[seat]
            for m in boards[seat]:
                finals[i][(_race_label(m), _name(m))] += 1

    for i in range(N):
        for (race, _name_), c in buys[i].items():
            on_off_buy[i][0 if race == tribes[i] else 1] += c

    print("=== purchases: on-tribe vs off-tribe split ===")
    for i in range(N):
        on, off = on_off_buy[i]
        tot = on + off or 1
        print(f"id{i} {tribes[i]:<10} buys: on-tribe {on:>4} ({100*on/tot:4.0f}%)  "
              f"off-tribe {off:>4} ({100*off/tot:4.0f}%)")

    print("\n=== top NON-tribe PURCHASES per identity ===")
    for i in range(N):
        top, tot = _top_nontribe(buys[i], tribes[i])
        print(f"\nid{i} {tribes[i]} (off-tribe buys={tot}):")
        for name, c in top:
            print(f"    {c:>3}  ({100*c/(tot or 1):4.0f}%)  {name}")

    print("\n=== top NON-tribe minions on FINAL boards per identity ===")
    for i in range(N):
        top, tot = _top_nontribe(finals[i], tribes[i])
        ontot = sum(c for (r, _), c in finals[i].items() if r == tribes[i])
        print(f"\nid{i} {tribes[i]} (final off-tribe={tot}, on-tribe={ontot}):")
        for name, c in top:
            print(f"    {c:>3}  ({100*c/(tot or 1):4.0f}%)  {name}")


if __name__ == "__main__":
    main()

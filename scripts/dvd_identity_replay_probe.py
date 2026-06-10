#!/usr/bin/env python3
"""DvD replay probe: 2 seats/identity, N games, with per-round trajectories.

Answers, for a v7 DvD checkpoint, in the real training setting (all identities
in one lobby, 2 seats each, facing each other):

  1. Do identities COMMIT to their tribe THROUGHOUT the game, or only flip to it
     right before dying? -> per-identity own-tribe fraction by "rounds before
     the seat's last round" (if it's high 3-4 rounds before the end, it's a real
     build, not a death-bed sell-and-switch).
  2. Per-identity stats: mean placement (1=best..8=worst), mean final-board stat
     sum (atk+hp), mean final tavern tier, mean own-tribe fraction.
  3. A couple of example final boards per identity.

Replays (per-round tribe composition + final board for every seat) are written
to <out>/replays.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

import src.envs  # noqa: F401
from src.agents.ppo_dvd_agent import PPODvDAgent
from src.bg_core.minion import Minion
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.placement import placement_for_seat
from src.envs.bglike.seat_config import lobby_from_learned_seats


def _race_label(m: Minion) -> str:
    return "NONE" if m.race is None else m.race.name


def _stat_sum(board) -> int:
    return int(sum(m.raw_attack + m.max_health for m in board))


def _own_count(board, tribe: str) -> int:
    return sum(1 for m in board if _race_label(m) == tribe)


def _tribe_comp(board) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for m in board:
        c[_race_label(m)] += 1
    return dict(c)


def _board_str(board, tribe: str) -> str:
    parts = []
    for m in board:
        lab = _race_label(m)
        mark = "*" if lab == tribe else " "
        g = "g" if m.is_golden else ""
        parts.append(f"{mark}{lab[:4]}{g} {m.raw_attack}/{m.max_health}")
    return " | ".join(parts)


def _final_boards_by_seat(state) -> Dict[int, tuple]:
    out: Dict[int, tuple] = {}
    final_tier: Dict[int, int] = {}
    for snap in state.eliminated:
        out[snap.seat] = tuple(snap.last_board)
        final_tier[snap.seat] = int(snap.tavern_tier)
    for seat in state.alive:
        out[seat] = tuple(state.players[seat].board)
        final_tier[seat] = int(state.players[seat].tavern_tier)
    return out, final_tier


class RecordingSibling:
    """Wraps the shared base agent under a fixed identity for one seat, and
    records (round_number -> own-tribe frac / tavern tier) at each shop act."""

    def __init__(self, base: PPODvDAgent, identity: int, seat: int, tribe: str,
                 recorder: dict):
        self._base = base
        self.identity = int(identity)
        self.seat = int(seat)
        self.tribe = tribe
        self._rec = recorder

    def act_structured(self, obs, legal_list, env, *, deterministic: bool = False):
        st = env.state
        seat = int(st.current_player_index)
        board = st.players[seat].board
        n = len(board)
        if n > 0:
            rnd = int(st.round_number)
            # keep the LAST observation of each round (≈ end-of-shop board)
            self._rec[seat][rnd] = {
                "own_frac": _own_count(board, self.tribe) / n,
                "tavern": int(st.players[seat].tavern_tier),
                "n": n,
            }
        with self._base._forced_identity_ctx(self.identity):
            return self._base.act_structured(obs, legal_list, env, deterministic=True)

    @property
    def training(self) -> bool:
        return self._base.training

    @training.setter
    def training(self, value: bool) -> None:
        self._base.training = value

    def eval(self):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument("--identity-tribes", default="DRAGON,ELEMENTAL,BEAST,PIRATE")
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tribes = [t.strip().upper() for t in args.identity_tribes.split(",")]
    N = len(tribes)
    out_dir = args.out or args.checkpoint.parent.parent / "replay_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = PPODvDAgent.load(str(args.checkpoint), device="cpu", seed=args.seed)
    base.eval()
    base.training = False

    seat_identity = {s: (s // 2) % N for s in range(8)}  # 2 seats per identity

    print(f"checkpoint: {args.checkpoint.name}")
    print(f"identity_tribes = {dict(enumerate(tribes))}")
    print(f"seat->identity  = {seat_identity}")
    print(f"games: {args.games}\n")

    # accumulators
    per_id_place: Dict[int, List[int]] = defaultdict(list)
    per_id_statsum: Dict[int, List[int]] = defaultdict(list)
    per_id_tier: Dict[int, List[int]] = defaultdict(list)
    per_id_ownfrac: Dict[int, List[float]] = defaultdict(list)
    # commitment: own-frac indexed by "rounds before the seat's LAST round"
    per_id_byback: Dict[int, Dict[int, List[float]]] = {i: defaultdict(list) for i in range(N)}
    examples: Dict[int, List[str]] = {i: [] for i in range(N)}
    replays: List[dict] = []

    for g in range(args.games):
        recorder: Dict[int, Dict[int, dict]] = {s: {} for s in range(8)}
        agents = {s: RecordingSibling(base, seat_identity[s], s, tribes[seat_identity[s]], recorder)
                  for s in range(8)}
        configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
        env = BGLobbyEnv(configs, learned_seats=tuple(range(8)), training_seats=(0,),
                         seed=args.seed + g, patch_dir=args.patch_dir, obs_kind="bglike_v5")
        env.reset(seed=args.seed + g)
        env.drain_until_lobby_done(deterministic=True)

        boards, final_tier = _final_boards_by_seat(env.state)
        game_rec: Dict[str, Any] = {"game": g, "seats": {}}
        for seat in range(8):
            ident = seat_identity[seat]
            tribe = tribes[ident]
            board = boards[seat]
            place = placement_for_seat(env.state, seat)
            ssum = _stat_sum(board)
            tier = final_tier[seat]
            n = len(board)
            ownf = _own_count(board, tribe) / n if n else 0.0

            per_id_place[ident].append(place)
            per_id_statsum[ident].append(ssum)
            per_id_tier[ident].append(tier)
            per_id_ownfrac[ident].append(ownf)

            # commitment trajectory: align by rounds-before-last
            rounds = sorted(recorder[seat].keys())
            if rounds:
                last = rounds[-1]
                for rnd in rounds:
                    back = last - rnd  # 0 = last round, 1 = one before, ...
                    if back <= 6:
                        per_id_byback[ident][back].append(recorder[seat][rnd]["own_frac"])

            if len(examples[ident]) < 3 and n > 0:
                examples[ident].append(
                    f"  g{g} seat{seat} place={place} tier={tier} "
                    f"own={ownf:.0%} stat_sum={ssum}\n      {_board_str(board, tribe)}")

            game_rec["seats"][seat] = {
                "identity": ident, "tribe": tribe, "placement": place,
                "final_tier": tier, "final_own_frac": round(ownf, 3),
                "final_stat_sum": ssum, "final_tribe_comp": _tribe_comp(board),
                "round_traj": {int(r): {"own_frac": round(recorder[seat][r]["own_frac"], 3),
                                         "tavern": recorder[seat][r]["tavern"],
                                         "n": recorder[seat][r]["n"]}
                                for r in sorted(recorder[seat].keys())},
            }
        replays.append(game_rec)
        print(f"game {g}: placements " +
              ", ".join(f"s{seat}(id{seat_identity[seat]})={placement_for_seat(env.state, seat)}"
                        for seat in range(8)))

    (out_dir / "replays.json").write_text(json.dumps(replays, indent=2))

    # ---------------- per-identity stats ----------------
    print("\n=== per-identity stats (over {} games x 2 seats = {} boards each) ==="
          .format(args.games, args.games * 2))
    print(f"{'id/tribe':<16}{'mean_place':>11}{'mean_statsum':>14}{'mean_tier':>11}{'mean_own%':>11}")
    for i in range(N):
        pl = per_id_place[i]
        print(f"id{i} {tribes[i]:<11}"
              f"{np.mean(pl):>11.2f}"
              f"{np.mean(per_id_statsum[i]):>14.1f}"
              f"{np.mean(per_id_tier[i]):>11.2f}"
              f"{100*np.mean(per_id_ownfrac[i]):>10.1f}%")

    # ---------------- commitment over time ----------------
    print("\n=== own-tribe fraction by rounds-before-elimination (commitment check) ===")
    print("  (if own% is already high 2-4 rounds before the end, it's a real build,")
    print("   not a last-second sell-and-switch)\n")
    header = "back→  " + "".join(f"{b:>8}" for b in range(6, -1, -1))
    print(f"{'id/tribe':<16}{header}")
    for i in range(N):
        row = f"id{i} {tribes[i]:<11}"
        for back in range(6, -1, -1):
            vals = per_id_byback[i].get(back, [])
            row += f"{(100*np.mean(vals)):>7.0f}%" if vals else f"{'-':>8}"
        print(row)
    print("\n(columns: '6' = 6 rounds before last shop ... '0' = final shop round)")

    # ---------------- example final boards ----------------
    print("\n=== example final boards (own-tribe minions marked with *) ===")
    for i in range(N):
        print(f"\nid{i} {tribes[i]}:")
        for ex in examples[i]:
            print(ex)

    print(f"\nreplays written to {out_dir / 'replays.json'}")


if __name__ == "__main__":
    main()

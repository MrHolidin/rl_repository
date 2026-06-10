#!/usr/bin/env python3
"""Analyze a DvD JSONL replay (from scripts/dvd_generate_replays.py).

Reads the shared-format replay and reports stats WITHOUT re-simulating. Current
report: how often each identity assembles triples (forged goldens, marked
``from_triple_merge``), with on-tribe vs off-tribe split and timing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", type=Path, help="JSONL replay path")
    args = ap.parse_args()

    with open(args.replay) as f:
        header = json.loads(f.readline())
    tribes = [t.upper() for t in header["identity_tribes"]]
    seat_identity = {int(k): v for k, v in header["seat_identity"].items()}
    N = len(tribes)
    games = int(header.get("games", 0))

    # per (ep, seat): running multiset of forged-golden card_ids seen on board;
    # an increase => a new triple was just forged.
    prev: dict = defaultdict(Counter)
    triples_on = defaultdict(int)    # identity -> on-tribe triples
    triples_off = defaultdict(int)   # identity -> off-tribe triples
    triple_round = defaultdict(list)  # identity -> list of round numbers
    seatgames = set()                 # (ep, seat) actually seen

    with open(args.replay) as f:
        f.readline()
        for line in f:
            r = json.loads(line)
            if r.get("type") != "frame":
                continue
            ep = r["ep"]
            st = r["state"]
            rnd = int(st.get("round", 0))
            for seat_s, p in st["players"].items():
                seat = int(seat_s)
                seatgames.add((ep, seat))
                forged = Counter()
                race_of = {}
                for m in p.get("board", []):
                    if m and m.get("from_triple_merge"):
                        cid = m["card_id"]
                        forged[cid] += 1
                        race_of[cid] = m.get("race")
                key = (ep, seat)
                before = prev[key]
                for cid, cnt in forged.items():
                    new = cnt - before.get(cid, 0)
                    if new > 0:
                        ident = seat_identity[seat]
                        race = race_of[cid]
                        for _ in range(new):
                            if race == tribes[ident]:
                                triples_on[ident] += 1
                            else:
                                triples_off[ident] += 1
                            triple_round[ident].append(rnd)
                prev[key] = forged

    n_seatgames = {i: sum(1 for (_, s) in seatgames if seat_identity[s] == i) for i in range(N)}

    print(f"replay: {args.replay.name}  ({games} games, identities={dict(enumerate(tribes))})\n")
    print(f"{'id/tribe':<16}{'triples/board':>14}{'on-tribe':>10}{'off-tribe':>11}"
          f"{'on%':>7}{'med.round':>11}")
    for i in range(N):
        on, off = triples_on[i], triples_off[i]
        tot = on + off
        boards = n_seatgames.get(i, 0) or 1
        rounds = sorted(triple_round[i])
        med = rounds[len(rounds) // 2] if rounds else 0
        print(f"id{i} {tribes[i]:<11}"
              f"{tot / boards:>14.2f}"
              f"{on:>10}{off:>11}"
              f"{(100 * on / tot if tot else 0):>6.0f}%{med:>11}")

    tot_all = sum(triples_on.values()) + sum(triples_off.values())
    tot_boards = sum(n_seatgames.values())
    print(f"\noverall: {tot_all} triples over {tot_boards} boards "
          f"= {tot_all / tot_boards:.2f}/board ({tot_all / max(games,1):.1f}/game across 8 seats)")


if __name__ == "__main__":
    main()

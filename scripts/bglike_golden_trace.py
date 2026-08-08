#!/usr/bin/env python3
"""Golden traces for the BGLike rules engine.

WHEN TO RUN THIS
================
Only when you are about to change, or have just changed, the **rules engine** --
``bg_core``, ``bg_catalog``, ``bg_recruitment``, ``bg_player_turn``, ``bg_combat``,
``bg_lobby``, or the observation builders in ``src/envs/bglike``.

It exists for one job: proving that a refactor which is supposed to change
nothing changed nothing. The intended use is a pair of runs around a change --

    python -m scripts.bglike_golden_trace record --out tests/fixtures/golden_bg.json
    ... do the refactor ...
    python -m scripts.bglike_golden_trace verify --trace tests/fixtures/golden_bg.json

Do NOT put this in the default test suite. A trace is a handful of full 8-seat
lobbies, which is seconds of work, and its value is concentrated in the moment
of a refactor rather than spread over every commit. The regular suite already
covers behaviour; this covers *equivalence*.

WHAT IT PINS
============
After every action, a canonical digest of the whole lobby -- each seat's board
(card id, stats, keywords, golden, ability count), hand, shop, gold, tier,
health, pending choice, plus round and elimination order -- is folded into a
hash chain. Observations for every registered ``obs_kind`` are hashed on the
same steps. A divergence therefore fails at the first step where anything
differs, not at the end, and ``verify`` prints the step, the seat and the field.

The driver is the scripted bots, not a checkpoint: their behaviour is fixed by
this repository rather than by a file that may be regenerated or moved, so a
trace stays meaningful across model versions.

THE FAILURE THIS IS REALLY FOR
==============================
RNG consumption order. A change can be perfectly correct and still shift the
random stream -- a new mechanic drawing from the same generator moves every
later draw, and old seeds stop reproducing. Nothing else catches that, and
review reliably misses it. New mechanics must take their own generator.

Coverage is reported on ``record`` (triples, magnetise, discover, golden,
ghosts, eliminations): a trace that never triggered a mechanic proves nothing
about it, so check the summary rather than trusting the seed count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import src.envs  # noqa: F401
from src.envs.bglike.heuristic_bots.bots import make_bot
from src.envs.bglike.lobby_env import BGLobbyEnv, _VALID_OBS_KINDS
from src.envs.bglike.seat_config import SeatConfig, SeatKind

DEFAULT_PATCH = "data/bgcore/19_6_0_74257"
# One bot per seat, fixed. Mixed on purpose: t1_random parks on tier 1 and
# triples a lot, t_up_random rushes tiers, structured and elemental build real
# boards -- between them the trace exercises paths a single bot would not.
DEFAULT_SEATS = (
    "structured", "t_up_random", "t1_random", "elemental",
    "structured", "t1_random", "t_up_random", "elemental",
)
DEFAULT_SEEDS = (101, 202, 303)


class _BotView:
    """What a scripted bot reads off the lobby for one seat.

    The bots take a per-seat view rather than an agent interface, so the trace
    drives them the same way ``bglike_elemental_vs_structured`` does.
    """

    def __init__(self, lobby, seat: int) -> None:
        self._lobby = lobby
        self._seat = seat
        self._mask: Optional[np.ndarray] = None

    @property
    def state(self):
        return self._lobby.state

    @property
    def _game(self):
        return self._lobby._game

    @property
    def patch(self):
        return self._lobby._game._patch

    @property
    def rl_pending(self):
        return self._lobby.rl_pending_for_seat(self._seat)

    def current_player(self) -> int:
        return self._seat

    def set_mask_override(self, mask) -> None:
        self._mask = None if mask is None else np.asarray(mask, dtype=bool)

    @property
    def legal_actions_mask(self) -> np.ndarray:
        if self._mask is not None:
            return self._mask
        return self._lobby.legal_mask_for_seat(self._seat)


def _minion_digest(m) -> tuple:
    if m is None:
        return ()
    return (
        m.card_id,
        int(m.raw_attack), int(m.max_health),
        int(m.base_attack), int(m.base_health),
        bool(m.has_shield), bool(m.is_golden),
        tuple(sorted(k.name for k in m.all_keywords)),
        len(m.abilities or ()),
    )


def _state_digest(state) -> str:
    """Canonical digest of everything a rules change could plausibly move."""
    rows: List[Any] = [int(state.round_number), int(state.combat_round)]
    for seat, p in enumerate(state.players):
        rows.append((
            seat,
            int(p.gold), int(p.health), int(p.tavern_tier),
            tuple(_minion_digest(m) for m in p.board),
            tuple(_minion_digest(m) for m in p.hand),
            tuple(_minion_digest(m) for m in p.shop),
            None if p.pending_choice is None else str(p.pending_choice.kind),
        ))
    rows.append(tuple((s.seat, int(s.tavern_tier)) for s in state.eliminated))
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, default=str).encode()
    ).hexdigest()[:32]


def _coverage(state, cov: Counter) -> None:
    for p in state.players:
        for m in list(p.board) + list(p.hand):
            if m is None:
                continue
            if m.is_golden:
                cov["golden"] += 1
            if getattr(m, "from_triple_merge", False):
                cov["triple"] += 1
            if len(m.abilities or ()) > 2:
                cov["magnetised"] += 1
        if p.pending_choice is not None:
            cov["pending_choice"] += 1
    cov["eliminations"] = max(cov["eliminations"], len(state.eliminated))


def _run_one(seed: int, *, patch_dir: str, seat_bots: Tuple[str, ...],
             obs_kinds: Tuple[str, ...], max_steps: int,
             cov: Counter) -> Dict[str, Any]:
    bots = [make_bot(name, seed=seed * 100 + s) for s, name in enumerate(seat_bots)]
    from src.agents.random_agent import RandomAgent

    # Seat controllers are irrelevant: every action below is chosen by a
    # scripted bot and applied with step_action. The lobby just needs a valid
    # seat config, so seat 0 carries a placeholder agent that is never asked.
    env = BGLobbyEnv(
        [SeatConfig(SeatKind.LEARNED, RandomAgent(seed=seed))]
        + [SeatConfig(SeatKind.RANDOM)] * 7,
        learned_seats=[0], seed=seed, patch_dir=patch_dir,
        obs_kind=obs_kinds[0], with_heroes=True,
    )
    env.reset(seed=seed)
    views = [_BotView(env, s) for s in range(8)]

    chain = hashlib.sha256()
    steps: List[Dict[str, Any]] = []
    n = 0
    while not env.lobby_done and n < max_steps:
        seat = env.current_seat()
        if not env._seat_can_act(seat):
            break
        obs_digest = {}
        for kind in obs_kinds:
            v = env.obs_for_seat(seat) if kind == env.obs_kind else None
            if v is None:
                continue
            obs_digest[kind] = hashlib.sha256(
                np.ascontiguousarray(v, dtype=np.float32).tobytes()
            ).hexdigest()[:16]

        mask = env.legal_mask_for_seat(seat)
        view = views[seat]
        view.set_mask_override(mask)
        try:
            action = int(bots[seat].choose_action(view))
        finally:
            view.set_mask_override(None)
        env.step_action(seat, action)

        record = {
            "i": n, "seat": int(seat),
            "action": int(action),
            "n_legal": int(mask.sum()),
            "state": _state_digest(env.state), "obs": obs_digest,
        }
        chain.update(json.dumps(record, sort_keys=True).encode())
        steps.append(record)
        _coverage(env.state, cov)
        n += 1

    return {"seed": seed, "steps": steps, "chain": chain.hexdigest()[:32],
            "n_steps": n, "rounds": int(env.state.round_number)}


def _collect(args, cov: Counter) -> List[Dict[str, Any]]:
    kinds = tuple(args.obs_kinds)
    unknown = [k for k in kinds if k not in _VALID_OBS_KINDS]
    if unknown:
        raise SystemExit(f"unknown obs_kind(s) {unknown}; valid: {sorted(_VALID_OBS_KINDS)}")
    out = []
    for seed in args.seeds:
        t = _run_one(seed, patch_dir=args.patch_dir, seat_bots=tuple(args.seats),
                     obs_kinds=kinds, max_steps=args.max_steps, cov=cov)
        print(f"  seed {seed}: {t['n_steps']} steps, {t['rounds']} rounds, chain {t['chain']}")
        out.append(t)
    return out


def record(args: argparse.Namespace) -> int:
    cov: Counter = Counter()
    traces = _collect(args, cov)
    payload = {
        "patch_dir": args.patch_dir,
        "seats": list(args.seats),
        "seeds": list(args.seeds),
        "obs_kinds": list(args.obs_kinds),
        "traces": traces,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    total = sum(t["n_steps"] for t in traces)
    print(f"\nwrote {args.out}: {len(traces)} lobbies, {total} steps")
    print("coverage (a mechanic at 0 is NOT pinned by this trace):")
    for k in ("triple", "golden", "magnetised", "pending_choice", "eliminations"):
        print(f"  {k:<16} {cov.get(k, 0)}")
    return 0


def verify(args: argparse.Namespace) -> int:
    ref = json.loads(args.trace.read_text())
    ns = argparse.Namespace(
        patch_dir=ref["patch_dir"], seats=ref["seats"], seeds=ref["seeds"],
        obs_kinds=ref["obs_kinds"], max_steps=args.max_steps,
    )
    cov: Counter = Counter()
    now = _collect(ns, cov)

    bad = 0
    for old, new in zip(ref["traces"], now):
        if old["chain"] == new["chain"]:
            print(f"  seed {old['seed']}: OK ({new['n_steps']} steps)")
            continue
        bad += 1
        print(f"  seed {old['seed']}: DIVERGED")
        for a, b in zip(old["steps"], new["steps"]):
            if a == b:
                continue
            print(f"    first divergence at step {a['i']} (seat {a['seat']})")
            for field in ("action", "n_legal", "state"):
                if a.get(field) != b.get(field):
                    print(f"      {field}: was {a.get(field)!r} -> now {b.get(field)!r}")
            for kind in a.get("obs", {}):
                if a["obs"].get(kind) != b.get("obs", {}).get(kind):
                    print(f"      obs[{kind}]: was {a['obs'][kind]} -> now {b['obs'].get(kind)}")
            break
        else:
            print(f"    lengths differ: was {len(old['steps'])} steps, now {len(new['steps'])}")
    if bad:
        print(f"\n{bad}/{len(ref['traces'])} lobbies diverged -- the change is NOT behaviour-preserving.")
        print("If the change was meant to alter behaviour, re-record. If it was a refactor,")
        print("check RNG draw order first: a new draw from an existing generator shifts everything after it.")
        return 1
    print(f"\nall {len(ref['traces'])} lobbies identical")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--patch-dir", default=DEFAULT_PATCH)
        p.add_argument("--max-steps", type=int, default=8000)

    r = sub.add_parser("record", help="record a baseline trace (run BEFORE the refactor)")
    common(r)
    r.add_argument("--out", type=Path, default=Path("tests/fixtures/golden_bg.json"))
    r.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    r.add_argument("--seats", nargs=8, default=list(DEFAULT_SEATS))
    r.add_argument("--obs-kinds", nargs="+", default=["bglike_v6_heroes"])
    r.set_defaults(fn=record)

    v = sub.add_parser("verify", help="replay and compare against a recorded trace")
    common(v)
    v.add_argument("--trace", type=Path, default=Path("tests/fixtures/golden_bg.json"))
    v.set_defaults(fn=verify)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

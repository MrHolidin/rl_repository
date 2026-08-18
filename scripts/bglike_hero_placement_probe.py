#!/usr/bin/env python3
"""Mean placement per hero: one fixed hero per seat, one checkpoint on all 8 seats.

Every lobby seats the same eight heroes, seat i always getting HEROES[i], so a
lobby is a complete round-robin of the set and the placements sum to 36 by
construction — the eight means are directly comparable and average to 4.5.

The tribe-preference vector is forced to zero (identity off), so a v13 net that
reads one sees a neutral vector instead of a random draw it would otherwise
chase.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

torch.set_num_threads(1)  # batch=1 forwards are faster single-threaded

import src.envs  # noqa: F401
from src.bg_catalog.patch_context import load_patch_context
from src.bg_recruitment import hero_passives
from src.envs.bglike import game as game_mod
from src.envs.bglike import lobby_env as lobby_mod
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.placement import placement_for_seat
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.evaluation.eval_checkpoints import load_training_agent_checkpoint
from src.training.bg_network_policy import obs_kind_for_checkpoint

NUM_SEATS = 8


def t_crit_975(n: int) -> float:
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262, 20: 2.093, 30: 2.045, 60: 2.000}
    if n <= 1:
        return float("nan")
    for k in sorted(table):
        if n - 1 <= k:
            return table[k]
    return 1.96


def mean_ci(vals: List[float]) -> Dict[str, float]:
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci95_half": float("nan")}
    mu = sum(vals) / n
    if n < 2:
        return {"n": n, "mean": mu, "ci95_half": float("nan")}
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    return {"n": n, "mean": mu, "ci95_half": t_crit_975(n) * math.sqrt(var / n)}


def pin_heroes(hero_ids: List[str], patch) -> None:
    """Seat i gets hero_ids[i], assigned where the random draw used to happen.

    Patching the assignment (rather than overwriting ``player.hero`` after
    reset) keeps the hero in place before the opening shop fill and before
    ``apply_hero_on_game_start`` — Millificent shapes that first shop and the
    start-of-game powers hand out their tokens, so a later overwrite would
    silently drop both.
    """
    state = {"i": 0}

    def assign_fixed(player, *, patch, rng):  # noqa: A002 - mirrors the real signature
        player.hero = patch.heroes[hero_ids[state["i"] % len(hero_ids)]]
        state["i"] += 1

    hero_passives.assign_random_hero = assign_fixed  # type: ignore[assignment]
    game_mod.draw_tribe_pref = lambda rng: (0.0,) * 7  # identity off
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--num-games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--patch-dir", type=str, default="data/bgcore/19_6_0_74257")
    ap.add_argument("--shop-excluded-count", type=int, default=1)
    ap.add_argument("--drain-steps", type=int, default=40000)
    ap.add_argument("--heroes", type=str, default=None,
                    help="8 comma-separated hero ids (default: first 8 of the patch pool, sorted)")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    patch = load_patch_context(args.patch_dir)
    pool = sorted(patch.hero_pool_ids)
    if args.heroes:
        hero_ids = [h.strip() for h in args.heroes.split(",") if h.strip()]
    else:
        hero_ids = pool[:NUM_SEATS]
    if len(hero_ids) != NUM_SEATS:
        raise SystemExit(f"need exactly {NUM_SEATS} heroes, got {len(hero_ids)}")
    unknown = [h for h in hero_ids if h not in patch.heroes]
    if unknown:
        raise SystemExit(f"unknown hero ids: {unknown} (pool: {pool})")

    ck = args.checkpoint.resolve()
    obs_kind = obs_kind_for_checkpoint(ck)
    with_heroes = True
    with_pref = obs_kind == "bglike_v7_pref"
    agent = load_training_agent_checkpoint(ck, device=args.device, seed=args.seed)
    if hasattr(agent, "eval"):
        agent.eval()
    print(f"checkpoint {ck.name} | obs {obs_kind} | heroes {hero_ids}", flush=True)

    counter = pin_heroes(hero_ids, patch)
    learned = tuple(range(NUM_SEATS))
    agents = {s: agent for s in learned}

    old_cap = lobby_mod.MAX_DRAIN_STEPS
    lobby_mod.MAX_DRAIN_STEPS = int(args.drain_steps)
    by_hero: Dict[str, List[int]] = defaultdict(list)
    games: List[dict] = []
    try:
        for g in range(args.num_games):
            seed = args.seed + g
            counter["i"] = 0
            configs = lobby_from_learned_seats(learned, agent_by_seat=agents)
            env = BGLobbyEnv(
                configs, learned_seats=learned, training_seats=learned, seed=seed,
                patch_dir=args.patch_dir, obs_kind=obs_kind,
                with_heroes=with_heroes, with_tribe_pref=with_pref,
                shop_excluded_count=args.shop_excluded_count,
            )
            env.reset(seed=seed)
            st = env.state
            seen = [st.players[s].hero.hero_id if st.players[s].hero else None for s in range(NUM_SEATS)]
            if seen != hero_ids:
                raise RuntimeError(f"hero pinning failed: {seen} != {hero_ids}")
            env.drain_until_lobby_done(deterministic=True)
            row = {}
            tiers = {}
            st_end = env.state
            final_by_seat = {}
            for snap in st_end.eliminated:
                final_by_seat[snap.seat] = snap.tavern_tier
            for s in st_end.alive:
                final_by_seat[s] = st_end.players[s].tavern_tier
            for s in range(NUM_SEATS):
                place = placement_for_seat(env.state, s)
                by_hero[hero_ids[s]].append(place)
                row[hero_ids[s]] = place
                # When each tier was first held, plus the tier the seat ended on:
                # "does this hero climb earlier" is a question about the rounds,
                # not only about where it finished.
                tiers[hero_ids[s]] = {
                    "final": int(final_by_seat.get(s, 0)),
                    "first_round": {int(k): int(v) for k, v in env.tier_first_round(s).items()},
                }
            games.append({"game": g, "seed": seed, "placements": row, "tiers": tiers,
                          "rounds": int(st_end.round_number)})
            if (g + 1) % 25 == 0:
                print(f"  {g + 1}/{args.num_games}", flush=True)
    finally:
        lobby_mod.MAX_DRAIN_STEPS = old_cap

    summary = {h: mean_ci([float(x) for x in by_hero[h]]) for h in hero_ids}
    payload = {
        "checkpoint": ck.name, "obs_kind": obs_kind, "heroes": hero_ids,
        "num_games": len(games), "seed": args.seed,
        "zero_tribe_pref": True, "summary": summary, "games": games,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload))

    print("\nmean placement per hero (1 = best, 8 = worst; the eight average to 4.5)")
    for h, st_ in sorted(summary.items(), key=lambda kv: kv[1]["mean"]):
        print(f"  {h:12s} {st_['mean']:.3f}  (95% CI ±{st_['ci95_half']:.3f}, n={st_['n']})")
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()

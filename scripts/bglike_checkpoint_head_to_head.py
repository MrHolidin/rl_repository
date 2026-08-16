#!/usr/bin/env python3
"""BGLike 4v4 checkpoint head-to-head: average placement by team."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

torch.set_num_threads(1)  # batch=1 forwards are faster single-threaded

import src.envs  # noqa: F401
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.lobby_env import OBS_KIND_BGLIKE_V7_PREF
from src.envs.bglike.placement import placement_for_seat
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.evaluation.eval_checkpoints import find_checkpoints, load_training_agent_checkpoint
from src.training.bg_network_policy import obs_kind_for_checkpoint

_T975 = {2: 4.302653, 5: 2.570582, 10: 2.228139, 20: 2.0860, 40: 2.0211, 80: 1.9944}


def t_crit_975(n: int) -> float:
    if n < 2:
        return float("nan")
    return _T975.get(n - 1, 1.96)


def _resolve_ckpt(checkpoint_dir: Path, prefix: str, step: int | None, path: Path | None) -> Tuple[Path, int]:
    if path is not None:
        p = path.resolve()
        from src.evaluation.eval_checkpoints import _step_from_filename

        s = _step_from_filename(p.name)
        return p, int(s if s is not None else -1)
    found = find_checkpoints(checkpoint_dir.resolve(), prefix=prefix)
    if not found:
        raise SystemExit(f"No checkpoints in {checkpoint_dir}")
    if step is not None:
        ck, st = min(found, key=lambda x: abs(x[1] - step))
        return ck, st
    return found[-1]


def run_head_to_head(
    *,
    ck_a: Path,
    ck_b: Path,
    seats_a: Sequence[int],
    seats_b: Sequence[int],
    num_games: int,
    seed: int,
    device: str,
    patch_dir: str | None = None,
    obs_kind: str | None = None,
    with_heroes: bool | None = None,
    zero_tribe_pref: bool = False,
) -> List[dict]:
    # Each checkpoint declares the observation it reads, so the two teams need
    # not share one: the lobby builds a per-seat layout. The observation is a
    # pure function of (state, seat), so seats reading different layouts of the
    # same state is well-defined. This is what lets a v12 checkpoint
    # (bglike_v6_heroes, 1123 floats) be scored against a v11_heroes one
    # (bglike_v5_heroes, 2683) in the same game.
    kind_a = obs_kind_for_checkpoint(ck_a)
    kind_b = obs_kind_for_checkpoint(ck_b)
    obs_kind_by_seat = {s: kind_a for s in seats_a}
    obs_kind_by_seat.update({s: kind_b for s in seats_b})
    lobby_kind = obs_kind or kind_a  # only covers seats no checkpoint claims
    if with_heroes is None:
        with_heroes = kind_a.endswith("_heroes") or kind_b.endswith("_heroes")
    if kind_a != kind_b:
        print(f"mixed observations: A reads {kind_a}, B reads {kind_b}", flush=True)

    agent_a = load_training_agent_checkpoint(ck_a, device=device, seed=seed)
    agent_b = load_training_agent_checkpoint(ck_b, device=device, seed=seed + 1)
    for ag in (agent_a, agent_b):
        if hasattr(ag, "eval"):
            ag.eval()
    agents = {}
    for s in seats_a:
        agents[s] = agent_a
    for s in seats_b:
        agents[s] = agent_b
    learned = tuple(sorted(set(seats_a) | set(seats_b)))
    configs = lobby_from_learned_seats(learned, agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=learned,
        training_seats=learned,
        seed=seed,
        patch_dir=patch_dir,
        obs_kind=lobby_kind,
        obs_kind_by_seat=obs_kind_by_seat,
        with_heroes=with_heroes,
        # A seat reading the preference layout needs the block to exist; seats on
        # older layouts never see it, so turning it on is free for them.
        with_tribe_pref=any(
            k == OBS_KIND_BGLIKE_V7_PREF for k in obs_kind_by_seat.values()
        ),
    )
    games: List[dict] = []
    import time as _time

    for g in range(num_games):
        _t0 = _time.perf_counter()
        env.reset(seed=seed + g)
        if zero_tribe_pref:
            # Score the identity-trained net on strength alone: every seat gets
            # an all-zero preference vector, so the block carries no signal and
            # no purchase is worth more than another. Vectors are drawn once at
            # reset, so overwriting them here holds for the whole game.
            for p in env.state.players:
                if getattr(p, "tribe_pref", None) is not None:
                    p.tribe_pref = tuple(0.0 for _ in p.tribe_pref)
        env.drain_until_lobby_done(deterministic=True)
        print(
            f"  game {g + 1}/{num_games}: {_time.perf_counter() - _t0:.1f}s",
            flush=True,
        )
        st = env.state
        placements = {s: placement_for_seat(st, s) for s in range(8)}
        games.append(
            {
                "game": g,
                "seed": seed + g,
                "winner": st.winner,
                "placements": placements,
                "team_a_placements": [placements[s] for s in seats_a],
                "team_b_placements": [placements[s] for s in seats_b],
                "boards": {s: board_snapshot(st, s) for s in range(8)},
                "tier_round": {s: env.tier_first_round(s) for s in range(8)},
            }
        )
    return games


def board_snapshot(state, seat: int) -> dict:
    """Final tavern tier and board composition for one seat.

    Placement alone cannot separate "plays the same game slightly better" from
    "plays a different game": the battle-shaping arm was within noise on lobby
    wins while reaching tier 6 in 0.5% of games against the baseline's 20.7%.
    """
    elim = {snap.seat: snap.tavern_tier for snap in state.eliminated}
    player = state.players[seat]
    tribes: Counter = Counter()
    tiers: Counter = Counter()
    keywords: Counter = Counter()
    golden = attack = health = 0
    for m in player.board:
        if m is None:
            continue
        tribes[getattr(getattr(m, "race", None), "name", "NONE")] += 1
        tiers[int(m.tier)] += 1
        golden += int(bool(m.is_golden))
        attack += int(m.raw_attack)
        health += int(m.max_health)
        for kw in m.all_keywords:
            keywords[kw.name] += 1
    return {
        "tier": int(elim.get(seat, player.tavern_tier)),
        "size": int(sum(tiers.values())),
        "golden": golden,
        "attack": attack,
        "health": health,
        "tribes": dict(tribes),
        "tiers": {str(k): v for k, v in tiers.items()},
        "keywords": dict(keywords),
    }


def summarize_paired(games: List[dict], seats_a, seats_b) -> dict:
    """Per-lobby paired difference of team mean placement.

    Placements in a lobby sum to 36, so the two team means always sum to 9.0
    and are perfectly anti-correlated. Independent per-team CIs therefore
    overstate the uncertainty; the paired difference is the honest statistic.
    """
    diffs = [
        sum(g["team_a_placements"]) / len(seats_a) - sum(g["team_b_placements"]) / len(seats_b)
        for g in games
    ]
    n = len(diffs)
    mean = sum(diffs) / n if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "ci95_half": float("nan"), "diffs": diffs}
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return {"n": n, "mean": mean, "ci95_half": t_crit_975(n) * math.sqrt(var / n), "diffs": diffs}


def summarize_placements(values: List[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci95_half": float("nan")}
    mu = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mu, "ci95_half": float("nan")}
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    half = t_crit_975(n) * math.sqrt(var / n)
    return {"n": n, "mean": mu, "ci95_half": half}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "runs/bglike/dist_ppo_005/checkpoints",
    )
    ap.add_argument("--prefix", type=str, default="dist_bglike_ppo")
    ap.add_argument("--ckpt-a", type=Path, default=None, help="Team A checkpoint path")
    ap.add_argument("--ckpt-b", type=Path, default=None, help="Team B checkpoint path")
    ap.add_argument("--step-a", type=int, default=5_000_000, help="Nearest step for team A")
    ap.add_argument("--step-b", type=int, default=2_500_000, help="Nearest step for team B")
    ap.add_argument("--seats-a", type=str, default="0,1,2,3", help="Seats for team A")
    ap.add_argument("--seats-b", type=str, default="4,5,6,7", help="Seats for team B")
    ap.add_argument("--label-a", type=str, default="5000k")
    ap.add_argument("--label-b", type=str, default="2500k")
    ap.add_argument("--num-games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--patch-dir", type=str, default=None,
        help="patch package dir (required for patch-pinned checkpoints)",
    )
    ap.add_argument(
        "--obs-kind", type=str, default=None,
        help="override the lobby default; each checkpoint's own layout is "
             "detected from its network type and applied per seat",
    )
    ap.add_argument(
        "--zero-tribe-pref", action="store_true",
        help="force every seat's tribe-preference vector to zero (identity off)",
    )
    ap.add_argument(
        "--with-heroes", dest="with_heroes", action="store_true", default=None,
        help="force heroes on (auto-enabled when either net reads a hero obs)",
    )
    args = ap.parse_args()

    seats_a = tuple(int(x.strip()) for x in args.seats_a.split(",") if x.strip())
    seats_b = tuple(int(x.strip()) for x in args.seats_b.split(",") if x.strip())
    if set(seats_a) & set(seats_b):
        raise SystemExit("seat sets must not overlap")
    if len(seats_a) != 4 or len(seats_b) != 4:
        raise SystemExit("expected 4 seats per team")

    ck_a, step_a = _resolve_ckpt(args.checkpoint_dir, args.prefix, args.step_a, args.ckpt_a)
    ck_b, step_b = _resolve_ckpt(args.checkpoint_dir, args.prefix, args.step_b, args.ckpt_b)

    games = run_head_to_head(
        ck_a=ck_a,
        ck_b=ck_b,
        seats_a=seats_a,
        seats_b=seats_b,
        num_games=args.num_games,
        seed=args.seed,
        device=args.device,
        patch_dir=args.patch_dir,
        obs_kind=args.obs_kind,
        with_heroes=args.with_heroes,
        zero_tribe_pref=bool(args.zero_tribe_pref),
    )
    all_a = [float(p) for g in games for p in g["team_a_placements"]]
    all_b = [float(p) for g in games for p in g["team_b_placements"]]
    sum_a = summarize_placements(all_a)
    sum_b = summarize_placements(all_b)
    wins_a = sum(1 for g in games if g["winner"] in seats_a)
    wins_b = sum(1 for g in games if g["winner"] in seats_b)

    print(f"team A ({args.label_a}): {ck_a.name}  seats={seats_a}")
    print(f"team B ({args.label_b}): {ck_b.name}  seats={seats_b}")
    print(f"games: {args.num_games}  seed: {args.seed}")
    print(
        f"mean placement {args.label_a}: {sum_a['mean']:.3f} "
        f"(95% CI ±{sum_a['ci95_half']:.3f}, n={sum_a['n']})  "
        f"[1=best, 8=worst]"
    )
    print(
        f"mean placement {args.label_b}: {sum_b['mean']:.3f} "
        f"(95% CI ±{sum_b['ci95_half']:.3f}, n={sum_b['n']})"
    )
    print(f"lobby wins: {args.label_a} {wins_a}/{args.num_games}, {args.label_b} {wins_b}/{args.num_games}")

    paired = summarize_paired(games, seats_a, seats_b)
    verdict = (
        f"{args.label_a} better" if paired["mean"] + paired["ci95_half"] < 0
        else f"{args.label_b} better" if paired["mean"] - paired["ci95_half"] > 0
        else "not separated"
    )
    print(
        f"\npaired per-lobby delta ({args.label_a} - {args.label_b}): "
        f"{paired['mean']:+.3f} +/- {paired['ci95_half']:.3f}  "
        f"(95% CI, n={paired['n']} lobbies)  -> {verdict}"
    )

    def team_boards(seats):
        return [g["boards"][s] for g in games for s in seats]

    ba, bb = team_boards(seats_a), team_boards(seats_b)
    print(f"\n{'':<24}{args.label_a:>12}{args.label_b:>12}")
    for name, key in (("mean final tier", "tier"), ("minions", "size"),
                      ("golden", "golden"), ("attack", "attack"), ("health", "health")):
        print(f"{name:<24}"
              + "".join(f"{sum(x[key] for x in b) / max(len(b), 1):>12.2f}" for b in (ba, bb)))
    def team_rounds(seats, tier):
        vals = [g["tier_round"][s][tier] for g in games for s in seats
                if tier in g["tier_round"][s]]
        return sum(vals) / len(vals) if vals else float("nan")

    for tier in (4, 5, 6):
        print(f"{'reached t' + str(tier) + ' (%)':<24}"
              + "".join(f"{100 * sum(x['tier'] >= tier for x in b) / max(len(b), 1):>12.1f}"
                        for b in (ba, bb)))
        print(f"{'  first at round':<24}"
              + "".join(f"{team_rounds(seats, tier):>12.1f}" for seats in (seats_a, seats_b)))
    tribes = sorted({t for b in (ba, bb) for x in b for t in x["tribes"]})
    print(f"\n{'tribe share (%)':<24}{args.label_a:>12}{args.label_b:>12}")
    for t in tribes:
        print(f"{t:<24}" + "".join(
            f"{100 * sum(x['tribes'].get(t, 0) for x in b) / max(sum(sum(x['tribes'].values()) for x in b), 1):>12.1f}"
            for b in (ba, bb)))

    payload = {
        "team_a": {"label": args.label_a, "checkpoint": ck_a.name, "step": step_a, "seats": seats_a},
        "team_b": {"label": args.label_b, "checkpoint": ck_b.name, "step": step_b, "seats": seats_b},
        "num_games": args.num_games,
        "seed": args.seed,
        "summary": {
            "mean_placement_a": sum_a,
            "mean_placement_b": sum_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
        },
        "games": games,
    }
    out = args.out_json
    if out is None:
        out = (
            args.checkpoint_dir.parent
            / f"head_to_head_{args.label_a}_vs_{args.label_b}_{args.num_games}g.json"
        )
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()

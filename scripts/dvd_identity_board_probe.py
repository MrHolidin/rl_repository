#!/usr/bin/env python3
"""Probe a DvD/v7 checkpoint: do the identities actually build different tribes?

For each population identity i, pin the agent to identity i (single-identity
mode) and play it on EVERY seat of N self-play lobbies, then report the tribe
composition of identity-i's final boards. If the identities diverged, their
tribe histograms differ; if they collapsed to one good-stuff pile, they look
identical (and dominated by NONE).

This answers the "noise vs commit" question directly from boards, without
trusting the in-training dvd_* metrics: we read what each identity *actually
builds*, averaged over games (so per-game noise washes out).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401
from src.bg_core.minion import Minion
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import lobby_from_learned_seats
import torch


def _load_dvd_agent(path: Path, *, device: str, seed: int):
    """Load a checkpoint as PPODvDAgent if its network is v7, else fail loud.

    NOTE: the shared ``load_training_agent_checkpoint`` dispatches on
    ``agent_kind`` only, and DvD agents inherit ``agent_kind=
    'ppo_minibg_structured'`` → they load as the *base* structured agent and
    lose the identity API. This probe loads the right class directly.
    """
    from src.agents.ppo_dvd_agent import PPODvDAgent

    ckpt = torch.load(str(path), map_location="cpu")
    nt = str(ckpt.get("ppo_network_type", ""))
    if nt != "bglike_structured_v7":
        raise SystemExit(f"checkpoint network is {nt!r}, not bglike_structured_v7")
    return PPODvDAgent.load(str(path), device=device, seed=seed)


def _race_label(m: Minion) -> str:
    return "NONE" if m.race is None else m.race.name


def _final_boards_by_seat(state) -> Dict[int, tuple]:
    out: Dict[int, tuple] = {}
    for snap in state.eliminated:
        out[snap.seat] = snap.last_board
    for seat in state.alive:
        out[seat] = tuple(state.players[seat].board)
    return out


def probe_identity(agent, identity: int, *, num_games: int, seed: int, patch_dir: str) -> dict:
    # Pin every seat to this identity (single-identity mode).
    agent.set_episode_identity(identity)
    agent.eval()
    agents = {s: agent for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=seed,
        patch_dir=patch_dir,
        obs_kind="bglike_v5",
    )
    race_counts: Counter = Counter()
    n_minions = 0
    n_boards = 0
    for g in range(num_games):
        # Re-pin each game: drain may not touch agent identity, but reset is safe.
        agent.set_episode_identity(identity)
        env.reset(seed=seed + g)
        env.drain_until_lobby_done(deterministic=True)
        boards = _final_boards_by_seat(env.state)
        for seat in range(8):
            n_boards += 1
            for m in boards[seat]:
                race_counts[_race_label(m)] += 1
                n_minions += 1
    total = sum(race_counts.values()) or 1
    pct = {k: round(100.0 * v / total, 1) for k, v in sorted(race_counts.items())}
    return {
        "identity": identity,
        "n_boards": n_boards,
        "n_minions": n_minions,
        "race_pct": pct,
        "dominant_non_none": max(
            ((k, v) for k, v in race_counts.items() if k != "NONE"),
            key=lambda kv: kv[1],
            default=("-", 0),
        )[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--num-identities", type=int, default=4)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    args = ap.parse_args()

    agent = _load_dvd_agent(args.checkpoint, device=args.device, seed=args.seed)
    if not hasattr(agent, "set_episode_identity"):
        raise SystemExit("checkpoint is not a DvD/v7 agent (no set_episode_identity)")

    print(f"checkpoint: {args.checkpoint.name}")
    print(f"identities: {args.num_identities}, games/identity: {args.games}\n")
    rows: List[dict] = []
    for i in range(args.num_identities):
        r = probe_identity(
            agent, i, num_games=args.games, seed=args.seed + i * 100000,
            patch_dir=args.patch_dir,
        )
        rows.append(r)
        print(f"id{i}: dominant_non_none={r['dominant_non_none']:>10}  race_pct={r['race_pct']}")

    # Pairwise divergence of tribe histograms (excluding NONE) — are they different?
    print("\n-- tribe vectors (non-NONE %) --")
    all_tribes = sorted({t for r in rows for t in r["race_pct"] if t != "NONE"})
    for r in rows:
        vec = {t: r["race_pct"].get(t, 0.0) for t in all_tribes}
        print(f"id{r['identity']}: {vec}")


if __name__ == "__main__":
    main()

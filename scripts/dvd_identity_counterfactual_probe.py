#!/usr/bin/env python3
"""Counterfactual DvD probe: does the identity INPUT change the policy at all?

Collect real shop-decision states by playing one identity, then for each state
re-run the policy under every identity one-hot (same core obs, only the identity
tail swapped) and measure how much the action distribution moves:

  * TV  = total-variation distance between identities' action distributions,
          averaged over decisions (0 = identity is inert; large = it matters).
  * dP_argmax = mean change in the top action's probability across identities.

This isolates wiring/representation from reward alignment:
  TV ~ 0      -> the identity input is causally inert (dead/untrained pathway).
  TV > 0 but boards identical (see board/mixed probes) -> input works, but the
              learned response is NOT tribe-aligned (a reward/credit problem).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.envs  # noqa: F401
from src.agents.ppo_dvd_agent import PPODvDAgent
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import lobby_from_learned_seats


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--num-identities", type=int, default=4)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    ap.add_argument("--max-states", type=int, default=400)
    args = ap.parse_args()

    agent = PPODvDAgent.load(str(args.checkpoint), device="cpu", seed=args.seed)
    agent.eval()
    net = agent.policy_net
    N = args.num_identities

    # --- collect (core_obs, legal_list) at shop decisions by playing identity 0 ---
    captured = []
    orig_act = agent.act_structured

    def _recording_act(obs, legal_list, env, *, deterministic=False):
        if len(captured) < args.max_states and len(legal_list) > 1:
            captured.append((np.asarray(obs, dtype=np.float32), list(legal_list)))
        return orig_act(obs, legal_list, env, deterministic=deterministic)

    agent.act_structured = _recording_act  # type: ignore
    agents = {s: agent for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    for g in range(args.games):
        agent.set_episode_identity(0)
        env = BGLobbyEnv(configs, learned_seats=tuple(range(8)), training_seats=(0,),
                         seed=args.seed + g, patch_dir=args.patch_dir, obs_kind="bglike_v5")
        env.reset(seed=args.seed + g)
        env.drain_until_lobby_done(deterministic=True)
        if len(captured) >= args.max_states:
            break
    agent.act_structured = orig_act  # type: ignore

    print(f"checkpoint: {args.checkpoint.name}")
    print(f"collected {len(captured)} multi-choice shop decisions\n")
    if not captured:
        raise SystemExit("no states captured")

    # --- per state: action dist under each identity (same core obs) ---
    tvs, dargmax, max_tv = [], [], 0.0
    for core_obs, legal in captured:
        dists = []
        for i in range(N):
            onehot = np.zeros(N, dtype=np.float32)
            onehot[i] = 1.0
            aug = np.concatenate([core_obs, onehot])
            obs_t = torch.as_tensor(aug, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, mask, _v, _c = net.policy_logits_and_value(
                    obs_t, [legal], return_cache=True)
                logits = logits.masked_fill(~mask, float("-inf"))
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            dists.append(p)
        # pairwise TV across identities
        pair_tv = [_tv(dists[a], dists[b])
                   for a in range(N) for b in range(a + 1, N)]
        tvs.append(float(np.mean(pair_tv)))
        max_tv = max(max_tv, float(np.max(pair_tv)))
        am = [int(np.argmax(d)) for d in dists]
        dargmax.append(0.0 if len(set(am)) == 1 else 1.0)

    print(f"mean pairwise TV(action dist) across identities = {np.mean(tvs):.4f}")
    print(f"max  pairwise TV (single decision)              = {max_tv:.4f}")
    print(f"fraction of decisions where argmax action differs by identity = "
          f"{np.mean(dargmax):.3f}")
    print()
    print("interpretation:")
    print("  TV ~ 0.00-0.02  -> identity input is essentially INERT (dead pathway)")
    print("  TV >> 0 but boards identical -> input works, response NOT tribe-aligned")


if __name__ == "__main__":
    main()

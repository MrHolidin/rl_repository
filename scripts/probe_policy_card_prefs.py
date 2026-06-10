#!/usr/bin/env python3
"""Behavioral fingerprint of a checkpoint: per-card buy tendencies when offered.

Runs full 8-seat self-play lobbies with the checkpoint driving every seat
(identity pinned per lobby, rotating). At every shop decision it computes the
policy distribution over legal structured actions and records, for each shop
offer: P(buy that slot) and whether the argmax action bought it. Aggregates by
card name so "does it ever buy Brann/Nomi when offered" is answered with data.

Also tracks action-type mix and per-round value estimates.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

import src.envs  # noqa: F401
from src.envs.bglike.lobby_env import BGLobbyEnv
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.minibg.structured_actions import StructActionType


def load_agent(path: Path, *, device: str, seed: int):
    from src.agents.ppo_dvd_agent import PPODvDAgent

    ckpt = torch.load(str(path), map_location="cpu")
    nt = str(ckpt.get("ppo_network_type", ""))
    if nt != "bglike_structured_v7":
        raise SystemExit(f"checkpoint network is {nt!r}, not bglike_structured_v7")
    return PPODvDAgent.load(str(path), device=device, seed=seed)


class _StateView:
    def __init__(self, lobby: BGLobbyEnv) -> None:
        self.state = lobby.state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patch-dir", default="data/bgcore/19_6_0_74257")
    ap.add_argument("--num-identities", type=int, default=4)
    ap.add_argument(
        "--pin-identity",
        type=int,
        default=None,
        help="pin all games to this identity instead of rotating",
    )
    args = ap.parse_args()

    agent = load_agent(args.checkpoint, device=args.device, seed=args.seed)
    agent.eval()
    agent.training = False

    agents = {s: agent for s in range(8)}
    configs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        configs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=args.seed,
        patch_dir=args.patch_dir,
        obs_kind="bglike_v5",
    )
    patch = env._game._patch

    # per-card aggregates over decisions where the card was a legal BUY
    offer_n = defaultdict(int)
    offer_pbuy = defaultdict(float)
    offer_bought = defaultdict(int)
    offer_round_sum = defaultdict(float)
    action_type_count = defaultdict(int)
    value_by_round = defaultdict(list)
    tier_of = {}

    n_decisions = 0
    for g in range(args.games):
        ident = args.pin_identity if args.pin_identity is not None else g % args.num_identities
        agent.set_episode_identity(ident)
        env.reset(seed=args.seed + 1000 * g)
        steps = 0
        while not env.lobby_done and steps < 20000:
            steps += 1
            cur = env.current_seat()
            if not env._seat_can_act(cur):
                if env.state.done:
                    break
                raise RuntimeError("drain stall in probe")
            obs = env.obs_for_seat(cur)
            legal = env.legal_structured_actions_for_seat(cur)
            if not legal:
                raise RuntimeError("empty legal in probe")

            # policy distribution under the pinned identity
            obs_aug = agent._augment_obs_np(obs, None)
            obs_t = torch.as_tensor(
                obs_aug, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            with torch.no_grad():
                logits, mask, value = agent.policy_net.policy_logits_and_value(
                    obs_t, [legal]
                )
                logits = logits.masked_fill(~mask, float("-inf"))
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            argmax_idx = int(np.argmax(probs))

            player = env.state.players[cur]
            rnd = env.state.round_number
            value_by_round[rnd].append(float(value.item()))

            for i, act in enumerate(legal):
                if act.type == StructActionType.BUY:
                    slot = act.args[0]
                    m = player.shop[slot]
                    if m is None:
                        continue
                    name = m.name or m.card_id
                    tier_of[name] = m.tier
                    offer_n[name] += 1
                    offer_pbuy[name] += float(probs[i])
                    offer_round_sum[name] += rnd
                    if i == argmax_idx:
                        offer_bought[name] += 1

            n_decisions += 1
            # step with the agent's own (deterministic eval) path to get board_perm
            struct_act, board_perm, _ = agent.act_structured(
                obs, legal, _StateView(env), deterministic=True
            )
            action_type_count[struct_act.type.name] += 1
            env.step_structured_for_seat(cur, struct_act, board_perm=board_perm)

    print(f"decisions: {n_decisions}, games: {args.games}\n")
    print("action type mix:")
    tot = sum(action_type_count.values())
    for k, v in sorted(action_type_count.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v:6d}  {100.0 * v / tot:5.1f}%")

    print("\nmean value estimate by round (n):")
    for rnd in sorted(value_by_round):
        vals = value_by_round[rnd]
        print(f"  r{rnd:2d}: {np.mean(vals):+.3f}  (n={len(vals)})")

    rows = []
    for name, n in offer_n.items():
        rows.append(
            (
                tier_of[name],
                name,
                n,
                offer_pbuy[name] / n,
                offer_bought[name] / n,
                offer_round_sum[name] / n,
            )
        )
    rows.sort(key=lambda r: (r[0], -r[3]))
    print("\nper-card buy tendencies (tier | name | times offered | mean P(buy) | argmax-buy rate | mean round):")
    for tier, name, n, pbuy, brate, mr in rows:
        print(f"  t{tier} {name:32s} n={n:5d}  P(buy)={pbuy:.3f}  bought={brate:.3f}  r~{mr:.1f}")


if __name__ == "__main__":
    main()

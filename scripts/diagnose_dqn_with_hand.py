"""Diagnose dqn_with_hand regression.

Loads a checkpoint and plays N episodes vs `random` and vs `balanced`. For
every state the agent acts in, records:
  - phase (SHOP | ORDER)
  - num_legal actions
  - q_spread = max(legal Q) - min(legal Q)
  - top1 - top2 (greedy gap)
  - chosen env action id

Then prints aggregate stats split by phase and opponent.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.agents.dqn.agent import DQNAgent
from src.envs.minibg.env import MiniBGEnv
from src.envs.minibg.heuristic_bots.bots import (
    BalancedBot,
    RandomBot,
    HeuristicBot,
)
from src.envs.minibg.action_map import (
    A_BUY_BASE,
    A_FINISH,
    A_LEVEL_UP,
    A_PLACE_BASE,
    A_ROLL,
    A_SELECT_ORDER_BASE,
    A_SELL_BASE,
    NUM_ENV_ACTIONS,
)
from src.envs.minibg.state import PlayerPhase


PLAYER_TOKENS = (1, -1)


def action_kind(a: int) -> str:
    if a == A_ROLL:
        return "ROLL"
    if a == A_LEVEL_UP:
        return "LEVEL_UP"
    if A_BUY_BASE <= a < A_BUY_BASE + 3:
        return "BUY"
    if A_SELL_BASE <= a < A_SELL_BASE + 4:
        return "SELL"
    if A_PLACE_BASE <= a < A_PLACE_BASE + 3:
        return "PLACE"
    if a == A_FINISH:
        return "FINISH"
    if a >= A_SELECT_ORDER_BASE:
        return "SELECT_ORDER"
    return "?"


def load_agent(ckpt_path: Path, device: str = "cuda") -> DQNAgent:
    return DQNAgent.load(str(ckpt_path), device=device)


def run_episodes(agent: DQNAgent, opponent: HeuristicBot, n: int, seed: int):
    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, 2**31 - 1))
    env = MiniBGEnv(seed=base_seed, battle_damage_shaping=0.0)

    # Per-phase records: list of dicts.
    records: Dict[str, List[dict]] = {"SHOP": [], "ORDER": []}
    outcomes = collections.Counter()
    total_steps = 0
    agent_steps = 0

    for ep in range(n):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        env.reset(seed=ep_seed)
        agent_token = PLAYER_TOKENS[int(rng.integers(0, 2))]

        while not env.done:
            cur_token = env.current_player_token
            mask = env.legal_actions_mask
            obs = env._get_obs()
            phase = env.state.players[env.current_player()].phase

            if cur_token == agent_token:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
                    mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(agent.device)
                    q = agent.q_network(obs_t, legal_mask=mask_t)[0]
                    q_np = q.detach().cpu().numpy()

                legal_idx = np.flatnonzero(mask)
                legal_q = q_np[legal_idx]
                spread = float(legal_q.max() - legal_q.min())
                if len(legal_q) >= 2:
                    top2 = np.sort(legal_q)[-2:]
                    gap = float(top2[1] - top2[0])
                else:
                    gap = 0.0
                chosen = int(legal_idx[int(np.argmax(legal_q))])

                rec = {
                    "round": env.state.round_number,
                    "n_legal": int(len(legal_idx)),
                    "q_spread": spread,
                    "top2_gap": gap,
                    "q_max": float(legal_q.max()),
                    "q_min": float(legal_q.min()),
                    "chosen_kind": action_kind(chosen),
                }
                if phase == PlayerPhase.ORDER:
                    records["ORDER"].append(rec)
                else:
                    records["SHOP"].append(rec)
                action = chosen
                agent_steps += 1
            else:
                action = opponent.choose_action(env)
            env.step(int(action))
            total_steps += 1

        winner = env.state.winner
        if winner == 0 or winner is None:
            outcomes["draw"] += 1
        elif winner == agent_token:
            outcomes["win"] += 1
        else:
            outcomes["loss"] += 1

    return records, outcomes, total_steps, agent_steps


def summarize(name: str, records: Dict[str, List[dict]], outcomes, total_steps, agent_steps) -> None:
    n = sum(outcomes.values())
    print(f"\n=== vs {name}  ({n} episodes) ===")
    print(f"  outcomes: win={outcomes['win']} draw={outcomes['draw']} loss={outcomes['loss']}")
    print(f"  total env steps={total_steps}  agent decisions={agent_steps}")
    for phase in ("SHOP", "ORDER"):
        rs = records[phase]
        if not rs:
            print(f"  [{phase}] no records")
            continue
        spreads = np.array([r["q_spread"] for r in rs])
        gaps = np.array([r["top2_gap"] for r in rs])
        n_legals = np.array([r["n_legal"] for r in rs])
        print(
            f"  [{phase}] count={len(rs):4d}  "
            f"avg_spread={spreads.mean():.4f}  p50={np.median(spreads):.4f}  "
            f"p95={np.percentile(spreads, 95):.4f}  max={spreads.max():.4f}"
        )
        print(
            f"            avg_top2_gap={gaps.mean():.4f}  p50={np.median(gaps):.4f}  "
            f"p95={np.percentile(gaps, 95):.4f}"
        )
        print(
            f"            avg_n_legal={n_legals.mean():.2f}  "
            f"min={n_legals.min()}  max={n_legals.max()}"
        )
        kind_counts = collections.Counter(r["chosen_kind"] for r in rs)
        print(
            f"            chosen kinds: {dict(kind_counts)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    agent = load_agent(Path(args.ckpt), device=args.device)
    print(f"Loaded {args.ckpt}; device={agent.device}")

    rb = RandomBot(seed=args.seed)
    bb = BalancedBot()

    rng = np.random.default_rng(args.seed)
    s1 = int(rng.integers(0, 2**31 - 1))
    s2 = int(rng.integers(0, 2**31 - 1))

    rec_r, out_r, ts_r, as_r = run_episodes(agent, rb, args.episodes, s1)
    rec_b, out_b, ts_b, as_b = run_episodes(agent, bb, args.episodes, s2)

    summarize("random", rec_r, out_r, ts_r, as_r)
    summarize("balanced", rec_b, out_b, ts_b, as_b)


if __name__ == "__main__":
    main()

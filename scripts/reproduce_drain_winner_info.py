#!/usr/bin/env python3
"""Reproduce stale ``step.info`` when episode ends inside ``_drain_opponent``.

Run from repo root:
  python scripts/reproduce_drain_winner_info.py [--games 500] [--checkpoint path/to.pt]

Compares ``StepResult.info['winner']`` after ``AgentPerspectiveEnv.step_structured`` with
``base_env.winner`` when the episode is done. Mismatch indicates the drain-termination path
returned ``info`` from the learner's non-terminal structured step instead of the terminal step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.envs  # noqa: F401
from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent
from src.envs import RewardConfig
from src.registry import make_game
from src.training.agent_perspective_env import AgentPerspectiveEnv, make_minibg_shaping_fn
from src.training.distributed_trainer import FixedOpponentSampler
from src.utils import freeze_agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400, help="Max episodes to try")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO / "runs/realbg/dist_ppo_007/checkpoints/dist_minibg_ppo_955594.pt",
    )
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    ck = args.checkpoint.resolve()
    if not ck.is_file():
        raise SystemExit(f"checkpoint not found: {ck}")

    from src.envs.minibg.heuristic_bots.agent_adapter import MiniBGHeuristicAgent
    from src.envs.minibg.heuristic_bots.tournament import make_bot

    # Same shaping as training configs
    base = make_game("minibg", reward_config=RewardConfig(), battle_damage_shaping=0.06)
    opp = MiniBGHeuristicAgent(make_bot("t1_random", args.seed))
    env = AgentPerspectiveEnv(
        base,
        FixedOpponentSampler(opp),
        agent_first_probability=0.5,
        shaping_fn=make_minibg_shaping_fn(0.06),
    )

    agent = MiniBGPPOStructuredAgent.load(str(ck), device="cpu", seed=args.seed)
    freeze_agent(agent)
    agent.train()

    mismatches = 0
    terminal_steps = 0
    mismatch_samples: list[tuple[int, object, object]] = []

    for ep in range(args.games):
        obs = env.reset()
        if hasattr(agent, "set_env"):
            agent.set_env(env)
        steps = 0
        while not env.done:
            legal_list = env.legal_structured_actions()
            struct_act, board_perm, _idx = agent.act_structured(
                obs, legal_list, env, deterministic=False
            )
            step = env.step_structured(struct_act, board_perm=board_perm)
            steps += 1
            if step.done:
                terminal_steps += 1
                info = step.info if isinstance(step.info, dict) else {}
                iw = info.get("winner")
                gw = getattr(env.base, "winner", None)
                ok = (iw == gw) or (iw is None and gw is None and not env.base.done)
                if env.base.done and iw != gw:
                    mismatches += 1
                    if len(mismatch_samples) < 8:
                        mismatch_samples.append((ep, iw, gw))
            obs = step.obs
            if steps > 100_000:
                raise RuntimeError("episode too long")

    print(f"episodes={args.games} terminal_agent_steps={terminal_steps}")
    print(f"mismatches (info['winner'] != base.winner when base.done): {mismatches}")
    if mismatches:
        print("samples (episode, info_winner, base_winner):")
        for row in mismatch_samples:
            print(f"  {row}")
        print("\nInterpretation: episode often ended in _drain_opponent; returned info was stale.")
    else:
        print("No mismatches in this run (try --games larger or another checkpoint / opponent).")


if __name__ == "__main__":
    main()

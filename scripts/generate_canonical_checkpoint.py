#!/usr/bin/env python3
"""Generate canonical DQN checkpoint (300 steps) and probe data for compat tests."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs import Connect4Env
from src.training.canonical_checkpoint import PROBE_ACTIONS, PROBE_SEED, train_and_probe


def main() -> None:
    fixtures = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    fixtures.mkdir(exist_ok=True)

    seed = 42
    agent, actions = train_and_probe(seed, steps=300)

    ckpt_path = fixtures / "canonical_dqn.pt"
    agent.save(str(ckpt_path))
    print(f"Saved checkpoint to {ckpt_path}")

    env2 = Connect4Env(rows=6, cols=7)
    env2.reset(seed=PROBE_SEED)
    probe_data = []
    for agent_a, opp_a in PROBE_ACTIONS:
        obs = env2._get_obs()
        legal = env2.legal_actions_mask.astype(bool)
        expected = actions[len(probe_data)] if len(probe_data) < len(actions) else 0
        probe_data.append({
            "obs": obs.tolist(),
            "legal_mask": legal.tolist(),
            "expected_action": expected,
        })
        step_res = env2.step(agent_a)
        if step_res.terminated or step_res.truncated:
            break
        step_res = env2.step(opp_a)
        if step_res.terminated or step_res.truncated:
            break

    probe_path = fixtures / "canonical_dqn_probe.json"
    probe_path.write_text(json.dumps({"probes": probe_data}, indent=2))
    print(f"Saved probe to {probe_path}")


if __name__ == "__main__":
    main()

"""Shared logic for canonical checkpoint generation and determinism tests."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from src.agents import DQNAgent
from src.envs import Connect4Env
from src.features.action_space import DiscreteActionSpace
from src.models import Connect4DQN
from src.training.opponent_sampler import RandomOpponentSampler
from src.training.trainer import Trainer, StartPolicy

PROBE_SEED = 99
PROBE_ACTIONS = [(3, 3), (4, 2), (1, 0)]


def train_and_probe(seed: int, steps: int = 300) -> Tuple[DQNAgent, List[int]]:
    """Train DQN for steps and return (agent, list of actions on probe states)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = Connect4Env(rows=6, cols=7)
    network = Connect4DQN(rows=6, cols=7, in_channels=2, num_actions=7)
    agent = DQNAgent(
        network=network,
        num_actions=7,
        seed=seed,
        action_space=DiscreteActionSpace(n=7),
        batch_size=32,
        replay_buffer_size=2000,
        device="cuda",
    )
    agent.train()

    trainer = Trainer(
        env,
        agent,
        opponent_sampler=RandomOpponentSampler(seed=seed),
        start_policy=StartPolicy.AGENT_FIRST,
        rng=np.random.default_rng(seed),
    )
    trainer.train(total_steps=steps, deterministic=False)

    agent.eval()
    env2 = Connect4Env(rows=6, cols=7)
    env2.reset(seed=PROBE_SEED)
    actions = []

    for agent_a, opp_a in PROBE_ACTIONS:
        obs = env2._get_obs()
        legal = env2.legal_actions_mask.astype(bool)
        action = agent.act(obs, legal_mask=legal, deterministic=True)
        actions.append(int(action))
        step_res = env2.step(agent_a)
        if step_res.terminated or step_res.truncated:
            break
        step_res = env2.step(opp_a)
        if step_res.terminated or step_res.truncated:
            break

    return agent, actions

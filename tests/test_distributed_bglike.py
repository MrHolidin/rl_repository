"""Distributed trainer BGLike wiring."""

from __future__ import annotations

import numpy as np

from src.agents.ppo_agent import PPOAgent, RolloutBuffer
from src.envs.bglike.action_map import NUM_ENV_ACTIONS
from src.envs.bglike.obs import OBS_DIM
from src.features.action_space import DiscreteActionSpace
from src.models.flat_mlp_ac import FlatMLPActorCritic
from src.models.ppo_policy_factory import build_ppo_actor_critic
from src.training.distributed_trainer import (
    _merge_buffers,
    _payload_to_buffer,
    _use_structured_collect,
)


def test_use_structured_collect_bglike_false():
    assert not _use_structured_collect({"game_id": "bglike", "use_structured": False})


def test_flat_mlp_ac_accepts_bglike_obs_dim():
    net = build_ppo_actor_critic("flat_mlp", (OBS_DIM,), NUM_ENV_ACTIONS, mlp_hidden_size=32)
    x = np.zeros(OBS_DIM, dtype=np.float32)
    import torch

    logits, value = net(torch.from_numpy(x).unsqueeze(0))
    assert logits.shape == (1, NUM_ENV_ACTIONS)
    assert value.shape == (1,)


def test_rollout_buffer_roundtrip():
    mg = {"game_id": "bglike", "use_structured": False}
    buf = RolloutBuffer()
    buf.add(
        obs=np.zeros(OBS_DIM, dtype=np.float32),
        action=1,
        reward=0.0,
        done=False,
        value=0.0,
        log_prob=-1.0,
        legal_mask=np.ones(NUM_ENV_ACTIONS, dtype=bool),
        next_obs=np.zeros(OBS_DIM, dtype=np.float32),
        next_legal_mask=np.ones(NUM_ENV_ACTIONS, dtype=bool),
        seat_id=0,
    )
    payload = __import__("pickle").dumps(buf)
    restored = _payload_to_buffer(payload, mg)
    merged = _merge_buffers([restored], mg)
    assert len(merged) == 1


def test_ppo_agent_save_load_flat_mlp(tmp_path):
    from src.models.ppo_policy_factory import PPO_NETWORK_FLAT_MLP, ppo_network_type_for_save

    net = FlatMLPActorCritic(OBS_DIM, NUM_ENV_ACTIONS, hidden_size=32)
    agent = PPOAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=NUM_ENV_ACTIONS,
        network=net,
        ppo_network_type=ppo_network_type_for_save(PPO_NETWORK_FLAT_MLP),
        ppo_network_kwargs=net.get_constructor_kwargs(),
        action_space=DiscreteActionSpace(NUM_ENV_ACTIONS),
        rollout_steps=64,
    )
    path = tmp_path / "agent.pt"
    agent.save(str(path))
    loaded = PPOAgent.load(str(path), device="cpu", seed=1)
    assert loaded.num_actions == NUM_ENV_ACTIONS

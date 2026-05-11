import numpy as np
import torch

import src.agents  # noqa: F401 — registers `ppo` with make_agent
from src.envs.minibg.obs import OBS_DIM
from src.models.minibg_slot_ac import MiniBGSlotActorCritic
from src.models.ppo_policy_factory import (
    PPO_NETWORK_MINIBG_SLOT,
    build_ppo_actor_critic,
    restore_ppo_actor_critic,
)
from src.registry import make_agent


def test_minibg_slot_actor_critic_forward():
    n_act = 80
    m = MiniBGSlotActorCritic(num_actions=n_act, slot_hidden=8, trunk_hidden=32)
    x = torch.randn(2, OBS_DIM)
    legal = torch.zeros(2, n_act, dtype=torch.bool)
    legal[:, :3] = True
    logits, v = m(x, legal_mask=legal)
    assert logits.shape == (2, n_act)
    assert v.shape == (2,)
    assert torch.isfinite(logits[0, :3]).all()
    assert (logits[0, 3:] == float("-inf")).all()


def test_build_ppo_minibg_slot():
    net = build_ppo_actor_critic(
        PPO_NETWORK_MINIBG_SLOT,
        (OBS_DIM,),
        num_actions=42,
        slot_hidden_channels=8,
        trunk_hidden_size=64,
        region_conv2_kernel=1,
    )
    assert isinstance(net, MiniBGSlotActorCritic)


def test_make_agent_ppo_minibg():
    agent = make_agent(
        "ppo",
        network_type="minibg_slot",
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=50,
        rollout_steps=32,
        minibatch_size=16,
        device="cpu",
    )
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    legal = np.zeros(50, dtype=bool)
    legal[0] = True
    a = agent.act(obs, legal_mask=legal, deterministic=True)
    assert 0 <= a < 50


def test_ppo_save_load_roundtrip(tmp_path):
    from src.agents.ppo_agent import PPOAgent

    path = tmp_path / "p.pt"
    agent = make_agent(
        "ppo",
        network_type="minibg_slot",
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=11,
        rollout_steps=8,
        minibatch_size=4,
        device="cpu",
    )
    agent.save(str(path))
    loaded = PPOAgent.load(str(path), device="cpu")
    assert loaded.num_actions == 11
    assert loaded._ppo_network_type == "minibg_slot"
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    legal = np.zeros(11, dtype=bool)
    legal[2] = True
    assert loaded.act(obs, legal_mask=legal) == agent.act(obs, legal_mask=legal, deterministic=True)


def test_restore_ppo_actor_critic_minibg_kw_only():
    net = restore_ppo_actor_critic(
        "minibg_slot",
        (OBS_DIM,),
        15,
        {"slot_hidden": 4, "trunk_hidden": 8, "region_conv2_kernel": 1},
    )
    x = torch.randn(1, OBS_DIM)
    lm = torch.zeros(1, 15, dtype=torch.bool)
    lm[0, 0] = True
    logits, _ = net(x, legal_mask=lm)
    assert logits.shape == (1, 15)

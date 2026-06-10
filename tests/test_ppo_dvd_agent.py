"""PPODvDAgent step-2 wiring: factory build, obs augmentation, identity gradient.

Verifies the trainer-facing plumbing without a full lobby rollout:
  * ``make_agent("ppo", network_type="bglike_structured_v7", ...)`` returns a
    PPODvDAgent wrapping a v7 net whose obs width includes the identity tail;
  * the agent appends a correct one-hot identity to observations;
  * gradient reaches ``identity_proj`` (the identity actually trains).
"""

from __future__ import annotations

import numpy as np
import torch

from src.agents import PPODvDAgent  # noqa: F401 (ensures "ppo" is registered)
from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.actions import NUM_ACTIONS
from src.envs.bglike.obs_v5 import OBS_DIM_V5
from src.models.bglike_structured_v7 import BGLikeStructuredV7
from src.registry import make_agent

_PATCH = "data/bgcore/15_6_2_36393"
_N_ID = 4


def _make_agent(num_identities: int = _N_ID, diversity_coef: float = 0.0):
    ctx = load_patch_context(_PATCH)
    return make_agent(
        "ppo",
        network_type="bglike_structured_v7",
        num_identities=num_identities,
        diversity_coef=diversity_coef,
        num_actions=NUM_ACTIONS,
        observation_shape=(OBS_DIM_V5,),
        observation_type="vector",
        num_pool_indices=ctx.num_pool_indices,
        device="cpu",
    )


def test_factory_returns_dvd_agent_over_v7():
    agent = _make_agent(num_identities=_N_ID, diversity_coef=0.0)
    assert isinstance(agent, PPODvDAgent)
    assert agent.num_identities == _N_ID
    assert agent.diversity_coef == 0.0
    assert isinstance(agent.policy_net, BGLikeStructuredV7)
    # Net declares the augmented input width; env still emits OBS_DIM_V5.
    assert agent.policy_net.obs_dim == OBS_DIM_V5 + _N_ID


def test_obs_augmentation_appends_active_identity_onehot():
    agent = _make_agent()
    agent._active_identity = 2
    aug = agent._augment_obs_np(np.zeros(OBS_DIM_V5, dtype=np.float32))
    assert aug.shape == (OBS_DIM_V5 + _N_ID,)
    tail = aug[OBS_DIM_V5:]
    assert tail.sum() == 1.0
    assert tail[2] == 1.0


def test_identity_index_stays_in_range_under_rotation():
    agent = _make_agent()
    for _ in range(200):
        agent._dvd_new_episode = True
        # mimic the start-of-episode rotation in act_structured
        if agent._dvd_new_episode:
            agent._active_identity = int(agent._id_rng.integers(agent.num_identities))
            agent._dvd_new_episode = False
        assert 0 <= agent._active_identity < _N_ID


def test_identity_receives_gradient_through_agent_net():
    agent = _make_agent()
    agent.policy_net.train()
    aug = agent._augment_obs_np(np.zeros(OBS_DIM_V5, dtype=np.float32))
    x = torch.as_tensor(aug, dtype=torch.float32).unsqueeze(0)
    state_emb, _ = agent.policy_net.encode_state(x)
    state_emb.sum().backward()
    grad = agent.policy_net.identity_proj.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad) > 0

"""Second value head for the shaping half of the reward.

The placement CE head spans exactly the eight placement rewards, so shaping can
never appear in its expectation and reaches the advantage unbaselined. These
tests pin the fix:

  * the head is zero-initialized, so a net without shaping is unchanged;
  * ``V = E[placement reward] + E[remaining shaping]``;
  * the reward splits back into its two halves without any extra plumbing from
    the collector (terminal row = placement + board shaping, the rest = per-step
    shaping), and the shaping half is what the head is regressed on;
  * the earned-shaping metrics report the true per-game total;
  * nets predating the head (v8) keep the old behaviour exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.action_map import NUM_ENV_ACTIONS
from src.envs.bglike.obs_v5_heroes import OBS_DIM_V5_HEROES
from src.envs.bglike.placement import placement_reward
from src.registry import make_agent

PATCH_DIR = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return load_patch_context(PATCH_DIR)


def _agent(patch, **overrides):
    kw = dict(
        network_type="bglike_structured_v11_heroes",
        observation_shape=(OBS_DIM_V5_HEROES,),
        observation_type="vector",
        num_actions=int(NUM_ENV_ACTIONS),
        num_pool_indices=patch.num_pool_indices,
        num_identities=4,
        slot_hidden_channels=32,
        card_emb_dim=16,
        entity_attention_layers=1,
        rollout_steps=16,
        ppo_epochs=1,
        minibatch_size=8,
        discount_factor=1.0,
        device="cpu",
    )
    kw.update(overrides)
    return make_agent("ppo", **kw)


# --------------------------------------------------------------------------- #
# Head contract
# --------------------------------------------------------------------------- #


def test_shape_head_is_zero_init_and_v_decomposes(patch):
    agent = _agent(patch)
    net = agent.policy_net
    trunk = torch.randn(5, net._state_summary_dim)

    shape_v = net.shape_value(trunk)
    assert shape_v.shape == (5,)
    assert torch.equal(shape_v, torch.zeros(5))
    # Zero-init → a run with no shaping sees exactly the pre-existing V.
    assert torch.allclose(net.value_from_trunk(trunk), net.placement_value(trunk))

    with torch.no_grad():
        net.critic_shape.bias.fill_(0.3)
    assert torch.allclose(
        net.value_from_trunk(trunk), net.placement_value(trunk) + 0.3, atol=1e-6
    )


# --------------------------------------------------------------------------- #
# Reward split
# --------------------------------------------------------------------------- #


def _one_segment(agent, per_step, terminal_shaping, place=2):
    """One seat, four rows: three shaped steps then the placement terminal."""
    rewards = np.array(
        [per_step, per_step, per_step, placement_reward(place) + terminal_shaping],
        dtype=np.float32,
    )
    dones = np.array([False, False, False, True])
    seats = np.zeros(4, dtype=np.int64)
    labels = [place] * 4
    return agent._shaping_returns(rewards, dones, seats, labels)


def test_split_recovers_the_shaping_half(patch):
    agent = _agent(patch)
    ret, stats = _one_segment(agent, per_step=0.05, terminal_shaping=0.2)

    # MC return of shaping alone, discount 1.0: 0.05*3 + 0.2 = 0.35 at the head
    # of the segment, shrinking by one step's shaping per row.
    assert ret == pytest.approx([0.35, 0.30, 0.25, 0.20], abs=1e-6)
    # The placement reward is fully subtracted — no trace of it in the target.
    assert stats["shaping_per_game"] == pytest.approx(0.35, abs=1e-6)
    assert stats["shaping_step_mean"] == pytest.approx(0.35 / 4, abs=1e-6)


def test_no_shaping_leaves_nothing_to_fit(patch):
    agent = _agent(patch)
    ret, stats = _one_segment(agent, per_step=0.0, terminal_shaping=0.0)
    assert np.allclose(ret, 0.0)
    assert stats["shaping_per_game"] == pytest.approx(0.0)
    assert stats["shaping_frac_of_return"] == pytest.approx(0.0)


def test_terminal_row_without_a_placement_label_contributes_nothing(patch):
    """A segment cut at the buffer edge cannot be decomposed — it must not hand
    the head a placement reward to fit."""
    agent = _agent(patch)
    rewards = np.array([0.05, 0.6], dtype=np.float32)
    dones = np.array([False, True])
    seats = np.zeros(2, dtype=np.int64)
    ret, _ = agent._shaping_returns(rewards, dones, seats, [-1, -1])
    assert ret == pytest.approx([0.05, 0.0], abs=1e-6)


def test_per_seat_split_does_not_mix_segments(patch):
    agent = _agent(patch)
    rewards = np.array(
        [0.05, 0.05, placement_reward(1), placement_reward(8)], dtype=np.float32
    )
    dones = np.array([False, False, True, True])
    seats = np.array([0, 1, 0, 1], dtype=np.int64)
    # Labels are the seat's own placement: seat 0 came 1st, seat 1 came 8th.
    ret, stats = agent._shaping_returns(rewards, dones, seats, [1, 8, 1, 8])
    # Each seat keeps its own 0.05 and neither inherits the other's.
    assert ret == pytest.approx([0.05, 0.05, 0.0, 0.0], abs=1e-6)
    assert stats["shaping_per_game"] == pytest.approx(0.05, abs=1e-6)


# --------------------------------------------------------------------------- #
# Nets predating the head
# --------------------------------------------------------------------------- #


def test_v8_net_keeps_the_old_behaviour(patch):
    from src.envs.bglike.obs_v5 import OBS_DIM_V5

    agent = _agent(
        patch,
        network_type="bglike_structured_v8",
        observation_shape=(int(OBS_DIM_V5),),
    )
    assert agent._distributional
    assert not agent._has_shape_value
    ret, stats = _one_segment(agent, per_step=0.05, terminal_shaping=0.2)
    assert ret is None
    assert stats == {}

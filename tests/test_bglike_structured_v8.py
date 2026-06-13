"""v8 distributional placement critic: init contract, warm-start, labels, CE update.

Contract:
  * a v7 ``state_dict`` loads into v8 with ``strict=False`` adding only the
    ``critic_dist.*`` keys;
  * zero-init dist head → uniform placement distribution → V == 0 exactly,
    and V always equals softmax(logits)·placement_reward_vec;
  * the PPO factory builds / serializes / restores v8;
  * ``close_rollout_segment(placement=...)`` stamps the label on every row of
    the seat segment (and only that segment);
  * the flat PPO update runs CE + emits placement_acc on a real-env rollout.
"""

from __future__ import annotations

import numpy as np
import torch

import src.envs  # noqa: F401
from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.placement import placement_reward
from src.models.bglike_structured_v7 import BGLikeStructuredV7
from src.models.bglike_structured_v8 import BGLikeStructuredV8, NUM_PLACEMENTS
from src.models.ppo_policy_factory import (
    PPO_NETWORK_BGLIKE_STRUCTURED_V8,
    build_ppo_actor_critic,
    default_ppo_network_kwargs,
    ppo_network_type_for_save,
    restore_ppo_actor_critic,
)

_PATCH = "data/bgcore/15_6_2_36393"
_N_ID = 4


def _base_kwargs(ctx):
    return dict(
        slot_hidden=48,
        state_dim=128,
        action_dim=64,
        num_pool_indices=ctx.num_pool_indices,
    )


def _v8(ctx=None):
    ctx = ctx or load_patch_context(_PATCH)
    return BGLikeStructuredV8(num_identities=_N_ID, **_base_kwargs(ctx)).eval()


def test_v7_checkpoint_warm_starts_v8():
    ctx = load_patch_context(_PATCH)
    v7 = BGLikeStructuredV7(num_identities=_N_ID, **_base_kwargs(ctx))
    v8 = _v8(ctx)
    missing, unexpected = v8.load_state_dict(v7.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {
        "critic_dist.0.weight",
        "critic_dist.0.bias",
        "critic_dist.2.weight",
        "critic_dist.2.bias",
        "placement_reward_vec",
    }


def test_zero_init_value_is_exactly_zero_and_uniform():
    v8 = _v8()
    trunk = torch.randn(5, v8._state_summary_dim)
    logits = v8.placement_logits(trunk)
    assert logits.shape == (5, NUM_PLACEMENTS)
    assert torch.allclose(logits, torch.zeros_like(logits))
    v = v8.value_from_trunk(trunk)
    assert torch.allclose(v, torch.zeros(5), atol=1e-7)


def test_value_equals_expectation_after_perturbation():
    v8 = _v8()
    with torch.no_grad():
        torch.nn.init.normal_(v8.critic_dist[-1].weight, std=0.5)
        torch.nn.init.normal_(v8.critic_dist[-1].bias, std=0.5)
    trunk = torch.randn(7, v8._state_summary_dim)
    probs = torch.softmax(v8.placement_logits(trunk), dim=-1)
    rew = torch.tensor([placement_reward(p) for p in range(1, 9)])
    expected = (probs * rew).sum(-1)
    assert torch.allclose(v8.value_from_trunk(trunk), expected, atol=1e-6)
    # values flow through the standard forward too
    from src.envs.bglike.obs_v5 import OBS_DIM_V5
    from src.envs.minibg.structured_actions import StructAction, StructActionType

    obs = torch.randn(2, OBS_DIM_V5 + _N_ID)
    legal = [[StructAction(StructActionType.ROLL)], [StructAction(StructActionType.ROLL)]]
    _, _, values = v8.policy_logits_and_value(obs, legal)
    _, cache = v8.encode_state(obs)
    assert torch.allclose(values, v8.value_from_trunk(cache["trunk"]), atol=1e-6)


def test_factory_build_serialize_restore_v8():
    ctx = load_patch_context(_PATCH)
    from src.envs.bglike.obs_v5 import OBS_DIM_V5

    net = build_ppo_actor_critic(
        PPO_NETWORK_BGLIKE_STRUCTURED_V8,
        (OBS_DIM_V5,),
        1,
        slot_hidden_channels=48,
        num_pool_indices=ctx.num_pool_indices,
    )
    assert isinstance(net, BGLikeStructuredV8)
    kw = default_ppo_network_kwargs(PPO_NETWORK_BGLIKE_STRUCTURED_V8, net)
    nt = ppo_network_type_for_save(PPO_NETWORK_BGLIKE_STRUCTURED_V8)
    restored = restore_ppo_actor_critic(nt, (OBS_DIM_V5,), 1, kw)
    assert isinstance(restored, BGLikeStructuredV8)


class _FakeBuf:
    """Minimal SegmentRolloutBuffer with placement labels."""

    def __init__(self, seat_ids):
        n = len(seat_ids)
        self.seat_ids = list(seat_ids)
        self.rewards = [0.0] * n
        self.dones = [False] * n
        self.obs = [np.zeros(3, dtype=np.float32)] * n
        self.placement_label = [-1] * n


def test_close_segment_stamps_placement_over_whole_segment():
    from src.agents.rollout_segments import close_rollout_segment

    # interleaved seats: 0 1 0 1 0
    buf = _FakeBuf([0, 1, 0, 1, 0])
    assert close_rollout_segment(buf, 0, 1.0, placement=3)
    assert buf.placement_label == [3, -1, 3, -1, 3]
    assert buf.dones == [False, False, False, False, True]
    assert close_rollout_segment(buf, 1, -1.0, placement=7)
    assert buf.placement_label == [3, 7, 3, 7, 3]


def test_close_segment_does_not_cross_previous_segment():
    from src.agents.rollout_segments import close_rollout_segment

    # seat 0 plays two episodes inside one buffer
    buf = _FakeBuf([0, 0, 0, 0])
    assert close_rollout_segment(buf, 0, 1.0, placement=2)
    assert buf.placement_label == [2, 2, 2, 2]
    # second episode's rows
    buf.seat_ids += [0, 0]
    buf.rewards += [0.0, 0.0]
    buf.dones += [False, False]
    buf.obs += [np.zeros(3, dtype=np.float32)] * 2
    buf.placement_label += [-1, -1]
    assert close_rollout_segment(buf, 0, -1.0, placement=8)
    assert buf.placement_label == [2, 2, 2, 2, 8, 8]
    assert buf.dones == [False, False, False, True, False, True]


def test_flat_update_runs_ce_and_emits_placement_acc():
    """End-to-end: collect a few rows from a real lobby with a v8 DvD agent,
    close segments with placements, run the PPO update."""
    from src.agents.ppo_dvd_agent import PPODvDAgent
    from src.envs.bglike.action_map import NUM_ENV_ACTIONS
    from src.envs.bglike.lobby_env import BGLobbyEnv
    from src.envs.bglike.obs_v5 import OBS_DIM_V5
    from src.envs.bglike.seat_config import lobby_from_learned_seats
    from src.agents.random_agent import RandomAgent

    ctx = load_patch_context("data/bgcore/19_6_0_74257")
    net = BGLikeStructuredV8(num_identities=_N_ID, **_base_kwargs(ctx))
    agent = PPODvDAgent(
        observation_shape=(int(OBS_DIM_V5),),
        observation_type="vector",
        num_actions=int(NUM_ENV_ACTIONS),
        network=net,
        ppo_network_type="bglike_structured_v8",
        num_identities=_N_ID,
        rollout_steps=8,
        minibatch_size=8,
        ppo_epochs=1,
        device="cpu",
        seed=0,
    )
    assert agent._distributional

    cfgs = lobby_from_learned_seats((0,), agent_by_seat={0: RandomAgent(seed=1)}, seed=1)
    env = BGLobbyEnv(
        cfgs,
        learned_seats=(0,),
        seed=1,
        patch_dir="data/bgcore/19_6_0_74257",
        obs_kind="bglike_v5",
    )
    env.reset()

    class _View:
        def __init__(self, lobby):
            self.state = lobby.state

    class _T:
        def __init__(self, obs, action, next_obs, legal, next_legal, seat):
            from src.agents.ppo_structured_minibg_agent import (
                INFO_STRUCT_LEGAL,
                INFO_STRUCT_NEXT_LEGAL,
            )

            self.obs = obs
            self.action = action
            self.reward = 0.0
            self.next_obs = next_obs
            self.terminated = False
            self.truncated = False
            self.info = {
                INFO_STRUCT_LEGAL: legal,
                INFO_STRUCT_NEXT_LEGAL: next_legal,
                "acting_seat": seat,
            }

    agent.train()
    n = 0
    while n < 10:
        cur = env.current_seat()
        if not env._seat_can_act(cur):
            break
        obs = env.obs_for_seat(cur)
        legal = env.legal_structured_actions_for_seat(cur)
        act, perm, idx = agent.act_structured(obs, legal, _View(env))
        env.step_structured_for_seat(cur, act, board_perm=perm)
        next_legal = env.legal_structured_actions_for_seat(env.current_seat())
        # raw (core-width) next obs, like the real trainer loop — the DvD agent
        # augments it itself at bootstrap time
        next_obs = env.obs_for_seat(env.current_seat())
        agent.observe(_T(agent._cache["obs"] if agent._cache else obs, idx, next_obs, legal, next_legal, cur))
        n += 1
    assert len(agent.rollout_buffer) >= 8
    # close every seat that has rows with an arbitrary placement
    for seat in sorted({s for s in agent.rollout_buffer.seat_ids}):
        agent.close_segment(seat, 1.0, placement=2)
    assert all(p >= 1 for p in agent.rollout_buffer.placement_label)

    metrics = agent.update()
    assert "placement_acc" in metrics
    assert np.isfinite(metrics["value_loss"])
    assert 0.0 <= metrics["placement_acc"] <= 1.0

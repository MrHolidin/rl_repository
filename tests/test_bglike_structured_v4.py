"""Smoke / unit tests for BGLikeStructuredV4 (round-level recurrent actor-critic)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.obs import OBS_DIM
from src.models.bglike_structured_v4 import BGLikeStructuredV4

_PATCH_36393 = "data/bgcore/15_6_2_36393"


def _make_v4(**overrides):
    ctx = load_patch_context(_PATCH_36393)
    kw = dict(
        slot_hidden=48,
        state_dim=128,
        action_dim=64,
        obs_layout="bglike",
        num_pool_indices=ctx.num_pool_indices,
        recurrent_hidden_dim=64,
    )
    kw.update(overrides)
    return BGLikeStructuredV4(**kw)


def test_v4_constructs_with_widened_state_summary():
    net = _make_v4(recurrent_hidden_dim=64)
    assert net.recurrent_hidden_dim == 64
    # state_summary widened by recurrent_hidden_dim.
    assert net.state_summary_ln.normalized_shape == (
        net._state_summary_base_dim + 64,
    )
    assert net.state_proj.in_features == net._state_summary_base_dim + 64
    # critic accepts the widened state_summary as input.
    assert net.critic[0].in_features == net._state_summary_base_dim + 64


def test_v4_forward_with_zero_hidden_runs():
    net = _make_v4()
    x = torch.zeros(2, OBS_DIM, dtype=torch.float32)
    state_emb, cache = net.encode_state(x)
    assert state_emb.shape == (2, net.state_dim)
    assert cache["h_prev"].shape == (2, net.recurrent_hidden_dim)
    # critic produces a single scalar per batch row.
    v = net.critic(cache["trunk"]).squeeze(-1)
    assert v.shape == (2,)


def test_v4_h_prev_changes_state_emb():
    net = _make_v4()
    x = torch.randn(3, OBS_DIM)
    h0 = net.zero_hidden(3)
    h_rand = torch.randn(3, net.recurrent_hidden_dim)
    state_emb_0, _ = net.encode_state(x, h_prev=h0)
    state_emb_r, _ = net.encode_state(x, h_prev=h_rand)
    # Different hidden states must produce different state_embs (otherwise
    # the GRU output isn't actually plumbed through the state summary).
    assert not torch.allclose(state_emb_0, state_emb_r, atol=1e-5)


def test_v4_step_round_hidden_changes_h():
    net = _make_v4()
    x = torch.randn(2, OBS_DIM)
    h0 = net.zero_hidden(2)
    state_emb, _ = net.encode_state(x, h_prev=h0)
    h1 = net.step_round_hidden(state_emb, h0)
    assert h1.shape == h0.shape
    # GRU was init'd small but state_emb is non-zero ⇒ h1 should not be exactly 0.
    assert h1.abs().sum().item() > 0.0


def test_v4_get_constructor_kwargs_round_trip():
    net = _make_v4(recurrent_hidden_dim=64)
    kw = net.get_constructor_kwargs()
    assert kw["recurrent_hidden_dim"] == 64
    assert "round_gru_init_scale" in kw
    # Reconstruct from kwargs should not throw.
    net2 = BGLikeStructuredV4(**kw)
    assert net2.recurrent_hidden_dim == net.recurrent_hidden_dim


def test_v4_h_prev_shape_validated():
    net = _make_v4()
    x = torch.zeros(2, OBS_DIM)
    bad_h = torch.zeros(2, net.recurrent_hidden_dim + 1)
    with pytest.raises(ValueError, match="h_prev"):
        net.encode_state(x, h_prev=bad_h)


def test_v4_save_load_via_factory(tmp_path):
    from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent

    net = _make_v4(recurrent_hidden_dim=32)
    agent = MiniBGPPOStructuredAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=64,
        network=net,
        ppo_network_type="bglike_structured_v4",
        ppo_network_kwargs=dict(net.get_constructor_kwargs()),
        device="cpu",
        rollout_steps=8,
    )
    assert agent._is_recurrent
    assert agent._recurrent_hidden_dim == 32

    path = tmp_path / "v4_agent.pt"
    agent.save(str(path))
    loaded = MiniBGPPOStructuredAgent.load(str(path), device="cpu")
    assert loaded._is_recurrent
    assert loaded._recurrent_hidden_dim == 32
    assert isinstance(loaded.policy_net, BGLikeStructuredV4)


def test_v4_recurrent_ppo_update_runs_on_synthetic_buffer():
    """End-to-end: stuff a tiny synthetic rollout into the agent and run one
    PPO update. Catches integration bugs between buffer fields, sequence
    grouping, and the per-timestep forward chain."""
    from src.agents.ppo_structured_minibg_agent import (
        MiniBGPPOStructuredAgent,
        StructuredMiniBGRolloutBuffer,
    )
    from src.envs.bglike.actions import BOARD_SIZE as _BG_BOARD
    from src.envs.minibg.structured_actions import StructAction, StructActionType

    net = _make_v4(recurrent_hidden_dim=32)
    agent = MiniBGPPOStructuredAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=64,
        network=net,
        ppo_network_type="bglike_structured_v4",
        ppo_network_kwargs=dict(net.get_constructor_kwargs()),
        device="cpu",
        rollout_steps=24,
        ppo_epochs=2,
        minibatch_size=8,
    )
    assert agent._is_recurrent

    # Construct 3 episodes × 2 seats × 4 steps each = 24 steps.
    # Each (episode, seat) sequence: 3 shop steps + 1 COMPLETE_TURN.
    rng = np.random.default_rng(0)
    H = agent._recurrent_hidden_dim
    buf = agent.rollout_buffer
    for ep in range(3):
        for seat in (0, 1):
            for t in range(4):
                is_complete = t == 3
                buf.add(
                    obs=rng.standard_normal(OBS_DIM, dtype=np.float32),
                    legal_list=[
                        StructAction(type=StructActionType.ROLL),
                        StructAction(type=StructActionType.COMPLETE_TURN),
                    ],
                    action_index=int(is_complete),
                    complete_turn=is_complete,
                    occupied_mask=np.zeros(_BG_BOARD, dtype=bool),
                    order_pick_row=np.full(_BG_BOARD, -1, dtype=np.int64),
                    reward=float(rng.normal()),
                    done=bool(is_complete and seat == 1 and t == 3 and ep == 2),
                    value=float(rng.normal() * 0.1),
                    log_prob=-float(np.log(2.0)),  # uniform over 2 legal
                    next_obs=rng.standard_normal(OBS_DIM, dtype=np.float32),
                    next_legal_list=[StructAction(type=StructActionType.ROLL)],
                    seat_id=seat,
                    h_prev=np.zeros(H, dtype=np.float32),
                    episode_id=ep,
                )

    assert len(buf) == 24
    metrics = agent.update()
    # Update should produce real metrics (not just the "not enough steps" stub).
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "bptt_sequences" in metrics
    # 3 episodes × 2 seats = 6 sequences.
    assert metrics["bptt_sequences"] == 6.0
    # Buffer cleared after update.
    assert len(agent.rollout_buffer) == 0


def test_v4_rollout_buffer_records_h_prev_and_episode_id():
    from src.agents.ppo_structured_minibg_agent import StructuredMiniBGRolloutBuffer
    from src.envs.minibg.structured_actions import StructAction, StructActionType

    buf = StructuredMiniBGRolloutBuffer()
    fake_obs = np.zeros(OBS_DIM, dtype=np.float32)
    fake_h = np.zeros(32, dtype=np.float32)
    legal = [StructAction(type=StructActionType.ROLL)]
    buf.add(
        obs=fake_obs,
        legal_list=legal,
        action_index=0,
        complete_turn=False,
        occupied_mask=np.zeros(7, dtype=bool),
        order_pick_row=np.full(7, -1, dtype=np.int64),
        reward=0.0,
        done=False,
        value=0.5,
        log_prob=-1.0,
        next_obs=fake_obs,
        next_legal_list=legal,
        seat_id=3,
        h_prev=fake_h,
        episode_id=7,
    )
    assert len(buf) == 1
    assert buf.h_prev[0].shape == (32,)
    assert buf.episode_ids[0] == 7
    buf.clear()
    assert len(buf.h_prev) == 0
    assert len(buf.episode_ids) == 0

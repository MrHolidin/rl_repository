"""Per-seat GAE with interleaved trajectories; segment closures; league outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.ppo_structured_minibg_agent import (
    INFO_STRUCT_LEGAL,
    INFO_STRUCT_NEXT_LEGAL,
    MiniBGPPOStructuredAgent,
)
from src.agents.rollout_segments import compute_gae_advantages
from src.envs.bglike.obs import OBS_DIM
from src.envs.bglike.placement import placement_reward, placement_score
from src.training.bglike_perspective import (
    league_outcomes_for_segment_closures,
    read_opponent_slot_by_seat,
)
from src.training.selfplay.league_state import SLOT_SCRIPTED


def test_gae_uniform_seat_ids_matches_linear_gae():
    """For non-interleaved buffers (e.g. MiniBG 2p, seat_ids all -1), per-seat
    GAE must be bit-identical to the standard linear GAE formula."""
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    dones = np.array([False, False, True], dtype=np.bool_)
    seat_ids = np.full(3, -1, dtype=np.int64)

    adv, ret = compute_gae_advantages(
        rewards, values, dones, seat_ids,
        discount_factor=1.0, gae_lambda=0.95, last_next_value=0.0,
    )

    # Hand-rolled linear GAE:
    #   t=2: delta = 1.0 - 0.3 = 0.7, gae = 0.7
    #   t=1: delta = 0 + 1.0*0.3 - 0.2 = 0.1, gae = 0.1 + 0.95*0.7 = 0.765
    #   t=0: delta = 0 + 1.0*0.2 - 0.1 = 0.1, gae = 0.1 + 0.95*0.765 = 0.82675
    assert adv == pytest.approx([0.82675, 0.765, 0.7])
    assert ret == pytest.approx([0.92675, 0.965, 1.0])


def test_gae_per_seat_propagates_terminal_reward_back_through_seat_only():
    """Two interleaved seats with terminal rewards on their last step. Per-seat
    GAE must propagate the +1 backward to seat 0's earlier action even though a
    seat-1 action sits between them in the buffer (which the buggy linear-with-
    seat-reset GAE would treat as a hard boundary and zero out)."""
    seat_ids = np.array([0, 1, 0, 1], dtype=np.int64)
    rewards = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
    values = np.array([0.1, 0.05, 0.1, 0.05], dtype=np.float32)
    dones = np.array([False, False, True, True], dtype=np.bool_)

    adv, ret = compute_gae_advantages(
        rewards, values, dones, seat_ids,
        discount_factor=1.0, gae_lambda=0.95, last_next_value=0.0,
    )

    # Seat 0 group [0, 2]:
    #   j=1 (t=2, done): delta = 1.0 - 0.1 = 0.9, gae = 0.9
    #   j=0 (t=0): bootstrap = values[2] = 0.1
    #     delta = 0 + 0.1 - 0.1 = 0.0
    #     gae = 0.0 + 0.95*0.9 = 0.855
    # Seat 1 group [1, 3]:
    #   j=1 (t=3, done): delta = -1.0 - 0.05 = -1.05, gae = -1.05
    #   j=0 (t=1): bootstrap = values[3] = 0.05
    #     delta = 0 + 0.05 - 0.05 = 0.0
    #     gae = 0.0 + 0.95*(-1.05) = -0.9975
    assert adv[0] == pytest.approx(0.855)
    assert adv[1] == pytest.approx(-0.9975)
    assert adv[2] == pytest.approx(0.9)
    assert adv[3] == pytest.approx(-1.05)

    # The buggy "treat seat boundary as terminal" GAE would give adv[0]=-0.1
    # (only the 1-step delta with zero bootstrap). Guard against regression.
    assert adv[0] > 0.5


def test_gae_per_seat_handles_multiple_episodes_for_same_seat():
    """Same seat_id appears in two episodes within one buffer. The mid-buffer
    dones=True from close_rollout_segment must reset the GAE accumulator so
    episode 1's terminal does not leak into episode 2's earlier transitions."""
    # Buffer: seat 0 plays game 1 (idx 0,1, terminates idx 1), then plays
    # game 2 (idx 2,3, terminates idx 3). Place 1.0 reward at game 1's end,
    # -1.0 at game 2's end.
    seat_ids = np.array([0, 0, 0, 0], dtype=np.int64)
    rewards = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    dones = np.array([False, True, False, True], dtype=np.bool_)

    adv, _ = compute_gae_advantages(
        rewards, values, dones, seat_ids,
        discount_factor=1.0, gae_lambda=1.0, last_next_value=0.0,
    )

    # Game 1 (lambda=1, so adv = MC return - V):
    #   t=1: delta = 1.0 - 0.5 = 0.5
    #   t=0: delta = 0 + 1.0*0.5 - 0.5 = 0; gae = 0 + 1*1*0.5 = 0.5
    # Game 2 (resets at t=3 because dones[3]=True; t=2 bootstraps from values[3]):
    #   t=3: delta = -1.0 - 0.5 = -1.5
    #   t=2: delta = 0 + 1.0*0.5 - 0.5 = 0; gae = 0 + 1*1*(-1.5) = -1.5
    # If episodes were leaking we'd see adv[0] mixed with game 2's reward.
    assert adv == pytest.approx([0.5, 0.5, -1.5, -1.5])


def test_gae_bootstrap_uses_last_next_value_for_buffer_tail():
    """When the very last in-buffer step has done=False, bootstrap from
    last_next_value (only used by single-seat / non-bglike callers)."""
    seat_ids = np.full(2, -1, dtype=np.int64)
    rewards = np.array([0.0, 0.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    dones = np.array([False, False], dtype=np.bool_)

    adv, _ = compute_gae_advantages(
        rewards, values, dones, seat_ids,
        discount_factor=1.0, gae_lambda=1.0, last_next_value=0.42,
    )

    # t=1: delta = 0 + 1.0*0.42 - 0 = 0.42; gae = 0.42
    # t=0: delta = 0 + 1.0*0 - 0 = 0; gae = 0 + 1*1*0.42 = 0.42
    assert adv == pytest.approx([0.42, 0.42])


def test_structured_close_segment_updates_last_step_for_seat():
    from src.models.minibg_structured_ac import MiniBGStructuredActorCritic

    net = MiniBGStructuredActorCritic(
        slot_hidden=16,
        trunk_hidden=32,
        obs_layout="bglike",
    )
    agent = MiniBGPPOStructuredAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=8,
        network=net,
        rollout_steps=64,
        device="cpu",
    )
    agent.train()
    legal: list = []
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    agent.rollout_buffer.add(
        obs=obs,
        legal_list=legal,
        action_index=0,
        complete_turn=False,
        occupied_mask=np.zeros(7, dtype=bool),
        order_pick_row=np.full(7, -1, dtype=np.int64),
        reward=0.0,
        done=False,
        value=0.0,
        log_prob=-1.0,
        next_obs=obs,
        next_legal_list=legal,
        seat_id=2,
    )
    assert agent.close_segment(2, placement_reward(4)) is True
    assert agent.rollout_buffer.rewards[0] == placement_reward(4)
    assert agent.rollout_buffer.dones[0] is True


def test_league_outcomes_per_segment_closure_dedupes_slots():
    closures = [
        {"seat": 1, "placement": 2, "placement_reward": placement_reward(2)},
        {"seat": 3, "placement": 6, "placement_reward": placement_reward(6)},
    ]
    slot_by_seat = {4: SLOT_SCRIPTED, 5: 7, 6: 7, 7: SLOT_SCRIPTED}
    outcomes = league_outcomes_for_segment_closures(closures, slot_by_seat)
    assert outcomes == [
        (SLOT_SCRIPTED, placement_score(2)),
        (7, placement_score(2)),
        (SLOT_SCRIPTED, placement_score(6)),
        (7, placement_score(6)),
    ]


def test_read_opponent_slot_by_seat_prefers_slot_by_seat():
    class Sampler:
        _slot_by_seat = {1: 3, 2: 4}
        _episode_slot_by_seat = {9: 9}

    assert read_opponent_slot_by_seat(Sampler()) == {1: 3, 2: 4}

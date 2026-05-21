"""Independent learner segments: GAE boundaries, observe, league outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.ppo_agent import PPOAgent, compute_gae_advantages
from src.envs.bglike.placement import placement_reward, placement_score
from src.features.action_space import DiscreteActionSpace
from src.models.simple_mlp import SimpleMLP
from src.training.bglike_perspective import (
    league_outcomes_for_segment_closures,
    read_opponent_slot_by_seat,
)
from src.training.selfplay.league_state import SLOT_SCRIPTED


def test_gae_zero_bootstrap_on_seat_change():
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    dones = np.array([False, False, True], dtype=np.float32)
    seat_ids = np.array([2, 5, 5], dtype=np.int64)

    adv, _ = compute_gae_advantages(
        rewards,
        values,
        dones,
        seat_ids,
        discount_factor=1.0,
        gae_lambda=1.0,
        last_next_value=0.0,
    )
    assert adv[0] == -0.5
    assert adv[1] == pytest.approx(0.4)
    assert adv[2] == pytest.approx(0.3)

    same_seats = np.array([2, 2, 2], dtype=np.int64)
    adv_same, _ = compute_gae_advantages(
        rewards,
        values,
        dones,
        same_seats,
        discount_factor=1.0,
        gae_lambda=1.0,
        last_next_value=0.0,
    )
    assert adv_same[0] != adv[0]


def test_observe_marks_done_on_seat_switch():
    net = SimpleMLP(input_size=8, num_actions=5, hidden_size=16)
    agent = PPOAgent(
        observation_shape=(8,),
        observation_type="vector",
        num_actions=5,
        network=net,
        action_space=DiscreteActionSpace(5),
        rollout_steps=64,
    )
    agent.train()
    legal = np.ones(5, dtype=bool)
    obs0 = np.zeros(8, dtype=np.float32)
    obs1 = np.ones(8, dtype=np.float32)
    agent.rollout_buffer.add(
        obs=obs0,
        action=1,
        reward=0.0,
        done=False,
        value=0.0,
        log_prob=-1.0,
        legal_mask=legal,
        next_obs=obs1,
        next_legal_mask=legal,
        seat_id=2,
    )
    agent._last_action_cache = {
        "obs": obs1,
        "legal_mask": legal,
        "action": 2,
        "value": 0.0,
        "log_prob": -1.0,
    }

    class T:
        def __init__(self, obs, action, seat, next_obs):
            self.obs = obs
            self.action = action
            self.reward = 0.0
            self.next_obs = next_obs
            self.terminated = False
            self.truncated = False
            self.legal_mask = legal
            self.next_legal_mask = legal
            self.info = {"acting_seat": seat}

    agent.observe(T(obs1, 2, 5, obs1))

    buf = agent.rollout_buffer
    assert len(buf) == 2
    assert buf.seat_ids == [2, 5]
    assert buf.dones[0] is True
    assert buf.dones[1] is False
    np.testing.assert_array_equal(buf.next_obs[0], obs0)


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

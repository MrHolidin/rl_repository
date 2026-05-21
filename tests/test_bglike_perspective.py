"""BGLike AgentPerspective + multi-current lobby."""

from __future__ import annotations

import numpy as np

from src.agents.random_agent import RandomAgent
from src.envs.bglike.action_map import NUM_ENV_ACTIONS
from src.envs.bglike.obs import OBS_DIM
from src.envs.bglike.placement import placement_reward
from src.training.bglike_perspective import (
    BGLikeAgentPerspectiveEnv,
    make_bglike_agent_perspective_env,
    make_bglike_shaping_fn,
)
from src.training.opponent_sampler import RandomOpponentSampler


class _Learner(RandomAgent):
    pass


def test_multi_current_agent_perspective_episode():
    learner = _Learner(seed=1)
    env = make_bglike_agent_perspective_env(
        RandomOpponentSampler(seed=2),
        num_current_seats=2,
        seed=10,
    )
    env.set_learner_agent(learner)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,)
    steps = 0
    while not env.done and steps < 800:
        mask = env.legal_actions_mask
        if not mask.any():
            break
        out = env.step(int(learner.act(obs, legal_mask=mask)))
        obs = out.obs
        assert not out.terminated or out.info.get("lobby_episode_done")
        steps += 1
    assert steps > 0


def test_final_reward_uses_placement_when_lobby_done():
    env = BGLikeAgentPerspectiveEnv.__new__(BGLikeAgentPerspectiveEnv)
    reward = env._final_reward_for_agent(
        {
            "acting_seat": 0,
            "placement_reward": placement_reward(3),
            "placements_current": {0: 3, 1: 5},
            "lobby_episode_done": True,
        }
    )
    assert reward == placement_reward(3)


def test_segment_closure_after_observe_closes_acting_seat():
    from src.agents.ppo_agent import PPOAgent, RolloutBuffer
    from src.features.action_space import DiscreteActionSpace
    from src.models.simple_mlp import SimpleMLP
    from src.training.bglike_perspective import apply_bglike_segment_closures_after_observe

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
    agent.rollout_buffer.add(
        obs=np.zeros(8, dtype=np.float32),
        action=1,
        reward=0.0,
        done=False,
        value=0.0,
        log_prob=-1.0,
        legal_mask=np.ones(5, dtype=bool),
        next_obs=np.zeros(8, dtype=np.float32),
        next_legal_mask=np.ones(5, dtype=bool),
        seat_id=3,
    )
    env = BGLikeAgentPerspectiveEnv.__new__(BGLikeAgentPerspectiveEnv)
    env._learner = agent
    apply_bglike_segment_closures_after_observe(
        env,
        {
            "segment_closures": [
                {"seat": 3, "placement": 4, "placement_reward": placement_reward(4)},
            ],
        },
    )
    assert agent.rollout_buffer.rewards[0] == placement_reward(4)
    assert agent.rollout_buffer.dones[0] is True


def test_shaping_only_on_combat_advanced():
    from unittest.mock import MagicMock

    base = MagicMock()
    base.current_seats = (0,)
    base.lobby.last_battle_signed.return_value = 0.5
    env = BGLikeAgentPerspectiveEnv.__new__(BGLikeAgentPerspectiveEnv)
    env._bg_base = base
    env._agent_token = 1
    env.shaping_fn = make_bglike_shaping_fn(0.1)
    step = MagicMock()
    step.terminated = False
    step.info = {"combat_advanced": False}
    assert env._reward_in_agent_perspective(step, agent_acted=True) == 0.0
    step.info = {"combat_advanced": True, "acting_seat": 0}
    assert env._reward_in_agent_perspective(step, agent_acted=True) == -0.05


def test_shaping_uses_acting_seat_not_average():
    from unittest.mock import MagicMock

    base = MagicMock()
    base.current_seats = (0, 1, 2, 3)

    def signed_for_seat(seat: int) -> float:
        return {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8}[seat]

    base.lobby.last_battle_signed.side_effect = signed_for_seat
    env = BGLikeAgentPerspectiveEnv.__new__(BGLikeAgentPerspectiveEnv)
    env._bg_base = base
    env._agent_token = 1
    env.shaping_fn = make_bglike_shaping_fn(0.1)
    step = MagicMock()
    step.terminated = False
    step.info = {"combat_advanced": True, "acting_seat": 2}
    assert env._reward_in_agent_perspective(step, agent_acted=True) == -0.06


def test_placement_reward_in_final_info():
    from src.envs.bglike.placement import placement_for_seat

    for place in range(1, 9):
        assert placement_reward(place) == (9 - 2 * place) / 7.0


def test_notify_episode_end_skips_league_record():
    from unittest.mock import MagicMock

    sampler = MagicMock()
    env = BGLikeAgentPerspectiveEnv.__new__(BGLikeAgentPerspectiveEnv)
    env.opponent_sampler = sampler
    env._episode_index = 0
    env._agent_token = 1
    env.notify_episode_end({"placements_current": {0: 2, 1: 6}})
    assert sampler.on_episode_end.call_count == 1
    _, info = sampler.on_episode_end.call_args[0]
    assert info["skip_league_record"] is True
    assert "agent_result" not in info
    assert info["placements_current"] == {0: 2, 1: 6}
    assert env._episode_index == 1


def test_ppo_close_segment_updates_last_step_for_seat():
    from src.agents.ppo_agent import PPOAgent, RolloutBuffer
    from src.features.action_space import DiscreteActionSpace
    from src.models.simple_mlp import SimpleMLP

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
    buf = agent.rollout_buffer
    buf.add(
        obs=np.zeros(8, dtype=np.float32),
        action=1,
        reward=0.0,
        done=False,
        value=0.0,
        log_prob=-1.0,
        legal_mask=np.ones(5, dtype=bool),
        next_obs=np.zeros(8, dtype=np.float32),
        next_legal_mask=np.ones(5, dtype=bool),
        seat_id=2,
    )
    buf.add(
        obs=np.ones(8, dtype=np.float32),
        action=2,
        reward=0.0,
        done=False,
        value=0.0,
        log_prob=-1.0,
        legal_mask=np.ones(5, dtype=bool),
        next_obs=np.ones(8, dtype=np.float32),
        next_legal_mask=np.ones(5, dtype=bool),
        seat_id=5,
    )
    assert agent.close_segment(2, placement_reward(4)) is True
    assert buf.rewards[0] == placement_reward(4)
    assert buf.dones[0] is True
    assert buf.dones[1] is False
    assert agent.close_segment(99, 0.0) is False

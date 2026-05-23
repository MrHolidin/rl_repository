"""Tests for unified lobby seat controller dispatch."""

from __future__ import annotations

import copy

import numpy as np

from src.agents import RandomAgent
from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent
from src.envs.bglike.action_map import NUM_ENV_ACTIONS
from src.envs.bglike.obs import OBS_DIM
from src.envs.bglike.lobby_env import make_bglike_training_env
from src.models.minibg_structured_ac import MiniBGStructuredActorCritic
from src.models.ppo_policy_factory import PPO_NETWORK_BGLIKE_STRUCTURED, ppo_network_type_for_save
from src.training.bglike_perspective import make_bglike_agent_perspective_env
from src.training.controller_step import describe_seat_controller, lobby_seat_step
from src.training.opponent_sampler import RandomOpponentSampler


def _make_structured_agent(*, seed: int = 0) -> MiniBGPPOStructuredAgent:
    net = MiniBGStructuredActorCritic(
        slot_hidden=16,
        trunk_hidden=32,
        obs_layout="bglike",
    )
    return MiniBGPPOStructuredAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=NUM_ENV_ACTIONS,
        network=net,
        ppo_network_type=ppo_network_type_for_save(PPO_NETWORK_BGLIKE_STRUCTURED),
        ppo_network_kwargs=net.get_constructor_kwargs(),
        seed=seed,
        rollout_steps=64,
    )


def test_lobby_seat_step_structured_opponent_via_step_auto():
    learner = _make_structured_agent(seed=1)
    opponent = copy.deepcopy(learner)
    opponent.eval()

    env = make_bglike_training_env(current_seats=(0,), seed=7)
    env.set_agents(learner, {s: opponent for s in range(1, 8)})
    env.reset(seed=11)

    inner = env.lobby
    steps = 0
    while not env.done and steps < 400:
        seat = env.acting_seat
        if seat is None:
            break
        if seat in env.current_seats:
            legal = env.legal_structured_actions()
            assert legal
            act, perm, _ = learner.act_structured(
                env._obs(),
                legal,
                env,
                deterministic=True,
            )
            env.step_structured(act, board_perm=perm)
        else:
            cur = inner.current_seat()
            assert inner._seat_can_act(cur)
            auto = inner.step_auto(cur, deterministic=True)
            assert auto.seat == cur
            assert isinstance(auto.action, int)
            assert auto.info.acting_seat == cur
            assert auto.controller
            assert auto.control_path in ("structured", "flat")
        steps += 1

    assert steps > 0


def test_lobby_seat_step_multi_current_structured_drain():
    agent = _make_structured_agent(seed=2)
    env = make_bglike_training_env(current_seats=(0, 1), seed=5)
    opponents = {s: RandomAgent(seed=s + 10) for s in range(2, 8)}
    env.set_agents(agent, opponents)
    env.reset(seed=13)

    inner = env.lobby
    seen_other_current = False
    steps = 0
    while not env.done and steps < 500:
        seat = env.acting_seat
        if seat is not None and seat in env.current_seats:
            legal = env.legal_structured_actions()
            if legal:
                act, perm, _ = agent.act_structured(
                    env._obs(),
                    legal,
                    env,
                    deterministic=True,
                )
                env.step_structured(act, board_perm=perm)
        else:
            cur = inner.current_seat()
            if inner._seat_can_act(cur):
                inner.step_auto(cur, deterministic=True)
                if cur in env.current_seats and cur != env.acting_seat:
                    seen_other_current = True
        steps += 1

    if not env.done:
        env.finish_lobby_to_end()
    assert steps > 0
    assert seen_other_current or env.done


def test_lobby_seat_step_direct_structured():
    agent = _make_structured_agent(seed=3)
    env = make_bglike_training_env(current_seats=(0,), seed=1)
    env.set_agents(agent, {s: RandomAgent(seed=s) for s in range(1, 8)})
    env.reset(seed=2)
    inner = env.lobby
    seat = inner.current_seat()
    assert inner._seat_can_act(seat)
    result = lobby_seat_step(
        inner,
        seat,
        agent,
        deterministic=True,
        controller_env=env,
    )
    assert result.info.acting_seat == seat
    assert isinstance(result.action, int)
    assert result.controller
    assert result.control_path == "structured"
    assert result.struct_action


def test_describe_seat_controller_heuristic():
    from src.envs.bglike.heuristic_bots import make_heuristic_agent

    agent = make_heuristic_agent("t1_random", seed=0)
    env = make_bglike_training_env(current_seats=(0,), seed=1)
    opponents = {s: agent if s == 1 else RandomAgent(seed=s + 10) for s in range(1, 8)}
    env.set_agents(RandomAgent(seed=2), opponents)
    env._opponent_slot_by_seat = {1: -2}
    env.reset(seed=3)
    inner = env.lobby
    label = describe_seat_controller(
        agent,
        seat=1,
        lobby=inner,
        controller_env=env,
    )
    assert "BGLikeHeuristicAgent" in label
    assert "t1_random" in label
    assert "slot=SCRIPTED" in label


def test_perspective_reset_with_structured_self_play_opponent():
    learner = _make_structured_agent(seed=4)
    opp = copy.deepcopy(learner)
    opp.eval()

    class _SingleSeatSampler(RandomOpponentSampler):
        def sample_for_seats(self, seats):
            return {int(s): opp for s in seats}

    env = make_bglike_agent_perspective_env(_SingleSeatSampler(), seed=9)
    env.set_learner_agent(learner)
    obs = env.reset()
    steps = 0
    while not env.done and steps < 300:
        legal = env.legal_structured_actions()
        if not legal:
            break
        act, perm, _ = learner.act_structured(obs, legal, env, deterministic=True)
        step = env.step_structured(act, board_perm=perm)
        obs = step.obs
        steps += 1
    assert steps > 0

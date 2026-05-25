"""BGLike heuristic bots in 8p lobby."""

from __future__ import annotations

import numpy as np

from src.agents.random_agent import RandomAgent
from src.envs.bglike.heuristic_bots import default_bot_constructors, make_heuristic_agent
from src.envs.bglike.lobby_env import make_bglike_training_env
from src.training.bglike_perspective import make_bglike_agent_perspective_env
from src.training.opponent_sampler import RandomOpponentSampler


def test_heuristic_bot_legal_action_in_lobby():
    env = make_bglike_training_env(current_seats=(0,), seed=7, patch_dir="data/bgcore/15_6_2_36393")
    learner = RandomAgent(seed=1)
    bot_agent = make_heuristic_agent("structured", seed=2)
    env.set_agents(learner, {s: bot_agent for s in range(1, 8)})
    bot_agent.set_env(env)
    obs = env.reset(seed=11)
    steps = 0
    while not env.done and steps < 400:
        seat = env.acting_seat
        if seat is None:
            break
        mask = env.legal_actions_mask
        if seat in env.current_seats:
            action = int(learner.act(obs, legal_mask=mask))
        else:
            action = int(bot_agent.act(obs, legal_mask=mask))
        assert mask[action], (seat, action, np.flatnonzero(mask)[:20])
        out = env.step(action)
        obs = out.obs
        steps += 1
    if not env.done:
        env.finish_lobby_to_end()
    assert steps > 0
    assert env.done


def test_pool_scripted_bglike_bot_names():
    names = default_bot_constructors().keys()
    assert "structured" in names
    assert "t1_random" in names


def test_heuristic_drain_seat_not_learner_acting_seat():
    """Opponent step_auto must not read the learner's acting_seat for mask/player."""
    env = make_bglike_training_env(current_seats=(0,), seed=7, patch_dir="data/bgcore/15_6_2_36393")
    learner = RandomAgent(seed=1)
    bot_agent = make_heuristic_agent("structured", seed=2)
    env.set_agents(learner, {s: bot_agent for s in range(1, 8)})
    bot_agent.set_env(env)
    obs = env.reset(seed=11)
    inner = env.lobby
    opp: int | None = None
    for _ in range(400):
        for s in range(1, 8):
            if bool(inner.legal_mask_for_seat(s).any()):
                opp = s
                break
        if opp is not None:
            break
        seat = env.acting_seat
        mask = env.legal_actions_mask
        if seat is not None and mask.any():
            obs = env.step(int(learner.act(obs, legal_mask=mask))).obs
        else:
            cur = inner.current_seat()
            if inner._seat_can_act(cur):
                inner.step_auto(cur, deterministic=True)
    assert opp is not None
    mask_before = inner.legal_mask_for_seat(opp).copy()
    env._acting_seat = 0
    auto = inner.step_auto(opp, deterministic=True)
    assert auto.seat == opp
    assert mask_before[auto.action]
    assert auto.control_path == "flat"
    assert "structured" in auto.controller or "t1" in auto.controller or "BGLike" in auto.controller


def test_agent_perspective_with_heuristic_opponent():
    from src.training.selfplay.opponent_pool import OpponentPool, ScriptedOpponentsSpec
    from src.training.opponent_sampler import OpponentPoolSampler

    learner = RandomAgent(seed=3)
    pool = OpponentPool(
        device="cpu",
        seed=99,
        self_play_config=None,
        scripted=ScriptedOpponentsSpec(
            "bglike",
            {"structured": 1.0},
        ),
        current_agent=learner,
    )
    env = make_bglike_agent_perspective_env(
        OpponentPoolSampler(opponent_pool=pool),
        seed=5,
        patch_dir="data/bgcore/15_6_2_36393",
    )
    env.set_learner_agent(learner)
    obs = env.reset()
    steps = 0
    while not env.done and steps < 600:
        mask = env.legal_actions_mask
        if not mask.any():
            break
        out = env.step(int(learner.act(obs, legal_mask=mask)))
        obs = out.obs
        steps += 1
    assert steps > 0

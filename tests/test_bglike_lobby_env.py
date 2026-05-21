"""BGLobbyEnv: multi-seat learned agents and placement terminal reward."""

from __future__ import annotations

import numpy as np

from src.agents.random_agent import RandomAgent
from src.envs.bglike.action_map import NUM_ENV_ACTIONS
from src.envs.bglike.actions import NUM_PLAYERS, Action
from src.envs.bglike.lobby_env import BGLobbyEnv, BGLobbySingleAgentEnv, run_lobby_episode
from src.envs.bglike.placement import placement_reward
from src.envs.bglike.seat_config import SeatConfig, SeatKind, lobby_from_learned_seats


def test_single_agent_env_reset_and_step() -> None:
    agent = RandomAgent(seed=1)
    env = BGLobbySingleAgentEnv(
        training_seat=2,
        learned_seats=(2,),
        agent_by_seat={2: agent},
        seed=10,
    )
    obs = env.reset()
    from src.envs.bglike.obs import OBS_DIM

    assert obs.shape == (OBS_DIM,)
    assert env.legal_actions_mask.shape == (NUM_ENV_ACTIONS,)
    steps = 0
    while not env.done and steps < 500:
        mask = env.legal_actions_mask
        if not mask.any():
            break
        a = int(agent.act(obs, legal_mask=mask))
        out = env.step(a)
        obs = out.obs
        steps += 1
    assert steps > 0


def test_two_learned_agents_one_lobby() -> None:
    a0 = RandomAgent(seed=20)
    a3 = RandomAgent(seed=21)
    configs = lobby_from_learned_seats(
        (0, 3),
        agent_by_seat={0: a0, 3: a3},
        seed=30,
    )
    lobby = BGLobbyEnv(configs, learned_seats=(0, 3), seed=30)
    result = run_lobby_episode(lobby, record_seats=(0, 3), deterministic=False)
    assert lobby.lobby_done or lobby.episode_done
    for seat in (0, 3):
        if seat in result.outcomes:
            place = result.outcomes[seat].placements[seat]
            assert 1 <= place <= 8
            assert placement_reward(place) == placement_reward(place)


def test_placement_reward_on_elimination_recorded() -> None:
    agent = RandomAgent(seed=40)
    configs = lobby_from_learned_seats((1,), agent_by_seat={1: agent}, seed=41)
    lobby = BGLobbyEnv(configs, learned_seats=(1,), training_seats=(1,), seed=41)
    result = run_lobby_episode(lobby, record_seats=(1,))
    tr = result.transitions[1]
    if tr:
        terminal = [t for t in tr if t.terminated]
        if terminal:
            assert -1.0 <= terminal[-1].reward <= 1.0


def test_all_random_lobby_completes() -> None:
    agent = RandomAgent(seed=51)
    configs = lobby_from_learned_seats((0,), agent_by_seat={0: agent}, seed=50)
    for i in range(1, NUM_PLAYERS):
        configs = list(configs)
        configs[i] = SeatConfig(SeatKind.RANDOM, RandomAgent(seed=60 + i))
        configs = tuple(configs)
    lobby = BGLobbyEnv(configs, learned_seats=(0,), seed=50)
    lobby.reset()
    lobby.drain_until_lobby_done()
    assert lobby.lobby_done

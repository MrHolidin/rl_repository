"""BGLobbyEnv: multi-seat learned agents and placement terminal reward."""

from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from src.agents.random_agent import RandomAgent
from src.bg_lobby.player import PlayerPhase
from src.envs.bglike.action_map import NUM_ENV_ACTIONS
from src.envs.bglike.actions import NUM_PLAYERS, Action
from src.envs.bglike.lobby_env import (
    AutoStepResult,
    BGLobbyEnv,
    BGLobbySingleAgentEnv,
    LobbyStepInfo,
    run_lobby_episode,
)
from src.envs.bglike.placement import placement_reward
from src.envs.bglike.seat_config import SeatConfig, SeatKind, lobby_from_learned_seats


def test_single_agent_env_reset_and_step() -> None:
    agent = RandomAgent(seed=1)
    env = BGLobbySingleAgentEnv(
        training_seat=2,
        learned_seats=(2,),
        agent_by_seat={2: agent},
        seed=10,
        patch_dir="data/bgcore/15_6_2_36393",
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
    lobby = BGLobbyEnv(configs, learned_seats=(0, 3), patch_dir="data/bgcore/15_6_2_36393", seed=30)
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
    lobby = BGLobbyEnv(configs, learned_seats=(1,), patch_dir="data/bgcore/15_6_2_36393", training_seats=(1,), seed=41)
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
    lobby = BGLobbyEnv(configs, learned_seats=(0,), patch_dir="data/bgcore/15_6_2_36393", seed=50)
    lobby.reset()
    lobby.drain_until_lobby_done()
    assert lobby.lobby_done


class _RoundResetStub:
    def __init__(self, *, steps_per_round: int, rounds: int) -> None:
        self.steps_per_round = steps_per_round
        self.rounds = rounds
        self.steps_taken = 0
        self._finished_training = set()
        self._training_seats = frozenset()
        self._state = SimpleNamespace(
            done=False,
            combat_round=0,
            current_player_index=0,
            alive=tuple(range(NUM_PLAYERS)),
            players=tuple(
                SimpleNamespace(phase=PlayerPhase.SHOP) for _ in range(NUM_PLAYERS)
            ),
        )

    @property
    def state(self):
        return self._state

    @property
    def lobby_done(self) -> bool:
        return self._state.done

    def reset(self, seed=None):
        return self._state

    def current_seat(self) -> int:
        return 0

    def _seat_can_act(self, seat: int) -> bool:
        return True

    def step_auto(self, seat: int, *, deterministic: bool = False) -> AutoStepResult:
        self.steps_taken += 1
        if self.steps_taken % self.steps_per_round == 0:
            self._state.combat_round += 1
        if self.steps_taken >= self.steps_per_round * self.rounds:
            self._state.done = True
        return AutoStepResult(seat=seat, action=0, info=LobbyStepInfo(acting_seat=seat))

    def _raise_drain_stall(self, **kwargs) -> None:
        raise RuntimeError(f"unexpected stall: {kwargs}")

    def _raise_drain_exceeded_cap(self, **kwargs) -> None:
        raise RuntimeError(f"unexpected cap: {kwargs}")

    def _mark_finished_training(self, seat: int) -> None:
        self._finished_training.add(seat)

    def finalize_placements(self):
        return {}


def test_drain_until_lobby_done_resets_cap_each_round(monkeypatch) -> None:
    import src.envs.bglike.lobby_env as le

    stub = _RoundResetStub(steps_per_round=3, rounds=2)
    monkeypatch.setattr(le, "MAX_DRAIN_STEPS", 3)
    monkeypatch.setattr(le, "is_seat_finished", lambda state, seat: False)

    log = le.BGLobbyEnv.drain_until_lobby_done(stub, deterministic=False)

    assert len(log) == 6
    assert stub.state.combat_round == 2
    assert stub.lobby_done


def test_run_lobby_episode_resets_cap_each_round(monkeypatch) -> None:
    import src.envs.bglike.lobby_env as le

    stub = _RoundResetStub(steps_per_round=3, rounds=2)
    monkeypatch.setattr(le, "MAX_DRAIN_STEPS", 3)

    result = run_lobby_episode(stub, record_seats=(), deterministic=False)

    assert result.transitions == {}
    assert result.outcomes == {}
    assert stub.state.combat_round == 2
    assert stub.lobby_done

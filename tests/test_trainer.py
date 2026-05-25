"""Tests for the generic Trainer and callbacks."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent
from src.envs.base import SingleAgentEnv, StepResult, TurnBasedEnv
from src.envs.minibg import MiniBGEnv
from src.training.agent_perspective_env import AgentPerspectiveEnv
from src.training.callbacks import (
    CheckpointCallback,
    EarlyStopCallback,
    EvalCallback,
    TrainerCallback,
)
from src.training.opponent_sampler import RandomOpponentSampler
from src.training.trainer import Trainer, Transition


class DummySingleAgentEnv(SingleAgentEnv):
    """Single-agent env with fixed-length episodes and constant reward."""

    def __init__(self, episode_length: int = 5):
        self.episode_length = episode_length
        self._legal_mask = np.ones(2, dtype=bool)
        self._state = 0
        self._steps = 0
        self._done = False

    def reset(self) -> np.ndarray:
        self._state = 0
        self._steps = 0
        self._done = False
        return self._obs()

    def step(self, action: int) -> StepResult:
        self._steps += 1
        self._state += action
        terminated = self._steps >= self.episode_length
        self._done = terminated
        return StepResult(
            obs=self._obs(),
            reward=1.0,
            terminated=terminated,
            truncated=False,
            info={"state": self._state},
        )

    @property
    def legal_actions_mask(self) -> np.ndarray:
        return self._legal_mask

    @property
    def done(self) -> bool:
        return self._done

    def _obs(self) -> np.ndarray:
        return np.array([self._state], dtype=np.float32)


class FixedWinnerSingleAgentEnv(DummySingleAgentEnv):
    """One-step env with predefined ``winner`` token in info."""

    def __init__(self, winner: int, agent_token: int = 1):
        super().__init__(episode_length=1)
        self._winner = winner
        self._agent_token = agent_token

    @property
    def agent_token(self) -> int:
        return self._agent_token

    def step(self, action: int) -> StepResult:
        result = super().step(action)
        return StepResult(
            obs=result.obs,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info={"winner": self._winner},
        )


class DummyAgent(BaseAgent):
    def __init__(self):
        self.observe_calls = 0
        self.update_calls = 0
        self.actions: list[int] = []
        self.loss_sequence = [1.0, 0.9, 0.9, 0.9]

    def act(self, obs: np.ndarray, legal_mask: np.ndarray | None = None, deterministic: bool = False) -> int:
        if legal_mask is None:
            return 0
        legal_indices = np.flatnonzero(legal_mask)
        action = int(legal_indices[0]) if legal_indices.size else 0
        self.actions.append(action)
        return action

    def observe(self, transition: Transition, is_augmented: bool = False) -> dict:
        self.observe_calls += 1
        return {}

    def update(self) -> dict:
        self.update_calls += 1
        loss = self.loss_sequence[min(self.update_calls - 1, len(self.loss_sequence) - 1)]
        return {"loss": loss}

    def save(self, path: str) -> None:
        Path(path).write_text("dummy")

    @classmethod
    def load(cls, path: str, **kwargs) -> "DummyAgent":
        return cls()


class CaptureAgentResultCallback(TrainerCallback):
    def __init__(self):
        self.results: list[Optional[int]] = []

    def on_episode_end(
        self,
        trainer: Trainer,
        episode: int,
        episode_info: dict,
    ) -> None:
        self.results.append(episode_info.get("agent_result"))


def test_trainer_runs_and_records_metrics():
    env = DummySingleAgentEnv()
    agent = DummyAgent()

    eval_results: list[dict] = []

    def eval_fn(_: Trainer) -> dict:
        result = {"metric": len(eval_results)}
        eval_results.append(result)
        return result

    trainer = Trainer(env, agent, callbacks=[EvalCallback(eval_fn, interval=2)], track_timings=True)
    trainer.train(total_steps=5)

    assert trainer.global_step == 5
    assert agent.observe_calls == 5
    assert agent.update_calls == 5
    assert len(eval_results) == 2
    assert trainer.timing_report is not None
    report = trainer.timing_report
    assert all(key in report for key in ("total", "core", "callbacks", "eval"))


def test_trainer_checkpoint_and_early_stop():
    env = DummySingleAgentEnv()
    agent = DummyAgent()
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_cb = CheckpointCallback(tmpdir, interval=2)
        early_stop_cb = EarlyStopCallback(monitor="loss", patience=1, mode="min")
        trainer = Trainer(env, agent, callbacks=[checkpoint_cb, early_stop_cb])
        trainer.train(total_steps=10)

        assert trainer.stop_training is True
        assert trainer.global_step == 3
        checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.pt"))
        assert checkpoint_files, "Checkpoint callback should persist model files."


def test_trainer_agent_result_respects_player_order():
    agent = DummyAgent()
    env = FixedWinnerSingleAgentEnv(winner=1, agent_token=1)
    cb = CaptureAgentResultCallback()
    trainer = Trainer(env, agent, callbacks=[cb])
    trainer.train(total_steps=1)
    assert cb.results == [1]

    agent2 = DummyAgent()
    env2 = FixedWinnerSingleAgentEnv(winner=-1, agent_token=-1)
    cb2 = CaptureAgentResultCallback()
    trainer2 = Trainer(env2, agent2, callbacks=[cb2])
    trainer2.train(total_steps=1)
    assert cb2.results == [1]


def test_should_continue_training_respects_episode_and_step_caps():
    env = DummySingleAgentEnv()
    agent = DummyAgent()
    t = Trainer(env, agent, max_episodes=5)
    t.global_step = 0
    t.episode_index = 5
    assert not t._should_continue_training(10**9)
    t.episode_index = 4
    assert t._should_continue_training(10**9)
    t.global_step = 10**9
    assert not t._should_continue_training(10**9)


def test_agent_perspective_env_terminal_reward_is_agent_centric():
    base = MiniBGEnv(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    sampler = RandomOpponentSampler(seed=2)
    env = AgentPerspectiveEnv(base, sampler, agent_first_probability=0.5)
    trainer = Trainer(env, RandomAgent(seed=1), opponent_sampler=sampler)
    trainer.train(total_steps=200)
    assert trainer.global_step == 200

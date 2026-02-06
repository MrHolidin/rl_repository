"""Tests for the generic Trainer and callbacks."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.base import StepResult, TurnBasedEnv
from src.training.callbacks import (
    CheckpointCallback,
    EarlyStopCallback,
    EvalCallback,
    TrainerCallback,
)
from src.training.trainer import StartPolicy, Trainer, Transition


class DummyEnv(TurnBasedEnv):
    def __init__(self, episode_length: int = 5):
        self.episode_length = episode_length
        self._legal_mask = np.ones(2, dtype=bool)
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._state = 0
        self._steps = 0
        return self._obs()

    def step(self, action: int) -> StepResult:
        self._steps += 1
        self._state += action
        terminated = self._steps >= self.episode_length
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

    def current_player(self) -> int:
        return 0

    def render(self) -> None:
        pass

    def _obs(self) -> np.ndarray:
        return np.array([self._state], dtype=np.float32)


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


class FixedWinnerEnv(DummyEnv):
    """Single-step environment that reports predefined winner token."""

    def __init__(self, winner: int):
        super().__init__(episode_length=1)
        self._winner = winner

    def step(self, action: int) -> StepResult:
        result = super().step(action)
        return StepResult(
            obs=result.obs,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info={"winner": self._winner},
        )


def test_trainer_runs_and_records_metrics():
    env = DummyEnv()
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
    # Eval should have run at steps 2 and 4
    assert len(eval_results) == 2
    assert trainer.timing_report is not None
    report = trainer.timing_report
    assert all(key in report for key in ("total", "core", "callbacks", "eval"))


def test_trainer_checkpoint_and_early_stop():
    env = DummyEnv()
    agent = DummyAgent()
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_cb = CheckpointCallback(tmpdir, interval=2)
        early_stop_cb = EarlyStopCallback(monitor="loss", patience=1, mode="min")
        trainer = Trainer(env, agent, callbacks=[checkpoint_cb, early_stop_cb])
        trainer.train(total_steps=10)

        # Early stop should trigger after loss stops improving
        assert trainer.stop_training is True
        # Step 1 loss=1.0 (best), step2 loss=0.9 (best), step3 loss=0.9 (no improvement) -> stop
        assert trainer.global_step == 3

        checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.pt"))
        assert checkpoint_files, "Checkpoint callback should persist model files."


def test_trainer_agent_result_respects_player_order():
    agent = DummyAgent()
    env = FixedWinnerEnv(winner=1)
    cb = CaptureAgentResultCallback()
    trainer = Trainer(env, agent, callbacks=[cb], start_policy=StartPolicy.AGENT_FIRST)
    trainer.train(total_steps=1)
    assert cb.results == [1]

    agent2 = DummyAgent()
    env2 = FixedWinnerEnv(winner=-1)
    cb2 = CaptureAgentResultCallback()
    trainer2 = Trainer(env2, agent2, callbacks=[cb2], start_policy=StartPolicy.OPPONENT_FIRST)
    trainer2.train(total_steps=1)
    assert cb2.results == [1]


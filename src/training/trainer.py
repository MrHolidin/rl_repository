"""Single-agent training loop.

Two-player turn alternation, opponent sampling, randomized openings and
zero-sum reward attribution live in
``src.training.agent_perspective_env.AgentPerspectiveEnv``. The trainer here
just consumes a `SingleAgentEnv` and runs ``(s, a, r, s')`` SARSA-style.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.base import SingleAgentEnv, StepResult
from src.training.opponent_sampler import OpponentSampler


@dataclass
class Transition:
    """Container for a single agent decision-point transition."""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    legal_mask: Optional[np.ndarray] = None
    next_legal_mask: Optional[np.ndarray] = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class StartPolicy(Enum):
    """Who acts first in an episode (consumed by `AgentPerspectiveEnv`)."""

    RANDOM = "random"
    AGENT_FIRST = "agent_first"
    OPPONENT_FIRST = "opponent_first"


class TrainerCallback:
    """Base class for trainer callbacks."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        pass

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        pass

    def on_episode_end(
        self,
        trainer: "Trainer",
        episode: int,
        episode_info: Dict[str, Any],
    ) -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass


class Trainer:
    """Generic single-agent training loop.

    The environment is expected to expose the `SingleAgentEnv` contract:
    ``reset()`` returns the obs at an agent decision point, ``step(action)``
    advances to the next decision point or terminal with a reward already
    in the agent's perspective. Two-player games should be wrapped in
    `AgentPerspectiveEnv` before being passed here.
    """

    def __init__(
        self,
        env: SingleAgentEnv,
        agent: BaseAgent,
        callbacks: Optional[Iterable[TrainerCallback]] = None,
        track_timings: bool = False,
        data_augment_fn: Optional[Callable[[Transition], Sequence[Transition]]] = None,
        max_episodes: Optional[int] = None,
        opponent_sampler: Optional[OpponentSampler] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.callbacks: List[TrainerCallback] = list(callbacks) if callbacks else []
        self.global_step = 0
        self.episode_index = 0
        self.stop_training = False
        self.track_timings = track_timings
        self._timings: Dict[str, float] = {"total": 0.0, "core": 0.0, "callbacks": 0.0}
        self.timing_report: Optional[Dict[str, float]] = None
        self.data_augment_fn = data_augment_fn
        self.max_episodes: Optional[int] = (
            int(max_episodes) if max_episodes is not None else None
        )
        # Kept for callback access (e.g. CheckpointCallback.on_checkpoint).
        # Trainer itself never drives the sampler -- `AgentPerspectiveEnv` does.
        self.opponent_sampler = opponent_sampler
        self._target_total_steps = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, total_steps: int, *, deterministic: bool = False) -> None:
        self._target_total_steps = total_steps
        for callback in self.callbacks:
            callback.on_train_begin(self)

        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        while self._should_continue_training(total_steps):
            iteration_start = perf_counter() if self.track_timings else None

            legal = self._as_bool_mask(self.env.legal_actions_mask)
            action = int(
                self.agent.act(obs, legal_mask=legal, deterministic=deterministic)
            )
            step = self.env.step(action)
            episode_length += 1
            episode_reward += float(step.reward)

            next_legal = (
                self._zero_mask_like(legal)
                if step.done
                else self._as_bool_mask(self.env.legal_actions_mask)
            )
            transition = Transition(
                obs=obs,
                action=action,
                reward=float(step.reward),
                next_obs=step.obs,
                terminated=step.terminated,
                truncated=step.truncated,
                info=step.info,
                legal_mask=legal,
                next_legal_mask=next_legal,
            )

            metrics = self._observe_and_update(transition)
            core_done = perf_counter() if self.track_timings else None
            self._after_transition(transition, metrics, iteration_start, core_done)

            if step.done:
                self._handle_episode_end(episode_reward, episode_length, step.info)
                episode_reward = 0.0
                episode_length = 0
                if self._should_continue_training(total_steps):
                    obs = self.env.reset()
            else:
                obs = step.obs

        for callback in self.callbacks:
            callback.on_train_end(self)

        if self.track_timings:
            self._finalize_timings()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_continue_training(self, total_steps: int) -> bool:
        if self.stop_training:
            return False
        if self.global_step >= total_steps:
            return False
        if self.max_episodes is not None and self.episode_index >= self.max_episodes:
            return False
        return True

    def _observe_and_update(self, transition: Transition) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        transitions: List[Transition] = [transition]
        if self.data_augment_fn is not None:
            extra = self.data_augment_fn(transition)
            if extra:
                if isinstance(extra, Transition):
                    transitions.append(extra)
                else:
                    transitions.extend(list(extra))

        for i, t in enumerate(transitions):
            is_augmented = i > 0
            observe_metrics = self.agent.observe(t, is_augmented=is_augmented) or {}
            self._merge_metrics(metrics, observe_metrics)

        update_metrics = self.agent.update() or {}
        self._merge_metrics(metrics, update_metrics)
        return metrics

    @staticmethod
    def _merge_metrics(dst: Dict[str, float], src: Dict[str, float]) -> None:
        for key, value in src.items():
            if (
                key in dst
                and isinstance(dst[key], (int, float))
                and isinstance(value, (int, float))
            ):
                dst[key] = 0.5 * (dst[key] + value)
            else:
                dst[key] = value

    def _after_transition(
        self,
        transition: Transition,
        metrics: Dict[str, float],
        iteration_start: Optional[float],
        core_done: Optional[float],
    ) -> None:
        if self.track_timings and iteration_start is not None and core_done is not None:
            self._timings["core"] += max(0.0, core_done - iteration_start)

        self.global_step += 1
        for callback in self.callbacks:
            cb_start = perf_counter() if self.track_timings else None
            callback.on_step_end(self, self.global_step, transition, metrics)
            if self.track_timings and cb_start is not None:
                self._timings["callbacks"] += perf_counter() - cb_start

        if self.track_timings and iteration_start is not None:
            self._timings["total"] += perf_counter() - iteration_start

    def _handle_episode_end(
        self,
        episode_reward: float,
        episode_length: int,
        info: Dict[str, Any],
    ) -> None:
        agent_token = int(getattr(self.env, "agent_token", 1))
        winner = info.get("winner") if isinstance(info, dict) else None
        agent_result = self._agent_relative_result(winner, agent_token)
        episode_info: Dict[str, Any] = {
            "reward": episode_reward,
            "length": episode_length,
            "info": info,
            "agent_token": agent_token,
            "agent_result": agent_result,
        }
        for callback in self.callbacks:
            callback.on_episode_end(self, self.episode_index, episode_info)
        self.env.notify_episode_end(info if isinstance(info, dict) else {})
        self.episode_index += 1

    @staticmethod
    def _agent_relative_result(winner: Optional[int], agent_token: int) -> Optional[int]:
        if winner is None:
            return None
        if winner == 0:
            return 0
        if winner == agent_token:
            return 1
        if winner == -agent_token:
            return -1
        return None

    @staticmethod
    def _as_bool_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        return np.asarray(mask, dtype=bool, order="C")

    @staticmethod
    def _zero_mask_like(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        return np.zeros_like(np.asarray(mask, dtype=bool))

    def _finalize_timings(self) -> None:
        total_time = self._timings["total"]
        core_time = min(self._timings["core"], total_time)
        callback_time = min(self._timings["callbacks"], total_time)
        eval_time = min(
            sum(getattr(cb, "time_spent", 0.0) for cb in self.callbacks),
            total_time,
        )

        def pct(value: float) -> float:
            return (value / total_time * 100.0) if total_time > 0 else 0.0

        self.timing_report = {
            "total": total_time,
            "core": core_time,
            "callbacks": callback_time,
            "eval": eval_time,
        }

        print(
            "[timings] total={total:.2f}s, core={core:.2f}s ({core_pct:.1f}%), "
            "callbacks={cb:.2f}s ({cb_pct:.1f}%), eval={eval:.2f}s ({eval_pct:.1f}%)".format(
                total=total_time,
                core=core_time,
                core_pct=pct(core_time),
                cb=callback_time,
                cb_pct=pct(callback_time),
                eval=eval_time,
                eval_pct=pct(eval_time),
            )
        )


__all__ = [
    "StartPolicy",
    "Trainer",
    "TrainerCallback",
    "Transition",
]

"""Generic training loop and callbacks for turn-based environments."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.base import StepResult, TurnBasedEnv


@dataclass
class Transition:
    """Container representing a single environment interaction."""

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


class EvalCallback(TrainerCallback):
    """Run evaluation at a fixed step interval."""

    def __init__(self, eval_fn: Callable[["Trainer"], Dict[str, float]], interval: int, name: str = "eval"):
        self.eval_fn = eval_fn
        self.interval = max(1, interval)
        self.name = name
        self.history: List[Dict[str, float]] = []
        self.time_spent: float = 0.0

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if step % self.interval == 0:
            start = perf_counter()
            result = self.eval_fn(trainer)
            self.time_spent += perf_counter() - start
            self.history.append(result)
            metrics.update({f"{self.name}/{k}": v for k, v in result.items()})


class CheckpointCallback(TrainerCallback):
    """Persist agent checkpoints during training."""

    def __init__(self, output_dir: str | Path, interval: int, prefix: str = "checkpoint"):
        self.output_dir = Path(output_dir)
        self.interval = max(1, interval)
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if step % self.interval == 0:
            path = self.output_dir / f"{self.prefix}_{step}.pt"
            trainer.agent.save(str(path))
            metrics["checkpoint_saved"] = step


class CSVLoggerCallback(TrainerCallback):
    """Append training metrics to a CSV file."""

    def __init__(self, csv_path: str | Path, fieldnames: Optional[Sequence[str]] = None):
        self.csv_path = Path(csv_path)
        self.fieldnames = list(fieldnames) if fieldnames is not None else None
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    def on_train_begin(self, trainer: "Trainer") -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="")
        if self.fieldnames is None:
            # Delay initialisation until first step to discover keys
            return
        self._writer = csv.DictWriter(self._file, fieldnames=["step"] + list(self.fieldnames))
        self._writer.writeheader()

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if self._file is None:
            return
        if self._writer is None:
            fieldnames = ["step"] + sorted(metrics.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()
        row = {"step": step}
        row.update(metrics)
        self._writer.writerow(row)
        self._file.flush()

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class WandbLoggerCallback(TrainerCallback):
    """Optional Weights & Biases logger."""

    def __init__(self, **wandb_init_kwargs: Any):
        try:
            import wandb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("wandb is not installed. Install wandb or disable WandbLoggerCallback.") from exc
        self.wandb = wandb
        self.wandb_init_kwargs = wandb_init_kwargs
        self._run = None

    def on_train_begin(self, trainer: "Trainer") -> None:
        self._run = self.wandb.init(**self.wandb_init_kwargs)

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if self._run is not None and metrics:
            self.wandb.log(metrics, step=step)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


class EarlyStopCallback(TrainerCallback):
    """Stop training when monitored metric satisfies condition."""

    def __init__(
        self,
        monitor: str,
        patience: int,
        mode: str = "max",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.mode = mode
        self._best: Optional[float] = None
        self._bad_steps = 0

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if self.monitor not in metrics:
            return
        value = metrics[self.monitor]
        improved = (
            self._best is None
            or (self.mode == "max" and value > self._best)
            or (self.mode == "min" and value < self._best)
        )
        if improved:
            self._best = value
            self._bad_steps = 0
        else:
            self._bad_steps += 1
            if self._bad_steps >= self.patience:
                trainer.stop_training = True


class Trainer:
    """Generic training loop that interacts with an environment via an agent."""

    def __init__(
        self,
        env: TurnBasedEnv,
        agent: BaseAgent,
        callbacks: Optional[Iterable[TrainerCallback]] = None,
        track_timings: bool = False,
    ) -> None:
        self.env = env
        self.agent = agent
        self.callbacks: List[TrainerCallback] = list(callbacks) if callbacks is not None else []
        self.global_step = 0
        self.episode_index = 0
        self.stop_training = False
        self.track_timings = track_timings
        self._timings: Dict[str, float] = {"total": 0.0, "core": 0.0, "callbacks": 0.0}
        self.timing_report: Optional[Dict[str, float]] = None

    def train(self, total_steps: int, *, deterministic: bool = False) -> None:
        """Run the training loop for the specified number of environment steps."""
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for callback in self.callbacks:
            callback.on_train_begin(self)

        while self.global_step < total_steps and not self.stop_training:
            legal_mask = getattr(self.env, "legal_actions_mask", None)
            if legal_mask is not None:
                legal_mask = np.asarray(legal_mask, dtype=bool)

            step_start = perf_counter() if self.track_timings else None

            action = self.agent.act(obs, legal_mask=legal_mask, deterministic=deterministic)
            step_result: StepResult = self.env.step(action)
            next_legal_mask = None
            if legal_mask is not None:
                if step_result.done:
                    next_legal_mask = np.zeros_like(legal_mask, dtype=bool)
                else:
                    next_legal_mask = np.asarray(self.env.legal_actions_mask, dtype=bool)

            transition = Transition(
                obs=obs,
                action=action,
                reward=step_result.reward,
                next_obs=step_result.obs,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
                info=step_result.info,
                legal_mask=legal_mask,
                next_legal_mask=next_legal_mask,
            )

            metrics = {}
            observe_metrics = self.agent.observe(transition) or {}
            update_metrics = self.agent.update() or {}
            metrics.update(observe_metrics)
            for key, value in update_metrics.items():
                if key in metrics and isinstance(metrics[key], (int, float)):
                    # average duplicate keys
                    metrics[key] = (metrics[key] + value) / 2.0
                else:
                    metrics[key] = value

            self.global_step += 1
            episode_reward += step_result.reward
            episode_length += 1

            if self.track_timings and step_start is not None:
                self._timings["core"] += perf_counter() - step_start

            for callback in self.callbacks:
                cb_start = perf_counter() if self.track_timings else None
                callback.on_step_end(self, self.global_step, transition, metrics)
                if self.track_timings and cb_start is not None:
                    elapsed = perf_counter() - cb_start
                    self._timings["callbacks"] += elapsed

            if step_result.done:
                episode_info = {
                    "reward": episode_reward,
                    "length": episode_length,
                    "info": step_result.info,
                }
                for callback in self.callbacks:
                    callback.on_episode_end(self, self.episode_index, episode_info)
                obs = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                self.episode_index += 1
            else:
                obs = step_result.obs

            if self.track_timings and step_start is not None:
                self._timings["total"] += perf_counter() - step_start

        for callback in self.callbacks:
            callback.on_train_end(self)

        if self.track_timings:
            self._finalize_timings()

    def _finalize_timings(self) -> None:
        total_time = self._timings["total"]
        core_time = self._timings["core"]
        callback_time = self._timings["callbacks"]
        eval_time = sum(
            getattr(cb, "time_spent", 0.0) for cb in self.callbacks if isinstance(cb, EvalCallback)
        )

        # Ensure non-negative residuals
        core_time = min(core_time, total_time)
        callback_time = min(callback_time, total_time)
        eval_time = min(eval_time, total_time)

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


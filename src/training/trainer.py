"""Generic training loop and callbacks for turn-based environments."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.base import StepResult, TurnBasedEnv
from src.training.opponent_sampler import OpponentSampler, RandomOpponentSampler


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


@dataclass
class PendingTransition:
    """Partial transition awaiting opponent response."""

    obs: np.ndarray
    action: int
    reward: float
    legal_mask: Optional[np.ndarray]


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
            if trainer.opponent_sampler is not None:
                trainer.opponent_sampler.on_checkpoint(path, trainer.episode_index)


class ProgressLoggerCallback(TrainerCallback):
    """Periodically print training progress with detailed metrics."""

    def __init__(
        self,
        step_interval: int = 10_000,
        *,
        print_fn: Callable[[str], None] = print,
        show_detailed_metrics: bool = True,
        detailed_metrics_interval: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        self.step_interval = max(1, step_interval) if step_interval else 0
        self.print_fn = print_fn
        self.show_detailed_metrics = show_detailed_metrics
        self.detailed_metrics_interval = detailed_metrics_interval or (step_interval // 2 if step_interval else 0)
        self.total_steps = total_steps
        
        # Episode statistics
        self._wins = 0
        self._draws = 0
        self._losses = 0
        self._episode_count = 0
        self._episode_metrics: List[Dict[str, float]] = []
        self._last_detailed_metrics_step = 0
        self._last_episode_summary_step = 0
        self._current_episode_train_steps = 0
        self._total_train_steps = 0
        self._latest_metrics: Dict[str, float] = {}

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Initialize tracking."""
        self._wins = 0
        self._draws = 0
        self._losses = 0
        self._episode_count = 0
        self._episode_metrics.clear()
        self._last_detailed_metrics_step = 0
        self._last_episode_summary_step = 0
        self._current_episode_train_steps = 0
        self._total_train_steps = 0
        self._latest_metrics.clear()
        self._last_total_train_steps = 0
        if hasattr(trainer, "_target_total_steps"):
            self.total_steps = trainer._target_total_steps

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        """Track metrics and print progress."""
        # Store latest metrics for detailed output
        if metrics:
            numeric_metrics = {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }
            self._episode_metrics.append(numeric_metrics)
            self._latest_metrics.update(numeric_metrics)
            
            # Count training steps (when loss or grad_norm is present)
            if "loss" in metrics or "grad_norm" in metrics:
                self._current_episode_train_steps += 1
                self._total_train_steps += 1
        
        # Print detailed metrics periodically (between episode summaries)
        if (
            self.show_detailed_metrics
            and self.detailed_metrics_interval > 0
            and step % self.detailed_metrics_interval == 0
            and step % self.step_interval != 0  # Don't print if we're about to print episode summary
        ):
            # Only print if we have detailed metrics
            if any(k in self._latest_metrics for k in ["max_q", "min_q", "avg_target_q"]):
                self._print_detailed_metrics(trainer, step)
                self._last_detailed_metrics_step = step

    def on_episode_end(
        self,
        trainer: "Trainer",
        episode: int,
        episode_info: Dict[str, Any],
    ) -> None:
        """Track episode statistics."""
        self._episode_count += 1
        info = episode_info.get("info", {})
        winner = info.get("winner") if isinstance(info, dict) else None
        
        if winner == 1:
            self._wins += 1
        elif winner == -1:
            self._losses += 1
        elif winner == 0:
            self._draws += 1
        
        # Print episode summary if at evaluation interval
        if self.step_interval and trainer.global_step - self._last_episode_summary_step >= self.step_interval:
            self._print_episode_summary(trainer, trainer.global_step, episode_info)
            self._last_episode_summary_step = trainer.global_step
            # Reset detailed metrics counter after episode summary
            self._last_detailed_metrics_step = trainer.global_step
            # Clear episode metrics after summary to start fresh
            self._episode_metrics.clear()
            self._current_episode_train_steps = 0
        else:
            # Clear episode metrics after tracking (but keep latest_metrics for detailed output)
            self._current_episode_train_steps = 0

    def _print_episode_summary(
        self,
        trainer: "Trainer",
        step: int,
        episode_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print episode summary with win/draw/loss rates."""
        total_episodes = max(1, self._episode_count)
        win_rate = self._wins / total_episodes
        draw_rate = self._draws / total_episodes
        loss_rate = self._losses / total_episodes
        
        # Compute average metrics from collected metrics since last summary
        avg_metrics = {}
        if self._episode_metrics:
            keys = set().union(*(m.keys() for m in self._episode_metrics))
            for key in keys:
                values = [m[key] for m in self._episode_metrics if key in m and m[key] != 0.0]
                if values:
                    avg_metrics[key] = sum(values) / len(values)
                elif self._latest_metrics.get(key, 0.0) != 0.0:
                    # Use latest if no average available
                    avg_metrics[key] = self._latest_metrics[key]
        
        # Get buffer info (use latest if available)
        buffer_size = self._latest_metrics.get("buffer_size", avg_metrics.get("buffer_size", 0))
        buffer_capacity = self._latest_metrics.get("buffer_capacity", avg_metrics.get("buffer_capacity", 0))
        if buffer_capacity == 0:
            # Try to get from agent
            if hasattr(trainer.agent, "replay_buffer"):
                buffer_size = len(trainer.agent.replay_buffer)
                buffer_capacity = trainer.agent.replay_buffer.capacity
        buffer_util = (buffer_size / buffer_capacity * 100) if buffer_capacity > 0 else 0.0
        
        # Get epsilon
        epsilon = getattr(trainer.agent, "epsilon", None)
        if epsilon is None:
            epsilon = self._latest_metrics.get("epsilon", avg_metrics.get("epsilon", 0.0))
        
        # Count training steps since last summary (from metrics that had loss/grad_norm)
        train_steps = sum(
            1 for m in self._episode_metrics if "loss" in m or "grad_norm" in m
        )
        # Use total train steps if available
        if train_steps == 0 and self._total_train_steps > 0:
            train_steps = self._total_train_steps - getattr(self, "_last_total_train_steps", 0)
            self._last_total_train_steps = self._total_train_steps
        
        # Get training metrics (prefer average over latest for episode summary)
        avg_loss = avg_metrics.get("loss", self._latest_metrics.get("loss", 0.0))
        avg_q = avg_metrics.get("avg_q", self._latest_metrics.get("avg_q", 0.0))
        td_error = avg_metrics.get("td_error", self._latest_metrics.get("td_error", 0.0))
        grad_norm = avg_metrics.get("grad_norm", self._latest_metrics.get("grad_norm", 0.0))
        
        # Build message (episode number without total, as total_steps is steps not episodes)
        episode_num = trainer.episode_index if hasattr(trainer, "episode_index") else self._episode_count
        
        msg = (
            f"Episode {episode_num} | "
            f"Win: {win_rate:.2%} | Draw: {draw_rate:.2%} | Loss: {loss_rate:.2%} | "
            f"Epsilon: {epsilon:.4f} | "
            f"Buffer: {int(buffer_size)}/{int(buffer_capacity)} ({buffer_util:.1f}%) | "
            f"Train steps: {train_steps} | "
            f"Loss: {avg_loss:.4f} | Avg Q: {avg_q:.4f} | TD Error: {td_error:.4f} | Grad Norm: {grad_norm:.4f}"
        )
        
        self.print_fn(msg)

    def _print_detailed_metrics(
        self,
        trainer: "Trainer",
        step: int,
    ) -> None:
        """Print detailed metrics from latest metrics."""
        max_q = self._latest_metrics.get("max_q", 0.0)
        min_q = self._latest_metrics.get("min_q", 0.0)
        avg_target_q = self._latest_metrics.get("avg_target_q", 0.0)
        grad_norm_clipped = self._latest_metrics.get("grad_norm_clipped", self._latest_metrics.get("grad_norm", 0.0))
        
        # Get step count from agent
        step_count = getattr(trainer.agent, "step_count", step)
        
        # Use total training steps tracked
        total_train_steps = self._total_train_steps
        
        msg = (
            f"  Detailed metrics: Max Q: {max_q:.4f} | Min Q: {min_q:.4f} | "
            f"Target Q: {avg_target_q:.4f} | Grad Norm (clipped): {grad_norm_clipped:.4f} | "
            f"Total train steps: {total_train_steps} | Step count: {step_count}"
        )
        
        self.print_fn(msg)


class CSVLoggerCallback(TrainerCallback):
    """Append training metrics to a CSV file."""

    def __init__(
        self,
        csv_path: str | Path,
        fieldnames: Optional[Sequence[str]] = None,
        mode: str = "step",
    ):
        self.csv_path = Path(csv_path)
        self.fieldnames = list(fieldnames) if fieldnames is not None else None
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: List[str] = []
        self.mode = mode if mode in {"step", "episode"} else "step"
        self._episode_metrics: List[Dict[str, float]] = []
        self._episode_count = 0
        self._wins = 0
        self._draws = 0
        self._losses = 0

    def on_train_begin(self, trainer: "Trainer") -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w+", newline="")
        self._fieldnames = ["step"]
        if self.mode == "episode":
            self._fieldnames.append("global_step")
            self._episode_metrics.clear()
            self._episode_count = 0
            self._wins = 0
            self._draws = 0
            self._losses = 0
        if self.fieldnames is not None:
            for name in self.fieldnames:
                if name not in self._fieldnames:
                    self._fieldnames.append(name)
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
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
        if self.mode == "episode":
            # Collect metrics for episode aggregation
            numeric_metrics = {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }
            # Always append, even if empty (to track that step occurred)
            self._episode_metrics.append(numeric_metrics)
            return
        # Step mode: write metrics immediately
        row: Dict[str, Any] = {"step": step, **metrics}

        new_fields = [key for key in row.keys() if key not in self._fieldnames]
        if new_fields:
            self._fieldnames.extend(sorted(new_fields))
            self._rewrite_with_new_fieldnames()

        self._write_row(row)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self.mode == "episode":
            self._episode_metrics.clear()
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            self._fieldnames = []

    def on_episode_end(
        self,
        trainer: "Trainer",
        episode: int,
        episode_info: Dict[str, Any],
    ) -> None:
        if self.mode != "episode" or self._file is None:
            return

        self._episode_count += 1
        row: Dict[str, Any] = {
            "step": self._episode_count,
            "global_step": trainer.global_step,
            "episode_reward": float(episode_info.get("reward", 0.0)),
            "episode_length": float(episode_info.get("length", 0)),
        }

        info = episode_info.get("info", {})
        winner = None
        if isinstance(info, dict):
            winner = info.get("winner")
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    row[f"info_{key}"] = float(value)

        if winner == 1:
            self._wins += 1
        elif winner == -1:
            self._losses += 1
        elif winner == 0:
            self._draws += 1
        if winner is not None:
            row["winner"] = winner

        total_episodes = max(1, self._episode_count)
        row["win_rate"] = self._wins / total_episodes
        row["draw_rate"] = self._draws / total_episodes
        row["loss_rate"] = self._losses / total_episodes

        epsilon = getattr(trainer.agent, "epsilon", None)
        if epsilon is not None:
            row["epsilon"] = float(epsilon)

        if self._episode_metrics:
            keys = set().union(*(m.keys() for m in self._episode_metrics))
            for key in sorted(keys):
                values = [m[key] for m in self._episode_metrics if key in m]
                if not values:
                    continue
                avg_value = sum(values) / len(values)
                if key.startswith("train_") or key.startswith("eval/") or "/" in key:
                    row[key] = avg_value
                else:
                    row[f"train_{key}"] = avg_value
        self._episode_metrics.clear()

        self._write_row(row)

    def _write_row(self, row: Dict[str, Any]) -> None:
        if self._file is None:
            return
        new_fields = [key for key in row.keys() if key not in self._fieldnames]
        if new_fields:
            self._fieldnames.extend(sorted(new_fields))
            self._rewrite_with_new_fieldnames()

        assert self._writer is not None
        ordered_row = {name: row.get(name, "") for name in self._fieldnames}
        self._writer.writerow(ordered_row)
        self._file.flush()

    def _rewrite_with_new_fieldnames(self) -> None:
        if self._file is None:
            return

        self._file.flush()
        self._file.seek(0)
        reader = csv.DictReader(self._file)
        existing_rows: List[Dict[str, Any]] = []
        if reader.fieldnames is not None:
            for row in reader:
                existing_rows.append(row)

        self._file.seek(0)
        self._file.truncate(0)
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()
        for row in existing_rows:
            ordered = {name: row.get(name, "") for name in self._fieldnames}
            self._writer.writerow(ordered)


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
        opponent_sampler: Optional[OpponentSampler] = None,
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
        self.opponent_sampler = opponent_sampler
        self._default_opponent_sampler = RandomOpponentSampler()
        self._current_opponent: Optional[BaseAgent] = None
        self._target_total_steps = 0

    def train(self, total_steps: int, *, deterministic: bool = False) -> None:
        """Run the training loop for the specified number of agent updates."""
        self._target_total_steps = total_steps
        reward_config = getattr(self.env, "reward_config", None)

        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        pending: Optional[PendingTransition] = None

        for callback in self.callbacks:
            callback.on_train_begin(self)

        self._prepare_opponent()

        while self.global_step < total_steps and not self.stop_training:
            iteration_start = perf_counter() if self.track_timings else None

            legal_mask = self._legal_mask_array(getattr(self.env, "legal_actions_mask", None))
            action = self.agent.act(obs, legal_mask=legal_mask, deterministic=deterministic)
            agent_step = self.env.step(action)
            episode_length += 1

            transition: Optional[Transition] = None
            metrics: Dict[str, float] = {}

            if agent_step.done:
                transition = Transition(
                    obs=obs,
                    action=action,
                    reward=agent_step.reward,
                    next_obs=agent_step.obs,
                    terminated=agent_step.terminated,
                    truncated=agent_step.truncated,
                    info=agent_step.info,
                    legal_mask=legal_mask,
                    next_legal_mask=self._zero_mask_like(legal_mask),
                )
                metrics = self._process_agent_transition(transition)
                episode_reward += agent_step.reward

                core_done = perf_counter() if self.track_timings else None
                self._after_transition(transition, metrics, iteration_start, core_done)

                obs, episode_reward, episode_length, pending = self._handle_episode_end(
                    episode_reward,
                    episode_length,
                    agent_step.info,
                )
                continue

            pending = PendingTransition(obs=obs, action=action, reward=agent_step.reward, legal_mask=legal_mask)
            obs = agent_step.obs

            opponent = self._current_opponent or self._prepare_opponent()
            opponent_mask = self._legal_mask_array(getattr(self.env, "legal_actions_mask", None))
            opponent_action = opponent.act(obs, legal_mask=opponent_mask, deterministic=False)
            opponent_step = self.env.step(opponent_action)
            episode_length += 1

            assert pending is not None  # for mypy/mind
            final_reward = self._compute_agent_reward(
                pending.reward,
                opponent_step,
                reward_config,
            )
            next_mask = (
                self._legal_mask_array(getattr(self.env, "legal_actions_mask", None))
                if not opponent_step.done
                else self._zero_mask_like(pending.legal_mask)
            )

            transition = Transition(
                obs=pending.obs,
                action=pending.action,
                reward=final_reward,
                next_obs=opponent_step.obs,
                terminated=opponent_step.terminated,
                truncated=opponent_step.truncated,
                info=opponent_step.info,
                legal_mask=pending.legal_mask,
                next_legal_mask=next_mask,
            )
            metrics = self._process_agent_transition(transition)
            episode_reward += final_reward
            pending = None

            core_done = perf_counter() if self.track_timings else None
            self._after_transition(transition, metrics, iteration_start, core_done)

            obs = opponent_step.obs
            if opponent_step.done:
                obs, episode_reward, episode_length, pending = self._handle_episode_end(
                    episode_reward,
                    episode_length,
                    opponent_step.info,
                )

        for callback in self.callbacks:
            callback.on_train_end(self)

        if self.track_timings:
            self._finalize_timings()

    def _process_agent_transition(self, transition: Transition) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        observe_metrics = self.agent.observe(transition) or {}
        metrics.update(observe_metrics)

        update_metrics = self.agent.update() or {}
        for key, value in update_metrics.items():
            if key in metrics and isinstance(metrics[key], (int, float)):
                metrics[key] = (metrics[key] + value) / 2.0
            else:
                metrics[key] = value
        return metrics

    def _after_transition(
        self,
        transition: Transition,
        metrics: Dict[str, float],
        iteration_start: Optional[float],
        core_done: Optional[float],
    ) -> None:
        if self.track_timings and iteration_start is not None and core_done is not None:
            self._timings["core"] += max(0.0, core_done - iteration_start)

        # Decay epsilon after each step (if agent supports it)
        # User should configure epsilon_decay appropriately for per-step updates
        if hasattr(self.agent, "epsilon") and hasattr(self.agent, "epsilon_decay") and hasattr(self.agent, "epsilon_min"):
            if self.agent.training and self.agent.epsilon > self.agent.epsilon_min:
                self.agent.epsilon *= self.agent.epsilon_decay
                # Ensure epsilon doesn't go below minimum
                if self.agent.epsilon < self.agent.epsilon_min:
                    self.agent.epsilon = self.agent.epsilon_min

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
    ) -> Tuple[np.ndarray, float, int, Optional[PendingTransition]]:
        episode_info = {
            "reward": episode_reward,
            "length": episode_length,
            "info": info,
        }
        for callback in self.callbacks:
            callback.on_episode_end(self, self.episode_index, episode_info)

        if self.opponent_sampler is not None:
            self.opponent_sampler.on_episode_end(self.episode_index, episode_info)

        obs = self.env.reset()
        self.episode_index += 1
        self._current_opponent = None

        if not self.stop_training and self.global_step < self._target_total_steps:
            self._prepare_opponent()

        return obs, 0.0, 0, None

    def _prepare_opponent(self) -> BaseAgent:
        sampler = self.opponent_sampler or self._default_opponent_sampler
        sampler.prepare(self.episode_index)
        opponent = sampler.sample()
        opponent.eval()
        if hasattr(opponent, "epsilon"):
            setattr(opponent, "epsilon", 0.0)
        self._current_opponent = opponent
        return opponent

    def _legal_mask_array(self, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if isinstance(mask, np.ndarray) and mask.dtype == bool:
            return mask.astype(bool, copy=True)
        return np.asarray(mask, dtype=bool)

    def _zero_mask_like(self, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        return np.zeros_like(mask, dtype=bool)

    def _compute_agent_reward(
        self,
        pending_reward: float,
        opponent_step: StepResult,
        reward_config: Any,
    ) -> float:
        """
        Compute final reward for agent's action after opponent's move.
        
        When opponent finishes the game (opponent_step.done == True):
        - If winner == 0: draw -> reward_config.draw
        - Otherwise: opponent won -> reward_config.loss (agent lost)
        
        When game continues: return pending_reward (shaping reward from agent's move).
        """
        if not opponent_step.done:
            # Game continues: return shaping reward from agent's move
            return pending_reward

        # Game ended after opponent's move
        winner = opponent_step.info.get("winner")
        if reward_config is not None:
            if winner == 0:
                # Draw
                return getattr(reward_config, "draw", 0.0)
            else:
                # Opponent won (regardless of whether winner == 1 or winner == -1)
                # Agent lost
                return getattr(reward_config, "loss", -1.0)

        # Fallback if no reward_config
        if winner == 0:
            return 0.0
        else:
            # Opponent won, agent lost
            return -1.0

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


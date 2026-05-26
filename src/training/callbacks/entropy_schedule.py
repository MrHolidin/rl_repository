"""Entropy coefficient scheduler for PPO-like agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.training.trainer import TrainerCallback, Transition

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class EntropyCoefScheduleCallback(TrainerCallback):
    """Linearly anneal ``agent.entropy_coef`` from ``start_coef`` to ``end_coef``.

    Schedule is parameterised by *steps*: at step 0 the coefficient is
    ``start_coef``; at step ``schedule_steps`` (and later) it is ``end_coef``.
    Updates are only written every ``interval_steps`` to keep CSV noise low.

    Distributed setup note: only the host runs PPO updates (and thus consults
    ``agent.entropy_coef``). Workers only do forward passes; they don't read
    this field. So updating on the host is sufficient.
    """

    def __init__(
        self,
        *,
        start_coef: float,
        end_coef: float,
        schedule_steps: int,
        interval_steps: int = 1000,
        metric_key: str = "entropy_coef",
    ) -> None:
        if schedule_steps <= 0:
            raise ValueError("schedule_steps must be > 0")
        if interval_steps <= 0:
            raise ValueError("interval_steps must be > 0")
        self.start_coef = float(start_coef)
        self.end_coef = float(end_coef)
        self.schedule_steps = int(schedule_steps)
        self.interval_steps = int(interval_steps)
        self.metric_key = metric_key
        # Track last step where we wrote the coef. Step counts can jump by ~rollout_steps
        # at a time in distributed mode (not by 1), so % interval_steps won't reliably
        # hit zero. Fire whenever we cross the next interval boundary.
        self._last_applied_step: int = -1

    def _coef_for_step(self, step: int) -> float:
        if step <= 0:
            return self.start_coef
        if step >= self.schedule_steps:
            return self.end_coef
        frac = float(step) / float(self.schedule_steps)
        return self.start_coef + (self.end_coef - self.start_coef) * frac

    def on_train_begin(self, trainer: "Trainer") -> None:
        agent = getattr(trainer, "agent", None)
        if agent is None or not hasattr(agent, "entropy_coef"):
            return
        agent.entropy_coef = self.start_coef

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: dict,
    ) -> None:
        # Fire on every crossing of an `interval_steps` boundary, not on exact
        # divisibility. In distributed mode `step` jumps by ~rollout_steps per
        # round, so a strict modulo never hits.
        if step - self._last_applied_step < self.interval_steps and self._last_applied_step >= 0:
            return
        agent = getattr(trainer, "agent", None)
        if agent is None or not hasattr(agent, "entropy_coef"):
            return
        coef = self._coef_for_step(step)
        agent.entropy_coef = coef
        self._last_applied_step = step
        if self.metric_key:
            metrics[self.metric_key] = coef


__all__ = ["EntropyCoefScheduleCallback"]

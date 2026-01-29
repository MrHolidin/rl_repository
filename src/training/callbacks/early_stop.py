"""Early stopping callback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.training.trainer import TrainerCallback, Transition

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class EarlyStopCallback(TrainerCallback):
    """Stop training when monitored metric satisfies condition."""

    def __init__(self, monitor: str, patience: int, mode: str = "max") -> None:
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
        metrics: dict,
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

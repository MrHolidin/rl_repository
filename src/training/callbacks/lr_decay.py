"""Learning rate decay callback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.training.trainer import TrainerCallback, Transition

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class LearningRateDecayCallback(TrainerCallback):
    """Multiply optimizer learning rate by decay_factor every N steps."""

    def __init__(
        self,
        *,
        interval_steps: int,
        decay_factor: float,
        min_lr: Optional[float] = None,
        optimizer_attr: str = "optimizer",
        metric_key: str = "learning_rate",
    ) -> None:
        if interval_steps <= 0:
            raise ValueError("interval_steps must be > 0")
        if decay_factor <= 0:
            raise ValueError("decay_factor must be > 0")
        self.interval_steps = interval_steps
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.optimizer_attr = optimizer_attr
        self.metric_key = metric_key
        self._optimizer = None
        self._warned_missing_optimizer = False

    def on_train_begin(self, trainer: "Trainer") -> None:
        self._optimizer = getattr(trainer.agent, self.optimizer_attr, None)
        if self._optimizer is None and not self._warned_missing_optimizer:
            print(
                f"[LearningRateDecayCallback] Agent has no '{self.optimizer_attr}'. "
                "Skipping LR decay."
            )
            self._warned_missing_optimizer = True

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: dict,
    ) -> None:
        if self._optimizer is None:
            return
        if step % self.interval_steps != 0:
            return

        new_lr = None
        for group in self._optimizer.param_groups:
            old_lr = group.get("lr", None)
            if old_lr is None:
                continue
            updated = old_lr * self.decay_factor
            if self.min_lr is not None:
                updated = max(updated, self.min_lr)
            group["lr"] = updated
            new_lr = updated

        if new_lr is None:
            return

        agent = getattr(trainer, "agent", None)
        if agent is not None and hasattr(agent, "learning_rate"):
            agent.learning_rate = new_lr

        if self.metric_key:
            metrics[self.metric_key] = new_lr

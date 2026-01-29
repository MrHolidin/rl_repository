"""Weights & Biases logging callback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.training.trainer import TrainerCallback, Transition

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class WandbLoggerCallback(TrainerCallback):
    """Optional Weights & Biases logger."""

    def __init__(self, **wandb_init_kwargs: Any):
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "wandb is not installed. Install wandb or disable WandbLoggerCallback."
            ) from exc
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
        metrics: dict,
    ) -> None:
        if self._run is not None and metrics:
            self.wandb.log(metrics, step=step)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

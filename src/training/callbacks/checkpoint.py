"""Checkpoint callback: persist agent checkpoints during training."""

from __future__ import annotations

from pathlib import Path

from src.training.trainer import Trainer, TrainerCallback, Transition


class CheckpointCallback(TrainerCallback):
    """Persist agent checkpoints during training."""

    def __init__(self, output_dir: str | Path, interval: int, prefix: str = "checkpoint"):
        self.output_dir = Path(output_dir)
        self.interval = max(1, interval)
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        transition: Transition,
        metrics: dict,
    ) -> None:
        if step % self.interval == 0:
            path = self.output_dir / f"{self.prefix}_{step}.pt"
            trainer.agent.save(str(path))
            metrics["checkpoint_saved"] = step
            if trainer.opponent_sampler is not None:
                trainer.opponent_sampler.on_checkpoint(path, trainer.episode_index)

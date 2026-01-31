"""Metrics file callback: log training metrics to CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.training.trainer import TrainerCallback, Transition

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class MetricsFileCallback(TrainerCallback):
    """Log training metrics to CSV."""

    FIELDS = (
        "step",
        "episode",
        "epsilon",
        "learning_rate",
        "avg_q",
        "avg_target_q",
        "target_q_p95",
        "target_q_max",
        "td_error",
        "td_error_p95",
        "td_error_max",
        "q_spread",
        "top2_gap",
        "grad_norm",
        "update_magnitude",
    )

    def __init__(
        self,
        run_dir: Path,
        interval: int = 100,
        *,
        filename: str = "metrics.csv",
    ):
        self.run_dir = Path(run_dir)
        self.interval = max(1, interval)
        self.path = self.run_dir / filename
        self._file = None
        self._writer = None
        self._last: Dict[str, Any] = {}

    def on_train_begin(self, trainer: "Trainer") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=list(self.FIELDS), extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()
        self._last.clear()

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if step % self.interval != 0:
            return

        agent = trainer.agent
        epsilon = getattr(agent, "epsilon", None)
        lr = getattr(agent, "learning_rate", None)
        if lr is None and hasattr(agent, "optimizer"):
            try:
                lr = agent.optimizer.param_groups[0].get("lr")
            except (IndexError, KeyError):
                pass

        self._last["epsilon"] = epsilon
        self._last["learning_rate"] = lr
        if metrics:
            for key in self.FIELDS:
                if key not in ("step", "episode") and key in metrics:
                    self._last[key] = metrics[key]

        row = {"step": step, "episode": trainer.episode_index}
        for key in self.FIELDS:
            if key not in row:
                row[key] = self._format(self._last.get(key))
        self._writer.writerow(row)
        self._file.flush()

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        return str(value)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

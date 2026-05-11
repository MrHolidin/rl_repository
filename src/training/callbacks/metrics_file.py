"""Metrics file callback: log training metrics to CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

from src.training.metrics_presets import LEGACY_DQN_METRICS_FIELDS
from src.training.trainer import TrainerCallback, Transition

if TYPE_CHECKING:
    from src.training.trainer import Trainer

FieldNames = Union[Sequence[str], Tuple[str, ...]]


class MetricsFileCallback(TrainerCallback):
    """Log training metrics to CSV.

    ``fieldnames`` defaults to the legacy DQN column set for backward compatibility.
    Training ``run`` resolves columns from ``agent.id`` (use ``preset: auto`` in yaml).
    """

    FIELDS = LEGACY_DQN_METRICS_FIELDS

    def __init__(
        self,
        run_dir: Path,
        interval: int = 100,
        *,
        filename: str = "metrics.csv",
        fieldnames: Optional[FieldNames] = None,
    ):
        self.run_dir = Path(run_dir)
        self.interval = max(1, interval)
        self.path = self.run_dir / filename
        self._fieldnames: Tuple[str, ...] = tuple(
            fieldnames if fieldnames is not None else LEGACY_DQN_METRICS_FIELDS
        )
        self._file = None
        self._writer = None
        self._last: Dict[str, Any] = {}

    def on_train_begin(self, trainer: "Trainer") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=list(self._fieldnames), extrasaction="ignore"
        )
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
            for key in self._fieldnames:
                if key not in ("step", "episode") and key in metrics:
                    self._last[key] = metrics[key]

        if step % self.interval != 0:
            return

        row = {"step": step, "episode": trainer.episode_index}
        for key in self._fieldnames:
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

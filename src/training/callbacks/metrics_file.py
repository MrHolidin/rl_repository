"""Metrics file callback: log training metrics to grouped CSV files.

Instead of one wide ``metrics.csv`` (60-80 columns, most blank per run), this
writes several narrow ``metrics_<group>.csv`` files (core / critic / rollout /
dvd / ...), each created lazily when its group first produces data and joinable
on ``step``. Routing is defined in :mod:`src.training.metric_groups`; any metric
the agent emits is captured (unregistered ones go to ``metrics_misc.csv``), so
new metrics never need a column list to be edited first.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from src.training.metric_groups import (
    DVD_IDENTITIES_GROUP,
    GROUP_COLUMNS,
    IGNORED_KEYS,
    display_name,
    header_for,
    is_wide_group,
    parse_dvd_identity,
    route,
)
from src.training.trainer import TrainerCallback, Transition

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class MetricsFileCallback(TrainerCallback):
    """Log training metrics to grouped CSV files (``<prefix>_<group>.csv``)."""

    def __init__(
        self,
        run_dir: Path,
        interval: int = 100,
        *,
        prefix: str = "metrics",
    ):
        self.run_dir = Path(run_dir)
        self.interval = max(1, interval)
        self.prefix = prefix
        self._files: Dict[str, Any] = {}
        self._writers: Dict[str, "csv.DictWriter"] = {}
        self._last: Dict[str, Any] = {}
        self._seen_groups: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_train_begin(self, trainer: "Trainer") -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Drop stale grouped files from a previous run so groups that no longer
        # appear don't linger with old data.
        for old in self.run_dir.glob(f"{self.prefix}_*.csv"):
            try:
                old.unlink()
            except OSError:
                pass
        self._last.clear()
        self._seen_groups.clear()

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        self._absorb(trainer, metrics)
        if step % self.interval != 0:
            return
        episode = getattr(trainer, "episode_index", 0)
        for group in sorted(self._seen_groups):
            for row in self._rows_for(group, step, episode):
                self._write(group, row)

    def on_train_end(self, trainer: "Trainer") -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._writers.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _absorb(self, trainer: "Trainer", metrics: Dict[str, float]) -> None:
        """Fold this step's metrics into the carry-forward state."""
        agent = getattr(trainer, "agent", None)
        self._last["epsilon"] = getattr(agent, "epsilon", None)
        self._last["learning_rate"] = self._learning_rate(agent)
        self._seen_groups.add("core")  # lr/epsilon always make core relevant

        if not metrics:
            return
        for key, value in metrics.items():
            if key in IGNORED_KEYS:
                continue
            self._last[key] = value
            self._seen_groups.add(route(key))

    @staticmethod
    def _learning_rate(agent: Any) -> Any:
        lr = getattr(agent, "learning_rate", None)
        if lr is None and hasattr(agent, "optimizer"):
            try:
                lr = agent.optimizer.param_groups[0].get("lr")
            except (IndexError, KeyError, AttributeError):
                lr = None
        return lr

    def _rows_for(self, group: str, step: int, episode: int) -> List[Dict[str, Any]]:
        index = {"step": step, "episode": episode}
        if is_wide_group(group):
            row = dict(index)
            for col in GROUP_COLUMNS[group]:
                row[col] = self._fmt(self._last.get(col))
            return [row]
        if group == DVD_IDENTITIES_GROUP:
            return self._dvd_identity_rows(index)
        # Generic long format: one (metric, value) row per key in this group.
        rows: List[Dict[str, Any]] = []
        for key, value in self._last.items():
            if route(key) != group:
                continue
            rows.append({**index, "metric": display_name(key), "value": self._fmt(value)})
        return rows

    def _dvd_identity_rows(self, index: Dict[str, Any]) -> List[Dict[str, Any]]:
        by_identity: Dict[int, Dict[str, Any]] = {}
        for key, value in self._last.items():
            parsed = parse_dvd_identity(key)
            if parsed is None:
                continue
            field, idx = parsed
            by_identity.setdefault(idx, {})[field] = value
        rows: List[Dict[str, Any]] = []
        for idx in sorted(by_identity):
            fields = by_identity[idx]
            rows.append(
                {
                    **index,
                    "identity": idx,
                    "place": self._fmt(fields.get("place")),
                    "tribe": self._fmt(fields.get("tribe")),
                    "assigned_frac": self._fmt(fields.get("assigned_frac")),
                }
            )
        return rows

    def _write(self, group: str, row: Dict[str, Any]) -> None:
        writer = self._writers.get(group)
        if writer is None:
            header: Tuple[str, ...] = header_for(group)
            path = self.run_dir / f"{self.prefix}_{group}.csv"
            f = open(path, "w", newline="")
            writer = csv.DictWriter(f, fieldnames=list(header), extrasaction="ignore")
            writer.writeheader()
            self._files[group] = f
            self._writers[group] = writer
        writer.writerow(row)
        self._files[group].flush()

    @staticmethod
    def _fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return str(value)
            return f"{value:.6g}"
        return str(value)

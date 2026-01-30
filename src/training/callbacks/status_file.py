"""Status file callback: heartbeat, progress, graceful stop via file or signal."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.training.trainer import TrainerCallback, Transition

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class StatusFileCallback(TrainerCallback):
    """Write status.json with heartbeat; check for stop file to request graceful shutdown."""

    def __init__(
        self,
        run_dir: Path,
        interval: int = 100,
        *,
        total_steps: Optional[int] = None,
    ):
        self.run_dir = Path(run_dir)
        self.interval = max(1, interval)
        self.total_steps = total_steps
        self.status_path = self.run_dir / "status.json"
        self.stop_path = self.run_dir / "stop"
        self._start_time: Optional[str] = None

    def on_train_begin(self, trainer: "Trainer") -> None:
        self._start_time = datetime.now(timezone.utc).isoformat()
        if self.total_steps is None:
            self.total_steps = getattr(trainer, "_target_total_steps", None)
        self._write_status(trainer, step=0)

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        if step % self.interval == 0:
            self._write_status(trainer, step)
        if self.stop_path.exists():
            trainer.stop_training = True

    def on_train_end(self, trainer: "Trainer") -> None:
        status = "stopped" if trainer.stop_training else "completed"
        self._write_status(trainer, trainer.global_step, status=status)

    def _write_status(
        self,
        trainer: "Trainer",
        step: int,
        *,
        status: str = "running",
    ) -> None:
        epsilon = getattr(trainer.agent, "epsilon", None)
        data: Dict[str, Any] = {
            "status": status,
            "step": step,
            "total_steps": self.total_steps,
            "episode": trainer.episode_index,
            "start_time": self._start_time,
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        }
        if epsilon is not None:
            data["epsilon"] = round(epsilon, 6)
        self._atomic_write(data)

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        """Write JSON atomically via tempfile + rename."""
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self.run_dir, prefix=".status_", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, self.status_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except Exception:
            # fallback: direct write (less safe but better than nothing)
            self.status_path.write_text(json.dumps(data, indent=2))

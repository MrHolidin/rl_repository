"""Eval callback: run evaluation at a fixed step interval."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Callable, Dict, List

from src.training.trainer import TrainerCallback, Transition

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class EvalCallback(TrainerCallback):
    """Run evaluation at a fixed step interval."""

    def __init__(
        self,
        eval_fn: Callable[["Trainer"], Dict[str, float]],
        interval: int,
        name: str = "eval",
    ):
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
        metrics: dict,
    ) -> None:
        if step % self.interval == 0:
            start = perf_counter()
            result = self.eval_fn(trainer)
            self.time_spent += perf_counter() - start
            self.history.append(result)
            metrics.update({f"{self.name}/{k}": v for k, v in result.items()})
            summary = " | ".join(
                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in result.items()
            )
            print(f"[{self.name}] step {step}: {summary}")

"""Performance tracking callback for detailed timing analysis."""

from __future__ import annotations

import functools
from collections import defaultdict
from time import perf_counter
from typing import Any, Callable, Dict, Optional

from src.training.trainer import Trainer, TrainerCallback, Transition


class PerformanceTrackerCallback(TrainerCallback):
    """Detailed performance tracker that monitors timing of all key operations."""

    def __init__(self, report_interval: int = 1000, print_report: bool = True):
        self.report_interval = report_interval
        self.print_report = print_report
        self._timings: Dict[str, list[float]] = defaultdict(list)
        self._counts: Dict[str, int] = defaultdict(int)
        self._original_methods: Dict[Any, Dict[str, Callable]] = {}
        self._wrapped_objects: Dict[int, Any] = {}
        self._trainer: Optional[Trainer] = None
        self._step_count = 0

    def on_train_begin(self, trainer: Trainer) -> None:
        self._trainer = trainer
        self._wrap_agent_methods(trainer.agent)
        self._wrap_env_methods(trainer.env)
        self._wrap_opponent_preparation(trainer)

    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        transition: Transition,
        metrics: Dict[str, float],
    ) -> None:
        self._step_count = step
        if self.report_interval > 0 and step % self.report_interval == 0:
            self._print_interim_report(step)

    def on_train_end(self, trainer: Trainer) -> None:
        self._restore_methods()
        if self.print_report:
            self._print_final_report()

    def _wrap_agent_methods(self, agent: Any, prefix: str = "agent") -> None:
        agent_id = id(agent)
        if agent_id not in self._original_methods:
            self._original_methods[agent_id] = {}
            self._wrapped_objects[agent_id] = agent

        if hasattr(agent, "act") and "act" not in self._original_methods[agent_id]:
            original_act = agent.act
            self._original_methods[agent_id]["act"] = original_act

            @functools.wraps(original_act)
            def tracked_act(*args, **kwargs):
                start = perf_counter()
                result = original_act(*args, **kwargs)
                self._timings[f"{prefix}.act"].append(perf_counter() - start)
                self._counts[f"{prefix}.act"] += 1
                return result

            agent.act = tracked_act

        if hasattr(agent, "observe") and "observe" not in self._original_methods[agent_id]:
            original_observe = agent.observe
            self._original_methods[agent_id]["observe"] = original_observe

            @functools.wraps(original_observe)
            def tracked_observe(*args, **kwargs):
                start = perf_counter()
                result = original_observe(*args, **kwargs)
                self._timings[f"{prefix}.observe"].append(perf_counter() - start)
                self._counts[f"{prefix}.observe"] += 1
                return result

            agent.observe = tracked_observe

        if hasattr(agent, "update") and "update" not in self._original_methods[agent_id]:
            original_update = agent.update
            self._original_methods[agent_id]["update"] = original_update

            @functools.wraps(original_update)
            def tracked_update(*args, **kwargs):
                start = perf_counter()
                result = original_update(*args, **kwargs)
                self._timings[f"{prefix}.update"].append(perf_counter() - start)
                self._counts[f"{prefix}.update"] += 1
                return result

            agent.update = tracked_update

    def _wrap_env_methods(self, env: Any) -> None:
        env_id = id(env)
        if env_id not in self._original_methods:
            self._original_methods[env_id] = {}
            self._wrapped_objects[env_id] = env

        if hasattr(env, "step") and "step" not in self._original_methods[env_id]:
            original_step = env.step
            self._original_methods[env_id]["step"] = original_step

            @functools.wraps(original_step)
            def tracked_step(*args, **kwargs):
                start = perf_counter()
                result = original_step(*args, **kwargs)
                self._timings["env.step"].append(perf_counter() - start)
                self._counts["env.step"] += 1
                return result

            env.step = tracked_step

        if hasattr(env, "reset") and "reset" not in self._original_methods[env_id]:
            original_reset = env.reset
            self._original_methods[env_id]["reset"] = original_reset

            @functools.wraps(original_reset)
            def tracked_reset(*args, **kwargs):
                start = perf_counter()
                result = original_reset(*args, **kwargs)
                self._timings["env.reset"].append(perf_counter() - start)
                self._counts["env.reset"] += 1
                return result

            env.reset = tracked_reset

    def _wrap_opponent_preparation(self, trainer: Trainer) -> None:
        if not hasattr(trainer, "_prepare_opponent"):
            return
        original_prepare = trainer._prepare_opponent
        tid = id(trainer)
        if tid in self._original_methods and "_prepare_opponent" in self._original_methods[tid]:
            return
        if tid not in self._original_methods:
            self._original_methods[tid] = {}
            self._wrapped_objects[tid] = trainer
        self._original_methods[tid]["_prepare_opponent"] = original_prepare

        @functools.wraps(original_prepare)
        def tracked_prepare_opponent():
            opponent = original_prepare()
            self._wrap_agent_methods(opponent, prefix="opponent")
            return opponent

        trainer._prepare_opponent = tracked_prepare_opponent

    def _restore_methods(self) -> None:
        for obj_id, methods in list(self._original_methods.items()):
            obj = self._wrapped_objects.get(obj_id)
            if obj is None:
                continue
            for method_name, original_method in methods.items():
                try:
                    setattr(obj, method_name, original_method)
                except (AttributeError, TypeError):
                    pass
        self._original_methods.clear()
        self._wrapped_objects.clear()

    def _print_interim_report(self, step: int) -> None:
        print(f"\n[Performance @ step {step:,}]")
        self._print_stats()

    def _print_final_report(self) -> None:
        print("\n" + "=" * 80)
        print("PERFORMANCE TRACKER - FINAL REPORT")
        print("=" * 80)
        self._print_stats()
        print("=" * 80)

    def _print_stats(self) -> None:
        if not self._timings:
            print("No timing data collected.")
            return
        stats = []
        total_time = 0.0
        for key in sorted(self._timings.keys()):
            times = self._timings[key]
            count = self._counts[key]
            if not times:
                continue
            total = sum(times)
            total_time += total
            avg = total / count if count > 0 else 0.0
            min_t = min(times)
            max_t = max(times)
            p50 = sorted(times)[len(times) // 2]
            p95 = sorted(times)[min(int(len(times) * 0.95), len(times) - 1)]
            p99 = sorted(times)[min(int(len(times) * 0.99), len(times) - 1)]
            stats.append({
                "operation": key,
                "count": count,
                "total": total,
                "avg": avg,
                "min": min_t,
                "max": max_t,
                "p50": p50,
                "p95": p95,
                "p99": p99,
            })
        stats.sort(key=lambda x: x["total"], reverse=True)
        print(f"\n{'Operation':<20} {'Count':>10} {'Total (s)':>12} {'Avg (ms)':>12} "
              f"{'Min (ms)':>12} {'Max (ms)':>12} {'P50 (ms)':>12} {'P95 (ms)':>12} {'P99 (ms)':>12}")
        print("-" * 120)
        for stat in stats:
            pct = (stat["total"] / total_time * 100) if total_time > 0 else 0.0
            print(f"{stat['operation']:<20} {stat['count']:>10,} {stat['total']:>12.2f} ({pct:>5.1f}%) "
                  f"{stat['avg']*1000:>12.3f} {stat['min']*1000:>12.3f} {stat['max']*1000:>12.3f} "
                  f"{stat['p50']*1000:>12.3f} {stat['p95']*1000:>12.3f} {stat['p99']*1000:>12.3f}")
        print("-" * 120)
        print(f"{'TOTAL':<20} {'':>10} {total_time:>12.2f} ({100.0:>5.1f}%)")
        print()
        if stats:
            t0 = stats[0]
            print(f"Top bottleneck: {t0['operation']} ({t0['total']:.2f}s, {t0['total']/total_time*100:.1f}%)")
            if len(stats) > 1:
                t1 = stats[1]
                print(f"Second: {t1['operation']} ({t1['total']:.2f}s, {t1['total']/total_time*100:.1f}%)")

    def get_stats(self) -> Dict[str, Any]:
        out = {}
        for key in sorted(self._timings.keys()):
            times = self._timings[key]
            count = self._counts[key]
            if not times:
                continue
            total = sum(times)
            avg = total / count if count > 0 else 0.0
            out[key] = {
                "count": count,
                "total": total,
                "avg": avg,
                "min": min(times),
                "max": max(times),
                "p50": sorted(times)[len(times) // 2],
                "p95": sorted(times)[min(int(len(times) * 0.95), len(times) - 1)],
                "p99": sorted(times)[min(int(len(times) * 0.99), len(times) - 1)],
            }
        return out

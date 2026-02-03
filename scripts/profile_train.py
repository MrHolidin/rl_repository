"""Profile training (CPU + GPU) for ~10 sec, then report steps and save trace."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

MAX_SEC = 8


def main() -> None:
    print("Starting profiler...", flush=True)
    config_path = Path("configs/connect4/self_play_distributional.yaml")
    run_dir = Path("runs/profile_1k")
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = config_path.read_text()
    data = yaml.safe_load(raw)
    data["train"] = data.get("train", {})
    data["train"]["total_steps"] = 50_000

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml.dump(data, default_flow_style=False, allow_unicode=True))
        tmp_config = f.name

    try:
        from src.training import run as run_module

        def stop_after() -> None:
            time.sleep(MAX_SEC)
            if run_module._current_trainer is not None:
                run_module._current_trainer.stop_training = True

        t = threading.Thread(target=stop_after, daemon=True)
        t.start()

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            print("GPU profiling enabled", flush=True)
        else:
            print("WARNING: CUDA not available, CPU only", flush=True)

        start = time.perf_counter()
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            run_module.run(config_path=tmp_config, run_dir=str(run_dir))
        elapsed = time.perf_counter() - start

        step = 0
        status_path = run_dir / "status.json"
        if status_path.exists():
            try:
                step = json.loads(status_path.read_text()).get("step", 0)
            except Exception:
                pass

        print(f"Training done: {step} steps in {elapsed:.1f}s", flush=True)
        print("Generating profiler report...", flush=True)
        out = []
        out.append(f"=== RESULTS ===")
        out.append(f"Steps: {step}, Elapsed: {elapsed:.1f}s, Steps/sec: {step/elapsed:.1f}")
        out.append(f"\n=== TOP 30 by CUDA time ===")
        out.append(prof.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
            row_limit=20,
        ))
        out.append(f"\n=== TOP 20 by CPU time ===")
        out.append(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

        report = "\n".join(out)
        report_path = run_dir / "profile_report.txt"
        report_path.write_text(report)
        print(f"Report saved to {report_path}")
    finally:
        Path(tmp_config).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

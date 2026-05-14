"""Profile training with cProfile only (no torch.profiler — avoids huge RAM from event buffers)."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training under cProfile and write text + .prof stats.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/minibg/ppo_structured.yaml")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/profile_cprofile")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--status-interval", type=int, default=500)
    parser.add_argument("--top", type=int, default=60, help="Lines in cProfile report (by cumtime).")
    parser.add_argument(
        "--strip-callbacks",
        action="store_true",
        help="Replace train.callbacks with a single high-interval metrics_file (lighter I/O).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(args.config.read_text())
    raw["train"] = dict(raw.get("train", {}))
    raw["train"]["total_steps"] = int(args.total_steps)
    if args.strip_callbacks:
        raw["train"]["callbacks"] = [{"type": "metrics_file", "params": {"interval": 1_000_000}}]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(run_dir)
    ) as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True)
        tmp_config = Path(f.name)

    from src.training import run as run_module

    prof_path = run_dir / "cprofile.prof"
    pr = cProfile.Profile()

    try:
        print(f"cProfile: config={args.config} total_steps={args.total_steps} run_dir={run_dir}", flush=True)
        t0 = time.perf_counter()
        pr.enable()
        run_module.run(
            config_path=tmp_config,
            run_dir=str(run_dir),
            status_interval=int(args.status_interval),
        )
        pr.disable()
        elapsed = time.perf_counter() - t0
    finally:
        tmp_config.unlink(missing_ok=True)

    pr.dump_stats(str(prof_path))

    step = 0
    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            step = int(json.loads(status_path.read_text()).get("step", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    s = pstats.Stats(str(prof_path))
    s.sort_stats("cumtime")
    buf = io.StringIO()
    s.stream = buf
    s.print_stats(int(args.top))

    lines = [
        f"=== cProfile only (no torch.profiler) ===",
        f"Steps: {step}  Elapsed: {elapsed:.1f}s  Steps/sec: {step / elapsed:.2f}" if elapsed > 0 else f"Steps: {step}",
        f"Raw stats: {prof_path}",
        "",
        f"=== Top {args.top} by cumtime ===",
        buf.getvalue(),
    ]
    report = "\n".join(lines)
    report_path = run_dir / "profile_report.txt"
    report_path.write_text(report)
    print(report)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()

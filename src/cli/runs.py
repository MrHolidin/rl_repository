"""CLI for managing training runs: list running, stop gracefully."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _process_alive(pid: int) -> bool:
    """Check if a process with given PID is running."""
    try:
        os.kill(pid, 0)  # signal 0 = check existence
        return True
    except (OSError, ProcessLookupError):
        return False


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # Handle various ISO formats
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def list_runs(runs_root: Path, all_runs: bool = False) -> None:
    """List training runs, optionally showing all (including completed/stopped)."""
    runs_root = Path(runs_root)
    if not runs_root.exists():
        print(f"Runs directory not found: {runs_root}")
        return

    now = datetime.now(timezone.utc)
    found = False

    for status_file in sorted(runs_root.glob("*/status.json")):
        run_dir = status_file.parent
        try:
            data = json.loads(status_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        status = data.get("status", "unknown")
        step = data.get("step", 0)
        total_steps = data.get("total_steps")
        episode = data.get("episode", 0)
        epsilon = data.get("epsilon")
        start_time = _parse_iso(data.get("start_time"))
        heartbeat = _parse_iso(data.get("last_heartbeat"))

        # Check if process is alive
        pid_file = run_dir / "pid"
        pid = None
        alive = False
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                alive = _process_alive(pid)
            except (ValueError, OSError):
                pass

        # Determine effective status
        if status == "running" and not alive:
            status = "dead"  # process crashed without updating status
        elif status == "running" and heartbeat:
            age = (now - heartbeat).total_seconds()
            if age > 120:  # 2 min stale heartbeat
                status = "stale"

        # Filter
        if not all_runs and status not in ("running",):
            continue

        found = True

        # Format progress
        if total_steps:
            progress = f"{step}/{total_steps} ({100*step/total_steps:.1f}%)"
        else:
            progress = str(step)

        # Format duration
        duration = ""
        if start_time:
            delta = (now - start_time).total_seconds()
            duration = _format_duration(delta)

        # Print
        line = f"  {run_dir.name:<40} {status:<10} step={progress:<16} ep={episode:<6}"
        if epsilon is not None:
            line += f" ε={epsilon:.4f}"
        if duration:
            line += f"  [{duration}]"
        if pid:
            line += f"  (pid={pid})"
        print(line)

    if not found:
        if all_runs:
            print("No runs found.")
        else:
            print("No running pipelines. Use --all to show completed/stopped runs.")


def stop_run(run_dir: Path) -> None:
    """Send stop signal to a training run."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    pid_file = run_dir / "pid"
    stop_file = run_dir / "stop"

    # Try SIGTERM first (instant)
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if _process_alive(pid):
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to PID {pid} ({run_dir.name})")
                return
        except (ValueError, OSError, ProcessLookupError) as e:
            print(f"Could not send signal: {e}")

    # Fallback: create stop file
    try:
        stop_file.touch()
        print(f"Created stop file: {stop_file}")
        print("Training will stop at next status check interval.")
    except OSError as e:
        print(f"Failed to create stop file: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli.runs list              # List running pipelines
  python -m src.cli.runs list --all        # List all runs (including completed)
  python -m src.cli.runs stop runs/my_run  # Stop a running pipeline
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_parser = subparsers.add_parser("list", help="List training runs")
    list_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory for runs (default: runs/)",
    )
    list_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show all runs, not just running",
    )

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop a training run")
    stop_parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to run directory to stop",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_runs(args.runs_dir, all_runs=args.all)
    elif args.command == "stop":
        stop_run(args.run_dir)


if __name__ == "__main__":
    main()

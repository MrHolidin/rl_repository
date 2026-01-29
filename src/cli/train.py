"""Training CLI: single entrypoint with --config and --run_dir."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.training.run import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training from a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Output run directory (default: runs/<config_stem>_<timestamp>).")
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
    else:
        stem = config_path.stem
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path("runs") / f"{stem}_{ts}"

    run(config_path=config_path, run_dir=run_dir)


if __name__ == "__main__":
    main()

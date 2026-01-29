"""Run metadata collection and persistence."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else False
    except Exception:
        return False


def _torch_version() -> Optional[str]:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None


def collect_meta(
    *,
    config_path: Path,
    command: str,
    seed: Optional[int],
    device: Optional[str],
) -> Dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "command": command,
        "start_time_iso": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "device": device,
        "python_version": sys.version.split()[0],
        "torch_version": _torch_version(),
        "config_path": str(config_path.resolve()),
    }


def write_meta_json(path: Path, meta: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(meta, f, indent=2)

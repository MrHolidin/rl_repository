#!/usr/bin/env python3
"""Check patch package coverage: catalog pool vs bindings.py EFFECTS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bg_catalog.patch_coverage import analyze_patch_coverage, format_report
from src.bg_catalog.patch_context import DEFAULT_PATCH_DIR


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate data/bgcore/<patch>/ catalog + bindings coverage."
    )
    p.add_argument(
        "patch_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_PATCH_DIR,
        help=f"patch package directory (default: {DEFAULT_PATCH_DIR})",
    )
    p.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="exit 1 when warnings are present (default: only errors)",
    )
    args = p.parse_args()

    report = analyze_patch_coverage(args.patch_dir)
    print(format_report(report))

    if not report.ok:
        raise SystemExit(1)
    if args.fail_on_warning and report.warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

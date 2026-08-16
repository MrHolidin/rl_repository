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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate data/bgcore/<patch>/ catalog + bindings coverage."
    )
    # No default: the package to check is always named explicitly, the same way
    # a PatchContext is never implicit (there is no "current" patch anymore).
    p.add_argument(
        "patch_dir",
        type=Path,
        help="patch package directory, e.g. data/bgcore/19_6_0_74257",
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

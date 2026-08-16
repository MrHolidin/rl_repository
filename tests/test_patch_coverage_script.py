"""Tests for patch coverage analysis."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.bg_catalog.patch_coverage import analyze_patch_coverage, format_report

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGES_DIR = _REPO_ROOT / "data" / "bgcore"
_PATCH_36393 = _PACKAGES_DIR / "15_6_2_36393"
_SCRIPT = _REPO_ROOT / "scripts" / "check_patch_coverage.py"


def _packages():
    return sorted(p for p in _PACKAGES_DIR.iterdir() if (p / "meta.json").is_file())


def test_36393_patch_has_no_errors():
    report = analyze_patch_coverage(_PATCH_36393)
    assert report.ok
    assert not report.errors
    assert report.pool_count == 81


def test_36393_patch_battlecries_bound():
    report = analyze_patch_coverage(_PATCH_36393)
    assert not any(i.code == "battlecry_unbound" for i in report.warnings)


def test_format_report_includes_header():
    report = analyze_patch_coverage(_PATCH_36393)
    text = format_report(report)
    assert "build 36393" in text
    assert "pool  81" in text


@pytest.mark.parametrize("patch_dir", _packages(), ids=lambda p: p.name)
def test_every_shipped_package_is_error_free(patch_dir):
    """Adding a package means adding a *passing* one — for every build we ship."""
    report = analyze_patch_coverage(patch_dir)
    assert report.ok, format_report(report)


@pytest.mark.parametrize("patch_dir", _packages(), ids=lambda p: p.name)
def test_coverage_cli_runs(patch_dir):
    """The CLI entry point, not just the library behind it.

    ``check_patch_coverage.py`` is the documented gate for adding a patch
    package, and it silently rotted once already (it imported a constant that
    had been removed from ``patch_context``) because only the library was
    covered here.
    """
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), str(patch_dir)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"build {analyze_patch_coverage(patch_dir).build}" in proc.stdout

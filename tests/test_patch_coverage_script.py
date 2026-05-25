"""Tests for patch coverage analysis."""

from pathlib import Path

from src.bg_catalog.patch_coverage import analyze_patch_coverage, format_report

_PATCH_36393 = Path(__file__).resolve().parents[1] / "data" / "bgcore" / "15_6_2_36393"


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

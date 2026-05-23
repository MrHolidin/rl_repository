"""Tests for patch coverage analysis."""

from src.bg_catalog.patch_coverage import analyze_patch_coverage, format_report
from src.bg_catalog.patch_context import DEFAULT_PATCH_DIR


def test_default_patch_has_no_errors():
    report = analyze_patch_coverage(DEFAULT_PATCH_DIR)
    assert report.ok
    assert not report.errors
    assert report.pool_count == 81


def test_default_patch_battlecries_bound():
    report = analyze_patch_coverage(DEFAULT_PATCH_DIR)
    assert not any(i.code == "battlecry_unbound" for i in report.warnings)


def test_format_report_includes_header():
    report = analyze_patch_coverage(DEFAULT_PATCH_DIR)
    text = format_report(report)
    assert "build 36393" in text
    assert "pool  81" in text

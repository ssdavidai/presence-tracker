"""
Tests for analysis/weekly_dashboard.py — Weekly HTML Presence Dashboard

Coverage:
  1. _week_dates() — generates correct 7-day window
  2. _week_stats() — aggregates metrics from day records correctly
  3. SVG helpers — produce non-empty SVG markup
     - _svg_dps_sparkline()
     - _svg_metric_grid()
     - _svg_recovery_chart()
     - _svg_meeting_bars()
  4. _source_coverage_table() — HTML table with correct structure
  5. generate_weekly_dashboard() — end-to-end:
     - Creates HTML file in the expected location
     - File is non-empty and contains key sections
     - Works when no days have data (all missing)
     - Works with partial week data (some days missing)
     - Works with full week data
     - Output path override respected
     - Summary stats are correct (best/worst day, averages)

Run with: python3 -m pytest tests/test_weekly_dashboard.py -v
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.weekly_dashboard import (
    _week_dates,
    _week_stats,
    _svg_dps_sparkline,
    _svg_metric_grid,
    _svg_recovery_chart,
    _svg_meeting_bars,
    _source_coverage_table,
    generate_weekly_dashboard,
    _dps_tier,
    _dps_colour,
    _cls_colour,
    _fdi_colour,
    _ras_colour,
    _recovery_colour,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

def _make_day(date_str: str, dps: float = 75.0, cls: float = 0.35, fdi: float = 0.70,
              ras: float = 0.80, recovery: float = 72.0, meeting_mins: int = 90,
              missing: bool = False) -> dict:
    """Build a minimal day record matching the rolling summary schema."""
    if missing:
        return {"date": date_str, "missing": True}
    return {
        "date": date_str,
        "missing": False,
        "presence_score": {"dps": dps},
        "metrics_avg": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": 0.20,
            "context_switch_cost": 0.15,
            "recovery_alignment_score": ras,
        },
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": 68.0,
            "resting_heart_rate": 57.0,
            "sleep_hours": 7.2,
            "sleep_performance": 85.0,
        },
        "calendar": {
            "total_meeting_minutes": meeting_mins,
            "event_count": 2,
        },
        "source_coverage": {
            "whoop": 10,
            "calendar": 10,
            "slack": 10,
            "rescuetime": 8,
            "omi": 0,
        },
    }


def _make_week(end_date_str: str, missing_indices: list[int] = None) -> list[dict]:
    """Build a 7-day list of day records."""
    dates = _week_dates(end_date_str)
    missing_indices = missing_indices or []
    return [
        _make_day(
            d,
            dps=70.0 + i * 3,
            cls=0.30 + i * 0.02,
            fdi=0.80 - i * 0.02,
            ras=0.85 - i * 0.02,
            recovery=75.0 + i * 1.5,
            meeting_mins=60 + i * 15,
            missing=(i in missing_indices),
        )
        for i, d in enumerate(dates)
    ]


# ─── Tests: _week_dates ────────────────────────────────────────────────────────

class TestWeekDates:
    def test_returns_7_dates(self):
        dates = _week_dates("2026-03-14")
        assert len(dates) == 7

    def test_last_date_is_end_date(self):
        dates = _week_dates("2026-03-14")
        assert dates[-1] == "2026-03-14"

    def test_first_date_is_6_days_before(self):
        dates = _week_dates("2026-03-14")
        assert dates[0] == "2026-03-08"

    def test_dates_are_consecutive(self):
        dates = _week_dates("2026-03-14")
        for i in range(1, len(dates)):
            prev = datetime.strptime(dates[i - 1], "%Y-%m-%d")
            curr = datetime.strptime(dates[i], "%Y-%m-%d")
            assert (curr - prev).days == 1

    def test_dates_in_ascending_order(self):
        dates = _week_dates("2026-03-14")
        assert dates == sorted(dates)


# ─── Tests: _week_stats ────────────────────────────────────────────────────────

class TestWeekStats:
    def test_all_missing_returns_zero_days(self):
        days = [_make_day(f"2026-03-{8 + i:02d}", missing=True) for i in range(7)]
        stats = _week_stats(days)
        assert stats["days_with_data"] == 0
        assert stats["avg_dps"] is None
        assert stats["avg_recovery"] is None

    def test_counts_valid_days(self):
        days = _make_week("2026-03-14", missing_indices=[0, 1])
        stats = _week_stats(days)
        assert stats["days_with_data"] == 5

    def test_avg_dps_computed(self):
        days = _make_week("2026-03-14")
        stats = _week_stats(days)
        assert stats["avg_dps"] is not None
        assert 60 < stats["avg_dps"] < 100

    def test_avg_cls_computed(self):
        days = _make_week("2026-03-14")
        stats = _week_stats(days)
        assert stats["avg_cls"] is not None
        assert 0.0 < stats["avg_cls"] < 1.0

    def test_total_meeting_minutes_sum(self):
        days = _make_week("2026-03-14")
        stats = _week_stats(days)
        expected = sum(60 + i * 15 for i in range(7))
        assert stats["total_meeting_minutes"] == expected

    def test_best_dps_day_is_highest(self):
        days = _make_week("2026-03-14")
        stats = _week_stats(days)
        # DPS goes 70, 73, 76, 79, 82, 85, 88 → highest is last day
        assert stats["best_dps_day"] is not None
        date, score = stats["best_dps_day"]
        assert date == "2026-03-14"

    def test_worst_dps_day_is_lowest(self):
        days = _make_week("2026-03-14")
        stats = _week_stats(days)
        # DPS goes 70, 73, 76, 79, 82, 85, 88 → lowest is first day
        assert stats["worst_dps_day"] is not None
        date, score = stats["worst_dps_day"]
        assert date == "2026-03-08"

    def test_best_worst_none_when_no_dps(self):
        days = [_make_day(f"2026-03-{8 + i:02d}", missing=True) for i in range(7)]
        stats = _week_stats(days)
        assert stats["best_dps_day"] is None
        assert stats["worst_dps_day"] is None


# ─── Tests: colour helpers ─────────────────────────────────────────────────────

class TestColourHelpers:
    def test_cls_low_is_green(self):
        assert _cls_colour(0.10) == "#4ade80"

    def test_cls_mid_is_yellow(self):
        assert _cls_colour(0.35) == "#facc15"

    def test_cls_high_is_orange(self):
        assert _cls_colour(0.65) == "#fb923c"

    def test_cls_very_high_is_red(self):
        assert _cls_colour(0.90) == "#f87171"

    def test_fdi_high_is_green(self):
        assert _fdi_colour(0.90) == "#4ade80"

    def test_fdi_low_is_red(self):
        assert _fdi_colour(0.10) == "#f87171"

    def test_recovery_high_is_green(self):
        assert _recovery_colour(80.0) == "#4ade80"

    def test_recovery_low_is_red(self):
        assert _recovery_colour(20.0) == "#f87171"


class TestDpsTier:
    def test_exceptional_at_90(self):
        assert _dps_tier(90) == "Exceptional"

    def test_strong_at_85(self):
        assert _dps_tier(85) == "Strong"

    def test_good_at_75(self):
        assert _dps_tier(75) == "Good"

    def test_moderate_at_65(self):
        assert _dps_tier(65) == "Moderate"

    def test_challenging_at_50(self):
        assert _dps_tier(50) == "Challenging"

    def test_difficult_at_30(self):
        assert _dps_tier(30) == "Difficult"

    def test_none_returns_dash(self):
        assert _dps_tier(None) == "—"


# ─── Tests: SVG generators ─────────────────────────────────────────────────────

class TestSvgDpsSparkline:
    def test_returns_nonempty_svg(self):
        days = _make_week("2026-03-14")
        svg = _svg_dps_sparkline(days)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_contains_data_points(self):
        days = _make_week("2026-03-14")
        svg = _svg_dps_sparkline(days)
        assert "<circle" in svg  # data point dots

    def test_handles_all_missing(self):
        days = [_make_day(f"2026-03-{8 + i:02d}", missing=True) for i in range(7)]
        svg = _svg_dps_sparkline(days)
        assert "<svg" in svg  # still renders

    def test_handles_partial_data(self):
        days = _make_week("2026-03-14", missing_indices=[2, 4])
        svg = _svg_dps_sparkline(days)
        assert "<svg" in svg


class TestSvgMetricGrid:
    def test_returns_nonempty_svg(self):
        days = _make_week("2026-03-14")
        svg = _svg_metric_grid(days)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_contains_bars(self):
        days = _make_week("2026-03-14")
        svg = _svg_metric_grid(days)
        assert "<rect" in svg

    def test_handles_all_missing(self):
        days = [_make_day(f"2026-03-{8 + i:02d}", missing=True) for i in range(7)]
        svg = _svg_metric_grid(days)
        assert "<svg" in svg


class TestSvgRecoveryChart:
    def test_returns_nonempty_svg(self):
        days = _make_week("2026-03-14")
        svg = _svg_recovery_chart(days)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_contains_recovery_area(self):
        days = _make_week("2026-03-14")
        svg = _svg_recovery_chart(days)
        assert "<path" in svg

    def test_handles_all_missing(self):
        days = [_make_day(f"2026-03-{8 + i:02d}", missing=True) for i in range(7)]
        svg = _svg_recovery_chart(days)
        assert "<svg" in svg


class TestSvgMeetingBars:
    def test_returns_nonempty_svg(self):
        days = _make_week("2026-03-14")
        svg = _svg_meeting_bars(days)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_contains_bars_when_meetings_exist(self):
        days = _make_week("2026-03-14")
        svg = _svg_meeting_bars(days)
        assert "<rect" in svg

    def test_handles_zero_meetings(self):
        days = _make_week("2026-03-14")
        for d in days:
            if not d.get("missing"):
                d["calendar"]["total_meeting_minutes"] = 0
        svg = _svg_meeting_bars(days)
        assert "<svg" in svg

    def test_handles_all_missing(self):
        days = [_make_day(f"2026-03-{8 + i:02d}", missing=True) for i in range(7)]
        svg = _svg_meeting_bars(days)
        assert "<svg" in svg


# ─── Tests: coverage table ─────────────────────────────────────────────────────

class TestSourceCoverageTable:
    def test_returns_html_table(self):
        days = _make_week("2026-03-14")
        html = _source_coverage_table(days)
        assert "<table" in html
        assert "</table>" in html

    def test_contains_source_rows(self):
        days = _make_week("2026-03-14")
        html = _source_coverage_table(days)
        assert "whoop" in html
        assert "slack" in html
        assert "calendar" in html

    def test_handles_all_missing(self):
        days = [_make_day(f"2026-03-{8 + i:02d}", missing=True) for i in range(7)]
        html = _source_coverage_table(days)
        assert "<table" in html
        # All cells should be missing (—)
        assert "cov-miss" in html


# ─── Tests: generate_weekly_dashboard ────────────────────────────────────────

class TestGenerateWeeklyDashboard:
    def _summary_for_days(self, end_date_str: str, missing_indices: list[int] = None):
        """Build a mock rolling summary dict."""
        dates = _week_dates(end_date_str)
        missing_indices = missing_indices or []
        days_dict = {}
        for i, d in enumerate(dates):
            if i not in missing_indices:
                day_rec = _make_day(
                    d,
                    dps=70.0 + i * 3,
                    cls=0.30 + i * 0.02,
                    fdi=0.80 - i * 0.02,
                    ras=0.85 - i * 0.02,
                    recovery=75.0 + i * 1.5,
                    meeting_mins=60 + i * 15,
                )
                # Remove "date" and "missing" keys — they're added by _load_week
                days_dict[d] = {k: v for k, v in day_rec.items() if k not in ("date", "missing")}
        return {"days": days_dict}

    def test_creates_html_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "week-test.html"
            with patch("analysis.weekly_dashboard.read_summary") as mock_summary:
                mock_summary.return_value = self._summary_for_days("2026-03-14")
                result = generate_weekly_dashboard("2026-03-14", output_path=out_path)
            assert result == out_path
            assert out_path.exists()

    def test_html_file_is_nonempty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "week-test.html"
            with patch("analysis.weekly_dashboard.read_summary") as mock_summary:
                mock_summary.return_value = self._summary_for_days("2026-03-14")
                generate_weekly_dashboard("2026-03-14", output_path=out_path)
            assert out_path.stat().st_size > 1000

    def test_html_contains_key_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "week-test.html"
            with patch("analysis.weekly_dashboard.read_summary") as mock_summary:
                mock_summary.return_value = self._summary_for_days("2026-03-14")
                generate_weekly_dashboard("2026-03-14", output_path=out_path)
            content = out_path.read_text()
        assert "Weekly Presence Dashboard" in content
        assert "DPS" in content
        assert "Recovery" in content
        assert "Meeting Load" in content
        assert "Source Coverage" in content

    def test_default_output_path(self):
        """When no output_path given, defaults to data/dashboard/week-YYYY-MM-DD.html."""
        from config import DATA_DIR
        expected = DATA_DIR / "dashboard" / "week-2026-03-14.html"
        with patch("analysis.weekly_dashboard.read_summary") as mock_summary:
            mock_summary.return_value = self._summary_for_days("2026-03-14")
            result = generate_weekly_dashboard("2026-03-14")
        assert result == expected

    def test_all_missing_days_still_generates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "week-empty.html"
            with patch("analysis.weekly_dashboard.read_summary") as mock_summary:
                mock_summary.return_value = {"days": {}}
                result = generate_weekly_dashboard("2026-03-14", output_path=out_path)
            assert out_path.exists()
            content = out_path.read_text()
            assert "0/7 days with data" in content

    def test_partial_week_generates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "week-partial.html"
            with patch("analysis.weekly_dashboard.read_summary") as mock_summary:
                mock_summary.return_value = self._summary_for_days("2026-03-14", missing_indices=[0, 1, 2])
                result = generate_weekly_dashboard("2026-03-14", output_path=out_path)
            assert out_path.exists()
            content = out_path.read_text()
            assert "4/7 days with data" in content

    def test_date_range_in_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "week-test.html"
            with patch("analysis.weekly_dashboard.read_summary") as mock_summary:
                mock_summary.return_value = self._summary_for_days("2026-03-14")
                generate_weekly_dashboard("2026-03-14", output_path=out_path)
            content = out_path.read_text()
        # Should contain something like "Mar 8 – Mar 14, 2026"
        assert "Mar" in content
        assert "2026" in content

    def test_full_week_shows_best_worst_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "week-full.html"
            with patch("analysis.weekly_dashboard.read_summary") as mock_summary:
                mock_summary.return_value = self._summary_for_days("2026-03-14")
                generate_weekly_dashboard("2026-03-14", output_path=out_path)
            content = out_path.read_text()
        assert "Best day" in content
        assert "Lowest day" in content

    def test_output_is_valid_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "week-test.html"
            with patch("analysis.weekly_dashboard.read_summary") as mock_summary:
                mock_summary.return_value = self._summary_for_days("2026-03-14")
                generate_weekly_dashboard("2026-03-14", output_path=out_path)
            content = out_path.read_text()
        assert content.startswith("<!DOCTYPE html>")
        assert "</html>" in content
        # Basic tag balance
        assert content.count("<body>") == content.count("</body>")
        assert content.count("<head>") == content.count("</head>")

    def test_weekly_stats_hero_values_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "week-test.html"
            with patch("analysis.weekly_dashboard.read_summary") as mock_summary:
                mock_summary.return_value = self._summary_for_days("2026-03-14")
                generate_weekly_dashboard("2026-03-14", output_path=out_path)
            content = out_path.read_text()
        # Hero should show Avg DPS, Recovery, FDI, CLS, Meetings
        assert "Avg DPS" in content
        assert "Avg Recovery" in content
        assert "Avg FDI" in content
        assert "Total Meetings" in content

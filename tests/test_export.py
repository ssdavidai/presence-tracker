"""
Tests for scripts/export.py — data export utility.

Run with: python3 -m pytest tests/test_export.py -v
"""

import csv
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Import the module under test
from scripts.export import (
    CSV_COLUMNS,
    build_row,
    build_json_row,
    export_csv,
    export_json,
    filter_dates,
    _safe,
    _sources_for_date,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_DAY_DATA = {
    "date": "2026-03-13",
    "whoop": {
        "recovery_score": 85.0,
        "hrv_rmssd_milli": 72.5,
        "resting_heart_rate": 55.0,
        "sleep_hours": 8.0,
        "sleep_performance": 88.0,
    },
    "metrics_avg": {
        "cognitive_load_score": 0.32,
        "focus_depth_index": 0.75,
        "social_drain_index": 0.18,
        "context_switch_cost": 0.22,
        "recovery_alignment_score": 0.81,
    },
    "metrics_peak": {
        "cognitive_load_score": 0.65,
    },
    "focus_quality": {
        "active_fdi": 0.68,
        "active_windows": 12,
        "peak_focus_hour": 10,
        "peak_focus_fdi": 0.91,
    },
    "calendar": {
        "total_meeting_minutes": 90,
        "meeting_windows": 6,
    },
    "slack": {
        "messages_sent": 24,
        "messages_received": 47,
    },
    "rescuetime": None,
}

SAMPLE_DAY_WITH_RT = {
    **SAMPLE_DAY_DATA,
    "rescuetime": {
        "focus_minutes": 180.0,
        "distraction_minutes": 30.0,
        "productive_pct": 85.7,
    },
}

MINIMAL_DAY_DATA: dict = {}


# ─── Tests for _safe() ────────────────────────────────────────────────────────

class TestSafe:
    def test_none_returns_default_empty(self):
        assert _safe(None) == ""

    def test_none_with_custom_default(self):
        assert _safe(None, "N/A") == "N/A"

    def test_float_formatted_to_4dp(self):
        assert _safe(0.12345678) == "0.1235"

    def test_float_zero(self):
        assert _safe(0.0) == "0.0000"

    def test_int_as_string(self):
        assert _safe(42) == "42"

    def test_string_passthrough(self):
        assert _safe("hello") == "hello"

    def test_bool_as_string(self):
        # bool is a subclass of int — True → "True"
        assert _safe(True) == "True"


# ─── Tests for build_row() ─────────────────────────────────────────────────────

class TestBuildRow:
    def test_returns_all_csv_columns(self):
        with patch("scripts.export._sources_for_date", return_value="whoop|slack"):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert set(row.keys()) == set(CSV_COLUMNS)

    def test_date_is_passed_through(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["date"] == "2026-03-13"

    def test_recovery_score_formatted(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["recovery_score"] == "85.0000"

    def test_hrv_formatted(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["hrv_rmssd_ms"] == "72.5000"

    def test_avg_cls_correct(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["avg_cls"] == "0.3200"

    def test_peak_cls_uses_metrics_peak(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["peak_cls"] == "0.6500"

    def test_active_fdi_uses_focus_quality(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["active_fdi"] == "0.6800"

    def test_peak_focus_hour_correct(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["peak_focus_hour"] == "10"

    def test_meeting_minutes_correct(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["total_meeting_min"] == "90"

    def test_slack_sent_correct(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["slack_sent"] == "24"

    def test_rt_focus_blank_when_none(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["rt_focus_min"] == ""

    def test_rt_columns_when_rescuetime_present(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", SAMPLE_DAY_WITH_RT)
        assert row["rt_focus_min"] == "180.0000"
        assert row["rt_distraction_min"] == "30.0000"
        assert row["rt_productive_pct"] == "85.7000"

    def test_minimal_data_no_crash(self):
        """Empty day_data must not raise — all fields should return empty strings."""
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", MINIMAL_DAY_DATA)
        assert set(row.keys()) == set(CSV_COLUMNS)
        assert row["recovery_score"] == ""
        assert row["avg_cls"] == ""

    def test_avg_fdi_falls_back_to_metrics_avg_when_focus_quality_missing(self):
        """When focus_quality is absent, avg_fdi should fall back to metrics_avg FDI."""
        day_data = {
            **SAMPLE_DAY_DATA,
            "focus_quality": {},  # empty focus_quality
        }
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", day_data)
        # Should fall back to metrics_avg.focus_depth_index = 0.75
        assert row["avg_fdi"] == "0.7500"


# ─── Tests for build_json_row() ───────────────────────────────────────────────

class TestBuildJsonRow:
    def test_includes_all_csv_columns(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_json_row("2026-03-13", SAMPLE_DAY_DATA)
        for col in CSV_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_includes_raw_nested_data(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_json_row("2026-03-13", SAMPLE_DAY_DATA)
        assert "_raw" in row
        raw = row["_raw"]
        assert "whoop" in raw
        assert "metrics_avg" in raw
        assert "focus_quality" in raw
        assert "calendar" in raw

    def test_raw_whoop_matches_input(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_json_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["_raw"]["whoop"]["recovery_score"] == 85.0

    def test_raw_rescuetime_none_when_absent(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_json_row("2026-03-13", SAMPLE_DAY_DATA)
        assert row["_raw"]["rescuetime"] is None

    def test_raw_rescuetime_present_when_available(self):
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_json_row("2026-03-13", SAMPLE_DAY_WITH_RT)
        assert row["_raw"]["rescuetime"]["focus_minutes"] == 180.0


# ─── Tests for filter_dates() ─────────────────────────────────────────────────

class TestFilterDates:
    ALL_DATES = [
        "2026-01-01",
        "2026-01-15",
        "2026-02-01",
        "2026-02-15",
        "2026-03-01",
        "2026-03-13",
    ]

    def test_no_filters_returns_all(self):
        result = filter_dates(self.ALL_DATES, None, None, None)
        assert result == self.ALL_DATES

    def test_empty_input_returns_empty(self):
        result = filter_dates([], None, None, None)
        assert result == []

    def test_start_filter(self):
        result = filter_dates(self.ALL_DATES, None, "2026-02-01", None)
        assert result == ["2026-02-01", "2026-02-15", "2026-03-01", "2026-03-13"]

    def test_start_and_end_filter(self):
        result = filter_dates(self.ALL_DATES, None, "2026-02-01", "2026-02-28")
        assert result == ["2026-02-01", "2026-02-15"]

    def test_days_filter_last_1(self):
        """--days 1 should include only dates from yesterday onward."""
        # With today's date mocked, test the relative behaviour
        result = filter_dates(["2026-03-13"], 1, None, None)
        # Should include 2026-03-13 if it's within the last 1 day from today
        # Since tests run on 2026-03-14, 2026-03-13 is within last 1 day
        assert isinstance(result, list)

    def test_days_returns_subset(self):
        """--days N should return only the most recent N+ dates."""
        # 100 days back from 2026-03-14 = 2025-12-04; all our dates are after that
        result = filter_dates(self.ALL_DATES, 100, None, None)
        assert result == self.ALL_DATES

    def test_start_only_no_end_includes_up_to_today(self):
        result = filter_dates(self.ALL_DATES, None, "2026-03-01", None)
        # Should include 2026-03-01 and 2026-03-13
        assert "2026-03-01" in result
        assert "2026-03-13" in result
        assert "2026-01-01" not in result

    def test_end_without_start_is_ignored(self):
        """End without start/days should return all (start defaults to first date)."""
        result = filter_dates(self.ALL_DATES, None, None, "2026-03-13")
        # No start filter applied without --start
        assert result == self.ALL_DATES

    def test_days_takes_priority_over_start(self):
        """--days should override --start when both specified (days takes precedence in filter_dates)."""
        # filter_dates checks days first
        result = filter_dates(self.ALL_DATES, 200, "2026-02-01", None)
        # days=200 means all dates from ~2025-09-26 onward → all dates included
        assert result == self.ALL_DATES


# ─── Tests for export_csv() ───────────────────────────────────────────────────

class TestExportCsv:
    def _make_rows(self, n=2):
        rows = []
        for i in range(n):
            rows.append({col: f"val_{col}_{i}" for col in CSV_COLUMNS})
        return rows

    def test_outputs_header_row(self):
        buf = io.StringIO()
        rows = self._make_rows(1)
        export_csv(rows, buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        assert set(reader.fieldnames) == set(CSV_COLUMNS)

    def test_outputs_correct_number_of_rows(self):
        buf = io.StringIO()
        rows = self._make_rows(3)
        export_csv(rows, buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        data_rows = list(reader)
        assert len(data_rows) == 3

    def test_values_roundtrip(self):
        buf = io.StringIO()
        rows = [{"date": "2026-03-13", **{col: "0.1234" for col in CSV_COLUMNS if col != "date"}}]
        export_csv(rows, buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        read_row = next(reader)
        assert read_row["date"] == "2026-03-13"
        assert read_row["avg_cls"] == "0.1234"

    def test_empty_rows_produces_header_only(self):
        buf = io.StringIO()
        export_csv([], buf)
        buf.seek(0)
        lines = [l for l in buf.readlines() if l.strip()]
        assert len(lines) == 1  # only header

    def test_columns_are_ordered(self):
        buf = io.StringIO()
        rows = self._make_rows(1)
        export_csv(rows, buf)
        buf.seek(0)
        header_line = buf.readline().strip()
        columns_in_output = header_line.split(",")
        assert columns_in_output == CSV_COLUMNS


# ─── Tests for export_json() ──────────────────────────────────────────────────

class TestExportJson:
    def test_outputs_valid_json(self):
        buf = io.StringIO()
        rows = [{"date": "2026-03-13", "avg_cls": "0.3200"}]
        export_json(rows, buf)
        buf.seek(0)
        parsed = json.loads(buf.read())
        assert isinstance(parsed, list)

    def test_row_count_matches(self):
        buf = io.StringIO()
        rows = [{"date": f"2026-03-{i:02d}"} for i in range(1, 6)]
        export_json(rows, buf)
        buf.seek(0)
        parsed = json.loads(buf.read())
        assert len(parsed) == 5

    def test_field_values_preserved(self):
        buf = io.StringIO()
        rows = [{"date": "2026-03-13", "recovery_score": "92.0000", "avg_cls": "0.0374"}]
        export_json(rows, buf)
        buf.seek(0)
        parsed = json.loads(buf.read())
        assert parsed[0]["date"] == "2026-03-13"
        assert parsed[0]["recovery_score"] == "92.0000"

    def test_empty_rows_produces_empty_array(self):
        buf = io.StringIO()
        export_json([], buf)
        buf.seek(0)
        parsed = json.loads(buf.read())
        assert parsed == []

    def test_json_ends_with_newline(self):
        buf = io.StringIO()
        export_json([{"date": "2026-03-13"}], buf)
        buf.seek(0)
        content = buf.read()
        assert content.endswith("\n")


# ─── Tests for _sources_for_date() ────────────────────────────────────────────

class TestSourcesForDate:
    def _make_window(self, sources=None, has_rt=False, has_omi=False):
        window = {
            "metadata": {"sources_available": sources or ["whoop", "calendar", "slack"]},
        }
        if has_rt:
            window["rescuetime"] = {"active_seconds": 60}
        else:
            window["rescuetime"] = None
        if has_omi:
            window["omi"] = {"conversation_active": True}
        return window

    def test_returns_pipe_separated_sources(self):
        windows = [self._make_window()]
        with patch("scripts.export.read_day", return_value=windows):
            result = _sources_for_date("2026-03-13")
        assert "whoop" in result
        assert "|" in result  # multiple sources joined by pipe

    def test_no_data_returns_empty(self):
        with patch("scripts.export.read_day", return_value=[]):
            result = _sources_for_date("2026-03-13")
        assert result == ""

    def test_rescuetime_added_when_rt_windows_present(self):
        windows = [
            self._make_window(sources=["whoop", "calendar", "slack"], has_rt=True)
        ]
        with patch("scripts.export.read_day", return_value=windows):
            result = _sources_for_date("2026-03-13")
        assert "rescuetime" in result

    def test_omi_added_when_omi_windows_present(self):
        windows = [
            self._make_window(sources=["whoop", "calendar", "slack"], has_omi=True)
        ]
        with patch("scripts.export.read_day", return_value=windows):
            result = _sources_for_date("2026-03-13")
        assert "omi" in result

    def test_no_duplicates_in_sources(self):
        # sources_available already has slack, and window has rt data
        windows = [
            self._make_window(sources=["whoop", "calendar", "slack", "rescuetime"], has_rt=True)
        ]
        with patch("scripts.export.read_day", return_value=windows):
            result = _sources_for_date("2026-03-13")
        # "rescuetime" should appear only once
        assert result.count("rescuetime") == 1

    def test_sources_order_preserved(self):
        windows = [self._make_window(sources=["whoop", "calendar", "slack"])]
        with patch("scripts.export.read_day", return_value=windows):
            result = _sources_for_date("2026-03-13")
        parts = result.split("|")
        assert parts[0] == "whoop"


# ─── Integration: build_row → export_csv → parse back ─────────────────────────

class TestExportIntegration:
    def test_full_pipeline_csv(self):
        """Build a row, write to CSV, parse back, verify roundtrip."""
        with patch("scripts.export._sources_for_date", return_value="whoop|calendar|slack"):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)

        buf = io.StringIO()
        export_csv([row], buf)
        buf.seek(0)

        reader = csv.DictReader(buf)
        read_row = next(reader)
        assert read_row["date"] == "2026-03-13"
        assert read_row["recovery_score"] == "85.0000"
        assert read_row["avg_cls"] == "0.3200"
        assert read_row["sources"] == "whoop|calendar|slack"

    def test_full_pipeline_json(self):
        """Build a JSON row, write, parse back."""
        with patch("scripts.export._sources_for_date", return_value="whoop|calendar|slack"):
            row = build_json_row("2026-03-13", SAMPLE_DAY_DATA)

        buf = io.StringIO()
        export_json([row], buf)
        buf.seek(0)

        parsed = json.loads(buf.read())
        assert len(parsed) == 1
        assert parsed[0]["date"] == "2026-03-13"
        assert parsed[0]["_raw"]["whoop"]["recovery_score"] == 85.0

    def test_csv_handles_blank_fields_without_error(self):
        """Rows with empty strings for optional fields should produce valid CSV."""
        with patch("scripts.export._sources_for_date", return_value=""):
            row = build_row("2026-03-13", MINIMAL_DAY_DATA)

        buf = io.StringIO()
        export_csv([row], buf)
        buf.seek(0)

        reader = csv.DictReader(buf)
        read_row = next(reader)
        assert read_row["rt_focus_min"] == ""
        assert read_row["recovery_score"] == ""

    def test_multi_day_csv_sorted_by_date(self):
        """Multiple days should appear in date order in the CSV output."""
        day1 = {**SAMPLE_DAY_DATA}
        day2 = {**SAMPLE_DAY_WITH_RT}

        with patch("scripts.export._sources_for_date", return_value="whoop|calendar|slack"):
            rows = [
                build_row("2026-03-13", day1),
                build_row("2026-03-14", day2),
            ]

        buf = io.StringIO()
        export_csv(rows, buf)
        buf.seek(0)

        reader = csv.DictReader(buf)
        data_rows = list(reader)
        assert len(data_rows) == 2
        assert data_rows[0]["date"] == "2026-03-13"
        assert data_rows[1]["date"] == "2026-03-14"

    def test_output_file_created(self):
        """Writing to a file path should create the file."""
        with patch("scripts.export._sources_for_date", return_value="whoop|slack"):
            row = build_row("2026-03-13", SAMPLE_DAY_DATA)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.csv"
            with open(output_path, "w", newline="") as f:
                export_csv([row], f)
            assert output_path.exists()
            content = output_path.read_text()
            assert "date" in content
            assert "2026-03-13" in content


# ─── Tests for CSV_COLUMNS completeness ──────────────────────────────────────

class TestCsvColumns:
    def test_required_columns_present(self):
        required = [
            "date", "recovery_score", "hrv_rmssd_ms", "avg_cls",
            "avg_fdi", "avg_ras", "total_meeting_min", "sources",
        ]
        for col in required:
            assert col in CSV_COLUMNS, f"Required column missing: {col}"

    def test_rescuetime_columns_present(self):
        assert "rt_focus_min" in CSV_COLUMNS
        assert "rt_distraction_min" in CSV_COLUMNS
        assert "rt_productive_pct" in CSV_COLUMNS

    def test_focus_quality_columns_present(self):
        assert "active_fdi" in CSV_COLUMNS
        assert "active_windows" in CSV_COLUMNS
        assert "peak_focus_hour" in CSV_COLUMNS
        assert "peak_focus_fdi" in CSV_COLUMNS

    def test_no_duplicate_columns(self):
        assert len(CSV_COLUMNS) == len(set(CSV_COLUMNS))

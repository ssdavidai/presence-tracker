"""
Tests for scripts/status.py — System Status CLI

Coverage:
  - _gather_status() returns the correct structure
  - _health_code() returns correct codes for healthy/warning/error states
  - _bar() renders correct block-character bars
  - _trend_arrow() detects up/down/flat trends
  - print_brief() output format
  - print_json() returns valid JSON with expected keys
  - Formatters produce non-empty lines for each section
  - No external dependencies called during tests (all mocked)
"""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.status import (
    _bar,
    _colour_cls,
    _colour_fdi,
    _colour_recovery,
    _fmt_anomaly_section,
    _fmt_cognitive_debt_section,
    _fmt_dashboard_section,
    _fmt_data_section,
    _fmt_metrics_section,
    _fmt_ml_section,
    _fmt_sources_section,
    _fmt_whoop_section,
    _gather_status,
    _health_code,
    _trend_arrow,
    print_brief,
    print_json,
    print_status,
)


# ─── Test helpers ─────────────────────────────────────────────────────────────

def _make_status(
    n_days: int = 5,
    age_days: int = 0,
    staleness_days: int = 0,
    oldest: str = "2026-03-09",
    newest: str = "2026-03-13",
    avg_cls: float = 0.35,
    avg_fdi: float = 0.65,
    avg_ras: float = 0.75,
    recovery: float = 78.0,
    hrv: float = 65.0,
    anomaly_count: int = 0,
    ml_days: int = 5,
    ml_min: int = 60,
    ready_to_train: bool = False,
    dashboard_count: int = 1,
) -> dict:
    """Build a minimal status dict for testing formatters."""
    return {
        "data": {
            "n_days": n_days,
            "oldest": oldest,
            "newest": newest,
            "age_days": age_days,
            "staleness_days": staleness_days,
            "dates": [oldest, newest],
        },
        "metrics": {
            "recent_cls": [avg_cls - 0.05, avg_cls, avg_cls + 0.01],
            "recent_fdi": [avg_fdi + 0.05, avg_fdi, avg_fdi - 0.01],
            "recent_ras": [avg_ras, avg_ras, avg_ras],
            "avg_cls_7d": avg_cls,
            "avg_fdi_7d": avg_fdi,
            "avg_ras_7d": avg_ras,
        },
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "resting_heart_rate": 55.0,
            "sleep_performance": 82.0,
            "sleep_hours": 7.5,
            "strain": 10.2,
        },
        "sources": {
            "coverage": {
                "whoop": 5,
                "calendar": 5,
                "slack": 4,
            },
            "source_days": 5,
        },
        "ml": {
            "days_of_data": ml_days,
            "min_days_required": ml_min,
            "ready_to_train": ready_to_train,
            "days_remaining_until_ready": ml_min - ml_days,
            "models_trained": {
                "anomaly_detector": False,
                "recovery_predictor": False,
                "focus_clusters": False,
                "feature_scaler": False,
            },
            "last_trained": None,
            "training_days_used": None,
        },
        "dashboard": {
            "count": dashboard_count,
            "latest": "2026-03-13.html" if dashboard_count > 0 else None,
        },
        "anomalies": (
            {"date": newest, "triggered": ["cls_spike"], "count": anomaly_count}
            if anomaly_count > 0
            else {}
        ),
        "cognitive_debt": {},  # Empty by default → "not yet meaningful" display
    }


# ─── _bar() ──────────────────────────────────────────────────────────────────

class TestBar:
    def test_full_bar(self):
        b = _bar(1.0, width=10)
        assert b == "▓" * 10

    def test_empty_bar(self):
        b = _bar(0.0, width=10)
        assert b == "░" * 10

    def test_half_bar(self):
        b = _bar(0.5, width=10)
        assert b.count("▓") == 5
        assert b.count("░") == 5

    def test_length_is_correct(self):
        for w in [5, 10, 12, 16]:
            assert len(_bar(0.7, width=w)) == w

    def test_clamps_above_one(self):
        b = _bar(1.5, width=10)
        assert b == "▓" * 10
        assert len(b) == 10

    def test_zero_width_returns_empty(self):
        assert _bar(0.5, width=0) == ""


# ─── _trend_arrow() ──────────────────────────────────────────────────────────

class TestTrendArrow:
    def test_up_trend(self):
        assert _trend_arrow([0.3, 0.35, 0.40]) == "↑"

    def test_down_trend(self):
        assert _trend_arrow([0.7, 0.65, 0.60]) == "↓"

    def test_flat_trend(self):
        assert _trend_arrow([0.5, 0.50, 0.51]) == "→"

    def test_single_value_returns_dot(self):
        assert _trend_arrow([0.5]) == "·"

    def test_empty_list_returns_dot(self):
        assert _trend_arrow([]) == "·"

    def test_exactly_threshold_up(self):
        # delta of exactly 0.03 IS > 0.03 → counted as up in the implementation
        assert _trend_arrow([0.50, 0.53]) == "↑"

    def test_just_above_threshold_up(self):
        assert _trend_arrow([0.50, 0.541]) == "↑"

    def test_just_below_threshold_down(self):
        # delta of -0.031 IS < -0.03 → down
        assert _trend_arrow([0.55, 0.519]) == "↓"


# ─── _health_code() ──────────────────────────────────────────────────────────

class TestHealthCode:
    def test_healthy_when_fresh(self):
        s = _make_status(n_days=7, staleness_days=0)
        assert _health_code(s) == 0

    def test_healthy_when_stale_one_day(self):
        # staleness=1 means data is from yesterday — normal since ingestion runs at 23:45
        s = _make_status(n_days=7, staleness_days=1)
        assert _health_code(s) == 0

    def test_warning_when_anomaly_triggered(self):
        s = _make_status(n_days=7, staleness_days=0, anomaly_count=1)
        assert _health_code(s) == 1

    def test_error_when_no_data(self):
        s = _make_status(n_days=0, staleness_days=0)
        assert _health_code(s) == 2

    def test_error_when_very_stale(self):
        # n_days=10 (10 days of historical data) but staleness=5 (last ingest was 5 days ago)
        # This is the key regression case: previously age_days=10 was used which confused count
        # with staleness, causing false ERROR on systems with > 3 days of collected data.
        s = _make_status(n_days=10, staleness_days=5)
        assert _health_code(s) == 2

    def test_many_days_of_data_but_fresh_is_healthy(self):
        # This is the regression test: 6+ days of collected data, freshly ingested today.
        # The old bug would treat age_days=6 (count) as staleness and return ERROR.
        s = _make_status(n_days=6, age_days=6, staleness_days=0)
        assert _health_code(s) == 0

    def test_error_overrides_anomaly(self):
        # Even with anomaly, primary classification is error due to no data
        s = _make_status(n_days=0, staleness_days=0, anomaly_count=1)
        assert _health_code(s) == 2

    def test_exactly_two_days_stale_is_warning(self):
        s = _make_status(n_days=5, staleness_days=2)
        assert _health_code(s) == 1

    def test_exactly_three_days_stale_is_warning(self):
        s = _make_status(n_days=5, staleness_days=3)
        assert _health_code(s) == 1

    def test_four_days_stale_is_error(self):
        s = _make_status(n_days=5, staleness_days=4)
        assert _health_code(s) == 2


# ─── Colour helpers ───────────────────────────────────────────────────────────

class TestColourHelpers:
    """Verify colour codes contain the value and don't crash."""

    def test_cls_green_for_low(self):
        result = _colour_cls(0.20)
        assert "0.20" in result

    def test_cls_yellow_for_medium(self):
        result = _colour_cls(0.45)
        assert "0.45" in result

    def test_cls_red_for_high(self):
        result = _colour_cls(0.75)
        assert "0.75" in result

    def test_fdi_green_for_high(self):
        result = _colour_fdi(0.80)
        assert "0.80" in result

    def test_fdi_red_for_low(self):
        result = _colour_fdi(0.30)
        assert "0.30" in result

    def test_recovery_green_for_high(self):
        result = _colour_recovery(85.0)
        assert "85" in result

    def test_recovery_yellow_for_medium(self):
        result = _colour_recovery(50.0)
        assert "50" in result

    def test_recovery_red_for_low(self):
        result = _colour_recovery(20.0)
        assert "20" in result


# ─── Section formatters ───────────────────────────────────────────────────────

class TestFmtDataSection:
    def test_returns_non_empty_list(self):
        s = _make_status(n_days=5, age_days=0)
        lines = _fmt_data_section(s["data"])
        assert len(lines) > 0

    def test_contains_day_count(self):
        s = _make_status(n_days=12, age_days=0)
        lines = _fmt_data_section(s["data"])
        text = "\n".join(lines)
        assert "12" in text

    def test_no_data_shows_run_hint(self):
        s = _make_status(n_days=0, age_days=0)
        lines = _fmt_data_section(s["data"])
        text = "\n".join(lines)
        assert "run_daily" in text.lower() or "run" in text.lower()

    def test_ml_progress_bar_present(self):
        s = _make_status(n_days=5)
        lines = _fmt_data_section(s["data"])
        text = "\n".join(lines)
        # Should show progress like "5/60" or bar characters
        assert "▓" in text or "░" in text

    def test_freshness_shows_today_when_staleness_zero(self):
        """staleness_days=0 should display 'today', not a days-ago number."""
        s = _make_status(n_days=6, age_days=6, staleness_days=0)
        lines = _fmt_data_section(s["data"])
        text = "\n".join(lines)
        assert "today" in text.lower()

    def test_freshness_shows_yesterday_when_staleness_one(self):
        s = _make_status(n_days=6, staleness_days=1)
        lines = _fmt_data_section(s["data"])
        text = "\n".join(lines)
        assert "yesterday" in text.lower()

    def test_freshness_shows_days_ago_when_stale(self):
        s = _make_status(n_days=6, staleness_days=4)
        lines = _fmt_data_section(s["data"])
        text = "\n".join(lines)
        assert "4 days ago" in text

    def test_ml_progress_uses_collected_count_not_staleness(self):
        """ML progress should show 6/60, not 0/60, even when data is fresh (staleness=0)."""
        s = _make_status(n_days=6, age_days=6, staleness_days=0)
        lines = _fmt_data_section(s["data"])
        text = "\n".join(lines)
        assert "6/60" in text


class TestFmtMetricsSection:
    def test_returns_lines(self):
        s = _make_status()
        lines = _fmt_metrics_section(s["metrics"])
        assert len(lines) >= 2

    def test_contains_cls(self):
        s = _make_status(avg_cls=0.42)
        lines = _fmt_metrics_section(s["metrics"])
        text = "\n".join(lines)
        assert "CLS" in text
        assert "0.42" in text

    def test_contains_fdi(self):
        s = _make_status(avg_fdi=0.77)
        lines = _fmt_metrics_section(s["metrics"])
        text = "\n".join(lines)
        assert "FDI" in text
        assert "0.77" in text

    def test_none_metrics_shows_no_history(self):
        s = _make_status()
        s["metrics"]["avg_cls_7d"] = None
        lines = _fmt_metrics_section(s["metrics"])
        text = "\n".join(lines)
        assert "no" in text.lower() or "n/a" in text.lower() or "history" in text.lower()


class TestFmtWhoopSection:
    def test_returns_lines(self):
        s = _make_status(recovery=78.0, hrv=65.0)
        lines = _fmt_whoop_section(s["whoop"], "2026-03-13")
        assert len(lines) >= 2

    def test_contains_recovery_score(self):
        s = _make_status(recovery=85.0)
        lines = _fmt_whoop_section(s["whoop"], "2026-03-13")
        text = "\n".join(lines)
        assert "85" in text

    def test_contains_hrv(self):
        s = _make_status(hrv=72.0)
        lines = _fmt_whoop_section(s["whoop"], "2026-03-13")
        text = "\n".join(lines)
        assert "72" in text

    def test_no_whoop_data_shows_placeholder(self):
        lines = _fmt_whoop_section({}, "2026-03-13")
        text = "\n".join(lines)
        assert "no whoop" in text.lower() or "n/a" in text.lower()


class TestFmtSourcesSection:
    def test_returns_lines(self):
        s = _make_status()
        lines = _fmt_sources_section(s["sources"])
        assert len(lines) >= 2

    def test_shows_all_five_sources(self):
        s = _make_status()
        lines = _fmt_sources_section(s["sources"])
        text = "\n".join(lines)
        for src in ["whoop", "calendar", "slack", "rescuetime", "omi"]:
            assert src in text

    def test_full_coverage_shows_checkmark(self):
        s = _make_status()
        s["sources"]["coverage"] = {"whoop": 5, "calendar": 5, "slack": 5}
        s["sources"]["source_days"] = 5
        lines = _fmt_sources_section(s["sources"])
        text = "\n".join(lines)
        assert "✓" in text

    def test_empty_coverage_no_crash(self):
        lines = _fmt_sources_section({"coverage": {}, "source_days": 0})
        assert len(lines) >= 1


class TestFmtMlSection:
    def test_shows_days_remaining_when_not_ready(self):
        s = _make_status(ml_days=5, ml_min=60)
        lines = _fmt_ml_section(s["ml"])
        text = "\n".join(lines)
        assert "55" in text  # 60 - 5 = 55 remaining

    def test_shows_ready_when_enough_data(self):
        s = _make_status(ml_days=60, ml_min=60, ready_to_train=True)
        s["ml"]["ready_to_train"] = True
        s["ml"]["days_remaining_until_ready"] = 0
        lines = _fmt_ml_section(s["ml"])
        text = "\n".join(lines)
        assert "train" in text.lower() or "sufficient" in text.lower()

    def test_shows_trained_models_when_present(self):
        s = _make_status(ml_days=65, ready_to_train=True)
        s["ml"]["models_trained"] = {
            "anomaly_detector": True,
            "recovery_predictor": True,
            "focus_clusters": False,
            "feature_scaler": True,
        }
        s["ml"]["last_trained"] = "2026-03-10"
        lines = _fmt_ml_section(s["ml"])
        text = "\n".join(lines)
        assert "anomaly_detector" in text or "recovery_predictor" in text


class TestFmtAnomalySection:
    def test_no_anomalies_shows_checkmark(self):
        lines = _fmt_anomaly_section({})
        text = "\n".join(lines)
        assert "✓" in text or "no" in text.lower()

    def test_anomaly_triggered_shows_warning(self):
        anomaly = {"date": "2026-03-13", "triggered": ["cls_spike"], "count": 1}
        lines = _fmt_anomaly_section(anomaly)
        text = "\n".join(lines)
        assert "⚠" in text or "alert" in text.lower() or "spike" in text.lower()

    def test_multiple_anomalies_all_listed(self):
        anomaly = {
            "date": "2026-03-13",
            "triggered": ["cls_spike", "fdi_collapse"],
            "count": 2,
        }
        lines = _fmt_anomaly_section(anomaly)
        text = "\n".join(lines)
        assert "cls" in text.lower()
        assert "fdi" in text.lower()


class TestFmtCognitiveDebtSection:
    def test_empty_dict_shows_not_meaningful(self):
        lines = _fmt_cognitive_debt_section({})
        text = "\n".join(lines)
        assert "not yet" in text.lower() or "need" in text.lower()

    def test_meaningful_shows_cdi_score(self):
        cdi = {
            "cdi": 63.0,
            "tier": "loading",
            "days_in_deficit": 5,
            "days_used": 10,
            "trend_5d": 0.04,
            "line": "🟠 CDI 63/100 — Loading",
        }
        lines = _fmt_cognitive_debt_section(cdi)
        text = "\n".join(lines)
        assert "63" in text

    def test_shows_tier(self):
        cdi = {
            "cdi": 75.0,
            "tier": "fatigued",
            "days_in_deficit": 8,
            "days_used": 14,
            "trend_5d": 0.06,
            "line": "🔴 CDI 75/100 — Fatigued",
        }
        lines = _fmt_cognitive_debt_section(cdi)
        text = "\n".join(lines)
        assert "fatigued" in text.lower()

    def test_shows_deficit_days(self):
        cdi = {
            "cdi": 55.0,
            "tier": "loading",
            "days_in_deficit": 4,
            "days_used": 7,
            "trend_5d": None,
            "line": "🟠 CDI 55/100 — Loading",
        }
        lines = _fmt_cognitive_debt_section(cdi)
        text = "\n".join(lines)
        assert "4" in text

    def test_surplus_tier_output(self):
        cdi = {
            "cdi": 22.0,
            "tier": "surplus",
            "days_in_deficit": 1,
            "days_used": 7,
            "trend_5d": -0.05,
            "line": "🟢 CDI 22/100 — Surplus",
        }
        lines = _fmt_cognitive_debt_section(cdi)
        text = "\n".join(lines)
        assert "surplus" in text.lower()

    def test_header_present(self):
        lines = _fmt_cognitive_debt_section({})
        assert len(lines) >= 1
        assert "Cognitive Debt" in lines[0]


class TestFmtDashboardSection:
    def test_no_dashboards_shows_hint(self):
        lines = _fmt_dashboard_section({"count": 0, "latest": None})
        text = "\n".join(lines)
        assert "generat" in text.lower() or "no" in text.lower()

    def test_dashboards_present_shows_count(self):
        lines = _fmt_dashboard_section({"count": 5, "latest": "2026-03-13.html"})
        text = "\n".join(lines)
        assert "5" in text

    def test_latest_file_name_shown(self):
        lines = _fmt_dashboard_section({"count": 3, "latest": "2026-03-13.html"})
        text = "\n".join(lines)
        assert "2026-03-13.html" in text


# ─── CLI output functions ─────────────────────────────────────────────────────

class TestPrintBrief:
    """print_brief() should produce a machine-parseable one-liner."""

    @patch("scripts.status._gather_status")
    def test_output_format(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=0)
        code = print_brief()
        captured = capsys.readouterr()
        assert captured.out.startswith("[")
        assert "data=" in captured.out
        assert "ml=" in captured.out

    @patch("scripts.status._gather_status")
    def test_exit_code_healthy(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=0)
        code = print_brief()
        assert code == 0

    @patch("scripts.status._gather_status")
    def test_exit_code_warning(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=2, staleness_days=2)
        code = print_brief()
        assert code == 1

    @patch("scripts.status._gather_status")
    def test_exit_code_error_no_data(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=0, age_days=0)
        code = print_brief()
        assert code == 2

    @patch("scripts.status._gather_status")
    def test_output_label_ok(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=0)
        print_brief()
        captured = capsys.readouterr()
        assert "[OK]" in captured.out

    @patch("scripts.status._gather_status")
    def test_output_label_warn(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=2, staleness_days=2)
        print_brief()
        captured = capsys.readouterr()
        assert "[WARN]" in captured.out

    @patch("scripts.status._gather_status")
    def test_output_label_error(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=0, age_days=0)
        print_brief()
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out


class TestPrintJson:
    """print_json() should produce valid, structured JSON."""

    @patch("scripts.status._gather_status")
    def test_valid_json(self, mock_gather, capsys):
        mock_gather.return_value = _make_status()
        print_json()
        captured = capsys.readouterr()
        # Should not raise
        parsed = json.loads(captured.out)
        assert isinstance(parsed, dict)

    @patch("scripts.status._gather_status")
    def test_has_required_keys(self, mock_gather, capsys):
        mock_gather.return_value = _make_status()
        print_json()
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        for key in ["data", "metrics", "whoop", "sources", "ml", "dashboard", "anomalies", "health"]:
            assert key in parsed, f"Missing key: {key}"

    @patch("scripts.status._gather_status")
    def test_health_is_string(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=0)
        print_json()
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["health"] in ("healthy", "warning", "error")

    @patch("scripts.status._gather_status")
    def test_health_string_for_warning(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=2, staleness_days=2)
        print_json()
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["health"] == "warning"

    @patch("scripts.status._gather_status")
    def test_generated_at_is_present(self, mock_gather, capsys):
        mock_gather.return_value = _make_status()
        print_json()
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "generated_at" in parsed


class TestPrintStatus:
    """print_status() full output smoke tests."""

    @patch("scripts.status._gather_status")
    def test_produces_output(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=0)
        code = print_status()
        captured = capsys.readouterr()
        assert len(captured.out) > 100  # Should produce substantial output

    @patch("scripts.status._gather_status")
    def test_shows_all_section_headers(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=0)
        print_status()
        captured = capsys.readouterr()
        # All 8 section numbers should appear (③ = CDI added in v8.0)
        for i in range(1, 9):
            assert str(i) in captured.out, f"Section ① ... ⑧ missing: {i}"

    @patch("scripts.status._gather_status")
    def test_exit_code_matches_health(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=0)
        code = print_status()
        assert code == 0

    @patch("scripts.status._gather_status")
    def test_warning_exit_code(self, mock_gather, capsys):
        mock_gather.return_value = _make_status(n_days=5, age_days=2, staleness_days=2)
        code = print_status()
        assert code == 1


# ─── _gather_status() integration (mocked store) ─────────────────────────────

class TestGatherStatus:
    """
    Test _gather_status() with mocked store layer.
    Validates that the status dict has the expected structure.
    """

    @patch("scripts.status.get_data_status")
    @patch("scripts.status.get_recent_summaries")
    @patch("scripts.status.get_date_range")
    @patch("scripts.status.get_data_staleness_days")
    @patch("scripts.status.get_data_age_days")
    @patch("scripts.status.read_day")
    @patch("scripts.status.list_available_dates")
    def test_returns_expected_keys(
        self,
        mock_dates,
        mock_read_day,
        mock_age,
        mock_staleness,
        mock_range,
        mock_recent,
        mock_ml,
    ):
        mock_dates.return_value = ["2026-03-13"]
        mock_read_day.return_value = [
            {
                "whoop": {
                    "recovery_score": 80.0,
                    "hrv_rmssd_milli": 65.0,
                    "resting_heart_rate": 55.0,
                    "sleep_performance": 82.0,
                    "sleep_hours": 7.5,
                    "strain": 10.0,
                },
                "metadata": {"sources_available": ["whoop", "calendar", "slack"]},
            }
        ]
        mock_age.return_value = 1
        mock_staleness.return_value = 0
        mock_range.return_value = ("2026-03-13", "2026-03-13")
        mock_recent.return_value = [
            {
                "metrics_avg": {
                    "cognitive_load_score": 0.35,
                    "focus_depth_index": 0.70,
                    "recovery_alignment_score": 0.80,
                },
                "focus_quality": {"active_fdi": 0.72},
            }
        ]
        mock_ml.return_value = {
            "days_of_data": 1,
            "min_days_required": 60,
            "ready_to_train": False,
            "days_remaining_until_ready": 59,
            "models_trained": {},
            "last_trained": None,
            "training_days_used": None,
        }

        with patch("scripts.status.check_anomalies", return_value={"alerts": {}, "any_triggered": False}):
            result = _gather_status()

        assert "data" in result
        assert "metrics" in result
        assert "whoop" in result
        assert "sources" in result
        assert "ml" in result
        assert "dashboard" in result
        assert "anomalies" in result
        # staleness_days should be present in the data block
        assert "staleness_days" in result["data"]

    @patch("scripts.status.get_data_status")
    @patch("scripts.status.get_recent_summaries")
    @patch("scripts.status.get_date_range")
    @patch("scripts.status.get_data_staleness_days")
    @patch("scripts.status.get_data_age_days")
    @patch("scripts.status.read_day")
    @patch("scripts.status.list_available_dates")
    def test_empty_dates_handled(
        self,
        mock_dates,
        mock_read_day,
        mock_age,
        mock_staleness,
        mock_range,
        mock_recent,
        mock_ml,
    ):
        mock_dates.return_value = []
        mock_read_day.return_value = []
        mock_age.return_value = 0
        mock_staleness.return_value = 0
        mock_range.return_value = (None, None)
        mock_recent.return_value = []
        mock_ml.return_value = {
            "days_of_data": 0,
            "min_days_required": 60,
            "ready_to_train": False,
            "days_remaining_until_ready": 60,
            "models_trained": {},
            "last_trained": None,
            "training_days_used": None,
        }

        result = _gather_status()
        assert result["data"]["n_days"] == 0
        assert result["metrics"]["avg_cls_7d"] is None

    @patch("scripts.status.get_data_status")
    @patch("scripts.status.get_recent_summaries")
    @patch("scripts.status.get_date_range")
    @patch("scripts.status.get_data_staleness_days")
    @patch("scripts.status.get_data_age_days")
    @patch("scripts.status.read_day")
    @patch("scripts.status.list_available_dates")
    def test_anomaly_detection_graceful(
        self,
        mock_dates,
        mock_read_day,
        mock_age,
        mock_staleness,
        mock_range,
        mock_recent,
        mock_ml,
    ):
        """If anomaly detection fails, _gather_status should not raise."""
        mock_dates.return_value = ["2026-03-13"]
        mock_read_day.return_value = [
            {"whoop": {}, "metadata": {"sources_available": ["whoop"]}}
        ]
        mock_age.return_value = 1
        mock_staleness.return_value = 0
        mock_range.return_value = ("2026-03-13", "2026-03-13")
        mock_recent.return_value = []
        mock_ml.return_value = {
            "days_of_data": 1, "min_days_required": 60, "ready_to_train": False,
            "days_remaining_until_ready": 59, "models_trained": {}, "last_trained": None,
            "training_days_used": None,
        }

        # Simulate check_anomalies raising
        with patch("scripts.status.check_anomalies", side_effect=Exception("oops")):
            result = _gather_status()  # Should not raise

        assert result["anomalies"] == {}

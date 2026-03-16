"""
Tests for analysis/load_forecast.py — Predicted Cognitive Load Forecast

Coverage:
  1. compute_load_forecast() — main function
     - Returns is_meaningful=False when no calendar and no history
     - Returns is_meaningful=False when < 2 days of history
     - Returns correct load_label for each CLS range
     - Returns correct bucket and statistics when history is available
     - Handles today_calendar=None gracefully (no calendar provided)
     - Handles empty calendar (0 meeting minutes)
     - Handles heavy meeting load (> 4.5h)
     - Confidence levels: high (≥8 days), medium (3-7), low (< 3)
     - Fallback to overall average when no matching bucket days exist

  2. format_forecast_line() — Slack formatter
     - Returns empty string when is_meaningful=False
     - Contains the load label and CLS value
     - Contains confidence note for low-confidence forecasts

  3. Helper functions
     - _bucket_for_minutes() maps meeting minutes to correct buckets
     - _cls_label() maps CLS values to correct labels
     - _percentile() computes correct 25th/75th percentiles
     - _fmt_minutes() formats minutes correctly
     - _build_narrative() generates non-empty narrative strings

  4. Integration with morning brief
     - _compute_load_forecast_for_brief() returns None when not meaningful
     - _compute_load_forecast_for_brief() returns dict when meaningful
     - format_morning_brief_message() includes forecast line when present

Run with: python3 -m pytest tests/test_load_forecast.py -v
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.load_forecast import (
    LoadForecast,
    _bucket_for_minutes,
    _build_narrative,
    _cls_label,
    _fmt_minutes,
    _percentile,
    compute_load_forecast,
    format_forecast_line,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_summary(days: dict) -> dict:
    """Build a rolling summary dict with the given days."""
    return {"days": days}


def _make_day(date_str: str, meeting_minutes: int, avg_cls: float) -> dict:
    """Build a minimal day summary entry."""
    return {
        "date": date_str,
        "calendar": {"total_meeting_minutes": meeting_minutes},
        "metrics_avg": {"cognitive_load_score": avg_cls},
    }


def _make_calendar(meeting_minutes: int) -> dict:
    """Build a minimal calendar dict."""
    return {
        "total_meeting_minutes": meeting_minutes,
        "event_count": max(1, meeting_minutes // 60),
        "events": [],
    }


def _history_days(
    today_str: str,
    days_back: int,
    meeting_minutes: int,
    avg_cls: float,
) -> dict:
    """
    Build `days_back` historical day entries (all before today_str)
    with the specified meeting load and CLS.
    """
    today_dt = datetime.strptime(today_str, "%Y-%m-%d")
    days = {}
    for i in range(1, days_back + 1):
        d = (today_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        days[d] = _make_day(d, meeting_minutes, avg_cls)
    return days


# ─── _bucket_for_minutes ──────────────────────────────────────────────────────

class TestBucketForMinutes:
    def test_zero_minutes_is_light(self):
        assert _bucket_for_minutes(0) == "light"

    def test_59_minutes_is_light(self):
        assert _bucket_for_minutes(59) == "light"

    def test_60_minutes_is_moderate(self):
        assert _bucket_for_minutes(60) == "moderate"

    def test_149_minutes_is_moderate(self):
        assert _bucket_for_minutes(149) == "moderate"

    def test_150_minutes_is_heavy(self):
        assert _bucket_for_minutes(150) == "heavy"

    def test_269_minutes_is_heavy(self):
        assert _bucket_for_minutes(269) == "heavy"

    def test_270_minutes_is_intense(self):
        assert _bucket_for_minutes(270) == "intense"

    def test_large_value_is_intense(self):
        assert _bucket_for_minutes(9000) == "intense"


# ─── _cls_label ───────────────────────────────────────────────────────────────

class TestClsLabel:
    def test_0_is_very_light(self):
        assert _cls_label(0.0) == "Very light"

    def test_0_15_is_very_light(self):
        assert _cls_label(0.15) == "Very light"

    def test_0_20_is_light(self):
        assert _cls_label(0.20) == "Light"

    def test_0_35_is_light(self):
        assert _cls_label(0.35) == "Light"

    def test_0_40_is_moderate(self):
        assert _cls_label(0.40) == "Moderate"

    def test_0_55_is_moderate(self):
        assert _cls_label(0.55) == "Moderate"

    def test_0_60_is_high(self):
        assert _cls_label(0.60) == "High"

    def test_0_75_is_high(self):
        assert _cls_label(0.75) == "High"

    def test_0_80_is_very_high(self):
        assert _cls_label(0.80) == "Very high"

    def test_1_0_is_very_high(self):
        assert _cls_label(1.0) == "Very high"


# ─── _percentile ──────────────────────────────────────────────────────────────

class TestPercentile:
    def test_single_value(self):
        assert _percentile([0.5], 25) == 0.5
        assert _percentile([0.5], 75) == 0.5

    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_two_values_p50(self):
        result = _percentile([0.2, 0.8], 50)
        assert abs(result - 0.5) < 0.001

    def test_sorted_list_p25(self):
        vals = [0.1, 0.2, 0.3, 0.4]
        result = _percentile(vals, 25)
        assert 0.1 <= result <= 0.3

    def test_sorted_list_p75(self):
        vals = [0.1, 0.2, 0.3, 0.4]
        result = _percentile(vals, 75)
        assert 0.2 <= result <= 0.4

    def test_p0_returns_min(self):
        vals = [0.3, 0.7, 0.5]
        assert _percentile(vals, 0) == 0.3

    def test_p100_returns_max(self):
        vals = [0.3, 0.7, 0.5]
        assert _percentile(vals, 100) == 0.7


# ─── _fmt_minutes ──────────────────────────────────────────────────────────────

class TestFmtMinutes:
    def test_zero_minutes(self):
        assert _fmt_minutes(0) == "0m"

    def test_45_minutes(self):
        assert _fmt_minutes(45) == "45m"

    def test_60_minutes(self):
        assert _fmt_minutes(60) == "1h"

    def test_90_minutes(self):
        assert _fmt_minutes(90) == "1h30m"

    def test_120_minutes(self):
        assert _fmt_minutes(120) == "2h"

    def test_150_minutes(self):
        assert _fmt_minutes(150) == "2h30m"


# ─── _build_narrative ─────────────────────────────────────────────────────────

class TestBuildNarrative:
    def test_no_meetings(self):
        text = _build_narrative(0.12, 0.08, 0.16, "Very light", "high", 0, 10, "light")
        assert "No meetings" in text
        assert "light" in text.lower()

    def test_with_meetings(self):
        text = _build_narrative(0.45, 0.38, 0.52, "Moderate", "medium", 90, 5, "moderate")
        assert "moderate" in text.lower()
        assert "CLS ~0.45" in text

    def test_low_confidence_note(self):
        text = _build_narrative(0.45, 0.38, 0.52, "Moderate", "low", 90, 2, "moderate")
        assert "low confidence" in text

    def test_medium_confidence_note(self):
        text = _build_narrative(0.45, 0.38, 0.52, "Moderate", "medium", 90, 5, "moderate")
        assert "based on 5 similar days" in text

    def test_high_load_tip(self):
        text = _build_narrative(0.75, 0.65, 0.82, "High", "high", 240, 8, "heavy")
        assert "protect" in text.lower() or "recovery" in text.lower()

    def test_moderate_load_tip(self):
        text = _build_narrative(0.52, 0.45, 0.60, "Moderate", "medium", 120, 5, "moderate")
        assert "front-load" in text.lower() or "focused" in text.lower() or "moderate" in text.lower()

    def test_no_meetings_deep_work_tip(self):
        text = _build_narrative(0.10, 0.08, 0.14, "Very light", "high", 0, 8, "light")
        assert "deep" in text.lower() or "uninterrupted" in text.lower()

    def test_returns_non_empty_string(self):
        text = _build_narrative(0.30, 0.25, 0.35, "Light", "high", 60, 8, "light")
        assert len(text) > 10


# ─── compute_load_forecast ────────────────────────────────────────────────────

class TestComputeLoadForecast:
    """Tests for the main compute_load_forecast function."""

    def test_not_meaningful_when_no_history(self):
        """No JSONL history → is_meaningful=False."""
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=[]),
            patch("analysis.load_forecast.read_summary", return_value={"days": {}}),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(60))
        assert not forecast.is_meaningful

    def test_not_meaningful_when_only_one_day(self):
        """Only 1 day of history → is_meaningful=False."""
        days = {"2026-03-15": _make_day("2026-03-15", 60, 0.40)}
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=["2026-03-15"]),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(days)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(60))
        assert not forecast.is_meaningful

    def test_meaningful_with_enough_history(self):
        """Sufficient history → is_meaningful=True."""
        history = _history_days("2026-03-16", 5, 90, 0.45)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(90))
        assert forecast.is_meaningful
        assert forecast.predicted_cls is not None

    def test_correct_prediction_for_matching_days(self):
        """Predicted CLS should match the historical avg for similar days."""
        # 5 historical days with 90 min meetings and CLS=0.45
        history = _history_days("2026-03-16", 5, 90, 0.45)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(90))
        assert forecast.is_meaningful
        assert abs(forecast.predicted_cls - 0.45) < 0.01

    def test_correct_load_label(self):
        """Load label should match the predicted CLS."""
        history = _history_days("2026-03-16", 5, 90, 0.50)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(90))
        assert forecast.load_label == "Moderate"

    def test_high_load_label_for_high_cls(self):
        """High CLS → High label."""
        history = _history_days("2026-03-16", 5, 90, 0.70)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(90))
        assert forecast.load_label == "High"

    def test_very_light_label_for_no_meetings(self):
        """No meetings + low historical CLS → Very light label."""
        history = _history_days("2026-03-16", 5, 0, 0.08)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(0))
        assert forecast.load_label == "Very light"

    def test_confidence_high_with_many_days(self):
        """≥ 8 matching days → 'high' confidence."""
        history = _history_days("2026-03-16", 10, 90, 0.45)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(90))
        assert forecast.confidence == "high"

    def test_confidence_medium_with_moderate_days(self):
        """3–7 matching days → 'medium' confidence."""
        history = _history_days("2026-03-16", 5, 90, 0.45)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(90))
        assert forecast.confidence == "medium"

    def test_confidence_low_with_few_days(self):
        """1–2 matching days → 'low' confidence."""
        history = _history_days("2026-03-16", 2, 90, 0.45)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(90))
        assert forecast.confidence == "low"

    def test_no_calendar_uses_zero_minutes(self):
        """None calendar → treated as 0 meeting minutes."""
        history = _history_days("2026-03-16", 5, 0, 0.10)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", None)
        assert forecast.meeting_minutes == 0

    def test_cls_low_leq_predicted_leq_cls_high(self):
        """Percentile range: low ≤ predicted ≤ high."""
        # Varied CLS values in the same bucket for a real percentile range
        today = "2026-03-16"
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        history = {}
        cls_vals = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        for i, cls_v in enumerate(cls_vals, 1):
            d = (today_dt - timedelta(days=i)).strftime("%Y-%m-%d")
            history[d] = _make_day(d, 90, cls_v)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast(today, _make_calendar(90))
        assert forecast.is_meaningful
        assert forecast.cls_low <= forecast.predicted_cls
        assert forecast.predicted_cls <= forecast.cls_high

    def test_fallback_to_overall_average_when_no_matching_bucket(self):
        """When no days match the bucket, fall back to overall CLS average."""
        today = "2026-03-16"
        # All historical days are 'light' (< 60 min). Today is 'heavy' (200 min).
        history = _history_days(today, 5, 30, 0.15)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast(today, _make_calendar(200))
        # Should still produce a meaningful forecast (using overall average)
        assert forecast.is_meaningful
        assert forecast.matching_days == 0
        assert forecast.confidence == "low"

    def test_forecast_does_not_include_today(self):
        """Today's date should NOT appear in the historical profile."""
        today = "2026-03-16"
        # History includes today as well as past days — today should be excluded
        history = _history_days(today, 5, 90, 0.45)
        history[today] = _make_day(today, 90, 0.99)  # today → should be ignored
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast(today, _make_calendar(90))
        # Predicted CLS should be ~0.45, not pulled towards 0.99
        assert forecast.is_meaningful
        assert forecast.predicted_cls < 0.60

    def test_meeting_minutes_stored_correctly(self):
        """meeting_minutes in forecast should match calendar input."""
        history = _history_days("2026-03-16", 5, 120, 0.50)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(120))
        assert forecast.meeting_minutes == 120

    def test_narrative_is_non_empty_when_meaningful(self):
        """Narrative should be a non-empty string when forecast is meaningful."""
        history = _history_days("2026-03-16", 5, 90, 0.45)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(90))
        assert forecast.is_meaningful
        assert len(forecast.narrative) > 10

    def test_to_dict_serializable(self):
        """to_dict() should produce JSON-serializable output."""
        history = _history_days("2026-03-16", 5, 90, 0.45)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            forecast = compute_load_forecast("2026-03-16", _make_calendar(90))
        d = forecast.to_dict()
        # Should not raise
        json.dumps(d)
        assert "predicted_cls" in d
        assert "load_label" in d
        assert "is_meaningful" in d


# ─── format_forecast_line ─────────────────────────────────────────────────────

class TestFormatForecastLine:
    def test_empty_when_not_meaningful(self):
        forecast = LoadForecast(date_str="2026-03-16", is_meaningful=False)
        assert format_forecast_line(forecast) == ""

    def test_empty_when_no_predicted_cls(self):
        forecast = LoadForecast(
            date_str="2026-03-16",
            predicted_cls=None,
            is_meaningful=True,
        )
        assert format_forecast_line(forecast) == ""

    def test_contains_load_label(self):
        forecast = LoadForecast(
            date_str="2026-03-16",
            predicted_cls=0.45,
            cls_low=0.38,
            cls_high=0.52,
            load_label="Moderate",
            confidence="high",
            matching_days=10,
            is_meaningful=True,
        )
        line = format_forecast_line(forecast)
        assert "Moderate" in line

    def test_contains_cls_value(self):
        forecast = LoadForecast(
            date_str="2026-03-16",
            predicted_cls=0.45,
            cls_low=0.38,
            cls_high=0.52,
            load_label="Moderate",
            confidence="medium",
            matching_days=5,
            is_meaningful=True,
        )
        line = format_forecast_line(forecast)
        assert "0.45" in line

    def test_low_confidence_shows_warning(self):
        forecast = LoadForecast(
            date_str="2026-03-16",
            predicted_cls=0.45,
            cls_low=0.38,
            cls_high=0.52,
            load_label="Moderate",
            confidence="low",
            matching_days=2,
            is_meaningful=True,
        )
        line = format_forecast_line(forecast)
        assert "low confidence" in line.lower() or "⚠️" in line

    def test_high_confidence_shows_matching_days(self):
        forecast = LoadForecast(
            date_str="2026-03-16",
            predicted_cls=0.45,
            cls_low=0.38,
            cls_high=0.52,
            load_label="Moderate",
            confidence="high",
            matching_days=12,
            is_meaningful=True,
        )
        line = format_forecast_line(forecast)
        assert "12" in line

    def test_shows_range_when_wide_enough(self):
        """When cls_high - cls_low > 0.05, range should be shown."""
        forecast = LoadForecast(
            date_str="2026-03-16",
            predicted_cls=0.50,
            cls_low=0.35,
            cls_high=0.65,
            load_label="Moderate",
            confidence="high",
            matching_days=10,
            is_meaningful=True,
        )
        line = format_forecast_line(forecast)
        # Range should be shown
        assert "0.35" in line or "0.65" in line

    def test_no_range_when_narrow(self):
        """When cls_high - cls_low <= 0.05, no range is shown."""
        forecast = LoadForecast(
            date_str="2026-03-16",
            predicted_cls=0.50,
            cls_low=0.48,
            cls_high=0.52,
            load_label="Moderate",
            confidence="high",
            matching_days=10,
            is_meaningful=True,
        )
        line = format_forecast_line(forecast)
        # Only point estimate shown
        assert "0.50" in line
        assert "0.48" not in line

    def test_contains_forecast_label_text(self):
        forecast = LoadForecast(
            date_str="2026-03-16",
            predicted_cls=0.45,
            cls_low=0.38,
            cls_high=0.52,
            load_label="Moderate",
            confidence="high",
            matching_days=10,
            is_meaningful=True,
        )
        line = format_forecast_line(forecast)
        assert "forecast" in line.lower() or "📊" in line


# ─── Integration with morning brief ──────────────────────────────────────────

class TestMorningBriefIntegration:
    """Test that the load forecast is wired into the morning brief correctly."""

    def test_compute_load_forecast_for_brief_returns_none_when_not_meaningful(self):
        """Helper should return None when forecast is not meaningful (no history)."""
        from analysis.morning_brief import _compute_load_forecast_for_brief
        # Patch the store functions used by compute_load_forecast so no history exists.
        # The helper uses a lazy import of compute_load_forecast inside the function,
        # so we patch through the analysis.load_forecast module directly.
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=[]),
            patch("analysis.load_forecast.read_summary", return_value={"days": {}}),
        ):
            result = _compute_load_forecast_for_brief("2026-03-16", _make_calendar(60))
        assert result is None

    def test_compute_load_forecast_for_brief_returns_dict_when_meaningful(self):
        """Helper should return a dict with expected keys when meaningful."""
        from analysis.morning_brief import _compute_load_forecast_for_brief
        history = _history_days("2026-03-16", 5, 90, 0.45)
        dates = sorted(history.keys())
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value=_make_summary(history)),
        ):
            result = _compute_load_forecast_for_brief("2026-03-16", _make_calendar(90))
        assert result is not None
        assert result["is_meaningful"] is True
        assert "line" in result
        assert "predicted_cls" in result
        assert "load_label" in result
        assert "confidence" in result

    def test_format_morning_brief_message_includes_forecast(self):
        """format_morning_brief_message() should include the forecast line when present."""
        from analysis.morning_brief import format_morning_brief_message
        brief = {
            "date": "2026-03-16",
            "whoop": {
                "recovery_score": 75.0,
                "hrv_rmssd_milli": 70.0,
                "sleep_hours": 7.5,
                "sleep_performance": 80.0,
                "resting_heart_rate": 55.0,
            },
            "readiness": {
                "tier": "good",
                "label": "Good",
                "recommendation": "Good day to work on complex tasks.",
            },
            "yesterday": {},
            "hrv_baseline": 70.0,
            "trend_context": {},
            "personal_baseline": None,
            "today_calendar": None,
            "cognitive_debt": None,
            "dps_trend": None,
            "tomorrow_focus_plan": None,
            # v14: Load forecast present
            "load_forecast": {
                "line": "📊 *Load forecast:* Moderate — CLS ~0.45 (10 similar days)",
                "predicted_cls": 0.45,
                "load_label": "Moderate",
                "confidence": "high",
                "matching_days": 10,
                "narrative": "90m of meetings → moderate load expected.",
                "is_meaningful": True,
            },
        }
        message = format_morning_brief_message(brief)
        assert "Load forecast" in message or "forecast" in message.lower() or "📊" in message

    def test_format_morning_brief_message_omits_forecast_when_absent(self):
        """format_morning_brief_message() should not error when load_forecast is None."""
        from analysis.morning_brief import format_morning_brief_message
        brief = {
            "date": "2026-03-16",
            "whoop": {
                "recovery_score": 75.0,
                "hrv_rmssd_milli": 70.0,
                "sleep_hours": 7.5,
                "sleep_performance": 80.0,
                "resting_heart_rate": 55.0,
            },
            "readiness": {
                "tier": "good",
                "label": "Good",
                "recommendation": "Good day to work on complex tasks.",
            },
            "yesterday": {},
            "hrv_baseline": 70.0,
            "trend_context": {},
            "personal_baseline": None,
            "today_calendar": None,
            "cognitive_debt": None,
            "dps_trend": None,
            "tomorrow_focus_plan": None,
            "load_forecast": None,  # No forecast
        }
        # Should not raise
        message = format_morning_brief_message(brief)
        assert len(message) > 0

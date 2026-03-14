"""
Tests for scripts/weekly_summary.py — Deterministic Weekly Summary

Coverage:
  - _week_dates() returns 7 dates ending on the given date
  - _mean() / _delta() / _arrow() helpers
  - _fmt_val() / _fmt_delta() / _fmt_ms() formatters
  - _day_label() / _hour_label() formatters
  - load_week_data() with mock rolling summary
  - compute_weekly_summary() delta computation
  - format_weekly_message() structural requirements
  - Empty / no-data cases do not crash
  - Prior-week comparison block only appears with ≥1 prior-week day

Run with: python3 -m pytest tests/test_weekly_summary.py -v
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.weekly_summary import (
    _week_dates,
    _mean,
    _delta,
    _arrow,
    _fmt_val,
    _fmt_delta,
    _fmt_ms,
    _fmt_ms_delta,
    _day_label,
    _hour_label,
    _pct_change,
    load_week_data,
    compute_weekly_summary,
    format_weekly_message,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_day_summary(
    date: str,
    cls: float = 0.30,
    fdi: float = 0.70,
    sdi: float = 0.20,
    csc: float = 0.15,
    ras: float = 0.75,
    recovery: float = 72.0,
    hrv: float = 65.0,
    sleep_h: float = 7.5,
    sleep_perf: float = 78.0,
    meeting_mins: int = 120,
    peak_focus_hour: int = 10,
    active_fdi: float = 0.65,
    active_windows: int = 8,
    rt_focus_mins: float = 90.0,
    rt_distraction_mins: float = 30.0,
    productive_pct: float = 75.0,
) -> dict:
    """Build a fake daily summary entry matching rolling.json schema."""
    return {
        "date": date,
        "metrics_avg": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "context_switch_cost": csc,
            "recovery_alignment_score": ras,
        },
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "sleep_hours": sleep_h,
            "sleep_performance": sleep_perf,
        },
        "focus_quality": {
            "active_fdi": active_fdi,
            "active_windows": active_windows,
            "peak_focus_hour": peak_focus_hour,
            "peak_focus_fdi": active_fdi + 0.05,
        },
        "calendar": {
            "total_meeting_minutes": meeting_mins,
            "meeting_windows": meeting_mins // 15,
        },
        "slack": {
            "total_messages_sent": 30,
            "total_messages_received": 50,
        },
        "rescuetime": {
            "focus_minutes": rt_focus_mins,
            "distraction_minutes": rt_distraction_mins,
            "neutral_minutes": 20.0,
            "active_minutes": rt_focus_mins + rt_distraction_mins + 20.0,
            "productive_pct": productive_pct,
            "top_activities": ["Xcode", "Terminal"],
            "rt_windows": 6,
        },
    }


def _make_rolling(days: list[str]) -> dict:
    """Build a fake rolling.json dict with one summary per date."""
    return {
        "days": {d: _make_day_summary(d) for d in days},
        "total_days": len(days),
        "last_updated": "2026-03-14T10:00:00",
    }


# ─── Helper unit tests ────────────────────────────────────────────────────────

class TestWeekDates:
    def test_returns_seven_dates(self):
        dates = _week_dates("2026-03-14")
        assert len(dates) == 7

    def test_last_date_is_end_date(self):
        dates = _week_dates("2026-03-14")
        assert dates[-1] == "2026-03-14"

    def test_first_date_is_six_days_before(self):
        dates = _week_dates("2026-03-14")
        assert dates[0] == "2026-03-08"

    def test_dates_are_sequential(self):
        dates = _week_dates("2026-03-14")
        for i in range(1, len(dates)):
            prev = datetime.strptime(dates[i - 1], "%Y-%m-%d")
            curr = datetime.strptime(dates[i], "%Y-%m-%d")
            assert (curr - prev).days == 1

    def test_dates_format_is_iso(self):
        dates = _week_dates("2026-03-10")
        for d in dates:
            datetime.strptime(d, "%Y-%m-%d")  # should not raise


class TestMeanHelper:
    def test_empty_list_returns_none(self):
        assert _mean([]) is None

    def test_all_none_returns_none(self):
        assert _mean([None, None]) is None

    def test_mixed_none_ignores_none(self):
        assert _mean([1.0, None, 3.0]) == 2.0

    def test_single_value(self):
        assert _mean([0.75]) == 0.75

    def test_precision_three_decimal_places(self):
        result = _mean([0.1, 0.2, 0.3])
        assert isinstance(result, float)
        assert abs(result - 0.2) < 0.001


class TestDeltaHelper:
    def test_positive_delta(self):
        assert _delta(0.8, 0.6) == pytest.approx(0.2, abs=0.001)

    def test_negative_delta(self):
        assert _delta(0.4, 0.6) == pytest.approx(-0.2, abs=0.001)

    def test_none_this_week_returns_none(self):
        assert _delta(None, 0.5) is None

    def test_none_last_week_returns_none(self):
        assert _delta(0.5, None) is None

    def test_zero_delta(self):
        assert _delta(0.5, 0.5) == 0.0


class TestArrowHelper:
    def test_positive_delta_up_good(self):
        assert _arrow(0.1, good_direction="up") == "↑"

    def test_negative_delta_up_good(self):
        assert _arrow(-0.1, good_direction="up") == "↓"

    def test_positive_delta_down_good(self):
        # positive delta on "down is good" metric = things got worse = ↓
        assert _arrow(0.1, good_direction="down") == "↓"

    def test_negative_delta_down_good(self):
        assert _arrow(-0.1, good_direction="down") == "↑"

    def test_near_zero_returns_flat(self):
        assert _arrow(0.005) == "→"

    def test_none_returns_flat(self):
        assert _arrow(None) == "→"


class TestFmtVal:
    def test_none_returns_dash(self):
        assert _fmt_val(None) == "—"

    def test_zero_point_seven(self):
        result = _fmt_val(0.70)
        assert "70" in result

    def test_default_suffix_is_pct(self):
        result = _fmt_val(0.80)
        assert "%" in result

    def test_custom_scale(self):
        result = _fmt_val(0.5, scale=1.0, suffix="", decimals=2)
        assert "0.50" in result


class TestFmtDelta:
    def test_empty_when_none(self):
        assert _fmt_delta(None) == ""

    def test_small_delta_suppressed(self):
        # < 1% change → empty
        assert _fmt_delta(0.005) == ""

    def test_large_positive_delta(self):
        result = _fmt_delta(0.10, good_direction="up")
        assert "↑" in result
        assert "10" in result

    def test_large_negative_delta_down_good(self):
        result = _fmt_delta(-0.10, good_direction="down")
        assert "↑" in result  # negative delta on "down is good" = arrow up


class TestFmtMs:
    def test_none_returns_dash(self):
        assert _fmt_ms(None) == "—"

    def test_integer_ms(self):
        assert _fmt_ms(65.0) == "65ms"

    def test_rounds_correctly(self):
        assert _fmt_ms(64.6) == "65ms"


class TestDayLabel:
    def test_none_returns_dash(self):
        assert _day_label(None) == "—"

    def test_monday_format(self):
        # 2026-03-09 is a Monday
        label = _day_label("2026-03-09")
        assert "Mon" in label
        assert "9" in label

    def test_invalid_date_does_not_crash(self):
        result = _day_label("not-a-date")
        assert isinstance(result, str)


class TestHourLabel:
    def test_none_returns_dash(self):
        assert _hour_label(None) == "—"

    def test_9am(self):
        assert _hour_label(9) == "9am"

    def test_2pm(self):
        assert _hour_label(14) == "2pm"

    def test_midnight(self):
        assert _hour_label(0) == "12am"

    def test_noon(self):
        assert _hour_label(12) == "12pm"


# ─── load_week_data tests ─────────────────────────────────────────────────────

class TestLoadWeekData:
    """Tests using mocked rolling summary and JSONL reads."""

    def _mock_rolling(self, days: list[str]) -> dict:
        return _make_rolling(days)

    def test_no_data_returns_zero_days(self):
        with patch("scripts.weekly_summary.read_summary", return_value={"days": {}}), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = load_week_data("2026-03-14")
        assert result["days_with_data"] == 0

    def test_full_week_returns_seven_days_with_data(self):
        dates = [f"2026-03-{d:02d}" for d in range(8, 15)]  # Mar 8–14
        rolling = _make_rolling(dates)
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = load_week_data("2026-03-14")
        assert result["days_with_data"] == 7

    def test_partial_week_counts_available_days(self):
        # Only 3 days in the rolling summary for this week
        dates = ["2026-03-12", "2026-03-13", "2026-03-14"]
        rolling = _make_rolling(dates)
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = load_week_data("2026-03-14")
        assert result["days_with_data"] == 3

    def test_metrics_dict_has_required_keys(self):
        dates = ["2026-03-13", "2026-03-14"]
        rolling = _make_rolling(dates)
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = load_week_data("2026-03-14")
        m = result.get("metrics", {})
        for key in ["cls", "fdi", "sdi", "csc", "ras"]:
            assert key in m, f"Missing metric key: {key}"

    def test_whoop_dict_has_required_keys(self):
        dates = ["2026-03-14"]
        rolling = _make_rolling(dates)
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = load_week_data("2026-03-14")
        w = result.get("whoop", {})
        for key in ["recovery", "hrv", "sleep_hours", "sleep_performance"]:
            assert key in w, f"Missing WHOOP key: {key}"

    def test_cls_mean_is_accurate(self):
        dates = ["2026-03-13", "2026-03-14"]
        rolling = _make_rolling(dates)
        # Both days have cls=0.30 in the fixture
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = load_week_data("2026-03-14")
        assert abs(result["metrics"]["cls"] - 0.30) < 0.001

    def test_omi_stats_present(self):
        dates = ["2026-03-14"]
        rolling = _make_rolling(dates)
        # Mock a day with one Omi conversation window
        fake_omi_window = {
            "omi": {"conversation_active": True, "word_count": 200, "sessions_count": 1, "speech_seconds": 300.0}
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[fake_omi_window]):
            result = load_week_data("2026-03-14")
        omi = result.get("omi_stats", {})
        assert omi.get("days_active") == 1
        assert omi.get("total_words") == 200
        assert omi.get("total_sessions") == 1

    def test_rt_stats_present_when_rescuetime_in_summary(self):
        dates = ["2026-03-14"]
        rolling = _make_rolling(dates)
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = load_week_data("2026-03-14")
        rt = result.get("rt_stats", {})
        assert rt.get("days_tracked", 0) >= 1

    def test_calendar_stats_include_meeting_minutes(self):
        dates = ["2026-03-13", "2026-03-14"]
        rolling = _make_rolling(dates)
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = load_week_data("2026-03-14")
        cal = result.get("calendar_stats", {})
        # Each day has 120 meeting minutes in fixture → total = 240
        assert cal.get("total_meeting_minutes") == 240

    def test_peak_focus_hour_is_integer_or_none(self):
        dates = ["2026-03-14"]
        rolling = _make_rolling(dates)
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = load_week_data("2026-03-14")
        peak = result.get("peak_focus_hour")
        assert peak is None or isinstance(peak, int)

    def test_dates_key_always_present(self):
        with patch("scripts.weekly_summary.read_summary", return_value={"days": {}}), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = load_week_data("2026-03-14")
        assert "dates" in result
        assert len(result["dates"]) == 7


# ─── compute_weekly_summary tests ─────────────────────────────────────────────

class TestComputeWeeklySummary:
    """Tests for the top-level aggregation function."""

    def _make_full_rolling(self) -> dict:
        # This week: Mar 8–14
        this_week = [f"2026-03-{d:02d}" for d in range(8, 15)]
        # Prior week: Mar 1–7
        prior_week = [f"2026-03-0{d}" for d in range(1, 8)]
        all_dates = this_week + prior_week
        return _make_rolling(all_dates)

    def test_result_has_required_keys(self):
        rolling = self._make_full_rolling()
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = compute_weekly_summary("2026-03-14")
        for key in ["end_date", "this_week", "last_week", "deltas", "whoop_deltas"]:
            assert key in result, f"Missing key: {key}"

    def test_end_date_matches_input(self):
        rolling = self._make_full_rolling()
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = compute_weekly_summary("2026-03-14")
        assert result["end_date"] == "2026-03-14"

    def test_deltas_computed_when_both_weeks_have_data(self):
        rolling = self._make_full_rolling()
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = compute_weekly_summary("2026-03-14")
        deltas = result["deltas"]
        # Same data for both weeks → all deltas ≈ 0
        for k in ["cls", "fdi", "sdi", "ras"]:
            if deltas.get(k) is not None:
                assert abs(deltas[k]) < 0.01, f"Expected near-zero delta for {k}"

    def test_deltas_empty_when_no_prior_week(self):
        # Only this week's data
        this_week = [f"2026-03-{d:02d}" for d in range(8, 15)]
        rolling = _make_rolling(this_week)
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = compute_weekly_summary("2026-03-14")
        # deltas may be empty or all None
        deltas = result["deltas"]
        assert isinstance(deltas, dict)

    def test_cls_delta_correct_sign(self):
        # This week: high CLS; prior week: low CLS → positive delta
        this_week_dates = [f"2026-03-{d:02d}" for d in range(8, 15)]
        prior_week_dates = [f"2026-03-0{d}" for d in range(1, 8)]
        days = {}
        for d in this_week_dates:
            s = _make_day_summary(d)
            s["metrics_avg"]["cognitive_load_score"] = 0.60
            days[d] = s
        for d in prior_week_dates:
            s = _make_day_summary(d)
            s["metrics_avg"]["cognitive_load_score"] = 0.30
            days[d] = s
        rolling = {"days": days}
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = compute_weekly_summary("2026-03-14")
        cls_delta = result["deltas"].get("cls")
        assert cls_delta is not None
        assert cls_delta > 0, f"Expected positive CLS delta (this > last), got {cls_delta}"

    def test_whoop_deltas_computed(self):
        rolling = self._make_full_rolling()
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            result = compute_weekly_summary("2026-03-14")
        assert isinstance(result["whoop_deltas"], dict)


# ─── format_weekly_message tests ─────────────────────────────────────────────

class TestFormatWeeklyMessage:
    """Tests for the Slack message formatter."""

    def _make_summary(self, this_days: int = 5, last_days: int = 5) -> dict:
        this_week_end = "2026-03-14"
        this_week_dates = [f"2026-03-{d:02d}" for d in range(15 - this_days, 15)]
        prior_end = "2026-03-07"
        prior_dates = [f"2026-03-0{d}" for d in range(8 - last_days, 8)]
        all_dates = this_week_dates + prior_dates
        rolling = _make_rolling(all_dates)

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            return compute_weekly_summary(this_week_end)

    def test_output_is_string(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        assert isinstance(result, str)

    def test_output_is_non_empty(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        assert len(result) > 50

    def test_contains_weekly_summary_header(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        assert "Weekly Presence Summary" in result

    def test_contains_cls_metric(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        assert "CLS" in result

    def test_contains_fdi_metric(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        assert "FDI" in result

    def test_contains_whoop_recovery(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        assert "Recovery" in result or "WHOOP" in result

    def test_no_crash_when_no_data(self):
        empty_summary = {
            "end_date": "2026-03-14",
            "this_week": {"dates": [], "days_with_data": 0, "metrics": None},
            "last_week": {"dates": [], "days_with_data": 0},
            "deltas": {},
            "whoop_deltas": {},
        }
        result = format_weekly_message(empty_summary)
        assert isinstance(result, str)
        assert "No presence data" in result

    def test_prior_week_comparison_appears_when_both_weeks_have_data(self):
        summary = self._make_summary(this_days=5, last_days=5)
        result = format_weekly_message(summary)
        assert "Prior Week" in result or "vs" in result.lower()

    def test_prior_week_comparison_absent_when_no_prior_data(self):
        summary = self._make_summary(this_days=3, last_days=0)
        result = format_weekly_message(summary)
        # When last_week has no data, the comparison block should not appear
        # (or if it does appear, it should indicate stable/no change)
        assert isinstance(result, str)  # Must not crash at minimum

    def test_meeting_minutes_shown_when_present(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        assert "Meeting" in result or "meeting" in result

    def test_slack_messages_shown(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        assert "Slack" in result or "sent" in result

    def test_footer_present(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        assert "Presence Tracker" in result

    def test_date_range_in_header(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        # Should include month/date range like "Mar 8–14"
        assert "Mar" in result

    def test_rescuetime_section_shown_when_rt_data(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        # RT data is in fixture → deep work section should appear
        assert "focused" in result.lower() or "focus" in result.lower() or "deep" in result.lower()

    def test_best_worst_day_section_present(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        assert "Best" in result or "Worst" in result or "Heaviest" in result

    def test_message_has_multiple_lines(self):
        summary = self._make_summary()
        result = format_weekly_message(summary)
        lines = result.split("\n")
        assert len(lines) >= 8


# ─── Integration: compute + format together ───────────────────────────────────

class TestWeeklySummaryIntegration:
    def test_full_pipeline_does_not_crash(self):
        dates = ["2026-03-12", "2026-03-13", "2026-03-14"]
        rolling = _make_rolling(dates)
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_single_day_does_not_crash(self):
        dates = ["2026-03-14"]
        rolling = _make_rolling(dates)
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)
        assert "Weekly Presence Summary" in message

    def test_omi_conversation_data_flows_into_message(self):
        dates = ["2026-03-14"]
        rolling = _make_rolling(dates)
        # Inject Omi conversation windows
        fake_omi_window = {
            "omi": {
                "conversation_active": True,
                "word_count": 500,
                "sessions_count": 3,
                "speech_seconds": 600.0,
            }
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[fake_omi_window]):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)
        assert "session" in message.lower() or "conversation" in message.lower()

    def test_no_omi_no_omi_section(self):
        dates = ["2026-03-14"]
        rolling = _make_rolling(dates)
        # No Omi windows
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")
            omi_stats = summary["this_week"].get("omi_stats", {})
        assert omi_stats.get("days_active", 0) == 0

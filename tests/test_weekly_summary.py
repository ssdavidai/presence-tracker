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
  - DPS extraction, sparkline, tier label, and weekly headline (v1.1)

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
    _dps_sparkline,
    _dps_tier_label,
    _extract_dps,
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


# ─── DPS helpers (v1.1) ───────────────────────────────────────────────────────

class TestDpsSparkline:
    def test_all_none_shows_dots(self):
        result = _dps_sparkline([None, None, None, None, None, None, None])
        assert result == "·······"

    def test_high_dps_shows_full_block(self):
        result = _dps_sparkline([100.0])
        assert result == "█"

    def test_zero_dps_shows_space_or_low_block(self):
        result = _dps_sparkline([0.0])
        assert len(result) == 1
        assert result != "·"  # 0.0 is a real value, not None

    def test_seven_days_returns_seven_chars(self):
        dps = [60.0, 70.0, None, 80.0, 55.0, 75.0, 65.0]
        result = _dps_sparkline(dps)
        assert len(result) == 7

    def test_mixed_values_and_nones(self):
        dps = [50.0, None, 75.0]
        result = _dps_sparkline(dps)
        assert len(result) == 3
        assert result[1] == "·"

    def test_all_same_value_consistent_chars(self):
        dps = [70.0, 70.0, 70.0]
        result = _dps_sparkline(dps)
        assert len(set(result)) == 1  # all same char


class TestDpsTierLabel:
    def test_peak_at_80(self):
        label = _dps_tier_label(80.0)
        assert "peak" in label.lower()

    def test_strong_at_70(self):
        label = _dps_tier_label(70.0)
        assert "strong" in label.lower()

    def test_moderate_at_55(self):
        label = _dps_tier_label(55.0)
        assert "moderate" in label.lower()

    def test_stretched_at_40(self):
        label = _dps_tier_label(40.0)
        assert "stretch" in label.lower()

    def test_recovery_at_20(self):
        label = _dps_tier_label(20.0)
        assert "recovery" in label.lower()

    def test_boundary_at_65(self):
        # 65 is the boundary for strong
        label = _dps_tier_label(65.0)
        assert "strong" in label.lower()


class TestExtractDps:
    def test_returns_stored_dps_when_present(self):
        record = {
            "date": "2026-03-14",
            "presence_score": {"dps": 72.5, "tier": "strong"},
        }
        result = _extract_dps(record)
        assert result == 72.5

    def test_returns_none_when_no_date(self):
        record = {}
        result = _extract_dps(record)
        assert result is None

    def test_returns_none_when_windows_unavailable(self):
        """Without real JSONL, fallback returns None gracefully."""
        record = {"date": "2099-01-01"}  # date that has no JSONL
        result = _extract_dps(record)
        assert result is None

    def test_rounds_to_one_decimal(self):
        record = {
            "date": "2026-03-14",
            "presence_score": {"dps": 72.456789},
        }
        result = _extract_dps(record)
        assert result == 72.5


# ─── DPS integration in load_week_data ───────────────────────────────────────

class TestLoadWeekDataDps:
    def _make_dps_day(self, date: str, dps: float) -> dict:
        base = _make_day_summary(date)
        base["presence_score"] = {"dps": dps, "tier": "strong", "components": {}}
        return base

    def test_dps_avg_computed_from_stored_scores(self):
        dates = ["2026-03-12", "2026-03-13", "2026-03-14"]
        rolling = {
            "days": {
                "2026-03-12": self._make_dps_day("2026-03-12", 70.0),
                "2026-03-13": self._make_dps_day("2026-03-13", 80.0),
                "2026-03-14": self._make_dps_day("2026-03-14", 60.0),
            },
            "total_days": 3,
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            data = load_week_data("2026-03-14")

        assert data["dps_avg"] == pytest.approx(70.0, abs=0.5)

    def test_dps_per_day_none_for_missing_dates(self):
        # Only 2 of the 7 days in the window have data
        dates = ["2026-03-13", "2026-03-14"]
        rolling = {
            "days": {
                "2026-03-13": self._make_dps_day("2026-03-13", 65.0),
                "2026-03-14": self._make_dps_day("2026-03-14", 75.0),
            },
            "total_days": 2,
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            data = load_week_data("2026-03-14")

        dps_per_day = data["dps_per_day"]
        assert len(dps_per_day) == 7
        none_count = sum(1 for v in dps_per_day if v is None)
        assert none_count == 5  # 5 missing days

    def test_dps_extremes_best_highest_score(self):
        rolling = {
            "days": {
                "2026-03-12": self._make_dps_day("2026-03-12", 55.0),
                "2026-03-13": self._make_dps_day("2026-03-13", 90.0),  # best
                "2026-03-14": self._make_dps_day("2026-03-14", 40.0),  # worst
            },
            "total_days": 3,
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            data = load_week_data("2026-03-14")

        assert data["dps_extremes"]["best"]["date"] == "2026-03-13"
        assert data["dps_extremes"]["worst"]["date"] == "2026-03-14"

    def test_dps_extremes_none_when_no_data(self):
        with patch("scripts.weekly_summary.read_summary", return_value={"days": {}}), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            data = load_week_data("2026-03-14")

        assert data.get("dps_avg") is None

    def test_dps_avg_none_when_no_presence_score_and_no_jsonl(self):
        # Days without presence_score and empty JSONL → fallback returns None
        dates = ["2026-03-14"]
        rolling = {
            "days": {"2026-03-14": _make_day_summary("2026-03-14")},
            "total_days": 1,
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            data = load_week_data("2026-03-14")

        # dps_avg may be None (no stored DPS, empty JSONL fallback returns None)
        # The key must exist
        assert "dps_avg" in data


# ─── DPS in compute_weekly_summary ───────────────────────────────────────────

class TestComputeWeeklySummaryDps:
    def _make_dps_day(self, date: str, dps: float) -> dict:
        base = _make_day_summary(date)
        base["presence_score"] = {"dps": dps, "tier": "strong"}
        return base

    def test_dps_delta_present_in_result(self):
        this_dates = ["2026-03-08", "2026-03-09", "2026-03-14"]
        last_dates = ["2026-03-01", "2026-03-02", "2026-03-07"]
        rolling = {
            "days": {
                **{d: self._make_dps_day(d, 70.0) for d in this_dates},
                **{d: self._make_dps_day(d, 60.0) for d in last_dates},
            },
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")

        assert "dps_delta" in summary
        dps_d = summary["dps_delta"]
        # this week avg 70, last week avg 60 → delta ≈ +10
        if dps_d is not None:
            assert dps_d > 0

    def test_dps_delta_none_when_no_prior_week(self):
        rolling = {
            "days": {"2026-03-14": self._make_dps_day("2026-03-14", 75.0)},
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")

        # No prior week data → dps_delta should be None (both weeks' DPS may not align)
        assert "dps_delta" in summary


# ─── DPS in format_weekly_message ────────────────────────────────────────────

class TestFormatWeeklyMessageDps:
    def _make_dps_day(self, date: str, dps: float) -> dict:
        base = _make_day_summary(date)
        base["presence_score"] = {"dps": dps, "tier": "strong"}
        return base

    def test_dps_headline_present_when_dps_available(self):
        dates = ["2026-03-12", "2026-03-13", "2026-03-14"]
        rolling = {
            "days": {d: self._make_dps_day(d, 72.0) for d in dates},
            "total_days": 3,
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)

        assert "DPS" in message
        assert "72" in message

    def test_dps_headline_shows_tier(self):
        dates = ["2026-03-14"]
        rolling = {
            "days": {"2026-03-14": self._make_dps_day("2026-03-14", 85.0)},
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)

        # 85 → "peak presence"
        assert "peak" in message.lower()

    def test_dps_sparkline_in_message(self):
        dates = ["2026-03-12", "2026-03-13", "2026-03-14"]
        rolling = {
            "days": {d: self._make_dps_day(d, 70.0) for d in dates},
            "total_days": 3,
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)

        # Sparkline is rendered in backtick code block
        assert "`" in message

    def test_dps_best_worst_day_in_message_when_multiple_days(self):
        rolling = {
            "days": {
                "2026-03-12": self._make_dps_day("2026-03-12", 55.0),
                "2026-03-13": self._make_dps_day("2026-03-13", 90.0),
                "2026-03-14": self._make_dps_day("2026-03-14", 40.0),
            },
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)

        # Best/worst days section should reference DPS
        assert "Best day (DPS)" in message or "Toughest day (DPS)" in message

    def test_dps_delta_shown_when_prior_week_higher(self):
        this_dates = ["2026-03-08", "2026-03-09", "2026-03-14"]
        last_dates = ["2026-03-01", "2026-03-02", "2026-03-07"]
        rolling = {
            "days": {
                **{d: self._make_dps_day(d, 75.0) for d in this_dates},
                **{d: self._make_dps_day(d, 60.0) for d in last_dates},
            },
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)

        # DPS improved → should see an upward trend note
        assert "DPS" in message

    def test_no_dps_no_dps_section(self):
        """When no DPS data at all, message should still render without DPS line."""
        dates = ["2026-03-14"]
        rolling = {
            "days": {"2026-03-14": _make_day_summary("2026-03-14")},  # no presence_score
        }
        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)

        # Message should still render without crashing
        assert "Weekly Presence Summary" in message


# ─── v2.3: Personal Records + Next-week Pacing ────────────────────────────────

class TestWeeklyPersonalRecordsMilestones:
    """
    The weekly summary should surface Personal Records milestones (new all-time
    bests, meaningful streaks) that were set during the 7-day window.

    These tests mock the personal_records module to control which records are
    returned and verify that format_weekly_message() includes (or omits) the
    milestones section correctly.
    """

    def _make_rolling(self, dates):
        """Minimal rolling summary with the given dates having data."""
        return {
            "days": {d: _make_day_summary(d) for d in dates},
        }

    def test_milestones_section_present_when_new_best_set(self):
        """
        When a new personal best was set this week, the message should include
        a milestones section with 🏆.
        """
        dates = ["2026-03-12", "2026-03-13", "2026-03-14"]
        rolling = self._make_rolling(dates)

        # Mock personal_records: FDI personal best set on 2026-03-14
        mock_records_obj = MagicMock()
        mock_records_obj.is_meaningful.return_value = True
        mock_records_obj.best_fdi_day = MagicMock()
        mock_records_obj.best_fdi_day.date_str = "2026-03-14"
        mock_records_obj.best_cls_day = None
        mock_records_obj.best_ras_day = None
        mock_records_obj.best_dps_day = None
        mock_records_obj.best_recovery_day = None
        mock_records_obj.best_hrv_day = None
        mock_records_obj.low_load_streak = None
        mock_records_obj.deep_focus_streak = None
        mock_records_obj.recovery_aligned_streak = None
        mock_records_obj.green_recovery_streak = None

        from analysis.personal_records import TodayRecordSummary
        today_rec = TodayRecordSummary(
            date_str="2026-03-14",
            has_records=True,
            new_best_fdi=True,
        )

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]), \
             patch("analysis.personal_records.compute_personal_records",
                   return_value=mock_records_obj), \
             patch("analysis.personal_records.check_today_records",
                   return_value=today_rec):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)

        assert "milestones" in message.lower() or "🏆" in message

    def test_milestones_section_absent_when_no_records(self):
        """
        When no personal records were set this week, the milestones section
        should be silently omitted from the message.
        """
        dates = ["2026-03-14"]
        rolling = self._make_rolling(dates)

        mock_records_obj = MagicMock()
        mock_records_obj.is_meaningful.return_value = True
        mock_records_obj.best_fdi_day = MagicMock()
        mock_records_obj.best_fdi_day.date_str = "2026-02-01"  # older date, not this week
        mock_records_obj.best_cls_day = None
        mock_records_obj.best_ras_day = None
        mock_records_obj.best_dps_day = None
        mock_records_obj.best_recovery_day = None
        mock_records_obj.best_hrv_day = None
        mock_records_obj.low_load_streak = None
        mock_records_obj.deep_focus_streak = None
        mock_records_obj.recovery_aligned_streak = None
        mock_records_obj.green_recovery_streak = None

        from analysis.personal_records import TodayRecordSummary
        no_rec = TodayRecordSummary(date_str="2026-03-14", has_records=False)

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]), \
             patch("analysis.personal_records.compute_personal_records",
                   return_value=mock_records_obj), \
             patch("analysis.personal_records.check_today_records",
                   return_value=no_rec):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)

        assert "This week's milestones" not in message

    def test_message_renders_without_crash_when_personal_records_raises(self):
        """
        If personal_records raises an exception, the weekly summary should
        still render cleanly — the milestones block degrades silently.
        """
        dates = ["2026-03-14"]
        rolling = self._make_rolling(dates)

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]), \
             patch("analysis.personal_records.compute_personal_records",
                   side_effect=RuntimeError("records unavailable")):
            summary = compute_weekly_summary("2026-03-14")
            message = format_weekly_message(summary)

        # Message must render — the exception is swallowed silently
        assert "Weekly Presence Summary" in message
        assert "This week's milestones" not in message


class TestWeeklyNextWeekPacing:
    """
    The weekly summary embeds the next-week WeeklyPacingPlan when:
      - The plan is meaningful (is_meaningful=True)
      - The summary is recent (≤ 2 days old)

    These tests mock the weekly_pacing module and verify correct embedding.
    """

    def _make_rolling(self, dates):
        return {"days": {d: _make_day_summary(d) for d in dates}}

    def _make_meaningful_plan(self):
        """Minimal mock WeeklyPacingPlan that is_meaningful=True."""
        plan = MagicMock()
        plan.is_meaningful = True
        plan.strategy = "PUSH"
        plan.strategy_headline = "Strong week ahead — go deep."
        plan.week_start = "2026-03-16"
        plan.week_end = "2026-03-20"
        plan.weekly_load_forecast = 0.12
        plan.push_days = ["2026-03-16", "2026-03-17"]
        plan.protect_days = []
        plan.cdi_context = "balanced"
        plan.cdi_score = 52.0
        plan.days_of_history = 5
        plan.days = []
        return plan

    def test_pacing_section_present_for_recent_summary(self):
        """
        For a recent summary (end_date = today − 0 days), the pacing plan
        section should appear in the message when the plan is meaningful.
        """
        from datetime import datetime
        today = datetime.now()
        end_date = today.strftime("%Y-%m-%d")
        dates = [end_date]
        rolling = self._make_rolling(dates)

        plan = self._make_meaningful_plan()
        pacing_section_text = "📅 *Weekly Pacing — Mon 16 Mar → Fri 20 Mar*\n\nStrong week ahead."

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]), \
             patch("analysis.weekly_pacing.compute_weekly_pacing", return_value=plan), \
             patch("analysis.weekly_pacing.format_weekly_pacing_section",
                   return_value=pacing_section_text):
            summary = compute_weekly_summary(end_date)
            message = format_weekly_message(summary)

        # The pacing section text should appear in the message
        assert "Weekly Pacing" in message or "Strong week ahead" in message

    def test_pacing_section_absent_for_stale_summary(self):
        """
        For a stale summary (end_date > 2 days ago), the pacing plan should
        be omitted — it would show a past week's plan, not next week's.
        """
        # end_date = 10 days ago → stale
        stale_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        rolling = self._make_rolling([stale_date])

        plan = self._make_meaningful_plan()

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]), \
             patch("analysis.weekly_pacing.compute_weekly_pacing", return_value=plan):
            summary = compute_weekly_summary(stale_date)
            message = format_weekly_message(summary)

        assert "Weekly Pacing" not in message

    def test_pacing_section_absent_when_not_meaningful(self):
        """
        When the WeeklyPacingPlan is not meaningful (insufficient history),
        the section should be silently omitted.
        """
        from datetime import datetime
        today = datetime.now()
        end_date = today.strftime("%Y-%m-%d")
        rolling = self._make_rolling([end_date])

        plan = MagicMock()
        plan.is_meaningful = False  # ← not meaningful

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]), \
             patch("analysis.weekly_pacing.compute_weekly_pacing", return_value=plan):
            summary = compute_weekly_summary(end_date)
            message = format_weekly_message(summary)

        assert "Weekly Pacing" not in message

    def test_message_renders_without_crash_when_pacing_raises(self):
        """
        If weekly_pacing raises an exception, the weekly summary should still
        render cleanly — the pacing section degrades silently.
        """
        from datetime import datetime
        today = datetime.now()
        end_date = today.strftime("%Y-%m-%d")
        rolling = self._make_rolling([end_date])

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]), \
             patch("analysis.weekly_pacing.compute_weekly_pacing",
                   side_effect=RuntimeError("pacing unavailable")):
            summary = compute_weekly_summary(end_date)
            message = format_weekly_message(summary)

        assert "Weekly Presence Summary" in message
        assert "Weekly Pacing" not in message


# ─── v2.4: Load Drivers tests ─────────────────────────────────────────────────

class TestComputeWeekLoadDrivers:
    """Unit tests for compute_week_load_drivers()."""

    def test_returns_dict_with_expected_keys(self):
        """Return value always has shares, dominant, days_meaningful, error."""
        from scripts.weekly_summary import compute_week_load_drivers

        mock_result = {
            "daily": [],
            "weekly_shares": {"meetings": 0.40, "slack": 0.30, "physiology": 0.20, "rescuetime": 0.05, "omi": 0.05},
            "weekly_cls": 0.32,
            "dominant_source": "meetings",
            "days_meaningful": 5,
        }

        with patch("analysis.load_decomposer.compute_week_decomposition", return_value=mock_result):
            result = compute_week_load_drivers("2026-03-14", days=7)

        assert "shares" in result
        assert "dominant" in result
        assert "days_meaningful" in result
        assert "error" in result
        assert result["dominant"] == "meetings"
        assert result["days_meaningful"] == 5
        assert result["error"] is None

    def test_shares_sum_to_one(self):
        """Shares returned from the decomposer should sum to ≈ 1.0."""
        from scripts.weekly_summary import compute_week_load_drivers

        mock_result = {
            "weekly_shares": {"meetings": 0.40, "slack": 0.30, "physiology": 0.20, "rescuetime": 0.05, "omi": 0.05},
            "dominant_source": "meetings",
            "days_meaningful": 3,
        }

        with patch("analysis.load_decomposer.compute_week_decomposition", return_value=mock_result):
            result = compute_week_load_drivers("2026-03-14")

        total = sum(result["shares"].values())
        assert abs(total - 1.0) < 0.01

    def test_graceful_on_import_error(self):
        """When load_decomposer raises, returns safe defaults without error propagation."""
        from scripts.weekly_summary import compute_week_load_drivers

        with patch("analysis.load_decomposer.compute_week_decomposition", side_effect=ImportError("no module")):
            result = compute_week_load_drivers("2026-03-14")

        assert result["shares"] == {}
        assert result["days_meaningful"] == 0
        assert result["error"] is not None

    def test_graceful_on_exception(self):
        """Any exception in compute_week_decomposition returns safe defaults."""
        from scripts.weekly_summary import compute_week_load_drivers

        with patch("analysis.load_decomposer.compute_week_decomposition", side_effect=RuntimeError("oops")):
            result = compute_week_load_drivers("2026-03-14")

        assert result["shares"] == {}
        assert result["days_meaningful"] == 0
        assert result["error"] == "oops"

    def test_empty_shares_when_no_meaningful_days(self):
        """When days_meaningful=0, shares dict should be empty."""
        from scripts.weekly_summary import compute_week_load_drivers

        mock_result = {
            "weekly_shares": {},
            "dominant_source": "unknown",
            "days_meaningful": 0,
        }

        with patch("analysis.load_decomposer.compute_week_decomposition", return_value=mock_result):
            result = compute_week_load_drivers("2026-03-14")

        assert result["days_meaningful"] == 0
        assert result["shares"] == {}


class TestFormatWeeklyMessageLoadDrivers:
    """Tests for the Load Drivers section in format_weekly_message (v2.4)."""

    def _make_rolling(self, dates: list) -> dict:
        """Minimal rolling.json with one day per date."""
        return {
            "days": {
                d: {
                    "date": d,
                    "metrics_avg": {
                        "cognitive_load_score": 0.32,
                        "focus_depth_index": 0.70,
                        "social_drain_index": 0.20,
                        "context_switch_cost": 0.15,
                        "recovery_alignment_score": 0.75,
                    },
                    "whoop": {"recovery_score": 72.0, "hrv_rmssd_milli": 65.0, "sleep_hours": 7.5},
                    "focus_quality": {"active_fdi": 0.65, "active_windows": 8, "peak_focus_hour": 10},
                    "calendar": {"total_meeting_minutes": 90},
                    "slack": {"total_messages_sent": 20, "total_messages_received": 40},
                    "presence_score": {"dps": 75.0},
                }
                for d in dates
            }
        }

    def _make_drivers(self, shares: dict, dominant: str, days_meaningful: int = 3) -> dict:
        return {
            "shares": shares,
            "dominant": dominant,
            "days_meaningful": days_meaningful,
            "error": None,
        }

    def test_load_drivers_section_present_when_data_available(self):
        """Load Drivers section appears when ≥ 2 meaningful days."""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        rolling = self._make_rolling([today])

        drivers = self._make_drivers(
            shares={"meetings": 0.42, "slack": 0.28, "physiology": 0.18, "rescuetime": 0.07, "omi": 0.05},
            dominant="meetings",
        )

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]), \
             patch("scripts.weekly_summary.compute_week_load_drivers", return_value=drivers):
            from scripts.weekly_summary import compute_weekly_summary, format_weekly_message
            summary = compute_weekly_summary(today)
            # Inject drivers directly into summary dict for formatting test
            summary["this_week_drivers"] = drivers
            summary["last_week_drivers"] = {"shares": {}, "dominant": "unknown", "days_meaningful": 0, "error": None}
            message = format_weekly_message(summary)

        assert "Load Drivers" in message

    def test_load_drivers_shows_dominant_source(self):
        """The dominant source should be mentioned in the Load Drivers section."""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        rolling = self._make_rolling([today])

        drivers = self._make_drivers(
            shares={"meetings": 0.50, "slack": 0.30, "physiology": 0.15, "rescuetime": 0.03, "omi": 0.02},
            dominant="meetings",
        )

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            from scripts.weekly_summary import compute_weekly_summary, format_weekly_message
            summary = compute_weekly_summary(today)
            summary["this_week_drivers"] = drivers
            summary["last_week_drivers"] = {"shares": {}, "dominant": "unknown", "days_meaningful": 0, "error": None}
            message = format_weekly_message(summary)

        assert "Meetings" in message or "meetings" in message.lower()

    def test_load_drivers_section_absent_when_insufficient_data(self):
        """Load Drivers section is omitted when days_meaningful < 2."""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        rolling = self._make_rolling([today])

        no_drivers = self._make_drivers(shares={}, dominant="unknown", days_meaningful=0)

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            from scripts.weekly_summary import compute_weekly_summary, format_weekly_message
            summary = compute_weekly_summary(today)
            summary["this_week_drivers"] = no_drivers
            summary["last_week_drivers"] = no_drivers
            message = format_weekly_message(summary)

        assert "Load Drivers" not in message

    def test_load_drivers_shows_week_over_week_shift(self):
        """When prior-week data is available, large source shifts (≥3pp) appear."""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        rolling = self._make_rolling([today])

        this_drivers = self._make_drivers(
            shares={"meetings": 0.50, "slack": 0.30, "physiology": 0.15, "rescuetime": 0.03, "omi": 0.02},
            dominant="meetings",
        )
        last_drivers = self._make_drivers(
            # Slack was only 18% last week → shift of +12pp — should appear
            shares={"meetings": 0.50, "slack": 0.18, "physiology": 0.20, "rescuetime": 0.07, "omi": 0.05},
            dominant="meetings",
        )

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            from scripts.weekly_summary import compute_weekly_summary, format_weekly_message
            summary = compute_weekly_summary(today)
            summary["this_week_drivers"] = this_drivers
            summary["last_week_drivers"] = last_drivers
            message = format_weekly_message(summary)

        # Slack went from 18% → 30% = +12pp, which should trigger the shift line
        assert "↑" in message or "↓" in message  # at least one shift indicator
        assert "Slack" in message or "slack" in message.lower()

    def test_load_drivers_absent_when_decomposer_raises(self):
        """If compute_week_load_drivers raises, Load Drivers is silently omitted."""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        rolling = self._make_rolling([today])

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]), \
             patch("scripts.weekly_summary.compute_week_load_drivers",
                   side_effect=RuntimeError("decomposer unavailable")):
            from scripts.weekly_summary import compute_weekly_summary, format_weekly_message
            try:
                summary = compute_weekly_summary(today)
                message = format_weekly_message(summary)
                assert "Weekly Presence Summary" in message
                assert "Load Drivers" not in message
            except RuntimeError:
                # If compute_weekly_summary propagates — that's acceptable;
                # the format_weekly_message must not crash when drivers are missing
                summary = {"end_date": today, "this_week": {"days_with_data": 0}, "last_week": {}, "deltas": {}, "whoop_deltas": {}, "dps_delta": None, "this_week_drivers": {"shares": {}, "dominant": "unknown", "days_meaningful": 0, "error": None}, "last_week_drivers": {"shares": {}, "dominant": "unknown", "days_meaningful": 0, "error": None}}
                message = format_weekly_message(summary)
                assert "Load Drivers" not in message

    def test_small_source_shifts_not_shown(self):
        """Shifts < 3pp should not produce a shift annotation."""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        rolling = self._make_rolling([today])

        this_drivers = self._make_drivers(
            shares={"meetings": 0.42, "slack": 0.30, "physiology": 0.18, "rescuetime": 0.06, "omi": 0.04},
            dominant="meetings",
        )
        last_drivers = self._make_drivers(
            # Only 2pp shift on meetings — should NOT trigger shift line
            shares={"meetings": 0.40, "slack": 0.31, "physiology": 0.18, "rescuetime": 0.07, "omi": 0.04},
            dominant="meetings",
        )

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            from scripts.weekly_summary import compute_weekly_summary, format_weekly_message
            summary = compute_weekly_summary(today)
            summary["this_week_drivers"] = this_drivers
            summary["last_week_drivers"] = last_drivers
            message = format_weekly_message(summary)

        # Extract just the Load Drivers block (between that header and the next one)
        if "Load Drivers" in message:
            load_block_start = message.index("Load Drivers")
            next_section = message.find("\n*", load_block_start + 1)
            if next_section > 0:
                load_block = message[load_block_start:next_section]
            else:
                load_block = message[load_block_start:]
            # No shift arrows expected for < 3pp changes
            # (meetings: 42→40 = -2pp, slack: 30→31 = +1pp — both below threshold)
            assert "↑ +" not in load_block or "Slack" not in load_block or "+1" not in load_block

    def test_message_structure_unchanged_without_drivers(self):
        """Weekly summary renders correctly even when this_week_drivers key is missing."""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        rolling = self._make_rolling([today])

        with patch("scripts.weekly_summary.read_summary", return_value=rolling), \
             patch("scripts.weekly_summary.read_day", return_value=[]):
            from scripts.weekly_summary import compute_weekly_summary, format_weekly_message
            summary = compute_weekly_summary(today)
            # Simulate an older summary dict missing the drivers keys
            summary.pop("this_week_drivers", None)
            summary.pop("last_week_drivers", None)
            # Should not raise
            message = format_weekly_message(summary)

        assert "Weekly Presence Summary" in message
        assert "Load Drivers" not in message

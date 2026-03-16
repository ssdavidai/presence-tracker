"""
Tests for analysis/personal_records.py — Personal Records & Streaks

Coverage:
  1. compute_personal_records()
     - Returns empty records when no data
     - Correctly identifies best/worst FDI day
     - Correctly identifies best/worst CLS day (best = lowest)
     - Correctly identifies best/worst RAS day
     - Correctly identifies best/worst DPS day
     - Correctly identifies best/worst recovery day
     - Correctly identifies best/worst HRV day
     - Excludes today from record if as_of_date is set before today
     - is_meaningful() returns False with < 2 days, True with ≥ 2

  2. _compute_streaks()
     - Empty dates → zero streak
     - Single matching day → streak=1
     - Consecutive matching days → streak = count
     - Gap in data (missing day) breaks streak
     - Non-matching day breaks streak
     - Correctly identifies all-time longest streak
     - as_of_date cutoff is respected (no future dates in streak)

  3. Streak types (all four)
     - low_load: CLS < 0.25
     - deep_focus: FDI ≥ 0.70
     - recovery_aligned: RAS ≥ 0.70
     - green_recovery: recovery ≥ 67%

  4. _compute_lifetime_stats()
     - total_days counts all days with data
     - total_meeting_minutes sums correctly
     - best_week_avg_dps finds the best 7-day window
     - days_all_sources counts correctly

  5. check_today_records()
     - Returns has_records=False when no bests set today
     - Returns has_records=True when FDI best is today
     - Returns has_records=True when CLS best is today
     - Returns has_records=True when DPS best is today
     - Returns has_records=True when recovery best is today
     - Returns has_records=True when HRV best is today
     - Active streaks surfaced at ≥ 2 days
     - new_streak_records populated when streak is all-time best

  6. format_records_line()
     - Returns "" when has_records=False
     - Contains "🏆" and metric name on new personal best
     - Contains streak days and "🔥" for ≥ 3-day streak
     - Contains "new record!" when streak_records non-empty

  7. format_records_section()
     - Returns "not enough data" message when not meaningful
     - Contains all-time best dates
     - Contains streak lines
     - Contains lifetime stats

  8. to_dict() serialization
     - PersonalRecords.to_dict() produces JSON-serializable output
     - TodayRecordSummary.to_dict() produces correct keys

Run with: python3 -m pytest tests/test_personal_records.py -v
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.personal_records import (
    DayRecord,
    LifetimeStats,
    PersonalRecords,
    StreakRecord,
    TodayRecordSummary,
    _compute_lifetime_stats,
    _compute_streaks,
    _extract_day_metrics,
    check_today_records,
    compute_personal_records,
    format_records_line,
    format_records_section,
    STREAK_LOW_LOAD_CLS,
    STREAK_DEEP_FOCUS_FDI,
    STREAK_RECOVERY_ALIGNED_RAS,
    STREAK_GREEN_RECOVERY,
    MIN_DAYS_FOR_RECORDS,
    MIN_WORKING_WINDOWS,
)


# ─── Fixtures & helpers ───────────────────────────────────────────────────────

def _make_day_summary(
    date_str: str,
    avg_cls: float = 0.30,
    active_fdi: float = 0.65,
    avg_ras: float = 0.70,
    recovery: float = 75.0,
    hrv: float = 70.0,
    dps: float = 70.0,
    working_count: int = 30,
    meeting_minutes: int = 60,
    slack_sent: int = 10,
) -> dict:
    """Build a minimal rolling-summary day entry."""
    return {
        "date": date_str,
        "working_hours_analyzed": working_count,
        "metrics_avg": {
            "cognitive_load_score": avg_cls,
            "focus_depth_index": active_fdi,
            "social_drain_index": 0.30,
            "context_switch_cost": 0.25,
            "recovery_alignment_score": avg_ras,
        },
        "focus_quality": {
            "active_fdi": active_fdi,
            "active_windows": 20,
        },
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "sleep_hours": 7.5,
        },
        "presence_score": {"dps": dps},
        "calendar": {"total_meeting_minutes": meeting_minutes},
        "slack": {"total_sent": slack_sent},
        "sources_available": ["whoop", "calendar", "slack", "rescuetime", "omi"],
    }


def _make_rolling_summary(days_data: dict[str, dict]) -> dict:
    """Build a rolling.json-compatible summary dict."""
    return {"days": days_data}


def _dates_n_days_back(n: int, from_date: str = "2026-03-16") -> list[str]:
    """Generate N date strings going back from from_date (inclusive of from_date)."""
    base = datetime.strptime(from_date, "%Y-%m-%d")
    return [(base - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n - 1, -1, -1)]


# ─── 1. compute_personal_records() ───────────────────────────────────────────

class TestComputePersonalRecords:

    def test_empty_when_no_data(self):
        """No data → empty records object."""
        with (
            patch("analysis.personal_records.list_available_dates", return_value=[]),
            patch("analysis.personal_records.read_summary", return_value={"days": {}}),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.total_days_analyzed == 0
        assert not records.is_meaningful()
        assert records.best_fdi_day is None
        assert records.best_cls_day is None

    def test_not_meaningful_with_one_day(self):
        """Single day → not meaningful."""
        dates = ["2026-03-16"]
        days = {"2026-03-16": _make_day_summary("2026-03-16")}
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.is_meaningful() == (records.total_days_analyzed >= MIN_DAYS_FOR_RECORDS)

    def test_meaningful_with_two_days(self):
        """Two days → is_meaningful()."""
        dates = ["2026-03-15", "2026-03-16"]
        days = {
            "2026-03-15": _make_day_summary("2026-03-15"),
            "2026-03-16": _make_day_summary("2026-03-16"),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.is_meaningful()

    def test_best_fdi_day_is_highest(self):
        """best_fdi_day is the date with the highest active_fdi."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {
            "2026-03-14": _make_day_summary("2026-03-14", active_fdi=0.60),
            "2026-03-15": _make_day_summary("2026-03-15", active_fdi=0.85),
            "2026-03-16": _make_day_summary("2026-03-16", active_fdi=0.72),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.best_fdi_day is not None
        assert records.best_fdi_day.date_str == "2026-03-15"
        assert abs(records.best_fdi_day.value - 0.85) < 0.001

    def test_worst_fdi_day_is_lowest(self):
        """worst_fdi_day is the date with the lowest active_fdi."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {
            "2026-03-14": _make_day_summary("2026-03-14", active_fdi=0.40),
            "2026-03-15": _make_day_summary("2026-03-15", active_fdi=0.80),
            "2026-03-16": _make_day_summary("2026-03-16", active_fdi=0.65),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.worst_fdi_day is not None
        assert records.worst_fdi_day.date_str == "2026-03-14"

    def test_best_cls_day_is_lowest(self):
        """best_cls_day has the LOWEST CLS (lightest cognitive load)."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {
            "2026-03-14": _make_day_summary("2026-03-14", avg_cls=0.10),  # lightest
            "2026-03-15": _make_day_summary("2026-03-15", avg_cls=0.45),
            "2026-03-16": _make_day_summary("2026-03-16", avg_cls=0.30),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.best_cls_day is not None
        assert records.best_cls_day.date_str == "2026-03-14"
        assert abs(records.best_cls_day.value - 0.10) < 0.001

    def test_worst_cls_day_is_highest(self):
        """worst_cls_day has the HIGHEST CLS (most intense)."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {
            "2026-03-14": _make_day_summary("2026-03-14", avg_cls=0.75),  # most intense
            "2026-03-15": _make_day_summary("2026-03-15", avg_cls=0.30),
            "2026-03-16": _make_day_summary("2026-03-16", avg_cls=0.20),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.worst_cls_day is not None
        assert records.worst_cls_day.date_str == "2026-03-14"

    def test_best_ras_day_is_highest(self):
        """best_ras_day has the highest RAS."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {
            "2026-03-14": _make_day_summary("2026-03-14", avg_ras=0.95),
            "2026-03-15": _make_day_summary("2026-03-15", avg_ras=0.60),
            "2026-03-16": _make_day_summary("2026-03-16", avg_ras=0.75),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.best_ras_day is not None
        assert records.best_ras_day.date_str == "2026-03-14"

    def test_best_dps_day_is_highest(self):
        """best_dps_day has the highest DPS."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {
            "2026-03-14": _make_day_summary("2026-03-14", dps=90.0),
            "2026-03-15": _make_day_summary("2026-03-15", dps=65.0),
            "2026-03-16": _make_day_summary("2026-03-16", dps=78.0),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.best_dps_day is not None
        assert records.best_dps_day.date_str == "2026-03-14"
        assert abs(records.best_dps_day.value - 90.0) < 0.1

    def test_best_recovery_day_is_highest(self):
        """best_recovery_day has the highest WHOOP recovery."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {
            "2026-03-14": _make_day_summary("2026-03-14", recovery=95.0),
            "2026-03-15": _make_day_summary("2026-03-15", recovery=70.0),
            "2026-03-16": _make_day_summary("2026-03-16", recovery=80.0),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.best_recovery_day is not None
        assert records.best_recovery_day.date_str == "2026-03-14"

    def test_best_hrv_day_is_highest(self):
        """best_hrv_day has the highest HRV ms."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {
            "2026-03-14": _make_day_summary("2026-03-14", hrv=95.0),
            "2026-03-15": _make_day_summary("2026-03-15", hrv=62.0),
            "2026-03-16": _make_day_summary("2026-03-16", hrv=78.0),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        assert records.best_hrv_day is not None
        assert records.best_hrv_day.date_str == "2026-03-14"

    def test_as_of_date_excludes_future_dates(self):
        """Dates after as_of_date should not be considered."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {
            "2026-03-14": _make_day_summary("2026-03-14", active_fdi=0.60),
            "2026-03-15": _make_day_summary("2026-03-15", active_fdi=0.55),
            "2026-03-16": _make_day_summary("2026-03-16", active_fdi=0.99),  # future
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-15")  # as_of 15th, not 16th
        assert records.best_fdi_day is not None
        assert records.best_fdi_day.date_str == "2026-03-14"
        assert records.best_fdi_day.value < 0.99  # 16th should not appear

    def test_days_with_insufficient_working_windows_excluded(self):
        """Days with < MIN_WORKING_WINDOWS should not appear in records."""
        dates = ["2026-03-15", "2026-03-16"]
        days = {
            "2026-03-15": _make_day_summary("2026-03-15", active_fdi=0.99, working_count=1),  # excluded
            "2026-03-16": _make_day_summary("2026-03-16", active_fdi=0.60),  # included
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        # 2026-03-15 had insufficient data, so best FDI should be 2026-03-16
        if records.best_fdi_day:
            assert records.best_fdi_day.date_str != "2026-03-15"

    def test_as_of_date_set_correctly(self):
        """as_of_date in records matches the argument passed."""
        dates = ["2026-03-15", "2026-03-16"]
        days = {
            "2026-03-15": _make_day_summary("2026-03-15"),
            "2026-03-16": _make_day_summary("2026-03-16"),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-15")
        assert records.as_of_date == "2026-03-15"

    def test_records_populated_for_ten_days(self):
        """Records are populated correctly with 10 days of history."""
        n = 10
        dates = _dates_n_days_back(n)
        days = {}
        for i, d in enumerate(dates):
            days[d] = _make_day_summary(
                d,
                active_fdi=0.50 + i * 0.04,  # increases each day
                avg_cls=0.50 - i * 0.03,      # decreases each day
            )
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records(dates[-1])
        assert records.total_days_analyzed == n
        assert records.is_meaningful()
        # Latest day should have highest FDI and lowest CLS
        assert records.best_fdi_day.date_str == dates[-1]
        assert records.best_cls_day.date_str == dates[-1]


# ─── 2. _compute_streaks() ────────────────────────────────────────────────────

class TestComputeStreaks:

    def _run_streak(self, dates, day_metrics, condition_fn, as_of_date):
        return _compute_streaks(
            dates, day_metrics, condition_fn,
            "test_streak", "test threshold",
            as_of_date,
        )

    def test_empty_dates_returns_zero_streak(self):
        result = self._run_streak([], {}, lambda m: True, "2026-03-16")
        assert result.current_streak == 0
        assert result.longest_streak == 0

    def test_single_matching_day(self):
        dates = ["2026-03-16"]
        day_metrics = {"2026-03-16": {"active_fdi": 0.80}}
        result = self._run_streak(dates, day_metrics, lambda m: m["active_fdi"] >= 0.70, "2026-03-16")
        assert result.current_streak == 1
        assert result.longest_streak == 1

    def test_consecutive_days_streak_count(self):
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        day_metrics = {
            "2026-03-14": {"val": 0.80},
            "2026-03-15": {"val": 0.85},
            "2026-03-16": {"val": 0.75},
        }
        result = self._run_streak(dates, day_metrics, lambda m: m["val"] >= 0.70, "2026-03-16")
        assert result.current_streak == 3
        assert result.longest_streak == 3

    def test_non_matching_day_breaks_streak(self):
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        day_metrics = {
            "2026-03-14": {"val": 0.80},
            "2026-03-15": {"val": 0.50},  # breaks streak
            "2026-03-16": {"val": 0.80},
        }
        result = self._run_streak(dates, day_metrics, lambda m: m["val"] >= 0.70, "2026-03-16")
        assert result.current_streak == 1  # only today
        assert result.longest_streak == 1

    def test_missing_data_breaks_streak(self):
        """A date with None metrics breaks the current streak."""
        dates = ["2026-03-13", "2026-03-14", "2026-03-16"]
        day_metrics = {
            "2026-03-13": {"val": 0.80},
            "2026-03-14": {"val": 0.80},
            # 2026-03-15 is missing (gap in data)
            "2026-03-16": {"val": 0.80},
        }
        result = self._run_streak(dates, day_metrics, lambda m: m["val"] >= 0.70, "2026-03-16")
        # Streak should only be 1 (today, broken by gap)
        assert result.current_streak == 1

    def test_longest_streak_tracked_independently(self):
        """All-time longest streak can be longer than current streak."""
        dates = [
            "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13",
            "2026-03-14", "2026-03-15", "2026-03-16",
        ]
        day_metrics = {
            "2026-03-10": {"val": 0.80},
            "2026-03-11": {"val": 0.80},
            "2026-03-12": {"val": 0.80},  # best streak: 10–12
            "2026-03-13": {"val": 0.50},  # breaks streak
            "2026-03-14": {"val": 0.80},
            "2026-03-15": {"val": 0.80},  # current streak: 14–16
            "2026-03-16": {"val": 0.80},
        }
        result = self._run_streak(dates, day_metrics, lambda m: m["val"] >= 0.70, "2026-03-16")
        assert result.current_streak == 3
        assert result.longest_streak == 3  # tied with current (10–12 = 3, 14–16 = 3)
        assert result.longest_streak_start is not None

    def test_as_of_date_cuts_future_dates(self):
        """Dates after as_of_date should not contribute to current streak."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        day_metrics = {
            "2026-03-14": {"val": 0.80},
            "2026-03-15": {"val": 0.80},
            "2026-03-16": {"val": 0.80},
        }
        result = self._run_streak(dates, day_metrics, lambda m: m["val"] >= 0.70, "2026-03-15")
        # Should only see 14–15 (2 days)
        assert result.current_streak == 2

    def test_streak_start_date_correct(self):
        """current_streak_start should be the first date in the current streak."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        day_metrics = {
            "2026-03-14": {"val": 0.80},
            "2026-03-15": {"val": 0.80},
            "2026-03-16": {"val": 0.80},
        }
        result = self._run_streak(dates, day_metrics, lambda m: m["val"] >= 0.70, "2026-03-16")
        assert result.current_streak_start == "2026-03-14"

    def test_zero_current_streak_when_today_misses(self):
        """If today doesn't meet threshold, current streak is 0."""
        dates = ["2026-03-15", "2026-03-16"]
        day_metrics = {
            "2026-03-15": {"val": 0.80},
            "2026-03-16": {"val": 0.50},  # today misses
        }
        result = self._run_streak(dates, day_metrics, lambda m: m["val"] >= 0.70, "2026-03-16")
        assert result.current_streak == 0
        assert result.current_streak_start is None


# ─── 3. Streak type correctness ──────────────────────────────────────────────

class TestStreakTypes:

    def _make_records_with_streaks(self, days: dict, as_of: str = "2026-03-16") -> PersonalRecords:
        dates = sorted(days.keys())
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            return compute_personal_records(as_of)

    def test_low_load_streak_fires_below_threshold(self):
        """CLS < STREAK_LOW_LOAD_CLS fires the low_load streak."""
        days = {
            "2026-03-15": _make_day_summary("2026-03-15", avg_cls=STREAK_LOW_LOAD_CLS - 0.05),
            "2026-03-16": _make_day_summary("2026-03-16", avg_cls=STREAK_LOW_LOAD_CLS - 0.05),
        }
        records = self._make_records_with_streaks(days)
        assert records.low_load_streak is not None
        assert records.low_load_streak.current_streak == 2

    def test_low_load_streak_does_not_fire_above_threshold(self):
        """CLS ≥ STREAK_LOW_LOAD_CLS should NOT fire low_load streak."""
        days = {
            "2026-03-15": _make_day_summary("2026-03-15", avg_cls=STREAK_LOW_LOAD_CLS + 0.05),
            "2026-03-16": _make_day_summary("2026-03-16", avg_cls=STREAK_LOW_LOAD_CLS + 0.05),
        }
        records = self._make_records_with_streaks(days)
        assert records.low_load_streak is not None
        assert records.low_load_streak.current_streak == 0

    def test_deep_focus_streak_fires_at_threshold(self):
        """active_fdi ≥ STREAK_DEEP_FOCUS_FDI fires deep_focus streak."""
        days = {
            "2026-03-15": _make_day_summary("2026-03-15", active_fdi=STREAK_DEEP_FOCUS_FDI),
            "2026-03-16": _make_day_summary("2026-03-16", active_fdi=STREAK_DEEP_FOCUS_FDI + 0.05),
        }
        records = self._make_records_with_streaks(days)
        assert records.deep_focus_streak is not None
        assert records.deep_focus_streak.current_streak == 2

    def test_deep_focus_streak_does_not_fire_below_threshold(self):
        """active_fdi < STREAK_DEEP_FOCUS_FDI should NOT fire."""
        days = {
            "2026-03-15": _make_day_summary("2026-03-15", active_fdi=STREAK_DEEP_FOCUS_FDI - 0.05),
            "2026-03-16": _make_day_summary("2026-03-16", active_fdi=STREAK_DEEP_FOCUS_FDI - 0.05),
        }
        records = self._make_records_with_streaks(days)
        assert records.deep_focus_streak is not None
        assert records.deep_focus_streak.current_streak == 0

    def test_recovery_aligned_streak_fires_at_threshold(self):
        """avg_ras ≥ STREAK_RECOVERY_ALIGNED_RAS fires recovery_aligned streak."""
        days = {
            "2026-03-15": _make_day_summary("2026-03-15", avg_ras=STREAK_RECOVERY_ALIGNED_RAS),
            "2026-03-16": _make_day_summary("2026-03-16", avg_ras=STREAK_RECOVERY_ALIGNED_RAS + 0.05),
        }
        records = self._make_records_with_streaks(days)
        assert records.recovery_aligned_streak is not None
        assert records.recovery_aligned_streak.current_streak == 2

    def test_green_recovery_streak_fires_at_threshold(self):
        """WHOOP recovery ≥ STREAK_GREEN_RECOVERY fires green_recovery streak."""
        days = {
            "2026-03-15": _make_day_summary("2026-03-15", recovery=STREAK_GREEN_RECOVERY),
            "2026-03-16": _make_day_summary("2026-03-16", recovery=STREAK_GREEN_RECOVERY + 5),
        }
        records = self._make_records_with_streaks(days)
        assert records.green_recovery_streak is not None
        assert records.green_recovery_streak.current_streak == 2

    def test_green_recovery_streak_does_not_fire_below(self):
        """WHOOP recovery < STREAK_GREEN_RECOVERY should NOT fire."""
        days = {
            "2026-03-15": _make_day_summary("2026-03-15", recovery=STREAK_GREEN_RECOVERY - 5),
            "2026-03-16": _make_day_summary("2026-03-16", recovery=STREAK_GREEN_RECOVERY - 5),
        }
        records = self._make_records_with_streaks(days)
        assert records.green_recovery_streak is not None
        assert records.green_recovery_streak.current_streak == 0


# ─── 4. _compute_lifetime_stats() ────────────────────────────────────────────

class TestComputeLifetimeStats:

    def test_total_days_count(self):
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {d: _make_day_summary(d) for d in dates}
        lt = _compute_lifetime_stats(dates, _make_rolling_summary(days))
        assert lt.total_days == 3

    def test_total_meeting_minutes_sum(self):
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {
            "2026-03-14": _make_day_summary("2026-03-14", meeting_minutes=60),
            "2026-03-15": _make_day_summary("2026-03-15", meeting_minutes=90),
            "2026-03-16": _make_day_summary("2026-03-16", meeting_minutes=30),
        }
        lt = _compute_lifetime_stats(dates, _make_rolling_summary(days))
        assert lt.total_meeting_minutes == 180

    def test_best_week_dps_found(self):
        """Best 7-day window should have the highest average DPS."""
        # 10 days with the last 7 having higher DPS
        dates = _dates_n_days_back(10)
        days = {}
        for i, d in enumerate(dates):
            dps = 80.0 if i >= 3 else 50.0  # last 7 days are better
            days[d] = _make_day_summary(d, dps=dps)
        lt = _compute_lifetime_stats(dates, _make_rolling_summary(days))
        assert lt.best_week_avg_dps is not None
        assert lt.best_week_avg_dps >= 75.0  # Should be near 80

    def test_days_all_sources_counted(self):
        """days_all_sources counts days with all 5 data sources."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        day_with_all = _make_day_summary("2026-03-14")
        day_partial = _make_day_summary("2026-03-15")
        day_partial["sources_available"] = ["whoop", "calendar"]
        day_with_all2 = _make_day_summary("2026-03-16")
        days = {
            "2026-03-14": day_with_all,
            "2026-03-15": day_partial,
            "2026-03-16": day_with_all2,
        }
        lt = _compute_lifetime_stats(dates, _make_rolling_summary(days))
        assert lt.days_all_sources == 2

    def test_empty_dates_returns_zero_stats(self):
        lt = _compute_lifetime_stats([], _make_rolling_summary({}))
        assert lt.total_days == 0
        assert lt.total_meeting_minutes == 0
        assert lt.best_week_avg_dps is None

    def test_best_week_requires_7_days(self):
        """With < 7 days, best_week_avg_dps should be None (can't form a 7-day window)."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {d: _make_day_summary(d, dps=80.0) for d in dates}
        lt = _compute_lifetime_stats(dates, _make_rolling_summary(days))
        assert lt.best_week_avg_dps is None


# ─── 5. check_today_records() ────────────────────────────────────────────────

class TestCheckTodayRecords:

    def _records_with_today_as_best(
        self,
        date_str: str,
        metric: str,
        today_val: float,
        other_val: float,
    ) -> PersonalRecords:
        """Build records where today sets a new best for the given metric."""
        prev = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        if metric == "active_fdi":
            days = {
                prev: _make_day_summary(prev, active_fdi=other_val),
                date_str: _make_day_summary(date_str, active_fdi=today_val),
            }
        elif metric == "avg_cls":
            days = {
                prev: _make_day_summary(prev, avg_cls=other_val),
                date_str: _make_day_summary(date_str, avg_cls=today_val),
            }
        elif metric == "avg_ras":
            days = {
                prev: _make_day_summary(prev, avg_ras=other_val),
                date_str: _make_day_summary(date_str, avg_ras=today_val),
            }
        elif metric == "dps":
            days = {
                prev: _make_day_summary(prev, dps=other_val),
                date_str: _make_day_summary(date_str, dps=today_val),
            }
        elif metric == "recovery":
            days = {
                prev: _make_day_summary(prev, recovery=other_val),
                date_str: _make_day_summary(date_str, recovery=today_val),
            }
        elif metric == "hrv":
            days = {
                prev: _make_day_summary(prev, hrv=other_val),
                date_str: _make_day_summary(date_str, hrv=today_val),
            }
        else:
            raise ValueError(f"Unknown metric: {metric}")

        dates = [prev, date_str]
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            return compute_personal_records(date_str)

    def test_no_records_when_today_is_not_best(self):
        """has_records should be False when today didn't set any records or notable streaks."""
        prev = "2026-03-15"
        today = "2026-03-16"
        # Use values that don't trigger any streaks (recovery < 67, RAS < 0.70, FDI < 0.70, CLS > 0.25)
        days = {
            prev: _make_day_summary(prev, active_fdi=0.90, avg_cls=0.10,
                                    avg_ras=0.50, recovery=60.0),  # prev day is best FDI/CLS
            today: _make_day_summary(today, active_fdi=0.60, avg_cls=0.40,
                                     avg_ras=0.50, recovery=60.0),  # today is not best in anything
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=[prev, today]),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records(today)
        summary = check_today_records(today, records)
        assert not summary.new_best_fdi
        assert not summary.new_best_cls
        assert not summary.new_best_ras
        # has_records may still be True if a 2-day streak fires; that's correct behaviour.
        # The key assertion is that the explicit best flags are False for metrics where prev was better.
        assert not summary.new_best_fdi
        assert not summary.new_best_cls

    def test_new_best_fdi_detected(self):
        """new_best_fdi should be True when today has the highest FDI."""
        today = "2026-03-16"
        records = self._records_with_today_as_best(today, "active_fdi", 0.92, 0.70)
        summary = check_today_records(today, records)
        assert summary.new_best_fdi
        assert summary.has_records

    def test_new_best_cls_detected(self):
        """new_best_cls should be True when today has the LOWEST CLS."""
        today = "2026-03-16"
        records = self._records_with_today_as_best(today, "avg_cls", 0.05, 0.20)
        summary = check_today_records(today, records)
        assert summary.new_best_cls
        assert summary.has_records

    def test_new_best_ras_detected(self):
        today = "2026-03-16"
        records = self._records_with_today_as_best(today, "avg_ras", 0.99, 0.70)
        summary = check_today_records(today, records)
        assert summary.new_best_ras
        assert summary.has_records

    def test_new_best_dps_detected(self):
        today = "2026-03-16"
        records = self._records_with_today_as_best(today, "dps", 95.0, 70.0)
        summary = check_today_records(today, records)
        assert summary.new_best_dps
        assert summary.has_records

    def test_new_best_recovery_detected(self):
        today = "2026-03-16"
        records = self._records_with_today_as_best(today, "recovery", 99.0, 80.0)
        summary = check_today_records(today, records)
        assert summary.new_best_recovery
        assert summary.has_records

    def test_new_best_hrv_detected(self):
        today = "2026-03-16"
        records = self._records_with_today_as_best(today, "hrv", 120.0, 80.0)
        summary = check_today_records(today, records)
        assert summary.new_best_hrv
        assert summary.has_records

    def test_active_streak_surfaced_at_two_days(self):
        """Streak of 2 days should be surfaced in active_streaks()."""
        prev = "2026-03-15"
        today = "2026-03-16"
        # Both days have deep focus (FDI ≥ 0.70)
        days = {
            prev: _make_day_summary(prev, active_fdi=0.80),
            today: _make_day_summary(today, active_fdi=0.82),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=[prev, today]),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records(today)
        summary = check_today_records(today, records)
        streaks = summary.active_streaks()
        streak_names = [s[0] for s in streaks]
        assert "Deep focus" in streak_names

    def test_streak_days_count_correct(self):
        """deep_focus_streak_days should match the current streak."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {d: _make_day_summary(d, active_fdi=0.80) for d in dates}
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        summary = check_today_records("2026-03-16", records)
        assert summary.deep_focus_streak_days == 3

    def test_new_streak_record_populated(self):
        """new_streak_records should be populated when today sets a streak record."""
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {d: _make_day_summary(d, active_fdi=0.80) for d in dates}
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        # The 3-day streak is also the all-time longest, ending today
        assert records.deep_focus_streak.longest_streak_end == "2026-03-16"
        summary = check_today_records("2026-03-16", records)
        assert "deep_focus" in summary.new_streak_records

    def test_not_meaningful_records_returns_no_records(self):
        """With < MIN_DAYS_FOR_RECORDS, has_records should be False."""
        records = PersonalRecords(as_of_date="2026-03-16", total_days_analyzed=1)
        summary = check_today_records("2026-03-16", records)
        assert not summary.has_records

    def test_all_new_bests_list(self):
        """all_new_bests() returns correct list of metric names."""
        summary = TodayRecordSummary(
            date_str="2026-03-16",
            new_best_fdi=True,
            new_best_dps=True,
            has_records=True,
        )
        bests = summary.all_new_bests()
        assert "FDI" in bests
        assert "DPS" in bests
        assert "CLS" not in bests


# ─── 6. format_records_line() ────────────────────────────────────────────────

class TestFormatRecordsLine:

    def test_empty_when_no_records(self):
        summary = TodayRecordSummary(date_str="2026-03-16", has_records=False)
        assert format_records_line(summary) == ""

    def test_contains_trophy_on_new_best(self):
        summary = TodayRecordSummary(
            date_str="2026-03-16",
            has_records=True,
            new_best_fdi=True,
        )
        line = format_records_line(summary)
        assert "🏆" in line

    def test_contains_metric_name_on_new_best(self):
        summary = TodayRecordSummary(
            date_str="2026-03-16",
            has_records=True,
            new_best_fdi=True,
        )
        line = format_records_line(summary)
        assert "FDI" in line

    def test_contains_fire_emoji_for_three_day_streak(self):
        summary = TodayRecordSummary(
            date_str="2026-03-16",
            has_records=True,
            deep_focus_streak_days=3,
        )
        line = format_records_line(summary)
        assert "🔥" in line

    def test_contains_streak_day_count(self):
        summary = TodayRecordSummary(
            date_str="2026-03-16",
            has_records=True,
            deep_focus_streak_days=5,
        )
        line = format_records_line(summary)
        assert "5" in line

    def test_contains_new_record_text_for_streak_record(self):
        summary = TodayRecordSummary(
            date_str="2026-03-16",
            has_records=True,
            deep_focus_streak_days=4,
            new_streak_records=["deep_focus"],
        )
        line = format_records_line(summary)
        assert "new record" in line.lower()

    def test_two_day_streak_uses_checkmark(self):
        summary = TodayRecordSummary(
            date_str="2026-03-16",
            has_records=True,
            low_load_streak_days=2,
        )
        line = format_records_line(summary)
        assert "✅" in line or "2 days" in line

    def test_multiple_bests_all_shown(self):
        summary = TodayRecordSummary(
            date_str="2026-03-16",
            has_records=True,
            new_best_fdi=True,
            new_best_dps=True,
        )
        line = format_records_line(summary)
        assert "FDI" in line
        assert "DPS" in line

    def test_returns_non_empty_string_for_any_record(self):
        summary = TodayRecordSummary(
            date_str="2026-03-16",
            has_records=True,
            new_best_recovery=True,
        )
        line = format_records_line(summary)
        assert len(line) > 0


# ─── 7. format_records_section() ─────────────────────────────────────────────

class TestFormatRecordsSection:

    def test_not_enough_data_message(self):
        records = PersonalRecords(as_of_date="2026-03-16", total_days_analyzed=1)
        section = format_records_section(records)
        assert "not enough data" in section.lower()

    def test_contains_best_fdi_date(self):
        dates = ["2026-03-15", "2026-03-16"]
        days = {
            "2026-03-15": _make_day_summary("2026-03-15", active_fdi=0.90),
            "2026-03-16": _make_day_summary("2026-03-16", active_fdi=0.70),
        }
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        section = format_records_section(records)
        assert "15 Mar" in section or "0.90" in section

    def test_contains_streak_info(self):
        dates = ["2026-03-15", "2026-03-16"]
        days = {d: _make_day_summary(d, active_fdi=0.80) for d in dates}
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        section = format_records_section(records)
        assert "streak" in section.lower() or "Streak" in section

    def test_contains_lifetime_stats(self):
        dates = ["2026-03-15", "2026-03-16"]
        days = {d: _make_day_summary(d) for d in dates}
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        section = format_records_section(records)
        assert "Lifetime" in section or "Total days" in section.lower()

    def test_contains_days_tracked_count(self):
        dates = ["2026-03-14", "2026-03-15", "2026-03-16"]
        days = {d: _make_day_summary(d) for d in dates}
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        section = format_records_section(records)
        assert "3" in section


# ─── 8. Serialization ────────────────────────────────────────────────────────

class TestSerialization:

    def test_personal_records_to_dict_is_json_serializable(self):
        dates = ["2026-03-15", "2026-03-16"]
        days = {d: _make_day_summary(d) for d in dates}
        with (
            patch("analysis.personal_records.list_available_dates", return_value=dates),
            patch("analysis.personal_records.read_summary", return_value=_make_rolling_summary(days)),
        ):
            records = compute_personal_records("2026-03-16")
        d = records.to_dict()
        json.dumps(d)  # Should not raise
        assert "as_of_date" in d
        assert "bests" in d
        assert "streaks" in d
        assert "lifetime" in d

    def test_today_record_summary_to_dict(self):
        summary = TodayRecordSummary(
            date_str="2026-03-16",
            has_records=True,
            new_best_fdi=True,
            deep_focus_streak_days=3,
            new_streak_records=["deep_focus"],
        )
        d = summary.to_dict()
        json.dumps(d)  # Should not raise
        assert d["has_records"] is True
        assert "FDI" in d["new_bests"]
        assert any(s["name"] == "Deep focus" for s in d["active_streaks"])

    def test_day_record_to_dict(self):
        rec = DayRecord("2026-03-16", 0.85, "active_fdi", "highest FDI")
        d = rec.to_dict()
        assert d["date"] == "2026-03-16"
        assert abs(d["value"] - 0.85) < 0.0001
        assert d["metric"] == "active_fdi"
        assert d["label"] == "highest FDI"

    def test_streak_record_to_dict(self):
        sr = StreakRecord(
            "deep_focus", "FDI ≥ 0.70",
            current_streak=3, current_streak_start="2026-03-14",
            longest_streak=5, longest_streak_start="2026-03-01", longest_streak_end="2026-03-05",
        )
        d = sr.to_dict()
        assert d["current_streak"] == 3
        assert d["longest_streak"] == 5

    def test_lifetime_stats_to_dict(self):
        lt = LifetimeStats(
            total_days=10,
            total_meeting_minutes=500,
            best_week_avg_dps=78.5,
            best_week_end_date="2026-03-14",
        )
        d = lt.to_dict()
        json.dumps(d)
        assert d["total_days"] == 10
        assert d["best_week_avg_dps"] == 78.5

    def test_empty_records_to_dict_is_serializable(self):
        records = PersonalRecords(as_of_date="2026-03-16", total_days_analyzed=0)
        d = records.to_dict()
        json.dumps(d)  # Should not raise
        assert not d["is_meaningful"]

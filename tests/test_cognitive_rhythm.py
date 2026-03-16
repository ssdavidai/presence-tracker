"""
Tests for analysis/cognitive_rhythm.py — Cognitive Rhythm Analysis

Coverage:
  - compute_cognitive_rhythm() with no data → is_meaningful=False
  - compute_cognitive_rhythm() with sufficient data → is_meaningful=True
  - Hourly profile: correct hour bucketing, avg_cls/fdi computation
  - Day-of-week profile: correct weekday aggregation
  - peak_focus_hours: top-n hours by avg_fdi
  - low_load_hours: top-n hours by lowest avg_cls
  - morning_vs_afternoon bias detection
  - Sparklines: correct length and character set
  - format_rhythm_line() output format
  - format_rhythm_section() output contains expected components
  - format_rhythm_terminal() output format
  - Graceful degradation with partially missing data
  - to_dict() serialisation

Run with: python3 -m pytest tests/test_cognitive_rhythm.py -v
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.cognitive_rhythm import (
    CognitiveRhythm,
    DowProfile,
    HourlyProfile,
    MIN_DAYS_FOR_RHYTHM,
    SPARK_CHARS,
    _build_dow_profile,
    _build_hourly_profile,
    _build_sparkline,
    _extract_low_load_hours,
    _extract_peak_focus_hours,
    _hour_label,
    _mean,
    _morning_vs_afternoon,
    _spark_bar,
    compute_cognitive_rhythm,
    format_rhythm_line,
    format_rhythm_section,
    format_rhythm_terminal,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_window(
    date_str: str,
    hour: int,
    cls_val: float = 0.30,
    fdi_val: float = 0.70,
    is_working: bool = True,
    in_meeting: bool = False,
    slack_messages: int = 0,
) -> dict:
    """Build a minimal JSONL window dict for testing."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour, minute=0)
    dow_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
               4: "Friday", 5: "Saturday", 6: "Sunday"}
    return {
        "date": date_str,
        "window_start": dt.isoformat(),
        "metadata": {
            "hour_of_day": hour,
            "day_of_week": dow_map[dt.weekday()],
            "is_working_hours": is_working,
        },
        "metrics": {
            "cognitive_load_score": cls_val,
            "focus_depth_index": fdi_val,
            "social_drain_index": 0.1,
            "context_switch_cost": 0.1,
            "recovery_alignment_score": 0.8,
        },
        "calendar": {"in_meeting": in_meeting, "events": []},
        "slack": {"total_messages": slack_messages},
        "whoop": {"recovery_score": 80.0, "hrv_rmssd_milli": 70.0},
    }


def _make_summary_days(*dates_and_cls: tuple) -> dict:
    """Build a minimal rolling summary dict for the given (date, cls, fdi) tuples."""
    days = {}
    for date_str, cls_val, fdi_val in dates_and_cls:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        days[date_str] = {
            "date": date_str,
            "metrics_avg": {
                "cognitive_load_score": cls_val,
                "focus_depth_index": fdi_val,
                "social_drain_index": 0.1,
                "context_switch_cost": 0.1,
                "recovery_alignment_score": 0.8,
            },
            "whoop": {"recovery_score": 80.0, "hrv_rmssd_milli": 70.0},
            "calendar": {"total_meeting_minutes": 60},
        }
    return {"days": days}


# ─── Unit tests: helpers ──────────────────────────────────────────────────────

class TestMean:
    def test_basic(self):
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_empty(self):
        assert _mean([]) is None

    def test_none_filtered(self):
        assert _mean([1.0, None, 3.0]) == pytest.approx(2.0)  # type: ignore


class TestSparkBar:
    def test_min_value_gives_first_char(self):
        result = _spark_bar(0.0, 0.0, 1.0)
        assert result == SPARK_CHARS[0]

    def test_max_value_gives_last_char(self):
        result = _spark_bar(1.0, 0.0, 1.0)
        assert result == SPARK_CHARS[-1]

    def test_mid_value(self):
        result = _spark_bar(0.5, 0.0, 1.0)
        # Should be near the middle char
        assert result in SPARK_CHARS[2:7]

    def test_equal_min_max_returns_middle(self):
        result = _spark_bar(0.5, 0.5, 0.5)
        assert result in SPARK_CHARS  # shouldn't crash


class TestBuildSparkline:
    def test_correct_length(self):
        values = [0.1, 0.5, 0.9, 0.3, None]
        result = _build_sparkline(values)
        assert len(result) == 5

    def test_none_renders_as_dot(self):
        # None values (including all-None) render as '·'
        values = [None]
        result = _build_sparkline(values)
        assert result == "·"

    def test_none_in_mixed_list_renders_as_dot(self):
        # None in a list with real values also renders as '·'
        values = [0.5, None, 0.5]
        result = _build_sparkline(values)
        assert result[1] == "·"

    def test_all_same_values(self):
        values = [0.5, 0.5, 0.5]
        result = _build_sparkline(values)
        assert len(result) == 3
        assert all(c in SPARK_CHARS for c in result)

    def test_invert_flips_ordering(self):
        values = [0.2, 0.8]  # high → should map to low block when inverted
        normal = _build_sparkline(values, invert=False)
        inverted = _build_sparkline(values, invert=True)
        # In inverted mode, 0.8 (second) should be smaller than 0.2 (first)
        assert SPARK_CHARS.index(inverted[1]) < SPARK_CHARS.index(inverted[0])


class TestHourLabel:
    def test_midnight(self):
        assert _hour_label(0) == "12am"

    def test_morning(self):
        assert _hour_label(9) == "9am"

    def test_noon(self):
        assert _hour_label(12) == "12pm"

    def test_afternoon(self):
        assert _hour_label(14) == "2pm"

    def test_evening(self):
        assert _hour_label(20) == "8pm"


# ─── Unit tests: hourly profile ───────────────────────────────────────────────

class TestBuildHourlyProfile:
    def test_basic_aggregation(self):
        dates = ["2026-03-13"]
        windows = [
            _make_window("2026-03-13", 9, cls_val=0.20, fdi_val=0.80),
            _make_window("2026-03-13", 9, cls_val=0.30, fdi_val=0.70),
            _make_window("2026-03-13", 10, cls_val=0.40, fdi_val=0.60),
        ]
        with patch("analysis.cognitive_rhythm.read_day", return_value=windows):
            profiles = _build_hourly_profile(dates)

        hour9 = next((p for p in profiles if p.hour == 9), None)
        assert hour9 is not None
        assert hour9.avg_cls == pytest.approx(0.25, abs=0.001)
        assert hour9.avg_fdi == pytest.approx(0.75, abs=0.001)
        assert hour9.window_count == 2

    def test_non_working_hours_excluded(self):
        dates = ["2026-03-13"]
        windows = [
            _make_window("2026-03-13", 9, cls_val=0.30, fdi_val=0.70, is_working=True),
            _make_window("2026-03-13", 9, cls_val=0.99, fdi_val=0.01, is_working=False),
        ]
        with patch("analysis.cognitive_rhythm.read_day", return_value=windows):
            profiles = _build_hourly_profile(dates)

        hour9 = next((p for p in profiles if p.hour == 9), None)
        assert hour9 is not None
        # Non-working window should be excluded
        assert hour9.avg_cls == pytest.approx(0.30, abs=0.001)
        assert hour9.window_count == 1

    def test_empty_windows_returns_profiles_with_none(self):
        dates = ["2026-03-13"]
        with patch("analysis.cognitive_rhythm.read_day", return_value=[]):
            profiles = _build_hourly_profile(dates)

        # Should still return profiles for all working hours
        assert len(profiles) > 0
        # All should have zero or None values
        assert all(p.avg_cls is None or p.window_count == 0 for p in profiles)

    def test_read_day_exception_graceful(self):
        dates = ["2026-03-13"]
        with patch("analysis.cognitive_rhythm.read_day", side_effect=Exception("IO Error")):
            profiles = _build_hourly_profile(dates)
        # Should return empty profiles without crashing
        assert isinstance(profiles, list)


# ─── Unit tests: DOW profile ──────────────────────────────────────────────────

class TestBuildDowProfile:
    def test_basic_day_aggregation(self):
        # 2026-03-13 is a Friday (dow=4)
        # 2026-03-14 is a Saturday (dow=5)
        dates = ["2026-03-13", "2026-03-14"]
        summary = _make_summary_days(
            ("2026-03-13", 0.30, 0.75),
            ("2026-03-14", 0.40, 0.65),
        )
        profiles = _build_dow_profile(dates, summary)

        assert len(profiles) == 7  # Mon–Sun

        # Friday (dow=4)
        fri = profiles[4]
        assert fri.label == "Fri"
        assert fri.avg_cls == pytest.approx(0.30, abs=0.001)
        assert fri.avg_fdi == pytest.approx(0.75, abs=0.001)
        assert fri.day_count == 1

        # Saturday (dow=5)
        sat = profiles[5]
        assert sat.label == "Sat"
        assert sat.avg_cls == pytest.approx(0.40, abs=0.001)

    def test_empty_dates_returns_empty_profiles(self):
        profiles = _build_dow_profile([], {})
        assert len(profiles) == 7
        assert all(p.day_count == 0 for p in profiles)

    def test_multiple_same_weekday_averages(self):
        # Two Fridays
        dates = ["2026-03-13", "2026-03-20"]
        summary = _make_summary_days(
            ("2026-03-13", 0.20, 0.80),
            ("2026-03-20", 0.40, 0.60),
        )
        profiles = _build_dow_profile(dates, summary)
        fri = profiles[4]
        assert fri.avg_cls == pytest.approx(0.30, abs=0.001)
        assert fri.avg_fdi == pytest.approx(0.70, abs=0.001)
        assert fri.day_count == 2

    def test_missing_day_from_summary_graceful(self):
        dates = ["2026-03-13"]
        # No matching entry in summary
        summary = {"days": {}}
        profiles = _build_dow_profile(dates, summary)
        assert len(profiles) == 7
        fri = profiles[4]
        assert fri.day_count == 1
        assert fri.avg_cls is None  # no data in summary


# ─── Unit tests: insights ─────────────────────────────────────────────────────

class TestExtractPeakFocusHours:
    def test_returns_top_n_by_fdi(self):
        hourly = [
            HourlyProfile(hour=9,  avg_cls=0.3, avg_fdi=0.90, window_count=5),
            HourlyProfile(hour=10, avg_cls=0.3, avg_fdi=0.80, window_count=5),
            HourlyProfile(hour=11, avg_cls=0.3, avg_fdi=0.70, window_count=5),
            HourlyProfile(hour=12, avg_cls=0.3, avg_fdi=0.60, window_count=5),
        ]
        result = _extract_peak_focus_hours(hourly, n=2)
        assert result == [9, 10]

    def test_filters_low_confidence_hours(self):
        hourly = [
            HourlyProfile(hour=9,  avg_cls=0.3, avg_fdi=0.95, window_count=1),  # low count
            HourlyProfile(hour=10, avg_cls=0.3, avg_fdi=0.80, window_count=5),  # ok
        ]
        result = _extract_peak_focus_hours(hourly, n=2)
        # Hour 9 excluded due to window_count < MIN_WINDOWS_PER_HOUR (2)
        assert 9 not in result
        assert 10 in result

    def test_returns_empty_when_no_data(self):
        hourly = [
            HourlyProfile(hour=9, avg_cls=None, avg_fdi=None, window_count=0),
        ]
        result = _extract_peak_focus_hours(hourly)
        assert result == []


class TestExtractLowLoadHours:
    def test_returns_lowest_cls_hours(self):
        hourly = [
            HourlyProfile(hour=14, avg_cls=0.10, avg_fdi=0.6, window_count=5),
            HourlyProfile(hour=9,  avg_cls=0.30, avg_fdi=0.9, window_count=5),
            HourlyProfile(hour=15, avg_cls=0.15, avg_fdi=0.6, window_count=5),
        ]
        result = _extract_low_load_hours(hourly, n=2)
        assert result[0] == 14  # lowest CLS first
        assert result[1] == 15


class TestMorningVsAfternoon:
    def test_morning_dominant(self):
        # Morning hours have high FDI
        hourly = [
            HourlyProfile(hour=9,  avg_cls=0.3, avg_fdi=0.90, window_count=5),
            HourlyProfile(hour=10, avg_cls=0.3, avg_fdi=0.88, window_count=5),
            HourlyProfile(hour=11, avg_cls=0.3, avg_fdi=0.85, window_count=5),
            HourlyProfile(hour=14, avg_cls=0.3, avg_fdi=0.60, window_count=5),
            HourlyProfile(hour=15, avg_cls=0.3, avg_fdi=0.62, window_count=5),
        ]
        assert _morning_vs_afternoon(hourly) == "morning"

    def test_afternoon_dominant(self):
        hourly = [
            HourlyProfile(hour=9,  avg_cls=0.3, avg_fdi=0.55, window_count=5),
            HourlyProfile(hour=10, avg_cls=0.3, avg_fdi=0.57, window_count=5),
            HourlyProfile(hour=14, avg_cls=0.3, avg_fdi=0.90, window_count=5),
            HourlyProfile(hour=15, avg_cls=0.3, avg_fdi=0.85, window_count=5),
            HourlyProfile(hour=16, avg_cls=0.3, avg_fdi=0.88, window_count=5),
        ]
        assert _morning_vs_afternoon(hourly) == "afternoon"

    def test_balanced(self):
        hourly = [
            HourlyProfile(hour=9,  avg_cls=0.3, avg_fdi=0.75, window_count=5),
            HourlyProfile(hour=10, avg_cls=0.3, avg_fdi=0.75, window_count=5),
            HourlyProfile(hour=14, avg_cls=0.3, avg_fdi=0.75, window_count=5),
            HourlyProfile(hour=15, avg_cls=0.3, avg_fdi=0.75, window_count=5),
        ]
        assert _morning_vs_afternoon(hourly) == "balanced"

    def test_no_data_returns_none(self):
        assert _morning_vs_afternoon([]) is None

    def test_only_morning_data_returns_none(self):
        # Missing afternoon hours → can't compare
        hourly = [
            HourlyProfile(hour=9, avg_cls=0.3, avg_fdi=0.80, window_count=5),
        ]
        result = _morning_vs_afternoon(hourly)
        assert result is None


# ─── Integration tests: compute_cognitive_rhythm ──────────────────────────────

class TestComputeCognitiveRhythm:

    def _make_windows_for_date(
        self,
        date_str: str,
        peak_hour: int = 9,
        low_hour: int = 14,
    ) -> list[dict]:
        """Create a set of windows for a day with clear peak/trough."""
        windows = []
        for h in range(8, 18):
            fdi = 0.90 if h == peak_hour else 0.60
            cls = 0.10 if h == low_hour else 0.35
            # 4 windows per hour (15-min each)
            for _ in range(4):
                windows.append(_make_window(date_str, h, cls_val=cls, fdi_val=fdi))
        return windows

    def test_insufficient_data_returns_not_meaningful(self):
        with patch("analysis.cognitive_rhythm.list_available_dates", return_value=[]):
            with patch("analysis.cognitive_rhythm.read_summary", return_value={}):
                rhythm = compute_cognitive_rhythm()
        assert not rhythm.is_meaningful
        assert rhythm.days_analyzed == 0

    def test_one_day_below_minimum_returns_not_meaningful(self):
        with patch("analysis.cognitive_rhythm.list_available_dates", return_value=["2026-03-13"]):
            with patch("analysis.cognitive_rhythm.read_day", return_value=[]):
                with patch("analysis.cognitive_rhythm.read_summary", return_value={}):
                    rhythm = compute_cognitive_rhythm()
        assert not rhythm.is_meaningful

    def test_sufficient_data_returns_meaningful(self):
        dates = [f"2026-03-{d:02d}" for d in range(10, 10 + MIN_DAYS_FOR_RHYTHM)]
        windows = self._make_windows_for_date(dates[0])

        with patch("analysis.cognitive_rhythm.list_available_dates", return_value=dates):
            with patch("analysis.cognitive_rhythm.read_day", return_value=windows):
                with patch("analysis.cognitive_rhythm.read_summary", return_value={}):
                    rhythm = compute_cognitive_rhythm()

        assert rhythm.is_meaningful
        assert rhythm.days_analyzed == MIN_DAYS_FOR_RHYTHM

    def test_peak_focus_hours_detected(self):
        dates = [f"2026-03-{d:02d}" for d in range(10, 14)]  # 4 days
        # All windows have high FDI at 9am, lower elsewhere
        def mock_read_day(date_str):
            return self._make_windows_for_date(date_str, peak_hour=9)

        with patch("analysis.cognitive_rhythm.list_available_dates", return_value=dates):
            with patch("analysis.cognitive_rhythm.read_day", side_effect=mock_read_day):
                with patch("analysis.cognitive_rhythm.read_summary", return_value={}):
                    rhythm = compute_cognitive_rhythm()

        assert 9 in rhythm.peak_focus_hours

    def test_morning_bias_detected(self):
        dates = [f"2026-03-{d:02d}" for d in range(10, 14)]

        def mock_read_day(date_str):
            windows = []
            for h in range(8, 18):
                fdi = 0.90 if h < 13 else 0.55
                for _ in range(4):
                    windows.append(_make_window(date_str, h, cls_val=0.3, fdi_val=fdi))
            return windows

        with patch("analysis.cognitive_rhythm.list_available_dates", return_value=dates):
            with patch("analysis.cognitive_rhythm.read_day", side_effect=mock_read_day):
                with patch("analysis.cognitive_rhythm.read_summary", return_value={}):
                    rhythm = compute_cognitive_rhythm()

        assert rhythm.morning_bias == "morning"

    def test_sparklines_correct_length(self):
        dates = [f"2026-03-{d:02d}" for d in range(10, 14)]
        windows = self._make_windows_for_date(dates[0])

        with patch("analysis.cognitive_rhythm.list_available_dates", return_value=dates):
            with patch("analysis.cognitive_rhythm.read_day", return_value=windows):
                with patch("analysis.cognitive_rhythm.read_summary", return_value={}):
                    rhythm = compute_cognitive_rhythm()

        # 8am to 8pm = 13 hours (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
        assert len(rhythm.hourly_fdi_sparkline) == 13
        assert len(rhythm.hourly_cls_sparkline) == 13
        # Mon–Sun = 7 chars
        assert len(rhythm.dow_fdi_sparkline) == 7

    def test_as_of_date_excludes_future(self):
        dates = ["2026-03-10", "2026-03-11", "2026-03-12", "2026-03-15"]  # 15th is after cutoff

        with patch("analysis.cognitive_rhythm.list_available_dates", return_value=dates):
            with patch("analysis.cognitive_rhythm.read_day", return_value=[]):
                with patch("analysis.cognitive_rhythm.read_summary", return_value={}):
                    rhythm = compute_cognitive_rhythm(as_of_date_str="2026-03-13")

        # Should only include dates <= 2026-03-13
        assert rhythm.days_analyzed == 3


# ─── Tests: to_dict serialisation ─────────────────────────────────────────────

class TestToDictSerialisation:
    def test_full_rhythm_to_dict(self):
        rhythm = CognitiveRhythm(
            hourly=[
                HourlyProfile(hour=9, avg_cls=0.25, avg_fdi=0.80, window_count=10),
            ],
            dow=[
                DowProfile(dow=0, label="Mon", avg_cls=0.30, avg_fdi=0.75,
                           avg_recovery=80.0, avg_meetings=60.0, day_count=2),
            ],
            peak_focus_hours=[9, 10],
            low_load_hours=[14, 15],
            best_focus_dow=0,
            heaviest_dow=3,
            morning_bias="morning",
            hourly_fdi_sparkline="▂▄▇█▇▆▅",
            hourly_cls_sparkline="▄▄▄▄▄▄▄",
            dow_fdi_sparkline="▄▇█▇▆▁▁",
            days_analyzed=10,
            date_range="2026-03-01 → 2026-03-14",
            is_meaningful=True,
        )
        d = rhythm.to_dict()

        assert d["is_meaningful"] is True
        assert d["days_analyzed"] == 10
        assert d["morning_bias"] == "morning"
        assert d["peak_focus_hours"] == [9, 10]
        assert len(d["hourly"]) == 1
        assert d["hourly"][0]["hour"] == 9
        assert d["hourly"][0]["avg_cls"] == pytest.approx(0.25, abs=0.001)
        assert len(d["dow"]) == 1
        assert d["dow"][0]["label"] == "Mon"

    def test_not_meaningful_to_dict(self):
        rhythm = CognitiveRhythm(is_meaningful=False, days_analyzed=1)
        d = rhythm.to_dict()
        assert d["is_meaningful"] is False
        assert d["days_analyzed"] == 1

    def test_to_dict_is_json_serialisable(self):
        import json
        rhythm = CognitiveRhythm(
            is_meaningful=False,
            days_analyzed=0,
            date_range="",
        )
        # Should not raise
        json.dumps(rhythm.to_dict())


# ─── Tests: formatters ────────────────────────────────────────────────────────

class TestFormatRhythmLine:
    def test_not_meaningful_returns_empty(self):
        rhythm = CognitiveRhythm(is_meaningful=False)
        assert format_rhythm_line(rhythm) == ""

    def test_includes_peak_focus_hours(self):
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            peak_focus_hours=[9, 10],
            morning_bias="morning",
            best_focus_dow=1,  # Tue
            days_analyzed=10,
        )
        line = format_rhythm_line(rhythm)
        assert "9am" in line
        assert "Rhythm" in line

    def test_includes_morning_bias(self):
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            peak_focus_hours=[],
            morning_bias="afternoon",
            days_analyzed=10,
        )
        line = format_rhythm_line(rhythm)
        assert "afternoon" in line

    def test_includes_best_dow(self):
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            peak_focus_hours=[],
            best_focus_dow=2,  # Wed
            days_analyzed=10,
        )
        line = format_rhythm_line(rhythm)
        assert "Wed" in line


class TestFormatRhythmSection:
    def test_not_meaningful_returns_empty(self):
        rhythm = CognitiveRhythm(is_meaningful=False)
        assert format_rhythm_section(rhythm) == ""

    def test_contains_header(self):
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            days_analyzed=7,
            hourly_fdi_sparkline="▂▄▇",
            hourly_cls_sparkline="▄▄▄",
            dow_fdi_sparkline="▄▇█▇▆▁▁",
            dow=[DowProfile(dow=i, label=d, avg_cls=None, avg_fdi=None,
                            avg_recovery=None, avg_meetings=None, day_count=0)
                 for i, d in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])],
        )
        section = format_rhythm_section(rhythm)
        assert "Rhythm" in section
        assert "7 days" in section

    def test_contains_sparkline(self):
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            days_analyzed=7,
            hourly_fdi_sparkline="▂▄▇▇▅▄▃▂▁▁▁▁▁",
            hourly_cls_sparkline="▄▄▄▄▄▄▄▄▄▄▄▄▄",
            dow_fdi_sparkline="▄▇█▇▆▁▁",
            dow=[DowProfile(dow=i, label=d, avg_cls=None, avg_fdi=None,
                            avg_recovery=None, avg_meetings=None, day_count=0)
                 for i, d in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])],
        )
        section = format_rhythm_section(rhythm)
        assert "▂▄▇" in section  # FDI sparkline visible

    def test_compact_mode_omits_detail(self):
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            days_analyzed=7,
            peak_focus_hours=[9, 10],
            hourly_fdi_sparkline="▂▄▇",
            hourly_cls_sparkline="▄▄▄",
            dow_fdi_sparkline="▄▇█",
            dow=[],
        )
        compact = format_rhythm_section(rhythm, compact=True)
        full = format_rhythm_section(rhythm, compact=False)
        # Compact should be shorter
        assert len(compact) < len(full)

    def test_morning_bias_shown_when_present(self):
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            days_analyzed=10,
            morning_bias="morning",
            hourly_fdi_sparkline="▂▄▇",
            hourly_cls_sparkline="▄▄▄",
            dow_fdi_sparkline="▄▇█▇▆▁▁",
            dow=[DowProfile(dow=i, label=d, avg_cls=None, avg_fdi=None,
                            avg_recovery=None, avg_meetings=None, day_count=0)
                 for i, d in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])],
        )
        section = format_rhythm_section(rhythm)
        assert "morning" in section.lower()


class TestFormatRhythmTerminal:
    def test_not_meaningful_returns_message(self):
        rhythm = CognitiveRhythm(is_meaningful=False, days_analyzed=1)
        output = format_rhythm_terminal(rhythm)
        assert "Not enough data" in output

    def test_meaningful_output_contains_sections(self):
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            days_analyzed=14,
            date_range="2026-03-01 → 2026-03-14",
            peak_focus_hours=[9, 10, 11],
            low_load_hours=[14, 15],
            morning_bias="morning",
            best_focus_dow=1,   # Tue
            heaviest_dow=3,     # Thu
            hourly_fdi_sparkline="▂▄▇▇▅▄▃▂▁▁▁▁▁",
            hourly_cls_sparkline="▄▄▄▄▄▄▄▄▄▄▄▄▄",
            dow_fdi_sparkline="▄▇█▇▆▁▁",
            dow=[DowProfile(dow=i, label=d, avg_cls=0.30, avg_fdi=0.70,
                            avg_recovery=80.0, avg_meetings=60.0, day_count=2)
                 for i, d in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])],
        )
        output = format_rhythm_terminal(rhythm)
        assert "Cognitive Rhythm" in output
        assert "9am" in output
        assert "14 days" in output
        assert "Tue" in output  # best_focus_dow


# ─── Edge case tests ──────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_windows_same_fdi_returns_valid_sparkline(self):
        dates = [f"2026-03-{d:02d}" for d in range(10, 14)]

        def mock_read_day(date_str):
            return [_make_window(date_str, h, fdi_val=0.75) for h in range(8, 18)]

        with patch("analysis.cognitive_rhythm.list_available_dates", return_value=dates):
            with patch("analysis.cognitive_rhythm.read_day", side_effect=mock_read_day):
                with patch("analysis.cognitive_rhythm.read_summary", return_value={}):
                    rhythm = compute_cognitive_rhythm()

        # Should not crash; sparklines should be non-empty
        assert len(rhythm.hourly_fdi_sparkline) > 0
        # All same value → all same sparkline char
        unique_chars = set(c for c in rhythm.hourly_fdi_sparkline if c != "·")
        assert len(unique_chars) <= 1  # all same

    def test_single_weekday_in_data(self):
        """Only Fridays in the data — all other DOW should have day_count=0."""
        # 2026-02-06 = Friday, 2026-02-13 = Friday, 2026-02-20 = Friday (all past)
        dates = ["2026-02-06", "2026-02-13", "2026-02-20"]

        summary = _make_summary_days(
            ("2026-02-06", 0.30, 0.75),
            ("2026-02-13", 0.28, 0.80),
            ("2026-02-20", 0.32, 0.72),
        )

        with patch("analysis.cognitive_rhythm.list_available_dates", return_value=dates):
            with patch("analysis.cognitive_rhythm.read_day", return_value=[]):
                with patch("analysis.cognitive_rhythm.read_summary", return_value=summary):
                    rhythm = compute_cognitive_rhythm()

        assert rhythm.is_meaningful  # 3 days >= MIN_DAYS_FOR_RHYTHM

        # Friday = dow index 4
        fri = rhythm.dow[4]
        assert fri.day_count == 3

        # All other days should have 0 count
        for i, d in enumerate(rhythm.dow):
            if i != 4:
                assert d.day_count == 0

    def test_read_summary_exception_graceful(self):
        dates = [f"2026-03-{d:02d}" for d in range(10, 14)]

        with patch("analysis.cognitive_rhythm.list_available_dates", return_value=dates):
            with patch("analysis.cognitive_rhythm.read_day", return_value=[]):
            # Returning {} simulates read_summary failure gracefully
                with patch("analysis.cognitive_rhythm.read_summary", side_effect=Exception("fail")):
                    try:
                        rhythm = compute_cognitive_rhythm()
                        # Should gracefully degrade
                        assert isinstance(rhythm, CognitiveRhythm)
                    except Exception:
                        pytest.fail("compute_cognitive_rhythm should not propagate exceptions")

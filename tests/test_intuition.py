"""
Tests for the Alfred Intuition Layer — pattern analytics.

Tests the pre-computation functions that build the rich context packet
for the weekly LLM report:
  - compute_hourly_patterns
  - compute_hrv_cls_correlation
  - compute_day_of_week_profile
  - compute_focus_window_analysis
  - compute_meeting_impact
  - compute_weekly_analytics
  - _build_analysis_prompt (structural / non-regression)

Run with: python3 -m pytest tests/test_intuition.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engine.chunker import build_windows
from analysis.intuition import (
    compute_hourly_patterns,
    compute_hrv_cls_correlation,
    compute_day_of_week_profile,
    compute_focus_window_analysis,
    compute_meeting_impact,
    compute_weekly_analytics,
    _build_analysis_prompt,
)


# ─── Shared fixtures ──────────────────────────────────────────────────────────

SAMPLE_WHOOP_GOOD = {
    "recovery_score": 85.0,
    "hrv_rmssd_milli": 78.0,
    "resting_heart_rate": 52.0,
    "sleep_performance": 88.0,
    "sleep_hours": 8.5,
    "strain": 12.0,
    "spo2_percentage": 95.5,
}

SAMPLE_WHOOP_LOW = {
    "recovery_score": 38.0,
    "hrv_rmssd_milli": 42.0,
    "resting_heart_rate": 64.0,
    "sleep_performance": 55.0,
    "sleep_hours": 5.5,
    "strain": 9.0,
    "spo2_percentage": 94.0,
}

SAMPLE_WHOOP_HIGH = {
    "recovery_score": 95.0,
    "hrv_rmssd_milli": 95.0,
    "resting_heart_rate": 50.0,
    "sleep_performance": 96.0,
    "sleep_hours": 9.0,
    "strain": 5.0,
    "spo2_percentage": 96.5,
}


def _make_calendar(hour: int = 10, duration: int = 60, attendees: int = 4):
    """Build a single-event calendar dict at the given hour."""
    return {
        "events": [
            {
                "id": "evt1",
                "title": "Weekly Sync",
                "start": f"2026-03-13T{hour:02d}:00:00+01:00",
                "end": f"2026-03-13T{hour:02d}:{duration:02d}:00+01:00" if duration < 60
                       else f"2026-03-13T{hour + duration // 60:02d}:{duration % 60:02d}:00+01:00",
                "duration_minutes": duration,
                "attendee_count": attendees,
                "organizer_email": "david@szabostuban.com",
                "is_all_day": False,
                "location": "",
                "status": "confirmed",
            }
        ],
        "event_count": 1,
        "total_meeting_minutes": duration,
        "max_concurrent_attendees": attendees,
    }


def _make_calendar_empty():
    return {"events": [], "event_count": 0, "total_meeting_minutes": 0, "max_concurrent_attendees": 0}


def _make_slack_active(hour: int = 14):
    """Add Slack messages in a specific hour."""
    windows = {}
    base_idx = hour * 4
    for offset in range(4):
        windows[base_idx + offset] = {
            "messages_sent": 4,
            "messages_received": 8,
            "total_messages": 12,
            "channels_active": 2,
        }
    return windows


def _make_windows_for_date(date_str: str, whoop=None, calendar=None, slack=None):
    return build_windows(
        date_str=date_str,
        whoop_data=whoop or SAMPLE_WHOOP_GOOD,
        calendar_data=calendar or _make_calendar_empty(),
        slack_windows=slack or {},
    )


def _multi_day_windows() -> list[dict]:
    """Return active windows for 3 days with varying recovery and meetings."""
    all_windows = []

    # Day 1: good recovery, heavy meetings
    all_windows.extend(_make_windows_for_date(
        "2026-03-10",
        whoop=SAMPLE_WHOOP_HIGH,
        calendar=_make_calendar(hour=9, duration=120, attendees=8),
        slack=_make_slack_active(hour=10),
    ))

    # Day 2: low recovery, moderate activity
    all_windows.extend(_make_windows_for_date(
        "2026-03-11",
        whoop=SAMPLE_WHOOP_LOW,
        calendar=_make_calendar(hour=14, duration=30, attendees=3),
        slack=_make_slack_active(hour=15),
    ))

    # Day 3: medium recovery, focus day
    all_windows.extend(_make_windows_for_date(
        "2026-03-12",
        whoop=SAMPLE_WHOOP_GOOD,
        calendar=_make_calendar_empty(),
        slack=_make_slack_active(hour=9),
    ))

    return all_windows


# ─── compute_hourly_patterns ──────────────────────────────────────────────────

class TestComputeHourlyPatterns:
    def test_empty_returns_empty(self):
        assert compute_hourly_patterns([]) == {}

    def test_only_working_hours_included(self):
        """Hours outside 7am-10pm should not appear."""
        windows = _make_windows_for_date(
            "2026-03-13",
            slack=_make_slack_active(hour=2),  # 2am activity — non-working
        )
        patterns = compute_hourly_patterns(windows)
        # 2am windows are outside working hours → should not appear
        assert 2 not in patterns

    def test_active_hours_are_present(self):
        """Hours with meetings or Slack should appear in patterns."""
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=11, duration=60),
        )
        patterns = compute_hourly_patterns(windows)
        # Meeting was at 11am → should appear
        assert 11 in patterns

    def test_avg_cls_is_in_range(self):
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=10, duration=60, attendees=5),
        )
        patterns = compute_hourly_patterns(windows)
        for h, v in patterns.items():
            if v["avg_cls"] is not None:
                assert 0.0 <= v["avg_cls"] <= 1.0, f"Hour {h}: avg_cls={v['avg_cls']} out of range"

    def test_avg_fdi_is_in_range(self):
        windows = _make_windows_for_date(
            "2026-03-13",
            slack=_make_slack_active(hour=14),
        )
        patterns = compute_hourly_patterns(windows)
        for h, v in patterns.items():
            if v["avg_fdi"] is not None:
                assert 0.0 <= v["avg_fdi"] <= 1.0, f"Hour {h}: avg_fdi={v['avg_fdi']} out of range"

    def test_idle_working_hours_excluded(self):
        """Working hours with no meetings and no Slack should not inflate averages."""
        windows = _make_windows_for_date(
            "2026-03-13",
            slack=_make_slack_active(hour=10),
        )
        patterns = compute_hourly_patterns(windows)
        # Only hour 10 should be active; hours like 7,8,9 have no activity
        active_hours = set(patterns.keys())
        # 10 should be present (has Slack)
        assert 10 in active_hours
        # 7 should not be present (no activity)
        assert 7 not in active_hours

    def test_active_windows_count_is_correct(self):
        """active_windows should reflect how many 15-min windows had activity."""
        windows = _make_windows_for_date(
            "2026-03-13",
            slack=_make_slack_active(hour=13),  # 4 windows at 13:00-13:45
        )
        patterns = compute_hourly_patterns(windows)
        if 13 in patterns:
            assert patterns[13]["active_windows"] == 4


# ─── compute_hrv_cls_correlation ─────────────────────────────────────────────

class TestComputeHrvClsCorrelation:
    def test_single_day_returns_insufficient(self):
        windows = _make_windows_for_date("2026-03-13", whoop=SAMPLE_WHOOP_GOOD)
        result = compute_hrv_cls_correlation(windows)
        assert result.get("insufficient_data") is True

    def test_multi_day_returns_correlation(self):
        all_windows = _multi_day_windows()
        result = compute_hrv_cls_correlation(all_windows)
        assert not result.get("insufficient_data")
        assert "correlation_direction" in result

    def test_direction_is_valid_string(self):
        all_windows = _multi_day_windows()
        result = compute_hrv_cls_correlation(all_windows)
        if not result.get("insufficient_data"):
            assert result["correlation_direction"] in ("inverse", "flat", "positive")

    def test_quartile_cls_values_in_range(self):
        all_windows = _multi_day_windows()
        result = compute_hrv_cls_correlation(all_windows)
        if not result.get("insufficient_data"):
            for key, val in result.get("quartile_cls", {}).items():
                if val is not None:
                    assert 0.0 <= val <= 1.0, f"Quartile {key}: {val} out of range"

    def test_note_is_always_string(self):
        for windows in [
            _make_windows_for_date("2026-03-13"),
            _multi_day_windows(),
        ]:
            result = compute_hrv_cls_correlation(windows)
            assert isinstance(result.get("note", ""), str)

    def test_low_hrv_high_cls_is_inverse(self):
        """3 days: low HRV with high meetings → should detect inverse or flat correlation."""
        all_windows = []
        # Day with high HRV, low meetings
        all_windows.extend(_make_windows_for_date(
            "2026-03-10",
            whoop={**SAMPLE_WHOOP_HIGH, "hrv_rmssd_milli": 100.0},
            calendar=_make_calendar_empty(),
        ))
        # Day with medium HRV, medium meetings
        all_windows.extend(_make_windows_for_date(
            "2026-03-11",
            whoop={**SAMPLE_WHOOP_GOOD, "hrv_rmssd_milli": 65.0},
            calendar=_make_calendar(hour=10, duration=60, attendees=4),
        ))
        # Day with low HRV, heavy meetings
        all_windows.extend(_make_windows_for_date(
            "2026-03-12",
            whoop={**SAMPLE_WHOOP_LOW, "hrv_rmssd_milli": 40.0},
            calendar=_make_calendar(hour=9, duration=120, attendees=8),
            slack=_make_slack_active(hour=14),
        ))
        result = compute_hrv_cls_correlation(all_windows)
        if not result.get("insufficient_data"):
            # High meetings on low HRV day should show inverse direction
            assert result["correlation_direction"] in ("inverse", "flat")


# ─── compute_day_of_week_profile ──────────────────────────────────────────────

class TestComputeDayOfWeekProfile:
    def test_returns_list_of_dicts(self):
        profiles = compute_day_of_week_profile(_multi_day_windows())
        assert isinstance(profiles, list)
        assert len(profiles) > 0

    def test_one_profile_per_date(self):
        all_windows = _multi_day_windows()
        profiles = compute_day_of_week_profile(all_windows)
        dates = [p["date"] for p in profiles]
        assert len(dates) == len(set(dates)), "Duplicate dates in profiles"

    def test_profiles_are_chronologically_sorted(self):
        profiles = compute_day_of_week_profile(_multi_day_windows())
        dates = [p["date"] for p in profiles]
        assert dates == sorted(dates)

    def test_required_keys_present(self):
        profiles = compute_day_of_week_profile(_multi_day_windows())
        required = [
            "date", "day_of_week", "recovery_score", "hrv_rmssd_milli",
            "avg_cls", "peak_cls", "avg_fdi_active", "avg_ras",
            "total_meeting_minutes", "active_windows",
        ]
        for p in profiles:
            for k in required:
                assert k in p, f"Missing key '{k}' in profile: {p}"

    def test_meeting_minutes_reflect_calendar(self):
        """Day with 60-min meeting should have >= 45 total_meeting_minutes."""
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=11, duration=60),
        )
        profiles = compute_day_of_week_profile(windows)
        assert len(profiles) == 1
        assert profiles[0]["total_meeting_minutes"] >= 45

    def test_metrics_all_in_range(self):
        for p in compute_day_of_week_profile(_multi_day_windows()):
            for key in ["avg_cls", "peak_cls", "avg_fdi_active", "avg_ras"]:
                val = p.get(key)
                if val is not None:
                    assert 0.0 <= val <= 1.0, f"{key}={val} out of range in {p['date']}"

    def test_active_windows_count(self):
        """Active window count should be positive when there is calendar/slack activity."""
        windows = _make_windows_for_date(
            "2026-03-13",
            slack=_make_slack_active(hour=10),
        )
        profiles = compute_day_of_week_profile(windows)
        assert profiles[0]["active_windows"] > 0


# ─── compute_focus_window_analysis ───────────────────────────────────────────

class TestComputeFocusWindowAnalysis:
    def test_empty_returns_insufficient(self):
        result = compute_focus_window_analysis([])
        assert result.get("insufficient_data") is True

    def test_no_active_windows_returns_insufficient(self):
        """If all windows are idle, no focus data to analyze."""
        windows = _make_windows_for_date("2026-03-13")  # No meetings, no slack
        result = compute_focus_window_analysis(windows)
        assert result.get("insufficient_data") is True

    def test_active_day_returns_analysis(self):
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=10, duration=60),
            slack=_make_slack_active(hour=14),
        )
        result = compute_focus_window_analysis(windows)
        assert not result.get("insufficient_data")
        assert "best_focus_hours" in result
        assert "peak_load_hours" in result

    def test_best_focus_hours_are_sorted_descending(self):
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=10, duration=60),
            slack=_make_slack_active(hour=14),
        )
        result = compute_focus_window_analysis(windows)
        if not result.get("insufficient_data"):
            fdis = [h["avg_fdi"] for h in result["best_focus_hours"]]
            assert fdis == sorted(fdis, reverse=True), "Best focus hours should be sorted desc"

    def test_peak_load_hours_are_sorted_descending(self):
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=10, duration=60),
            slack=_make_slack_active(hour=14),
        )
        result = compute_focus_window_analysis(windows)
        if not result.get("insufficient_data"):
            clss = [h["avg_cls"] for h in result["peak_load_hours"]]
            assert clss == sorted(clss, reverse=True), "Peak load hours should be sorted desc"

    def test_hour_values_are_in_range(self):
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=10, duration=60),
            slack=_make_slack_active(hour=14),
        )
        result = compute_focus_window_analysis(windows)
        if not result.get("insufficient_data"):
            for h_item in result.get("best_focus_hours", []):
                assert 0.0 <= h_item["avg_fdi"] <= 1.0
            for h_item in result.get("peak_load_hours", []):
                assert 0.0 <= h_item["avg_cls"] <= 1.0

    def test_meeting_hours_appear_as_high_load(self):
        """Hours with meetings should appear in peak_load_hours."""
        windows = _make_windows_for_date(
            "2026-03-13",
            whoop={**SAMPLE_WHOOP_LOW, "recovery_score": 30.0},  # Low recovery boosts CLS
            calendar=_make_calendar(hour=11, duration=60, attendees=8),
        )
        result = compute_focus_window_analysis(windows)
        if not result.get("insufficient_data"):
            peak_hours = [h["hour"] for h in result["peak_load_hours"]]
            # 11:00 should be the peak load hour
            assert "11:00" in peak_hours, f"Expected 11:00 in peak hours, got {peak_hours}"


# ─── compute_meeting_impact ───────────────────────────────────────────────────

class TestComputeMeetingImpact:
    def test_no_meetings_returns_zeros(self):
        windows = _make_windows_for_date("2026-03-13", slack=_make_slack_active(hour=10))
        result = compute_meeting_impact(windows)
        assert result["meeting_window_count"] == 0
        assert result["total_meeting_minutes_week"] == 0
        assert result["avg_cls_in_meeting"] is None

    def test_meeting_cls_higher_than_focused(self):
        """CLS should be higher during meetings than during focused (Slack-only) work."""
        windows = _make_windows_for_date(
            "2026-03-13",
            whoop=SAMPLE_WHOOP_LOW,  # Low recovery → higher baseline CLS
            calendar=_make_calendar(hour=11, duration=60, attendees=8),
            slack=_make_slack_active(hour=9),  # Non-meeting Slack hour
        )
        result = compute_meeting_impact(windows)
        if result["avg_cls_in_meeting"] is not None and result["avg_cls_focused_work"] is not None:
            assert result["avg_cls_in_meeting"] >= result["avg_cls_focused_work"], (
                f"Meeting CLS ({result['avg_cls_in_meeting']}) should be >= "
                f"focused CLS ({result['avg_cls_focused_work']})"
            )

    def test_cls_premium_is_non_negative_for_meetings(self):
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=11, duration=60, attendees=8),
            slack=_make_slack_active(hour=14),
        )
        result = compute_meeting_impact(windows)
        if result["meeting_cls_premium"] is not None:
            # Premium could be positive (meetings cost more) or near zero
            assert result["meeting_cls_premium"] >= -0.05, (
                "Meeting CLS premium should not be significantly negative"
            )

    def test_total_meeting_minutes_correct(self):
        """A 60-min meeting covering 4 windows should show 60 total minutes."""
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=10, duration=60),
        )
        result = compute_meeting_impact(windows)
        assert result["total_meeting_minutes_week"] == 60

    def test_required_keys_present(self):
        windows = _make_windows_for_date("2026-03-13")
        result = compute_meeting_impact(windows)
        required = [
            "meeting_window_count", "total_meeting_minutes_week",
            "avg_cls_in_meeting", "avg_cls_focused_work",
            "avg_fdi_in_meeting", "avg_fdi_focused_work",
            "meeting_cls_premium", "short_meeting_fragmentation_pct",
        ]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_fragmentation_pct_in_range(self):
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=11, duration=20),  # Short meeting
        )
        result = compute_meeting_impact(windows)
        assert 0.0 <= result["short_meeting_fragmentation_pct"] <= 100.0


# ─── compute_weekly_analytics ─────────────────────────────────────────────────

class TestComputeWeeklyAnalytics:
    def test_empty_returns_empty(self):
        assert compute_weekly_analytics([]) == {}

    def test_returns_all_sections(self):
        all_windows = _multi_day_windows()
        result = compute_weekly_analytics(all_windows)
        assert "day_profiles" in result
        assert "hourly_patterns" in result
        assert "focus_analysis" in result
        assert "meeting_impact" in result
        assert "hrv_cls_correlation" in result

    def test_day_profiles_count_matches_days(self):
        all_windows = _multi_day_windows()
        result = compute_weekly_analytics(all_windows)
        profiles = result["day_profiles"]
        # We built 3 days of data
        assert len(profiles) == 3

    def test_no_crash_with_single_day(self):
        windows = _make_windows_for_date(
            "2026-03-13",
            calendar=_make_calendar(hour=10, duration=60),
            slack=_make_slack_active(hour=14),
        )
        result = compute_weekly_analytics(windows)
        assert result is not None
        assert "day_profiles" in result


# ─── _build_analysis_prompt ───────────────────────────────────────────────────

class TestBuildAnalysisPrompt:
    """Structural tests — validate that the prompt contains expected sections."""

    def _make_summaries(self):
        """Minimal daily summaries for prompt testing."""
        return [
            {
                "date": "2026-03-10",
                "whoop": {"recovery_score": 85.0, "hrv_rmssd_milli": 78.0},
                "metrics_avg": {"cognitive_load_score": 0.35, "focus_depth_index": 0.65,
                                "social_drain_index": 0.40, "recovery_alignment_score": 0.75},
                "calendar": {"total_meeting_minutes": 90, "meeting_windows": 6},
                "slack": {"total_messages_sent": 15, "total_messages_received": 40},
            },
            {
                "date": "2026-03-11",
                "whoop": {"recovery_score": 42.0, "hrv_rmssd_milli": 45.0},
                "metrics_avg": {"cognitive_load_score": 0.55, "focus_depth_index": 0.45,
                                "social_drain_index": 0.58, "recovery_alignment_score": 0.38},
                "calendar": {"total_meeting_minutes": 180, "meeting_windows": 12},
                "slack": {"total_messages_sent": 25, "total_messages_received": 60},
            },
        ]

    def test_prompt_is_non_empty_string(self):
        prompt = _build_analysis_prompt(self._make_summaries(), [])
        assert isinstance(prompt, str)
        assert len(prompt) > 500

    def test_prompt_contains_metrics_context(self):
        """The prompt must explain what the metrics mean."""
        prompt = _build_analysis_prompt(self._make_summaries(), [])
        assert "CLS" in prompt
        assert "FDI" in prompt
        assert "RAS" in prompt

    def test_prompt_contains_weekly_overview(self):
        prompt = _build_analysis_prompt(self._make_summaries(), [])
        assert "Average recovery" in prompt or "recovery" in prompt.lower()
        assert "cognitive load" in prompt.lower()

    def test_prompt_contains_slack_channel(self):
        """Prompt must instruct the LLM to send to the correct channel."""
        from config import SLACK_DM_CHANNEL
        prompt = _build_analysis_prompt(self._make_summaries(), [])
        assert SLACK_DM_CHANNEL in prompt

    def test_prompt_includes_daily_summaries(self):
        """Full daily summary JSON should be embedded in the prompt."""
        prompt = _build_analysis_prompt(self._make_summaries(), [])
        assert "2026-03-10" in prompt
        assert "2026-03-11" in prompt

    def test_prompt_includes_window_data_when_provided(self):
        """When windows_sample is provided, analytics should appear in prompt."""
        all_windows = _multi_day_windows()
        summaries = self._make_summaries()
        prompt = _build_analysis_prompt(summaries, all_windows)
        # The prompt should contain hourly analytics or meeting impact stats
        assert (
            "focus hours" in prompt.lower()
            or "Best focus" in prompt
            or "Meeting" in prompt
            or "meeting" in prompt.lower()
        )

    def test_prompt_without_windows_does_not_crash(self):
        """Empty windows_sample should still produce a valid prompt."""
        prompt = _build_analysis_prompt(self._make_summaries(), [])
        assert isinstance(prompt, str)
        assert len(prompt) > 200

    def test_prompt_contains_day_table_with_summaries(self):
        """Day-by-day breakdown should appear if day profiles are computed."""
        all_windows = _multi_day_windows()
        prompt = _build_analysis_prompt(self._make_summaries(), all_windows)
        # Day table header or content should be in the prompt
        assert "Day-by-day" in prompt or "day-by-day" in prompt.lower()

    def test_prompt_has_required_report_sections(self):
        """The prompt must instruct the LLM to produce all required sections."""
        prompt = _build_analysis_prompt(self._make_summaries(), [])
        assert "Health Baseline" in prompt
        assert "Cognitive Load Pattern" in prompt
        assert "Meeting Load" in prompt
        assert "Recovery Alignment" in prompt
        assert "Key Insight" in prompt
        assert "Recommendation" in prompt

    def test_prompt_contains_tone_instructions(self):
        """Alfred's tone instructions must be present."""
        prompt = _build_analysis_prompt(self._make_summaries(), [])
        assert "Alfred" in prompt
        assert "no bullet" in prompt.lower() or "no filler" in prompt.lower() or "no fluff" in prompt.lower()

    def test_prompt_with_empty_summaries_does_not_crash(self):
        prompt = _build_analysis_prompt([], [])
        assert isinstance(prompt, str)

    def test_hrv_correlation_in_prompt_when_multi_day(self):
        """HRV–CLS correlation section should appear in prompt when data supports it."""
        all_windows = _multi_day_windows()
        prompt = _build_analysis_prompt(self._make_summaries(), all_windows)
        assert "HRV" in prompt or "hrv" in prompt.lower()

    def test_prompt_with_windows_is_longer_than_without(self):
        """Providing window data should produce a richer (longer) prompt."""
        summaries = self._make_summaries()
        prompt_no_windows = _build_analysis_prompt(summaries, [])
        prompt_with_windows = _build_analysis_prompt(summaries, _multi_day_windows())
        assert len(prompt_with_windows) >= len(prompt_no_windows), (
            "Prompt with window analytics should be at least as long as without"
        )

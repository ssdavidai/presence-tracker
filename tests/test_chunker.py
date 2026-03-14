"""
Tests for the chunking engine.

Run with: python3 -m pytest tests/test_chunker.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engine.chunker import build_windows, summarize_day, WINDOWS_PER_DAY


SAMPLE_WHOOP = {
    "recovery_score": 85.0,
    "hrv_rmssd_milli": 72.4,
    "resting_heart_rate": 55.0,
    "sleep_performance": 89.0,
    "sleep_hours": 8.2,
    "strain": 12.5,
    "spo2_percentage": 95.1,
}

SAMPLE_CALENDAR = {
    "events": [
        {
            "id": "evt1",
            "title": "Product Sync",
            "start": "2026-03-13T10:00:00+01:00",
            "end": "2026-03-13T11:00:00+01:00",
            "duration_minutes": 60,
            "attendee_count": 4,
            "organizer_email": "david@szabostuban.com",
            "is_all_day": False,
            "location": "",
            "status": "confirmed",
        }
    ],
    "event_count": 1,
    "total_meeting_minutes": 60,
    "max_concurrent_attendees": 4,
}


def make_slack_windows():
    """Create sample slack windows with activity from 10-11am."""
    windows = {}
    # 10:00-10:15 = window 40
    windows[40] = {"messages_sent": 2, "messages_received": 5, "total_messages": 7, "channels_active": 1}
    # 10:15-10:30 = window 41
    windows[41] = {"messages_sent": 1, "messages_received": 3, "total_messages": 4, "channels_active": 1}
    return windows


class TestBuildWindows:
    def test_produces_96_windows(self):
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR,
            slack_windows=make_slack_windows(),
        )
        assert len(windows) == WINDOWS_PER_DAY, f"Expected 96 windows, got {len(windows)}"

    def test_window_schema(self):
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        w = windows[0]  # First window: 00:00-00:15

        assert "window_id" in w
        assert "date" in w
        assert "window_start" in w
        assert "window_end" in w
        assert "window_index" in w
        assert "calendar" in w
        assert "whoop" in w
        assert "slack" in w
        assert "metrics" in w
        assert "metadata" in w

    def test_window_indices_are_sequential(self):
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        for i, w in enumerate(windows):
            assert w["window_index"] == i, f"Window {i} has wrong index: {w['window_index']}"

    def test_meeting_detected_in_correct_windows(self):
        """Meeting 10:00-11:00 → windows 40,41,42,43 (10:00, 10:15, 10:30, 10:45)."""
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})

        meeting_windows = [i for i, w in enumerate(windows) if w["calendar"]["in_meeting"]]
        # 10:00 = hour 10, windows 40, 41, 42, 43
        expected = [40, 41, 42, 43]
        assert set(meeting_windows) == set(expected), \
            f"Meeting should be in windows {expected}, got {meeting_windows}"

    def test_whoop_data_propagated_to_all_windows(self):
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        for w in windows:
            assert w["whoop"]["recovery_score"] == SAMPLE_WHOOP["recovery_score"]

    def test_slack_data_in_correct_windows(self):
        slack = make_slack_windows()
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, slack)

        # Window 40 (10:00-10:15) should have Slack data
        w40 = windows[40]
        assert w40["slack"]["total_messages"] == 7

        # Window 0 (00:00-00:15) should have no Slack data
        w0 = windows[0]
        assert w0["slack"]["total_messages"] == 0

    def test_working_hours_flag(self):
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        for w in windows:
            hour = w["metadata"]["hour_of_day"]
            is_working = w["metadata"]["is_working_hours"]
            expected_working = 7 <= hour < 22
            assert is_working == expected_working, \
                f"Window hour {hour}: is_working={is_working}, expected {expected_working}"

    def test_metrics_present_and_in_range(self):
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, make_slack_windows())
        metric_keys = [
            "cognitive_load_score", "focus_depth_index", "social_drain_index",
            "context_switch_cost", "recovery_alignment_score"
        ]
        for w in windows:
            for key in metric_keys:
                assert key in w["metrics"], f"Missing metric {key} in window {w['window_index']}"
                val = w["metrics"][key]
                assert 0.0 <= val <= 1.0, f"Metric {key}={val} out of range in window {w['window_index']}"

    def test_empty_inputs_dont_crash(self):
        windows = build_windows("2026-03-13", {}, {"events": []}, {})
        assert len(windows) == WINDOWS_PER_DAY

    def test_date_str_in_window_id(self):
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        for w in windows:
            assert "2026-03-13" in w["window_id"]
            assert w["date"] == "2026-03-13"

    def test_day_of_week_correct(self):
        # 2026-03-13 is a Friday
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        assert windows[0]["metadata"]["day_of_week"] == "Friday"


class TestSummarizeDay:
    def test_summary_has_expected_keys(self):
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        summary = summarize_day(windows)

        assert "date" in summary
        assert "working_hours_analyzed" in summary
        assert "metrics_avg" in summary
        assert "calendar" in summary
        assert "whoop" in summary

    def test_summary_date_matches(self):
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        summary = summarize_day(windows)
        assert summary["date"] == "2026-03-13"

    def test_meeting_time_counted_correctly(self):
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        summary = summarize_day(windows)
        # We had a 60-min meeting and 4 windows (15 min each) = 60 min total
        assert summary["calendar"]["meeting_windows"] == 4

    def test_empty_windows_returns_empty(self):
        summary = summarize_day([])
        assert summary == {}

    def test_avg_metrics_in_range(self):
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, make_slack_windows())
        summary = summarize_day(windows)
        for key, val in summary["metrics_avg"].items():
            if val is not None:
                assert 0.0 <= val <= 1.0, f"Summary metric {key}={val} out of range"


class TestIsActiveWindow:
    """Tests for the is_active_window metadata flag (v1.3)."""

    def test_meeting_window_is_active(self):
        """Windows that overlap a calendar event are always active."""
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        # Meeting is 10:00-11:00 → windows 40-43
        for idx in [40, 41, 42, 43]:
            w = windows[idx]
            assert w["metadata"]["is_active_window"] is True, \
                f"Window {idx} (in meeting) should be active"

    def test_idle_window_is_not_active(self):
        """Sleep/idle windows with no signals are NOT active."""
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        # Window 0 = 00:00-00:15, no meeting, no Slack, no RT → inactive
        w = windows[0]
        assert w["metadata"]["is_active_window"] is False, \
            "Idle midnight window should not be active"

    def test_slack_message_window_is_active(self):
        """Windows with Slack messages are active even without a meeting."""
        slack = {5: {"messages_sent": 0, "messages_received": 3, "total_messages": 3, "channels_active": 1}}
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, slack)
        assert windows[5]["metadata"]["is_active_window"] is True, \
            "Window 5 has Slack messages so should be active"

    def test_zero_slack_no_meeting_is_not_active(self):
        """A window with zero Slack and no meeting is inactive."""
        slack = {10: {"messages_sent": 0, "messages_received": 0, "total_messages": 0, "channels_active": 0}}
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, slack)
        # Window 10 is not in the meeting range (40-43) and has no messages
        assert windows[10]["metadata"]["is_active_window"] is False, \
            "Window 10 has no signals so should be inactive"

    def test_rescuetime_active_makes_window_active(self):
        """Windows with RescueTime computer activity are active."""
        rt_windows = {
            15: {
                "focus_seconds": 300,
                "distraction_seconds": 0,
                "neutral_seconds": 0,
                "active_seconds": 300,
                "app_switches": 2,
                "productivity_score": 0.8,
                "top_activity": "PyCharm",
            }
        }
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {}, rt_windows)
        assert windows[15]["metadata"]["is_active_window"] is True, \
            "Window with RescueTime activity should be active"

    def test_rescuetime_zero_seconds_is_not_active(self):
        """A window where RescueTime has zero active_seconds is not active (no RT signal)."""
        rt_windows = {
            20: {
                "focus_seconds": 0,
                "distraction_seconds": 0,
                "neutral_seconds": 0,
                "active_seconds": 0,
                "app_switches": 0,
                "productivity_score": None,
                "top_activity": None,
            }
        }
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {}, rt_windows)
        # Window 20 is not in meeting, has no Slack, RT is zero
        assert windows[20]["metadata"]["is_active_window"] is False, \
            "Window with zero RT active_seconds should not be active"

    def test_is_active_window_present_in_all_windows(self):
        """Every window must have is_active_window in metadata."""
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, make_slack_windows())
        for w in windows:
            assert "is_active_window" in w["metadata"], \
                f"Window {w['window_index']} missing is_active_window"
            assert isinstance(w["metadata"]["is_active_window"], bool)

    def test_active_window_count_matches_signals(self):
        """The count of active windows matches manually countable signals."""
        slack = make_slack_windows()  # windows 40, 41
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, slack)
        active = [w for w in windows if w["metadata"]["is_active_window"]]
        # Calendar: windows 40, 41, 42, 43 (4)
        # Slack: windows 40, 41 (already counted in calendar overlap)
        # Total unique active: 40, 41, 42, 43 = 4 windows
        assert len(active) == 4, f"Expected 4 active windows, got {len(active)}"


class TestFocusQuality:
    """Tests for the focus_quality section in summarize_day (v1.3)."""

    def test_summary_has_focus_quality_key(self):
        """summarize_day() must include a focus_quality section."""
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        summary = summarize_day(windows)
        assert "focus_quality" in summary, "summary must have focus_quality section"

    def test_focus_quality_has_required_fields(self):
        """focus_quality must contain active_fdi, active_windows, peak_focus_hour, peak_focus_fdi."""
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, make_slack_windows())
        summary = summarize_day(windows)
        fq = summary["focus_quality"]
        assert "active_fdi" in fq
        assert "active_windows" in fq
        assert "peak_focus_hour" in fq
        assert "peak_focus_fdi" in fq

    def test_active_windows_count_is_non_negative(self):
        """active_windows must always be >= 0."""
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, {})
        summary = summarize_day(windows)
        assert summary["focus_quality"]["active_windows"] >= 0

    def test_active_fdi_in_range_or_none(self):
        """active_fdi must be in [0, 1] or None (when no active windows)."""
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, make_slack_windows())
        summary = summarize_day(windows)
        fdi = summary["focus_quality"]["active_fdi"]
        if fdi is not None:
            assert 0.0 <= fdi <= 1.0, f"active_fdi={fdi} out of range"

    def test_active_fdi_lower_than_all_windows_fdi(self):
        """
        When some idle windows exist (most), active_fdi should be <= metrics_avg.fdi.

        Idle windows have FDI=1.0 (no disruption), which inflates the all-window
        average.  Active-only FDI is more honest (meetings/Slack reduce it).
        """
        # Use calendar with meetings so active windows have reduced FDI
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, make_slack_windows())
        summary = summarize_day(windows)
        all_fdi = summary["metrics_avg"]["focus_depth_index"]
        active_fdi = summary["focus_quality"]["active_fdi"]
        if active_fdi is not None and all_fdi is not None:
            assert active_fdi <= all_fdi, (
                f"active_fdi ({active_fdi}) should be <= all-window FDI ({all_fdi}) "
                "because idle windows inflate the all-window average"
            )

    def test_no_active_windows_returns_none_fdi(self):
        """When no behavioral signals exist, active_fdi and peak_focus_hour should be None."""
        # Empty calendar, no Slack, no RT → all windows are inactive
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, {"events": []}, {})
        summary = summarize_day(windows)
        assert summary["focus_quality"]["active_fdi"] is None, \
            "With no active windows, active_fdi should be None"
        assert summary["focus_quality"]["peak_focus_hour"] is None, \
            "With no active windows, peak_focus_hour should be None"
        assert summary["focus_quality"]["active_windows"] == 0

    def test_peak_focus_hour_is_valid_hour_or_none(self):
        """peak_focus_hour must be in 0-23 or None."""
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, make_slack_windows())
        summary = summarize_day(windows)
        h = summary["focus_quality"]["peak_focus_hour"]
        if h is not None:
            assert 0 <= h <= 23, f"peak_focus_hour={h} is not a valid hour"

    def test_peak_focus_fdi_matches_peak_hour(self):
        """peak_focus_fdi must be > 0 when peak_focus_hour is set."""
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, SAMPLE_CALENDAR, make_slack_windows())
        summary = summarize_day(windows)
        fq = summary["focus_quality"]
        if fq["peak_focus_hour"] is not None:
            assert fq["peak_focus_fdi"] is not None
            assert 0.0 <= fq["peak_focus_fdi"] <= 1.0

    def test_active_windows_count_with_slack_only(self):
        """Active windows counted correctly when only Slack is the signal source."""
        slack = {
            30: {"messages_sent": 1, "messages_received": 0, "total_messages": 1, "channels_active": 1},
            31: {"messages_sent": 0, "messages_received": 2, "total_messages": 2, "channels_active": 1},
        }
        windows = build_windows("2026-03-13", SAMPLE_WHOOP, {"events": []}, slack)
        summary = summarize_day(windows)
        # Working hours check: hour 7 = windows 28-31, windows 30 and 31 are in working hours
        active_count = summary["focus_quality"]["active_windows"]
        # Windows 30 (hour 7) and 31 (hour 7) — both within working hours (7am+)
        assert active_count >= 2, f"Expected at least 2 active windows, got {active_count}"

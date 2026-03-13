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

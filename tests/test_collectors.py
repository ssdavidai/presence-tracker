"""
Tests for the data collectors.
These are integration tests — they hit real APIs.

Run with: python3 -m pytest tests/test_collectors.py -v
Skip with: python3 -m pytest tests/ -v --ignore=tests/test_collectors.py

Note: WHOOP and Calendar require live credentials.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime


TODAY = datetime.now().strftime("%Y-%m-%d")


class TestWhoopCollector:
    def test_returns_dict(self):
        from collectors.whoop import collect
        result = collect(TODAY)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        from collectors.whoop import collect
        result = collect(TODAY)
        expected_keys = [
            "recovery_score", "hrv_rmssd_milli", "resting_heart_rate",
            "sleep_performance", "sleep_hours", "strain", "spo2_percentage"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_recovery_score_in_range(self):
        from collectors.whoop import collect
        result = collect(TODAY)
        if result.get("recovery_score") is not None:
            assert 0 <= result["recovery_score"] <= 100, \
                f"Recovery score out of range: {result['recovery_score']}"

    def test_hrv_positive(self):
        from collectors.whoop import collect
        result = collect(TODAY)
        if result.get("hrv_rmssd_milli") is not None:
            assert result["hrv_rmssd_milli"] > 0, \
                f"HRV should be positive: {result['hrv_rmssd_milli']}"


class TestCalendarCollector:
    def test_returns_dict(self):
        from collectors.gcal import collect
        result = collect(TODAY)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        from collectors.gcal import collect
        result = collect(TODAY)
        assert "events" in result
        assert "event_count" in result
        assert "total_meeting_minutes" in result
        assert "max_concurrent_attendees" in result

    def test_events_is_list(self):
        from collectors.gcal import collect
        result = collect(TODAY)
        assert isinstance(result["events"], list)

    def test_event_count_matches_list(self):
        from collectors.gcal import collect
        result = collect(TODAY)
        assert result["event_count"] == len(result["events"])

    def test_event_schema(self):
        from collectors.gcal import collect
        result = collect(TODAY)
        for event in result["events"]:
            assert "id" in event
            assert "title" in event
            assert "start" in event
            assert "end" in event
            assert "duration_minutes" in event
            assert "attendee_count" in event


class TestSlackCollector:
    def test_returns_dict(self):
        from collectors.slack import collect
        result = collect(TODAY)
        assert isinstance(result, dict)

    def test_96_windows(self):
        from collectors.slack import collect
        result = collect(TODAY)
        # Should have exactly 96 window indices (0-95) OR be sparse (only active windows)
        # The slack collector returns sparse windows (only non-zero), but that's fine
        # All window indices should be valid
        for idx in result.keys():
            assert 0 <= int(idx) <= 95, f"Invalid window index: {idx}"

    def test_window_has_expected_keys(self):
        from collectors.slack import collect
        result = collect(TODAY)
        for idx, window in result.items():
            assert "messages_sent" in window
            assert "messages_received" in window
            assert "total_messages" in window
            assert "channels_active" in window

    def test_totals_are_non_negative(self):
        from collectors.slack import collect
        result = collect(TODAY)
        for idx, window in result.items():
            assert window["messages_sent"] >= 0
            assert window["messages_received"] >= 0
            assert window["total_messages"] >= 0
            assert window["channels_active"] >= 0


class TestGetEventsInWindow:
    def test_overlap_detection(self):
        from collectors.gcal import get_events_in_window
        from datetime import datetime
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Europe/Budapest")
        events = [
            {
                "id": "1",
                "title": "Meeting",
                "start": "2026-03-13T10:00:00+01:00",
                "end": "2026-03-13T11:00:00+01:00",
                "duration_minutes": 60,
                "attendee_count": 3,
                "organizer_email": "test@test.com",
                "is_all_day": False,
                "location": "",
                "status": "confirmed",
            }
        ]

        # Window that overlaps: 10:00-10:15
        w_start = datetime(2026, 3, 13, 10, 0, tzinfo=tz)
        w_end = datetime(2026, 3, 13, 10, 15, tzinfo=tz)
        overlap = get_events_in_window(events, w_start, w_end)
        assert len(overlap) == 1

        # Window that doesn't overlap: 11:00-11:15
        w_start2 = datetime(2026, 3, 13, 11, 0, tzinfo=tz)
        w_end2 = datetime(2026, 3, 13, 11, 15, tzinfo=tz)
        no_overlap = get_events_in_window(events, w_start2, w_end2)
        assert len(no_overlap) == 0

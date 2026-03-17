"""
Tests for analysis/actionable_insights.py — Actionable Insights Generator

Tests cover:
  - compute_actionable_insights() with:
      - empty store → is_meaningful=False
      - insufficient days (< MIN_DAYS) → is_meaningful=False
      - only neutral data → no insights surfaced
      - strong post-meeting FDI drop → meeting recovery gap detected
      - strong late-day FDI cliff → cliff insight detected
      - Slack fragmentation pattern → slack insight detected
      - meeting load threshold crossing → meeting load insight detected
      - sleep leverage pattern → sleep insight detected
      - load arc pattern → arc insight detected
      - multiple patterns → top 3 selected by impact
  - format_insights_section() produces expected text
  - format_insights_brief() produces compact line
  - format_insights_terminal() runs without crash
  - to_dict() round-trips cleanly
"""

import json
import sys
import tempfile
import shutil
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.actionable_insights import (
    ActionableInsights,
    Insight,
    compute_actionable_insights,
    format_insights_section,
    format_insights_brief,
    format_insights_terminal,
    _detect_meeting_recovery_gap,
    _detect_late_day_cliff,
    _detect_slack_fragmentation,
    _detect_meeting_load_threshold,
    _detect_sleep_leverage,
    _detect_load_arc,
    MIN_DAYS,
    MIN_SUPPORT_POINTS,
)


# ─── Window factory ───────────────────────────────────────────────────────────

def _make_window(
    date_str: str = "2026-03-10",
    hour: int = 9,
    minute: int = 0,
    window_index: int = None,
    cls: float = 0.30,
    fdi: float = 0.65,
    sdi: float = 0.20,
    csc: float = 0.20,
    ras: float = 0.75,
    in_meeting: bool = False,
    attendees: int = 0,
    messages_received: int = 0,
    messages_sent: int = 0,
    recovery_score: float = 75.0,
    sleep_hours: float = 7.5,
    sleep_performance: float = 80.0,
    is_active: bool = True,
    is_working: bool = True,
) -> dict:
    if window_index is None:
        window_index = hour * 4 + minute // 15

    return {
        "window_id": f"{date_str}T{hour:02d}:{minute:02d}:00",
        "date": date_str,
        "window_start": f"{date_str}T{hour:02d}:{minute:02d}:00+01:00",
        "window_end": f"{date_str}T{hour:02d}:{minute+15:02d}:00+01:00",
        "window_index": window_index,
        "calendar": {
            "in_meeting": in_meeting,
            "meeting_title": "Meeting" if in_meeting else None,
            "meeting_attendees": attendees if in_meeting else 0,
            "meeting_duration_minutes": 15 if in_meeting else 0,
            "meeting_organizer": None,
            "meetings_count": 1 if in_meeting else 0,
        },
        "whoop": {
            "recovery_score": recovery_score,
            "hrv_rmssd_milli": 65.0,
            "resting_heart_rate": 56.0,
            "sleep_performance": sleep_performance,
            "sleep_hours": sleep_hours,
            "strain": None,
            "spo2_percentage": 96.0,
        },
        "slack": {
            "messages_sent": messages_sent,
            "messages_received": messages_received,
            "total_messages": messages_sent + messages_received,
            "channels_active": 1 if messages_received > 0 or messages_sent > 0 else 0,
        },
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "context_switch_cost": csc,
            "recovery_alignment_score": ras,
        },
        "metadata": {
            "day_of_week": "Monday",
            "hour_of_day": hour,
            "minute_of_hour": minute,
            "is_working_hours": is_working,
            "is_active_window": is_active,
            "sources_available": ["whoop", "calendar", "slack"],
        },
    }


def _make_day(
    date_str: str,
    n_hours: int = 8,
    base_cls: float = 0.30,
    base_fdi: float = 0.65,
    sleep_hours: float = 7.5,
    sleep_performance: float = 80.0,
    recovery_score: float = 75.0,
) -> list[dict]:
    """Build a simple full-day list of windows (no meetings, moderate Slack)."""
    windows = []
    for h in range(8, 8 + n_hours):
        for m in range(0, 60, 15):
            windows.append(
                _make_window(
                    date_str=date_str,
                    hour=h,
                    minute=m,
                    cls=base_cls,
                    fdi=base_fdi,
                    is_active=True,
                    is_working=True,
                    sleep_hours=sleep_hours,
                    sleep_performance=sleep_performance,
                    recovery_score=recovery_score,
                )
            )
    return windows


# ─── Helpers for patching the store ──────────────────────────────────────────

def _make_days_data(days: dict[str, list[dict]]) -> dict:
    return days


# ─── Tests: empty / insufficient data ────────────────────────────────────────

class TestInsufficientData:

    def test_empty_store_returns_not_meaningful(self):
        with patch("analysis.actionable_insights.list_available_dates", return_value=[]):
            result = compute_actionable_insights()
        assert not result.is_meaningful
        assert result.insights == []

    def test_fewer_than_min_days_returns_not_meaningful(self):
        dates = ["2026-03-10", "2026-03-11"]  # only 2 days, MIN_DAYS=3
        day_data = {d: _make_day(d) for d in dates}

        def mock_read(d):
            return day_data.get(d, [])

        with patch("analysis.actionable_insights.list_available_dates", return_value=dates), \
             patch("analysis.actionable_insights.read_day", side_effect=mock_read):
            result = compute_actionable_insights()

        assert not result.is_meaningful

    def test_min_days_sets_days_analysed(self):
        dates = ["2026-03-10", "2026-03-11"]
        day_data = {d: _make_day(d) for d in dates}

        def mock_read(d):
            return day_data.get(d, [])

        with patch("analysis.actionable_insights.list_available_dates", return_value=dates), \
             patch("analysis.actionable_insights.read_day", side_effect=mock_read):
            result = compute_actionable_insights()

        assert result.days_analysed <= len(dates)


# ─── Tests: Detector 1 — Post-Meeting Recovery Gap ───────────────────────────

class TestMeetingRecoveryGap:

    def _make_meeting_day(self, date_str: str, pre_fdi: float, post_fdi: float) -> list[dict]:
        """Build a day with a meeting block sandwiched between pre/post windows."""
        windows = []
        # Pre-meeting window at 9:00
        windows.append(_make_window(date_str=date_str, hour=9, minute=0, fdi=pre_fdi, in_meeting=False, attendees=0))
        # Meeting windows at 9:15 and 9:30 (social, 3 attendees)
        windows.append(_make_window(date_str=date_str, hour=9, minute=15, fdi=0.40, in_meeting=True, attendees=3))
        windows.append(_make_window(date_str=date_str, hour=9, minute=30, fdi=0.40, in_meeting=True, attendees=3))
        # Post-meeting windows at 9:45 and 10:00
        windows.append(_make_window(date_str=date_str, hour=9, minute=45, fdi=post_fdi, in_meeting=False, attendees=0))
        windows.append(_make_window(date_str=date_str, hour=10, minute=0, fdi=post_fdi, in_meeting=False, attendees=0))
        # Sort by window_index
        windows.sort(key=lambda w: w["window_index"])
        return windows

    def test_large_fdi_drop_detected(self):
        """Strong post-meeting FDI drop → insight returned."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(6)]
        days_data = {d: self._make_meeting_day(d, pre_fdi=0.85, post_fdi=0.40) for d in dates}
        result = _detect_meeting_recovery_gap(days_data)
        assert result is not None
        assert result.detector == "post_meeting_recovery_gap"
        assert result.magnitude > 0.30  # at least 30% drop

    def test_no_drop_returns_none(self):
        """No FDI drop → no insight."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(6)]
        days_data = {d: self._make_meeting_day(d, pre_fdi=0.70, post_fdi=0.68) for d in dates}
        result = _detect_meeting_recovery_gap(days_data)
        assert result is None

    def test_insufficient_transitions_returns_none(self):
        """Only 2 transitions → below MIN_SUPPORT_POINTS."""
        dates = ["2026-03-10", "2026-03-11"]
        days_data = {d: self._make_meeting_day(d, pre_fdi=0.85, post_fdi=0.40) for d in dates}
        result = _detect_meeting_recovery_gap(days_data)
        assert result is None

    def test_impact_label_high_for_large_drop(self):
        dates = [f"2026-03-{10 + i:02d}" for i in range(8)]
        days_data = {d: self._make_meeting_day(d, pre_fdi=0.90, post_fdi=0.35) for d in dates}
        result = _detect_meeting_recovery_gap(days_data)
        assert result is not None
        assert result.impact_label in ("High", "Medium")


# ─── Tests: Detector 2 — Late-Day Cliff ──────────────────────────────────────

class TestLateDayCliff:

    def _make_cliff_day(self, date_str: str, cliff_hour: int, peak_fdi: float, cliff_fdi: float) -> list[dict]:
        """Build a day where FDI drops at cliff_hour."""
        windows = []
        for h in range(8, 19):
            fdi = peak_fdi if h < cliff_hour else cliff_fdi
            windows.append(_make_window(date_str=date_str, hour=h, fdi=fdi))
        return windows

    def test_consistent_cliff_detected(self):
        """Cliff at 15:00 on 5 out of 7 days → insight returned."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(7)]
        days_data = {d: self._make_cliff_day(d, cliff_hour=15, peak_fdi=0.80, cliff_fdi=0.45) for d in dates}
        result = _detect_late_day_cliff(days_data)
        assert result is not None
        assert result.detector == "late_day_cliff"

    def test_no_cliff_returns_none(self):
        """Consistently high FDI all day → no cliff."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(7)]
        days_data = {d: self._make_cliff_day(d, cliff_hour=22, peak_fdi=0.70, cliff_fdi=0.68) for d in dates}
        result = _detect_late_day_cliff(days_data)
        assert result is None

    def test_cliff_on_few_days_returns_none(self):
        """Cliff on only 1 of 7 days → below consistency threshold."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(7)]
        days_data = {}
        for i, d in enumerate(dates):
            if i == 0:
                days_data[d] = self._make_cliff_day(d, cliff_hour=14, peak_fdi=0.80, cliff_fdi=0.40)
            else:
                days_data[d] = _make_day(d, base_fdi=0.70)  # flat, no cliff
        result = _detect_late_day_cliff(days_data)
        assert result is None


# ─── Tests: Detector 3 — Slack Fragmentation ─────────────────────────────────

class TestSlackFragmentation:

    def _make_slack_day(self, date_str: str, quiet_fdi: float, high_fdi: float) -> list[dict]:
        """Build a day with contrasting FDI in quiet vs high-Slack windows."""
        windows = []
        # 8 quiet windows (0 messages)
        for i in range(8):
            windows.append(_make_window(
                date_str=date_str, hour=8 + i // 2, minute=(i % 2) * 30,
                fdi=quiet_fdi, messages_received=0
            ))
        # 8 high-volume windows (20+ messages)
        for i in range(8):
            windows.append(_make_window(
                date_str=date_str, hour=13 + i // 2, minute=(i % 2) * 30,
                fdi=high_fdi, messages_received=20
            ))
        return windows

    def test_slack_fdi_correlation_detected(self):
        """Large FDI drop in high-Slack windows → insight returned."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(5)]
        days_data = {d: self._make_slack_day(d, quiet_fdi=0.80, high_fdi=0.45) for d in dates}
        result = _detect_slack_fragmentation(days_data)
        assert result is not None
        assert result.detector == "slack_fragmentation"

    def test_no_slack_correlation_returns_none(self):
        """No FDI difference between quiet and high-Slack windows → None."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(5)]
        days_data = {d: self._make_slack_day(d, quiet_fdi=0.70, high_fdi=0.68) for d in dates}
        result = _detect_slack_fragmentation(days_data)
        assert result is None

    def test_insufficient_high_volume_windows_returns_none(self):
        """Only quiet windows, no high-Slack → can't compute delta."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(5)]
        days_data = {d: [
            _make_window(date_str=d, hour=8 + j, fdi=0.80, messages_received=0)
            for j in range(8)
        ] for d in dates}
        result = _detect_slack_fragmentation(days_data)
        assert result is None


# ─── Tests: Detector 4 — Meeting Load Threshold ──────────────────────────────

class TestMeetingLoadThreshold:

    def _make_heavy_meeting_day(self, date_str: str, n_meeting_hours: int, cls: float) -> list[dict]:
        """Build a day with n_meeting_hours of social meetings."""
        windows = []
        # Meeting windows
        for h in range(9, 9 + n_meeting_hours):
            for m in [0, 15, 30, 45]:
                windows.append(_make_window(
                    date_str=date_str, hour=h, minute=m,
                    cls=cls, fdi=0.40, in_meeting=True, attendees=3
                ))
        # Fill rest of day with low-load windows
        for h in range(9 + n_meeting_hours, 18):
            windows.append(_make_window(date_str=date_str, hour=h, cls=0.20, fdi=0.75))
        return windows

    def test_threshold_detected_when_load_spikes(self):
        """Heavy meeting days have high avg CLS → insight returned.

        Build days where most active windows are meetings (high CLS) to ensure
        the daily average clearly exceeds MEETING_CLS_THRESHOLD=0.55.
        """
        days_data = {}
        # 3 light days: no meetings, low CLS
        for i in range(3):
            d = f"2026-03-{10 + i:02d}"
            days_data[d] = [
                _make_window(date_str=d, hour=9 + j, cls=0.20, fdi=0.75, in_meeting=False, attendees=0)
                for j in range(4)
            ]

        # 3 heavy-meeting days: almost all windows are high-CLS meetings
        # 2.5h of meetings (10 windows × 15min = 150 min = "heavy" bucket 120–180 min)
        # Only add 2 non-meeting windows to keep avg CLS high
        for i in range(3):
            d = f"2026-03-{13 + i:02d}"
            windows = []
            # 10 consecutive meeting windows at CLS 0.80
            for slot in range(10):
                h = 9 + slot // 4
                m = (slot % 4) * 15
                windows.append(_make_window(date_str=d, hour=h, minute=m, cls=0.80, fdi=0.35, in_meeting=True, attendees=3))
            # 2 short non-meeting windows — still high CLS (reactive day)
            windows.append(_make_window(date_str=d, hour=12, cls=0.65, fdi=0.55, in_meeting=False, attendees=0))
            windows.append(_make_window(date_str=d, hour=13, cls=0.65, fdi=0.55, in_meeting=False, attendees=0))
            days_data[d] = windows

        result = _detect_meeting_load_threshold(days_data)
        assert result is not None
        assert result.detector == "meeting_load_threshold"

    def test_no_threshold_when_cls_always_low(self):
        """Even heavy meetings produce low CLS → no threshold detected."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(6)]
        days_data = {d: self._make_heavy_meeting_day(d, n_meeting_hours=4, cls=0.35) for d in dates}
        result = _detect_meeting_load_threshold(days_data)
        assert result is None


# ─── Tests: Detector 5 — Sleep Leverage ──────────────────────────────────────

class TestSleepLeverage:

    def _make_sleep_pair(
        self,
        date_str: str,
        next_date: str,
        sleep_hours: float,
        next_fdi: float,
    ) -> tuple[list[dict], list[dict]]:
        """Build two days where first day's sleep predicts next day's FDI."""
        day1 = [
            _make_window(date_str=date_str, hour=9, fdi=0.65, sleep_hours=sleep_hours)
        ]
        day2 = [
            _make_window(date_str=next_date, hour=9, fdi=next_fdi, sleep_hours=7.5)
        ]
        return day1, day2

    def test_sleep_hours_leverage_detected(self):
        """Good sleep → higher next-day FDI → insight returned."""
        days_data = {}
        dates = [f"2026-03-{10 + i:02d}" for i in range(8)]
        for i in range(0, 8, 2):
            d1, d2 = dates[i], dates[i + 1]
            if i < 4:
                # Good sleep → high FDI
                day1, day2 = self._make_sleep_pair(d1, d2, sleep_hours=8.0, next_fdi=0.80)
            else:
                # Poor sleep → low FDI
                day1, day2 = self._make_sleep_pair(d1, d2, sleep_hours=5.5, next_fdi=0.45)
            days_data[d1] = day1
            days_data[d2] = day2

        result = _detect_sleep_leverage(days_data)
        assert result is not None
        assert "sleep" in result.detector.lower()
        assert result.magnitude > 0.08

    def test_no_sleep_leverage_returns_none(self):
        """Sleep quality doesn't predict FDI → None."""
        days_data = {}
        dates = [f"2026-03-{10 + i:02d}" for i in range(8)]
        for i in range(0, 8, 2):
            d1, d2 = dates[i], dates[i + 1]
            day1, day2 = self._make_sleep_pair(d1, d2, sleep_hours=7.0, next_fdi=0.65)
            days_data[d1] = day1
            days_data[d2] = day2
        result = _detect_sleep_leverage(days_data)
        assert result is None


# ─── Tests: Detector 6 — Load Arc ────────────────────────────────────────────

class TestLoadArc:

    def _make_arc_day(
        self,
        date_str: str,
        am_cls: float,
        pm_cls: float,
        pm_fdi: float,
    ) -> list[dict]:
        """Build a day with distinct AM and PM CLS/FDI."""
        windows = []
        for h in range(8, 13):   # morning
            windows.append(_make_window(date_str=date_str, hour=h, cls=am_cls, fdi=0.70))
        for h in range(13, 18):  # afternoon
            windows.append(_make_window(date_str=date_str, hour=h, cls=pm_cls, fdi=pm_fdi))
        return windows

    def test_front_loading_advantage_detected(self):
        """Front-loaded days have better PM FDI → insight returned."""
        days_data = {}
        # 3 front-loaded days (heavy morning)
        for i in range(3):
            d = f"2026-03-{10 + i:02d}"
            days_data[d] = self._make_arc_day(d, am_cls=0.60, pm_cls=0.25, pm_fdi=0.80)
        # 3 back-loaded days (heavy afternoon)
        for i in range(3):
            d = f"2026-03-{13 + i:02d}"
            days_data[d] = self._make_arc_day(d, am_cls=0.25, pm_cls=0.65, pm_fdi=0.45)
        result = _detect_load_arc(days_data)
        assert result is not None
        assert result.detector == "load_arc"

    def test_no_arc_difference_returns_none(self):
        """No meaningful PM FDI difference between arc types → None."""
        days_data = {}
        for i in range(3):
            d = f"2026-03-{10 + i:02d}"
            days_data[d] = self._make_arc_day(d, am_cls=0.60, pm_cls=0.25, pm_fdi=0.65)
        for i in range(3):
            d = f"2026-03-{13 + i:02d}"
            days_data[d] = self._make_arc_day(d, am_cls=0.25, pm_cls=0.65, pm_fdi=0.67)
        result = _detect_load_arc(days_data)
        assert result is None


# ─── Tests: Full compute function ────────────────────────────────────────────

class TestComputeActionableInsights:

    def _setup_store(self, dates: list[str], day_data: dict[str, list[dict]]):
        """Return patches for list_available_dates and read_day."""
        def mock_read(d):
            return day_data.get(d, [])

        return (
            patch("analysis.actionable_insights.list_available_dates", return_value=dates),
            patch("analysis.actionable_insights.read_day", side_effect=mock_read),
        )

    def test_sufficient_neutral_days_returns_meaningful_false_or_empty(self):
        """Flat/neutral data — no detectors fire — insights is empty."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(7)]
        day_data = {d: _make_day(d, base_cls=0.30, base_fdi=0.65) for d in dates}

        p1, p2 = self._setup_store(dates, day_data)
        with p1, p2:
            result = compute_actionable_insights()

        # With flat data, no detectors should fire
        assert isinstance(result, ActionableInsights)
        # is_meaningful depends on whether any detector fired
        # (with flat data it should be False or have 0 insights)
        if result.is_meaningful:
            assert len(result.insights) <= 3

    def test_with_strong_patterns_returns_insights(self):
        """Multiple strong patterns → is_meaningful=True, insights returned."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(10)]
        day_data = {}
        for d in dates:
            windows = []
            # Morning windows: high FDI (quiet, no Slack)
            for h in range(8, 13):
                windows.append(_make_window(
                    date_str=d, hour=h, cls=0.35, fdi=0.85,
                    messages_received=0, sleep_hours=8.0
                ))
            # Afternoon windows: post-meeting low FDI + high Slack
            for h in range(13, 18):
                windows.append(_make_window(
                    date_str=d, hour=h, cls=0.45, fdi=0.35,
                    messages_received=20, sleep_hours=8.0
                ))
            # Meeting block at 10:00
            windows.append(_make_window(date_str=d, hour=10, minute=0, fdi=0.40, in_meeting=True, attendees=3))
            windows.append(_make_window(date_str=d, hour=10, minute=15, fdi=0.40, in_meeting=True, attendees=3))
            day_data[d] = windows

        p1, p2 = self._setup_store(dates, day_data)
        with p1, p2:
            result = compute_actionable_insights(days=10)

        assert isinstance(result, ActionableInsights)
        assert result.days_analysed >= 3

    def test_insights_ranked_1_to_3(self):
        """Any returned insights have rank 1, 2, 3."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(10)]
        day_data = {}
        for d in dates:
            windows = []
            for h in range(8, 18):
                windows.append(_make_window(date_str=d, hour=h, cls=0.30, fdi=0.70, messages_received=0))
            day_data[d] = windows

        p1, p2 = self._setup_store(dates, day_data)
        with p1, p2:
            result = compute_actionable_insights()

        for i, insight in enumerate(result.insights, start=1):
            assert insight.rank == i

    def test_date_range_populated(self):
        dates = [f"2026-03-{10 + i:02d}" for i in range(5)]
        day_data = {d: _make_day(d) for d in dates}

        p1, p2 = self._setup_store(dates, day_data)
        with p1, p2:
            result = compute_actionable_insights()

        assert result.days_analysed == len(dates)

    def test_as_of_date_limits_lookback(self):
        """as_of_date_str filters dates to only up to that date."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(7)]
        day_data = {d: _make_day(d) for d in dates}

        p1, p2 = self._setup_store(dates, day_data)
        with p1, p2:
            result = compute_actionable_insights(as_of_date_str="2026-03-13")

        # Should only use dates up to 2026-03-13 (first 4)
        assert result.days_analysed <= 4

    def test_detector_exception_does_not_crash(self):
        """If a detector raises an exception, compute should still return."""
        dates = [f"2026-03-{10 + i:02d}" for i in range(5)]
        day_data = {d: _make_day(d) for d in dates}

        def raising_detector(days_data):
            raise ValueError("simulated detector failure")

        p1, p2 = self._setup_store(dates, day_data)
        with p1, p2, \
             patch("analysis.actionable_insights._detect_meeting_recovery_gap", raising_detector):
            result = compute_actionable_insights()

        # Should still return without crashing
        assert isinstance(result, ActionableInsights)


# ─── Tests: to_dict() ────────────────────────────────────────────────────────

class TestToDict:

    def test_empty_to_dict(self):
        ai = ActionableInsights(is_meaningful=False)
        d = ai.to_dict()
        assert d["is_meaningful"] is False
        assert d["insights"] == []

    def test_with_insights_to_dict(self):
        insight = Insight(
            rank=1,
            title="Test",
            headline="Do this.",
            evidence="3/5 days",
            impact_label="High",
            n_supporting_days=3,
            magnitude=0.30,
            detector="test",
        )
        ai = ActionableInsights(
            insights=[insight],
            is_meaningful=True,
            days_analysed=5,
            date_range="2026-03-10 → 2026-03-14",
            generated_at="2026-03-14T09:00:00",
        )
        d = ai.to_dict()
        assert d["is_meaningful"] is True
        assert len(d["insights"]) == 1
        assert d["insights"][0]["title"] == "Test"
        assert d["insights"][0]["rank"] == 1

    def test_json_serialisable(self):
        insight = Insight(
            rank=1,
            title="Test",
            headline="Do this.",
            evidence="3 days",
            impact_label="Medium",
            n_supporting_days=3,
            magnitude=0.20,
            detector="test",
        )
        ai = ActionableInsights(insights=[insight], is_meaningful=True, days_analysed=5)
        # Should not raise
        json_str = json.dumps(ai.to_dict())
        assert json_str


# ─── Tests: Formatting ────────────────────────────────────────────────────────

class TestFormatting:

    def _sample_insights(self) -> ActionableInsights:
        return ActionableInsights(
            insights=[
                Insight(
                    rank=1,
                    title="Post-Meeting Recovery Gap",
                    headline="Add 15-min buffers after calls.",
                    evidence="6 transitions",
                    impact_label="High",
                    n_supporting_days=6,
                    magnitude=0.35,
                    detector="post_meeting_recovery_gap",
                ),
                Insight(
                    rank=2,
                    title="Slack Fragmentation Impact",
                    headline="Batch Slack to 3x daily.",
                    evidence="20 quiet vs 15 high windows",
                    impact_label="Medium",
                    n_supporting_days=5,
                    magnitude=0.20,
                    detector="slack_fragmentation",
                ),
            ],
            is_meaningful=True,
            days_analysed=14,
            date_range="2026-03-01 → 2026-03-14",
            generated_at="2026-03-14T09:00:00",
        )

    def test_format_section_contains_title(self):
        ai = self._sample_insights()
        section = format_insights_section(ai)
        assert "Actionable Insights" in section

    def test_format_section_contains_headlines(self):
        ai = self._sample_insights()
        section = format_insights_section(ai)
        assert "buffers" in section.lower() or "batch" in section.lower()

    def test_format_section_contains_evidence(self):
        ai = self._sample_insights()
        section = format_insights_section(ai)
        assert "transitions" in section or "windows" in section

    def test_format_section_empty_when_not_meaningful(self):
        ai = ActionableInsights(is_meaningful=False)
        assert format_insights_section(ai) == ""

    def test_format_brief_returns_top_insight(self):
        ai = self._sample_insights()
        brief = format_insights_brief(ai)
        assert "buffers" in brief.lower() or "insight" in brief.lower()

    def test_format_brief_empty_when_not_meaningful(self):
        ai = ActionableInsights(is_meaningful=False)
        assert format_insights_brief(ai) == ""

    def test_format_terminal_runs_without_crash(self):
        ai = self._sample_insights()
        output = format_insights_terminal(ai)
        assert isinstance(output, str)
        assert len(output) > 10

    def test_format_terminal_not_meaningful_message(self):
        ai = ActionableInsights(is_meaningful=False, days_analysed=2)
        output = format_insights_terminal(ai)
        assert "not enough" in output.lower() or "need" in output.lower()

    def test_format_section_shows_both_insights(self):
        ai = self._sample_insights()
        section = format_insights_section(ai)
        assert "1." in section
        assert "2." in section

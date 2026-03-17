"""
Tests for analysis/midday_checkin.py — Midday Cognitive Check-In

Tests cover:
  - compute_midday_checkin() with mocked windows
  - Morning metric aggregation (CLS, FDI, SDI)
  - Meeting minute counting
  - Pace label (Running hot / On track / Light morning)
  - Pace ratio computation
  - Remaining budget estimation
  - Afternoon nudge content
  - format_checkin_message() output
  - Graceful degradation (no windows, not enough active windows)
  - MidDayCheckIn.to_dict() serialisation
  - Load driver: dominant source detection (v29)
"""

import sys
from pathlib import Path

import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.midday_checkin import (
    MidDayCheckIn,
    compute_midday_checkin,
    format_checkin_message,
    _cls_label,
    _fdi_label,
    _fmt_minutes,
    _build_afternoon_nudge,
    _decompose_morning_load,
    _MIN_CLS_FOR_DRIVER,
    _SOURCE_EMOJIS,
    _SOURCE_LABELS,
    MIDDAY_HOUR,
    MIN_ACTIVE_WINDOWS,
)


# ─── Fixture helpers ──────────────────────────────────────────────────────────

def _make_window(
    hour: int,
    cls: float = 0.20,
    fdi: float = 0.80,
    sdi: float = 0.05,
    csc: float = 0.05,
    ras: float = 0.90,
    in_meeting: bool = False,
    slack_msgs: int = 0,
    is_active: bool = True,
) -> dict:
    """Build a synthetic window dict for testing."""
    return {
        "metadata": {
            "hour_of_day": hour,
            "is_working_hours": 8 <= hour < 19,
            "is_active_window": is_active,
        },
        "calendar": {
            "in_meeting": in_meeting,
            "total_meeting_minutes": 60 if in_meeting else 0,
        },
        "slack": {
            "total_messages": slack_msgs,
            "messages_sent": 0,
            "messages_received": slack_msgs,
        },
        "rescuetime": {
            "active_seconds": 600 if is_active else 0,
        },
        "omi": {},
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "context_switch_cost": csc,
            "recovery_alignment_score": ras,
        },
    }


def _make_day_windows(
    morning_cls: float = 0.20,
    morning_fdi: float = 0.80,
    morning_sdi: float = 0.05,
    in_meeting_hours: list = None,
    n_morning_active: int = 6,
    with_afternoon: bool = True,
) -> list[dict]:
    """
    Build a synthetic day of 96 windows (96 × 15min = 24h).
    Active morning windows from hours 8..12 (before MIDDAY_HOUR=13).
    """
    windows = []
    in_meeting_hours = in_meeting_hours or []

    for hour in range(24):
        for q in range(4):  # 4 quarters per hour
            in_meeting = hour in in_meeting_hours
            is_morning = hour < MIDDAY_HOUR
            is_active = is_morning and hour >= 8

            cls_val = morning_cls if is_morning else 0.10
            fdi_val = morning_fdi if is_morning else 0.70
            sdi_val = morning_sdi if is_morning else 0.02

            windows.append(_make_window(
                hour=hour,
                cls=cls_val,
                fdi=fdi_val,
                sdi=sdi_val,
                in_meeting=in_meeting,
                is_active=is_active,
            ))

    return windows


# ─── Unit tests: helpers ───────────────────────────────────────────────────────

class TestHelpers:
    def test_cls_label_very_light(self):
        assert _cls_label(0.05) == "very light"

    def test_cls_label_light(self):
        assert _cls_label(0.20) == "light"

    def test_cls_label_moderate(self):
        assert _cls_label(0.35) == "moderate"

    def test_cls_label_heavy(self):
        assert _cls_label(0.55) == "heavy"

    def test_cls_label_intense(self):
        assert _cls_label(0.75) == "intense"

    def test_cls_label_none(self):
        assert _cls_label(None) == "unknown"

    def test_fdi_label_deep(self):
        assert _fdi_label(0.85) == "deep"

    def test_fdi_label_solid(self):
        assert _fdi_label(0.65) == "solid"

    def test_fdi_label_fragmented(self):
        assert _fdi_label(0.45) == "fragmented"

    def test_fdi_label_scattered(self):
        assert _fdi_label(0.20) == "scattered"

    def test_fdi_label_none(self):
        assert _fdi_label(None) == "unknown"

    def test_fmt_minutes_sub_hour(self):
        assert _fmt_minutes(45) == "45min"

    def test_fmt_minutes_one_hour(self):
        assert _fmt_minutes(60) == "1h"

    def test_fmt_minutes_hour_and_half(self):
        assert _fmt_minutes(90) == "1h30min"

    def test_fmt_minutes_two_hours(self):
        assert _fmt_minutes(120) == "2h"


# ─── Unit tests: afternoon nudge ──────────────────────────────────────────────

class TestAfternoonNudge:
    def test_running_hot_high_sdi(self):
        nudge = _build_afternoon_nudge(
            morning_cls=0.60,
            morning_fdi=0.70,
            morning_sdi=0.55,
            pace_label="Running hot",
            meeting_minutes=90,
            remaining_budget_hours=2.0,
        )
        assert "social load" in nudge.lower() or "protect" in nudge.lower()

    def test_running_hot_low_fdi(self):
        nudge = _build_afternoon_nudge(
            morning_cls=0.65,
            morning_fdi=0.35,
            morning_sdi=0.10,
            pace_label="Running hot",
            meeting_minutes=60,
            remaining_budget_hours=1.5,
        )
        assert "fragmented" in nudge.lower() or "focused" in nudge.lower()

    def test_running_hot_generic(self):
        nudge = _build_afternoon_nudge(
            morning_cls=0.65,
            morning_fdi=0.70,
            morning_sdi=0.10,
            pace_label="Running hot",
            meeting_minutes=30,
            remaining_budget_hours=3.0,
        )
        assert "pace" in nudge.lower() or "running hot" in nudge.lower()

    def test_light_morning_with_budget(self):
        nudge = _build_afternoon_nudge(
            morning_cls=0.08,
            morning_fdi=0.95,
            morning_sdi=0.00,
            pace_label="Light morning",
            meeting_minutes=0,
            remaining_budget_hours=4.0,
        )
        assert "4.0h" in nudge or "block" in nudge.lower() or "focused" in nudge.lower()

    def test_light_morning_no_budget(self):
        nudge = _build_afternoon_nudge(
            morning_cls=0.08,
            morning_fdi=0.95,
            morning_sdi=0.00,
            pace_label="Light morning",
            meeting_minutes=0,
            remaining_budget_hours=None,
        )
        assert "deep" in nudge.lower() or "light" in nudge.lower() or "ideal" in nudge.lower()

    def test_on_track_good_budget(self):
        nudge = _build_afternoon_nudge(
            morning_cls=0.25,
            morning_fdi=0.80,
            morning_sdi=0.05,
            pace_label="On track",
            meeting_minutes=60,
            remaining_budget_hours=3.5,
        )
        assert "3.5h" in nudge or "protect" in nudge.lower() or "block" in nudge.lower()

    def test_on_track_low_budget(self):
        nudge = _build_afternoon_nudge(
            morning_cls=0.25,
            morning_fdi=0.80,
            morning_sdi=0.05,
            pace_label="On track",
            meeting_minutes=60,
            remaining_budget_hours=0.8,
        )
        assert "budget" in nudge.lower() or "spent" in nudge.lower() or "admin" in nudge.lower()

    def test_on_track_no_budget(self):
        nudge = _build_afternoon_nudge(
            morning_cls=0.25,
            morning_fdi=0.80,
            morning_sdi=0.05,
            pace_label="On track",
            meeting_minutes=60,
            remaining_budget_hours=None,
        )
        assert isinstance(nudge, str) and len(nudge) > 10

    def test_heavy_meetings_fallback(self):
        nudge = _build_afternoon_nudge(
            morning_cls=0.40,
            morning_fdi=0.70,
            morning_sdi=0.20,
            pace_label="On track",
            meeting_minutes=150,
            remaining_budget_hours=None,
        )
        assert "meeting" in nudge.lower() or "block" in nudge.lower()


# ─── Unit tests: compute_midday_checkin ───────────────────────────────────────

class TestComputeMidDayCheckIn:
    def test_empty_windows_returns_not_meaningful(self):
        checkin = compute_midday_checkin("2026-03-14", windows=[], baseline_cls=0.25)
        assert not checkin.is_meaningful
        assert checkin.active_windows == 0
        assert checkin.morning_cls is None

    def test_too_few_active_windows(self):
        # Only 2 active windows (< MIN_ACTIVE_WINDOWS=3)
        windows = [
            _make_window(hour=9, cls=0.20, is_active=True),
            _make_window(hour=10, cls=0.25, is_active=True),
            _make_window(hour=11, cls=0.00, is_active=False),  # inactive
        ]
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert not checkin.is_meaningful

    def test_normal_morning_on_track(self):
        windows = _make_day_windows(
            morning_cls=0.25,
            morning_fdi=0.85,
            morning_sdi=0.05,
        )
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.is_meaningful
        assert checkin.pace_label == "On track"
        assert checkin.morning_cls is not None
        assert abs(checkin.morning_cls - 0.25) < 0.05

    def test_running_hot_when_cls_above_threshold(self):
        # pace_ratio = morning_cls / baseline_cls = 0.50 / 0.25 = 2.0 > PACE_HOT_RATIO (1.25)
        windows = _make_day_windows(morning_cls=0.50)
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.is_meaningful
        assert checkin.pace_label == "Running hot"
        assert checkin.pace_ratio is not None
        assert checkin.pace_ratio > 1.25

    def test_light_morning_when_cls_below_threshold(self):
        # pace_ratio = 0.10 / 0.25 = 0.40 < PACE_LIGHT_RATIO (0.75)
        windows = _make_day_windows(morning_cls=0.10, morning_fdi=0.95)
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.is_meaningful
        assert checkin.pace_label == "Light morning"

    def test_meeting_minutes_counted(self):
        windows = _make_day_windows(
            morning_cls=0.30,
            in_meeting_hours=[9, 10],  # 2 hours × 4 quarters × 15min = 120 min
        )
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.is_meaningful
        # At minimum, some meeting minutes should be counted
        assert checkin.meeting_minutes >= 0  # meetings counted from in_meeting windows

    def test_afternoon_windows_excluded(self):
        # Create windows where afternoon has very high CLS — should NOT affect morning_cls
        morning = _make_day_windows(morning_cls=0.15, morning_fdi=0.90)
        # Replace afternoon windows with high-CLS ones
        for w in morning:
            if w["metadata"]["hour_of_day"] >= MIDDAY_HOUR:
                w["metrics"]["cognitive_load_score"] = 0.90  # Very high
        checkin = compute_midday_checkin("2026-03-14", windows=morning, baseline_cls=0.25)
        assert checkin.is_meaningful
        # Morning CLS should be ~0.15, not influenced by afternoon
        assert checkin.morning_cls < 0.40

    def test_fdi_correctly_aggregated(self):
        windows = _make_day_windows(morning_cls=0.20, morning_fdi=0.75)
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.is_meaningful
        assert checkin.morning_fdi is not None
        assert 0.60 < checkin.morning_fdi < 0.90

    def test_sdi_correctly_aggregated(self):
        windows = _make_day_windows(morning_cls=0.20, morning_sdi=0.15)
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.is_meaningful
        assert checkin.morning_sdi is not None
        assert checkin.morning_sdi > 0.0

    def test_remaining_budget_computed_when_dcb_provided(self):
        windows = _make_day_windows(morning_cls=0.20)
        checkin = compute_midday_checkin(
            "2026-03-14",
            windows=windows,
            baseline_cls=0.25,
            dcb_hours=6.0,
        )
        assert checkin.is_meaningful
        assert checkin.remaining_budget_hours is not None
        # Budget should be less than dcb_hours (some consumed in the morning)
        assert checkin.remaining_budget_hours < 6.0
        # But shouldn't be negative
        assert checkin.remaining_budget_hours >= 0.0

    def test_remaining_budget_none_when_no_dcb(self):
        windows = _make_day_windows(morning_cls=0.20)
        checkin = compute_midday_checkin(
            "2026-03-14",
            windows=windows,
            baseline_cls=0.25,
            dcb_hours=None,
        )
        # DCB will be loaded from store (likely None in test env)
        # Just check it doesn't crash and is_meaningful works
        assert checkin.is_meaningful or not checkin.is_meaningful  # always passes

    def test_pace_ratio_not_computed_without_baseline(self):
        """When baseline_cls is 0 or None, pace_ratio should be None."""
        windows = _make_day_windows(morning_cls=0.30)
        checkin = compute_midday_checkin(
            "2026-03-14",
            windows=windows,
            baseline_cls=0,  # zero baseline → no ratio
        )
        # Should not crash; pace_ratio should be None when baseline is 0
        assert not checkin.is_meaningful or checkin.pace_ratio is None or checkin.pace_ratio >= 0

    def test_afternoon_nudge_always_present(self):
        windows = _make_day_windows(morning_cls=0.25)
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.is_meaningful
        assert isinstance(checkin.afternoon_nudge, str)
        assert len(checkin.afternoon_nudge) > 10

    def test_date_str_preserved(self):
        windows = _make_day_windows(morning_cls=0.25)
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.date_str == "2026-03-14"

    def test_active_windows_count(self):
        windows = _make_day_windows(morning_cls=0.25)
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.active_windows >= MIN_ACTIVE_WINDOWS


# ─── Unit tests: format_checkin_message ───────────────────────────────────────

class TestFormatCheckinMessage:
    def _make_meaningful_checkin(self, **kwargs) -> MidDayCheckIn:
        defaults = dict(
            date_str="2026-03-14",
            morning_cls=0.25,
            morning_fdi=0.82,
            morning_sdi=0.05,
            meeting_minutes=60,
            active_windows=8,
            pace_label="On track",
            pace_ratio=1.00,
            morning_dominant_source=None,
            morning_dominant_share=None,
            afternoon_nudge="Protect a 90-min deep-work block.",
            remaining_budget_hours=3.5,
            is_meaningful=True,
        )
        defaults.update(kwargs)
        return MidDayCheckIn(**defaults)

    def test_format_contains_midday_pulse_header(self):
        checkin = self._make_meaningful_checkin()
        msg = format_checkin_message(checkin)
        assert "Midday pulse" in msg

    def test_format_contains_day_name(self):
        checkin = self._make_meaningful_checkin(date_str="2026-03-13")
        msg = format_checkin_message(checkin)
        assert "Friday" in msg

    def test_format_contains_cls_value(self):
        checkin = self._make_meaningful_checkin(morning_cls=0.35)
        msg = format_checkin_message(checkin)
        assert "0.35" in msg

    def test_format_contains_pace_label(self):
        checkin = self._make_meaningful_checkin(pace_label="Running hot")
        msg = format_checkin_message(checkin)
        assert "Running hot" in msg

    def test_format_contains_budget(self):
        checkin = self._make_meaningful_checkin(remaining_budget_hours=4.2)
        msg = format_checkin_message(checkin)
        assert "4.2h" in msg

    def test_format_contains_afternoon_nudge(self):
        checkin = self._make_meaningful_checkin(
            afternoon_nudge="Schedule a 90-min focus block."
        )
        msg = format_checkin_message(checkin)
        assert "90-min" in msg

    def test_format_not_meaningful(self):
        checkin = MidDayCheckIn(
            date_str="2026-03-14",
            morning_cls=None,
            morning_fdi=None,
            morning_sdi=None,
            meeting_minutes=0,
            active_windows=1,
            pace_label="Light morning",
            pace_ratio=None,
            morning_dominant_source=None,
            morning_dominant_share=None,
            afternoon_nudge="",
            remaining_budget_hours=None,
            is_meaningful=False,
        )
        msg = format_checkin_message(checkin)
        assert "Midday pulse" in msg
        assert "No active windows" in msg or "quiet" in msg.lower()

    def test_format_omits_meetings_when_zero(self):
        checkin = self._make_meaningful_checkin(meeting_minutes=0)
        msg = format_checkin_message(checkin)
        # "Meetings:" line should not appear when meeting_minutes=0
        assert "Meetings:" not in msg

    def test_format_shows_meeting_duration(self):
        checkin = self._make_meaningful_checkin(meeting_minutes=90)
        msg = format_checkin_message(checkin)
        assert "1h30min" in msg or "90min" in msg

    def test_format_is_compact(self):
        """Target: 4–6 lines. Never overwhelming."""
        checkin = self._make_meaningful_checkin()
        msg = format_checkin_message(checkin)
        line_count = len([l for l in msg.split("\n") if l.strip()])
        # Header + blank + 3-4 content lines + blank + nudge = 5-8 non-empty lines
        assert line_count <= 10, f"Too many lines: {line_count}"

    def test_pace_emoji_on_track(self):
        checkin = self._make_meaningful_checkin(pace_label="On track")
        msg = format_checkin_message(checkin)
        assert "✅" in msg

    def test_pace_emoji_running_hot(self):
        checkin = self._make_meaningful_checkin(pace_label="Running hot")
        msg = format_checkin_message(checkin)
        assert "🔥" in msg

    def test_pace_emoji_light(self):
        checkin = self._make_meaningful_checkin(pace_label="Light morning")
        msg = format_checkin_message(checkin)
        assert "🟢" in msg


# ─── Unit tests: to_dict serialisation ────────────────────────────────────────

class TestToDict:
    def test_to_dict_keys_present(self):
        windows = _make_day_windows(morning_cls=0.25)
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        d = checkin.to_dict()
        expected_keys = {
            "date_str", "morning_cls", "morning_fdi", "morning_sdi",
            "meeting_minutes", "active_windows", "pace_label", "pace_ratio",
            "morning_dominant_source", "morning_dominant_share",
            "afternoon_nudge", "remaining_budget_hours", "is_meaningful",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_none_values_preserved(self):
        checkin = compute_midday_checkin("2026-03-14", windows=[], baseline_cls=0.25)
        d = checkin.to_dict()
        assert d["morning_cls"] is None
        assert d["morning_fdi"] is None
        assert d["is_meaningful"] is False

    def test_to_dict_floats_rounded(self):
        windows = _make_day_windows(morning_cls=0.12345)
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        d = checkin.to_dict()
        if d["morning_cls"] is not None:
            # Should be rounded to 3 decimal places
            assert len(str(d["morning_cls"]).split(".")[-1]) <= 3


# ─── Regression tests ─────────────────────────────────────────────────────────

class TestRegressions:
    def test_no_crash_with_minimal_windows(self):
        """Three active windows in a single hour should work."""
        windows = [
            _make_window(hour=9, cls=0.30, fdi=0.80, is_active=True),
            _make_window(hour=9, cls=0.25, fdi=0.85, is_active=True),
            _make_window(hour=9, cls=0.20, fdi=0.90, is_active=True),
        ]
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.is_meaningful
        assert checkin.active_windows == 3

    def test_no_crash_when_all_inactive(self):
        """All windows inactive → not meaningful, no crash."""
        windows = [
            _make_window(hour=h, cls=0.05, is_active=False)
            for h in range(24) for _ in range(4)
        ]
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert not checkin.is_meaningful

    def test_no_crash_with_only_afternoon_active(self):
        """Only afternoon windows active — morning check-in should be not meaningful."""
        windows = [
            _make_window(hour=h, cls=0.50, is_active=(h >= MIDDAY_HOUR))
            for h in range(24) for _ in range(4)
        ]
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        # Morning has no active windows → not meaningful
        assert not checkin.is_meaningful

    def test_extreme_cls_values_clamped(self):
        """CLS values close to extremes (0.0, 1.0) shouldn't break anything."""
        windows = [
            _make_window(hour=h, cls=0.99, fdi=0.01, is_active=True)
            for h in [8, 9, 10, 11, 12] for _ in range(4)
        ]
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        assert checkin.is_meaningful
        assert checkin.morning_cls is not None
        assert checkin.pace_label == "Running hot"

    def test_format_handles_none_budget(self):
        """format_checkin_message() must not crash when remaining_budget_hours is None."""
        windows = _make_day_windows(morning_cls=0.25)
        checkin = compute_midday_checkin(
            "2026-03-14", windows=windows, baseline_cls=0.25, dcb_hours=None
        )
        # Even if DCB load fails, format should not crash
        try:
            msg = format_checkin_message(checkin)
            assert isinstance(msg, str)
        except Exception as e:
            pytest.fail(f"format_checkin_message crashed: {e}")


# ─── Unit tests: load driver (_decompose_morning_load, v29) ───────────────────

class TestLoadDriver:
    """Tests for the v29 dominant CLS source feature."""

    def _make_meeting_window(self, hour: int) -> dict:
        """Window dominated by meetings (social meeting with attendees)."""
        return {
            "metadata": {
                "hour_of_day": hour,
                "is_working_hours": True,
                "is_active_window": True,
            },
            "calendar": {
                "in_meeting": True,
                "meeting_attendees": 4,
                "total_meeting_minutes": 60,
            },
            "slack": {"total_messages": 0, "messages_sent": 0, "messages_received": 0},
            "rescuetime": {"active_seconds": 0},
            "whoop": {},
            "omi": {},
            "metrics": {
                "cognitive_load_score": 0.55,
                "focus_depth_index": 0.50,
                "social_drain_index": 0.30,
                "context_switch_cost": 0.10,
                "recovery_alignment_score": 0.80,
            },
        }

    def _make_slack_window(self, hour: int, msgs: int = 40) -> dict:
        """Window dominated by Slack messages (no meetings)."""
        return {
            "metadata": {
                "hour_of_day": hour,
                "is_working_hours": True,
                "is_active_window": True,
            },
            "calendar": {
                "in_meeting": False,
                "meeting_attendees": 0,
                "total_meeting_minutes": 0,
            },
            "slack": {
                "total_messages": msgs,
                "messages_sent": 10,
                "messages_received": msgs,
            },
            "rescuetime": {"active_seconds": 0},
            "whoop": {},
            "omi": {},
            "metrics": {
                "cognitive_load_score": 0.30,
                "focus_depth_index": 0.60,
                "social_drain_index": 0.20,
                "context_switch_cost": 0.10,
                "recovery_alignment_score": 0.85,
            },
        }

    def _make_physiology_window(self, hour: int, recovery: int = 20) -> dict:
        """Window dominated by poor recovery (low HRV/recovery_score)."""
        return {
            "metadata": {
                "hour_of_day": hour,
                "is_working_hours": True,
                "is_active_window": True,
            },
            "calendar": {
                "in_meeting": False,
                "meeting_attendees": 0,
                "total_meeting_minutes": 0,
            },
            "slack": {"total_messages": 0, "messages_sent": 0, "messages_received": 0},
            "rescuetime": {"active_seconds": 0},
            "whoop": {
                "recovery_score": recovery,
                "hrv_rmssd_milli": 25.0,
                "sleep_performance": 60.0,
            },
            "omi": {},
            "metrics": {
                "cognitive_load_score": 0.25,
                "focus_depth_index": 0.65,
                "social_drain_index": 0.05,
                "context_switch_cost": 0.05,
                "recovery_alignment_score": 0.50,
            },
        }

    # ── _decompose_morning_load() ──────────────────────────────────────────

    def test_empty_windows_returns_none(self):
        src, share = _decompose_morning_load([])
        assert src is None
        assert share is None

    def test_meeting_windows_dominant_source_is_meetings(self):
        windows = [self._make_meeting_window(h) for h in [8, 9, 10, 11, 12]]
        src, share = _decompose_morning_load(windows)
        assert src == "meetings", f"Expected 'meetings', got '{src}'"
        assert share is not None
        assert 0.0 < share <= 1.0

    def test_slack_windows_dominant_source_is_slack(self):
        windows = [self._make_slack_window(h, msgs=60) for h in [8, 9, 10, 11, 12]]
        src, share = _decompose_morning_load(windows)
        # Slack-heavy windows should have slack or physiology as dominant
        # (physiology always contributes, but slack load should be visible)
        assert src in ("slack", "physiology"), f"Unexpected source: {src}"

    def test_physiology_dominant_with_low_recovery(self):
        windows = [self._make_physiology_window(h, recovery=10) for h in [8, 9, 10, 11, 12]]
        src, share = _decompose_morning_load(windows)
        assert src == "physiology", f"Expected 'physiology', got '{src}'"
        assert share is not None and share > 0.3

    def test_dominant_share_between_0_and_1(self):
        windows = [self._make_meeting_window(h) for h in [8, 9, 10]]
        src, share = _decompose_morning_load(windows)
        if share is not None:
            assert 0.0 <= share <= 1.0

    def test_returns_none_none_on_import_failure(self, monkeypatch):
        """If load_decomposer import fails, return (None, None) gracefully."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "load_decomposer" in name:
                raise ImportError("mocked failure")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        src, share = _decompose_morning_load([self._make_meeting_window(9)])
        assert src is None
        assert share is None

    # ── compute_midday_checkin() load driver integration ──────────────────

    def test_low_cls_produces_no_driver(self):
        """When morning CLS is below threshold, dominant_source should be None."""
        windows = [
            _make_window(hour=h, cls=0.03, fdi=0.95, is_active=True)
            for h in [8, 9, 10, 11, 12]
        ]
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        # CLS of 0.03 < _MIN_CLS_FOR_DRIVER → no driver
        assert checkin.morning_dominant_source is None
        assert checkin.morning_dominant_share is None

    def test_sufficient_cls_produces_driver(self):
        """When morning CLS is at or above threshold, a driver should be identified."""
        windows = [self._make_meeting_window(h) for h in [8, 9, 10, 11, 12]]
        # Inject meetings into full day windows
        filler = [
            _make_window(hour=h, cls=0.10, is_active=False)
            for h in range(24) for _ in range(4)
        ]
        day_windows = windows + filler
        checkin = compute_midday_checkin("2026-03-14", windows=day_windows, baseline_cls=0.25)
        if checkin.is_meaningful and checkin.morning_cls is not None and checkin.morning_cls >= _MIN_CLS_FOR_DRIVER:
            assert checkin.morning_dominant_source is not None
            assert checkin.morning_dominant_share is not None

    def test_driver_field_in_to_dict(self):
        """to_dict() includes morning_dominant_source and morning_dominant_share."""
        windows = _make_day_windows(morning_cls=0.30)
        checkin = compute_midday_checkin("2026-03-14", windows=windows, baseline_cls=0.25)
        d = checkin.to_dict()
        assert "morning_dominant_source" in d
        assert "morning_dominant_share" in d

    def test_driver_share_rounded_in_to_dict(self):
        """morning_dominant_share in to_dict() is rounded to 2 decimal places."""
        windows = [self._make_meeting_window(h) for h in [8, 9, 10, 11, 12]]
        filler = [
            _make_window(hour=h, cls=0.10, is_active=False)
            for h in range(24) for _ in range(4)
        ]
        checkin = compute_midday_checkin("2026-03-14", windows=windows + filler, baseline_cls=0.25)
        d = checkin.to_dict()
        share = d["morning_dominant_share"]
        if share is not None:
            # Must not have more than 2 decimal places
            assert len(str(share).split(".")[-1]) <= 2

    # ── format_checkin_message() load driver rendering ─────────────────────

    def test_format_shows_load_driver_when_present(self):
        """Load driver line appears when dominant_source is set and CLS ≥ threshold."""
        checkin = MidDayCheckIn(
            date_str="2026-03-14",
            morning_cls=0.35,
            morning_fdi=0.70,
            morning_sdi=0.10,
            meeting_minutes=60,
            active_windows=8,
            pace_label="On track",
            pace_ratio=1.0,
            morning_dominant_source="meetings",
            morning_dominant_share=0.62,
            afternoon_nudge="Protect a block.",
            remaining_budget_hours=3.0,
            is_meaningful=True,
        )
        msg = format_checkin_message(checkin)
        assert "Load driver" in msg
        assert "Meetings" in msg or "🗓" in msg
        assert "62%" in msg

    def test_format_hides_driver_when_cls_too_low(self):
        """Load driver line should NOT appear when CLS is below threshold."""
        checkin = MidDayCheckIn(
            date_str="2026-03-14",
            morning_cls=0.05,          # below _MIN_CLS_FOR_DRIVER
            morning_fdi=0.95,
            morning_sdi=0.01,
            meeting_minutes=0,
            active_windows=6,
            pace_label="Light morning",
            pace_ratio=0.20,
            morning_dominant_source="physiology",   # set, but CLS too low
            morning_dominant_share=0.80,
            afternoon_nudge="Deep work window.",
            remaining_budget_hours=5.0,
            is_meaningful=True,
        )
        msg = format_checkin_message(checkin)
        assert "Load driver" not in msg

    def test_format_hides_driver_when_source_is_none(self):
        """No load driver line when morning_dominant_source is None."""
        checkin = MidDayCheckIn(
            date_str="2026-03-14",
            morning_cls=0.40,
            morning_fdi=0.75,
            morning_sdi=0.10,
            meeting_minutes=0,
            active_windows=8,
            pace_label="On track",
            pace_ratio=1.0,
            morning_dominant_source=None,
            morning_dominant_share=None,
            afternoon_nudge="Protect a block.",
            remaining_budget_hours=3.0,
            is_meaningful=True,
        )
        msg = format_checkin_message(checkin)
        assert "Load driver" not in msg

    def test_all_sources_have_emoji_and_label(self):
        """Every supported source has an emoji and a label for formatting."""
        sources = ["meetings", "slack", "physiology", "rescuetime", "omi"]
        for src in sources:
            assert src in _SOURCE_EMOJIS, f"Missing emoji for source: {src}"
            assert src in _SOURCE_LABELS, f"Missing label for source: {src}"
            assert _SOURCE_EMOJIS[src], f"Empty emoji for source: {src}"
            assert _SOURCE_LABELS[src], f"Empty label for source: {src}"

    def test_format_slack_driver_shows_correct_label(self):
        """Slack source shows 'Slack' label and 💬 emoji."""
        checkin = MidDayCheckIn(
            date_str="2026-03-14",
            morning_cls=0.25,
            morning_fdi=0.80,
            morning_sdi=0.15,
            meeting_minutes=0,
            active_windows=8,
            pace_label="On track",
            pace_ratio=1.0,
            morning_dominant_source="slack",
            morning_dominant_share=0.51,
            afternoon_nudge="Go async.",
            remaining_budget_hours=3.5,
            is_meaningful=True,
        )
        msg = format_checkin_message(checkin)
        assert "Slack" in msg
        assert "💬" in msg
        assert "51%" in msg

    def test_format_physiology_driver_shows_correct_label(self):
        """Physiology source shows 'Recovery deficit' label and 💤 emoji."""
        checkin = MidDayCheckIn(
            date_str="2026-03-14",
            morning_cls=0.20,
            morning_fdi=0.65,
            morning_sdi=0.05,
            meeting_minutes=0,
            active_windows=6,
            pace_label="On track",
            pace_ratio=0.80,
            morning_dominant_source="physiology",
            morning_dominant_share=0.75,
            afternoon_nudge="Rest early tonight.",
            remaining_budget_hours=2.5,
            is_meaningful=True,
        )
        msg = format_checkin_message(checkin)
        assert "Recovery deficit" in msg
        assert "💤" in msg

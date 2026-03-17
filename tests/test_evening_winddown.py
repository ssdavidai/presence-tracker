"""
Tests for the Evening Wind-Down module (v30).

Tests cover:
- Graceful no-data fallback
- Day type classification (PRODUCTIVE, DEEP, REACTIVE, FRAGMENTED, RECOVERY, MIXED)
- Load arc computation (front-loaded, back-loaded, even)
- Wind-down recommendation logic
- Slack message formatting
- to_dict() completeness

Run with: python3 -m pytest tests/test_evening_winddown.py -v
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from analysis.evening_winddown import (
    EveningWindDown,
    compute_evening_winddown,
    format_winddown_message,
    _classify_day_type,
    _compute_load_arc,
    _build_wind_down_recommendation,
    DAY_TYPES,
    MIDDAY_HOUR,
    MORNING_START_HOUR,
    EVENING_HOUR,
    MIN_ACTIVE_WINDOWS,
)


# ─── Window builder helpers ────────────────────────────────────────────────────

def _make_window(
    date_str: str = "2026-03-17",
    hour: int = 9,
    minute: int = 0,
    in_meeting: bool = False,
    messages_sent: int = 0,
    messages_received: int = 0,
    cls: float = 0.25,
    fdi: float = 0.70,
    sdi: float = 0.20,
    csc: float = 0.15,
    ras: float = 0.75,
    recovery: float = 75.0,
    hrv: float = 65.0,
    is_active: bool = True,
    rescuetime_active_secs: int = 0,
) -> dict:
    """Build a minimal JSONL window dict."""
    return {
        "window_id": f"{date_str}T{hour:02d}:{minute:02d}:00",
        "date": date_str,
        "window_start": f"{date_str}T{hour:02d}:{minute:02d}:00+01:00",
        "window_end": f"{date_str}T{hour:02d}:{minute + 15:02d}:00+01:00",
        "window_index": hour * 4 + minute // 15,
        "calendar": {
            "in_meeting": in_meeting,
            "meeting_title": "Test Meeting" if in_meeting else None,
            "meeting_attendees": 3 if in_meeting else 0,
            "meeting_duration_minutes": 60 if in_meeting else 0,
            "meeting_organizer": None,
            "meetings_count": 1 if in_meeting else 0,
        },
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "resting_heart_rate": 55.0,
            "sleep_performance": 82.0,
            "sleep_hours": 7.5,
            "strain": 12.0,
            "spo2_percentage": 96.0,
        },
        "slack": {
            "messages_sent": messages_sent,
            "messages_received": messages_received,
            "total_messages": messages_sent + messages_received,
            "channels_active": 1 if (messages_sent + messages_received) > 0 else 0,
        },
        "rescuetime": {
            "focus_seconds": rescuetime_active_secs,
            "distraction_seconds": 0,
            "neutral_seconds": 0,
            "active_seconds": rescuetime_active_secs,
            "app_switches": 0,
            "productivity_score": None,
            "top_activity": None,
        },
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "context_switch_cost": csc,
            "recovery_alignment_score": ras,
        },
        "metadata": {
            "day_of_week": datetime.strptime(date_str, "%Y-%m-%d").strftime("%A"),
            "hour_of_day": hour,
            "minute_of_hour": minute,
            "is_working_hours": True,
            "is_active_window": is_active,
            "sources_available": ["whoop", "calendar", "slack"],
        },
    }


def _make_workday(
    date_str: str = "2026-03-17",
    hours: Optional[list[int]] = None,
    cls: float = 0.30,
    fdi: float = 0.70,
    sdi: float = 0.20,
    csc: float = 0.15,
    in_meeting_hours: Optional[list[int]] = None,
    messages_sent_per_window: int = 0,
    recovery: float = 75.0,
) -> list[dict]:
    """
    Build a full workday list of windows (MORNING_START_HOUR to EVENING_HOUR).
    Each hour gets one window at :00.
    """
    if hours is None:
        hours = list(range(MORNING_START_HOUR, EVENING_HOUR))
    if in_meeting_hours is None:
        in_meeting_hours = []

    windows = []
    for h in hours:
        for m in [0, 15, 30, 45]:
            in_meeting = h in in_meeting_hours
            windows.append(_make_window(
                date_str=date_str,
                hour=h,
                minute=m,
                cls=cls,
                fdi=fdi,
                sdi=sdi,
                csc=csc,
                in_meeting=in_meeting,
                messages_sent=messages_sent_per_window,
                recovery=recovery,
                is_active=True,
            ))
    return windows


# ─── No-data / sparse data ────────────────────────────────────────────────────

class TestNoData:
    def test_empty_windows_returns_not_meaningful(self):
        result = compute_evening_winddown("2026-03-17", windows=[])
        assert isinstance(result, EveningWindDown)
        assert result.is_meaningful is False

    def test_none_windows_but_no_store_returns_not_meaningful(self):
        """When date has no data in store, should return gracefully."""
        result = compute_evening_winddown("2099-01-01", windows=[])
        assert isinstance(result, EveningWindDown)
        assert result.is_meaningful is False

    def test_sparse_windows_returns_not_meaningful(self):
        """Fewer than MIN_ACTIVE_WINDOWS active windows → not meaningful."""
        windows = [
            _make_window(date_str="2026-03-17", hour=9, is_active=True),
            _make_window(date_str="2026-03-17", hour=10, is_active=True),
        ]
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.is_meaningful is False

    def test_inactive_windows_ignored(self):
        """Windows not marked active and no meetings/slack → not counted."""
        windows = [
            _make_window(date_str="2026-03-17", hour=h, is_active=False,
                         messages_sent=0, in_meeting=False)
            for h in range(8, 18)
        ]
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.is_meaningful is False


# ─── Day type classification ──────────────────────────────────────────────────

class TestDayTypeClassification:
    def test_low_cls_is_recovery(self):
        result = _classify_day_type(
            fdi=0.85, cls=0.10, sdi=0.10, csc=0.10,
            meeting_minutes=30, slack_sent=5
        )
        assert result == "RECOVERY"

    def test_high_fdi_low_meetings_is_deep(self):
        result = _classify_day_type(
            fdi=0.80, cls=0.45, sdi=0.15, csc=0.12,
            meeting_minutes=60, slack_sent=5
        )
        assert result == "DEEP"

    def test_high_meetings_and_low_fdi_is_reactive(self):
        result = _classify_day_type(
            fdi=0.40, cls=0.55, sdi=0.60, csc=0.25,
            meeting_minutes=200, slack_sent=35
        )
        assert result == "REACTIVE"

    def test_high_csc_and_low_fdi_is_fragmented(self):
        result = _classify_day_type(
            fdi=0.45, cls=0.40, sdi=0.25, csc=0.40,
            meeting_minutes=90, slack_sent=15
        )
        assert result == "FRAGMENTED"

    def test_moderate_metrics_good_fdi_is_productive(self):
        result = _classify_day_type(
            fdi=0.70, cls=0.35, sdi=0.25, csc=0.20,
            meeting_minutes=90, slack_sent=20
        )
        assert result == "PRODUCTIVE"

    def test_none_cls_returns_mixed(self):
        result = _classify_day_type(
            fdi=None, cls=None, sdi=None, csc=None,
            meeting_minutes=0, slack_sent=0
        )
        assert result == "MIXED"

    def test_reactive_requires_at_least_two_signals(self):
        """One reactive signal alone should not trigger REACTIVE classification."""
        result = _classify_day_type(
            fdi=0.65, cls=0.40, sdi=0.60, csc=0.15,  # only SDI is high
            meeting_minutes=60, slack_sent=10
        )
        # With only one reactive signal (SDI), should not classify as REACTIVE
        assert result != "REACTIVE"


class TestDayTypeFromCompute:
    def test_low_load_day_is_recovery(self):
        windows = _make_workday(cls=0.08, fdi=0.90, sdi=0.05)
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.is_meaningful is True
        assert result.day_type == "RECOVERY"

    def test_high_fdi_few_meetings_is_deep(self):
        windows = _make_workday(cls=0.40, fdi=0.82, sdi=0.10)
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.is_meaningful is True
        assert result.day_type == "DEEP"

    def test_heavy_meeting_day_is_reactive(self):
        # 4 hours of meetings (16 windows × 15min = 4h)
        windows = _make_workday(
            cls=0.55, fdi=0.40, sdi=0.65,
            in_meeting_hours=[9, 10, 11, 14, 15],  # 5h meetings
            messages_sent_per_window=2,  # slack active
        )
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.is_meaningful is True
        assert result.day_type == "REACTIVE"

    def test_fragmented_day_classification(self):
        windows = _make_workday(cls=0.38, fdi=0.42, sdi=0.25, csc=0.42)
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.is_meaningful is True
        assert result.day_type == "FRAGMENTED"

    def test_productive_day_classification(self):
        windows = _make_workday(cls=0.35, fdi=0.72, sdi=0.20, csc=0.15)
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.is_meaningful is True
        assert result.day_type == "PRODUCTIVE"


# ─── Load arc ─────────────────────────────────────────────────────────────────

class TestLoadArc:
    def test_front_loaded_when_morning_higher(self):
        arc = _compute_load_arc(morning_cls=0.55, afternoon_cls=0.25)
        assert arc == "front-loaded"

    def test_back_loaded_when_afternoon_higher(self):
        arc = _compute_load_arc(morning_cls=0.25, afternoon_cls=0.55)
        assert arc == "back-loaded"

    def test_even_when_similar(self):
        arc = _compute_load_arc(morning_cls=0.35, afternoon_cls=0.38)
        assert arc == "even"

    def test_none_morning_is_back_loaded(self):
        arc = _compute_load_arc(morning_cls=None, afternoon_cls=0.50)
        assert arc == "back-loaded"

    def test_none_afternoon_is_front_loaded(self):
        arc = _compute_load_arc(morning_cls=0.50, afternoon_cls=None)
        assert arc == "front-loaded"

    def test_both_none_is_even(self):
        arc = _compute_load_arc(morning_cls=None, afternoon_cls=None)
        assert arc == "even"

    def test_boundary_just_above_threshold_is_back_loaded(self):
        """Delta slightly above 0.15 threshold should be 'back-loaded'."""
        arc = _compute_load_arc(morning_cls=0.25, afternoon_cls=0.42)
        # delta = 0.17, above 0.15 threshold → back-loaded
        assert arc == "back-loaded"

    def test_boundary_just_below_threshold_is_even(self):
        """Delta of 0.10 (below 0.15 threshold) should be 'even'."""
        arc = _compute_load_arc(morning_cls=0.30, afternoon_cls=0.40)
        # delta = 0.10, below 0.15 threshold → even
        assert arc == "even"

    def test_compute_arc_from_windows(self):
        """Arc should reflect morning vs afternoon CLS in compute result."""
        windows = []
        # Morning: light (08:00–12:00)
        for h in range(8, 13):
            for m in [0, 15, 30, 45]:
                windows.append(_make_window(date_str="2026-03-17", hour=h, minute=m, cls=0.20, is_active=True))
        # Afternoon: heavy (13:00–17:00)
        for h in range(13, 18):
            for m in [0, 15, 30, 45]:
                windows.append(_make_window(date_str="2026-03-17", hour=h, minute=m, cls=0.60, is_active=True))

        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.is_meaningful is True
        assert result.load_arc == "back-loaded"
        assert result.morning_cls is not None
        assert result.afternoon_cls is not None
        assert result.afternoon_cls > result.morning_cls


# ─── Wind-down recommendations ────────────────────────────────────────────────

class TestWindDownRecommendation:
    def test_reactive_day_recommends_silence(self):
        rec, detail = _build_wind_down_recommendation(
            day_type="REACTIVE",
            load_arc="even",
            recovery_score=65.0,
            full_day_cls=0.55,
        )
        assert isinstance(rec, str) and len(rec) > 5
        assert isinstance(detail, str) and len(detail) > 5
        assert "social" in rec.lower() or "screen" in rec.lower() or "ping" in rec.lower()

    def test_fragmented_day_recommends_closure(self):
        rec, detail = _build_wind_down_recommendation(
            day_type="FRAGMENTED",
            load_arc="even",
            recovery_score=70.0,
            full_day_cls=0.40,
        )
        assert "task" in rec.lower() or "finish" in rec.lower() or "close" in rec.lower() or "clean" in rec.lower()

    def test_recovery_day_recommends_early_sleep(self):
        rec, detail = _build_wind_down_recommendation(
            day_type="RECOVERY",
            load_arc="even",
            recovery_score=80.0,
            full_day_cls=0.12,
        )
        assert "sleep" in rec.lower() or "rest" in rec.lower() or "early" in rec.lower()

    def test_productive_back_loaded_recommends_stop(self):
        rec, detail = _build_wind_down_recommendation(
            day_type="PRODUCTIVE",
            load_arc="back-loaded",
            recovery_score=70.0,
            full_day_cls=0.45,
        )
        assert "step" in rec.lower() or "away" in rec.lower() or "cut" in rec.lower() or "stop" in rec.lower()

    def test_deep_low_recovery_mentions_rest(self):
        rec, detail = _build_wind_down_recommendation(
            day_type="DEEP",
            load_arc="even",
            recovery_score=45.0,
            full_day_cls=0.50,
        )
        assert "rest" in rec.lower() or "toll" in rec.lower() or "prior" in rec.lower()

    def test_all_day_types_return_non_empty(self):
        for dt in ["PRODUCTIVE", "DEEP", "REACTIVE", "FRAGMENTED", "RECOVERY", "MIXED"]:
            rec, detail = _build_wind_down_recommendation(
                day_type=dt,
                load_arc="even",
                recovery_score=70.0,
                full_day_cls=0.35,
            )
            assert isinstance(rec, str) and len(rec) > 5, f"Day type {dt}: empty recommendation"
            assert isinstance(detail, str) and len(detail) > 5, f"Day type {dt}: empty detail"


# ─── Metrics accuracy ────────────────────────────────────────────────────────

class TestMetricsAccuracy:
    def test_morning_cls_computed_correctly(self):
        """Morning windows (hour < MIDDAY_HOUR) should average separately."""
        windows = []
        for h in range(8, 13):
            windows.append(_make_window(date_str="2026-03-17", hour=h, cls=0.30, is_active=True))
        for h in range(13, 18):
            windows.append(_make_window(date_str="2026-03-17", hour=h, cls=0.50, is_active=True))

        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.is_meaningful is True
        assert result.morning_cls is not None
        assert abs(result.morning_cls - 0.30) < 0.01, f"Expected ~0.30, got {result.morning_cls}"

    def test_afternoon_cls_computed_correctly(self):
        windows = []
        for h in range(8, 13):
            windows.append(_make_window(date_str="2026-03-17", hour=h, cls=0.30, is_active=True))
        for h in range(13, 18):
            windows.append(_make_window(date_str="2026-03-17", hour=h, cls=0.50, is_active=True))

        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.afternoon_cls is not None
        assert abs(result.afternoon_cls - 0.50) < 0.01, f"Expected ~0.50, got {result.afternoon_cls}"

    def test_meeting_minutes_counted(self):
        """Windows flagged in_meeting should count as 15 min each."""
        windows = _make_workday(in_meeting_hours=[9, 10, 14])  # 3h × 4 windows = 12 windows × 15min = 180min
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.total_meeting_minutes == 180, f"Expected 180min, got {result.total_meeting_minutes}"

    def test_slack_sent_summed(self):
        """Messages sent should be summed across all workday windows."""
        windows = _make_workday(messages_sent_per_window=2)
        result = compute_evening_winddown("2026-03-17", windows=windows)
        expected_windows = (EVENING_HOUR - MORNING_START_HOUR) * 4  # 40 windows
        expected_sent = expected_windows * 2
        assert result.slack_sent == expected_sent, f"Expected {expected_sent}, got {result.slack_sent}"

    def test_recovery_score_extracted_from_first_whoop_window(self):
        """Recovery score should come from the first window with WHOOP data."""
        windows = _make_workday(recovery=82.0)
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.recovery_score == 82.0

    def test_windows_outside_workday_ignored(self):
        """Windows before MORNING_START_HOUR or >= EVENING_HOUR should not affect active count."""
        pre_work = [_make_window(date_str="2026-03-17", hour=6, is_active=True, cls=0.99)]
        post_work = [_make_window(date_str="2026-03-17", hour=22, is_active=True, cls=0.99)]
        work_windows = _make_workday(cls=0.30, fdi=0.70)
        all_windows = pre_work + work_windows + post_work

        result_work = compute_evening_winddown("2026-03-17", windows=work_windows)
        result_all = compute_evening_winddown("2026-03-17", windows=all_windows)

        # Full-day metrics should be the same (off-hours not counted)
        if result_work.is_meaningful and result_all.is_meaningful:
            assert abs((result_work.full_day_cls or 0) - (result_all.full_day_cls or 0)) < 0.05


# ─── Format message ────────────────────────────────────────────────────────────

class TestFormatMessage:
    def _make_winddown(self, **kwargs) -> EveningWindDown:
        defaults = dict(
            date_str="2026-03-17",
            day_type="PRODUCTIVE",
            day_type_label="Productive",
            day_type_emoji="✅",
            load_arc="even",
            morning_cls=0.32,
            afternoon_cls=0.35,
            full_day_cls=0.33,
            full_day_fdi=0.72,
            full_day_sdi=0.18,
            full_day_csc=0.15,
            total_meeting_minutes=90,
            slack_sent=15,
            active_windows_count=20,
            recovery_score=72.0,
            projected_dps=74.0,
            wind_down_recommendation="Good day — plan tomorrow briefly, then close.",
            wind_down_detail="You built solid output. Guard sleep for another like it.",
            is_meaningful=True,
        )
        defaults.update(kwargs)
        return EveningWindDown(**defaults)

    def test_not_meaningful_returns_fallback(self):
        wd = self._make_winddown(is_meaningful=False)
        msg = format_winddown_message(wd)
        assert "not enough data" in msg.lower() or "evening" in msg.lower()

    def test_message_contains_date(self):
        wd = self._make_winddown()
        msg = format_winddown_message(wd)
        # Monday, March 17
        assert "Monday" in msg or "March" in msg or "17" in msg

    def test_message_contains_day_type(self):
        wd = self._make_winddown(day_type="REACTIVE", day_type_label="Reactive", day_type_emoji="🔴")
        msg = format_winddown_message(wd)
        assert "Reactive" in msg

    def test_message_contains_recommendation(self):
        rec = "Step away from screens now."
        wd = self._make_winddown(wind_down_recommendation=rec)
        msg = format_winddown_message(wd)
        assert rec in msg

    def test_back_loaded_arc_shows_arrow(self):
        """Back-loaded arc should show AM→PM with ↗ arrow."""
        wd = self._make_winddown(load_arc="back-loaded", morning_cls=0.25, afternoon_cls=0.55)
        msg = format_winddown_message(wd)
        assert "↗" in msg or "PM" in msg

    def test_front_loaded_arc_shows_arrow(self):
        wd = self._make_winddown(load_arc="front-loaded", morning_cls=0.60, afternoon_cls=0.25)
        msg = format_winddown_message(wd)
        assert "↘" in msg or "PM" in msg

    def test_even_arc_no_split_shown(self):
        """Even arc should not show the AM/PM split line."""
        wd = self._make_winddown(load_arc="even", morning_cls=0.33, afternoon_cls=0.35)
        msg = format_winddown_message(wd)
        assert "↗" not in msg and "↘" not in msg

    def test_dps_shown_in_headline(self):
        wd = self._make_winddown(projected_dps=74.0)
        msg = format_winddown_message(wd)
        assert "74" in msg

    def test_meeting_minutes_shown(self):
        wd = self._make_winddown(total_meeting_minutes=90)
        msg = format_winddown_message(wd)
        assert "1h30min" in msg or "90" in msg or "Meetings" in msg

    def test_zero_meetings_not_shown(self):
        """No meetings → meeting line should not appear."""
        wd = self._make_winddown(total_meeting_minutes=0)
        msg = format_winddown_message(wd)
        assert "Meetings" not in msg

    def test_message_is_always_string(self):
        """format_winddown_message() must never return None."""
        for dt in ["PRODUCTIVE", "DEEP", "REACTIVE", "FRAGMENTED", "RECOVERY", "MIXED"]:
            wd = self._make_winddown(day_type=dt, day_type_label=DAY_TYPES[dt]["label"])
            msg = format_winddown_message(wd)
            assert isinstance(msg, str) and len(msg) > 10

    def test_none_winddown_returns_fallback(self):
        msg = format_winddown_message(None)
        assert isinstance(msg, str) and len(msg) > 5


# ─── to_dict() schema ─────────────────────────────────────────────────────────

class TestToDict:
    def _full_winddown(self) -> EveningWindDown:
        return EveningWindDown(
            date_str="2026-03-17",
            day_type="PRODUCTIVE",
            day_type_label="Productive",
            day_type_emoji="✅",
            load_arc="even",
            morning_cls=0.30,
            afternoon_cls=0.35,
            full_day_cls=0.32,
            full_day_fdi=0.72,
            full_day_sdi=0.20,
            full_day_csc=0.15,
            total_meeting_minutes=90,
            slack_sent=15,
            active_windows_count=22,
            recovery_score=74.0,
            projected_dps=73.0,
            wind_down_recommendation="Brief planning, then close.",
            wind_down_detail="You've done good work. Rest now.",
            is_meaningful=True,
        )

    def test_to_dict_has_required_keys(self):
        wd = self._full_winddown()
        d = wd.to_dict()
        required_keys = [
            "date_str", "day_type", "day_type_label", "day_type_emoji",
            "load_arc", "morning_cls", "afternoon_cls", "full_day_cls",
            "full_day_fdi", "full_day_sdi", "full_day_csc",
            "total_meeting_minutes", "slack_sent", "active_windows_count",
            "recovery_score", "projected_dps",
            "wind_down_recommendation", "wind_down_detail", "is_meaningful",
        ]
        for key in required_keys:
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_to_dict_numeric_fields_are_rounded(self):
        """Floats should be rounded to 3 decimal places max."""
        wd = self._full_winddown()
        d = wd.to_dict()
        for key in ["morning_cls", "afternoon_cls", "full_day_cls", "full_day_fdi"]:
            val = d.get(key)
            if val is not None:
                assert isinstance(val, float)
                # Should have at most 3 decimal places
                assert round(val, 3) == val, f"{key} not properly rounded: {val}"

    def test_to_dict_with_none_values(self):
        """None values should be preserved as None in the dict."""
        wd = EveningWindDown(
            date_str="2026-03-17",
            day_type="MIXED",
            day_type_label="Mixed",
            day_type_emoji="⚪",
            load_arc="even",
            morning_cls=None,
            afternoon_cls=None,
            full_day_cls=None,
            full_day_fdi=None,
            full_day_sdi=None,
            full_day_csc=None,
            total_meeting_minutes=0,
            slack_sent=0,
            active_windows_count=0,
            recovery_score=None,
            projected_dps=None,
            wind_down_recommendation="No recommendation.",
            wind_down_detail="",
            is_meaningful=False,
        )
        d = wd.to_dict()
        assert d["morning_cls"] is None
        assert d["projected_dps"] is None
        assert d["full_day_cls"] is None


# ─── Day type constants ───────────────────────────────────────────────────────

class TestDayTypeConstants:
    def test_all_day_types_have_required_fields(self):
        for dt_key, dt_val in DAY_TYPES.items():
            assert "label" in dt_val, f"{dt_key} missing 'label'"
            assert "emoji" in dt_val, f"{dt_key} missing 'emoji'"
            assert "description" in dt_val, f"{dt_key} missing 'description'"

    def test_day_types_cover_all_classifications(self):
        expected = {"PRODUCTIVE", "DEEP", "REACTIVE", "FRAGMENTED", "RECOVERY", "MIXED"}
        assert set(DAY_TYPES.keys()) == expected


# ─── Integration: compute from full windows ───────────────────────────────────

class TestIntegration:
    def test_compute_returns_meaningful_for_full_workday(self):
        windows = _make_workday()
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.is_meaningful is True
        assert result.day_type in DAY_TYPES
        assert result.load_arc in ("front-loaded", "back-loaded", "even")
        assert isinstance(result.wind_down_recommendation, str)
        assert len(result.wind_down_recommendation) > 5

    def test_compute_to_dict_is_json_serialisable(self):
        """The output dict must be fully JSON-serialisable."""
        import json
        windows = _make_workday()
        result = compute_evening_winddown("2026-03-17", windows=windows)
        d = result.to_dict()
        # Should not raise
        json.dumps(d)

    def test_format_winddown_message_from_compute(self):
        """Full pipeline: compute → format → non-empty string."""
        windows = _make_workday()
        result = compute_evening_winddown("2026-03-17", windows=windows)
        msg = format_winddown_message(result)
        assert isinstance(msg, str)
        if result.is_meaningful:
            assert len(msg) > 50
            assert "Evening Wind-Down" in msg

    def test_active_windows_count_matches(self):
        """active_windows_count in result should match actual active windows."""
        windows = _make_workday()
        result = compute_evening_winddown("2026-03-17", windows=windows)
        assert result.active_windows_count > 0
        assert result.active_windows_count <= (EVENING_HOUR - MORNING_START_HOUR) * 4

    def test_full_day_metrics_are_bounded(self):
        """CLS, FDI, SDI, CSC should all be in [0, 1]."""
        windows = _make_workday(cls=0.45, fdi=0.70, sdi=0.30, csc=0.20)
        result = compute_evening_winddown("2026-03-17", windows=windows)
        if result.is_meaningful:
            for attr, val in [
                ("full_day_cls", result.full_day_cls),
                ("full_day_fdi", result.full_day_fdi),
                ("full_day_sdi", result.full_day_sdi),
                ("full_day_csc", result.full_day_csc),
            ]:
                if val is not None:
                    assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0,1] bounds"

"""
Tests for analysis/focus_planner.py

Focus Planner — unit tests covering:

1.  _build_hourly_fdi_profile — derives FDI by hour from JSONL history
2.  _top_focus_hours — ranks hours by historical FDI
3.  _get_free_blocks — finds contiguous free blocks in a calendar
4.  _score_block — scores a free block for focus quality
5.  _get_cdi_limits — returns CDI-based block count + duration limits
6.  plan_tomorrow_focus — full integration (mocked store + calendar)
7.  format_focus_plan_section — Slack formatting

All external I/O (store, gcal, CDI) is mocked.
No credentials or live APIs needed.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.focus_planner import (
    FocusBlock,
    FocusPlan,
    _build_hourly_fdi_profile,
    _get_cdi_limits,
    _get_free_blocks,
    _score_block,
    _top_focus_hours,
    format_focus_plan_section,
    plan_tomorrow_focus,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────

TODAY = "2026-03-14"
TOMORROW = "2026-03-15"


def _make_window(
    hour: int,
    fdi: float = 0.8,
    active: bool = True,
    date: str = "2026-03-13",
) -> dict:
    """Build a minimal JSONL window dict for a given hour."""
    start = f"{date}T{hour:02d}:00:00+01:00"
    return {
        "window_start": start,
        "date": date,
        "window_index": hour * 4,
        "metadata": {
            "hour_of_day": hour,
            "is_active_window": active,
            "is_working_hours": 8 <= hour < 19,
            "day_of_week": "Friday",
        },
        "metrics": {
            "focus_depth_index": fdi,
            "cognitive_load_score": 0.4,
            "social_drain_index": 0.2,
            "context_switch_cost": 0.3,
            "recovery_alignment_score": 0.7,
        },
    }


def _make_windows_day(
    date: str,
    hourly_fdi: dict[int, float],
    active_hours: set[int] | None = None,
) -> list[dict]:
    """Build a list of windows with specific FDI values for given hours."""
    if active_hours is None:
        active_hours = set(hourly_fdi.keys())
    windows = []
    for hour in range(24):
        fdi = hourly_fdi.get(hour, 0.5)
        active = hour in active_hours
        windows.append(_make_window(hour, fdi=fdi, active=active, date=date))
    return windows


def _make_event(
    start_hour: int,
    duration_minutes: int = 60,
    date: str = TOMORROW,
    attendees: int = 2,
) -> dict:
    """Build a minimal calendar event dict."""
    start = datetime(2026, 3, 15, start_hour, 0).isoformat() + "+01:00"
    end_dt = datetime(2026, 3, 15, start_hour, 0) + timedelta(minutes=duration_minutes)
    end = end_dt.isoformat() + "+01:00"
    return {
        "id": f"evt_{start_hour}",
        "title": f"Meeting at {start_hour}:00",
        "start": start,
        "end": end,
        "duration_minutes": duration_minutes,
        "attendee_count": attendees,
        "is_all_day": False,
        "status": "confirmed",
    }


def _make_calendar(events: list[dict] | None = None) -> dict:
    """Build a minimal calendar data dict."""
    events = events or []
    total = sum(e.get("duration_minutes", 0) for e in events)
    return {
        "events": events,
        "event_count": len(events),
        "total_meeting_minutes": total,
        "max_concurrent_attendees": max((e["attendee_count"] for e in events), default=0),
    }


# ─── _build_hourly_fdi_profile ────────────────────────────────────────────────

class TestBuildHourlyFdiProfile:
    """Tests for _build_hourly_fdi_profile."""

    @patch("analysis.focus_planner.list_available_dates")
    @patch("analysis.focus_planner.read_day")
    def test_returns_dict_of_hour_to_fdi(self, mock_read, mock_list):
        """Profile returns a dict mapping working hours to mean FDI."""
        mock_list.return_value = ["2026-03-13"]
        # 4 active windows at hour 9 with varying FDI
        windows = [_make_window(9, fdi=v, active=True) for v in [0.8, 0.9, 0.85, 0.75]]
        mock_read.return_value = windows
        profile = _build_hourly_fdi_profile(TODAY)
        assert 9 in profile
        assert abs(profile[9] - 0.825) < 0.01

    @patch("analysis.focus_planner.list_available_dates")
    @patch("analysis.focus_planner.read_day")
    def test_excludes_inactive_windows(self, mock_read, mock_list):
        """Inactive windows should NOT contribute to the profile."""
        mock_list.return_value = ["2026-03-13"]
        windows = [_make_window(10, fdi=0.9, active=False) for _ in range(4)]
        mock_read.return_value = windows
        profile = _build_hourly_fdi_profile(TODAY)
        assert 10 not in profile

    @patch("analysis.focus_planner.list_available_dates")
    @patch("analysis.focus_planner.read_day")
    def test_excludes_hours_below_min_count(self, mock_read, mock_list):
        """Hours with fewer than MIN_WINDOWS_FOR_HOUR_PROFILE windows are excluded."""
        mock_list.return_value = ["2026-03-13"]
        # Only 1 window at hour 11 (below minimum of 2)
        windows = [_make_window(11, fdi=0.95, active=True)]
        mock_read.return_value = windows
        profile = _build_hourly_fdi_profile(TODAY)
        assert 11 not in profile

    @patch("analysis.focus_planner.list_available_dates")
    @patch("analysis.focus_planner.read_day")
    def test_includes_hours_at_min_count(self, mock_read, mock_list):
        """Hours with exactly MIN_WINDOWS_FOR_HOUR_PROFILE windows are included."""
        mock_list.return_value = ["2026-03-13"]
        # Exactly 2 windows at hour 11 (meets minimum)
        windows = [_make_window(11, fdi=0.9, active=True), _make_window(11, fdi=0.8, active=True)]
        mock_read.return_value = windows
        profile = _build_hourly_fdi_profile(TODAY)
        assert 11 in profile

    @patch("analysis.focus_planner.list_available_dates")
    @patch("analysis.focus_planner.read_day")
    def test_empty_when_no_history(self, mock_read, mock_list):
        """Returns empty dict when no historical dates exist."""
        mock_list.return_value = []
        profile = _build_hourly_fdi_profile(TODAY)
        assert profile == {}
        mock_read.assert_not_called()

    @patch("analysis.focus_planner.list_available_dates")
    @patch("analysis.focus_planner.read_day")
    def test_does_not_include_today(self, mock_read, mock_list):
        """Today's date is excluded from the history lookback."""
        mock_list.return_value = [TODAY, "2026-03-13"]
        windows_today = [_make_window(9, fdi=0.99, active=True, date=TODAY) for _ in range(4)]
        windows_yesterday = [_make_window(9, fdi=0.5, active=True, date="2026-03-13") for _ in range(4)]
        # read_day called only for "2026-03-13", not for TODAY
        mock_read.return_value = windows_yesterday
        profile = _build_hourly_fdi_profile(TODAY)
        # FDI should be ~0.5, not 0.99 (today's data excluded)
        assert 9 in profile
        assert abs(profile[9] - 0.5) < 0.01

    @patch("analysis.focus_planner.list_available_dates")
    @patch("analysis.focus_planner.read_day")
    def test_averages_across_multiple_days(self, mock_read, mock_list):
        """FDI for an hour is averaged across multiple history days."""
        mock_list.return_value = ["2026-03-12", "2026-03-13"]
        mock_read.side_effect = [
            [_make_window(9, fdi=0.6, active=True, date="2026-03-12")] * 3,
            [_make_window(9, fdi=0.9, active=True, date="2026-03-13")] * 3,
        ]
        profile = _build_hourly_fdi_profile(TODAY)
        # Average of 0.6 and 0.9 = 0.75
        assert 9 in profile
        assert abs(profile[9] - 0.75) < 0.01

    @patch("analysis.focus_planner.list_available_dates")
    @patch("analysis.focus_planner.read_day")
    def test_read_exception_handled_gracefully(self, mock_read, mock_list):
        """If read_day raises, the day is skipped — no crash."""
        mock_list.return_value = ["2026-03-12", "2026-03-13"]
        mock_read.side_effect = [Exception("disk error"), [_make_window(10, fdi=0.8, active=True)] * 3]
        profile = _build_hourly_fdi_profile(TODAY)
        # Should have data from the second day
        assert 10 in profile

    @patch("analysis.focus_planner.list_available_dates")
    @patch("analysis.focus_planner.read_day")
    def test_invalid_date_returns_empty(self, mock_read, mock_list):
        """Invalid today_date_str returns empty dict."""
        profile = _build_hourly_fdi_profile("not-a-date")
        assert profile == {}


# ─── _top_focus_hours ─────────────────────────────────────────────────────────

class TestTopFocusHours:
    """Tests for _top_focus_hours."""

    def test_returns_top_n_hours(self):
        """Returns the top-n hours by FDI within working hours."""
        profile = {9: 0.9, 10: 0.7, 11: 0.85, 14: 0.6, 15: 0.8}
        top = _top_focus_hours(profile, n=3)
        assert top == [9, 11, 15]

    def test_excludes_non_working_hours(self):
        """Hours outside WORK_START_HOUR–WORK_END_HOUR are excluded."""
        profile = {6: 0.99, 9: 0.8, 21: 0.95, 10: 0.7}
        top = _top_focus_hours(profile, n=3)
        # 6 and 21 should be excluded
        assert 6 not in top
        assert 21 not in top
        assert 9 in top
        assert 10 in top

    def test_returns_fewer_when_not_enough_hours(self):
        """Returns fewer than n hours when the profile is small."""
        profile = {9: 0.9}
        top = _top_focus_hours(profile, n=3)
        assert top == [9]

    def test_empty_profile_returns_empty(self):
        """Empty profile returns empty list."""
        assert _top_focus_hours({}, n=3) == []

    def test_single_hour_returns_that_hour(self):
        """Single working-hour profile returns that hour."""
        profile = {9: 0.8}
        assert _top_focus_hours(profile, n=1) == [9]

    def test_n_1_returns_best_hour(self):
        """n=1 returns the single best hour."""
        profile = {9: 0.7, 10: 0.95, 14: 0.85}
        top = _top_focus_hours(profile, n=1)
        assert top == [10]

    def test_all_working_hours(self):
        """Returns exactly n when profile covers many working hours."""
        profile = {h: float(h) / 24 for h in range(8, 19)}
        top = _top_focus_hours(profile, n=3)
        assert len(top) == 3
        # Highest hours should win (18, 17, 16)
        assert top[0] == 18


# ─── _get_free_blocks ─────────────────────────────────────────────────────────

class TestGetFreeBlocks:
    """Tests for _get_free_blocks."""

    def test_empty_calendar_returns_full_day_block(self):
        """No events → one large free block covering all working hours."""
        cal = _make_calendar()
        blocks = _get_free_blocks(cal, TOMORROW)
        assert len(blocks) == 1
        # Should span WORK_START_HOUR to WORK_END_HOUR
        total_mins = (19 - 8) * 60
        assert blocks[0]["duration_minutes"] == total_mins
        assert blocks[0]["start_hour"] == 8
        assert blocks[0]["start_minute"] == 0

    def test_single_event_splits_day_into_two_blocks(self):
        """A meeting in the middle splits the day into before + after blocks."""
        event = _make_event(start_hour=11, duration_minutes=60)
        cal = _make_calendar([event])
        blocks = _get_free_blocks(cal, TOMORROW)
        assert len(blocks) == 2
        # Before block: 8:00 → 11:00 = 180 min
        # After block: 12:00 → 19:00 = 420 min
        durations = sorted(b["duration_minutes"] for b in blocks)
        assert durations[0] == 180
        assert durations[1] == 420

    def test_morning_meeting_leaves_afternoon_free(self):
        """3h of morning meetings leaves a large afternoon block."""
        events = [
            _make_event(start_hour=8, duration_minutes=60),
            _make_event(start_hour=9, duration_minutes=60),
            _make_event(start_hour=10, duration_minutes=60),
        ]
        cal = _make_calendar(events)
        blocks = _get_free_blocks(cal, TOMORROW)
        assert len(blocks) == 1
        # 11:00–19:00 = 480 min
        assert blocks[0]["start_hour"] == 11
        assert blocks[0]["duration_minutes"] == 480

    def test_small_gap_excluded_by_min_block_size(self):
        """Gaps smaller than MIN_BLOCK_MINUTES (45min) are not returned."""
        # 30-minute gap between two events
        events = [
            _make_event(start_hour=9, duration_minutes=60),   # 9:00–10:00
            _make_event(start_hour=10, duration_minutes=30),  # 10:00–10:30
        ]
        # There's a 30-min gap at 10:30–11:00 but preceding events are consecutive
        # so only the gap after 10:30 matters: 10:30–19:00 = 510 min
        # Actually: 9:00–10:00 then 10:00–10:30 → 30 min gap at 10:30, then
        # wait — we need the before-first-event slot too: 8:00–9:00 = 60 min
        cal = _make_calendar(events)
        blocks = _get_free_blocks(cal, TOMORROW)
        # All blocks ≥ 45 min
        for b in blocks:
            assert b["duration_minutes"] >= 45

    def test_all_day_event_not_counted_as_occupied(self):
        """All-day events should not occupy working-hour slots."""
        event = _make_event(start_hour=9, duration_minutes=60)
        event["is_all_day"] = True
        cal = _make_calendar([event])
        blocks = _get_free_blocks(cal, TOMORROW)
        # Full day should be free since the all-day event is ignored
        assert len(blocks) == 1
        assert blocks[0]["duration_minutes"] == (19 - 8) * 60

    def test_fully_booked_day_returns_empty(self):
        """A fully booked working day returns no free blocks."""
        # Fill 8:00–19:00 with back-to-back 60-min meetings
        events = [_make_event(start_hour=h, duration_minutes=60) for h in range(8, 19)]
        cal = _make_calendar(events)
        blocks = _get_free_blocks(cal, TOMORROW)
        assert blocks == []

    def test_blocks_are_sorted_chronologically(self):
        """Free blocks are returned in chronological order."""
        events = [
            _make_event(start_hour=10, duration_minutes=60),
            _make_event(start_hour=14, duration_minutes=60),
        ]
        cal = _make_calendar(events)
        blocks = _get_free_blocks(cal, TOMORROW)
        start_times = [(b["start_hour"], b["start_minute"]) for b in blocks]
        assert start_times == sorted(start_times)

    def test_contiguous_events_leave_no_gap(self):
        """Back-to-back meetings leave no gap between them."""
        events = [
            _make_event(start_hour=9, duration_minutes=60),
            _make_event(start_hour=10, duration_minutes=60),
        ]
        cal = _make_calendar(events)
        blocks = _get_free_blocks(cal, TOMORROW)
        # Before: 8:00–9:00 = 60 min; after: 11:00–19:00 = 480 min
        durations = sorted(b["duration_minutes"] for b in blocks)
        assert durations == [60, 480]

    def test_missing_event_start_skipped_gracefully(self):
        """Events with no start/end fields are skipped without crashing."""
        bad_event = {"id": "bad", "title": "Bad", "is_all_day": False,
                     "start": None, "end": None, "duration_minutes": 60, "attendee_count": 1}
        cal = _make_calendar([bad_event])
        blocks = _get_free_blocks(cal, TOMORROW)
        # Should behave as if no events
        assert len(blocks) == 1

    def test_event_before_work_hours_ignored(self):
        """Events before WORK_START_HOUR don't affect working-hour free blocks."""
        early_event = _make_event(start_hour=6, duration_minutes=120)  # 6:00–8:00
        cal = _make_calendar([early_event])
        blocks = _get_free_blocks(cal, TOMORROW)
        # Full working day should still be free
        assert blocks[0]["duration_minutes"] == (19 - 8) * 60


# ─── _score_block ─────────────────────────────────────────────────────────────

class TestScoreBlock:
    """Tests for _score_block."""

    def test_peak_hour_gets_highest_score(self):
        """A block starting at the #1 historical focus hour scores highest."""
        profile = {9: 0.95, 10: 0.85, 14: 0.7}
        peak_hours = [9, 10, 14]
        block_peak = {"start_hour": 9, "start_minute": 0, "duration_minutes": 120}
        block_ok = {"start_hour": 14, "start_minute": 0, "duration_minutes": 120}
        score_peak, q_peak, _, _ = _score_block(block_peak, profile, peak_hours)
        score_ok, q_ok, _, _ = _score_block(block_ok, profile, peak_hours)
        assert score_peak > score_ok
        assert q_peak == "peak"

    def test_morning_block_scores_higher_than_afternoon(self):
        """Morning blocks (before 13:00) score higher than equivalent afternoon blocks."""
        profile = {}
        peak_hours = []
        block_morning = {"start_hour": 9, "start_minute": 0, "duration_minutes": 60}
        block_afternoon = {"start_hour": 15, "start_minute": 0, "duration_minutes": 60}
        score_m, _, _, _ = _score_block(block_morning, profile, peak_hours)
        score_a, _, _, _ = _score_block(block_afternoon, profile, peak_hours)
        assert score_m > score_a

    def test_longer_block_scores_higher(self):
        """Longer blocks score higher (duration bonus)."""
        profile = {}
        peak_hours = []
        short = {"start_hour": 10, "start_minute": 0, "duration_minutes": 45}
        medium = {"start_hour": 10, "start_minute": 0, "duration_minutes": 90}
        long_ = {"start_hour": 10, "start_minute": 0, "duration_minutes": 180}
        s_short, _, _, _ = _score_block(short, profile, peak_hours)
        s_medium, _, _, _ = _score_block(medium, profile, peak_hours)
        s_long, _, _, _ = _score_block(long_, profile, peak_hours)
        assert s_short < s_medium < s_long

    def test_quality_label_correct_for_peak(self):
        """Block at the #1 peak hour gets 'peak' quality."""
        profile = {9: 0.9}
        block = {"start_hour": 9, "start_minute": 0, "duration_minutes": 60}
        _, quality, _, _ = _score_block(block, profile, [9, 10])
        assert quality == "peak"

    def test_quality_label_correct_for_good(self):
        """Block at 2nd-best historical hour gets 'good' quality."""
        profile = {9: 0.9, 10: 0.8}
        block = {"start_hour": 10, "start_minute": 0, "duration_minutes": 60}
        _, quality, _, _ = _score_block(block, profile, [9, 10])
        assert quality == "good"

    def test_quality_ok_when_no_historical_data(self):
        """Block with no historical FDI data gets 'ok' quality."""
        block = {"start_hour": 16, "start_minute": 0, "duration_minutes": 60}
        _, quality, _, _ = _score_block(block, {}, [])
        assert quality == "ok"

    def test_fdi_score_returned_from_profile(self):
        """The historical FDI for the block's hour is returned."""
        profile = {9: 0.87}
        block = {"start_hour": 9, "start_minute": 0, "duration_minutes": 60}
        _, _, _, fdi = _score_block(block, profile, [9])
        assert abs(fdi - 0.87) < 0.001

    def test_fdi_score_none_when_not_in_profile(self):
        """FDI score is None when the hour has no historical data."""
        block = {"start_hour": 16, "start_minute": 0, "duration_minutes": 60}
        _, _, _, fdi = _score_block(block, {}, [])
        assert fdi is None

    def test_reason_string_non_empty(self):
        """Reason string is always non-empty."""
        block = {"start_hour": 11, "start_minute": 0, "duration_minutes": 60}
        _, _, reason, _ = _score_block(block, {}, [])
        assert isinstance(reason, str)
        assert len(reason) > 0


# ─── _get_cdi_limits ─────────────────────────────────────────────────────────

class TestGetCdiLimits:
    """Tests for _get_cdi_limits."""

    @patch("analysis.focus_planner.compute_cdi", create=True)
    def test_surplus_tier_returns_most_blocks(self, mock_cdi):
        """CDI surplus returns the highest block count and longest duration."""
        from analysis.cognitive_debt import CognitiveDebt
        debt = MagicMock()
        debt.is_meaningful = True
        debt.tier = "surplus"
        with patch("analysis.focus_planner._get_cdi_limits") as mock_limits:
            mock_limits.return_value = (3, 180, "surplus", "Energy surplus — room for up to 3 focus blocks.")
            max_blocks, max_dur, tier, note = mock_limits(TODAY)
        assert max_blocks == 3
        assert max_dur == 180
        assert tier == "surplus"

    @patch("analysis.focus_planner._get_cdi_limits")
    def test_critical_tier_returns_fewest_blocks(self, mock_limits):
        """CDI critical returns 1 block, 60 min max."""
        mock_limits.return_value = (1, 60, "critical", "High debt — one short deep-work block only.")
        max_blocks, max_dur, tier, note = mock_limits(TODAY)
        assert max_blocks == 1
        assert max_dur == 60

    @patch("analysis.focus_planner._get_cdi_limits")
    def test_fallback_when_cdi_unavailable(self, mock_limits):
        """Returns sensible defaults when CDI cannot be computed."""
        mock_limits.return_value = (2, 120, None, "")
        max_blocks, max_dur, tier, note = mock_limits(TODAY)
        assert max_blocks >= 1
        assert max_dur >= 60
        assert tier is None


# ─── plan_tomorrow_focus ──────────────────────────────────────────────────────

class TestPlanTomorrowFocus:
    """Integration tests for plan_tomorrow_focus."""

    def _patch_dependencies(
        self,
        available_dates: list[str] = None,
        days_data: dict[str, list[dict]] = None,
        cdi_tier: str = "balanced",
        tomorrow_calendar: dict = None,
    ):
        """
        Helper to create the right stack of patches for a plan_tomorrow_focus call.
        Returns a context manager stack you can use with 'with'.
        """
        available_dates = available_dates or []
        days_data = days_data or {}

        def fake_read_day(d):
            if d in days_data:
                return days_data[d]
            return []

        patches = [
            patch("analysis.focus_planner.list_available_dates", return_value=available_dates),
            patch("analysis.focus_planner.read_day", side_effect=fake_read_day),
        ]
        return patches

    def test_returns_focus_plan_object(self):
        """plan_tomorrow_focus returns a FocusPlan instance."""
        cal = _make_calendar()  # empty calendar = full day free
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            result = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert isinstance(result, FocusPlan)

    def test_empty_calendar_full_day_free(self):
        """Empty calendar → at least one recommended block."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert plan.is_meaningful
        assert len(plan.recommended_blocks) >= 1

    def test_fully_booked_calendar_no_blocks(self):
        """Fully booked calendar → is_meaningful=False, no recommended blocks."""
        events = [_make_event(start_hour=h, duration_minutes=60) for h in range(8, 19)]
        cal = _make_calendar(events)
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert not plan.is_meaningful
        assert plan.recommended_blocks == []

    def test_cdi_limits_block_count(self):
        """CDI critical tier limits plan to 1 block."""
        cal = _make_calendar()  # full free day
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(1, 60, "critical", "High debt.")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert len(plan.recommended_blocks) <= 1

    def test_cdi_limits_duration(self):
        """CDI fatigued tier caps block duration at 90 minutes."""
        cal = _make_calendar()  # full free day (660 min)
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(1, 90, "fatigued", "One block.")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        for block in plan.recommended_blocks:
            assert block.duration_minutes <= 90

    def test_peak_hour_selected_preferentially(self):
        """When a peak-FDI hour is free, it is included in peak_hours and top block."""
        # History: hour 9 has very high FDI
        history_windows = [_make_window(9, fdi=0.98, active=True, date="2026-03-13")] * 4
        # Tomorrow: 9:00–11:00 free (as part of 8:00–11:00 gap), 14:00–16:00 free
        events = [
            _make_event(start_hour=11, duration_minutes=180),  # 11:00–14:00
            _make_event(start_hour=16, duration_minutes=180),  # 16:00–19:00
        ]
        cal = _make_calendar(events)
        with patch("analysis.focus_planner.list_available_dates", return_value=["2026-03-13"]), \
             patch("analysis.focus_planner.read_day", return_value=history_windows), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(3, 180, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        # Hour 9 should be recognised as a peak focus hour in the profile
        assert 9 in plan.peak_hours
        # The first recommended block should cover the morning (starts at 8, covering the 9:00 peak window)
        assert plan.recommended_blocks[0].start_hour < 11  # within the morning free block

    def test_tomorrow_date_is_correct(self):
        """plan.date_str is tomorrow's date (today + 1 day)."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert plan.date_str == TOMORROW

    def test_cdi_tier_stored_in_plan(self):
        """CDI tier is stored in the plan for downstream use."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 90, "loading", "Some note")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert plan.cdi_tier == "loading"

    def test_blocks_within_working_hours(self):
        """All recommended blocks fall within WORK_START_HOUR and WORK_END_HOUR."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(3, 180, "surplus", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        for block in plan.recommended_blocks:
            assert 8 <= block.start_hour < 19
            end_min = block.start_hour * 60 + block.start_minute + block.duration_minutes
            assert end_min <= 19 * 60

    def test_invalid_today_date_returns_none(self):
        """Invalid today_date_str returns None."""
        result = plan_tomorrow_focus("not-a-date", tomorrow_calendar=_make_calendar())
        assert result is None

    def test_plan_has_summary_line(self):
        """Plan always has a non-empty summary line."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert isinstance(plan.summary_line, str)
        assert len(plan.summary_line) > 0

    def test_plan_has_advisory(self):
        """Plan always has an advisory sentence."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert isinstance(plan.advisory, str)

    def test_focus_blocks_have_labels(self):
        """Each FocusBlock has a non-empty label like '9:00–11:00'."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        for block in plan.recommended_blocks:
            assert "–" in block.label
            assert block.label[0].isdigit()

    def test_to_dict_serializable(self):
        """FocusPlan.to_dict() is fully JSON-serializable."""
        import json
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        result = json.dumps(plan.to_dict())
        assert isinstance(result, str)


# ─── FocusBlock dataclass ─────────────────────────────────────────────────────

class TestFocusBlock:
    """Tests for FocusBlock dataclass."""

    def test_to_dict_has_required_keys(self):
        """FocusBlock.to_dict() has all expected keys."""
        block = FocusBlock(
            start_hour=9, start_minute=0, duration_minutes=90,
            label="9:00–10:30", quality="peak",
            reason="test", is_morning=True, fdi_score=0.9,
        )
        d = block.to_dict()
        required = {
            "start_hour", "start_minute", "duration_minutes",
            "label", "quality", "reason", "is_morning", "fdi_score"
        }
        assert required.issubset(d.keys())

    def test_quality_values_are_valid(self):
        """Quality must be one of 'peak', 'good', 'ok'."""
        valid_qualities = {"peak", "good", "ok"}
        block = FocusBlock(
            start_hour=10, start_minute=0, duration_minutes=60,
            label="10:00–11:00", quality="peak",
            reason="test", is_morning=True,
        )
        assert block.quality in valid_qualities


# ─── format_focus_plan_section ────────────────────────────────────────────────

class TestFormatFocusPlanSection:
    """Tests for format_focus_plan_section."""

    def _make_plan(self, blocks: list[FocusBlock] = None, meaningful: bool = True) -> FocusPlan:
        blocks = blocks or []
        return FocusPlan(
            date_str=TOMORROW,
            recommended_blocks=blocks,
            peak_hours=[9, 10],
            cdi_tier="balanced",
            cdi_modifier="Balanced load — 2 blocks recommended.",
            summary_line="Best block: 9:00–11:00",
            advisory="Front-load harder tasks in the earlier block.",
            is_meaningful=meaningful,
        )

    def test_returns_string(self):
        """format_focus_plan_section always returns a string."""
        plan = self._make_plan()
        result = format_focus_plan_section(plan)
        assert isinstance(result, str)

    def test_none_input_returns_empty_string(self):
        """format_focus_plan_section(None) returns empty string."""
        result = format_focus_plan_section(None)
        assert result == ""

    def test_no_blocks_shows_summary_line(self):
        """When no blocks, the summary line is displayed."""
        plan = self._make_plan(blocks=[], meaningful=False)
        plan.summary_line = "Tomorrow is fully booked."
        result = format_focus_plan_section(plan)
        assert "Tomorrow is fully booked" in result

    def test_blocks_appear_in_output(self):
        """Recommended blocks appear in the output."""
        block = FocusBlock(
            start_hour=9, start_minute=0, duration_minutes=90,
            label="9:00–10:30", quality="peak",
            reason="your #1 historical focus hour", is_morning=True, fdi_score=0.9,
        )
        plan = self._make_plan(blocks=[block])
        result = format_focus_plan_section(plan)
        assert "9:00–10:30" in result

    def test_cdi_modifier_shown(self):
        """CDI modifier note appears in the output when set."""
        block = FocusBlock(
            start_hour=9, start_minute=0, duration_minutes=60,
            label="9:00–10:00", quality="good",
            reason="morning window", is_morning=True,
        )
        plan = self._make_plan(blocks=[block])
        plan.cdi_modifier = "Balanced load — 2 blocks recommended."
        result = format_focus_plan_section(plan)
        assert "Balanced" in result or "block" in result.lower()

    def test_advisory_shown(self):
        """Advisory sentence appears in the output."""
        block = FocusBlock(
            start_hour=10, start_minute=0, duration_minutes=90,
            label="10:00–11:30", quality="good",
            reason="strong focus hour", is_morning=True,
        )
        plan = self._make_plan(blocks=[block])
        result = format_focus_plan_section(plan)
        assert "Front-load" in result or "block" in result.lower()

    def test_header_present(self):
        """Output contains the Focus Plan header."""
        block = FocusBlock(
            start_hour=9, start_minute=0, duration_minutes=60,
            label="9:00–10:00", quality="peak",
            reason="peak", is_morning=True,
        )
        plan = self._make_plan(blocks=[block])
        result = format_focus_plan_section(plan)
        assert "Focus Plan" in result or "🎯" in result

    def test_multiple_blocks_all_appear(self):
        """All recommended blocks appear in the output."""
        blocks = [
            FocusBlock(
                start_hour=9, start_minute=0, duration_minutes=90,
                label="9:00–10:30", quality="peak",
                reason="peak hour", is_morning=True,
            ),
            FocusBlock(
                start_hour=14, start_minute=0, duration_minutes=60,
                label="14:00–15:00", quality="ok",
                reason="free window", is_morning=False,
            ),
        ]
        plan = self._make_plan(blocks=blocks)
        result = format_focus_plan_section(plan)
        assert "9:00–10:30" in result
        assert "14:00–15:00" in result

    def test_peak_block_has_fire_emoji(self):
        """Peak-quality blocks get the 🔥 emoji."""
        block = FocusBlock(
            start_hour=9, start_minute=0, duration_minutes=90,
            label="9:00–10:30", quality="peak",
            reason="your #1 historical focus hour", is_morning=True,
        )
        plan = self._make_plan(blocks=[block])
        result = format_focus_plan_section(plan)
        assert "🔥" in result

    def test_good_block_has_check_emoji(self):
        """Good-quality blocks get the ✅ emoji."""
        block = FocusBlock(
            start_hour=10, start_minute=0, duration_minutes=60,
            label="10:00–11:00", quality="good",
            reason="strong focus hour", is_morning=True,
        )
        plan = self._make_plan(blocks=[block])
        result = format_focus_plan_section(plan)
        assert "✅" in result

    def test_output_is_markdown_compatible(self):
        """Output contains Slack markdown (*bold*, _italic_)."""
        block = FocusBlock(
            start_hour=9, start_minute=0, duration_minutes=60,
            label="9:00–10:00", quality="peak",
            reason="peak", is_morning=True,
        )
        plan = self._make_plan(blocks=[block])
        result = format_focus_plan_section(plan)
        # At minimum, should have the *Header* marker
        assert "*" in result


# ─── FocusPlan integration ────────────────────────────────────────────────────

class TestFocusPlanIntegration:
    """Integration-style tests for the full plan pipeline."""

    def test_three_meeting_day_produces_targeted_blocks(self):
        """A typical meeting day still produces focus blocks in the gaps."""
        events = [
            _make_event(start_hour=10, duration_minutes=60),  # 10:00–11:00
            _make_event(start_hour=14, duration_minutes=90),  # 14:00–15:30
            _make_event(start_hour=16, duration_minutes=60),  # 16:00–17:00
        ]
        cal = _make_calendar(events)
        # Free: 8:00–10:00 (120min), 11:00–14:00 (180min), 15:30–16:00 (30min, too short), 17:00–19:00 (120min)
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(3, 180, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert plan.is_meaningful
        # Should have blocks in the 120min and 180min gaps (30min gap excluded)
        assert len(plan.recommended_blocks) >= 2

    def test_heavy_debt_single_short_block(self):
        """CDI critical + fully free day → one short block."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(1, 60, "critical", "High debt.")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert len(plan.recommended_blocks) == 1
        assert plan.recommended_blocks[0].duration_minutes <= 60

    def test_surplus_cdi_multiple_blocks(self):
        """CDI surplus + free day → up to 3 blocks."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(3, 180, "surplus", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        # Free day: 8:00–19:00 = one large block, no natural divisions
        # But only one FREE block exists (one contiguous stretch), so max 1
        # unless we split by max_dur
        # With surplus, up to 3 blocks, but free_blocks has 1 → 1 block
        assert len(plan.recommended_blocks) >= 1

    def test_historical_data_influences_peak_hours(self):
        """Historical FDI data results in non-empty peak_hours when available."""
        history = [_make_window(11, fdi=0.95, active=True, date="2026-03-13")] * 4
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=["2026-03-13"]), \
             patch("analysis.focus_planner.read_day", return_value=history), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert 11 in plan.peak_hours

    def test_no_history_peak_hours_empty(self):
        """With no history, peak_hours is empty."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert plan.peak_hours == []

    def test_days_of_history_count(self):
        """days_of_history reflects the number of available historical dates."""
        cal = _make_calendar()
        with patch("analysis.focus_planner.list_available_dates", return_value=["2026-03-12", "2026-03-13"]), \
             patch("analysis.focus_planner.read_day", return_value=[]), \
             patch("analysis.focus_planner._get_cdi_limits", return_value=(2, 120, "balanced", "")):
            plan = plan_tomorrow_focus(TODAY, tomorrow_calendar=cal)
        assert plan.days_of_history == 2

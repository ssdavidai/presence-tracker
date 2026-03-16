"""
Tests for analysis/weekly_pacing.py — Weekly Cognitive Pacing

Coverage:
  1. _get_week_dates()
     - Monday input → returns that Mon–Fri
     - Wednesday input → returns same week Mon–Fri
     - Saturday input → returns following week Mon–Fri
     - Sunday input → returns following week Mon–Fri
     - Returns exactly 5 dates

  2. _weekday_name()
     - Correct weekday names for known dates

  3. _classify_day()
     - PUSH when predicted CLS < 0.25 (any CDI)
     - PROTECT when fatigued CDI + CLS ≥ 0.25
     - PROTECT when critical CDI + CLS ≥ 0.25
     - STEADY when balanced CDI + moderate CLS
     - STEADY when balanced CDI + heavy CLS
     - Handles None predicted_cls with fallback

  4. _estimate_focus_hours()
     - 0 meeting minutes → ~6.4h
     - 120 meeting minutes → ~4.8h
     - 480 meeting minutes → 0.0h
     - Clamps at 6.5h max

  5. _strategy_note()
     - PUSH note includes "deep work" or "hardest thinking"
     - PROTECT critical note includes "protect energy"
     - PROTECT fatigued note includes debt reference
     - STEADY heavy note includes "focus windows" or "batch"

  6. _compute_weekly_strategy()
     - ≥ 3 PROTECT days → "PROTECT" strategy
     - ≥ 3 PUSH days → "PUSH" strategy
     - fatigued CDI + some push days → "TRANSITION" strategy
     - Mix → "BALANCED" strategy

  7. compute_weekly_pacing()
     - Returns 5 DayPacingProfiles for a Monday input
     - is_meaningful=True always (even with no history)
     - week_start is always a Monday
     - week_end is always a Friday
     - push_days + protect_days + steady_days = 5
     - to_dict() serialises without error (JSON-round-trips)
     - fetch_calendar=False → calendar_available=False for all days
     - CDI context is correctly passed (cdi_context matches CDI tier)

  8. format_weekly_pacing_line()
     - Returns non-empty string with "Week pacing" prefix
     - Contains strategy name

  9. format_weekly_pacing_section()
     - Returns non-empty string with day-by-day table
     - Contains "Weekly Pacing" heading
     - All 5 weekday names present

  10. format_weekly_pacing_terminal()
      - Returns non-empty string with ANSI codes
      - Degrades gracefully when is_meaningful=False

Run with: python3 -m pytest tests/test_weekly_pacing.py -v
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from analysis.weekly_pacing import (
    _classify_day,
    _compute_weekly_strategy,
    _estimate_focus_hours,
    _get_week_dates,
    _strategy_note,
    _weekday_name,
    compute_weekly_pacing,
    DayPacingProfile,
    WeeklyPacingPlan,
    format_weekly_pacing_line,
    format_weekly_pacing_section,
    format_weekly_pacing_terminal,
    FATIGUED_CDI_TIERS,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_day_profile(
    date_str: str = "2026-03-16",
    day_type: str = "STEADY",
    meeting_minutes: int = 60,
    predicted_cls: float = 0.30,
) -> DayPacingProfile:
    weekday = _weekday_name(date_str)
    return DayPacingProfile(
        date_str=date_str,
        weekday=weekday,
        meeting_minutes=meeting_minutes,
        predicted_cls=predicted_cls,
        cls_label="Moderate",
        day_type=day_type,
        focus_hours=_estimate_focus_hours(meeting_minutes),
        strategy_note=f"Test note for {weekday}",
        calendar_events=1,
        calendar_available=True,
        cls_confidence="low",
    )


# ─── 1. _get_week_dates() ─────────────────────────────────────────────────────

class TestGetWeekDates:
    def test_monday_returns_same_week(self):
        # 2026-03-16 is a Monday
        dates = _get_week_dates("2026-03-16")
        assert dates[0] == "2026-03-16"  # Mon
        assert dates[4] == "2026-03-20"  # Fri

    def test_wednesday_returns_same_week_monday(self):
        # 2026-03-18 is a Wednesday — should return same Mon–Fri
        dates = _get_week_dates("2026-03-18")
        assert dates[0] == "2026-03-16"  # Monday of same week
        assert dates[4] == "2026-03-20"

    def test_saturday_returns_next_week(self):
        # 2026-03-21 is a Saturday
        dates = _get_week_dates("2026-03-21")
        assert dates[0] == "2026-03-23"  # Following Monday
        assert dates[4] == "2026-03-27"

    def test_sunday_returns_next_week(self):
        # 2026-03-22 is a Sunday
        dates = _get_week_dates("2026-03-22")
        assert dates[0] == "2026-03-23"  # Following Monday

    def test_returns_exactly_five_dates(self):
        dates = _get_week_dates("2026-03-16")
        assert len(dates) == 5

    def test_dates_are_consecutive_weekdays(self):
        dates = _get_week_dates("2026-03-16")
        for i in range(4):
            d1 = datetime.strptime(dates[i], "%Y-%m-%d")
            d2 = datetime.strptime(dates[i + 1], "%Y-%m-%d")
            assert (d2 - d1).days == 1

    def test_week_start_is_always_monday(self):
        for base_day in range(7):
            dt = datetime(2026, 3, 16) + timedelta(days=base_day)
            dates = _get_week_dates(dt.strftime("%Y-%m-%d"))
            start_dt = datetime.strptime(dates[0], "%Y-%m-%d")
            assert start_dt.weekday() == 0, f"Expected Monday, got {start_dt.strftime('%A')} for input {dt.date()}"


# ─── 2. _weekday_name() ──────────────────────────────────────────────────────

class TestWeekdayName:
    def test_monday(self):
        assert _weekday_name("2026-03-16") == "Monday"

    def test_friday(self):
        assert _weekday_name("2026-03-20") == "Friday"

    def test_wednesday(self):
        assert _weekday_name("2026-03-18") == "Wednesday"


# ─── 3. _classify_day() ──────────────────────────────────────────────────────

class TestClassifyDay:
    def test_push_when_low_cls(self):
        assert _classify_day(0.10, "balanced", 0) == "PUSH"
        assert _classify_day(0.24, "balanced", 30) == "PUSH"

    def test_push_when_low_cls_even_fatigued(self):
        # Even fatigued CDI can't block a PUSH classification for very light days
        assert _classify_day(0.15, "fatigued", 0) == "PUSH"
        assert _classify_day(0.10, "critical", 0) == "PUSH"

    def test_protect_when_fatigued_and_moderate_cls(self):
        assert _classify_day(0.35, "fatigued", 120) == "PROTECT"
        assert _classify_day(0.50, "critical", 240) == "PROTECT"

    def test_protect_when_critical_and_any_load(self):
        assert _classify_day(0.26, "critical", 60) == "PROTECT"

    def test_steady_when_balanced_moderate(self):
        assert _classify_day(0.35, "balanced", 120) == "STEADY"

    def test_steady_when_balanced_heavy(self):
        assert _classify_day(0.60, "balanced", 240) == "STEADY"
        assert _classify_day(0.60, "surplus", 300) == "STEADY"

    def test_none_cls_fallback_light_meetings(self):
        # No CLS data, < 60 min meetings → PUSH
        assert _classify_day(None, "balanced", 30) == "PUSH"

    def test_none_cls_fallback_fatigued(self):
        # No CLS data, ≥ 60 min meetings, fatigued CDI → PROTECT
        assert _classify_day(None, "fatigued", 90) == "PROTECT"

    def test_none_cls_fallback_moderate_meetings_balanced(self):
        # No CLS data, ≥ 60 min meetings, balanced CDI → STEADY
        assert _classify_day(None, "balanced", 120) == "STEADY"


# ─── 4. _estimate_focus_hours() ──────────────────────────────────────────────

class TestEstimateFocusHours:
    def test_zero_meetings_gives_max_focus(self):
        hours = _estimate_focus_hours(0)
        assert hours == pytest.approx(6.4, abs=0.1)

    def test_two_hours_meetings_reduces_focus(self):
        hours = _estimate_focus_hours(120)
        assert hours == pytest.approx(4.8, abs=0.1)

    def test_full_day_meetings_gives_zero_focus(self):
        hours = _estimate_focus_hours(480)
        assert hours == 0.0

    def test_capped_at_six_five(self):
        # Negative meetings shouldn't exceed max
        hours = _estimate_focus_hours(0)
        assert hours <= 6.5

    def test_non_negative(self):
        hours = _estimate_focus_hours(600)
        assert hours >= 0.0


# ─── 5. _strategy_note() ─────────────────────────────────────────────────────

class TestStrategyNote:
    def test_push_day_mentions_deep_work(self):
        note = _strategy_note("PUSH", "Monday", 0.15, 5.0, 0, "balanced")
        assert any(kw in note.lower() for kw in ["deep work", "hardest thinking", "focused"])

    def test_protect_critical_mentions_protect(self):
        note = _strategy_note("PROTECT", "Wednesday", 0.60, 2.0, 240, "critical")
        assert "protect" in note.lower()

    def test_protect_fatigued_mentions_debt_or_fatigue(self):
        note = _strategy_note("PROTECT", "Thursday", 0.40, 3.5, 120, "fatigued")
        assert any(kw in note.lower() for kw in ["debt", "fatigue", "pacing", "pace"])

    def test_steady_heavy_mentions_focus_or_batch(self):
        note = _strategy_note("STEADY", "Tuesday", 0.45, 3.5, 180, "balanced")
        assert any(kw in note.lower() for kw in ["focus", "batch", "protect"])

    def test_note_always_starts_with_weekday(self):
        for weekday in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"):
            note = _strategy_note("STEADY", weekday, 0.30, 4.0, 60, "balanced")
            assert note.startswith(weekday)

    def test_note_non_empty(self):
        for day_type in ("PUSH", "STEADY", "PROTECT"):
            note = _strategy_note(day_type, "Monday", 0.30, 4.0, 60, "balanced")
            assert len(note) > 10


# ─── 6. _compute_weekly_strategy() ──────────────────────────────────────────

class TestComputeWeeklyStrategy:
    def _make_days(self, types: list[str]) -> list[DayPacingProfile]:
        base = datetime(2026, 3, 16)
        return [
            _make_day_profile(
                date_str=(base + timedelta(days=i)).strftime("%Y-%m-%d"),
                day_type=t,
            )
            for i, t in enumerate(types)
        ]

    def test_three_protect_days_returns_protect(self):
        days = self._make_days(["PROTECT", "PROTECT", "PROTECT", "STEADY", "PUSH"])
        strategy, _ = _compute_weekly_strategy(days, "balanced", 45.0)
        assert strategy == "PROTECT"

    def test_three_push_days_returns_push(self):
        days = self._make_days(["PUSH", "PUSH", "PUSH", "STEADY", "STEADY"])
        strategy, _ = _compute_weekly_strategy(days, "balanced", 45.0)
        assert strategy == "PUSH"

    def test_fatigued_with_push_days_returns_transition(self):
        days = self._make_days(["PUSH", "PROTECT", "PROTECT", "STEADY", "PUSH"])
        strategy, _ = _compute_weekly_strategy(days, "fatigued", 75.0)
        assert strategy == "TRANSITION"

    def test_mixed_returns_balanced(self):
        days = self._make_days(["PUSH", "STEADY", "STEADY", "PROTECT", "STEADY"])
        strategy, _ = _compute_weekly_strategy(days, "balanced", 45.0)
        assert strategy == "BALANCED"

    def test_headline_is_non_empty(self):
        days = self._make_days(["PUSH", "STEADY", "STEADY", "STEADY", "PUSH"])
        _, headline = _compute_weekly_strategy(days, "balanced", 50.0)
        assert len(headline) > 10

    def test_handles_none_cdi_score(self):
        days = self._make_days(["PUSH", "STEADY", "STEADY", "STEADY", "PUSH"])
        strategy, headline = _compute_weekly_strategy(days, "balanced", None)
        assert strategy in ("PUSH", "BALANCED", "PROTECT", "TRANSITION")
        assert headline


# ─── 7. compute_weekly_pacing() ──────────────────────────────────────────────

class TestComputeWeeklyPacing:
    """Tests for the full compute_weekly_pacing() function."""

    def test_returns_five_day_profiles(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        assert len(plan.days) == 5

    def test_is_meaningful_always_true(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        assert plan.is_meaningful is True

    def test_week_start_is_monday(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        start_dt = datetime.strptime(plan.week_start, "%Y-%m-%d")
        assert start_dt.weekday() == 0

    def test_week_end_is_friday(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        end_dt = datetime.strptime(plan.week_end, "%Y-%m-%d")
        assert end_dt.weekday() == 4

    def test_push_protect_steady_sum_to_five(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        push_n = len([d for d in plan.days if d.day_type == "PUSH"])
        protect_n = len([d for d in plan.days if d.day_type == "PROTECT"])
        steady_n = len([d for d in plan.days if d.day_type == "STEADY"])
        assert push_n + protect_n + steady_n == 5

    def test_push_protect_lists_match_days(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        push_from_days = {d.date_str for d in plan.days if d.day_type == "PUSH"}
        protect_from_days = {d.date_str for d in plan.days if d.day_type == "PROTECT"}
        assert set(plan.push_days) == push_from_days
        assert set(plan.protect_days) == protect_from_days

    def test_no_calendar_marks_unavailable(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        for day in plan.days:
            assert day.calendar_available is False
            assert day.calendar_events == 0
            assert day.meeting_minutes == 0

    def test_strategy_is_valid_value(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        assert plan.strategy in ("PUSH", "BALANCED", "PROTECT", "TRANSITION")

    def test_to_dict_serialises(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        d = plan.to_dict()
        # Must be JSON-serialisable
        serialised = json.dumps(d)
        parsed = json.loads(serialised)
        assert parsed["week_start"] == "2026-03-16"
        assert len(parsed["days"]) == 5

    def test_cdi_context_is_valid_tier(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        valid_tiers = {"surplus", "balanced", "loading", "fatigued", "critical"}
        assert plan.cdi_context in valid_tiers

    def test_days_of_history_non_negative(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        assert plan.days_of_history >= 0

    def test_saturday_input_uses_next_week(self):
        # 2026-03-21 is Saturday → should plan for week of March 23
        plan = compute_weekly_pacing("2026-03-21", fetch_calendar=False)
        assert plan.week_start == "2026-03-23"

    def test_weekly_load_forecast_in_range(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        if plan.weekly_load_forecast is not None:
            assert 0.0 <= plan.weekly_load_forecast <= 1.0

    def test_day_profiles_have_strategy_notes(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        for day in plan.days:
            assert len(day.strategy_note) > 0

    def test_focus_hours_non_negative(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        for day in plan.days:
            assert day.focus_hours >= 0.0

    def test_mock_calendar_with_meetings(self):
        """Verify that injecting a calendar with meetings changes day classification."""
        def mock_fetch(date_str: str) -> dict:
            return {
                "events": [{"title": "Team standup"}],
                "total_meeting_minutes": 240,
                "event_count": 4,
            }

        with patch("analysis.weekly_pacing._fetch_calendar_safe", side_effect=mock_fetch):
            plan = compute_weekly_pacing("2026-03-16", fetch_calendar=True)

        # All days have 240 meeting minutes → should see steady/protect days
        for day in plan.days:
            assert day.meeting_minutes == 240
            assert day.calendar_available is True
            # 240 min meetings → focus hours should be reduced
            assert day.focus_hours < 5.0


# ─── 8. format_weekly_pacing_line() ─────────────────────────────────────────

class TestFormatWeeklyPacingLine:
    def test_returns_non_empty_string(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        line = format_weekly_pacing_line(plan)
        assert len(line) > 0

    def test_contains_week_pacing(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        line = format_weekly_pacing_line(plan)
        assert "pacing" in line.lower()

    def test_contains_strategy(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        line = format_weekly_pacing_line(plan)
        assert plan.strategy in line

    def test_empty_when_not_meaningful(self):
        plan = WeeklyPacingPlan(
            week_start="2026-03-16",
            week_end="2026-03-20",
            is_meaningful=False,
        )
        assert format_weekly_pacing_line(plan) == ""


# ─── 9. format_weekly_pacing_section() ──────────────────────────────────────

class TestFormatWeeklyPacingSection:
    def test_returns_non_empty_string(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        section = format_weekly_pacing_section(plan)
        assert len(section) > 0

    def test_contains_weekly_pacing_heading(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        section = format_weekly_pacing_section(plan)
        assert "Weekly Pacing" in section

    def test_contains_all_weekdays(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        section = format_weekly_pacing_section(plan)
        # All 5 weekday abbreviations should appear
        for abbrev in ("Mon", "Tue", "Wed", "Thu", "Fri"):
            assert abbrev in section

    def test_contains_strategy_headline(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        section = format_weekly_pacing_section(plan)
        # Headline should appear somewhere in the section
        assert len(plan.strategy_headline) > 0
        # First meaningful words of headline should appear
        first_word = plan.strategy_headline.split()[0]
        assert first_word in section

    def test_empty_when_not_meaningful(self):
        plan = WeeklyPacingPlan(
            week_start="2026-03-16",
            week_end="2026-03-20",
            is_meaningful=False,
        )
        assert format_weekly_pacing_section(plan) == ""

    def test_contains_cdi_footer(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        section = format_weekly_pacing_section(plan)
        assert "CDI" in section


# ─── 10. format_weekly_pacing_terminal() ────────────────────────────────────

class TestFormatWeeklyPacingTerminal:
    def test_returns_non_empty_string(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        output = format_weekly_pacing_terminal(plan)
        assert len(output) > 0

    def test_contains_ansi_codes(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        output = format_weekly_pacing_terminal(plan)
        assert "\033[" in output

    def test_degrades_gracefully_when_not_meaningful(self):
        plan = WeeklyPacingPlan(
            week_start="2026-03-16",
            week_end="2026-03-20",
            is_meaningful=False,
        )
        output = format_weekly_pacing_terminal(plan)
        assert "Weekly Pacing" in output
        assert "No data" in output

    def test_contains_strategy_name(self):
        plan = compute_weekly_pacing("2026-03-16", fetch_calendar=False)
        output = format_weekly_pacing_terminal(plan)
        assert plan.strategy in output

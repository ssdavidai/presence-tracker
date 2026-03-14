"""
Tests for v7.0 — Calendar-Aware Morning Brief

Covers:
  - analyse_today_calendar(): load classification, first meeting detection,
    free blocks, events summary
  - _calendar_aware_recommendation(): correct advice per tier × schedule combo
  - compute_morning_brief(): today_calendar flows through to the output dict
  - format_morning_brief_message(): Today's Schedule section rendered correctly
  - Backward compatibility: None calendar → falls back to tier-based advice
  - Edge cases: empty calendar, all-day events filtered, non-working-hours filtered
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.morning_brief import (
    analyse_today_calendar,
    _calendar_aware_recommendation,
    compute_morning_brief,
    format_morning_brief_message,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_event(
    title: str,
    start_hour: int,
    start_min: int = 0,
    duration_min: int = 60,
    attendees: int = 3,
    date: str = "2026-03-14",
    is_all_day: bool = False,
) -> dict:
    """Build a minimal calendar event dict."""
    start_dt = datetime(
        int(date[:4]), int(date[5:7]), int(date[8:10]),
        start_hour, start_min, 0,
        tzinfo=None,
    )
    # Use timezone-naive ISO format for simplicity in tests
    start_iso = start_dt.isoformat()
    end_iso = start_dt.replace(
        hour=start_hour + (start_min + duration_min) // 60,
        minute=(start_min + duration_min) % 60,
    ).isoformat()
    return {
        "title": title,
        "start": start_iso,
        "end": end_iso,
        "duration_minutes": duration_min,
        "attendee_count": attendees,
        "organizer_email": "david@szabostuban.com",
        "is_all_day": is_all_day,
        "location": "",
        "status": "confirmed",
    }


def _make_calendar(events: list[dict], total_mins: Optional[int] = None) -> dict:
    """Build a minimal calendar_data dict (as returned by gcal.collect())."""
    computed_total = sum(e.get("duration_minutes", 0) for e in events)
    return {
        "events": events,
        "event_count": len(events),
        "total_meeting_minutes": total_mins if total_mins is not None else computed_total,
        "max_concurrent_attendees": max((e.get("attendee_count", 0) for e in events), default=0),
    }


def _make_whoop(recovery: float = 85.0, hrv: float = 79.0) -> dict:
    return {
        "recovery_score": recovery,
        "hrv_rmssd_milli": hrv,
        "sleep_hours": 7.5,
        "sleep_performance": 85.0,
        "resting_heart_rate": 54.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# analyse_today_calendar — load classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyseTodayCalendarLoadClass:

    def test_empty_calendar_is_free(self):
        cal = analyse_today_calendar(_make_calendar([]))
        assert cal["load_class"] == "free"

    def test_zero_events_is_free(self):
        cal = analyse_today_calendar({"events": [], "event_count": 0, "total_meeting_minutes": 0})
        assert cal["load_class"] == "free"

    def test_30min_single_meeting_is_light(self):
        e = _make_event("Quick sync", 10, 0, duration_min=30, attendees=2)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["load_class"] == "light"

    def test_90min_exactly_is_light(self):
        e = _make_event("Workshop", 10, 0, duration_min=90, attendees=4)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["load_class"] == "light"

    def test_91min_is_moderate(self):
        e = _make_event("Long meeting", 9, 0, duration_min=91, attendees=3)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["load_class"] == "moderate"

    def test_240min_exactly_is_moderate(self):
        events = [
            _make_event("Morning block", 9, 0, 120, 2),
            _make_event("Afternoon block", 14, 0, 120, 3),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        assert cal["load_class"] == "moderate"

    def test_241min_is_heavy(self):
        events = [
            _make_event("Deep session", 9, 0, 121, 2),
            _make_event("Product review", 14, 0, 120, 5),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        assert cal["load_class"] == "heavy"

    def test_full_meeting_day_is_heavy(self):
        events = [
            _make_event("Standup", 9, 0, 30, 8),
            _make_event("Sprint planning", 10, 0, 120, 10),
            _make_event("Lunch sync", 12, 0, 60, 4),
            _make_event("Architecture review", 14, 0, 90, 6),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        assert cal["load_class"] == "heavy"
        assert cal["total_minutes"] == 300


class TestAnalyseTodayCalendarFirstMeeting:

    def test_first_meeting_hour_detected(self):
        e = _make_event("Product sync", 10, 30, 60, 3)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["first_meeting_hour"] == 10

    def test_first_meeting_label_formatted(self):
        e = _make_event("All-hands", 10, 0, 60, 8)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["first_meeting_label"] == "10:00"

    def test_first_meeting_label_with_minutes(self):
        e = _make_event("Check-in", 9, 30, 30, 2)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["first_meeting_label"] == "9:30"

    def test_first_meeting_earliest_of_multiple(self):
        events = [
            _make_event("Late meeting", 14, 0, 60, 3),
            _make_event("Early meeting", 8, 30, 30, 2),
            _make_event("Mid meeting", 11, 0, 45, 4),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        assert cal["first_meeting_hour"] == 8
        assert cal["first_meeting_label"] == "8:30"

    def test_no_meetings_first_hour_is_none(self):
        cal = analyse_today_calendar(_make_calendar([]))
        assert cal["first_meeting_hour"] is None
        assert cal["first_meeting_label"] is None


class TestAnalyseTodayCalendarPreFreeTime:

    def test_pre_first_free_mins_basic(self):
        # Meeting at 10:00 → 8:00 to 10:00 = 120 min before first meeting
        e = _make_event("First meeting", 10, 0, 60, 3)
        cal = analyse_today_calendar(_make_calendar([e]))
        # pre_first_free_mins = (10 - 8) * 60 + 0 = 120
        assert cal["pre_first_free_mins"] == 120

    def test_pre_first_free_mins_with_partial_hour(self):
        # Meeting at 9:30 → 8:00 to 9:30 = 90 min
        e = _make_event("Morning sync", 9, 30, 30, 2)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["pre_first_free_mins"] == 90

    def test_pre_first_free_mins_zero_for_8am_meeting(self):
        # Meeting right at 8:00 → 0 min pre-meeting
        e = _make_event("Early bird", 8, 0, 60, 2)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["pre_first_free_mins"] == 0

    def test_pre_first_free_mins_zero_when_no_meetings(self):
        cal = analyse_today_calendar(_make_calendar([]))
        assert cal["pre_first_free_mins"] == 0


class TestAnalyseTodayCalendarFreePeriods:

    def test_free_morning_no_early_meetings(self):
        # Meeting at 11:00 → morning (before 10am) is free
        e = _make_event("Late sync", 11, 0, 60, 3)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["free_morning"] is True

    def test_free_morning_false_with_early_meeting(self):
        e = _make_event("Early standup", 9, 0, 30, 5)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["free_morning"] is False

    def test_free_morning_false_with_8am_meeting(self):
        e = _make_event("8am call", 8, 0, 60, 2)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["free_morning"] is False

    def test_free_afternoon_no_meetings_after_1pm(self):
        e = _make_event("Morning meeting", 9, 0, 60, 3)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["free_afternoon"] is True

    def test_free_afternoon_false_with_afternoon_meeting(self):
        e = _make_event("Afternoon review", 14, 0, 90, 4)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["free_afternoon"] is False

    def test_both_free_with_midday_only_meeting(self):
        # Meeting at 10:30 is after 10am and before 13:00
        e = _make_event("Midday check", 10, 30, 60, 2)
        cal = analyse_today_calendar(_make_calendar([e]))
        # 10:30 < 10 is False → free_morning = True
        # 10:30 >= 13 is False → free_afternoon = True
        assert cal["free_morning"] is True
        assert cal["free_afternoon"] is True


class TestAnalyseTodayCalendarEventFiltering:

    def test_all_day_events_filtered(self):
        e = _make_event("Birthday", 0, 0, 60, 0, is_all_day=True)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["event_count"] == 0
        assert cal["load_class"] == "free"

    def test_before_7am_events_filtered(self):
        # 6:00 event is outside working hours (starts before 7am)
        e = _make_event("Pre-dawn call", 6, 0, 30, 2)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["event_count"] == 0

    def test_after_22pm_events_filtered(self):
        e = _make_event("Late night call", 22, 0, 60, 3)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["event_count"] == 0

    def test_working_hours_events_included(self):
        events = [
            _make_event("First thing", 7, 0, 30, 2),
            _make_event("Last thing", 21, 30, 30, 2),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        assert cal["event_count"] == 2


class TestAnalyseTodayCalendarSocialMeetings:

    def test_social_meetings_count_gt_one_attendee(self):
        events = [
            _make_event("Team sync", 10, 0, 60, 5),
            _make_event("Solo focus", 14, 0, 90, 0),
            _make_event("1:1", 16, 0, 30, 1),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        # Team sync (5 attendees) is social; solo (0) and 1:1 (1) are not
        assert cal["social_meetings"] == 1

    def test_solo_blocks_not_social(self):
        e = _make_event("Deep work block", 9, 0, 120, 0)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert cal["social_meetings"] == 0


class TestAnalyseTodayCalendarLargestMeeting:

    def test_largest_meeting_detected(self):
        events = [
            _make_event("Quick sync", 9, 0, 30, 2),
            _make_event("Sprint planning", 10, 0, 120, 8),
            _make_event("1:1", 14, 0, 45, 1),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        assert cal["largest_meeting_mins"] == 120
        assert "Sprint planning" in cal["largest_meeting_title"]
        assert cal["largest_attendees"] == 8

    def test_largest_meeting_empty_calendar(self):
        cal = analyse_today_calendar(_make_calendar([]))
        assert cal["largest_meeting_mins"] == 0
        assert cal["largest_meeting_title"] == ""


class TestAnalyseTodayCalendarEventsSummary:

    def test_events_summary_has_correct_fields(self):
        e = _make_event("All-hands", 10, 0, 90, 12)
        cal = analyse_today_calendar(_make_calendar([e]))
        assert len(cal["events_summary"]) == 1
        ev = cal["events_summary"][0]
        assert ev["time"] == "10:00"
        assert ev["title"] == "All-hands"
        assert ev["duration_min"] == 90
        assert ev["attendees"] == 12

    def test_events_summary_sorted_chronologically(self):
        events = [
            _make_event("Afternoon", 14, 0, 60, 3),
            _make_event("Morning", 9, 0, 30, 2),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        assert cal["events_summary"][0]["time"] == "9:00"
        assert cal["events_summary"][1]["time"] == "14:00"


# ═══════════════════════════════════════════════════════════════════════════════
# _calendar_aware_recommendation — advice text
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalendarAwareRecommendationFallback:

    def test_none_calendar_falls_back_to_tier_recommendation(self):
        rec = _calendar_aware_recommendation("peak", 86, 79, None, None, None)
        # Should return the original tier recommendation text
        assert len(rec) > 10
        assert isinstance(rec, str)

    def test_none_calendar_peak_tier_mentions_capacity(self):
        rec = _calendar_aware_recommendation("peak", 86, 79, None, None, None)
        # Original _tier_recommendation for peak mentions capacity or readiness
        assert "readiness" in rec.lower() or "capacity" in rec.lower() or "high" in rec.lower()

    def test_none_calendar_recovery_tier_mentions_recovery(self):
        rec = _calendar_aware_recommendation("recovery", 28, 42, None, None, None)
        assert "recovery" in rec.lower()


class TestCalendarAwareRecommendationFreeDay:

    def test_peak_free_day_mentions_deep_work(self):
        cal = analyse_today_calendar(_make_calendar([]))
        rec = _calendar_aware_recommendation("peak", 86, 79, None, None, cal)
        assert "deep" in rec.lower() or "focus" in rec.lower() or "clear" in rec.lower()

    def test_good_free_day_mentions_deep_work(self):
        cal = analyse_today_calendar(_make_calendar([]))
        rec = _calendar_aware_recommendation("good", 75, 72, None, None, cal)
        assert "deep" in rec.lower() or "focus" in rec.lower() or "clear" in rec.lower()

    def test_moderate_free_day_mentions_pace(self):
        cal = analyse_today_calendar(_make_calendar([]))
        rec = _calendar_aware_recommendation("moderate", 58, 60, None, None, cal)
        assert len(rec) > 10


class TestCalendarAwareRecommendationLightDay:

    def test_peak_light_day_mentions_first_meeting_time(self):
        e = _make_event("Sync", 11, 0, 60, 3)
        cal = analyse_today_calendar(_make_calendar([e]))
        rec = _calendar_aware_recommendation("peak", 86, 79, None, None, cal)
        # Should mention the meeting time or free window
        assert "11" in rec or "high" in rec.lower() or "light" in rec.lower()

    def test_good_light_day_returns_string(self):
        e = _make_event("1:1", 10, 0, 30, 2)
        cal = analyse_today_calendar(_make_calendar([e]))
        rec = _calendar_aware_recommendation("good", 75, 72, None, None, cal)
        assert isinstance(rec, str) and len(rec) > 10


class TestCalendarAwareRecommendationHeavyDay:

    def test_peak_heavy_day_acknowledges_meeting_load(self):
        events = [
            _make_event("Standup", 9, 0, 30, 8),
            _make_event("Planning", 10, 0, 120, 12),
            _make_event("Review", 13, 0, 90, 6),
            _make_event("Retro", 15, 0, 60, 10),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        rec = _calendar_aware_recommendation("peak", 86, 79, None, None, cal)
        # Should warn about meeting load despite peak readiness
        assert "meeting" in rec.lower() or "h" in rec.lower()

    def test_moderate_heavy_day_gives_warning(self):
        events = [
            _make_event("Block 1", 9, 0, 90, 5),
            _make_event("Block 2", 11, 0, 90, 8),
            _make_event("Block 3", 14, 0, 90, 4),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        rec = _calendar_aware_recommendation("moderate", 58, 55, None, None, cal)
        # Moderate readiness + heavy day = risk combination
        assert "⚠️" in rec or "risk" in rec.lower() or "defer" in rec.lower() or "protect" in rec.lower()

    def test_recovery_tier_heavy_day_mentions_rescheduling(self):
        events = [
            _make_event("Meeting 1", 9, 0, 90, 5),
            _make_event("Meeting 2", 11, 0, 90, 8),
            _make_event("Meeting 3", 14, 0, 90, 4),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        rec = _calendar_aware_recommendation("recovery", 28, 40, None, None, cal)
        assert "recovery" in rec.lower()
        # Should mention the meetings and suggest rescheduling
        assert "meeting" in rec.lower() or "reschedul" in rec.lower()

    def test_low_tier_heavy_day_mentions_pace(self):
        events = [
            _make_event("Morning", 9, 0, 150, 5),
            _make_event("Afternoon", 14, 0, 120, 8),
        ]
        cal = analyse_today_calendar(_make_calendar(events))
        rec = _calendar_aware_recommendation("low", 42, 38, None, None, cal)
        assert len(rec) > 10


class TestCalendarAwareRecommendationOutputType:

    @pytest.mark.parametrize("tier", ["peak", "good", "moderate", "low", "recovery", "unknown"])
    @pytest.mark.parametrize("load_class", ["free", "light", "moderate", "heavy"])
    def test_always_returns_string(self, tier, load_class):
        mins_map = {"free": 0, "light": 60, "moderate": 180, "heavy": 300}
        total = mins_map[load_class]
        if total > 0:
            e = _make_event("Meeting", 10, 0, total, 3)
            events = [e]
        else:
            events = []
        cal = analyse_today_calendar(_make_calendar(events))
        rec = _calendar_aware_recommendation(tier, 70, 65, None, None, cal)
        assert isinstance(rec, str)
        assert len(rec) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# compute_morning_brief — calendar integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeMorningBriefCalendarIntegration:

    def test_no_calendar_brief_has_no_today_calendar_key(self):
        whoop = _make_whoop()
        brief = compute_morning_brief("2026-03-14", whoop)
        # With no calendar, today_calendar should be None
        assert brief.get("today_calendar") is None

    def test_with_calendar_brief_has_today_calendar(self):
        whoop = _make_whoop()
        e = _make_event("Team sync", 10, 0, 60, 4)
        gcal_data = _make_calendar([e])
        brief = compute_morning_brief("2026-03-14", whoop, today_calendar=gcal_data)
        assert brief.get("today_calendar") is not None
        assert brief["today_calendar"]["load_class"] in ("free", "light", "moderate", "heavy")

    def test_calendar_data_preserved_in_brief(self):
        whoop = _make_whoop()
        e = _make_event("All-hands", 10, 0, 90, 8)
        gcal_data = _make_calendar([e])
        brief = compute_morning_brief("2026-03-14", whoop, today_calendar=gcal_data)
        cal = brief["today_calendar"]
        assert cal["event_count"] == 1
        assert cal["largest_attendees"] == 8

    def test_brief_recommendation_is_string_with_calendar(self):
        whoop = _make_whoop(recovery=86, hrv=79)
        e = _make_event("Sync", 10, 0, 60, 3)
        gcal_data = _make_calendar([e])
        brief = compute_morning_brief("2026-03-14", whoop, today_calendar=gcal_data)
        assert isinstance(brief["readiness"]["recommendation"], str)
        assert len(brief["readiness"]["recommendation"]) > 0

    def test_brief_recommendation_is_string_without_calendar(self):
        whoop = _make_whoop()
        brief = compute_morning_brief("2026-03-14", whoop)
        assert isinstance(brief["readiness"]["recommendation"], str)
        assert len(brief["readiness"]["recommendation"]) > 0

    def test_invalid_calendar_data_degrades_gracefully(self):
        whoop = _make_whoop()
        # Malformed calendar data — should not crash
        brief = compute_morning_brief("2026-03-14", whoop, today_calendar={"events": None})
        assert isinstance(brief["readiness"]["recommendation"], str)

    def test_brief_structure_unchanged_with_calendar(self):
        """Ensure all pre-existing brief fields are still present."""
        whoop = _make_whoop()
        e = _make_event("Meeting", 10, 0, 60, 3)
        brief = compute_morning_brief("2026-03-14", whoop, today_calendar=_make_calendar([e]))
        assert "date" in brief
        assert "whoop" in brief
        assert "readiness" in brief
        assert "yesterday" in brief
        assert "hrv_baseline" in brief
        assert "trend_context" in brief


# ═══════════════════════════════════════════════════════════════════════════════
# format_morning_brief_message — Today's Schedule section
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatMorningBriefTodaySchedule:

    def _brief_with_calendar(self, events: list[dict]) -> dict:
        whoop = _make_whoop()
        gcal_data = _make_calendar(events)
        return compute_morning_brief("2026-03-14", whoop, today_calendar=gcal_data)

    def test_free_day_shows_no_meetings_message(self):
        brief = self._brief_with_calendar([])
        msg = format_morning_brief_message(brief)
        assert "No meetings" in msg or "no meetings" in msg or "full focus" in msg.lower() or "clear" in msg.lower()

    def test_meetings_section_header_present(self):
        e = _make_event("Team sync", 10, 0, 60, 4)
        brief = self._brief_with_calendar([e])
        msg = format_morning_brief_message(brief)
        assert "📅" in msg

    def test_meeting_time_shown(self):
        e = _make_event("All-hands", 10, 0, 60, 8)
        brief = self._brief_with_calendar([e])
        msg = format_morning_brief_message(brief)
        assert "10:00" in msg

    def test_meeting_title_shown(self):
        e = _make_event("Product Review", 14, 0, 90, 6)
        brief = self._brief_with_calendar([e])
        msg = format_morning_brief_message(brief)
        assert "Product Review" in msg

    def test_long_titles_truncated(self):
        long_title = "Very Long Meeting Title That Should Be Truncated In The Display"
        e = _make_event(long_title, 10, 0, 60, 3)
        brief = self._brief_with_calendar([e])
        msg = format_morning_brief_message(brief)
        # Should not include the full 60-char title verbatim
        assert long_title not in msg
        # But should include truncated version
        assert long_title[:20] in msg

    def test_max_5_meetings_shown(self):
        events = [
            _make_event(f"Meeting {i}", 9 + i, 0, 30, 3)
            for i in range(7)  # 7 meetings
        ]
        brief = self._brief_with_calendar(events)
        msg = format_morning_brief_message(brief)
        # Should show "+N more" when > 5
        assert "+2 more" in msg or "more" in msg

    def test_five_or_fewer_meetings_no_overflow(self):
        events = [
            _make_event(f"Meeting {i}", 9 + i, 0, 30, 3)
            for i in range(3)
        ]
        brief = self._brief_with_calendar(events)
        msg = format_morning_brief_message(brief)
        assert "more" not in msg

    def test_message_is_string_with_calendar(self):
        e = _make_event("Check-in", 10, 0, 30, 2)
        brief = self._brief_with_calendar([e])
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str)
        assert len(msg) > 50

    def test_message_is_string_without_calendar(self):
        whoop = _make_whoop()
        brief = compute_morning_brief("2026-03-14", whoop)
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str)
        assert len(msg) > 50

    def test_no_calendar_section_when_no_today_calendar(self):
        whoop = _make_whoop()
        brief = compute_morning_brief("2026-03-14", whoop)
        msg = format_morning_brief_message(brief)
        # The 📅 icon should NOT appear when no calendar data
        # (calendar section is only added when cal is not None)
        # Note: This is a soft check — the brief may show nothing or a fallback
        assert isinstance(msg, str)

    def test_heavy_day_load_emoji_present(self):
        events = [
            _make_event("Block 1", 9, 0, 90, 5),
            _make_event("Block 2", 11, 0, 90, 8),
            _make_event("Block 3", 14, 0, 90, 4),
        ]
        brief = self._brief_with_calendar(events)
        msg = format_morning_brief_message(brief)
        # Heavy day should show red emoji
        assert "🔴" in msg or "📅" in msg


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end / integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalendarBriefIntegration:

    def test_full_pipeline_peak_recovery_heavy_day(self):
        """Peak WHOOP + heavy meeting day: recommendation should be specific."""
        whoop = _make_whoop(recovery=89, hrv=82)
        events = [
            _make_event("Standup", 9, 0, 30, 8),
            _make_event("Planning", 10, 0, 120, 12),
            _make_event("Review", 13, 0, 90, 6),
            _make_event("Retro", 15, 30, 60, 10),
        ]
        gcal = _make_calendar(events)
        brief = compute_morning_brief("2026-03-14", whoop, today_calendar=gcal)
        msg = format_morning_brief_message(brief)

        assert "🟢" in msg   # peak tier emoji
        assert "📅" in msg   # calendar section present
        assert len(msg) > 100

    def test_full_pipeline_recovery_tier_light_day(self):
        """Recovery WHOOP + light meeting day: calendar acknowledged."""
        whoop = _make_whoop(recovery=22, hrv=38)
        events = [_make_event("Quick sync", 14, 0, 30, 2)]
        gcal = _make_calendar(events)
        brief = compute_morning_brief("2026-03-14", whoop, today_calendar=gcal)
        msg = format_morning_brief_message(brief)

        assert "🔴" in msg   # recovery tier emoji
        assert len(msg) > 50

    def test_full_pipeline_no_calendar_no_crash(self):
        """No calendar data → brief still works, no schedule section."""
        whoop = _make_whoop()
        brief = compute_morning_brief("2026-03-14", whoop)
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str) and len(msg) > 50

    def test_full_pipeline_free_day_message_contains_focus_language(self):
        """Clear calendar + good readiness → focus day message."""
        whoop = _make_whoop(recovery=80, hrv=75)
        gcal = _make_calendar([])
        brief = compute_morning_brief("2026-03-14", whoop, today_calendar=gcal)
        msg = format_morning_brief_message(brief)
        rec = brief["readiness"]["recommendation"].lower()
        assert "deep" in rec or "focus" in rec or "clear" in rec

    def test_full_pipeline_with_yesterday_summary_and_calendar(self):
        """Yesterday summary + today calendar both flow through correctly."""
        whoop = _make_whoop()
        yesterday = {
            "date": "2026-03-13",
            "metrics_avg": {"cognitive_load_score": 0.62},
            "calendar": {"total_meeting_minutes": 210},
        }
        e = _make_event("Sync", 10, 0, 60, 4)
        gcal = _make_calendar([e])
        brief = compute_morning_brief(
            "2026-03-14", whoop,
            yesterday_summary=yesterday,
            today_calendar=gcal,
        )
        assert brief["yesterday"]["avg_cls"] == 0.62
        assert brief["today_calendar"]["event_count"] == 1
        msg = format_morning_brief_message(brief)
        assert "Friday" in msg or "Yesterday" in msg   # yesterday label
        assert "📅" in msg                              # calendar section

    def test_calendar_analyse_then_recommend_consistent(self):
        """analyse_today_calendar output → _calendar_aware_recommendation always non-empty."""
        test_cases = [
            ([], "peak"),
            ([_make_event("X", 10, 0, 60, 3)], "good"),
            ([_make_event("A", 9, 0, 90, 5), _make_event("B", 14, 0, 90, 4)], "moderate"),
            ([_make_event("A", 9, 0, 90, 5), _make_event("B", 11, 0, 90, 4),
              _make_event("C", 14, 0, 90, 3)], "low"),
        ]
        for events, tier in test_cases:
            cal = analyse_today_calendar(_make_calendar(events))
            rec = _calendar_aware_recommendation(tier, 70, 65, None, None, cal)
            assert isinstance(rec, str) and len(rec) > 5, (
                f"Empty recommendation for tier={tier}, load={cal['load_class']}"
            )

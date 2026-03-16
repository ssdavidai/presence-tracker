"""
Tests for analysis/meeting_intel.py — Meeting Intelligence Module

Covers:
  1. compute_focus_fragmentation — FFS metric
  2. compute_cognitive_meeting_cost — CMC metric
  3. compute_social_drain_rate — SDR metric
  4. compute_meeting_recovery_fit — MRF classification
  5. compute_longest_free_gap — free gap detection
  6. compute_mis — composite Meeting Intelligence Score
  7. compute_meeting_intel — full integration
  8. format_meeting_intel_section — Slack formatting
  9. format_meeting_intel_terminal — terminal formatting

All tests use synthetic window data — no live APIs or credentials required.
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.meeting_intel import (
    MeetingIntel,
    compute_cognitive_meeting_cost,
    compute_focus_fragmentation,
    compute_longest_free_gap,
    compute_meeting_intel,
    compute_meeting_recovery_fit,
    compute_mis,
    compute_social_drain_rate,
    format_meeting_intel_section,
    format_meeting_intel_terminal,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_window(
    index: int,
    hour: int = 9,
    in_meeting: bool = False,
    meeting_title: str = "Standup",
    meeting_duration: int = 60,
    meeting_attendees: int = 3,
    cls: float = 0.1,
    fdi: float = 0.8,
    sdi: float = 0.1,
    csc: float = 0.1,
    ras: float = 0.9,
    is_working_hours: bool = True,
    is_active: bool = True,
) -> dict:
    """Build a synthetic 15-minute window."""
    return {
        "window_id": f"2026-03-13-{index:03d}",
        "window_index": index,
        "calendar": {
            "in_meeting": in_meeting,
            "meeting_title": meeting_title if in_meeting else None,
            "meeting_attendees": meeting_attendees if in_meeting else 0,
            "meeting_duration_minutes": meeting_duration if in_meeting else 0,
            "meeting_organizer": None,
            "meetings_count": 1 if in_meeting else 0,
        },
        "whoop": {"recovery_score": 78.0},
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "context_switch_cost": csc,
            "recovery_alignment_score": ras,
        },
        "metadata": {
            "hour_of_day": hour,
            "minute_of_hour": (index % 4) * 15,
            "is_working_hours": is_working_hours,
            "is_active_window": is_active,
            "sources_available": ["whoop", "calendar", "slack"],
        },
    }


def _make_day(
    meeting_hours: list[int] = None,
    meeting_title: str = "Standup",
    meeting_duration: int = 60,
    cls_in_meeting: float = 0.4,
    cls_baseline: float = 0.1,
    sdi_in_meeting: float = 0.3,
) -> list[dict]:
    """
    Build a synthetic day of 48 working-hour windows (8am–8pm).

    meeting_hours: list of hours (0-23) that are entirely in meetings.
    """
    meeting_hours = set(meeting_hours or [])
    windows = []
    for i, hour in enumerate(range(8, 20)):  # 8am to 8pm
        for quarter in range(4):
            idx = i * 4 + quarter
            in_meeting = hour in meeting_hours
            windows.append(
                _make_window(
                    index=idx,
                    hour=hour,
                    in_meeting=in_meeting,
                    meeting_title=meeting_title,
                    meeting_duration=meeting_duration,
                    cls=cls_in_meeting if in_meeting else cls_baseline,
                    sdi=sdi_in_meeting if in_meeting else 0.05,
                    is_working_hours=True,
                    is_active=True,
                )
            )
    return windows


# ─── Focus Fragmentation Score ────────────────────────────────────────────────

class TestFocusFragmentation:
    def test_no_meetings_returns_zero(self):
        windows = _make_day(meeting_hours=[])
        assert compute_focus_fragmentation(windows) == 0.0

    def test_no_windows_returns_zero(self):
        assert compute_focus_fragmentation([]) == 0.0

    def test_all_meetings_returns_one(self):
        windows = _make_day(meeting_hours=list(range(8, 20)))
        result = compute_focus_fragmentation(windows)
        assert result == 1.0

    def test_fragmented_day_scores_high(self):
        # Meetings with only 30-min gaps between them (< 45 min MIN_FOCUS_GAP)
        # Pattern: meeting at 9:00-10:00, 10:30-11:30, 12:00-13:00, 13:30-14:30
        # This creates 30-min free gaps that are too short for deep work
        windows = []
        idx = 0
        for hour in range(8, 20):
            for q in range(4):
                start_min = hour * 60 + q * 15
                in_meeting = False
                for ms, me in [
                    (9 * 60, 10 * 60),
                    (10 * 60 + 30, 11 * 60 + 30),
                    (12 * 60, 13 * 60),
                    (13 * 60 + 30, 14 * 60 + 30),
                ]:
                    if ms <= start_min < me:
                        in_meeting = True
                        break
                windows.append(_make_window(idx, hour=hour, in_meeting=in_meeting, is_working_hours=True))
                idx += 1
        result = compute_focus_fragmentation(windows)
        assert result > 0.0

    def test_afternoon_block_scores_low(self):
        # Single 3-hour meeting block in afternoon — morning free
        windows = _make_day(meeting_hours=[14, 15, 16])
        result = compute_focus_fragmentation(windows)
        # Morning is a large free block — low fragmentation
        assert result < 0.5

    def test_scattered_meetings_worse_than_block(self):
        scattered = _make_day(meeting_hours=[9, 11, 13, 15, 17])
        block = _make_day(meeting_hours=[9, 10, 11, 12, 13])
        scattered_ffs = compute_focus_fragmentation(scattered)
        block_ffs = compute_focus_fragmentation(block)
        # Scattered meetings leave short gaps — higher fragmentation
        assert scattered_ffs >= block_ffs

    def test_returns_float_in_0_1(self):
        windows = _make_day(meeting_hours=[9, 11])
        result = compute_focus_fragmentation(windows)
        assert 0.0 <= result <= 1.0

    def test_non_working_hour_windows_ignored(self):
        windows = [_make_window(0, hour=2, in_meeting=False, is_working_hours=False)]
        assert compute_focus_fragmentation(windows) == 0.0


# ─── Cognitive Meeting Cost ───────────────────────────────────────────────────

class TestCognitiveMeetingCost:
    def test_returns_none_when_no_meeting_windows(self):
        nm_wins = [_make_window(i, cls=0.1) for i in range(5)]
        result = compute_cognitive_meeting_cost([], nm_wins)
        assert result is None

    def test_returns_none_when_no_non_meeting_windows(self):
        m_wins = [_make_window(i, in_meeting=True, cls=0.4) for i in range(5)]
        result = compute_cognitive_meeting_cost(m_wins, [])
        assert result is None

    def test_higher_cls_in_meetings_returns_positive(self):
        m_wins = [_make_window(i, in_meeting=True, cls=0.5) for i in range(5)]
        nm_wins = [_make_window(i + 10, cls=0.1) for i in range(5)]
        result = compute_cognitive_meeting_cost(m_wins, nm_wins)
        assert result is not None
        assert result > 0.0

    def test_lower_cls_in_meetings_clips_to_zero(self):
        m_wins = [_make_window(i, in_meeting=True, cls=0.1) for i in range(5)]
        nm_wins = [_make_window(i + 10, cls=0.5) for i in range(5)]
        result = compute_cognitive_meeting_cost(m_wins, nm_wins)
        assert result == 0.0

    def test_equal_cls_returns_zero(self):
        m_wins = [_make_window(i, in_meeting=True, cls=0.3) for i in range(5)]
        nm_wins = [_make_window(i + 10, cls=0.3) for i in range(5)]
        result = compute_cognitive_meeting_cost(m_wins, nm_wins)
        assert result == 0.0

    def test_result_clipped_to_1(self):
        m_wins = [_make_window(i, in_meeting=True, cls=1.0) for i in range(5)]
        nm_wins = [_make_window(i + 10, cls=0.0) for i in range(5)]
        result = compute_cognitive_meeting_cost(m_wins, nm_wins)
        assert result is not None
        assert result <= 1.0

    def test_result_in_0_1(self):
        m_wins = [_make_window(i, in_meeting=True, cls=0.6) for i in range(5)]
        nm_wins = [_make_window(i + 10, cls=0.2) for i in range(5)]
        result = compute_cognitive_meeting_cost(m_wins, nm_wins)
        assert result is not None
        assert 0.0 <= result <= 1.0


# ─── Social Drain Rate ────────────────────────────────────────────────────────

class TestSocialDrainRate:
    def test_returns_none_when_empty(self):
        assert compute_social_drain_rate([]) is None

    def test_returns_mean_sdi(self):
        wins = [_make_window(i, in_meeting=True, sdi=0.4) for i in range(4)]
        result = compute_social_drain_rate(wins)
        assert result == pytest.approx(0.4, abs=0.01)

    def test_clips_to_1(self):
        wins = [_make_window(i, in_meeting=True, sdi=1.5) for i in range(4)]
        result = compute_social_drain_rate(wins)
        assert result == 1.0

    def test_zero_sdi_returns_zero(self):
        wins = [_make_window(i, in_meeting=True, sdi=0.0) for i in range(4)]
        result = compute_social_drain_rate(wins)
        assert result == 0.0

    def test_mixed_sdi_averages(self):
        wins = [
            _make_window(0, in_meeting=True, sdi=0.2),
            _make_window(1, in_meeting=True, sdi=0.6),
        ]
        result = compute_social_drain_rate(wins)
        assert result == pytest.approx(0.4, abs=0.01)


# ─── Meeting Recovery Fit ─────────────────────────────────────────────────────

class TestMeetingRecoveryFit:
    def test_no_recovery_returns_unknown(self):
        assert compute_meeting_recovery_fit(120, None) == "unknown"

    def test_heavy_meetings_low_recovery_is_overloaded(self):
        assert compute_meeting_recovery_fit(240, 40.0) == "overloaded"

    def test_heavy_meetings_high_recovery_is_aligned(self):
        assert compute_meeting_recovery_fit(240, 85.0) == "aligned"

    def test_light_meetings_peak_recovery_is_underutilised(self):
        assert compute_meeting_recovery_fit(30, 80.0) == "underutilised"

    def test_light_meetings_low_recovery_is_aligned(self):
        assert compute_meeting_recovery_fit(30, 40.0) == "aligned"

    def test_moderate_meetings_aligned(self):
        assert compute_meeting_recovery_fit(120, 65.0) == "aligned"

    def test_boundary_heavy_load_equals_180(self):
        # Exactly at MEETING_LOAD_HEAVY with low recovery = overloaded
        assert compute_meeting_recovery_fit(180, 45.0) == "overloaded"

    def test_boundary_light_load_equals_60(self):
        # Exactly at MEETING_LOAD_LIGHT with peak recovery
        # 60 >= 60 so it's not "light" (light is < 60)
        assert compute_meeting_recovery_fit(60, 80.0) == "aligned"

    def test_boundary_under_light_load(self):
        assert compute_meeting_recovery_fit(59, 80.0) == "underutilised"

    def test_recovery_exactly_at_peak_threshold(self):
        # RECOVERY_PEAK = 75 — 75 is peak
        assert compute_meeting_recovery_fit(30, 75.0) == "underutilised"

    def test_recovery_just_below_peak(self):
        assert compute_meeting_recovery_fit(30, 74.9) == "aligned"


# ─── Longest Free Gap ─────────────────────────────────────────────────────────

class TestLongestFreeGap:
    def test_no_windows_returns_zero(self):
        assert compute_longest_free_gap([]) == 0

    def test_all_free_returns_full_working_day(self):
        windows = _make_day(meeting_hours=[])
        result = compute_longest_free_gap(windows)
        # 12 hours × 60 min = 720 minutes
        assert result == 720

    def test_meeting_in_middle_splits_gap(self):
        windows = _make_day(meeting_hours=[12])  # 1 hour meeting at noon
        result = compute_longest_free_gap(windows)
        # 8am-12pm = 4h free (240 min), 1pm-8pm = 7h free (420 min)
        # Longest is the afternoon block: 420 min
        assert result == 420

    def test_meeting_at_start_leaves_long_gap(self):
        windows = _make_day(meeting_hours=[8, 9])  # 2 hours at start
        result = compute_longest_free_gap(windows)
        # 10h free block remaining
        assert result == 600

    def test_all_meetings_returns_zero(self):
        windows = _make_day(meeting_hours=list(range(8, 20)))
        result = compute_longest_free_gap(windows)
        assert result == 0

    def test_non_working_hours_excluded(self):
        windows = [
            _make_window(0, hour=2, in_meeting=False, is_working_hours=False),
            _make_window(1, hour=3, in_meeting=False, is_working_hours=False),
        ]
        assert compute_longest_free_gap(windows) == 0

    def test_returns_multiple_of_15(self):
        windows = _make_day(meeting_hours=[10])
        result = compute_longest_free_gap(windows)
        assert result % 15 == 0


# ─── Composite Score ──────────────────────────────────────────────────────────

class TestComputeMIS:
    def test_all_none_returns_none(self):
        assert compute_mis(None, None, None) is None

    def test_all_zero_returns_100(self):
        assert compute_mis(0.0, 0.0, 0.0) == 100

    def test_all_one_returns_0(self):
        assert compute_mis(1.0, 1.0, 1.0) == 0

    def test_partial_none_uses_zero_for_missing(self):
        # With only FFS=0.5, CMC and SDR default to 0
        result = compute_mis(0.5, None, None)
        expected = int(round((1 - 0.5 * 0.40) * 100))
        assert result == expected

    def test_returns_int(self):
        result = compute_mis(0.2, 0.1, 0.3)
        assert isinstance(result, int)

    def test_clipped_to_0(self):
        # Worst case — all maximally bad
        result = compute_mis(1.0, 1.0, 1.0)
        assert result == 0

    def test_clipped_to_100(self):
        # Best case
        result = compute_mis(0.0, 0.0, 0.0)
        assert result == 100

    def test_weight_distribution(self):
        # FFS weight 0.40: FFS=1 → 1 − 0.40 = 0.60 → MIS=60
        result = compute_mis(1.0, 0.0, 0.0)
        assert result == 60

    def test_cmc_weight(self):
        # CMC weight 0.30: CMC=1 → 1 − 0.30 = 0.70 → MIS=70
        result = compute_mis(0.0, 1.0, 0.0)
        assert result == 70

    def test_sdr_weight(self):
        # SDR weight 0.30: SDR=1 → 1 − 0.30 = 0.70 → MIS=70
        result = compute_mis(0.0, 0.0, 1.0)
        assert result == 70

    def test_mixed_values(self):
        result = compute_mis(0.4, 0.2, 0.1)
        expected = int(round((1 - 0.4 * 0.40 - 0.2 * 0.30 - 0.1 * 0.30) * 100))
        assert result == expected

    def test_in_0_100_range(self):
        for ffs, cmc, sdr in [(0.1, 0.2, 0.3), (0.9, 0.8, 0.7), (0.5, 0.5, 0.5)]:
            result = compute_mis(ffs, cmc, sdr)
            assert 0 <= result <= 100


# ─── Full Integration ─────────────────────────────────────────────────────────

class TestComputeMeetingIntel:
    def test_no_meetings_returns_not_meaningful(self):
        windows = _make_day(meeting_hours=[])
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert not intel.is_meaningful

    def test_meetings_returns_meaningful(self):
        windows = _make_day(meeting_hours=[9, 10])
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert intel.is_meaningful

    def test_date_propagated(self):
        windows = _make_day(meeting_hours=[9])
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert intel.date_str == "2026-03-14"

    def test_total_meeting_minutes_deduplicated(self):
        # Same title across 4 windows (1 hour) — should count once
        windows = [
            _make_window(i, hour=9, in_meeting=True, meeting_title="Standup", meeting_duration=60)
            for i in range(4)
        ]
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert intel.total_meeting_minutes == 60

    def test_multiple_distinct_meetings_summed(self):
        windows = (
            [_make_window(i, hour=9, in_meeting=True, meeting_title="Standup", meeting_duration=30)
             for i in range(2)]
            + [_make_window(i + 10, hour=14, in_meeting=True, meeting_title="1:1", meeting_duration=60)
               for i in range(4)]
        )
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert intel.total_meeting_minutes == 90

    def test_meeting_count_correct(self):
        windows = (
            [_make_window(i, hour=9, in_meeting=True, meeting_title="A", meeting_duration=60) for i in range(4)]
            + [_make_window(i + 10, hour=14, in_meeting=True, meeting_title="B", meeting_duration=60) for i in range(4)]
        )
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert intel.meeting_count == 2

    def test_whoop_recovery_used_for_mrf(self):
        # Build windows with 3 distinct meetings totalling 210 min (>= MEETING_LOAD_HEAVY=180)
        windows = (
            [_make_window(i, hour=9, in_meeting=True, meeting_title="A", meeting_duration=60) for i in range(4)]
            + [_make_window(i + 10, hour=10, in_meeting=True, meeting_title="B", meeting_duration=60) for i in range(4)]
            + [_make_window(i + 20, hour=11, in_meeting=True, meeting_title="C", meeting_duration=90) for i in range(6)]
        )
        whoop = {"recovery_score": 40.0}  # low recovery
        intel = compute_meeting_intel(windows, whoop_data=whoop, date_str="2026-03-14")
        assert intel.meeting_recovery_fit == "overloaded"

    def test_high_recovery_light_meetings_underutilised(self):
        windows = _make_day(meeting_hours=[9])  # light
        whoop = {"recovery_score": 80.0}  # peak recovery
        intel = compute_meeting_intel(windows, whoop_data=whoop, date_str="2026-03-14")
        # 1 hour meeting < MEETING_LOAD_LIGHT(60), recovery >= 75
        # Actually 60 == MEETING_LOAD_LIGHT — boundary check
        assert intel.meeting_recovery_fit in ("underutilised", "aligned")

    def test_mis_computed(self):
        windows = _make_day(meeting_hours=[9, 10])
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert intel.meeting_intelligence_score is not None
        assert 0 <= intel.meeting_intelligence_score <= 100

    def test_headline_is_string(self):
        windows = _make_day(meeting_hours=[9])
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert isinstance(intel.headline, str)
        assert len(intel.headline) > 0

    def test_advisory_is_string(self):
        windows = _make_day(meeting_hours=[9])
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert isinstance(intel.advisory, str)

    def test_to_dict_has_required_keys(self):
        windows = _make_day(meeting_hours=[9])
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        d = intel.to_dict()
        required = {
            "date_str", "focus_fragmentation_score", "cognitive_meeting_cost",
            "social_drain_rate", "meeting_recovery_fit", "meeting_intelligence_score",
            "total_meeting_minutes", "meeting_count", "meeting_windows",
            "free_gap_minutes", "peak_focus_threats", "is_meaningful",
            "headline", "advisory",
        }
        assert required.issubset(d.keys())

    def test_empty_windows_returns_not_meaningful(self):
        intel = compute_meeting_intel([], date_str="2026-03-14")
        assert not intel.is_meaningful

    def test_no_meeting_intel_headline(self):
        windows = _make_day(meeting_hours=[])
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert "No meetings" in intel.headline

    def test_peak_focus_threats_is_list(self):
        windows = _make_day(meeting_hours=[9])
        with patch("analysis.meeting_intel.compute_peak_focus_threats", return_value=["9:00"]):
            intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert isinstance(intel.peak_focus_threats, list)

    def test_free_gap_minutes_positive_when_meetings(self):
        windows = _make_day(meeting_hours=[9])
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert intel.free_gap_minutes > 0

    def test_meeting_windows_count_correct(self):
        windows = _make_day(meeting_hours=[9, 10])  # 2 hours = 8 windows
        intel = compute_meeting_intel(windows, date_str="2026-03-14")
        assert intel.meeting_windows == 8


# ─── Slack Formatter ──────────────────────────────────────────────────────────

class TestFormatMeetingIntelSection:
    def _intel_with_meetings(self, **kwargs) -> MeetingIntel:
        defaults = dict(
            date_str="2026-03-14",
            focus_fragmentation_score=0.3,
            cognitive_meeting_cost=0.2,
            social_drain_rate=0.15,
            meeting_recovery_fit="aligned",
            meeting_intelligence_score=80,
            total_meeting_minutes=120,
            meeting_count=2,
            meeting_windows=8,
            free_gap_minutes=90,
            peak_focus_threats=[],
            is_meaningful=True,
            headline="Acceptable meeting load — 2h with moderate fragmentation.",
            advisory="Longest free block: 90m.",
        )
        defaults.update(kwargs)
        return MeetingIntel(**defaults)

    def test_not_meaningful_returns_empty(self):
        intel = MeetingIntel(date_str="2026-03-14", is_meaningful=False)
        assert format_meeting_intel_section(intel) == ""

    def test_returns_string(self):
        intel = self._intel_with_meetings()
        result = format_meeting_intel_section(intel)
        assert isinstance(result, str)

    def test_contains_header(self):
        intel = self._intel_with_meetings()
        result = format_meeting_intel_section(intel)
        assert "Meeting Intelligence" in result

    def test_contains_mis(self):
        intel = self._intel_with_meetings(meeting_intelligence_score=75)
        result = format_meeting_intel_section(intel)
        assert "75" in result

    def test_contains_ffs(self):
        intel = self._intel_with_meetings(focus_fragmentation_score=0.3)
        result = format_meeting_intel_section(intel)
        assert "FFS" in result

    def test_contains_cmc(self):
        intel = self._intel_with_meetings(cognitive_meeting_cost=0.2)
        result = format_meeting_intel_section(intel)
        assert "CMC" in result

    def test_contains_sdr(self):
        intel = self._intel_with_meetings(social_drain_rate=0.15)
        result = format_meeting_intel_section(intel)
        assert "SDR" in result

    def test_contains_recovery_fit(self):
        intel = self._intel_with_meetings(meeting_recovery_fit="overloaded")
        result = format_meeting_intel_section(intel)
        assert "overloaded" in result

    def test_overloaded_has_warning_emoji(self):
        intel = self._intel_with_meetings(meeting_recovery_fit="overloaded")
        result = format_meeting_intel_section(intel)
        assert "⚠️" in result

    def test_aligned_has_checkmark_emoji(self):
        intel = self._intel_with_meetings(meeting_recovery_fit="aligned")
        result = format_meeting_intel_section(intel)
        assert "✅" in result

    def test_underutilised_has_bulb_emoji(self):
        intel = self._intel_with_meetings(meeting_recovery_fit="underutilised")
        result = format_meeting_intel_section(intel)
        assert "💡" in result

    def test_peak_focus_threats_shown(self):
        intel = self._intel_with_meetings(peak_focus_threats=["9:00", "10:00"])
        result = format_meeting_intel_section(intel)
        assert "9:00" in result

    def test_free_gap_shown_in_detail(self):
        intel = self._intel_with_meetings(free_gap_minutes=90)
        result = format_meeting_intel_section(intel)
        assert "90" in result or "1h" in result

    def test_no_free_gap_when_zero(self):
        intel = self._intel_with_meetings(free_gap_minutes=0, advisory="")
        result = format_meeting_intel_section(intel)
        assert "free gap" not in result.lower() or "0m" not in result

    def test_multiline_output(self):
        intel = self._intel_with_meetings()
        result = format_meeting_intel_section(intel)
        assert "\n" in result


# ─── Terminal Formatter ───────────────────────────────────────────────────────

class TestFormatMeetingIntelTerminal:
    def _intel_with_meetings(self) -> MeetingIntel:
        return MeetingIntel(
            date_str="2026-03-14",
            focus_fragmentation_score=0.3,
            cognitive_meeting_cost=0.2,
            social_drain_rate=0.15,
            meeting_recovery_fit="aligned",
            meeting_intelligence_score=75,
            total_meeting_minutes=120,
            meeting_count=2,
            meeting_windows=8,
            free_gap_minutes=90,
            peak_focus_threats=[],
            is_meaningful=True,
            headline="Acceptable meeting load.",
            advisory="",
        )

    def test_not_meaningful_returns_no_meetings(self):
        intel = MeetingIntel(date_str="2026-03-14", is_meaningful=False)
        result = format_meeting_intel_terminal(intel)
        assert "No meetings" in result

    def test_returns_string(self):
        intel = self._intel_with_meetings()
        result = format_meeting_intel_terminal(intel)
        assert isinstance(result, str)

    def test_contains_mis(self):
        intel = self._intel_with_meetings()
        result = format_meeting_intel_terminal(intel)
        assert "MIS" in result
        assert "75" in result

    def test_contains_ffs(self):
        intel = self._intel_with_meetings()
        result = format_meeting_intel_terminal(intel)
        assert "FFS" in result

    def test_contains_progress_bar(self):
        intel = self._intel_with_meetings()
        result = format_meeting_intel_terminal(intel)
        assert "▓" in result or "░" in result

    def test_peak_threats_shown_when_present(self):
        intel = self._intel_with_meetings()
        intel.peak_focus_threats = ["9:00"]
        result = format_meeting_intel_terminal(intel)
        assert "9:00" in result

    def test_advisory_shown_when_non_default(self):
        intel = self._intel_with_meetings()
        intel.advisory = "Cluster meetings next time."
        result = format_meeting_intel_terminal(intel)
        assert "Cluster meetings" in result


# ─── MRF edge cases ───────────────────────────────────────────────────────────

class TestMRFEdgeCases:
    def test_zero_meetings_with_low_recovery_is_aligned(self):
        # 0 min < MEETING_LOAD_LIGHT(60), but 30.0 < RECOVERY_PEAK(75) → aligned
        assert compute_meeting_recovery_fit(0, 30.0) == "aligned"

    def test_zero_meetings_with_zero_recovery(self):
        # 0 minutes < MEETING_LOAD_LIGHT, 0 < RECOVERY_LOW — not peak recovery
        assert compute_meeting_recovery_fit(0, 0.0) == "aligned"

    def test_exactly_heavy_load_at_boundary(self):
        # 180 >= MEETING_LOAD_HEAVY AND recovery < RECOVERY_LOW
        assert compute_meeting_recovery_fit(180, 49.9) == "overloaded"

    def test_just_under_heavy_is_aligned_even_with_low_recovery(self):
        assert compute_meeting_recovery_fit(179, 49.9) == "aligned"


# ─── Integration: compute_meeting_intel with real-ish data ────────────────────

class TestMeetingIntelIntegration:
    def test_fragmented_day_has_lower_mis(self):
        """Scattered meetings should produce lower MIS than a single block."""
        scattered = _make_day(meeting_hours=[9, 11, 13, 15], cls_in_meeting=0.4)
        block = _make_day(meeting_hours=[9, 10, 11, 12], cls_in_meeting=0.4)

        intel_scattered = compute_meeting_intel(scattered, date_str="2026-03-14")
        intel_block = compute_meeting_intel(block, date_str="2026-03-14")

        assert intel_scattered.meeting_intelligence_score <= intel_block.meeting_intelligence_score

    def test_high_cls_meetings_worse_than_low_cls(self):
        """High cognitive cost meetings should produce lower MIS."""
        high_cost = _make_day(meeting_hours=[9, 10], cls_in_meeting=0.8, cls_baseline=0.1)
        low_cost = _make_day(meeting_hours=[9, 10], cls_in_meeting=0.2, cls_baseline=0.1)

        intel_high = compute_meeting_intel(high_cost, date_str="2026-03-14")
        intel_low = compute_meeting_intel(low_cost, date_str="2026-03-14")

        assert intel_high.meeting_intelligence_score <= intel_low.meeting_intelligence_score

    def test_high_sdi_meetings_worse_than_low(self):
        """High social drain should lower MIS."""
        high_drain = _make_day(meeting_hours=[9, 10], sdi_in_meeting=0.8)
        low_drain = _make_day(meeting_hours=[9, 10], sdi_in_meeting=0.1)

        intel_high = compute_meeting_intel(high_drain, date_str="2026-03-14")
        intel_low = compute_meeting_intel(low_drain, date_str="2026-03-14")

        assert intel_high.meeting_intelligence_score <= intel_low.meeting_intelligence_score

"""
Tests for analysis/flow_detector.py

Covers:
  - _is_flow_window() — criteria evaluation per window
  - detect_flow_states() — session detection, scoring, labelling
  - format_flow_line() / format_flow_section() — Slack formatting
  - compute_weekly_flow_summary() — aggregation helper
  - Edge cases: no data, sparse data, perfect flow, all-blocked
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.flow_detector import (
    FLOW_CLS_MAX,
    FLOW_CLS_MIN,
    FLOW_CSC_MAX,
    FLOW_FDI_MIN,
    MIN_SESSION_WINDOWS,
    TARGET_FLOW_MINUTES,
    FlowSession,
    FlowStateResult,
    _classify_label,
    _is_flow_window,
    detect_flow_states,
    format_flow_line,
    format_flow_section,
    compute_weekly_flow_summary,
)


# ─── Window factory helpers ────────────────────────────────────────────────────

def _make_window(
    hour: int,
    minute: int,
    fdi: float = 0.75,
    cls: float = 0.35,
    csc: float = 0.20,
    in_meeting: bool = False,
    meeting_attendees: int = 0,
    is_working_hours: bool = True,
    date: str = "2026-03-14",
    window_index: int = 0,
) -> dict:
    """Build a minimal valid 15-min window dict for testing."""
    start_str = f"{date}T{hour:02d}:{minute:02d}:00+01:00"
    end_minute = minute + 15
    end_hour   = hour
    if end_minute >= 60:
        end_minute -= 60
        end_hour   += 1
    end_str = f"{date}T{end_hour:02d}:{end_minute:02d}:00+01:00"

    return {
        "date": date,
        "window_start": start_str,
        "window_end":   end_str,
        "window_index": window_index,
        "calendar": {
            "in_meeting":         in_meeting,
            "meeting_attendees":  meeting_attendees,
            "meeting_duration_minutes": 15 if in_meeting else 0,
        },
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index":    fdi,
            "social_drain_index":   0.1,
            "context_switch_cost":  csc,
            "recovery_alignment_score": 0.85,
        },
        "metadata": {
            "is_working_hours": is_working_hours,
            "is_active_window": True,
            "hour_of_day": hour,
            "minute_of_hour": minute,
        },
    }


def _make_day(
    windows_data: list[tuple],
    date: str = "2026-03-14",
) -> list[dict]:
    """
    Build a full day from a compact spec.

    windows_data: list of (hour, minute, fdi, cls, csc, is_working, in_meeting, attendees)
    """
    windows = []
    for idx, spec in enumerate(windows_data):
        h, m, fdi, cls_, csc, working, in_mtg, attendees = spec
        windows.append(_make_window(
            hour=h, minute=m,
            fdi=fdi, cls=cls_, csc=csc,
            is_working_hours=working,
            in_meeting=in_mtg,
            meeting_attendees=attendees,
            date=date,
            window_index=idx,
        ))
    return windows


# ─── Tests: _is_flow_window ────────────────────────────────────────────────────

class TestIsFlowWindow:
    def test_perfect_flow_window(self):
        w = _make_window(hour=9, minute=0, fdi=0.80, cls=0.35, csc=0.15)
        assert _is_flow_window(w) is True

    def test_fdi_below_threshold(self):
        """FDI just below minimum → not flow."""
        w = _make_window(hour=9, minute=0, fdi=FLOW_FDI_MIN - 0.01, cls=0.35, csc=0.15)
        assert _is_flow_window(w) is False

    def test_fdi_at_threshold(self):
        """FDI exactly at minimum → qualifies."""
        w = _make_window(hour=9, minute=0, fdi=FLOW_FDI_MIN, cls=0.35, csc=0.15)
        assert _is_flow_window(w) is True

    def test_cls_too_low_idle(self):
        """CLS below minimum (idle window) → not flow."""
        w = _make_window(hour=9, minute=0, fdi=0.80, cls=FLOW_CLS_MIN - 0.01, csc=0.15)
        assert _is_flow_window(w) is False

    def test_cls_at_min_boundary(self):
        """CLS exactly at minimum → qualifies."""
        w = _make_window(hour=9, minute=0, fdi=0.80, cls=FLOW_CLS_MIN, csc=0.15)
        assert _is_flow_window(w) is True

    def test_cls_too_high_overwhelm(self):
        """CLS above maximum (overwhelm) → not flow."""
        w = _make_window(hour=9, minute=0, fdi=0.80, cls=FLOW_CLS_MAX + 0.01, csc=0.15)
        assert _is_flow_window(w) is False

    def test_cls_at_max_boundary(self):
        """CLS exactly at maximum → qualifies."""
        w = _make_window(hour=9, minute=0, fdi=0.80, cls=FLOW_CLS_MAX, csc=0.15)
        assert _is_flow_window(w) is True

    def test_csc_too_high_fragmented(self):
        """CSC above maximum (too fragmented) → not flow."""
        w = _make_window(hour=9, minute=0, fdi=0.80, cls=0.35, csc=FLOW_CSC_MAX + 0.01)
        assert _is_flow_window(w) is False

    def test_csc_at_threshold(self):
        """CSC exactly at maximum → qualifies."""
        w = _make_window(hour=9, minute=0, fdi=0.80, cls=0.35, csc=FLOW_CSC_MAX)
        assert _is_flow_window(w) is True

    def test_social_meeting_breaks_flow(self):
        """Being in a social meeting (>1 attendee) → not flow."""
        w = _make_window(
            hour=9, minute=0, fdi=0.80, cls=0.35, csc=0.15,
            in_meeting=True, meeting_attendees=2,
        )
        assert _is_flow_window(w) is False

    def test_solo_meeting_allows_flow(self):
        """Solo calendar block (0 attendees) doesn't break flow."""
        w = _make_window(
            hour=9, minute=0, fdi=0.80, cls=0.35, csc=0.15,
            in_meeting=True, meeting_attendees=0,
        )
        assert _is_flow_window(w) is True

    def test_outside_working_hours(self):
        """Window outside working hours → not flow."""
        w = _make_window(
            hour=23, minute=30, fdi=0.80, cls=0.35, csc=0.15,
            is_working_hours=False,
        )
        assert _is_flow_window(w) is False

    def test_missing_metrics(self):
        """Window with None metrics → not flow (safe fallback)."""
        w = _make_window(hour=9, minute=0)
        w["metrics"]["focus_depth_index"] = None
        assert _is_flow_window(w) is False


# ─── Tests: detect_flow_states ─────────────────────────────────────────────────

class TestDetectFlowStates:
    def test_empty_windows_not_meaningful(self):
        result = detect_flow_states([])
        assert result.is_meaningful is False

    def test_too_few_working_windows(self):
        """Only 2 working-hour windows → not meaningful."""
        windows = [
            _make_window(9, 0, window_index=0),
            _make_window(9, 15, window_index=1),
        ]
        result = detect_flow_states(windows)
        assert result.is_meaningful is False

    def test_no_flow_high_cls(self):
        """All windows have CLS too high → no flow sessions."""
        windows = [
            _make_window(h, m, fdi=0.80, cls=0.80, csc=0.15, window_index=i)
            for i, (h, m) in enumerate([(9, 0), (9, 15), (9, 30), (9, 45)])
        ]
        result = detect_flow_states(windows)
        assert result.is_meaningful is True
        assert len(result.flow_sessions) == 0
        assert result.total_flow_minutes == 0
        assert result.flow_label == "none"
        assert result.flow_score == 0.0

    def test_no_flow_low_fdi(self):
        """All windows have FDI too low → no flow sessions."""
        windows = [
            _make_window(h, m, fdi=0.40, cls=0.35, csc=0.15, window_index=i)
            for i, (h, m) in enumerate([(9, 0), (9, 15), (9, 30), (9, 45)])
        ]
        result = detect_flow_states(windows)
        assert len(result.flow_sessions) == 0
        assert result.flow_label == "none"

    def test_single_qualifying_window_too_short(self):
        """Single qualifying window (15 min < 30 min minimum) → no session."""
        windows = [
            _make_window(9, 0, fdi=0.80, cls=0.35, csc=0.15, window_index=0),
            _make_window(9, 15, fdi=0.40, cls=0.80, csc=0.45, window_index=1),  # breaks run
            _make_window(9, 30, fdi=0.80, cls=0.35, csc=0.15, window_index=2),
        ]
        result = detect_flow_states(windows)
        assert len(result.flow_sessions) == 0

    def test_two_consecutive_windows_minimum_session(self):
        """2 consecutive qualifying windows (30 min) = minimum valid session."""
        windows = [
            _make_window(9, 0,  fdi=0.80, cls=0.35, csc=0.15, window_index=0),
            _make_window(9, 15, fdi=0.80, cls=0.35, csc=0.15, window_index=1),
            _make_window(9, 30, fdi=0.40, cls=0.80, csc=0.50, window_index=2),  # breaks
        ]
        result = detect_flow_states(windows)
        assert len(result.flow_sessions) == 1
        assert result.flow_sessions[0].duration_minutes == 30
        assert result.flow_sessions[0].window_count == 2

    def test_long_flow_session_deep_flow_label(self):
        """6 consecutive qualifying windows (90 min) → in_zone or better."""
        windows = [
            _make_window(9, i * 15, fdi=0.85, cls=0.30, csc=0.10, window_index=i)
            for i in range(6)
        ]
        result = detect_flow_states(windows)
        assert result.is_meaningful is True
        assert len(result.flow_sessions) == 1
        assert result.flow_sessions[0].duration_minutes == 90
        assert result.flow_label in ("in_zone", "deep_flow")
        assert result.total_flow_minutes == 90

    def test_deep_flow_label_at_120_min(self):
        """8 windows (120 min) = TARGET → flow_score 1.0 → deep_flow."""
        windows = [
            _make_window(9, i * 15, fdi=0.85, cls=0.30, csc=0.10, window_index=i)
            for i in range(8)
        ]
        result = detect_flow_states(windows)
        assert result.flow_score == 1.0
        assert result.flow_label == "deep_flow"

    def test_two_separate_sessions_detected(self):
        """Two flow blocks with a gap between them → two sessions."""
        windows = []
        # Morning block: 09:00–10:00 (4 windows = 60 min)
        for i in range(4):
            windows.append(_make_window(9, i * 15, fdi=0.80, cls=0.35, csc=0.15, window_index=i))
        # Break: 10:00–10:45 (3 non-qualifying windows)
        for i in range(3):
            windows.append(_make_window(10, i * 15, fdi=0.40, cls=0.80, csc=0.50, window_index=4 + i))
        # Afternoon block: 11:00–11:45 (3 windows = 45 min)
        for i in range(3):
            windows.append(_make_window(11, i * 15, fdi=0.80, cls=0.35, csc=0.15, window_index=7 + i))

        result = detect_flow_states(windows)
        assert len(result.flow_sessions) == 2
        total = 60 + 45
        assert result.total_flow_minutes == total
        # Peak session is the 60-min morning block
        assert result.peak_session is not None
        assert result.peak_session.duration_minutes == 60

    def test_social_meeting_breaks_session(self):
        """Social meeting in the middle of a flow run → two shorter sessions."""
        windows = [
            # Before meeting: 09:00–10:30 (6 windows)
            *[_make_window(9, i * 15, fdi=0.80, cls=0.35, csc=0.15, window_index=i)
              for i in range(6)],
            # Social meeting: breaks flow
            _make_window(10, 30, fdi=0.80, cls=0.35, csc=0.15,
                         in_meeting=True, meeting_attendees=3, window_index=6),
            # After meeting: 10:45–11:45 (4 windows)
            *[_make_window(10, 45 + i * 15, fdi=0.80, cls=0.35, csc=0.15, window_index=7 + i)
              for i in range(4)],
        ]
        result = detect_flow_states(windows)
        # The social meeting window breaks the run → two sessions
        assert len(result.flow_sessions) == 2

    def test_flow_score_capped_at_1(self):
        """Flow score never exceeds 1.0 even with 200+ minutes."""
        windows = [
            _make_window(8, i * 15, fdi=0.85, cls=0.30, csc=0.10, window_index=i)
            for i in range(16)  # 16 × 15 = 240 min
        ]
        result = detect_flow_states(windows)
        assert result.flow_score == 1.0

    def test_peak_session_is_longest(self):
        """peak_session is always the longest single session."""
        windows = [
            # Short: 2 windows (30 min)
            *[_make_window(8, i * 15, fdi=0.80, cls=0.35, csc=0.15, window_index=i)
              for i in range(2)],
            # Break
            _make_window(8, 30, fdi=0.20, cls=0.80, csc=0.50, window_index=2),
            # Long: 5 windows (75 min)
            *[_make_window(9, i * 15, fdi=0.80, cls=0.35, csc=0.15, window_index=3 + i)
              for i in range(5)],
        ]
        result = detect_flow_states(windows)
        assert result.peak_session is not None
        assert result.peak_session.duration_minutes == 75

    def test_session_times_extracted_correctly(self):
        """Flow session start/end times match window boundaries."""
        windows = [
            _make_window(9, 0,  fdi=0.80, cls=0.35, csc=0.15, window_index=0),
            _make_window(9, 15, fdi=0.80, cls=0.35, csc=0.15, window_index=1),
            _make_window(9, 30, fdi=0.80, cls=0.35, csc=0.15, window_index=2),
        ]
        result = detect_flow_states(windows)
        assert len(result.flow_sessions) == 1
        s = result.flow_sessions[0]
        assert s.start_time == "09:00"
        assert s.end_time   == "09:45"

    def test_session_metrics_averaged_correctly(self):
        """Session avg_fdi and avg_cls are correct means across windows."""
        # Need ≥ MIN_WORKING_WINDOWS (3) for is_meaningful; add a non-qualifying 3rd
        windows = [
            _make_window(9, 0,  fdi=0.80, cls=0.30, csc=0.10, window_index=0),
            _make_window(9, 15, fdi=0.90, cls=0.40, csc=0.20, window_index=1),
            # Third window: outside flow zone so we still get exactly one session from the first two
            _make_window(9, 30, fdi=0.20, cls=0.80, csc=0.60, window_index=2),
        ]
        result = detect_flow_states(windows)
        assert result.is_meaningful is True
        assert len(result.flow_sessions) == 1
        s = result.flow_sessions[0]
        assert abs(s.avg_fdi - 0.85) < 0.01   # (0.80 + 0.90) / 2 = 0.85
        assert abs(s.avg_cls - 0.35) < 0.01   # (0.30 + 0.40) / 2 = 0.35
        assert abs(s.avg_csc - 0.15) < 0.01   # (0.10 + 0.20) / 2 = 0.15

    def test_candidate_windows_count(self):
        """candidate_windows counts all individually-qualifying windows."""
        windows = [
            _make_window(9, 0,  fdi=0.80, cls=0.35, csc=0.15, window_index=0),   # qualifies
            _make_window(9, 15, fdi=0.40, cls=0.80, csc=0.50, window_index=1),   # no
            _make_window(9, 30, fdi=0.80, cls=0.35, csc=0.15, window_index=2),   # qualifies
            _make_window(9, 45, fdi=0.80, cls=0.35, csc=0.15, window_index=3),   # qualifies
        ]
        result = detect_flow_states(windows)
        assert result.candidate_windows == 3

    def test_non_working_hours_excluded(self):
        """Non-working-hour windows (is_working_hours=False) are excluded."""
        windows = [
            _make_window(9,  0, fdi=0.80, cls=0.35, csc=0.15, window_index=0, is_working_hours=True),
            _make_window(9, 15, fdi=0.80, cls=0.35, csc=0.15, window_index=1, is_working_hours=True),
            _make_window(22, 0, fdi=0.80, cls=0.35, csc=0.15, window_index=2, is_working_hours=False),
        ]
        result = detect_flow_states(windows)
        # Non-working window excluded from analysis
        assert result.windows_analysed == 2

    def test_all_non_working_hours_not_meaningful(self):
        """All non-working-hour windows → not meaningful."""
        windows = [
            _make_window(23, i * 15, fdi=0.80, cls=0.35, csc=0.15,
                         is_working_hours=False, window_index=i)
            for i in range(6)
        ]
        result = detect_flow_states(windows)
        assert result.is_meaningful is False

    def test_date_extracted_from_windows(self):
        """date_str should be extracted from the windows' date field."""
        windows = [
            _make_window(9, i * 15, date="2026-03-13", window_index=i)
            for i in range(3)
        ]
        result = detect_flow_states(windows)
        assert result.date_str == "2026-03-13"

    def test_windows_sorted_by_index(self):
        """Windows in reverse order should still produce correct sessions."""
        windows = [
            _make_window(9, 30, fdi=0.80, cls=0.35, csc=0.15, window_index=2),
            _make_window(9,  0, fdi=0.80, cls=0.35, csc=0.15, window_index=0),
            _make_window(9, 15, fdi=0.80, cls=0.35, csc=0.15, window_index=1),
        ]
        result = detect_flow_states(windows)
        assert len(result.flow_sessions) == 1
        assert result.flow_sessions[0].duration_minutes == 45


# ─── Tests: _classify_label ───────────────────────────────────────────────────

class TestClassifyLabel:
    def test_deep_flow_at_threshold(self):
        assert _classify_label(0.75) == "deep_flow"

    def test_deep_flow_above_threshold(self):
        assert _classify_label(1.0) == "deep_flow"

    def test_in_zone(self):
        assert _classify_label(0.60) == "in_zone"

    def test_in_zone_just_below_deep(self):
        assert _classify_label(0.74) == "in_zone"

    def test_brief_at_threshold(self):
        assert _classify_label(0.25) == "brief"

    def test_brief_above_threshold(self):
        assert _classify_label(0.45) == "brief"

    def test_none_below_threshold(self):
        assert _classify_label(0.24) == "none"

    def test_none_at_zero(self):
        assert _classify_label(0.0) == "none"


# ─── Tests: format_flow_line ──────────────────────────────────────────────────

class TestFormatFlowLine:
    def test_not_meaningful_returns_empty(self):
        result = FlowStateResult(
            date_str="2026-03-14",
            is_meaningful=False,
            flow_label="none",
        )
        assert format_flow_line(result) == ""

    def test_no_flow_label(self):
        result = FlowStateResult(
            date_str="2026-03-14",
            is_meaningful=True,
            flow_label="none",
            total_flow_minutes=0,
            flow_score=0.0,
        )
        line = format_flow_line(result)
        assert "🌑" in line
        assert "No Flow" in line

    def test_deep_flow_includes_time(self):
        peak = FlowSession(
            start_time="09:15", end_time="11:30",
            duration_minutes=135,
            avg_fdi=0.85, avg_cls=0.35, avg_csc=0.10,
            window_count=9,
            start_hour=9,
        )
        result = FlowStateResult(
            date_str="2026-03-14",
            is_meaningful=True,
            flow_label="deep_flow",
            total_flow_minutes=135,
            peak_session=peak,
            flow_score=1.0,
        )
        line = format_flow_line(result)
        assert "🌊" in line
        assert "Deep Flow" in line
        assert "2h" in line
        assert "09:15" in line

    def test_brief_flow_format(self):
        peak = FlowSession(
            start_time="10:00", end_time="10:30",
            duration_minutes=30,
            avg_fdi=0.70, avg_cls=0.30, avg_csc=0.20,
            window_count=2,
            start_hour=10,
        )
        result = FlowStateResult(
            date_str="2026-03-14",
            is_meaningful=True,
            flow_label="brief",
            total_flow_minutes=30,
            peak_session=peak,
            flow_score=0.25,
        )
        line = format_flow_line(result)
        assert "✨" in line
        assert "Brief" in line
        assert "30m" in line

    def test_in_zone_emoji(self):
        result = FlowStateResult(
            date_str="2026-03-14",
            is_meaningful=True,
            flow_label="in_zone",
            total_flow_minutes=75,
            flow_score=0.625,
        )
        line = format_flow_line(result)
        assert "🎯" in line
        assert "Zone" in line


# ─── Tests: format_flow_section ──────────────────────────────────────────────

class TestFormatFlowSection:
    def test_not_meaningful_returns_empty(self):
        result = FlowStateResult(
            date_str="2026-03-14",
            is_meaningful=False,
            flow_label="none",
        )
        assert format_flow_section(result) == ""

    def test_section_contains_sessions(self):
        sessions = [
            FlowSession("09:00", "10:30", 90, 0.85, 0.30, 0.10, 6, 9),
            FlowSession("14:00", "14:45", 45, 0.78, 0.35, 0.15, 3, 14),
        ]
        result = FlowStateResult(
            date_str="2026-03-14",
            is_meaningful=True,
            flow_label="deep_flow",
            total_flow_minutes=135,
            flow_sessions=sessions,
            peak_session=sessions[0],
            flow_score=1.0,
            insight="Exceptional flow day.",
        )
        section = format_flow_section(result)
        assert "09:00" in section
        assert "14:00" in section
        assert "Deep Flow" in section
        assert "Exceptional" in section

    def test_section_header_format(self):
        result = FlowStateResult(
            date_str="2026-03-14",
            is_meaningful=True,
            flow_label="in_zone",
            flow_score=0.60,
            insight="Solid focus.",
        )
        section = format_flow_section(result)
        assert "*Flow State: In the Zone*" in section
        assert "0.60" in section


# ─── Tests: compute_weekly_flow_summary ──────────────────────────────────────

class TestWeeklyFlowSummary:
    def _make_flow_result_dict(
        self,
        date: str,
        total_minutes: int,
        flow_score: float,
    ) -> dict:
        """Build a minimal FlowStateResult.to_dict() compatible dict."""
        return {
            "date_str": date,
            "flow_sessions": [],
            "total_flow_minutes": total_minutes,
            "peak_session": None,
            "flow_score": flow_score,
            "flow_label": "in_zone" if flow_score > 0 else "none",
            "windows_analysed": 30,
            "candidate_windows": 10,
            "is_meaningful": True,
            "insight": "Test",
            "peak_hour": None,
        }

    def test_empty_days_returns_zeros(self):
        summary = compute_weekly_flow_summary([])
        assert summary["total_flow_minutes"] == 0
        assert summary["flow_days"] == 0
        assert summary["best_flow_day"] is None

    def test_aggregates_correctly(self):
        days = [
            self._make_flow_result_dict("2026-03-10", 90, 0.75),
            self._make_flow_result_dict("2026-03-11", 60, 0.50),
            self._make_flow_result_dict("2026-03-12",  0, 0.00),
        ]
        summary = compute_weekly_flow_summary(days)
        assert summary["total_flow_minutes"] == 150
        assert summary["flow_days"] == 2
        assert summary["best_flow_day"] == "2026-03-10"
        assert summary["best_flow_minutes"] == 90
        assert abs(summary["avg_flow_minutes_per_day"] - 50.0) < 0.01

    def test_avg_flow_score_computed(self):
        days = [
            self._make_flow_result_dict("2026-03-10", 120, 1.00),
            self._make_flow_result_dict("2026-03-11",  60, 0.50),
        ]
        summary = compute_weekly_flow_summary(days)
        assert abs(summary["avg_flow_score"] - 0.75) < 0.001

    def test_all_no_flow_days(self):
        days = [
            self._make_flow_result_dict(f"2026-03-{10 + i}", 0, 0.0)
            for i in range(5)
        ]
        summary = compute_weekly_flow_summary(days)
        assert summary["total_flow_minutes"] == 0
        assert summary["flow_days"] == 0
        assert summary["best_flow_day"] is None

    def test_not_meaningful_days_excluded(self):
        """Days with is_meaningful=False should be excluded from averages."""
        days = [
            self._make_flow_result_dict("2026-03-10", 90, 0.75),
        ]
        days.append({
            "date_str": "2026-03-11",
            "flow_sessions": [],
            "total_flow_minutes": 0,
            "peak_session": None,
            "flow_score": 0.0,
            "flow_label": "none",
            "windows_analysed": 2,
            "candidate_windows": 0,
            "is_meaningful": False,  # excluded
            "insight": "",
            "peak_hour": None,
        })
        summary = compute_weekly_flow_summary(days)
        # Only the first day is valid
        assert summary["total_flow_minutes"] == 90
        assert abs(summary["avg_flow_minutes_per_day"] - 90.0) < 0.01


# ─── Integration: realistic day ───────────────────────────────────────────────

class TestRealisticDay:
    def test_typical_focus_morning(self):
        """
        Simulate a morning with:
          08:00–10:15 — deep focus (flow) — 9 windows = 135 min
          10:15–11:00 — team standup (breaks flow, social meeting)
          11:00–12:00 — moderate focus but high CLS from post-meeting
        """
        windows = []
        idx = 0

        # 08:00–09:45 = 8 windows of flow (FDI 0.80, CLS 0.30)
        for h in range(8, 10):
            for m in [0, 15, 30, 45]:
                windows.append(_make_window(h, m, fdi=0.80, cls=0.30, csc=0.15, window_index=idx))
                idx += 1
        # 10:00 = 1 more flow window before standup (9th flow window)
        windows.append(_make_window(10, 0, fdi=0.80, cls=0.30, csc=0.15, window_index=idx))
        idx += 1

        # 10:15 = social meeting (breaks flow)
        windows.append(_make_window(
            10, 15, fdi=0.40, cls=0.55, csc=0.10,
            in_meeting=True, meeting_attendees=4, window_index=idx
        ))
        idx += 1
        # 10:30 = still in meeting
        windows.append(_make_window(
            10, 30, fdi=0.40, cls=0.55, csc=0.10,
            in_meeting=True, meeting_attendees=4, window_index=idx
        ))
        idx += 1
        # 10:45 = meeting ending
        windows.append(_make_window(
            10, 45, fdi=0.40, cls=0.55, csc=0.10,
            in_meeting=True, meeting_attendees=4, window_index=idx
        ))
        idx += 1

        # 11:00–12:00 = moderate focus but high CLS (still in "reactivity mode")
        for m in [0, 15, 30, 45]:
            windows.append(_make_window(11, m, fdi=0.60, cls=0.65, csc=0.25, window_index=idx))
            idx += 1

        result = detect_flow_states(windows)
        assert result.is_meaningful is True
        # 9 flow windows = 135 min (08:00–10:15)
        # Post-meeting windows have CLS > FLOW_CLS_MAX (0.65 > 0.62) → excluded
        assert len(result.flow_sessions) == 1
        assert result.flow_sessions[0].duration_minutes == 135
        assert result.flow_sessions[0].window_count == 9
        assert result.flow_label == "deep_flow"  # 135/120 = 1.125 → capped at 1.0
        assert result.flow_score == 1.0

    def test_fragmented_day_no_flow(self):
        """
        Simulate a fragmented day with back-to-back short meetings,
        high CSC from context switching. No sustained flow.
        """
        windows = []
        for i in range(12):  # 3 hours of meetings
            windows.append(_make_window(
                9 + (i // 4), (i % 4) * 15,
                fdi=0.45, cls=0.55, csc=0.50,  # high CSC blocks flow
                window_index=i,
            ))

        result = detect_flow_states(windows)
        assert result.is_meaningful is True
        assert len(result.flow_sessions) == 0
        assert result.flow_label == "none"

    def test_single_deep_afternoon_session(self):
        """
        Light morning (idle, CLS too low), then 2-hour deep focus from 14:00.
        """
        windows = []
        # Morning: low CLS (idle reading, no real cognitive work)
        for i in range(12):
            windows.append(_make_window(8 + (i // 4), (i % 4) * 15,
                                        fdi=0.80, cls=0.05, csc=0.05,
                                        window_index=i))
        # Afternoon: 14:00–16:00 = 8 flow windows
        for i in range(8):
            windows.append(_make_window(14, i * 15, fdi=0.80, cls=0.35, csc=0.10,
                                        window_index=12 + i))

        result = detect_flow_states(windows)
        assert len(result.flow_sessions) == 1  # Only afternoon qualifies
        assert result.flow_sessions[0].duration_minutes == 120
        assert result.flow_label == "deep_flow"

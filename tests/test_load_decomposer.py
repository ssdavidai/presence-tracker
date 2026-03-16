"""
Tests for analysis/load_decomposer.py — Cognitive Load Decomposer

Coverage:
  1. _clamp / _norm helpers
  2. _physiological_readiness mirroring metrics.py
  3. _decompose_window
     - Empty window (all zeros) → physiology only
     - Social meeting → meetings component > 0
     - Solo meeting (attendees ≤ 1) → meetings component = 0
     - Slack messages → slack component > 0
     - RescueTime low productivity → rescuetime component > 0
     - Omi conversation → omi component > 0
     - Components sum approximately to cls_computed
  4. compute_load_decomposition
     - No data → is_meaningful=False
     - Insufficient active windows → is_meaningful=False
     - Valid day → is_meaningful=True, source_shares sum ≈ 1.0
     - source_shares keys cover all 5 sources
     - dominant_source matches max share
     - insight_lines is a list
     - to_dict() round-trips through JSON
  5. compute_week_decomposition
     - No data → days_meaningful=0
     - Weekly shares are averages
  6. _build_insights
     - Low load → early exit
     - Meeting dominant → mentions meetings in insight
     - Slack dominant → mentions Slack in insight
     - Physiology dominant → mentions recovery in insight
  7. Formatting
     - format_decomposition_line returns empty when not meaningful
     - format_decomposition_line contains 'Load breakdown' when meaningful
     - format_decomposition_section returns empty when not meaningful
     - format_decomposition_section contains all 5 source emojis
     - format_decomposition_terminal contains date
     - format_week_decomposition_section returns empty when no data
"""

import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from analysis.load_decomposer import (
    LoadDecomposition,
    _build_insights,
    _clamp,
    _decompose_window,
    _norm,
    _physiological_readiness,
    compute_load_decomposition,
    compute_week_decomposition,
    format_decomposition_line,
    format_decomposition_section,
    format_decomposition_terminal,
    format_week_decomposition_section,
)


# ─── Helper factories ─────────────────────────────────────────────────────────

def _make_window(
    in_meeting=False,
    meeting_attendees=0,
    slack_received=0,
    recovery_score=85.0,
    hrv=75.0,
    sleep_perf=85.0,
    rt_active=0,
    rt_prod=None,
    omi_active=False,
    omi_words=0,
    cls_stored=None,
) -> dict:
    return {
        "calendar": {
            "in_meeting": in_meeting,
            "meeting_attendees": meeting_attendees,
        },
        "whoop": {
            "recovery_score": recovery_score,
            "hrv_rmssd_milli": hrv,
            "sleep_performance": sleep_perf,
        },
        "slack": {
            "messages_received": slack_received,
        },
        "rescuetime": {
            "active_seconds": rt_active,
            "productivity_score": rt_prod,
        },
        "omi": {
            "conversation_active": omi_active,
            "word_count": omi_words,
        },
        "metrics": {
            "cognitive_load_score": cls_stored or 0.1,
        },
    }


def _make_day_windows(n=96, **kwargs) -> list:
    return [_make_window(**kwargs) for _ in range(n)]


# ─── 1. Helpers ───────────────────────────────────────────────────────────────

class TestClampNorm:
    def test_clamp_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below_zero(self):
        assert _clamp(-0.1) == 0.0

    def test_clamp_above_one(self):
        assert _clamp(1.5) == 1.0

    def test_norm_zero(self):
        assert _norm(0, max_val=100) == 0.0

    def test_norm_max(self):
        assert _norm(100, max_val=100) == 1.0

    def test_norm_half(self):
        assert abs(_norm(50, max_val=100) - 0.5) < 1e-9

    def test_norm_none(self):
        assert _norm(None, max_val=100) == 0.0

    def test_norm_clamps_above_max(self):
        assert _norm(200, max_val=100) == 1.0


# ─── 2. Physiological readiness ───────────────────────────────────────────────

class TestPhysiologicalReadiness:
    def test_perfect_readiness(self):
        # High recovery + HRV + sleep → close to 1.0
        r = _physiological_readiness(100.0, 130.0, 100.0)
        assert r > 0.9

    def test_zero_readiness(self):
        r = _physiological_readiness(0.0, 30.0, 0.0)
        assert r < 0.3

    def test_none_all_defaults_to_half(self):
        r = _physiological_readiness(None, None, None)
        assert 0.4 < r < 0.6

    def test_only_recovery_score(self):
        r = _physiological_readiness(85.0)
        assert 0.7 < r < 1.0


# ─── 3. Window decomposition ──────────────────────────────────────────────────

class TestDecomposeWindow:
    def test_empty_window_physiology_only(self):
        w = _make_window(
            in_meeting=False, meeting_attendees=0, slack_received=0,
            rt_active=0, rt_prod=None, omi_active=False,
        )
        comp = _decompose_window(w)
        assert comp["meetings"] == 0.0
        assert comp["slack"] == 0.0
        assert comp["omi"] == 0.0
        assert comp["physiology"] > 0.0  # recovery_inverse always present

    def test_social_meeting_raises_meetings(self):
        w = _make_window(in_meeting=True, meeting_attendees=4)
        comp = _decompose_window(w)
        assert comp["meetings"] > 0.0

    def test_solo_meeting_no_meetings_component(self):
        # attendees <= 1 → solo block, no meetings overhead
        w = _make_window(in_meeting=True, meeting_attendees=1)
        comp = _decompose_window(w)
        assert comp["meetings"] == 0.0

    def test_slack_messages_raise_slack_component(self):
        w = _make_window(slack_received=20)
        comp = _decompose_window(w)
        assert comp["slack"] > 0.0

    def test_rt_low_productivity_raises_rescuetime(self):
        w = _make_window(rt_active=600, rt_prod=0.1)
        comp = _decompose_window(w)
        assert comp["rescuetime"] > 0.0

    def test_rt_inactive_no_rescuetime(self):
        # Machine not active enough → RT signal skipped
        w = _make_window(rt_active=10, rt_prod=0.1)
        comp = _decompose_window(w)
        assert comp["rescuetime"] == 0.0

    def test_omi_conversation_raises_omi(self):
        w = _make_window(omi_active=True, omi_words=200)
        comp = _decompose_window(w)
        assert comp["omi"] > 0.0

    def test_omi_inactive_no_omi(self):
        w = _make_window(omi_active=False)
        comp = _decompose_window(w)
        assert comp["omi"] == 0.0

    def test_components_sum_approximately_to_cls_computed(self):
        w = _make_window(in_meeting=True, meeting_attendees=3, slack_received=10,
                         rt_active=600, rt_prod=0.5, omi_active=True, omi_words=100)
        comp = _decompose_window(w)
        total = comp["meetings"] + comp["slack"] + comp["physiology"] + comp["rescuetime"] + comp["omi"]
        # Within 1% of cls_computed (rounding differences)
        assert abs(total - comp["cls_computed"]) < 0.02

    def test_cls_computed_in_range(self):
        w = _make_window(in_meeting=True, meeting_attendees=8, slack_received=25,
                         rt_active=900, rt_prod=0.0, recovery_score=20.0)
        comp = _decompose_window(w)
        assert 0.0 <= comp["cls_computed"] <= 1.0

    def test_high_load_window(self):
        # Heavy meeting + lots of Slack + low recovery → CLS should be high
        w = _make_window(
            in_meeting=True, meeting_attendees=8, slack_received=25,
            recovery_score=20.0, hrv=30.0, sleep_perf=40.0,
        )
        comp = _decompose_window(w)
        assert comp["cls_computed"] > 0.4

    def test_zero_load_window(self):
        # Perfect recovery, no meetings, no Slack → CLS should be very low
        w = _make_window(
            in_meeting=False, slack_received=0, recovery_score=100.0,
            hrv=130.0, sleep_perf=100.0,
        )
        comp = _decompose_window(w)
        assert comp["cls_computed"] < 0.1


# ─── 4. compute_load_decomposition ───────────────────────────────────────────

class TestComputeLoadDecomposition:
    def test_no_data_returns_not_meaningful(self):
        with patch("analysis.load_decomposer.read_day", return_value=[]):
            result = compute_load_decomposition("2025-01-01")
        assert not result.is_meaningful

    def test_insufficient_active_windows_not_meaningful(self):
        # Only 1 window with CLS > MIN_WINDOW_CLS → not meaningful (need >= 2)
        windows = [_make_window(recovery_score=100.0, slack_received=0)] * 1
        # Ensure CLS is low enough to fall below MIN_WINDOW_CLS for remaining windows
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        # 1 window — below MIN_ACTIVE_WINDOWS=2
        assert not result.is_meaningful

    def test_valid_day_returns_meaningful(self):
        windows = _make_day_windows(n=96, slack_received=5)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        assert result.is_meaningful

    def test_source_shares_sum_to_one(self):
        windows = _make_day_windows(n=96, slack_received=5, in_meeting=True, meeting_attendees=3)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        assert result.is_meaningful
        total = sum(result.source_shares.values())
        assert abs(total - 1.0) < 0.01

    def test_all_five_sources_in_shares(self):
        windows = _make_day_windows(n=10, slack_received=5)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        for key in ["meetings", "slack", "physiology", "rescuetime", "omi"]:
            assert key in result.source_shares

    def test_dominant_source_matches_max_share(self):
        windows = _make_day_windows(n=10, recovery_score=20.0, hrv=30.0)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        assert result.is_meaningful
        expected_dom = max(result.source_shares, key=lambda k: result.source_shares[k])
        assert result.dominant_source == expected_dom

    def test_meeting_heavy_day_meetings_dominant(self):
        windows = _make_day_windows(
            n=10,
            in_meeting=True, meeting_attendees=6,
            recovery_score=90.0, hrv=90.0, slack_received=0,
        )
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        assert result.is_meaningful
        assert result.dominant_source == "meetings"

    def test_insight_lines_is_list(self):
        windows = _make_day_windows(n=10, slack_received=5)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        assert isinstance(result.insight_lines, list)

    def test_to_dict_json_roundtrip(self):
        windows = _make_day_windows(n=10, slack_received=5)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        d = result.to_dict()
        serialised = json.dumps(d)
        restored = json.loads(serialised)
        assert restored["date_str"] == "2025-01-01"
        assert "source_shares" in restored

    def test_total_cls_mean_non_negative(self):
        windows = _make_day_windows(n=96)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        assert result.total_cls_mean >= 0.0

    def test_windows_analysed_equals_window_count(self):
        windows = _make_day_windows(n=20)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        assert result.windows_analysed == 20

    def test_rescuetime_dominant_on_distracted_day(self):
        windows = _make_day_windows(
            n=10,
            recovery_score=90.0, hrv=90.0,
            rt_active=900, rt_prod=0.0,   # 100% distracted
            in_meeting=False, slack_received=0,
        )
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        assert result.is_meaningful
        assert result.dominant_source == "rescuetime"

    def test_omi_dominant_on_conversation_heavy_day(self):
        windows = _make_day_windows(
            n=10,
            recovery_score=90.0, hrv=90.0,
            omi_active=True, omi_words=500,
            in_meeting=False, slack_received=0, rt_active=0, rt_prod=None,
        )
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            result = compute_load_decomposition("2025-01-01")
        assert result.is_meaningful
        assert result.dominant_source == "omi"


# ─── 5. compute_week_decomposition ───────────────────────────────────────────

class TestComputeWeekDecomposition:
    def test_no_data_returns_zero_meaningful(self):
        with patch("analysis.load_decomposer.read_day", return_value=[]):
            week = compute_week_decomposition("2025-01-07", days=7)
        assert week["days_meaningful"] == 0

    def test_week_returns_correct_count(self):
        windows = _make_day_windows(n=10, slack_received=5)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            week = compute_week_decomposition("2025-01-07", days=7)
        assert week["days_meaningful"] == 7

    def test_weekly_shares_sum_to_one(self):
        windows = _make_day_windows(n=10, slack_received=5)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            week = compute_week_decomposition("2025-01-07", days=7)
        total = sum(week["weekly_shares"].values())
        assert abs(total - 1.0) < 0.01

    def test_dominant_source_present(self):
        windows = _make_day_windows(n=10, in_meeting=True, meeting_attendees=5,
                                    recovery_score=90.0, hrv=90.0)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            week = compute_week_decomposition("2025-01-07", days=3)
        assert week["dominant_source"] in ["meetings", "slack", "physiology", "rescuetime", "omi"]

    def test_daily_list_length_matches_days(self):
        windows = _make_day_windows(n=10)
        with patch("analysis.load_decomposer.read_day", return_value=windows):
            week = compute_week_decomposition("2025-01-07", days=5)
        assert len(week["daily"]) == 5


# ─── 6. _build_insights ───────────────────────────────────────────────────────

class TestBuildInsights:
    def test_low_total_load_early_exit(self):
        lines = _build_insights(
            date_str="2025-01-01",
            shares={"meetings": 0.0, "slack": 0.1, "physiology": 0.9, "rescuetime": 0.0, "omi": 0.0},
            source_cls={},
            total_cls_mean=0.05,   # below 0.10 threshold
            active_cls_mean=0.05,
            dominant="physiology",
        )
        assert len(lines) == 1
        assert "minimal" in lines[0].lower()

    def test_meeting_dominant_mentions_meetings(self):
        lines = _build_insights(
            date_str="2025-01-01",
            shares={"meetings": 0.50, "slack": 0.20, "physiology": 0.20, "rescuetime": 0.05, "omi": 0.05},
            source_cls={},
            total_cls_mean=0.50,
            active_cls_mean=0.50,
            dominant="meetings",
        )
        assert any("meeting" in l.lower() for l in lines)

    def test_slack_dominant_mentions_slack(self):
        lines = _build_insights(
            date_str="2025-01-01",
            shares={"meetings": 0.1, "slack": 0.50, "physiology": 0.30, "rescuetime": 0.05, "omi": 0.05},
            source_cls={},
            total_cls_mean=0.40,
            active_cls_mean=0.40,
            dominant="slack",
        )
        assert any("slack" in l.lower() for l in lines)

    def test_physiology_dominant_mentions_recovery(self):
        lines = _build_insights(
            date_str="2025-01-01",
            shares={"meetings": 0.05, "slack": 0.05, "physiology": 0.60, "rescuetime": 0.25, "omi": 0.05},
            source_cls={},
            total_cls_mean=0.40,
            active_cls_mean=0.40,
            dominant="physiology",
        )
        assert any("recovery" in l.lower() or "sleep" in l.lower() for l in lines)

    def test_omi_dominant_mentions_conversation(self):
        lines = _build_insights(
            date_str="2025-01-01",
            shares={"meetings": 0.05, "slack": 0.05, "physiology": 0.20, "rescuetime": 0.05, "omi": 0.65},
            source_cls={},
            total_cls_mean=0.40,
            active_cls_mean=0.40,
            dominant="omi",
        )
        assert any("conversation" in l.lower() or "talk" in l.lower() for l in lines)

    def test_max_three_lines(self):
        lines = _build_insights(
            date_str="2025-01-01",
            shares={"meetings": 0.40, "slack": 0.30, "physiology": 0.30, "rescuetime": 0.0, "omi": 0.0},
            source_cls={},
            total_cls_mean=0.60,
            active_cls_mean=0.60,
            dominant="meetings",
        )
        assert len(lines) <= 3


# ─── 7. Formatting ───────────────────────────────────────────────────────────

class TestFormatting:
    def _make_meaningful_decomp(self) -> LoadDecomposition:
        return LoadDecomposition(
            date_str="2025-01-01",
            total_cls_mean=0.45,
            active_cls_mean=0.48,
            windows_analysed=96,
            active_windows=50,
            source_shares={
                "meetings": 0.42,
                "slack": 0.28,
                "physiology": 0.18,
                "rescuetime": 0.10,
                "omi": 0.02,
            },
            source_cls={
                "meetings": 0.19,
                "slack": 0.13,
                "physiology": 0.08,
                "rescuetime": 0.05,
                "omi": 0.01,
            },
            dominant_source="meetings",
            insight_lines=["Meetings were the biggest driver today."],
            is_meaningful=True,
        )

    def _make_not_meaningful_decomp(self) -> LoadDecomposition:
        return LoadDecomposition(date_str="2025-01-01", is_meaningful=False)

    def test_decomposition_line_empty_when_not_meaningful(self):
        d = self._make_not_meaningful_decomp()
        assert format_decomposition_line(d) == ""

    def test_decomposition_line_contains_load_breakdown(self):
        d = self._make_meaningful_decomp()
        line = format_decomposition_line(d)
        assert "Load breakdown" in line or "load breakdown" in line.lower()

    def test_decomposition_line_contains_meetings_pct(self):
        d = self._make_meaningful_decomp()
        line = format_decomposition_line(d)
        # Meetings at 42% should be in the line
        assert "42%" in line

    def test_decomposition_section_empty_when_not_meaningful(self):
        d = self._make_not_meaningful_decomp()
        assert format_decomposition_section(d) == ""

    def test_decomposition_section_contains_all_five_emojis(self):
        d = self._make_meaningful_decomp()
        section = format_decomposition_section(d)
        for emoji in ["📅", "💬", "💤", "💻", "🎙"]:
            assert emoji in section

    def test_decomposition_section_contains_heading(self):
        d = self._make_meaningful_decomp()
        section = format_decomposition_section(d)
        assert "drove" in section.lower() or "load" in section.lower()

    def test_decomposition_terminal_contains_date(self):
        d = self._make_meaningful_decomp()
        term = format_decomposition_terminal(d)
        assert "2025-01-01" in term

    def test_decomposition_terminal_not_meaningful(self):
        d = self._make_not_meaningful_decomp()
        term = format_decomposition_terminal(d)
        assert "Not enough" in term or "not enough" in term.lower()

    def test_week_section_empty_when_no_data(self):
        week = {"days_meaningful": 0, "weekly_shares": {}, "dominant_source": "unknown"}
        assert format_week_decomposition_section(week) == ""

    def test_week_section_contains_weekly_heading(self):
        week = {
            "days_meaningful": 5,
            "weekly_shares": {
                "meetings": 0.40,
                "slack": 0.25,
                "physiology": 0.20,
                "rescuetime": 0.10,
                "omi": 0.05,
            },
            "weekly_cls": 0.35,
            "dominant_source": "meetings",
        }
        section = format_week_decomposition_section(week)
        assert "Weekly" in section or "week" in section.lower()

    def test_week_section_mentions_dominant(self):
        week = {
            "days_meaningful": 5,
            "weekly_shares": {
                "meetings": 0.40,
                "slack": 0.25,
                "physiology": 0.20,
                "rescuetime": 0.10,
                "omi": 0.05,
            },
            "weekly_cls": 0.35,
            "dominant_source": "meetings",
        }
        section = format_week_decomposition_section(week)
        assert "Meetings" in section or "meeting" in section.lower()

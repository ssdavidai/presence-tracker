"""
Tests for scripts/report.py — Per-Day Terminal Report

Coverage:
  - build_report() returns correct structure for a day with data
  - build_report() returns None when no data exists
  - _bar() renders correct block characters and respects width
  - _heatmap_line() produces correct characters for each CLS tier
  - _label_cls / _label_fdi / _label_ras return correct labels
  - _tier_label() maps recovery scores to correct tier strings
  - _delta_str() formats positive/negative deltas correctly
  - print_compact() outputs non-empty text without crashing
  - print_full() outputs non-empty text without crashing
  - print_full() includes expected sections
  - --json mode output is valid JSON with expected keys
  - --compare flag populates baseline in report dict
  - --windows flag triggers window table rendering
  - CLI exits with error on invalid date
  - CLI exits with error when no data for date
  - _meeting_blocks() groups contiguous meeting windows correctly
  - _omi_stats() returns None when no Omi windows
  - _omi_stats() aggregates correctly when Omi data present
  - _rt_stats() returns None when no RescueTime windows
  - _rt_stats() aggregates correctly when RT data present
  - _baseline() returns None when no summary data available
  - _baseline() computes correct averages from rolling summary

All external I/O (store reads) is mocked so tests run offline.
"""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.report import (
    _bar,
    _baseline,
    _delta_str,
    _heatmap_line,
    _label_cls,
    _label_fdi,
    _label_ras,
    _meeting_blocks,
    _omi_stats,
    _rt_stats,
    _tier_label,
    build_report,
    print_compact,
    print_full,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_window(
    index: int,
    hour: int,
    cls: float = 0.30,
    fdi: float = 0.70,
    sdi: float = 0.10,
    csc: float = 0.10,
    ras: float = 0.80,
    is_working: bool = True,
    is_active: bool = True,
    in_meeting: bool = False,
    attendees: int = 0,
    slack_sent: int = 0,
    slack_recv: int = 0,
    omi_active: bool = False,
    omi_words: int = 0,
    rt_active: int = 0,
) -> dict:
    minute = (index % 4) * 15
    date = "2026-03-14"
    w = {
        "window_id": f"{date}T{hour:02d}:{minute:02d}:00",
        "date": date,
        "window_start": f"{date}T{hour:02d}:{minute:02d}:00+01:00",
        "window_end": f"{date}T{hour:02d}:{(minute + 15) % 60:02d}:00+01:00",
        "window_index": index,
        "calendar": {
            "in_meeting": in_meeting,
            "meeting_title": "Test Meeting" if in_meeting else None,
            "meeting_attendees": attendees,
            "meeting_duration_minutes": 30 if in_meeting else 0,
            "meeting_organizer": None,
            "meetings_count": 1 if in_meeting else 0,
        },
        "whoop": {
            "recovery_score": 85.0,
            "hrv_rmssd_milli": 72.5,
            "resting_heart_rate": 55.0,
            "sleep_performance": 88.0,
            "sleep_hours": 8.0,
            "strain": 13.0,
            "spo2_percentage": 96.0,
        },
        "slack": {
            "messages_sent": slack_sent,
            "messages_received": slack_recv,
            "total_messages": slack_sent + slack_recv,
            "channels_active": 0,
        },
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "context_switch_cost": csc,
            "recovery_alignment_score": ras,
        },
        "metadata": {
            "day_of_week": "Saturday",
            "hour_of_day": hour,
            "minute_of_hour": minute,
            "is_working_hours": is_working,
            "sources_available": ["whoop", "calendar", "slack"],
            "is_active_window": is_active,
        },
    }
    if omi_active:
        w["omi"] = {
            "conversation_active": True,
            "word_count": omi_words,
            "speech_seconds": 60.0,
            "audio_seconds": 120.0,
            "sessions_count": 1,
            "speech_ratio": 0.5,
            # v10.1: topic classifier fields (optional — mirrors real JSONL output)
            "topic_category": None,
            "cognitive_density": None,
            "cls_weight": 1.0,
            "sdi_weight": 1.0,
            "topic_signals": [],
        }
    if rt_active > 0:
        w["rescuetime"] = {
            "active_seconds": rt_active,
            "focus_seconds": rt_active // 2,
            "distraction_seconds": rt_active // 4,
            "neutral_seconds": rt_active // 4,
            "productivity_score": 0.70,
            "app_switches": 3,
            "top_activity": "VS Code",
        }
    return w


def _make_day_windows(n_active: int = 5) -> list[dict]:
    """Create a minimal set of 96 windows for a test day."""
    windows = []
    # 96 windows: indices 0..95
    # working hours: 7am–10pm → indices 28..88 (roughly)
    for i in range(96):
        hour = i // 4
        is_working = 7 <= hour <= 21
        is_active = is_working and i < (28 + n_active * 4)
        w = _make_window(
            index=i,
            hour=hour,
            cls=0.35 if is_active else 0.04,
            fdi=0.70 if is_active else 0.95,
            sdi=0.15 if is_active else 0.00,
            csc=0.10 if is_active else 0.00,
            ras=0.80,
            is_working=is_working,
            is_active=is_active,
            slack_sent=2 if is_active else 0,
            slack_recv=5 if is_active else 0,
        )
        windows.append(w)
    return windows


SAMPLE_ROLLING = {
    "days": {
        "2026-03-13": {
            "date": "2026-03-13",
            "metrics_avg": {
                "cognitive_load_score": 0.32,
                "focus_depth_index": 0.75,
                "social_drain_index": 0.18,
                "context_switch_cost": 0.22,
                "recovery_alignment_score": 0.81,
            },
            "whoop": {"recovery_score": 78.0, "hrv_rmssd_milli": 65.0},
        },
        "2026-03-14": {
            "date": "2026-03-14",
            "metrics_avg": {
                "cognitive_load_score": 0.40,
                "focus_depth_index": 0.68,
                "social_drain_index": 0.20,
                "context_switch_cost": 0.25,
                "recovery_alignment_score": 0.78,
            },
            "whoop": {"recovery_score": 85.0, "hrv_rmssd_milli": 72.0},
        },
    },
    "total_days": 2,
    "last_updated": "2026-03-14T23:45:00",
}


# ─── _bar() ───────────────────────────────────────────────────────────────────

class TestBar:
    def test_zero_is_all_empty(self):
        result = _bar(0.0, width=10)
        assert result.count("░") == 10
        assert "▓" not in result

    def test_one_is_all_filled(self):
        result = _bar(1.0, width=10)
        assert "▓" * 10 in result
        assert "░" not in result.replace("\033[92m", "").replace("\033[0m", "")

    def test_half_fills_half(self):
        result = _bar(0.5, width=10)
        # Strip ANSI
        clean = "".join(c for c in result if c in ("▓", "░"))
        assert len(clean) == 10
        assert clean.count("▓") == 5
        assert clean.count("░") == 5

    def test_width_is_respected(self):
        for w in [5, 8, 12, 20]:
            result = _bar(0.5, width=w)
            clean = "".join(c for c in result if c in ("▓", "░"))
            assert len(clean) == w


# ─── _heatmap_line() ──────────────────────────────────────────────────────────

class TestHeatmapLine:
    def test_very_light_is_light_shade(self):
        hmap = _heatmap_line({7: 0.05}, start=7, end=7)
        assert "░" in hmap

    def test_mild_is_medium_shade(self):
        hmap = _heatmap_line({7: 0.15}, start=7, end=7)
        assert "▒" in hmap

    def test_moderate_is_dark_shade(self):
        hmap = _heatmap_line({7: 0.35}, start=7, end=7)
        assert "▓" in hmap

    def test_heavy_is_full_block(self):
        hmap = _heatmap_line({7: 0.60}, start=7, end=7)
        assert "█" in hmap

    def test_missing_hour_is_dot(self):
        hmap = _heatmap_line({}, start=7, end=7)
        assert "·" in hmap

    def test_length_matches_range(self):
        hmap = _heatmap_line({}, start=7, end=22)
        assert len(hmap) == 16  # 7..22 inclusive = 16 chars

    def test_full_working_day(self):
        cls_vals = {h: 0.30 for h in range(7, 22)}
        hmap = _heatmap_line(cls_vals, start=7, end=21)
        assert len(hmap) == 15


# ─── Label functions ──────────────────────────────────────────────────────────

class TestLabelCls:
    def test_light(self):
        assert _label_cls(0.10) == "light"

    def test_moderate(self):
        assert _label_cls(0.30) == "moderate"

    def test_heavy(self):
        assert _label_cls(0.50) == "heavy"

    def test_intense(self):
        assert _label_cls(0.70) == "intense"

    def test_boundaries(self):
        assert _label_cls(0.20) == "moderate"  # boundary: >= 0.20 → moderate
        assert _label_cls(0.40) == "heavy"     # boundary: >= 0.40 → heavy


class TestLabelFdi:
    def test_fragmented(self):
        assert _label_fdi(0.20) == "fragmented"

    def test_interrupted(self):
        assert _label_fdi(0.40) == "interrupted"

    def test_solid(self):
        assert _label_fdi(0.65) == "solid"

    def test_deep(self):
        assert _label_fdi(0.85) == "deep"


class TestLabelRas:
    def test_misaligned(self):
        assert _label_ras(0.20) == "misaligned"

    def test_moderate(self):
        assert _label_ras(0.45) == "moderate"

    def test_aligned(self):
        assert _label_ras(0.65) == "aligned"

    def test_excellent(self):
        assert _label_ras(0.85) == "excellent"


class TestTierLabel:
    def test_peak(self):
        assert _tier_label(85.0) == "Peak"

    def test_good(self):
        assert _tier_label(70.0) == "Good"

    def test_moderate(self):
        assert _tier_label(55.0) == "Moderate"

    def test_low(self):
        assert _tier_label(40.0) == "Low"

    def test_recovery(self):
        assert _tier_label(20.0) == "Recovery"

    def test_none_is_unknown(self):
        assert _tier_label(None) == "unknown"

    def test_boundary_80(self):
        assert _tier_label(80.0) == "Peak"

    def test_boundary_67(self):
        assert _tier_label(67.0) == "Good"


# ─── _delta_str() ─────────────────────────────────────────────────────────────

class TestDeltaStr:
    def test_flat_returns_arrow(self):
        result = _delta_str(0.30, 0.30)
        assert "→" in result

    def test_positive_shows_plus(self):
        result = _delta_str(0.40, 0.30)
        clean = "".join(c for c in result if c not in "\033[0123456789;m")
        assert "+" in clean

    def test_negative_shows_minus(self):
        result = _delta_str(0.20, 0.30)
        clean = "".join(c for c in result if c not in "\033[0123456789;m")
        assert "-" in clean

    def test_none_today_returns_empty(self):
        assert _delta_str(None, 0.30) == ""

    def test_none_baseline_returns_empty(self):
        assert _delta_str(0.30, None) == ""

    def test_both_none_returns_empty(self):
        assert _delta_str(None, None) == ""

    def test_small_delta_below_threshold_is_flat(self):
        # < 0.01 difference → flat arrow
        result = _delta_str(0.305, 0.300)
        assert "→" in result


# ─── _meeting_blocks() ───────────────────────────────────────────────────────

class TestMeetingBlocks:
    def test_no_meetings_returns_empty(self):
        windows = _make_day_windows(0)
        blocks = _meeting_blocks(windows)
        assert blocks == []

    def test_single_meeting_block(self):
        windows = _make_day_windows(0)
        # Mark two consecutive windows as a meeting
        windows[32]["calendar"]["in_meeting"] = True
        windows[32]["calendar"]["meeting_title"] = "Standup"
        windows[32]["calendar"]["meeting_attendees"] = 5
        windows[33]["calendar"]["in_meeting"] = True
        windows[33]["calendar"]["meeting_title"] = "Standup"
        windows[33]["calendar"]["meeting_attendees"] = 5
        blocks = _meeting_blocks(windows)
        assert len(blocks) == 1
        assert blocks[0]["title"] == "Standup"
        assert blocks[0]["windows"] == 2
        assert blocks[0]["attendees"] == 5

    def test_two_separate_meetings(self):
        windows = _make_day_windows(0)
        windows[32]["calendar"]["in_meeting"] = True
        windows[32]["calendar"]["meeting_title"] = "A"
        windows[40]["calendar"]["in_meeting"] = True
        windows[40]["calendar"]["meeting_title"] = "B"
        blocks = _meeting_blocks(windows)
        assert len(blocks) == 2

    def test_meeting_title_changes_creates_new_block(self):
        windows = _make_day_windows(0)
        windows[32]["calendar"]["in_meeting"] = True
        windows[32]["calendar"]["meeting_title"] = "Meeting A"
        windows[33]["calendar"]["in_meeting"] = True
        windows[33]["calendar"]["meeting_title"] = "Meeting B"
        blocks = _meeting_blocks(windows)
        assert len(blocks) == 2


# ─── _omi_stats() ─────────────────────────────────────────────────────────────

class TestOmiStats:
    def test_no_omi_returns_none(self):
        windows = _make_day_windows(3)
        assert _omi_stats(windows) is None

    def test_with_omi_returns_dict(self):
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=150)
        windows[33] = _make_window(33, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=200)
        result = _omi_stats(windows)
        assert result is not None
        assert result["total_words"] == 350
        assert result["conversation_windows"] == 2

    def test_sessions_summed(self):
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=100)
        windows[32]["omi"]["sessions_count"] = 2
        windows[33] = _make_window(33, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=100)
        windows[33]["omi"]["sessions_count"] = 3
        result = _omi_stats(windows)
        assert result["total_sessions"] == 5

    def test_speech_minutes_computed(self):
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True)
        windows[32]["omi"]["speech_seconds"] = 120.0  # 2 minutes
        result = _omi_stats(windows)
        assert result["total_speech_minutes"] == 2.0

    # ── v10.1 topic fields ────────────────────────────────────────────────────

    def test_no_topic_data_dominant_topic_is_none(self):
        """When no topic_category in omi windows, dominant_topic is None."""
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=200)
        # topic_category is None in the default _make_window helper
        result = _omi_stats(windows)
        assert result["dominant_topic"] is None
        assert result["category_counts"] is None

    def test_topic_category_counted(self):
        """When topic_category is set, category_counts reflects it."""
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=300)
        windows[32]["omi"]["topic_category"] = "work_technical"
        windows[33] = _make_window(33, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=200)
        windows[33]["omi"]["topic_category"] = "work_technical"
        result = _omi_stats(windows)
        assert result["category_counts"] == {"work_technical": 2}
        assert result["dominant_topic"] == "work_technical"

    def test_mixed_topics_dominant_selected(self):
        """When multiple categories, dominant_topic is the most frequent."""
        windows = _make_day_windows(0)
        # 2 personal windows, 1 technical window
        for i, (cat, words) in enumerate([("personal", 200), ("personal", 150), ("work_technical", 400)]):
            idx = 32 + i
            windows[idx] = _make_window(idx, 8, is_working=True, is_active=True,
                                        omi_active=True, omi_words=words)
            windows[idx]["omi"]["topic_category"] = cat
        result = _omi_stats(windows)
        assert result["dominant_topic"] == "personal"
        assert result["category_counts"]["personal"] == 2
        assert result["category_counts"]["work_technical"] == 1

    def test_cognitive_density_averaged(self):
        """mean_cognitive_density should average density across Omi windows."""
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=200)
        windows[32]["omi"]["cognitive_density"] = 0.6
        windows[33] = _make_window(33, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=200)
        windows[33]["omi"]["cognitive_density"] = 0.8
        result = _omi_stats(windows)
        assert result["mean_cognitive_density"] == pytest.approx(0.7, abs=0.01)

    def test_no_cognitive_density_returns_none(self):
        """mean_cognitive_density is None when no density values in windows."""
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=200)
        # cognitive_density is None in default _make_window omi block
        result = _omi_stats(windows)
        assert result["mean_cognitive_density"] is None

    def test_unknown_category_excluded_from_counts(self):
        """'unknown' category should not appear in category_counts."""
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=100)
        windows[32]["omi"]["topic_category"] = "unknown"
        result = _omi_stats(windows)
        assert result["category_counts"] is None or "unknown" not in (result["category_counts"] or {})

    def test_result_has_all_required_keys(self):
        """_omi_stats result must contain both v2 and v10.1 keys."""
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=100)
        result = _omi_stats(windows)
        assert result is not None
        # v2.0 keys
        assert "conversation_windows" in result
        assert "total_sessions" in result
        assert "total_words" in result
        assert "total_speech_minutes" in result
        # v10.1 keys
        assert "dominant_topic" in result
        assert "category_counts" in result
        assert "mean_cognitive_density" in result


# ─── _rt_stats() ──────────────────────────────────────────────────────────────

class TestRtStats:
    def test_no_rt_returns_none(self):
        windows = _make_day_windows(3)
        assert _rt_stats(windows) is None

    def test_with_rt_returns_dict(self):
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   rt_active=600)  # 10 min
        windows[33] = _make_window(33, 8, is_working=True, is_active=True,
                                   rt_active=600)
        result = _rt_stats(windows)
        assert result is not None
        assert result["active_minutes"] == 20.0

    def test_productive_pct_computed(self):
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   rt_active=600)
        # focus = rt_active // 2 = 300s, active = 600s → 50%
        result = _rt_stats(windows)
        assert result["productive_pct"] == 50.0

    def test_zero_active_returns_none(self):
        windows = _make_day_windows(0)
        # window has rescuetime key but active_seconds=0 → should not count
        windows[32]["rescuetime"] = {"active_seconds": 0, "focus_seconds": 0,
                                     "distraction_seconds": 0, "neutral_seconds": 0}
        result = _rt_stats(windows)
        assert result is None


# ─── _baseline() ──────────────────────────────────────────────────────────────

class TestBaseline:
    def test_no_summary_returns_none(self):
        with patch("scripts.report.read_summary", return_value={"days": {}}):
            result = _baseline(7)
        assert result is None

    def test_with_data_returns_averages(self):
        with patch("scripts.report.read_summary", return_value=SAMPLE_ROLLING):
            result = _baseline(7)
        assert result is not None
        assert result["cls"] is not None
        assert result["fdi"] is not None
        assert result["days"] == 2

    def test_cls_average_is_correct(self):
        with patch("scripts.report.read_summary", return_value=SAMPLE_ROLLING):
            result = _baseline(7)
        # (0.32 + 0.40) / 2 = 0.36
        assert abs(result["cls"] - 0.36) < 0.001

    def test_days_capped_by_available(self):
        with patch("scripts.report.read_summary", return_value=SAMPLE_ROLLING):
            result = _baseline(30)  # request 30 days but only 2 available
        assert result["days"] == 2


# ─── build_report() ───────────────────────────────────────────────────────────

class TestBuildReport:
    def setup_method(self):
        self.windows = _make_day_windows(10)

    def test_returns_none_for_missing_date(self):
        with patch("scripts.report.read_day", return_value=[]):
            result = build_report("2026-01-01")
        assert result is None

    def test_returns_dict_for_valid_day(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14")
        assert isinstance(result, dict)

    def test_has_required_top_level_keys(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14")
        required = ["date", "day_of_week", "whoop", "metrics", "focus",
                    "slack", "calendar", "omi", "rescuetime",
                    "sources_available", "hourly_cls"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_date_matches_input(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14")
        assert result["date"] == "2026-03-14"

    def test_day_of_week_is_correct(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14")
        assert result["day_of_week"] == "Saturday"

    def test_metrics_has_five_keys(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14")
        assert set(result["metrics"].keys()) == {"cls", "fdi", "sdi", "csc", "ras"}

    def test_focus_has_required_keys(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14")
        assert "active_fdi" in result["focus"]
        assert "active_windows" in result["focus"]
        assert "peak_focus_hour" in result["focus"]

    def test_slack_totals_are_correct(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14")
        expected_sent = sum(w["slack"]["messages_sent"] for w in self.windows)
        assert result["slack"]["messages_sent"] == expected_sent

    def test_baseline_none_when_compare_zero(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14", compare_days=0)
        assert result["baseline"] is None

    def test_baseline_populated_when_compare_nonzero(self):
        with patch("scripts.report.read_day", return_value=self.windows), \
             patch("scripts.report.read_summary", return_value=SAMPLE_ROLLING):
            result = build_report("2026-03-14", compare_days=7)
        assert result["baseline"] is not None

    def test_omi_none_when_no_omi_windows(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14")
        assert result["omi"] is None

    def test_omi_populated_when_omi_windows_present(self):
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=250)
        with patch("scripts.report.read_day", return_value=windows):
            result = build_report("2026-03-14")
        assert result["omi"] is not None
        assert result["omi"]["total_words"] == 250

    def test_hourly_cls_only_working_hours(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14")
        # Non-working hours should not appear (or have zero active windows)
        for h in result["hourly_cls"]:
            assert 7 <= int(h) <= 21, f"Non-working hour {h} in hourly_cls"

    def test_sources_available_is_sorted_list(self):
        with patch("scripts.report.read_day", return_value=self.windows):
            result = build_report("2026-03-14")
        assert isinstance(result["sources_available"], list)
        assert result["sources_available"] == sorted(result["sources_available"])


# ─── print_compact() ─────────────────────────────────────────────────────────

class TestPrintCompact:
    def test_does_not_crash(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_compact(report)
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_output_contains_date(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_compact(report)
        out = capsys.readouterr().out
        assert "2026-03-14" in out

    def test_output_contains_heatmap_chars(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_compact(report)
        out = capsys.readouterr().out
        heatmap_chars = {"░", "▒", "▓", "█", "·"}
        assert any(c in out for c in heatmap_chars)

    def test_output_contains_cls(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_compact(report)
        out = capsys.readouterr().out
        assert "CLS" in out


# ─── print_full() ─────────────────────────────────────────────────────────────

class TestPrintFull:
    def test_does_not_crash(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        assert len(capsys.readouterr().out) > 0

    def test_output_contains_whoop_section(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "WHOOP" in out

    def test_output_contains_metrics_section(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "CLS" in out
        assert "FDI" in out
        assert "RAS" in out

    def test_output_contains_hourly_section(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "Hourly" in out

    def test_output_contains_focus_section(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "Focus" in out

    def test_output_contains_slack_section(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "Slack" in out

    def test_output_contains_sources(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "Sources" in out

    def test_no_meetings_shows_no_meetings_message(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "No meetings" in out

    def test_meetings_shown_when_present(self, capsys):
        windows = _make_day_windows(0)
        windows[32]["calendar"]["in_meeting"] = True
        windows[32]["calendar"]["meeting_title"] = "Standup"
        windows[32]["calendar"]["meeting_attendees"] = 4
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "Standup" in out

    def test_omi_section_shown_when_present(self, capsys):
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   omi_active=True, omi_words=300)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "Omi" in out

    def test_omi_section_absent_when_no_omi(self, capsys):
        windows = _make_day_windows(5)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "Omi" not in out

    def test_rt_section_shown_when_present(self, capsys):
        windows = _make_day_windows(0)
        windows[32] = _make_window(32, 8, is_working=True, is_active=True,
                                   rt_active=900)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert "RescueTime" in out

    def test_baseline_delta_shown_when_compare_set(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows), \
             patch("scripts.report.read_summary", return_value=SAMPLE_ROLLING):
            report = build_report("2026-03-14", compare_days=7)
        print_full(report)
        out = capsys.readouterr().out
        assert "average" in out.lower() or "Δ" in out

    def test_full_output_is_multi_line(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report)
        out = capsys.readouterr().out
        assert out.count("\n") >= 20

    def test_show_windows_includes_table_header(self, capsys):
        windows = _make_day_windows(5)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report, show_windows=True)
        out = capsys.readouterr().out
        assert "Time" in out and "CLS" in out and "FDI" in out

    def test_show_windows_false_no_table(self, capsys):
        windows = _make_day_windows(5)
        with patch("scripts.report.read_day", return_value=windows):
            report = build_report("2026-03-14")
        print_full(report, show_windows=False)
        out = capsys.readouterr().out
        # Table header should not appear when show_windows=False
        assert "Time    CLS  FDI" not in out


# ─── JSON output ─────────────────────────────────────────────────────────────

class TestJsonOutput:
    def test_json_is_valid(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows), \
             patch("scripts.report.list_available_dates", return_value=["2026-03-14"]):
            from scripts.report import main as report_main
            import sys
            old_argv = sys.argv
            sys.argv = ["report.py", "2026-03-14", "--json"]
            try:
                report_main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, dict)

    def test_json_has_expected_keys(self, capsys):
        windows = _make_day_windows(10)
        with patch("scripts.report.read_day", return_value=windows), \
             patch("scripts.report.list_available_dates", return_value=["2026-03-14"]):
            from scripts.report import main as report_main
            import sys
            old_argv = sys.argv
            sys.argv = ["report.py", "2026-03-14", "--json"]
            try:
                report_main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv
        out = capsys.readouterr().out
        parsed = json.loads(out)
        for key in ["date", "whoop", "metrics", "focus", "slack", "calendar"]:
            assert key in parsed


# ─── CLI edge cases ───────────────────────────────────────────────────────────

class TestCLIEdgeCases:
    def test_invalid_date_exits_with_error(self, capsys):
        import sys
        old_argv = sys.argv
        sys.argv = ["report.py", "not-a-date"]
        try:
            from scripts.report import main as report_main
            with pytest.raises(SystemExit) as exc_info:
                report_main()
            assert exc_info.value.code != 0
        finally:
            sys.argv = old_argv

    def test_missing_date_exits_with_error(self, capsys):
        import sys
        old_argv = sys.argv
        sys.argv = ["report.py", "2020-01-01"]
        try:
            from scripts.report import main as report_main
            with patch("scripts.report.list_available_dates", return_value=[]):
                with pytest.raises(SystemExit) as exc_info:
                    report_main()
            assert exc_info.value.code != 0
        finally:
            sys.argv = old_argv

    def test_valid_date_exits_zero(self):
        windows = _make_day_windows(10)
        import sys
        old_argv = sys.argv
        sys.argv = ["report.py", "2026-03-14"]
        try:
            from scripts.report import main as report_main
            with patch("scripts.report.read_day", return_value=windows), \
                 patch("scripts.report.list_available_dates", return_value=["2026-03-14"]):
                # Should not raise SystemExit (implicit exit code 0)
                report_main()
        except SystemExit as e:
            assert e.code == 0 or e.code is None
        finally:
            sys.argv = old_argv


# ─── Trend table tests (v9.1) ─────────────────────────────────────────────────

class TestBuildTrendRows:
    """Tests for build_trend_rows() — multi-day summary extraction."""

    def test_empty_store_returns_empty_list(self):
        from scripts.report import build_trend_rows
        with patch("scripts.report.list_available_dates", return_value=[]):
            rows = build_trend_rows(14)
        assert rows == []

    def test_returns_one_row_per_available_day(self):
        from scripts.report import build_trend_rows
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=["2026-03-13", "2026-03-14"]), \
             patch("scripts.report.read_day", return_value=windows):
            rows = build_trend_rows(14)
        assert len(rows) == 2

    def test_row_has_required_keys(self):
        from scripts.report import build_trend_rows
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.report.read_day", return_value=windows):
            rows = build_trend_rows(14)
        assert len(rows) == 1
        row = rows[0]
        for key in ["date", "dow", "dps", "cls", "fdi", "ras", "recovery", "meeting_mins"]:
            assert key in row, f"Missing key: {key}"

    def test_days_limit_is_respected(self):
        from scripts.report import build_trend_rows
        available = [f"2026-03-{d:02d}" for d in range(1, 15)]  # 14 dates
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=available), \
             patch("scripts.report.read_day", return_value=windows):
            rows = build_trend_rows(7)
        assert len(rows) == 7

    def test_dates_in_ascending_order(self):
        from scripts.report import build_trend_rows
        available = ["2026-03-13", "2026-03-14"]
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=available), \
             patch("scripts.report.read_day", return_value=windows):
            rows = build_trend_rows(14)
        dates = [r["date"] for r in rows]
        assert dates == sorted(dates)

    def test_dps_is_numeric_or_none(self):
        from scripts.report import build_trend_rows
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.report.read_day", return_value=windows):
            rows = build_trend_rows(1)
        row = rows[0]
        assert row["dps"] is None or isinstance(row["dps"], (int, float))

    def test_cls_fdi_ras_are_numeric_or_none(self):
        from scripts.report import build_trend_rows
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.report.read_day", return_value=windows):
            rows = build_trend_rows(1)
        row = rows[0]
        for key in ["cls", "fdi", "ras"]:
            assert row[key] is None or isinstance(row[key], float), f"{key} not float/None"

    def test_dow_is_three_letter_string(self):
        from scripts.report import build_trend_rows
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.report.read_day", return_value=windows):
            rows = build_trend_rows(1)
        assert rows[0]["dow"] in {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}


class TestDpsEmoji:
    """Tests for _dps_emoji() — compact emoji+score formatting."""

    def test_exceptional_gets_star_emoji(self):
        from scripts.report import _dps_emoji
        result = _dps_emoji(90.0)
        assert "🌟" in result

    def test_strong_gets_check_emoji(self):
        from scripts.report import _dps_emoji
        result = _dps_emoji(80.0)
        assert "✅" in result

    def test_good_gets_yellow_circle(self):
        from scripts.report import _dps_emoji
        result = _dps_emoji(65.0)
        assert "🟡" in result

    def test_moderate_gets_orange_circle(self):
        from scripts.report import _dps_emoji
        result = _dps_emoji(50.0)
        assert "🟠" in result

    def test_poor_gets_red_circle(self):
        from scripts.report import _dps_emoji
        result = _dps_emoji(30.0)
        assert "🔴" in result

    def test_none_returns_placeholder(self):
        from scripts.report import _dps_emoji
        result = _dps_emoji(None)
        assert "—" in result

    def test_returns_string(self):
        from scripts.report import _dps_emoji
        assert isinstance(_dps_emoji(75.0), str)
        assert isinstance(_dps_emoji(None), str)

    def test_score_included_in_output(self):
        from scripts.report import _dps_emoji
        result = _dps_emoji(88.0)
        assert "88" in result


class TestSparkbar:
    """Tests for _sparkbar() — compact progress bar for trend table."""

    def test_full_value_all_filled(self):
        from scripts.report import _sparkbar
        bar = _sparkbar(1.0, 6)
        assert bar == "▓▓▓▓▓▓"

    def test_zero_value_all_empty(self):
        from scripts.report import _sparkbar
        bar = _sparkbar(0.0, 6)
        assert bar == "░░░░░░"

    def test_half_value_half_filled(self):
        from scripts.report import _sparkbar
        bar = _sparkbar(0.5, 6)
        assert bar.count("▓") == 3
        assert bar.count("░") == 3

    def test_width_respected(self):
        from scripts.report import _sparkbar
        for width in [4, 6, 8, 10]:
            bar = _sparkbar(0.5, width)
            assert len(bar) == width

    def test_none_returns_dots(self):
        from scripts.report import _sparkbar
        bar = _sparkbar(None, 6)
        assert bar == "······"

    def test_out_of_range_clamped(self):
        from scripts.report import _sparkbar
        bar_over = _sparkbar(1.5, 6)
        bar_under = _sparkbar(-0.5, 6)
        assert len(bar_over) == 6
        assert len(bar_under) == 6
        assert bar_over == "▓▓▓▓▓▓"
        assert bar_under == "░░░░░░"


class TestPrintTrend:
    """Tests for print_trend() — multi-day trend table rendering."""

    def test_prints_something_with_data(self, capsys):
        from scripts.report import print_trend
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=["2026-03-13", "2026-03-14"]), \
             patch("scripts.report.read_day", return_value=windows), \
             patch("scripts.report._cdi_tier_short", return_value="balanced"):
            print_trend(14)
        out = capsys.readouterr().out
        assert len(out.strip()) > 0

    def test_no_data_prints_error(self, capsys):
        from scripts.report import print_trend
        with patch("scripts.report.list_available_dates", return_value=[]):
            print_trend(14)
        # Should not crash; error goes to stderr
        out = capsys.readouterr()
        assert True  # just verifying no exception

    def test_output_contains_date(self, capsys):
        from scripts.report import print_trend
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.report.read_day", return_value=windows), \
             patch("scripts.report._cdi_tier_short", return_value="balanced"):
            print_trend(14)
        out = capsys.readouterr().out
        assert "2026-03-14" in out

    def test_output_contains_averages_footer(self, capsys):
        from scripts.report import print_trend
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=["2026-03-13", "2026-03-14"]), \
             patch("scripts.report.read_day", return_value=windows), \
             patch("scripts.report._cdi_tier_short", return_value="balanced"):
            print_trend(14)
        out = capsys.readouterr().out
        assert "Averages" in out

    def test_output_contains_trend_header(self, capsys):
        from scripts.report import print_trend
        windows = _make_day_windows(20)
        with patch("scripts.report.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.report.read_day", return_value=windows), \
             patch("scripts.report._cdi_tier_short", return_value="balanced"):
            print_trend(7)
        out = capsys.readouterr().out
        assert "Trend" in out

    def test_each_day_appears_as_row(self, capsys):
        from scripts.report import print_trend
        windows = _make_day_windows(20)
        dates = ["2026-03-13", "2026-03-14"]
        with patch("scripts.report.list_available_dates", return_value=dates), \
             patch("scripts.report.read_day", return_value=windows), \
             patch("scripts.report._cdi_tier_short", return_value="balanced"):
            print_trend(14)
        out = capsys.readouterr().out
        for d in dates:
            assert d in out

    def test_cli_trend_flag_triggers_trend_mode(self, capsys):
        """--trend N should call print_trend, not per-day report."""
        import sys
        windows = _make_day_windows(20)
        old_argv = sys.argv
        sys.argv = ["report.py", "--trend", "7"]
        try:
            from scripts.report import main as report_main
            with patch("scripts.report.list_available_dates", return_value=["2026-03-14"]), \
                 patch("scripts.report.read_day", return_value=windows), \
                 patch("scripts.report._cdi_tier_short", return_value="balanced"):
                report_main()
        except SystemExit as e:
            assert e.code == 0 or e.code is None
        finally:
            sys.argv = old_argv
        out = capsys.readouterr().out
        assert "Trend" in out or "2026-03-14" in out


class TestCdiTierShort:
    """Tests for _cdi_tier_short() — CDI tier lookup for trend table."""

    def test_returns_string(self):
        from scripts.report import _cdi_tier_short
        from unittest.mock import MagicMock
        mock_debt = MagicMock()
        mock_debt.tier = "balanced"
        with patch("scripts.report._cdi_tier_short.__module__"), \
             patch("analysis.cognitive_debt.compute_cdi", return_value=mock_debt):
            # Direct call should not crash
            result = _cdi_tier_short("2026-03-14")
        # Either a real result or fallback "—"
        assert isinstance(result, str)

    def test_exception_returns_dash(self):
        from scripts.report import _cdi_tier_short
        # When compute_cdi raises, should return "—" gracefully
        with patch("analysis.cognitive_debt.compute_cdi", side_effect=Exception("fail")):
            result = _cdi_tier_short("2026-03-14")
        assert result == "—"

    def test_none_cdi_returns_dash(self):
        from scripts.report import _cdi_tier_short
        with patch("analysis.cognitive_debt.compute_cdi", return_value=None):
            result = _cdi_tier_short("2026-03-14")
        assert result == "—"


class TestPrintWeek:
    """
    Tests for print_week() — the terminal weekly summary.

    Covers:
      - Output is non-empty and contains expected sections
      - No crash when called with a valid date
      - No crash when called with no data for the week
      - Week-over-week delta marker is present when prior week exists
      - CLI --week flag triggers print_week, not per-day report
      - Invalid date string does not crash (graceful error)
      - Coverage bars appear in output
      - Best/worst day section appears when data is present
    """

    # ── Fixtures ──────────────────────────────────────────────────────────────

    _ROLLING_FULL = {
        "days": {
            "2026-03-13": {
                "date": "2026-03-13",
                "metrics_avg": {
                    "cognitive_load_score": 0.32,
                    "focus_depth_index": 0.75,
                    "social_drain_index": 0.18,
                    "context_switch_cost": 0.22,
                    "recovery_alignment_score": 0.81,
                },
                "focus_quality": {
                    "active_fdi": 0.72,
                    "active_windows": 12,
                    "peak_focus_hour": 10,
                    "peak_focus_fdi": 0.85,
                },
                "whoop": {
                    "recovery_score": 78.0,
                    "hrv_rmssd_milli": 65.0,
                    "sleep_hours": 7.5,
                    "sleep_performance": 82.0,
                    "resting_heart_rate": 56.0,
                },
                "calendar": {"total_meeting_minutes": 90},
                "slack": {"total_messages_sent": 12, "total_messages_received": 55},
            },
            "2026-03-14": {
                "date": "2026-03-14",
                "metrics_avg": {
                    "cognitive_load_score": 0.40,
                    "focus_depth_index": 0.68,
                    "social_drain_index": 0.20,
                    "context_switch_cost": 0.25,
                    "recovery_alignment_score": 0.78,
                },
                "focus_quality": {
                    "active_fdi": 0.65,
                    "active_windows": 10,
                    "peak_focus_hour": 11,
                    "peak_focus_fdi": 0.80,
                },
                "whoop": {
                    "recovery_score": 85.0,
                    "hrv_rmssd_milli": 72.0,
                    "sleep_hours": 8.0,
                    "sleep_performance": 88.0,
                    "resting_heart_rate": 54.0,
                },
                "calendar": {"total_meeting_minutes": 120},
                "slack": {"total_messages_sent": 18, "total_messages_received": 70},
            },
        },
        "total_days": 2,
        "last_updated": "2026-03-14T23:45:00",
    }

    def _patch_week(self, rolling=None):
        """Return context managers for mocking week data."""
        rolling = rolling or self._ROLLING_FULL
        windows = _make_day_windows(20)
        return (
            patch("scripts.report.list_available_dates",
                  return_value=["2026-03-13", "2026-03-14"]),
            patch("scripts.report.read_day", return_value=windows),
            patch("scripts.report._cdi_tier_short", return_value="balanced"),
            # weekly_summary uses its own store reads — patch both
            patch("engine.store.read_summary", return_value=rolling),
            patch("engine.store.read_day", return_value=windows),
        )

    # ── Tests ─────────────────────────────────────────────────────────────────

    def test_prints_something_with_data(self, capsys):
        from scripts.report import print_week
        patches = self._patch_week()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        assert len(out) > 100

    def test_output_contains_weekly_header(self, capsys):
        from scripts.report import print_week
        patches = self._patch_week()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        assert "Weekly Summary" in out

    def test_output_contains_date_range(self, capsys):
        from scripts.report import print_week
        patches = self._patch_week()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        # Should mention the week range (Mar 8 → Mar 14)
        assert "Mar 14" in out or "Mar  8" in out or "Mar 8" in out

    def test_output_contains_cognitive_metrics_section(self, capsys):
        from scripts.report import print_week
        patches = self._patch_week()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        assert "CLS" in out
        assert "FDI" in out
        assert "RAS" in out

    def test_output_contains_whoop_section(self, capsys):
        from scripts.report import print_week
        patches = self._patch_week()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        assert "WHOOP" in out
        assert "Recovery" in out

    def test_output_contains_best_worst_section(self, capsys):
        from scripts.report import print_week
        patches = self._patch_week()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        assert "Best" in out or "Worst" in out or "Lightest" in out

    def test_output_contains_activity_totals(self, capsys):
        from scripts.report import print_week
        patches = self._patch_week()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        assert "Meetings" in out or "Slack" in out

    def test_output_contains_data_coverage_section(self, capsys):
        from scripts.report import print_week
        patches = self._patch_week()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        assert "Coverage" in out or "WHOOP" in out

    def test_no_crash_when_no_data(self, capsys):
        """When no data exists for the week, print_week should not crash."""
        from scripts.report import print_week
        empty_rolling = {"days": {}, "total_days": 0, "last_updated": None}
        patches = (
            patch("scripts.report.list_available_dates", return_value=[]),
            patch("scripts.report.read_day", return_value=[]),
            patch("scripts.report._cdi_tier_short", return_value="—"),
            patch("engine.store.read_summary", return_value=empty_rolling),
            patch("engine.store.read_day", return_value=[]),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        # Should print something (header at minimum)
        assert len(out) > 0

    def test_no_crash_on_invalid_date(self, capsys):
        """An invalid date string should print an error without raising."""
        from scripts.report import print_week
        # Should not raise; should handle gracefully
        print_week("not-a-date")

    def test_no_crash_when_end_date_is_none(self, capsys):
        """Calling with None should default to today without crashing."""
        from scripts.report import print_week
        empty_rolling = {"days": {}, "total_days": 0, "last_updated": None}
        patches = (
            patch("scripts.report.list_available_dates", return_value=[]),
            patch("scripts.report.read_day", return_value=[]),
            patch("scripts.report._cdi_tier_short", return_value="—"),
            patch("engine.store.read_summary", return_value=empty_rolling),
            patch("engine.store.read_day", return_value=[]),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week(None)

    def test_cli_week_flag_triggers_week_mode(self, capsys):
        """--week flag should call print_week, not the per-day report."""
        import sys
        windows = _make_day_windows(20)
        rolling = self._ROLLING_FULL
        old_argv = sys.argv
        sys.argv = ["report.py", "--week"]
        try:
            from scripts.report import main as report_main
            patches = (
                patch("scripts.report.list_available_dates",
                      return_value=["2026-03-13", "2026-03-14"]),
                patch("scripts.report.read_day", return_value=windows),
                patch("scripts.report._cdi_tier_short", return_value="balanced"),
                patch("engine.store.read_summary", return_value=rolling),
                patch("engine.store.read_day", return_value=windows),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                try:
                    report_main()
                except SystemExit as e:
                    assert e.code == 0 or e.code is None
        finally:
            sys.argv = old_argv
        out = capsys.readouterr().out
        assert "Weekly Summary" in out

    def test_daily_breakdown_section_present(self, capsys):
        """Per-day breakdown table should appear when data exists."""
        from scripts.report import print_week
        patches = self._patch_week()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        assert "Daily Breakdown" in out or "2026-03-14" in out

    def test_output_is_multi_line(self, capsys):
        """Weekly output should span multiple lines (not a one-liner)."""
        from scripts.report import print_week
        patches = self._patch_week()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            print_week("2026-03-14")
        out = capsys.readouterr().out
        assert out.count("\n") >= 10

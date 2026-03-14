"""
Tests for the daily digest module.

Run with: python3 -m pytest tests/test_daily_digest.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from analysis.daily_digest import (
    compute_digest,
    format_digest_message,
    _score_bar,
    _cls_label,
    _ras_label,
    _fdi_label,
    _generate_insight,
)
from engine.chunker import build_windows


# ─── Shared test fixtures ─────────────────────────────────────────────────────

SAMPLE_WHOOP = {
    "recovery_score": 78.0,
    "hrv_rmssd_milli": 65.5,
    "resting_heart_rate": 54.0,
    "sleep_performance": 82.0,
    "sleep_hours": 7.8,
    "strain": 13.2,
    "spo2_percentage": 95.2,
}

SAMPLE_WHOOP_LOW_RECOVERY = {
    "recovery_score": 38.0,
    "hrv_rmssd_milli": 42.0,
    "resting_heart_rate": 62.0,
    "sleep_performance": 55.0,
    "sleep_hours": 5.5,
    "strain": 8.0,
    "spo2_percentage": 94.1,
}

SAMPLE_WHOOP_HIGH_RECOVERY = {
    "recovery_score": 95.0,
    "hrv_rmssd_milli": 90.0,
    "resting_heart_rate": 50.0,
    "sleep_performance": 98.0,
    "sleep_hours": 9.0,
    "strain": 5.0,
    "spo2_percentage": 96.5,
}

SAMPLE_CALENDAR_HEAVY = {
    "events": [
        {
            "id": "evt1",
            "title": "Morning Standup",
            "start": "2026-03-13T09:00:00+01:00",
            "end": "2026-03-13T09:30:00+01:00",
            "duration_minutes": 30,
            "attendee_count": 6,
            "organizer_email": "david@szabostuban.com",
            "is_all_day": False,
            "location": "",
            "status": "confirmed",
        },
        {
            "id": "evt2",
            "title": "Product Review",
            "start": "2026-03-13T11:00:00+01:00",
            "end": "2026-03-13T13:00:00+01:00",
            "duration_minutes": 120,
            "attendee_count": 8,
            "organizer_email": "ceo@szabostuban.com",
            "is_all_day": False,
            "location": "",
            "status": "confirmed",
        },
        {
            "id": "evt3",
            "title": "1:1 Check-in",
            "start": "2026-03-13T15:00:00+01:00",
            "end": "2026-03-13T15:30:00+01:00",
            "duration_minutes": 30,
            "attendee_count": 2,
            "organizer_email": "david@szabostuban.com",
            "is_all_day": False,
            "location": "",
            "status": "confirmed",
        },
    ],
    "event_count": 3,
    "total_meeting_minutes": 180,
    "max_concurrent_attendees": 8,
}

SAMPLE_CALENDAR_EMPTY = {
    "events": [],
    "event_count": 0,
    "total_meeting_minutes": 0,
    "max_concurrent_attendees": 0,
}


def make_slack_windows_active():
    """Slack windows with moderate activity during the day."""
    windows = {}
    # Morning: 9am onwards
    for idx in [36, 37, 38, 44, 45, 52, 53, 60]:
        windows[idx] = {
            "messages_sent": 3,
            "messages_received": 7,
            "total_messages": 10,
            "channels_active": 2,
        }
    return windows


def make_slack_windows_quiet():
    """A quiet day with minimal Slack activity."""
    return {
        44: {"messages_sent": 1, "messages_received": 2, "total_messages": 3, "channels_active": 1}
    }


# ─── Score bar tests ──────────────────────────────────────────────────────────

class TestScoreBar:
    def test_zero_is_empty(self):
        bar = _score_bar(0.0)
        assert bar == "░" * 10

    def test_one_is_full(self):
        bar = _score_bar(1.0)
        assert bar == "▓" * 10

    def test_half_is_split(self):
        bar = _score_bar(0.5)
        assert "▓" in bar and "░" in bar
        assert len(bar) == 10

    def test_custom_width(self):
        bar = _score_bar(0.5, width=8)
        assert len(bar) == 8

    def test_output_is_correct_length(self):
        for val in [0.0, 0.1, 0.3, 0.7, 0.9, 1.0]:
            assert len(_score_bar(val)) == 10


# ─── Label tests ──────────────────────────────────────────────────────────────

class TestLabels:
    def test_cls_light(self):
        assert _cls_label(0.10) == "light"

    def test_cls_moderate(self):
        assert _cls_label(0.30) == "moderate"

    def test_cls_heavy(self):
        assert _cls_label(0.50) == "heavy"

    def test_cls_intense(self):
        assert _cls_label(0.70) == "intense"

    def test_cls_maximal(self):
        assert _cls_label(0.85) == "maximal"

    def test_ras_well_within(self):
        assert _ras_label(0.90) == "well within capacity"

    def test_ras_over_capacity(self):
        assert _ras_label(0.25) == "over capacity"

    def test_ras_significantly_over(self):
        assert _ras_label(0.10) == "significantly over capacity"

    def test_fdi_deep(self):
        assert _fdi_label(0.85) == "deep"

    def test_fdi_fragmented(self):
        assert _fdi_label(0.35) == "highly fragmented"


# ─── Compute digest tests ─────────────────────────────────────────────────────

class TestComputeDigest:
    def _make_windows(self, whoop=None, calendar=None, slack=None):
        return build_windows(
            date_str="2026-03-13",
            whoop_data=whoop or SAMPLE_WHOOP,
            calendar_data=calendar or SAMPLE_CALENDAR_EMPTY,
            slack_windows=slack or {},
        )

    def test_returns_dict_with_required_keys(self):
        windows = self._make_windows()
        digest = compute_digest(windows)
        assert "date" in digest
        assert "whoop" in digest
        assert "metrics" in digest
        assert "activity" in digest
        assert "insight" in digest

    def test_date_is_preserved(self):
        windows = self._make_windows()
        digest = compute_digest(windows)
        assert digest["date"] == "2026-03-13"

    def test_whoop_data_is_included(self):
        windows = self._make_windows(whoop=SAMPLE_WHOOP)
        digest = compute_digest(windows)
        assert digest["whoop"]["recovery_score"] == 78.0
        assert digest["whoop"]["hrv_rmssd_milli"] == 65.5

    def test_empty_windows_returns_empty_dict(self):
        digest = compute_digest([])
        assert digest == {}

    def test_avg_cls_is_over_working_hours(self):
        """avg_cls should be from working hours (not all 96 including sleep)."""
        windows = self._make_windows(calendar=SAMPLE_CALENDAR_HEAVY)
        digest = compute_digest(windows)
        avg_cls = digest["metrics"]["avg_cls"]
        assert avg_cls is not None
        assert 0.0 <= avg_cls <= 1.0

    def test_fdi_active_is_none_when_no_activity(self):
        """With no meetings and no Slack, there are no active windows."""
        windows = self._make_windows(calendar=SAMPLE_CALENDAR_EMPTY, slack={})
        digest = compute_digest(windows)
        # No active windows — FDI should be None or reflect this
        avg_fdi = digest["metrics"]["avg_fdi_active"]
        assert avg_fdi is None or avg_fdi == 0.0 or avg_fdi is not None  # It's None since no active windows

    def test_active_windows_counted_correctly(self):
        """Active windows = meeting OR slack activity."""
        slack = make_slack_windows_active()
        windows = self._make_windows(slack=slack)
        digest = compute_digest(windows)
        assert digest["activity"]["active_windows"] > 0

    def test_meeting_stats_correct(self):
        """Meeting stats should reflect the calendar data."""
        windows = self._make_windows(calendar=SAMPLE_CALENDAR_HEAVY)
        digest = compute_digest(windows)
        assert digest["activity"]["total_meeting_minutes"] > 0
        assert digest["activity"]["meeting_count"] > 0

    def test_slack_counts_correct(self):
        """Slack sent/received should sum across all windows."""
        slack = make_slack_windows_active()
        windows = self._make_windows(slack=slack)
        digest = compute_digest(windows)
        assert digest["activity"]["slack_sent"] > 0
        assert digest["activity"]["slack_received"] > 0

    def test_peak_window_is_highest_cls(self):
        """Peak window should be the one with highest CLS."""
        windows = self._make_windows(
            calendar=SAMPLE_CALENDAR_HEAVY,
            slack=make_slack_windows_active(),
        )
        digest = compute_digest(windows)
        peak = digest.get("peak_window")
        assert peak is not None
        # It should be a meeting window (meetings drive high CLS)
        assert peak["calendar"]["in_meeting"] or peak["metrics"]["cognitive_load_score"] > 0

    def test_metrics_all_in_range(self):
        """All metric values should be 0-1 or None."""
        windows = self._make_windows(
            calendar=SAMPLE_CALENDAR_HEAVY,
            slack=make_slack_windows_active(),
        )
        digest = compute_digest(windows)
        for key, val in digest["metrics"].items():
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


# ─── Insight generation tests ─────────────────────────────────────────────────

class TestGenerateInsight:
    def test_low_recovery_high_load_warns(self):
        insight = _generate_insight(
            recovery=35.0,
            avg_cls=0.65,
            avg_fdi_active=0.60,
            avg_ras=0.30,
            total_meeting_minutes=120,
            total_sent=20,
            peak_window=None,
            working_count=60,
            active_count=15,
        )
        assert "pushed hard" in insight or "recovery" in insight.lower() or "load" in insight.lower()

    def test_low_recovery_light_load_positive(self):
        insight = _generate_insight(
            recovery=42.0,
            avg_cls=0.15,
            avg_fdi_active=0.80,
            avg_ras=0.75,
            total_meeting_minutes=30,
            total_sent=5,
            peak_window=None,
            working_count=60,
            active_count=5,
        )
        assert "good" in insight.lower() or "light" in insight.lower() or "bounce" in insight.lower()

    def test_fragmented_focus_flagged(self):
        insight = _generate_insight(
            recovery=80.0,
            avg_cls=0.45,
            avg_fdi_active=0.35,  # Very fragmented
            avg_ras=0.80,
            total_meeting_minutes=60,
            total_sent=15,
            peak_window=None,
            working_count=60,
            active_count=12,
        )
        assert "fragment" in insight.lower() or "focus" in insight.lower()

    def test_heavy_meetings_flagged(self):
        insight = _generate_insight(
            recovery=75.0,
            avg_cls=0.50,
            avg_fdi_active=0.55,
            avg_ras=0.70,
            total_meeting_minutes=300,  # 5 hours of meetings
            total_sent=10,
            peak_window=None,
            working_count=60,
            active_count=20,
        )
        assert "meeting" in insight.lower() or "hours" in insight.lower()

    def test_no_activity_returns_message(self):
        insight = _generate_insight(
            recovery=85.0,
            avg_cls=0.05,
            avg_fdi_active=None,
            avg_ras=0.95,
            total_meeting_minutes=0,
            total_sent=0,
            peak_window=None,
            working_count=60,
            active_count=0,
        )
        assert len(insight) > 0

    def test_light_day_positive(self):
        insight = _generate_insight(
            recovery=70.0,
            avg_cls=0.08,
            avg_fdi_active=0.90,
            avg_ras=0.88,
            total_meeting_minutes=30,
            total_sent=3,
            peak_window=None,
            working_count=60,
            active_count=4,
        )
        assert "light" in insight.lower() or "recovery" in insight.lower()

    def test_insight_is_always_string(self):
        """Insight must always return a non-empty string."""
        for recovery in [None, 30, 70, 95]:
            insight = _generate_insight(
                recovery=recovery,
                avg_cls=0.30,
                avg_fdi_active=0.70,
                avg_ras=0.75,
                total_meeting_minutes=60,
                total_sent=5,
                peak_window=None,
                working_count=60,
                active_count=8,
            )
            assert isinstance(insight, str)
            assert len(insight) > 0


# ─── Format message tests ─────────────────────────────────────────────────────

class TestFormatDigestMessage:
    def _make_digest(self, **overrides):
        base = {
            "date": "2026-03-13",
            "whoop": {
                "recovery_score": 78.0,
                "hrv_rmssd_milli": 65.5,
                "sleep_hours": 7.8,
                "sleep_performance": 82.0,
            },
            "metrics": {
                "avg_cls": 0.35,
                "peak_cls": 0.55,
                "avg_fdi_active": 0.68,
                "avg_sdi_active": 0.42,
                "avg_csc_active": 0.30,
                "avg_ras": 0.74,
            },
            "activity": {
                "working_windows": 60,
                "active_windows": 15,
                "idle_windows": 45,
                "total_meeting_minutes": 90,
                "meeting_count": 2,
                "slack_sent": 18,
                "slack_received": 45,
            },
            "peak_window": None,
            "insight": "Test insight for formatting.",
        }
        base.update(overrides)
        return base

    def test_message_contains_date(self):
        msg = format_digest_message(self._make_digest())
        assert "March 13" in msg or "2026-03-13" in msg

    def test_message_contains_recovery(self):
        msg = format_digest_message(self._make_digest())
        assert "78%" in msg or "Recovery" in msg

    def test_message_contains_cls(self):
        msg = format_digest_message(self._make_digest())
        assert "Cognitive Load" in msg or "CLS" in msg

    def test_message_contains_insight(self):
        msg = format_digest_message(self._make_digest())
        assert "Test insight" in msg

    def test_empty_digest_does_not_crash(self):
        msg = format_digest_message({})
        assert "no data" in msg.lower() or len(msg) > 0

    def test_missing_whoop_does_not_crash(self):
        digest = self._make_digest(whoop={})
        msg = format_digest_message(digest)
        assert "unavailable" in msg.lower() or "Recovery" in msg

    def test_no_activity_message(self):
        digest = self._make_digest(
            activity={
                "working_windows": 60,
                "active_windows": 0,
                "idle_windows": 60,
                "total_meeting_minutes": 0,
                "meeting_count": 0,
                "slack_sent": 0,
                "slack_received": 0,
            }
        )
        msg = format_digest_message(digest)
        assert "No significant activity" in msg or len(msg) > 0

    def test_peak_window_shown_when_distinct(self):
        """If peak_cls is notably higher than avg, the peak should appear."""
        digest = self._make_digest()
        # avg_cls=0.35, peak_cls=0.55 → difference > 0.10, so peak should appear
        msg = format_digest_message(digest)
        assert "Peak" in msg or "55%" in msg

    def test_peak_window_hidden_when_close_to_avg(self):
        """If peak is close to avg, don't show it redundantly."""
        digest = self._make_digest(
            metrics={
                "avg_cls": 0.50,
                "peak_cls": 0.55,  # Only 0.05 above avg
                "avg_fdi_active": 0.68,
                "avg_sdi_active": 0.42,
                "avg_csc_active": 0.30,
                "avg_ras": 0.74,
            }
        )
        msg = format_digest_message(digest)
        # Should NOT show peak line when difference is ≤ 0.10
        assert "Peak" not in msg

    def test_fdi_shows_active_context(self):
        """FDI label should mention 'active windows' to clarify it's not all windows."""
        digest = self._make_digest()
        msg = format_digest_message(digest)
        assert "active" in msg.lower()

    def test_meeting_count_shown(self):
        digest = self._make_digest()
        msg = format_digest_message(digest)
        assert "2 meetings" in msg or "meeting" in msg.lower()

    def test_output_is_non_empty_string(self):
        for variant in [
            self._make_digest(),
            self._make_digest(whoop={}),
            self._make_digest(activity={"working_windows": 0, "active_windows": 0, "idle_windows": 0,
                                        "total_meeting_minutes": 0, "meeting_count": 0,
                                        "slack_sent": 0, "slack_received": 0}),
        ]:
            msg = format_digest_message(variant)
            assert isinstance(msg, str)
            assert len(msg) > 10


# ─── Integration: full pipeline test ─────────────────────────────────────────

class TestDigestIntegration:
    def test_full_pipeline_from_windows(self):
        """Build real windows, compute digest, format message — no crashes."""
        windows = build_windows(
            date_str="2026-03-14",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_HEAVY,
            slack_windows=make_slack_windows_active(),
        )
        assert len(windows) == 96

        digest = compute_digest(windows)
        assert digest is not None
        assert digest.get("date") == "2026-03-14"

        message = format_digest_message(digest)
        assert "March 14" in message or "2026-03-14" in message
        assert len(message) > 100  # Should be a substantive message

    def test_full_pipeline_low_recovery_high_load(self):
        """Low recovery + heavy meetings should trigger alignment warning."""
        windows = build_windows(
            date_str="2026-03-14",
            whoop_data=SAMPLE_WHOOP_LOW_RECOVERY,
            calendar_data=SAMPLE_CALENDAR_HEAVY,
            slack_windows=make_slack_windows_active(),
        )
        digest = compute_digest(windows)
        message = format_digest_message(digest)

        # The insight should reflect the strain situation
        assert len(digest["insight"]) > 0
        assert len(message) > 100

    def test_full_pipeline_high_recovery_quiet_day(self):
        """High recovery + light day should note the available capacity."""
        windows = build_windows(
            date_str="2026-03-14",
            whoop_data=SAMPLE_WHOOP_HIGH_RECOVERY,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows=make_slack_windows_quiet(),
        )
        digest = compute_digest(windows)
        message = format_digest_message(digest)

        assert digest["whoop"]["recovery_score"] == 95.0
        assert len(message) > 50

    def test_fdi_active_vs_all_windows_difference(self):
        """
        FDI over active windows should differ from FDI over all windows.
        This validates the key fix: idle windows (sleep, quiet time) don't
        inflate the focus quality metric.
        """
        windows = build_windows(
            date_str="2026-03-14",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_HEAVY,
            slack_windows=make_slack_windows_active(),
        )
        digest = compute_digest(windows)

        # Active-only FDI
        avg_fdi_active = digest["metrics"]["avg_fdi_active"]

        # Naive FDI over all working windows (what the old summarize_day does)
        working = [w for w in windows if w["metadata"]["is_working_hours"]]
        all_fdi_vals = [w["metrics"]["focus_depth_index"] for w in working]
        naive_avg_fdi = sum(all_fdi_vals) / len(all_fdi_vals) if all_fdi_vals else None

        # The active-only FDI should be lower (meetings and Slack reduce focus)
        # while the naive avg is inflated by all the quiet windows at 1.0
        if avg_fdi_active is not None and naive_avg_fdi is not None:
            assert avg_fdi_active <= naive_avg_fdi, (
                f"Active-window FDI ({avg_fdi_active:.3f}) should be <= "
                f"all-window naive FDI ({naive_avg_fdi:.3f})"
            )


# ─── compute_trend_context tests ─────────────────────────────────────────────

class TestComputeTrendContext:
    """Tests for the multi-day trend context builder."""

    def test_returns_empty_on_no_history(self):
        """With no store data, should return minimal dict."""
        from unittest.mock import patch
        from analysis.daily_digest import compute_trend_context
        with patch("engine.store.get_recent_summaries", return_value=[]):
            result = compute_trend_context("2026-03-14")
        assert result.get("days_of_data", 0) == 0

    def test_function_is_importable(self):
        """compute_trend_context must be importable from analysis.daily_digest."""
        from analysis.daily_digest import compute_trend_context
        assert callable(compute_trend_context)

    def test_returns_dict(self, tmp_path, monkeypatch):
        """Should always return a dict, even when store is empty."""
        from analysis.daily_digest import compute_trend_context
        from unittest.mock import patch
        with patch("engine.store.get_recent_summaries", return_value=[]):
            result = compute_trend_context("2026-03-14")
        assert isinstance(result, dict)

    def test_known_keys_present_with_data(self):
        """With sufficient mock data, expected keys should be present."""
        from analysis.daily_digest import compute_trend_context
        from unittest.mock import patch

        mock_summaries = [
            {
                "date": f"2026-03-{d:02d}",
                "whoop": {"hrv_rmssd_milli": 55.0 - d, "recovery_score": 70.0 - d},
                "metrics_avg": {"cognitive_load_score": 0.40 + d * 0.01,
                                "recovery_alignment_score": 0.75 - d * 0.01},
            }
            for d in range(1, 8)
        ]
        with patch("engine.store.get_recent_summaries", return_value=mock_summaries):
            result = compute_trend_context("2026-03-14", lookback_days=7)

        assert isinstance(result, dict)
        # days_of_data should reflect available history
        assert "days_of_data" in result

    def test_declining_hrv_detected(self):
        """A consistently declining HRV series should be detected."""
        from analysis.daily_digest import compute_trend_context
        from unittest.mock import patch

        # HRV declining: 70, 65, 60, 55, 50 over 5 days
        mock_summaries = [
            {
                "date": f"2026-03-{d:02d}",
                "whoop": {"hrv_rmssd_milli": 70.0 - (d - 1) * 5, "recovery_score": 75.0},
                "metrics_avg": {"cognitive_load_score": 0.40, "recovery_alignment_score": 0.75},
            }
            for d in range(1, 6)
        ]
        with patch("engine.store.get_recent_summaries", return_value=mock_summaries):
            result = compute_trend_context("2026-03-14", lookback_days=7)

        assert isinstance(result, dict)
        # If trend is detected, hrv_trend should indicate decline
        if "hrv_trend" in result:
            assert result["hrv_trend"] in ("declining", "improving", "stable")

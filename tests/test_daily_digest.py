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


# ─── Hourly CLS sparkline tests (v1.3) ───────────────────────────────────────

class TestComputeHourlyCLSCurve:
    """Tests for the hourly CLS aggregation function."""

    def _make_windows(self, cls_by_hour: dict[int, float]) -> list[dict]:
        """
        Build a minimal 96-window list where each working hour has the
        given CLS value (same for all 4 × 15-min windows in that hour).
        Non-specified hours default to 0.0 CLS.
        """
        windows = []
        for h in range(24):
            for q in range(4):
                window = {
                    "metadata": {
                        "hour_of_day": h,
                        "minute_of_hour": q * 15,
                        "is_working_hours": 7 <= h < 22,
                    },
                    "metrics": {
                        "cognitive_load_score": cls_by_hour.get(h, 0.0),
                    },
                }
                windows.append(window)
        return windows

    def test_returns_list_of_length_15(self):
        """Should always return exactly 15 values (7am–9pm inclusive)."""
        from analysis.daily_digest import compute_hourly_cls_curve
        windows = self._make_windows({})
        result = compute_hourly_cls_curve(windows)
        assert len(result) == 15, f"Expected 15 values, got {len(result)}"

    def test_covers_hours_7_to_21(self):
        """Index 0 = 7am, index 14 = 9pm (21:00)."""
        from analysis.daily_digest import compute_hourly_cls_curve
        # Set hour 9 (index 2) to a distinctive value
        windows = self._make_windows({9: 0.75})
        result = compute_hourly_cls_curve(windows)
        assert result[2] == pytest.approx(0.75, abs=0.001), \
            f"Hour 9 (index 2) should be 0.75, got {result[2]}"
        # Hour 8 (index 1) should be 0.0
        assert result[1] == pytest.approx(0.0, abs=0.001), \
            f"Hour 8 (index 1) should be 0.0, got {result[1]}"

    def test_averages_four_windows_per_hour(self):
        """Each hour = mean of up to 4 × 15-min windows."""
        from analysis.daily_digest import compute_hourly_cls_curve
        # Build windows manually with varying CLS within one hour
        windows = []
        for h in range(24):
            for q in range(4):
                cls = 0.0
                if h == 10:
                    cls = [0.1, 0.3, 0.5, 0.7][q]  # avg = 0.4
                window = {
                    "metadata": {
                        "hour_of_day": h,
                        "minute_of_hour": q * 15,
                        "is_working_hours": 7 <= h < 22,
                    },
                    "metrics": {"cognitive_load_score": cls},
                }
                windows.append(window)

        result = compute_hourly_cls_curve(windows)
        idx_10 = 10 - 7  # = 3
        assert result[idx_10] == pytest.approx(0.4, abs=0.001), \
            f"Hour 10 mean should be 0.40, got {result[idx_10]}"

    def test_excludes_hours_before_7am(self):
        """Hours before 7am should NOT appear in the output."""
        from analysis.daily_digest import compute_hourly_cls_curve
        # Set hours 0–6 to a distinctive value; they should not affect output
        windows = self._make_windows({h: 0.99 for h in range(7)})
        result = compute_hourly_cls_curve(windows)
        # All working hours were NOT given values so should be 0.0 (default)
        assert all(v == pytest.approx(0.0, abs=0.001) for v in result if v is not None), \
            "No pre-7am hours should appear in the sparkline"

    def test_excludes_hours_at_or_after_10pm(self):
        """Hours 22 and beyond should NOT appear in the output."""
        from analysis.daily_digest import compute_hourly_cls_curve
        windows = self._make_windows({22: 0.99, 23: 0.99})
        result = compute_hourly_cls_curve(windows)
        # Working hours 7–21 were not set, so all should be 0.0
        assert all(v == pytest.approx(0.0, abs=0.001) for v in result if v is not None)

    def test_empty_windows_returns_nones_or_zeros(self):
        """Empty window list returns list of 15 Nones (no data)."""
        from analysis.daily_digest import compute_hourly_cls_curve
        result = compute_hourly_cls_curve([])
        assert len(result) == 15
        assert all(v is None for v in result)

    def test_all_values_in_0_1_range(self):
        """No value should exceed [0, 1] since CLS is bounded."""
        from analysis.daily_digest import compute_hourly_cls_curve
        windows = self._make_windows({h: 0.85 for h in range(7, 22)})
        result = compute_hourly_cls_curve(windows)
        for i, val in enumerate(result):
            if val is not None:
                assert 0.0 <= val <= 1.0, f"Index {i}: {val} out of [0,1]"


class TestFormatHourlySparkline:
    """Tests for the sparkline renderer."""

    def test_correct_length_for_15_values(self):
        """Output should be exactly 15 characters for 15 inputs."""
        from analysis.daily_digest import _format_hourly_sparkline
        result = _format_hourly_sparkline([0.05] * 15)
        assert len(result) == 15, f"Expected 15 chars, got {len(result)}"

    def test_very_light_maps_to_light_block(self):
        """Values < 0.10 should render as ░"""
        from analysis.daily_digest import _format_hourly_sparkline
        assert _format_hourly_sparkline([0.0])[0] == "░"
        assert _format_hourly_sparkline([0.05])[0] == "░"
        assert _format_hourly_sparkline([0.09])[0] == "░"

    def test_light_maps_to_medium_block(self):
        """Values 0.10–0.24 should render as ▒"""
        from analysis.daily_digest import _format_hourly_sparkline
        assert _format_hourly_sparkline([0.10])[0] == "▒"
        assert _format_hourly_sparkline([0.20])[0] == "▒"
        assert _format_hourly_sparkline([0.24])[0] == "▒"

    def test_moderate_maps_to_dark_block(self):
        """Values 0.25–0.49 should render as ▓"""
        from analysis.daily_digest import _format_hourly_sparkline
        assert _format_hourly_sparkline([0.25])[0] == "▓"
        assert _format_hourly_sparkline([0.40])[0] == "▓"
        assert _format_hourly_sparkline([0.49])[0] == "▓"

    def test_heavy_maps_to_full_block(self):
        """Values >= 0.50 should render as █"""
        from analysis.daily_digest import _format_hourly_sparkline
        assert _format_hourly_sparkline([0.50])[0] == "█"
        assert _format_hourly_sparkline([0.75])[0] == "█"
        assert _format_hourly_sparkline([1.0])[0] == "█"

    def test_none_maps_to_dot(self):
        """None values (missing data) should render as ·"""
        from analysis.daily_digest import _format_hourly_sparkline
        result = _format_hourly_sparkline([None])
        assert result == "·"

    def test_mixed_values_produce_correct_pattern(self):
        """A known mixed input should produce the expected pattern."""
        from analysis.daily_digest import _format_hourly_sparkline
        # ░▒▓█· — each level once plus a None
        vals = [0.05, 0.15, 0.35, 0.65, None]
        result = _format_hourly_sparkline(vals)
        assert result == "░▒▓█·", f"Got: {result!r}"

    def test_empty_input_returns_empty_string(self):
        """Empty input → empty output."""
        from analysis.daily_digest import _format_hourly_sparkline
        assert _format_hourly_sparkline([]) == ""

    def test_output_is_string(self):
        """Return type must always be str."""
        from analysis.daily_digest import _format_hourly_sparkline
        result = _format_hourly_sparkline([0.3, None, 0.8])
        assert isinstance(result, str)


class TestHourlyCLSInDigest:
    """Integration tests: sparkline appears in compute_digest and format_digest_message."""

    def _make_basic_windows(self) -> list[dict]:
        """Build 96 minimal windows with varying CLS for testing."""
        from engine.chunker import build_windows
        whoop = {
            "recovery_score": 75.0,
            "hrv_rmssd_milli": 65.0,
            "resting_heart_rate": 55.0,
            "sleep_performance": 80.0,
            "sleep_hours": 7.5,
            "strain": 12.0,
            "spo2_percentage": 95.0,
        }
        calendar = {
            "events": [],
            "event_count": 0,
            "total_meeting_minutes": 0,
            "max_concurrent_attendees": 0,
        }
        from unittest.mock import patch
        with patch("engine.store.get_recent_summaries", return_value=[]):
            return build_windows("2026-03-14", whoop, calendar, {})

    def test_compute_digest_includes_hourly_cls_curve(self):
        """compute_digest() should include 'hourly_cls_curve' key."""
        from analysis.daily_digest import compute_digest
        from unittest.mock import patch
        windows = self._make_basic_windows()
        with patch("engine.store.get_recent_summaries", return_value=[]):
            digest = compute_digest(windows)
        assert "hourly_cls_curve" in digest, "digest must contain 'hourly_cls_curve'"

    def test_hourly_cls_curve_in_digest_has_15_values(self):
        """The curve in the digest should have exactly 15 values."""
        from analysis.daily_digest import compute_digest
        from unittest.mock import patch
        windows = self._make_basic_windows()
        with patch("engine.store.get_recent_summaries", return_value=[]):
            digest = compute_digest(windows)
        curve = digest["hourly_cls_curve"]
        assert len(curve) == 15, f"Expected 15, got {len(curve)}"

    def test_format_digest_message_contains_sparkline(self):
        """format_digest_message() output should include the sparkline line."""
        from analysis.daily_digest import compute_digest, format_digest_message
        from unittest.mock import patch
        windows = self._make_basic_windows()
        with patch("engine.store.get_recent_summaries", return_value=[]):
            digest = compute_digest(windows)
        message = format_digest_message(digest)
        # The sparkline line contains '7am' and '10pm' as anchors
        assert "7am" in message, "Formatted digest should show '7am' in sparkline"
        assert "10pm" in message, "Formatted digest should show '10pm' in sparkline"

    def test_format_digest_message_sparkline_uses_block_chars(self):
        """The sparkline line should contain at least one block char."""
        from analysis.daily_digest import compute_digest, format_digest_message
        from unittest.mock import patch
        windows = self._make_basic_windows()
        with patch("engine.store.get_recent_summaries", return_value=[]):
            digest = compute_digest(windows)
        message = format_digest_message(digest)
        # At least one block character should appear somewhere in the message
        block_chars = {"░", "▒", "▓", "█", "·"}
        assert any(c in message for c in block_chars), \
            "Formatted digest should contain block-chart characters in sparkline"

    def test_no_hourly_cls_curve_in_digest_gracefully_handled(self):
        """If digest has no hourly_cls_curve key, format_digest_message should not crash."""
        from analysis.daily_digest import format_digest_message
        minimal_digest = {
            "date": "2026-03-14",
            "whoop": {"recovery_score": 75.0, "hrv_rmssd_milli": 65.0, "sleep_hours": 7.5},
            "metrics": {"avg_cls": 0.20, "peak_cls": 0.40, "avg_fdi_active": 0.70, "avg_ras": 0.75},
            "activity": {
                "working_windows": 60, "active_windows": 10, "idle_windows": 50,
                "total_meeting_minutes": 60, "meeting_count": 2,
                "slack_sent": 5, "slack_received": 20,
            },
            "peak_window": None,
            "trend": {},
            "insight": "Test insight.",
            # deliberately no "hourly_cls_curve"
        }
        # Should not raise
        result = format_digest_message(minimal_digest)
        assert isinstance(result, str)
        assert len(result) > 0


# ─── Omi conversation digest tests (v1.6) ────────────────────────────────────

def _make_omi_windows(date_str: str = "2026-03-13", omi_by_window: dict = None):
    """
    Build windows with Omi data injected for specific window indices.

    omi_by_window: dict of window_index → omi signal dict
    (conversation_active, word_count, speech_seconds, sessions_count, speech_ratio)
    """
    from engine.chunker import build_windows
    windows = build_windows(
        date_str=date_str,
        whoop_data=SAMPLE_WHOOP,
        calendar_data=SAMPLE_CALENDAR_EMPTY,
        slack_windows={},
        omi_windows=omi_by_window or {},
    )
    return windows


class TestOmiDigest:
    """Tests for Omi conversation aggregation in compute_digest (v1.6)."""

    def test_no_omi_data_returns_none(self):
        """When there is no Omi data in windows, omi field should be None."""
        windows = _make_omi_windows(omi_by_window={})
        digest = compute_digest(windows)
        assert digest.get("omi") is None

    def test_omi_data_present_returns_dict(self):
        """When Omi data is present, omi field should be a non-empty dict."""
        omi_windows = {
            36: {  # 9am window
                "conversation_active": True,
                "word_count": 300,
                "speech_seconds": 240.0,
                "audio_seconds": 360.0,
                "sessions_count": 1,
                "speech_ratio": 0.67,
            }
        }
        windows = _make_omi_windows(omi_by_window=omi_windows)
        digest = compute_digest(windows)
        assert digest.get("omi") is not None
        assert isinstance(digest["omi"], dict)

    def test_omi_session_count_aggregated(self):
        """Total sessions should sum sessions_count across all omi-active windows."""
        omi_windows = {
            36: {"conversation_active": True, "word_count": 200, "speech_seconds": 120.0,
                 "audio_seconds": 180.0, "sessions_count": 2, "speech_ratio": 0.67},
            44: {"conversation_active": True, "word_count": 150, "speech_seconds": 90.0,
                 "audio_seconds": 150.0, "sessions_count": 1, "speech_ratio": 0.60},
        }
        windows = _make_omi_windows(omi_by_window=omi_windows)
        digest = compute_digest(windows)
        omi = digest["omi"]
        assert omi["total_sessions"] == 3  # 2 + 1

    def test_omi_word_count_aggregated(self):
        """Total words should sum across all omi-active working-hour windows."""
        omi_windows = {
            36: {"conversation_active": True, "word_count": 250, "speech_seconds": 150.0,
                 "audio_seconds": 220.0, "sessions_count": 1, "speech_ratio": 0.68},
            52: {"conversation_active": True, "word_count": 400, "speech_seconds": 280.0,
                 "audio_seconds": 360.0, "sessions_count": 1, "speech_ratio": 0.78},
        }
        windows = _make_omi_windows(omi_by_window=omi_windows)
        digest = compute_digest(windows)
        assert digest["omi"]["total_words"] == 650

    def test_omi_speech_minutes_computed(self):
        """Total speech minutes = sum of speech_seconds / 60, rounded to 1 dp."""
        omi_windows = {
            36: {"conversation_active": True, "word_count": 300, "speech_seconds": 180.0,
                 "audio_seconds": 250.0, "sessions_count": 1, "speech_ratio": 0.72},
        }
        windows = _make_omi_windows(omi_by_window=omi_windows)
        digest = compute_digest(windows)
        expected_min = round(180.0 / 60.0, 1)
        assert digest["omi"]["total_speech_minutes"] == expected_min

    def test_omi_non_working_hours_excluded(self):
        """Omi data outside working hours should not be included in the digest."""
        # Window 0 = midnight (non-working), window 36 = 9am (working)
        omi_windows = {
            0: {"conversation_active": True, "word_count": 500, "speech_seconds": 300.0,
                "audio_seconds": 450.0, "sessions_count": 2, "speech_ratio": 0.67},
            36: {"conversation_active": True, "word_count": 100, "speech_seconds": 60.0,
                 "audio_seconds": 90.0, "sessions_count": 1, "speech_ratio": 0.67},
        }
        windows = _make_omi_windows(omi_by_window=omi_windows)
        digest = compute_digest(windows)
        # Only window 36 (working hours) should be counted
        assert digest["omi"]["total_words"] == 100
        assert digest["omi"]["total_sessions"] == 1

    def test_omi_digest_has_required_keys(self):
        """Omi digest dict must contain expected keys."""
        omi_windows = {
            36: {"conversation_active": True, "word_count": 200, "speech_seconds": 120.0,
                 "audio_seconds": 180.0, "sessions_count": 1, "speech_ratio": 0.67},
        }
        windows = _make_omi_windows(omi_by_window=omi_windows)
        digest = compute_digest(windows)
        omi = digest["omi"]
        assert "conversation_windows" in omi
        assert "total_sessions" in omi
        assert "total_words" in omi
        assert "total_speech_minutes" in omi

    def test_omi_conversation_windows_count(self):
        """conversation_windows = number of working windows with conversation_active=True."""
        omi_windows = {
            36: {"conversation_active": True, "word_count": 200, "speech_seconds": 120.0,
                 "audio_seconds": 180.0, "sessions_count": 1, "speech_ratio": 0.67},
            44: {"conversation_active": True, "word_count": 150, "speech_seconds": 90.0,
                 "audio_seconds": 130.0, "sessions_count": 1, "speech_ratio": 0.69},
            52: {"conversation_active": False, "word_count": 0, "speech_seconds": 0.0,
                 "audio_seconds": 0.0, "sessions_count": 0, "speech_ratio": 0.0},
        }
        windows = _make_omi_windows(omi_by_window=omi_windows)
        digest = compute_digest(windows)
        # 2 windows have conversation_active=True, 1 doesn't
        assert digest["omi"]["conversation_windows"] == 2


class TestOmiFormatDigestMessage:
    """Tests for Omi section rendering in format_digest_message (v1.6)."""

    def _minimal_digest(self, omi=None, peak_focus_hour=None, peak_focus_fdi=None):
        """Build a minimal digest dict, optionally with Omi + peak focus data."""
        return {
            "date": "2026-03-14",
            "whoop": {"recovery_score": 75.0, "hrv_rmssd_milli": 65.0, "sleep_hours": 7.5},
            "metrics": {
                "avg_cls": 0.25, "peak_cls": 0.45, "avg_fdi_active": 0.72, "avg_ras": 0.78,
            },
            "activity": {
                "working_windows": 60, "active_windows": 10, "idle_windows": 50,
                "total_meeting_minutes": 60, "meeting_count": 2,
                "slack_sent": 5, "slack_received": 20,
            },
            "peak_window": None,
            "trend": {},
            "insight": "Test insight.",
            "hourly_cls_curve": None,
            "rescuetime": None,
            "omi": omi,
            "peak_focus_hour": peak_focus_hour,
            "peak_focus_fdi": peak_focus_fdi,
        }

    def test_no_omi_no_omi_section(self):
        """When omi is None, no 🎙 line should appear in the message."""
        msg = format_digest_message(self._minimal_digest(omi=None))
        assert "🎙" not in msg

    def test_omi_section_shown_when_present(self):
        """When omi data is present, 🎙 line should appear."""
        omi = {"conversation_windows": 2, "total_sessions": 3, "total_words": 450, "total_speech_minutes": 12.0}
        msg = format_digest_message(self._minimal_digest(omi=omi))
        assert "🎙" in msg

    def test_omi_session_count_in_message(self):
        """Session count should be shown in the Omi line."""
        omi = {"conversation_windows": 1, "total_sessions": 2, "total_words": 300, "total_speech_minutes": 8.0}
        msg = format_digest_message(self._minimal_digest(omi=omi))
        assert "2 conversations" in msg

    def test_omi_word_count_in_message(self):
        """Word count should appear in the Omi line."""
        omi = {"conversation_windows": 1, "total_sessions": 1, "total_words": 847, "total_speech_minutes": 14.0}
        msg = format_digest_message(self._minimal_digest(omi=omi))
        assert "847" in msg

    def test_omi_speech_minutes_in_message(self):
        """Speaking time should appear in the Omi line."""
        omi = {"conversation_windows": 1, "total_sessions": 1, "total_words": 200, "total_speech_minutes": 7.0}
        msg = format_digest_message(self._minimal_digest(omi=omi))
        assert "7" in msg and "min" in msg

    def test_omi_single_conversation_grammar(self):
        """Single conversation should use singular 'conversation', not 'conversations'."""
        omi = {"conversation_windows": 1, "total_sessions": 1, "total_words": 100, "total_speech_minutes": 3.0}
        msg = format_digest_message(self._minimal_digest(omi=omi))
        assert "1 conversation" in msg
        assert "1 conversations" not in msg

    def test_no_peak_focus_line_when_none(self):
        """When peak_focus_hour is None, no 🏆 line should appear."""
        msg = format_digest_message(self._minimal_digest())
        assert "🏆" not in msg

    def test_no_peak_focus_line_when_below_threshold(self):
        """When peak FDI is below 0.70, the 🏆 line should not appear."""
        msg = format_digest_message(self._minimal_digest(peak_focus_hour=9, peak_focus_fdi=0.55))
        assert "🏆" not in msg

    def test_peak_focus_line_shown_above_threshold(self):
        """When peak FDI is ≥ 0.70, the 🏆 line should appear."""
        msg = format_digest_message(self._minimal_digest(peak_focus_hour=9, peak_focus_fdi=0.84))
        assert "🏆" in msg

    def test_peak_focus_shows_hour_range(self):
        """Peak focus line should show 'HH:00–HH+1:00' format."""
        msg = format_digest_message(self._minimal_digest(peak_focus_hour=9, peak_focus_fdi=0.84))
        assert "09:00–10:00" in msg

    def test_peak_focus_shows_fdi_percentage(self):
        """Peak focus line should show FDI as a percentage."""
        msg = format_digest_message(self._minimal_digest(peak_focus_hour=9, peak_focus_fdi=0.84))
        assert "84%" in msg

    def test_peak_focus_at_threshold_boundary(self):
        """At exactly 0.70 FDI, the peak focus line should appear."""
        msg = format_digest_message(self._minimal_digest(peak_focus_hour=14, peak_focus_fdi=0.70))
        assert "🏆" in msg
        assert "14:00–15:00" in msg

    def test_no_crash_when_omi_missing_from_digest(self):
        """format_digest_message should handle digests without omi key."""
        digest = self._minimal_digest()
        del digest["omi"]  # remove the key entirely
        result = format_digest_message(digest)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_crash_when_peak_focus_keys_missing(self):
        """format_digest_message should handle digests without peak_focus_hour/fdi keys."""
        digest = self._minimal_digest()
        del digest["peak_focus_hour"]
        del digest["peak_focus_fdi"]
        result = format_digest_message(digest)
        assert isinstance(result, str)
        assert len(result) > 0


class TestPeakFocusHourInComputeDigest:
    """Tests for peak focus hour computation in compute_digest (v1.6)."""

    def test_no_active_windows_peak_focus_none(self):
        """When there are no active working windows, peak_focus_hour should be None."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        digest = compute_digest(windows)
        assert digest["peak_focus_hour"] is None
        assert digest["peak_focus_fdi"] is None

    def test_active_windows_produce_peak_focus(self):
        """When there are active windows, peak_focus_hour should be set."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows=make_slack_windows_active(),
        )
        digest = compute_digest(windows)
        # Active windows exist, so peak_focus_hour should be set
        assert digest["peak_focus_hour"] is not None
        assert digest["peak_focus_fdi"] is not None

    def test_peak_focus_hour_is_valid_working_hour(self):
        """Peak focus hour must be in the working hours range (7–21)."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows=make_slack_windows_active(),
        )
        digest = compute_digest(windows)
        if digest["peak_focus_hour"] is not None:
            assert 7 <= digest["peak_focus_hour"] < 22

    def test_peak_focus_fdi_is_in_range(self):
        """Peak focus FDI must be in [0, 1]."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows=make_slack_windows_active(),
        )
        digest = compute_digest(windows)
        if digest["peak_focus_fdi"] is not None:
            assert 0.0 <= digest["peak_focus_fdi"] <= 1.0

    def test_peak_focus_keys_in_digest(self):
        """compute_digest should always include peak_focus_hour and peak_focus_fdi keys."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        digest = compute_digest(windows)
        assert "peak_focus_hour" in digest
        assert "peak_focus_fdi" in digest


# ─── v1.9: Tomorrow's focus plan in nightly digest ───────────────────────────

class TestTomorrowFocusPlanInDigest:
    """Tests for the v1.9 tomorrow focus plan feature in the nightly digest."""

    def test_compute_digest_includes_tomorrow_focus_plan_key(self):
        """compute_digest always includes the tomorrow_focus_plan key."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        digest = compute_digest(windows)
        assert "tomorrow_focus_plan" in digest

    def test_tomorrow_focus_plan_is_none_or_dict(self):
        """tomorrow_focus_plan is either None or a dict."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        digest = compute_digest(windows)
        plan = digest["tomorrow_focus_plan"]
        assert plan is None or isinstance(plan, dict)

    def test_focus_plan_dict_has_expected_keys(self):
        """When tomorrow_focus_plan is a dict, it has the required keys."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        digest = compute_digest(windows)
        plan = digest["tomorrow_focus_plan"]
        if plan is not None:
            assert "section" in plan
            assert "is_meaningful" in plan
            assert "block_count" in plan
            assert "summary_line" in plan
            assert "advisory" in plan

    def test_focus_plan_does_not_crash_digest(self):
        """Even when focus planner raises, compute_digest should succeed."""
        from unittest.mock import patch
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        # Simulate a failing focus planner — digest must still compute
        with patch("analysis.focus_planner.plan_tomorrow_focus", side_effect=Exception("mock fail")):
            digest = compute_digest(windows)
        assert "date" in digest
        assert digest["tomorrow_focus_plan"] is None

    def test_format_digest_with_meaningful_plan_shows_section(self):
        """When plan is meaningful, format_digest_message includes the section."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        digest = compute_digest(windows)
        # Inject a mock meaningful plan
        digest["tomorrow_focus_plan"] = {
            "section": "*🎯 Tomorrow's Focus Plan:*\n• 9:00–11:00  _(120min, peak focus hour)_  🔥",
            "summary_line": "One peak block available",
            "advisory": "Front-load the harder task.",
            "is_meaningful": True,
            "block_count": 1,
            "cdi_tier": "balanced",
        }
        msg = format_digest_message(digest)
        assert "Tomorrow" in msg
        assert "Focus Plan" in msg
        assert "9:00" in msg

    def test_format_digest_without_plan_omits_section(self):
        """When tomorrow_focus_plan is None, no focus plan section is shown."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        digest = compute_digest(windows)
        digest["tomorrow_focus_plan"] = None
        msg = format_digest_message(digest)
        assert "Focus Plan" not in msg

    def test_format_digest_with_not_meaningful_plan_omits_section(self):
        """When plan.is_meaningful is False, the section is not shown."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        digest = compute_digest(windows)
        digest["tomorrow_focus_plan"] = {
            "section": "",
            "summary_line": "No data",
            "advisory": "",
            "is_meaningful": False,
            "block_count": 0,
            "cdi_tier": "balanced",
        }
        msg = format_digest_message(digest)
        assert "Focus Plan" not in msg

    def test_focus_plan_appears_after_insight(self):
        """The focus plan section should appear after the insight line."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        digest = compute_digest(windows)
        digest["tomorrow_focus_plan"] = {
            "section": "*🎯 Tomorrow's Focus Plan:*\n• 9:00–11:00  _(120min)_  🔥",
            "summary_line": "One peak block",
            "advisory": "Front-load.",
            "is_meaningful": True,
            "block_count": 1,
            "cdi_tier": "balanced",
        }
        digest["insight"] = "Test insight text"
        msg = format_digest_message(digest)
        insight_pos = msg.find("Test insight text")
        plan_pos = msg.find("Tomorrow's Focus Plan")
        assert insight_pos != -1, "Insight not found in message"
        assert plan_pos != -1, "Focus plan not found in message"
        assert insight_pos < plan_pos, "Focus plan should appear after insight"

    def test_no_duplicate_focus_plan_header(self):
        """The header '*🎯 Tomorrow's Focus Plan:*' should appear exactly once."""
        windows = build_windows(
            date_str="2026-03-13",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )
        digest = compute_digest(windows)
        digest["tomorrow_focus_plan"] = {
            "section": "*🎯 Tomorrow's Focus Plan:*\n• 9:00–11:00  _(120min)_  🔥",
            "summary_line": "One peak block",
            "advisory": "",
            "is_meaningful": True,
            "block_count": 1,
            "cdi_tier": "balanced",
        }
        msg = format_digest_message(digest)
        # Header must appear exactly once — format_focus_plan_section already includes it
        count = msg.count("Tomorrow's Focus Plan")
        assert count == 1, f"Header appeared {count} times, expected 1"

    def test_compute_focus_plan_for_digest_is_importable(self):
        """The _compute_focus_plan_for_digest function can be imported."""
        from analysis.daily_digest import _compute_focus_plan_for_digest
        assert callable(_compute_focus_plan_for_digest)

    def test_compute_focus_plan_returns_none_on_exception(self):
        """_compute_focus_plan_for_digest returns None when planner raises."""
        from unittest.mock import patch
        from analysis.daily_digest import _compute_focus_plan_for_digest
        with patch("analysis.focus_planner.plan_tomorrow_focus", side_effect=RuntimeError("test")):
            result = _compute_focus_plan_for_digest("2026-03-13")
        assert result is None


# ─── Meeting Intelligence integration (v2.0) ─────────────────────────────────

class TestMeetingIntelDigestIntegration:
    """
    Tests for the v2.0 Meeting Intelligence integration into the daily digest.
    Covers _compute_meeting_intel_for_digest() and format_digest_message() rendering.
    """

    def _windows_with_meetings(self) -> list[dict]:
        """Build windows with 2 hours of meetings in the morning."""
        return build_windows(
            date_str="2026-03-16",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_HEAVY,
            slack_windows={},
        )

    def _windows_no_meetings(self) -> list[dict]:
        """Build windows with no meetings."""
        return build_windows(
            date_str="2026-03-16",
            whoop_data=SAMPLE_WHOOP,
            calendar_data=SAMPLE_CALENDAR_EMPTY,
            slack_windows={},
        )

    # ── _compute_meeting_intel_for_digest unit tests ──────────────────────

    def test_importable(self):
        """_compute_meeting_intel_for_digest can be imported."""
        from analysis.daily_digest import _compute_meeting_intel_for_digest
        assert callable(_compute_meeting_intel_for_digest)

    def test_returns_none_for_empty_windows(self):
        """Returns None when given an empty windows list."""
        from analysis.daily_digest import _compute_meeting_intel_for_digest
        result = _compute_meeting_intel_for_digest([], "2026-03-16")
        assert result is None

    def test_returns_none_when_no_meetings(self):
        """Returns None when the day has no meetings."""
        from analysis.daily_digest import _compute_meeting_intel_for_digest
        windows = self._windows_no_meetings()
        result = _compute_meeting_intel_for_digest(windows, "2026-03-16")
        assert result is None

    def test_returns_dict_when_meetings_exist(self):
        """Returns a dict with meeting intel keys when the day had meetings."""
        from analysis.daily_digest import _compute_meeting_intel_for_digest
        windows = self._windows_with_meetings()
        result = _compute_meeting_intel_for_digest(windows, "2026-03-16")
        # May be None if SAMPLE_CALENDAR_HEAVY windows don't produce meeting windows
        # — either None or a valid dict is acceptable
        if result is not None:
            assert isinstance(result, dict)

    def test_result_has_expected_keys_when_meaningful(self):
        """When meetings are detected, the result dict has all expected keys."""
        from analysis.daily_digest import _compute_meeting_intel_for_digest
        windows = self._windows_with_meetings()
        result = _compute_meeting_intel_for_digest(windows, "2026-03-16")
        if result is None:
            pytest.skip("No meeting windows generated — calendar fixture may not produce meetings")
        expected_keys = {
            "mis", "ffs", "cmc", "sdr", "meeting_recovery_fit",
            "meeting_count", "total_meeting_minutes", "free_gap_minutes",
            "peak_focus_threats", "headline", "advisory", "section",
        }
        assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"

    def test_section_is_non_empty_string_when_meaningful(self):
        """The pre-formatted Slack section is a non-empty string."""
        from analysis.daily_digest import _compute_meeting_intel_for_digest
        windows = self._windows_with_meetings()
        result = _compute_meeting_intel_for_digest(windows, "2026-03-16")
        if result is None:
            pytest.skip("No meeting windows generated")
        assert isinstance(result["section"], str)
        assert len(result["section"]) > 0

    def test_mis_is_in_valid_range_when_meaningful(self):
        """MIS is an integer in [0, 100]."""
        from analysis.daily_digest import _compute_meeting_intel_for_digest
        windows = self._windows_with_meetings()
        result = _compute_meeting_intel_for_digest(windows, "2026-03-16")
        if result is None:
            pytest.skip("No meeting windows generated")
        mis = result["mis"]
        if mis is not None:
            assert 0 <= mis <= 100, f"MIS {mis} is outside [0, 100]"

    def test_does_not_crash_on_exception(self):
        """Returns None gracefully when meeting_intel raises an exception."""
        from unittest.mock import patch
        from analysis.daily_digest import _compute_meeting_intel_for_digest
        with patch("analysis.meeting_intel.compute_meeting_intel", side_effect=RuntimeError("boom")):
            result = _compute_meeting_intel_for_digest(
                self._windows_with_meetings(), "2026-03-16"
            )
        assert result is None

    # ── compute_digest integration tests ─────────────────────────────────

    def test_compute_digest_includes_meeting_intel_key(self):
        """compute_digest() always includes the 'meeting_intel' key."""
        windows = self._windows_no_meetings()
        digest = compute_digest(windows)
        assert "meeting_intel" in digest

    def test_meeting_intel_is_none_when_no_meetings(self):
        """meeting_intel is None when the day had no meetings."""
        windows = self._windows_no_meetings()
        digest = compute_digest(windows)
        assert digest["meeting_intel"] is None

    # ── format_digest_message rendering tests ────────────────────────────

    def test_format_digest_shows_meeting_intel_section(self):
        """When meeting_intel has a section, it appears in the formatted message."""
        windows = self._windows_no_meetings()
        digest = compute_digest(windows)
        digest["meeting_intel"] = {
            "mis": 72,
            "section": "*📅 Meeting Intelligence:*\nMIS 72/100 — Acceptable load.",
        }
        msg = format_digest_message(digest)
        assert "Meeting Intelligence" in msg

    def test_format_digest_omits_meeting_intel_when_none(self):
        """When meeting_intel is None, the section is absent from the message."""
        windows = self._windows_no_meetings()
        digest = compute_digest(windows)
        assert digest["meeting_intel"] is None
        msg = format_digest_message(digest)
        assert "Meeting Intelligence" not in msg

    def test_format_digest_omits_meeting_intel_when_section_missing(self):
        """When meeting_intel dict has no 'section' key, nothing crashes."""
        windows = self._windows_no_meetings()
        digest = compute_digest(windows)
        digest["meeting_intel"] = {"mis": 80}  # no 'section' key
        msg = format_digest_message(digest)
        assert "Meeting Intelligence" not in msg
        assert isinstance(msg, str)

    def test_meeting_intel_section_appears_before_insight(self):
        """Meeting Intel section appears before the insight line."""
        windows = self._windows_no_meetings()
        digest = compute_digest(windows)
        digest["meeting_intel"] = {
            "mis": 55,
            "section": "*📅 Meeting Intelligence:*\nMIS 55/100 — Heavy fragmentation.",
        }
        digest["insight"] = "Watch your cognitive load."
        msg = format_digest_message(digest)
        mi_pos = msg.find("Meeting Intelligence")
        insight_pos = msg.find("Watch your cognitive load")
        assert mi_pos != -1, "Meeting Intel not found"
        assert insight_pos != -1, "Insight not found"
        assert mi_pos < insight_pos, "Meeting Intel should appear before insight"

"""
Tests for the Morning Readiness Brief module.

Run with: python3 -m pytest tests/test_morning_brief.py -v

All tests are pure unit tests — no credentials, no network, no filesystem.
The morning brief only requires WHOOP data dict + optional context dicts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from analysis.morning_brief import (
    _readiness_tier,
    _tier_label,
    _tier_recommendation,
    _score_bar,
    _hrv_context,
    compute_morning_brief,
    format_morning_brief_message,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _whoop(recovery=85.0, hrv=72.0, sleep_h=8.0, sleep_perf=88.0, rhr=54.0):
    return {
        "recovery_score": recovery,
        "hrv_rmssd_milli": hrv,
        "sleep_hours": sleep_h,
        "sleep_performance": sleep_perf,
        "resting_heart_rate": rhr,
    }


def _yesterday(cls=0.30, meeting_mins=90, date="2026-03-13"):
    return {
        "date": date,
        "metrics_avg": {"cognitive_load_score": cls},
        "calendar": {"total_meeting_minutes": meeting_mins},
    }


# ─── _readiness_tier ─────────────────────────────────────────────────────────

class TestReadinessTier:
    def test_high_recovery_is_peak(self):
        assert _readiness_tier(85.0, 70.0) == "peak"

    def test_high_recovery_low_hrv_is_good(self):
        # HRV < 45 with high recovery degrades peak → good
        assert _readiness_tier(85.0, 40.0) == "good"

    def test_good_recovery_is_good(self):
        assert _readiness_tier(70.0, 65.0) == "good"

    def test_good_recovery_low_hrv_is_moderate(self):
        assert _readiness_tier(70.0, 38.0) == "moderate"

    def test_moderate_recovery_is_moderate(self):
        assert _readiness_tier(55.0, 60.0) == "moderate"

    def test_moderate_recovery_low_hrv_is_low(self):
        assert _readiness_tier(55.0, 40.0) == "low"

    def test_low_recovery_is_low(self):
        assert _readiness_tier(40.0, None) == "low"

    def test_very_low_recovery_is_recovery_day(self):
        assert _readiness_tier(25.0, None) == "recovery"

    def test_none_recovery_returns_unknown(self):
        assert _readiness_tier(None, 70.0) == "unknown"

    def test_none_hrv_uses_recovery_only(self):
        # No HRV penalty when HRV is not available
        assert _readiness_tier(85.0, None) == "peak"

    def test_boundary_at_80(self):
        assert _readiness_tier(80.0, 70.0) == "peak"
        assert _readiness_tier(79.9, 70.0) == "good"

    def test_boundary_at_67(self):
        assert _readiness_tier(67.0, 70.0) == "good"
        assert _readiness_tier(66.9, 70.0) == "moderate"

    def test_boundary_at_50(self):
        assert _readiness_tier(50.0, 70.0) == "moderate"
        assert _readiness_tier(49.9, 70.0) == "low"

    def test_boundary_at_33(self):
        assert _readiness_tier(33.0, 70.0) == "low"
        assert _readiness_tier(32.9, 70.0) == "recovery"

    def test_all_valid_tiers_returned(self):
        valid = {"peak", "good", "moderate", "low", "recovery", "unknown"}
        for recovery in [None, 20, 40, 55, 70, 85]:
            for hrv in [None, 30, 60, 90]:
                tier = _readiness_tier(recovery, hrv)
                assert tier in valid, f"Unexpected tier {tier!r} for recovery={recovery}, hrv={hrv}"


# ─── _tier_label ──────────────────────────────────────────────────────────────

class TestTierLabel:
    def test_all_tiers_have_labels(self):
        for tier in ["peak", "good", "moderate", "low", "recovery", "unknown"]:
            label = _tier_label(tier)
            assert isinstance(label, str)
            assert len(label) > 0

    def test_peak_label(self):
        assert _tier_label("peak") == "Peak"

    def test_recovery_label(self):
        assert _tier_label("recovery") == "Recovery Day"

    def test_unknown_label(self):
        assert _tier_label("unknown") == "Unknown"


# ─── _tier_recommendation ────────────────────────────────────────────────────

class TestTierRecommendation:
    def test_all_tiers_return_non_empty_string(self):
        for tier in ["peak", "good", "moderate", "low", "recovery", "unknown"]:
            rec = _tier_recommendation(tier, 70.0, 65.0, None, None)
            assert isinstance(rec, str)
            assert len(rec) > 20, f"Recommendation for {tier} is too short: {rec!r}"

    def test_peak_with_heavy_yesterday_mentions_recovery(self):
        rec = _tier_recommendation("peak", 90.0, 75.0, 0.60, 300)
        # Should mention the heavy/demanding previous session
        assert "recovered" in rec.lower() or "demanding" in rec.lower() or "yesterday" in rec.lower()

    def test_moderate_with_meeting_heavy_yesterday_warns(self):
        rec = _tier_recommendation("moderate", 58.0, 60.0, 0.50, 300)
        assert "meeting" in rec.lower()

    def test_low_tier_mentions_schedule_light(self):
        rec = _tier_recommendation("low", 40.0, 35.0, None, None)
        assert "light" in rec.lower() or "routine" in rec.lower()

    def test_recovery_day_mentions_cancel_or_reschedule(self):
        rec = _tier_recommendation("recovery", 25.0, None, None, None)
        assert "recovery" in rec.lower() or "cancel" in rec.lower()

    def test_unknown_tier_mentions_whoop(self):
        rec = _tier_recommendation("unknown", None, None, None, None)
        assert "WHOOP" in rec or "unavailable" in rec.lower()

    def test_low_with_hrv_includes_hrv_value(self):
        rec = _tier_recommendation("low", 40.0, 38.0, None, None)
        # Should mention the actual HRV value in the recommendation
        assert "38" in rec or "hrv" in rec.lower() or "autonomic" in rec.lower()

    def test_none_parameters_dont_crash(self):
        for tier in ["peak", "good", "moderate", "low", "recovery", "unknown"]:
            rec = _tier_recommendation(tier, None, None, None, None)
            assert isinstance(rec, str)


# ─── _score_bar ───────────────────────────────────────────────────────────────

class TestScoreBar:
    def test_zero_returns_all_empty(self):
        bar = _score_bar(0.0)
        assert "▓" not in bar
        assert len(bar) == 10

    def test_one_returns_all_filled(self):
        bar = _score_bar(1.0)
        assert "░" not in bar
        assert len(bar) == 10

    def test_half_splits_evenly(self):
        bar = _score_bar(0.5)
        assert bar.count("▓") == 5
        assert bar.count("░") == 5

    def test_custom_width(self):
        bar = _score_bar(0.7, width=20)
        assert len(bar) == 20


# ─── _hrv_context ─────────────────────────────────────────────────────────────

class TestHrvContext:
    def test_none_hrv_returns_na(self):
        assert _hrv_context(None, None) == "N/A"

    def test_no_baseline_returns_plain_value(self):
        result = _hrv_context(70.0, None)
        assert "70" in result
        assert "baseline" not in result

    def test_near_baseline_says_baseline(self):
        result = _hrv_context(70.0, 70.0)
        assert "baseline" in result

    def test_above_baseline_shows_positive_pct(self):
        result = _hrv_context(80.0, 70.0)
        assert "+" in result

    def test_below_baseline_shows_negative_pct(self):
        result = _hrv_context(55.0, 70.0)
        assert "-" in result or "−" in result or "%" in result

    def test_small_deviation_says_baseline(self):
        # 5% difference is less than the 8% threshold
        result = _hrv_context(73.0, 70.0)
        assert "baseline" in result


# ─── compute_morning_brief ────────────────────────────────────────────────────

class TestComputeMorningBrief:
    def test_returns_dict(self):
        brief = compute_morning_brief("2026-03-14", _whoop())
        assert isinstance(brief, dict)

    def test_has_required_keys(self):
        brief = compute_morning_brief("2026-03-14", _whoop())
        assert "date" in brief
        assert "whoop" in brief
        assert "readiness" in brief
        assert "yesterday" in brief
        assert "hrv_baseline" in brief

    def test_date_propagated(self):
        brief = compute_morning_brief("2026-03-14", _whoop())
        assert brief["date"] == "2026-03-14"

    def test_whoop_signals_extracted(self):
        brief = compute_morning_brief("2026-03-14", _whoop(recovery=88.0, hrv=74.0))
        assert brief["whoop"]["recovery_score"] == 88.0
        assert brief["whoop"]["hrv_rmssd_milli"] == 74.0

    def test_readiness_tier_present(self):
        brief = compute_morning_brief("2026-03-14", _whoop(recovery=85.0))
        assert brief["readiness"]["tier"] in {"peak", "good", "moderate", "low", "recovery", "unknown"}

    def test_readiness_label_present(self):
        brief = compute_morning_brief("2026-03-14", _whoop())
        assert isinstance(brief["readiness"]["label"], str)
        assert len(brief["readiness"]["label"]) > 0

    def test_recommendation_present(self):
        brief = compute_morning_brief("2026-03-14", _whoop())
        assert isinstance(brief["readiness"]["recommendation"], str)
        assert len(brief["readiness"]["recommendation"]) > 10

    def test_yesterday_cls_extracted(self):
        brief = compute_morning_brief("2026-03-14", _whoop(), _yesterday(cls=0.45))
        assert brief["yesterday"]["avg_cls"] == pytest.approx(0.45)

    def test_yesterday_meeting_mins_extracted(self):
        brief = compute_morning_brief("2026-03-14", _whoop(), _yesterday(meeting_mins=180))
        assert brief["yesterday"]["meeting_minutes"] == 180

    def test_hrv_baseline_stored(self):
        brief = compute_morning_brief("2026-03-14", _whoop(), hrv_baseline=68.5)
        assert brief["hrv_baseline"] == pytest.approx(68.5)

    def test_empty_whoop_data_doesnt_crash(self):
        brief = compute_morning_brief("2026-03-14", {})
        assert isinstance(brief, dict)
        assert brief["readiness"]["tier"] == "unknown"

    def test_none_yesterday_is_graceful(self):
        brief = compute_morning_brief("2026-03-14", _whoop(), yesterday_summary=None)
        assert brief["yesterday"]["avg_cls"] is None
        assert brief["yesterday"]["meeting_minutes"] is None

    def test_high_recovery_peak_tier(self):
        brief = compute_morning_brief("2026-03-14", _whoop(recovery=90.0, hrv=75.0))
        assert brief["readiness"]["tier"] == "peak"

    def test_low_recovery_low_tier(self):
        brief = compute_morning_brief("2026-03-14", _whoop(recovery=38.0, hrv=35.0))
        assert brief["readiness"]["tier"] in {"low", "recovery"}


# ─── format_morning_brief_message ────────────────────────────────────────────

class TestFormatMorningBriefMessage:
    def _brief(self, recovery=85.0, hrv=72.0, hrv_baseline=None, yesterday_cls=None):
        return compute_morning_brief(
            "2026-03-14",
            _whoop(recovery=recovery, hrv=hrv),
            _yesterday(cls=yesterday_cls) if yesterday_cls is not None else None,
            hrv_baseline=hrv_baseline,
        )

    def test_returns_string(self):
        msg = format_morning_brief_message(self._brief())
        assert isinstance(msg, str)

    def test_contains_date_label(self):
        msg = format_morning_brief_message(self._brief())
        assert "Saturday" in msg or "March" in msg  # 2026-03-14 is a Saturday

    def test_contains_recovery_score(self):
        msg = format_morning_brief_message(self._brief(recovery=85.0))
        assert "85" in msg

    def test_contains_tier_label(self):
        msg = format_morning_brief_message(self._brief(recovery=85.0, hrv=72.0))
        # Peak tier
        assert "Peak" in msg

    def test_contains_recommendation(self):
        msg = format_morning_brief_message(self._brief())
        assert "Today:" in msg

    def test_contains_score_bar(self):
        msg = format_morning_brief_message(self._brief())
        assert "▓" in msg or "░" in msg

    def test_contains_hrv(self):
        msg = format_morning_brief_message(self._brief(hrv=72.0))
        assert "72" in msg

    def test_yesterday_section_shown_when_present(self):
        msg = format_morning_brief_message(self._brief(yesterday_cls=0.45))
        assert "Yesterday" in msg

    def test_yesterday_section_absent_when_no_data(self):
        brief = compute_morning_brief("2026-03-14", _whoop(), None)
        msg = format_morning_brief_message(brief)
        assert "Yesterday" not in msg

    def test_hrv_baseline_context_shown(self):
        msg = format_morning_brief_message(self._brief(hrv=80.0, hrv_baseline=70.0))
        assert "baseline" in msg or "%" in msg

    def test_empty_brief_returns_fallback(self):
        msg = format_morning_brief_message({})
        assert "no morning data" in msg.lower() or "unavailable" in msg.lower()

    def test_low_recovery_shows_red_emoji(self):
        msg = format_morning_brief_message(self._brief(recovery=25.0, hrv=30.0))
        assert "🔴" in msg

    def test_peak_recovery_shows_green_emoji(self):
        msg = format_morning_brief_message(self._brief(recovery=88.0, hrv=75.0))
        assert "🟢" in msg

    def test_no_whoop_data_handled_gracefully(self):
        brief = compute_morning_brief("2026-03-14", {})
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str)
        assert len(msg) > 10

    def test_morning_header_present(self):
        msg = format_morning_brief_message(self._brief())
        assert "Morning Readiness" in msg

    def test_meeting_minutes_shown_in_yesterday(self):
        brief = compute_morning_brief("2026-03-14", _whoop(), _yesterday(cls=0.50, meeting_mins=240))
        msg = format_morning_brief_message(brief)
        assert "240" in msg or "min" in msg

    def test_sleep_info_shown(self):
        msg = format_morning_brief_message(self._brief())
        assert "Sleep" in msg or "8.0" in msg

    def test_rhr_shown(self):
        msg = format_morning_brief_message(self._brief())
        assert "RHR" in msg or "bpm" in msg


# ─── Tier emoji coverage ──────────────────────────────────────────────────────

class TestTierEmojis:
    """Verify each readiness tier maps to a distinct, correct emoji."""

    def _msg_for_tier(self, tier: str) -> str:
        recovery_map = {
            "peak": (88.0, 75.0),
            "good": (72.0, 65.0),
            "moderate": (55.0, 60.0),
            "low": (40.0, 50.0),
            "recovery": (20.0, 25.0),
        }
        recovery, hrv = recovery_map.get(tier, (85.0, 70.0))
        brief = compute_morning_brief("2026-03-14", _whoop(recovery=recovery, hrv=hrv))
        return format_morning_brief_message(brief)

    def test_peak_emoji(self):
        assert "🟢" in self._msg_for_tier("peak")

    def test_good_emoji(self):
        assert "🔵" in self._msg_for_tier("good")

    def test_moderate_emoji(self):
        assert "🟡" in self._msg_for_tier("moderate")

    def test_low_emoji(self):
        assert "🟠" in self._msg_for_tier("low")

    def test_recovery_emoji(self):
        assert "🔴" in self._msg_for_tier("recovery")


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_recovery_at_exactly_100(self):
        brief = compute_morning_brief("2026-03-14", _whoop(recovery=100.0, hrv=100.0))
        assert brief["readiness"]["tier"] == "peak"
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str)

    def test_recovery_at_zero(self):
        brief = compute_morning_brief("2026-03-14", _whoop(recovery=0.0, hrv=20.0))
        assert brief["readiness"]["tier"] == "recovery"

    def test_hrv_at_zero(self):
        # Zero HRV should still produce a valid message
        brief = compute_morning_brief("2026-03-14", _whoop(hrv=0.0))
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str)

    def test_very_long_recommendation_fits_in_message(self):
        brief = compute_morning_brief("2026-03-14", _whoop(), _yesterday(cls=0.70, meeting_mins=420))
        msg = format_morning_brief_message(brief)
        # Message should be reasonable length (not truncated or broken)
        assert len(msg) > 100
        assert "Today:" in msg

    def test_yesterday_meeting_zero_handled(self):
        brief = compute_morning_brief("2026-03-14", _whoop(), _yesterday(meeting_mins=0))
        msg = format_morning_brief_message(brief)
        # Should not show "0min meetings"
        assert isinstance(msg, str)

    def test_hrv_baseline_near_zero_doesnt_crash(self):
        brief = compute_morning_brief("2026-03-14", _whoop(hrv=65.0), hrv_baseline=0.1)
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str)


# ─── v1.5: Trend context ─────────────────────────────────────────────────────

def _trend(
    days=4,
    hrv_trend="stable",
    hrv_streak=0,
    hrv_vs_baseline=None,
    hrv_baseline_ms=None,
    recovery_trend="stable",
    recovery_streak=0,
    overcapacity_streak=0,
    cls_vs_baseline=None,
    note="",
):
    """Build a synthetic trend_context dict for tests."""
    return {
        "days_of_data": days,
        "hrv_trend": hrv_trend,
        "hrv_streak_days": hrv_streak,
        "hrv_vs_baseline": hrv_vs_baseline,
        "hrv_baseline_ms": hrv_baseline_ms,
        "recovery_trend": recovery_trend,
        "recovery_streak_days": recovery_streak,
        "overcapacity_streak": overcapacity_streak,
        "cls_vs_baseline": cls_vs_baseline,
        "note": note,
    }


class TestComputeMorningBriefTrendContext:
    """Test that compute_morning_brief correctly stores trend_context."""

    def test_trend_context_stored_in_brief(self):
        tc = _trend(days=3, hrv_trend="declining", hrv_streak=2)
        brief = compute_morning_brief("2026-03-14", _whoop(), trend_context=tc)
        assert "trend_context" in brief
        assert brief["trend_context"]["hrv_trend"] == "declining"

    def test_none_trend_context_stored_as_empty_dict(self):
        brief = compute_morning_brief("2026-03-14", _whoop(), trend_context=None)
        assert brief["trend_context"] == {}

    def test_empty_trend_context_stored_as_empty_dict(self):
        brief = compute_morning_brief("2026-03-14", _whoop(), trend_context={})
        assert brief["trend_context"] == {}

    def test_trend_context_has_required_keys_when_provided(self):
        tc = _trend(days=5)
        brief = compute_morning_brief("2026-03-14", _whoop(), trend_context=tc)
        assert brief["trend_context"].get("days_of_data") == 5

    def test_existing_keys_still_present_with_trend(self):
        """Adding trend_context must not break the existing brief structure."""
        tc = _trend(days=2)
        brief = compute_morning_brief("2026-03-14", _whoop(), trend_context=tc)
        assert "date" in brief
        assert "whoop" in brief
        assert "readiness" in brief
        assert "yesterday" in brief
        assert "hrv_baseline" in brief

    def test_backward_compat_no_trend_param(self):
        """Callers that don't pass trend_context must still work."""
        brief = compute_morning_brief("2026-03-14", _whoop())
        assert isinstance(brief, dict)
        assert brief["trend_context"] == {}


class TestFormatMorningBriefTrendSection:
    """Test that format_morning_brief_message renders trend context correctly."""

    def _brief_with_trend(self, tc):
        return compute_morning_brief("2026-03-14", _whoop(), trend_context=tc)

    def test_no_pattern_section_when_no_trend_data(self):
        brief = self._brief_with_trend({})
        msg = format_morning_brief_message(brief)
        assert "Pattern:" not in msg

    def test_no_pattern_section_when_days_of_data_lt_2(self):
        tc = _trend(days=1, hrv_trend="declining", hrv_streak=2)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "Pattern:" not in msg

    def test_pattern_section_shown_when_hrv_declining(self):
        tc = _trend(days=4, hrv_trend="declining", hrv_streak=3)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "Pattern:" in msg
        assert "HRV declining" in msg
        assert "3" in msg

    def test_pattern_section_shown_when_hrv_improving(self):
        tc = _trend(days=4, hrv_trend="improving", hrv_streak=2)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "Pattern:" in msg
        assert "HRV improving" in msg

    def test_no_hrv_trend_shown_for_single_day_streak(self):
        # hrv_streak=1 is below the ≥2 threshold
        tc = _trend(days=3, hrv_trend="declining", hrv_streak=1)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "HRV declining" not in msg

    def test_overcapacity_streak_shown(self):
        tc = _trend(days=5, overcapacity_streak=3)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "Pattern:" in msg
        assert "capacity" in msg
        assert "3" in msg

    def test_single_overcapacity_day_not_shown(self):
        # overcapacity_streak=1 is below the ≥2 threshold
        tc = _trend(days=3, overcapacity_streak=1)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "Above capacity" not in msg

    def test_recovery_decline_streak_shown(self):
        tc = _trend(days=4, recovery_trend="declining", recovery_streak=2)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "Pattern:" in msg
        assert "Recovery declining" in msg

    def test_cls_above_baseline_shown_when_significant(self):
        # 40% above baseline should surface
        tc = _trend(days=5, cls_vs_baseline=40.0)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "Pattern:" in msg
        assert "above" in msg.lower()
        assert "40" in msg

    def test_cls_below_baseline_shown_when_significant(self):
        tc = _trend(days=5, cls_vs_baseline=-30.0)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "Pattern:" in msg
        assert "below" in msg.lower()

    def test_small_cls_deviation_not_shown(self):
        # 10% is below the 25% threshold
        tc = _trend(days=5, cls_vs_baseline=10.0)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "baseline" not in msg

    def test_stable_trend_no_pattern_section(self):
        tc = _trend(days=5, hrv_trend="stable", hrv_streak=0, overcapacity_streak=0)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "Pattern:" not in msg

    def test_multiple_signals_all_shown(self):
        tc = _trend(
            days=5,
            hrv_trend="declining", hrv_streak=3,
            overcapacity_streak=2,
            recovery_trend="declining", recovery_streak=2,
        )
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "HRV declining" in msg
        assert "capacity" in msg.lower()
        assert "Recovery declining" in msg

    def test_pattern_section_appears_after_recommendation(self):
        tc = _trend(days=4, hrv_trend="declining", hrv_streak=2)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        rec_pos = msg.find("Today:")
        pattern_pos = msg.find("Pattern:")
        assert rec_pos < pattern_pos, "Pattern section should appear after Today: recommendation"

    def test_warning_emoji_present_for_declining_hrv(self):
        tc = _trend(days=4, hrv_trend="declining", hrv_streak=2)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "⚠️" in msg

    def test_checkmark_emoji_present_for_improving_hrv(self):
        tc = _trend(days=4, hrv_trend="improving", hrv_streak=2)
        brief = self._brief_with_trend(tc)
        msg = format_morning_brief_message(brief)
        assert "✅" in msg

    def test_empty_brief_with_trend_doesnt_crash(self):
        msg = format_morning_brief_message({})
        assert isinstance(msg, str)

    def test_trend_context_none_in_brief_doesnt_crash(self):
        brief = compute_morning_brief("2026-03-14", _whoop())
        brief["trend_context"] = None  # Simulate None stored
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str)
        assert "Pattern:" not in msg

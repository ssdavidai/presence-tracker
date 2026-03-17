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


# ─── DPS Trend in morning brief ───────────────────────────────────────────────

class TestComputeMorningBriefDpsTrend:
    """Tests for the DPS trend field injected into compute_morning_brief()."""

    def test_dps_trend_key_always_present(self):
        """compute_morning_brief() must always include 'dps_trend' key."""
        brief = compute_morning_brief("2026-03-14", _whoop())
        assert "dps_trend" in brief

    def test_dps_trend_is_none_or_dict(self):
        """dps_trend must be None (insufficient data) or a dict."""
        brief = compute_morning_brief("2026-03-14", _whoop())
        value = brief.get("dps_trend")
        assert value is None or isinstance(value, dict)

    def test_dps_trend_dict_has_required_keys(self, monkeypatch):
        """When DPS trend data is available, the dict has all expected keys."""
        import analysis.morning_brief as mb_mod
        fake_trend = {
            "mean_dps": 68.0,
            "delta_dps": -12.0,
            "trend_direction": "declining",
            "days_used": 5,
            "recent_scores": [72, 65, 58],
            "line": "📉 Day quality declining: 72 → 65 → 58 (−14 pts)",
        }
        monkeypatch.setattr(mb_mod, "_compute_dps_trend_for_brief", lambda *a: fake_trend)
        brief = compute_morning_brief("2026-03-14", _whoop())
        dps_trend = brief.get("dps_trend")
        assert dps_trend is not None
        for key in ("mean_dps", "delta_dps", "trend_direction", "days_used", "recent_scores", "line"):
            assert key in dps_trend, f"Missing key in dps_trend: {key}"

    def test_dps_trend_none_when_helper_returns_none(self, monkeypatch):
        """When _compute_dps_trend_for_brief returns None, brief['dps_trend'] is None."""
        import analysis.morning_brief as mb_mod
        monkeypatch.setattr(mb_mod, "_compute_dps_trend_for_brief", lambda *a: None)
        brief = compute_morning_brief("2026-03-14", _whoop())
        assert brief.get("dps_trend") is None

    def test_dps_trend_present_when_helper_returns_data(self, monkeypatch):
        """When _compute_dps_trend_for_brief returns data, brief['dps_trend'] is that data."""
        import analysis.morning_brief as mb_mod
        fake = {"mean_dps": 72.0, "delta_dps": 8.0, "trend_direction": "improving",
                "days_used": 5, "recent_scores": [62, 68, 74],
                "line": "📈 Day quality improving: 62 → 68 → 74 (+12 pts)"}
        monkeypatch.setattr(mb_mod, "_compute_dps_trend_for_brief", lambda *a: fake)
        brief = compute_morning_brief("2026-03-14", _whoop())
        assert brief.get("dps_trend") == fake

    def test_existing_brief_keys_unaffected(self, monkeypatch):
        """Adding dps_trend must not remove or corrupt other brief fields."""
        import analysis.morning_brief as mb_mod
        monkeypatch.setattr(mb_mod, "_compute_dps_trend_for_brief", lambda *a: None)
        brief = compute_morning_brief("2026-03-14", _whoop())
        for key in ("date", "whoop", "readiness", "yesterday", "cognitive_debt"):
            assert key in brief, f"Key '{key}' missing from brief after DPS trend addition"


class TestFormatMorningBriefDpsTrend:
    """Tests for DPS trend rendering in format_morning_brief_message()."""

    def _brief_with_dps_trend(self, dps_trend: dict | None) -> dict:
        """Build a minimal brief dict with the given dps_trend."""
        return {
            "date": "2026-03-14",
            "whoop": {"recovery_score": 78.0, "hrv_rmssd_milli": 65.0,
                      "sleep_hours": 7.5, "sleep_performance": 82.0, "resting_heart_rate": 56.0},
            "readiness": {"tier": "good", "label": "Good", "recommendation": "Productive day."},
            "yesterday": {"date": "2026-03-13", "avg_cls": 0.45, "meeting_minutes": 90},
            "hrv_baseline": 70.0,
            "trend_context": {},
            "personal_baseline": None,
            "today_calendar": None,
            "cognitive_debt": None,
            "dps_trend": dps_trend,
        }

    def test_no_dps_trend_section_when_none(self):
        """When dps_trend is None, no DPS trend line appears in the message."""
        brief = self._brief_with_dps_trend(None)
        msg = format_morning_brief_message(brief)
        assert "Day quality" not in msg
        assert "📈" not in msg
        assert "📉" not in msg

    def test_no_dps_trend_section_when_empty_line(self):
        """When dps_trend has an empty line, no DPS trend appears."""
        brief = self._brief_with_dps_trend({
            "mean_dps": 70.0, "delta_dps": 1.0, "trend_direction": "stable",
            "days_used": 3, "recent_scores": [69, 71, 70], "line": "",
        })
        msg = format_morning_brief_message(brief)
        assert "Day quality" not in msg

    def test_declining_trend_line_appears(self):
        """Declining DPS trend should render in the message."""
        brief = self._brief_with_dps_trend({
            "mean_dps": 63.0,
            "delta_dps": -15.0,
            "trend_direction": "declining",
            "days_used": 5,
            "recent_scores": [72, 65, 58],
            "line": "📉 Day quality declining: 72 → 65 → 58 (−15 pts)",
        })
        msg = format_morning_brief_message(brief)
        assert "Day quality declining" in msg
        assert "📉" in msg

    def test_improving_trend_line_appears(self):
        """Improving DPS trend should render in the message."""
        brief = self._brief_with_dps_trend({
            "mean_dps": 68.0,
            "delta_dps": 12.0,
            "trend_direction": "improving",
            "days_used": 5,
            "recent_scores": [58, 65, 73],
            "line": "📈 Day quality improving: 58 → 65 → 73 (+15 pts)",
        })
        msg = format_morning_brief_message(brief)
        assert "Day quality improving" in msg
        assert "📈" in msg

    def test_stable_trend_line_appears(self):
        """Stable DPS trend with line set should render."""
        brief = self._brief_with_dps_trend({
            "mean_dps": 70.0,
            "delta_dps": 1.5,
            "trend_direction": "stable",
            "days_used": 6,
            "recent_scores": [69, 72, 70],
            "line": "➡️ Day quality stable (6d avg: 70/100)",
        })
        msg = format_morning_brief_message(brief)
        assert "Day quality stable" in msg
        assert "➡️" in msg

    def test_sparkline_scores_appear_in_message(self):
        """The recent score sparkline should appear in the message."""
        brief = self._brief_with_dps_trend({
            "mean_dps": 65.0,
            "delta_dps": -14.0,
            "trend_direction": "declining",
            "days_used": 5,
            "recent_scores": [78, 65, 52],
            "line": "📉 Day quality declining: 78 → 65 → 52 (−26 pts)",
        })
        msg = format_morning_brief_message(brief)
        assert "78" in msg
        assert "65" in msg
        assert "52" in msg

    def test_dps_trend_appears_after_cdi(self):
        """DPS trend section should appear after CDI in the message (if both present)."""
        brief = self._brief_with_dps_trend({
            "mean_dps": 63.0,
            "delta_dps": -12.0,
            "trend_direction": "declining",
            "days_used": 5,
            "recent_scores": [72, 65, 58],
            "line": "📉 Day quality declining: 72 → 65 → 58 (−12 pts)",
        })
        brief["cognitive_debt"] = {
            "cdi": 72,
            "tier": "fatigued",
            "line": "🟠 CDI 72/100 — Fatigued (5 deficit days, trend ↑)",
        }
        msg = format_morning_brief_message(brief)
        cdi_pos = msg.find("CDI")
        dps_pos = msg.find("Day quality")
        assert cdi_pos >= 0, "CDI line missing from message"
        assert dps_pos >= 0, "DPS trend line missing from message"
        assert cdi_pos < dps_pos, "DPS trend should appear after CDI"

    def test_no_crash_when_dps_trend_missing_keys(self):
        """Partial dps_trend dict (missing keys) should not crash the formatter."""
        brief = self._brief_with_dps_trend({"line": "📉 Day quality declining: 70 → 60"})
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str)

    def test_no_crash_when_dps_trend_is_empty_dict(self):
        """Empty dps_trend dict should not crash — just no DPS line in output."""
        brief = self._brief_with_dps_trend({})
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str)
        assert "Day quality" not in msg

    def test_no_crash_when_brief_has_no_dps_trend_key(self):
        """Brief without 'dps_trend' key (old-format brief) should not crash."""
        brief = self._brief_with_dps_trend(None)
        del brief["dps_trend"]
        msg = format_morning_brief_message(brief)
        assert isinstance(msg, str)


class TestFormatDpsTrendLine:
    """Unit tests for _format_dps_trend_line() directly."""

    def test_import(self):
        from analysis.morning_brief import _format_dps_trend_line
        assert callable(_format_dps_trend_line)

    def test_improving_contains_arrow_emoji(self):
        from analysis.morning_brief import _format_dps_trend_line
        trend = {"trend_direction": "improving", "delta_dps": 12.0, "mean_dps": 70.0, "days_used": 5}
        result = _format_dps_trend_line(trend, [60, 68, 74])
        assert "📈" in result

    def test_declining_contains_down_arrow_emoji(self):
        from analysis.morning_brief import _format_dps_trend_line
        trend = {"trend_direction": "declining", "delta_dps": -14.0, "mean_dps": 64.0, "days_used": 5}
        result = _format_dps_trend_line(trend, [78, 65, 55])
        assert "📉" in result

    def test_stable_with_enough_days_has_right_arrow(self):
        from analysis.morning_brief import _format_dps_trend_line
        trend = {"trend_direction": "stable", "delta_dps": 1.0, "mean_dps": 70.0, "days_used": 5}
        result = _format_dps_trend_line(trend, [69, 71, 70])
        assert "➡️" in result

    def test_stable_with_few_days_returns_empty(self):
        from analysis.morning_brief import _format_dps_trend_line
        trend = {"trend_direction": "stable", "delta_dps": 1.0, "mean_dps": 70.0, "days_used": 4}
        result = _format_dps_trend_line(trend, [69, 71, 70])
        assert result == ""

    def test_improving_includes_plus_delta(self):
        from analysis.morning_brief import _format_dps_trend_line
        trend = {"trend_direction": "improving", "delta_dps": 15.0, "mean_dps": 72.0, "days_used": 5}
        result = _format_dps_trend_line(trend, [58, 68, 74])
        assert "15" in result
        assert "+" in result or "pts" in result

    def test_declining_includes_minus_delta(self):
        from analysis.morning_brief import _format_dps_trend_line
        trend = {"trend_direction": "declining", "delta_dps": -18.0, "mean_dps": 62.0, "days_used": 5}
        result = _format_dps_trend_line(trend, [80, 68, 54])
        assert "18" in result
        assert "−" in result

    def test_sparkline_scores_appear_in_output(self):
        from analysis.morning_brief import _format_dps_trend_line
        trend = {"trend_direction": "improving", "delta_dps": 10.0, "mean_dps": 68.0, "days_used": 4}
        result = _format_dps_trend_line(trend, [55, 63, 71])
        assert "55" in result
        assert "63" in result
        assert "71" in result

    def test_empty_scores_does_not_crash(self):
        from analysis.morning_brief import _format_dps_trend_line
        trend = {"trend_direction": "improving", "delta_dps": 8.0, "mean_dps": 70.0, "days_used": 4}
        result = _format_dps_trend_line(trend, [])
        assert isinstance(result, str)

    def test_returns_string_for_all_directions(self):
        from analysis.morning_brief import _format_dps_trend_line
        for direction in ("improving", "declining", "stable"):
            trend = {"trend_direction": direction, "delta_dps": 5.0, "mean_dps": 70.0, "days_used": 5}
            result = _format_dps_trend_line(trend, [65, 70, 72])
            assert isinstance(result, str)


class TestComputeDpsTrendForBrief:
    """Unit tests for _compute_dps_trend_for_brief() (the helper wrapper)."""

    def test_returns_none_when_trend_none(self, monkeypatch):
        """When compute_dps_trend returns None, helper returns None."""
        import analysis.morning_brief as mb_mod
        import analysis.presence_score as ps_mod
        monkeypatch.setattr(ps_mod, "compute_dps_trend", lambda *a, **kw: None)
        monkeypatch.setattr(ps_mod, "get_historical_dps", lambda *a, **kw: [])
        from analysis.morning_brief import _compute_dps_trend_for_brief
        result = _compute_dps_trend_for_brief("2026-03-14")
        assert result is None

    def test_returns_dict_when_trend_available(self, monkeypatch):
        """When compute_dps_trend has data, helper returns a dict."""
        import analysis.presence_score as ps_mod
        fake_trend = {
            "mean_dps": 68.0, "delta_dps": -12.0, "trend_direction": "declining",
            "days_used": 5, "best_day": "2026-03-10", "best_dps": 78.0,
            "worst_day": "2026-03-14", "worst_dps": 54.0,
        }
        fake_history = [
            {"date": "2026-03-10", "dps": 78.0, "tier": "strong", "is_meaningful": True},
            {"date": "2026-03-12", "dps": 65.0, "tier": "good", "is_meaningful": True},
            {"date": "2026-03-14", "dps": 54.0, "tier": "moderate", "is_meaningful": True},
        ]
        monkeypatch.setattr(ps_mod, "compute_dps_trend", lambda *a, **kw: fake_trend)
        monkeypatch.setattr(ps_mod, "get_historical_dps", lambda *a, **kw: fake_history)
        from analysis.morning_brief import _compute_dps_trend_for_brief
        result = _compute_dps_trend_for_brief("2026-03-14")
        assert result is not None
        assert isinstance(result, dict)

    def test_result_has_line_key(self, monkeypatch):
        """The helper result must include a 'line' key."""
        import analysis.presence_score as ps_mod
        fake_trend = {
            "mean_dps": 68.0, "delta_dps": -12.0, "trend_direction": "declining",
            "days_used": 5, "best_day": "2026-03-10", "best_dps": 78.0,
            "worst_day": "2026-03-14", "worst_dps": 54.0,
        }
        fake_history = [
            {"date": "2026-03-12", "dps": 68.0, "tier": "good", "is_meaningful": True},
            {"date": "2026-03-13", "dps": 62.0, "tier": "good", "is_meaningful": True},
            {"date": "2026-03-14", "dps": 54.0, "tier": "moderate", "is_meaningful": True},
        ]
        monkeypatch.setattr(ps_mod, "compute_dps_trend", lambda *a, **kw: fake_trend)
        monkeypatch.setattr(ps_mod, "get_historical_dps", lambda *a, **kw: fake_history)
        from analysis.morning_brief import _compute_dps_trend_for_brief
        result = _compute_dps_trend_for_brief("2026-03-14")
        assert result is not None
        assert "line" in result

    def test_graceful_on_exception(self, monkeypatch):
        """Any exception inside the helper returns None without crashing."""
        import analysis.presence_score as ps_mod
        monkeypatch.setattr(ps_mod, "compute_dps_trend",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")))
        from analysis.morning_brief import _compute_dps_trend_for_brief
        result = _compute_dps_trend_for_brief("2026-03-14")
        assert result is None

    def test_result_includes_recent_scores(self, monkeypatch):
        """Helper must include recent_scores derived from the last 3 history entries."""
        import analysis.presence_score as ps_mod
        fake_trend = {
            "mean_dps": 70.0, "delta_dps": 10.0, "trend_direction": "improving",
            "days_used": 5, "best_day": "2026-03-14", "best_dps": 75.0,
            "worst_day": "2026-03-10", "worst_dps": 55.0,
        }
        fake_history = [
            {"date": "2026-03-10", "dps": 55.0, "tier": "moderate", "is_meaningful": True},
            {"date": "2026-03-11", "dps": 60.0, "tier": "good", "is_meaningful": True},
            {"date": "2026-03-12", "dps": 65.0, "tier": "good", "is_meaningful": True},
            {"date": "2026-03-13", "dps": 70.0, "tier": "good", "is_meaningful": True},
            {"date": "2026-03-14", "dps": 75.0, "tier": "strong", "is_meaningful": True},
        ]
        monkeypatch.setattr(ps_mod, "compute_dps_trend", lambda *a, **kw: fake_trend)
        monkeypatch.setattr(ps_mod, "get_historical_dps", lambda *a, **kw: fake_history)
        from analysis.morning_brief import _compute_dps_trend_for_brief
        result = _compute_dps_trend_for_brief("2026-03-14")
        assert result is not None
        assert "recent_scores" in result
        # Last 3 meaningful: 65, 70, 75
        assert result["recent_scores"] == [65, 70, 75]


# ─── Cognitive Rhythm integration (v18.0) ────────────────────────────────────

class TestComputeCognitiveRhythmForBrief:
    """Unit tests for _compute_cognitive_rhythm_for_brief()."""

    def test_returns_none_when_not_meaningful(self, monkeypatch):
        """Returns None when rhythm has is_meaningful=False (insufficient data)."""
        import analysis.cognitive_rhythm as cr_mod
        from analysis.cognitive_rhythm import CognitiveRhythm
        not_meaningful = CognitiveRhythm(is_meaningful=False, days_analyzed=0)
        monkeypatch.setattr(cr_mod, "compute_cognitive_rhythm", lambda *a, **kw: not_meaningful)
        from analysis.morning_brief import _compute_cognitive_rhythm_for_brief
        result = _compute_cognitive_rhythm_for_brief("2026-03-14")
        assert result is None

    def test_returns_dict_when_meaningful(self, monkeypatch):
        """Returns a dict when rhythm data is meaningful."""
        import analysis.cognitive_rhythm as cr_mod
        from analysis.cognitive_rhythm import CognitiveRhythm
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            days_analyzed=10,
            peak_focus_hours=[9, 10, 11],
            low_load_hours=[14, 15, 16],
            morning_bias="morning",
            best_focus_dow=2,  # Wednesday
            hourly_fdi_sparkline="▁▂▄▇█▇▅▄▃▃▂▂▂",
            dow_fdi_sparkline="▃▅▇▅▃▁▁",
            date_range="2026-03-04 → 2026-03-14",
        )
        monkeypatch.setattr(cr_mod, "compute_cognitive_rhythm", lambda *a, **kw: rhythm)
        from analysis.morning_brief import _compute_cognitive_rhythm_for_brief
        result = _compute_cognitive_rhythm_for_brief("2026-03-14")
        assert result is not None
        assert result["is_meaningful"] is True
        assert "line" in result
        assert isinstance(result["line"], str)

    def test_result_includes_peak_hours(self, monkeypatch):
        """Result must include peak_focus_hours list."""
        import analysis.cognitive_rhythm as cr_mod
        from analysis.cognitive_rhythm import CognitiveRhythm
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            days_analyzed=5,
            peak_focus_hours=[9, 10],
            morning_bias="morning",
            best_focus_dow=1,
            hourly_fdi_sparkline="▂▄▇▆▄▃▂▁",
            dow_fdi_sparkline="▃▅▇▅▃▁▁",
        )
        monkeypatch.setattr(cr_mod, "compute_cognitive_rhythm", lambda *a, **kw: rhythm)
        from analysis.morning_brief import _compute_cognitive_rhythm_for_brief
        result = _compute_cognitive_rhythm_for_brief("2026-03-14")
        assert result is not None
        assert result["peak_focus_hours"] == [9, 10]

    def test_graceful_on_exception(self, monkeypatch):
        """Any exception returns None without crashing."""
        import analysis.cognitive_rhythm as cr_mod
        monkeypatch.setattr(
            cr_mod, "compute_cognitive_rhythm",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated failure")),
        )
        from analysis.morning_brief import _compute_cognitive_rhythm_for_brief
        result = _compute_cognitive_rhythm_for_brief("2026-03-14")
        assert result is None

    def test_returns_none_when_line_is_empty(self, monkeypatch):
        """Returns None when format_rhythm_line produces empty string."""
        import analysis.cognitive_rhythm as cr_mod
        from analysis.cognitive_rhythm import CognitiveRhythm
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            days_analyzed=3,
            peak_focus_hours=[],  # no peak hours → empty line
            morning_bias=None,
            best_focus_dow=None,
        )
        monkeypatch.setattr(cr_mod, "compute_cognitive_rhythm", lambda *a, **kw: rhythm)
        from analysis.morning_brief import _compute_cognitive_rhythm_for_brief
        result = _compute_cognitive_rhythm_for_brief("2026-03-14")
        # With no peak hours and no bias, format_rhythm_line returns ""
        # The helper should return None
        assert result is None

    def test_message_contains_rhythm_line(self, monkeypatch):
        """format_morning_brief_message includes rhythm when present."""
        import analysis.cognitive_rhythm as cr_mod
        from analysis.cognitive_rhythm import CognitiveRhythm
        rhythm = CognitiveRhythm(
            is_meaningful=True,
            days_analyzed=7,
            peak_focus_hours=[9, 10, 11],
            morning_bias="morning",
            best_focus_dow=2,
            hourly_fdi_sparkline="▂▄▇▆▄▃▂▁▁▁▁▁▁",
            dow_fdi_sparkline="▃▅▇▅▃▁▁",
        )
        monkeypatch.setattr(cr_mod, "compute_cognitive_rhythm", lambda *a, **kw: rhythm)

        brief = {
            "date": "2026-03-14",
            "whoop": {
                "recovery_score": 85.0,
                "hrv_rmssd_milli": 72.0,
                "sleep_hours": 8.0,
                "sleep_performance": 88.0,
                "resting_heart_rate": 54.0,
            },
            "readiness": {"tier": "peak", "label": "Peak", "recommendation": "Go hard."},
            "yesterday": {},
            "hrv_baseline": 70.0,
            "trend_context": {},
            "cognitive_debt": None,
            "cognitive_budget": None,
            "dps_trend": None,
            "load_forecast": None,
            "tomorrow_focus_plan": None,
            "today_calendar": None,
            "cognitive_rhythm": {
                "line": "⏱ *Rhythm:* Peak focus 9am–10am · morning-biased · Best day: Wed",
                "peak_focus_hours": [9, 10, 11],
                "morning_bias": "morning",
                "best_focus_dow": "Wed",
                "hourly_fdi_sparkline": "▂▄▇▆▄▃▂▁▁▁▁▁▁",
                "days_analyzed": 7,
                "is_meaningful": True,
            },
        }
        from analysis.morning_brief import format_morning_brief_message
        msg = format_morning_brief_message(brief)
        assert "Rhythm" in msg
        assert "Peak focus" in msg

    def test_message_omits_rhythm_when_none(self):
        """format_morning_brief_message omits rhythm section when cognitive_rhythm is None."""
        brief = {
            "date": "2026-03-14",
            "whoop": {
                "recovery_score": 85.0,
                "hrv_rmssd_milli": 72.0,
                "sleep_hours": 8.0,
                "sleep_performance": 88.0,
                "resting_heart_rate": 54.0,
            },
            "readiness": {"tier": "peak", "label": "Peak", "recommendation": "Go hard."},
            "yesterday": {},
            "hrv_baseline": 70.0,
            "trend_context": {},
            "cognitive_debt": None,
            "cognitive_budget": None,
            "dps_trend": None,
            "load_forecast": None,
            "tomorrow_focus_plan": None,
            "today_calendar": None,
            "cognitive_rhythm": None,
        }
        from analysis.morning_brief import format_morning_brief_message
        msg = format_morning_brief_message(brief)
        assert "Rhythm" not in msg

    def test_compute_morning_brief_includes_rhythm_key(self, monkeypatch):
        """compute_morning_brief dict always has a 'cognitive_rhythm' key."""
        import analysis.cognitive_rhythm as cr_mod
        from analysis.cognitive_rhythm import CognitiveRhythm
        not_meaningful = CognitiveRhythm(is_meaningful=False)
        monkeypatch.setattr(cr_mod, "compute_cognitive_rhythm", lambda *a, **kw: not_meaningful)

        # Mock out all the side-effect helpers
        import analysis.morning_brief as mb_mod
        monkeypatch.setattr(mb_mod, "_compute_cdi_for_brief", lambda *a: None)
        monkeypatch.setattr(mb_mod, "_compute_dps_trend_for_brief", lambda *a: None)
        monkeypatch.setattr(mb_mod, "_compute_focus_plan_for_brief", lambda *a, **kw: None)
        monkeypatch.setattr(mb_mod, "_compute_load_forecast_for_brief", lambda *a, **kw: None)
        monkeypatch.setattr(mb_mod, "_compute_cognitive_budget_for_brief", lambda *a, **kw: None)

        from analysis.morning_brief import compute_morning_brief
        result = compute_morning_brief(
            today_date="2026-03-14",
            whoop_data={
                "recovery_score": 85.0,
                "hrv_rmssd_milli": 72.0,
                "sleep_hours": 8.0,
                "sleep_performance": 88.0,
                "resting_heart_rate": 54.0,
            },
        )
        assert "cognitive_rhythm" in result


class TestComputeConversationForBrief:
    """Tests for _compute_conversation_for_brief (v23.0)."""

    def test_returns_none_when_not_meaningful(self, monkeypatch):
        """Returns None when ConversationIntelligence.is_meaningful is False."""
        import analysis.conversation_intelligence as ci_mod
        from analysis.conversation_intelligence import ConversationIntelligence

        empty_ci = ConversationIntelligence(
            date_range="2026-03-01 → 2026-03-07",
            days_requested=7,
            days_with_data=0,
            total_speech_minutes=0.0,
            total_words=0,
            avg_speech_minutes_per_day=0.0,
            avg_words_per_day=0.0,
            avg_cognitive_density=0.0,
            peak_conversation_hour=9,
            heavy_days=[],
            light_days=[],
            language_split={},
            dominant_language="unknown",
            topic_distribution={},
            dominant_topic="unknown",
            hourly_profile={},
            daily_summaries=[],
            is_meaningful=False,
            trend_direction="stable",
            trend_description="No data.",
            insight_lines=[],
        )
        monkeypatch.setattr(ci_mod, "analyse_conversation_history", lambda **kw: empty_ci)

        from analysis.morning_brief import _compute_conversation_for_brief
        result = _compute_conversation_for_brief("2026-03-07")
        assert result is None

    def test_returns_dict_when_meaningful(self, monkeypatch):
        """Returns a dict with expected keys when CI is meaningful."""
        import analysis.conversation_intelligence as ci_mod
        from analysis.conversation_intelligence import ConversationIntelligence

        meaningful_ci = ConversationIntelligence(
            date_range="2026-03-01 → 2026-03-07",
            days_requested=7,
            days_with_data=4,
            total_speech_minutes=280.0,
            total_words=21000,
            avg_speech_minutes_per_day=70.0,
            avg_words_per_day=5250.0,
            avg_cognitive_density=0.48,
            peak_conversation_hour=10,
            heavy_days=["2026-03-05"],
            light_days=["2026-03-02"],
            language_split={"en": 6, "hu": 2},
            dominant_language="en",
            topic_distribution={"work_technical": 0.60, "personal": 0.40},
            dominant_topic="work_technical",
            hourly_profile={10: 3200, 11: 2800, 14: 1500},
            daily_summaries=[],
            is_meaningful=True,
            trend_direction="stable",
            trend_description="Conversation volume stable.",
            insight_lines=["📈 Some insight here."],
        )
        monkeypatch.setattr(ci_mod, "analyse_conversation_history", lambda **kw: meaningful_ci)

        from analysis.morning_brief import _compute_conversation_for_brief
        result = _compute_conversation_for_brief("2026-03-07")
        assert result is not None
        assert result["is_meaningful"] is True
        assert "line" in result
        assert "avg_speech_min" in result
        assert "dominant_language" in result
        assert "trend_direction" in result
        assert "days_with_data" in result

    def test_result_line_is_nonempty(self, monkeypatch):
        """The 'line' key in the result is a non-empty string."""
        import analysis.conversation_intelligence as ci_mod
        from analysis.conversation_intelligence import ConversationIntelligence

        meaningful_ci = ConversationIntelligence(
            date_range="2026-03-01 → 2026-03-07",
            days_requested=7,
            days_with_data=5,
            total_speech_minutes=350.0,
            total_words=26250,
            avg_speech_minutes_per_day=70.0,
            avg_words_per_day=5250.0,
            avg_cognitive_density=0.45,
            peak_conversation_hour=9,
            heavy_days=[],
            light_days=[],
            language_split={"en": 8, "hu": 2},
            dominant_language="en",
            topic_distribution={"work_technical": 0.70},
            dominant_topic="work_technical",
            hourly_profile={9: 4000, 10: 3200},
            daily_summaries=[],
            is_meaningful=True,
            trend_direction="increasing",
            trend_description="Conversation volume trending up.",
            insight_lines=[],
        )
        monkeypatch.setattr(ci_mod, "analyse_conversation_history", lambda **kw: meaningful_ci)

        from analysis.morning_brief import _compute_conversation_for_brief
        result = _compute_conversation_for_brief("2026-03-07")
        assert result is not None
        assert len(result["line"]) > 0

    def test_graceful_on_exception(self, monkeypatch):
        """Returns None when analyse_conversation_history raises any exception."""
        import analysis.conversation_intelligence as ci_mod
        monkeypatch.setattr(
            ci_mod,
            "analyse_conversation_history",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("transcript dir missing")),
        )

        from analysis.morning_brief import _compute_conversation_for_brief
        result = _compute_conversation_for_brief("2026-03-07")
        assert result is None

    def test_result_dominant_language_matches_ci(self, monkeypatch):
        """dominant_language in result matches the ConversationIntelligence object."""
        import analysis.conversation_intelligence as ci_mod
        from analysis.conversation_intelligence import ConversationIntelligence

        ci = ConversationIntelligence(
            date_range="2026-03-01 → 2026-03-07",
            days_requested=7,
            days_with_data=3,
            total_speech_minutes=120.0,
            total_words=9000,
            avg_speech_minutes_per_day=40.0,
            avg_words_per_day=3000.0,
            avg_cognitive_density=0.35,
            peak_conversation_hour=15,
            heavy_days=[],
            light_days=[],
            language_split={"hu": 5, "en": 1},
            dominant_language="hu",
            topic_distribution={"personal": 0.80},
            dominant_topic="personal",
            hourly_profile={15: 2000},
            daily_summaries=[],
            is_meaningful=True,
            trend_direction="decreasing",
            trend_description="Conversation volume trending down.",
            insight_lines=[],
        )
        monkeypatch.setattr(ci_mod, "analyse_conversation_history", lambda **kw: ci)

        from analysis.morning_brief import _compute_conversation_for_brief
        result = _compute_conversation_for_brief("2026-03-07")
        assert result is not None
        assert result["dominant_language"] == "hu"
        assert result["trend_direction"] == "decreasing"
        assert result["days_with_data"] == 3


class TestConversationIntelligenceInBriefMessage:
    """Tests for conversation intelligence rendering in format_morning_brief_message."""

    def _make_brief(self, ci_data=None):
        """Build a minimal valid brief dict with optional conversation_intelligence."""
        return {
            "date": "2026-03-14",
            "whoop": {
                "recovery_score": 85.0,
                "hrv_rmssd_milli": 72.0,
                "sleep_hours": 8.0,
                "sleep_performance": 88.0,
                "resting_heart_rate": 54.0,
            },
            "readiness": {"tier": "peak", "label": "Peak", "recommendation": "Go hard."},
            "yesterday": {},
            "hrv_baseline": 70.0,
            "trend_context": {},
            "cognitive_debt": None,
            "cognitive_budget": None,
            "dps_trend": None,
            "load_forecast": None,
            "tomorrow_focus_plan": None,
            "today_calendar": None,
            "cognitive_rhythm": None,
            "conversation_intelligence": ci_data,
        }

    def test_message_contains_conversation_line_when_meaningful(self):
        """format_morning_brief_message includes CI line when is_meaningful=True."""
        brief = self._make_brief({
            "line": "🗣 Conversation (4d): 72 min/day · peak 10:00 · English · →",
            "avg_speech_min": 72.0,
            "dominant_language": "en",
            "trend_direction": "stable",
            "days_with_data": 4,
            "is_meaningful": True,
        })
        from analysis.morning_brief import format_morning_brief_message
        msg = format_morning_brief_message(brief)
        assert "Conversation" in msg
        assert "72 min/day" in msg

    def test_message_omits_conversation_when_none(self):
        """format_morning_brief_message omits CI section when conversation_intelligence is None."""
        brief = self._make_brief(None)
        from analysis.morning_brief import format_morning_brief_message
        msg = format_morning_brief_message(brief)
        # There might be "Conversation" elsewhere; check the specific line format
        assert "🗣 Conversation" not in msg

    def test_message_omits_conversation_when_not_meaningful(self):
        """format_morning_brief_message omits CI section when is_meaningful=False."""
        brief = self._make_brief({
            "line": "",
            "avg_speech_min": 0.0,
            "dominant_language": "unknown",
            "trend_direction": "stable",
            "days_with_data": 0,
            "is_meaningful": False,
        })
        from analysis.morning_brief import format_morning_brief_message
        msg = format_morning_brief_message(brief)
        assert "🗣 Conversation" not in msg

    def test_compute_morning_brief_includes_conversation_key(self, monkeypatch):
        """compute_morning_brief dict always includes a 'conversation_intelligence' key."""
        import analysis.morning_brief as mb_mod
        monkeypatch.setattr(mb_mod, "_compute_cdi_for_brief", lambda *a: None)
        monkeypatch.setattr(mb_mod, "_compute_cdi_forecast_for_brief", lambda *a: None)
        monkeypatch.setattr(mb_mod, "_compute_dps_trend_for_brief", lambda *a: None)
        monkeypatch.setattr(mb_mod, "_compute_focus_plan_for_brief", lambda *a, **kw: None)
        monkeypatch.setattr(mb_mod, "_compute_load_forecast_for_brief", lambda *a, **kw: None)
        monkeypatch.setattr(mb_mod, "_compute_cognitive_budget_for_brief", lambda *a, **kw: None)
        monkeypatch.setattr(mb_mod, "_compute_cognitive_rhythm_for_brief", lambda *a: None)
        monkeypatch.setattr(mb_mod, "_compute_conversation_for_brief", lambda *a: None)
        monkeypatch.setattr(mb_mod, "_compute_ml_recovery_for_brief", lambda *a: None)
        monkeypatch.setattr(mb_mod, "_compute_sleep_focus_for_brief", lambda *a: None)
        monkeypatch.setattr(mb_mod, "_compute_weekly_pacing_for_brief", lambda *a, **kw: None)

        from analysis.morning_brief import compute_morning_brief
        result = compute_morning_brief(
            today_date="2026-03-14",
            whoop_data={
                "recovery_score": 85.0,
                "hrv_rmssd_milli": 72.0,
                "sleep_hours": 8.0,
                "sleep_performance": 88.0,
                "resting_heart_rate": 54.0,
            },
        )
        assert "conversation_intelligence" in result

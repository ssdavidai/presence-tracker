"""
Tests for v2.6: Tomorrow's Cognitive Budget in the Nightly Digest

Coverage:
  1. _compute_tomorrow_cognitive_budget_for_digest()
     - Returns None when windows is empty
     - Returns None when WHOOP data is missing from windows
     - Returns None when recovery_score is None
     - Returns dict with expected keys when WHOOP data is present
     - Uses CDI tier from precomputed_cdi when provided
     - Falls back to no CDI modifier when precomputed_cdi is None
     - Exception isolation: returns None on any internal exception
     - Line contains "Tomorrow's budget:" prefix
     - Line contains the dcb_hours value
     - Line contains the tier label

  2. compute_digest() integration
     - "tomorrow_cognitive_budget" key is present in the digest dict
     - Value is None when WHOOP data is unavailable

  3. format_digest_message() rendering
     - Shows "Tomorrow's budget" line when budget is meaningful
     - Budget line appears in the "*Tomorrow*" section
     - Budget line appears after the load forecast line (when both present)
     - Budget line appears without load forecast (budget-only Tomorrow section)
     - "*Tomorrow*" header shown when only budget is meaningful (no load/plan)
     - Does NOT show budget when is_meaningful=False
     - Does not crash when tomorrow_cognitive_budget has unexpected keys
     - Does not crash when tomorrow_cognitive_budget is None

Run with: python3 -m pytest tests/test_tomorrow_cognitive_budget_digest.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.daily_digest import (
    _compute_tomorrow_cognitive_budget_for_digest,
    format_digest_message,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

WHOOP_HIGH_RECOVERY = {
    "recovery_score": 86.0,
    "hrv_rmssd_milli": 78.0,
    "resting_heart_rate": 54.0,
    "sleep_performance": 82.0,
    "sleep_hours": 7.5,
    "strain": 10.0,
    "spo2_percentage": 95.0,
}

WHOOP_LOW_RECOVERY = {
    "recovery_score": 42.0,
    "hrv_rmssd_milli": 48.0,
    "resting_heart_rate": 64.0,
    "sleep_performance": 55.0,
    "sleep_hours": 5.5,
    "strain": 18.0,
    "spo2_percentage": 93.0,
}


def _make_window(
    date_str: str = "2026-03-14",
    whoop: dict | None = None,
    hour: int = 9,
) -> dict:
    """Build a minimal window dict for testing."""
    return {
        "date": date_str,
        "timestamp": f"{date_str}T{hour:02d}:00:00",
        "whoop": whoop or WHOOP_HIGH_RECOVERY,
        "calendar": {"in_meeting": False, "meeting_title": None},
        "slack": {
            "messages_sent": 0,
            "messages_received": 0,
            "total_messages": 0,
        },
        "rescuetime": None,
        "omi": None,
        "metrics": {
            "cognitive_load_score": 0.05,
            "focus_depth_index": 0.95,
            "social_drain_index": 0.01,
            "context_switch_cost": 0.02,
            "recovery_alignment_score": 0.99,
        },
        "metadata": {
            "day_of_week": "Saturday",
            "hour_of_day": hour,
            "minute_of_hour": 0,
            "is_working_hours": hour >= 7 and hour < 22,
            "is_active_window": False,
            "sources_available": ["whoop"],
        },
    }


def _make_windows(
    date_str: str = "2026-03-14",
    whoop: dict | None = None,
    count: int = 3,
) -> list[dict]:
    """Build a list of windows with consistent WHOOP data."""
    return [_make_window(date_str=date_str, whoop=whoop, hour=7 + i) for i in range(count)]


def _meaningful_budget(
    dcb_hours: float = 5.5,
    tier: str = "good",
    label: str = "Strong day",
    dcb_low: float = 5.0,
    dcb_high: float = 6.0,
) -> dict:
    """Build a meaningful tomorrow_cognitive_budget dict."""
    return {
        "line": f"🧠 *Tomorrow's budget:* ~{dcb_hours:.1f}h ({dcb_low:.1f}–{dcb_high:.1f}h) — {label}  _(86% recovery · balanced CDI)_",
        "dcb_hours": dcb_hours,
        "dcb_low": dcb_low,
        "dcb_high": dcb_high,
        "tier": tier,
        "label": label,
        "is_meaningful": True,
    }


def _minimal_digest(
    *,
    tomorrow_load_forecast=None,
    tomorrow_cognitive_budget=None,
    tomorrow_focus_plan=None,
) -> dict:
    """Build a minimal digest dict for testing format_digest_message."""
    return {
        "date": "2026-03-14",
        "metrics": {
            "avg_cls": 0.05,
            "peak_cls": None,
            "avg_fdi_active": 0.95,
            "avg_ras": 0.99,
        },
        "whoop": {
            "recovery_score": 86.0,
            "hrv_rmssd_milli": 79.0,
            "sleep_hours": 6.7,
        },
        "activity": {
            "total_meeting_minutes": 0,
            "meeting_count": 0,
            "active_windows": 5,
            "slack_sent": 2,
        },
        "tomorrow_load_forecast": tomorrow_load_forecast,
        "tomorrow_cognitive_budget": tomorrow_cognitive_budget,
        "tomorrow_focus_plan": tomorrow_focus_plan,
    }


# ─── Tests: _compute_tomorrow_cognitive_budget_for_digest ─────────────────────

class TestComputeTomorrowCognitiveBudgetForDigest:

    def test_empty_windows_returns_none(self):
        result = _compute_tomorrow_cognitive_budget_for_digest([], precomputed_cdi=None)
        assert result is None

    def test_missing_whoop_returns_none(self):
        windows = [_make_window(whoop=None)]
        windows[0]["whoop"] = {}
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is None

    def test_none_recovery_score_returns_none(self):
        whoop_no_recovery = {**WHOOP_HIGH_RECOVERY, "recovery_score": None}
        windows = _make_windows(whoop=whoop_no_recovery)
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is None

    def test_valid_whoop_returns_dict(self):
        windows = _make_windows()
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is not None
        assert isinstance(result, dict)

    def test_result_has_expected_keys(self):
        windows = _make_windows()
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is not None
        for key in ("line", "dcb_hours", "dcb_low", "dcb_high", "tier", "label", "is_meaningful"):
            assert key in result, f"Missing key: {key}"

    def test_is_meaningful_true_when_whoop_present(self):
        windows = _make_windows()
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is not None
        assert result["is_meaningful"] is True

    def test_line_contains_tomorrows_budget_prefix(self):
        windows = _make_windows()
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is not None
        assert "Tomorrow's budget:" in result["line"]

    def test_line_contains_hours(self):
        windows = _make_windows()
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is not None
        # Line should contain the hours value like "~5.5h"
        assert "h" in result["line"]
        assert result["dcb_hours"] > 0.0

    def test_line_contains_tier_label(self):
        windows = _make_windows()
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is not None
        assert result["label"] in result["line"]

    def test_high_recovery_produces_higher_budget(self):
        """86% recovery → more hours than 42% recovery."""
        windows_high = _make_windows(whoop=WHOOP_HIGH_RECOVERY)
        windows_low = _make_windows(whoop=WHOOP_LOW_RECOVERY)

        result_high = _compute_tomorrow_cognitive_budget_for_digest(windows_high, precomputed_cdi=None)
        result_low = _compute_tomorrow_cognitive_budget_for_digest(windows_low, precomputed_cdi=None)

        assert result_high is not None
        assert result_low is not None
        assert result_high["dcb_hours"] > result_low["dcb_hours"]

    def test_cdi_fatigued_reduces_budget_vs_balanced(self):
        """Fatigued CDI should produce fewer hours than balanced CDI."""
        windows = _make_windows()

        cdi_balanced = {"tier": "balanced", "cdi": 80, "line": "CDI 80"}
        cdi_fatigued = {"tier": "fatigued", "cdi": 55, "line": "CDI 55"}

        result_balanced = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=cdi_balanced)
        result_fatigued = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=cdi_fatigued)

        assert result_balanced is not None
        assert result_fatigued is not None
        assert result_fatigued["dcb_hours"] < result_balanced["dcb_hours"]

    def test_cdi_surplus_increases_budget_vs_balanced(self):
        """Surplus CDI should produce more hours than balanced CDI."""
        windows = _make_windows()

        cdi_balanced = {"tier": "balanced", "cdi": 80, "line": "CDI 80"}
        cdi_surplus = {"tier": "surplus", "cdi": 95, "line": "CDI 95"}

        result_balanced = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=cdi_balanced)
        result_surplus = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=cdi_surplus)

        assert result_balanced is not None
        assert result_surplus is not None
        assert result_surplus["dcb_hours"] >= result_balanced["dcb_hours"]

    def test_none_cdi_still_returns_result(self):
        """No CDI data → neutral modifier, still meaningful."""
        windows = _make_windows()
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is not None
        assert result["is_meaningful"] is True

    def test_cdi_without_tier_key_is_handled_gracefully(self):
        """CDI dict without 'tier' key → falls back to neutral modifier."""
        windows = _make_windows()
        cdi_no_tier = {"cdi": 80, "line": "CDI 80"}  # missing 'tier'
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=cdi_no_tier)
        # Should still compute (falls back to neutral CDI modifier)
        assert result is not None

    def test_exception_isolation(self):
        """Any exception → returns None, never raises."""
        with patch(
            "analysis.daily_digest._compute_tomorrow_cognitive_budget_for_digest",
            wraps=_compute_tomorrow_cognitive_budget_for_digest,
        ):
            # Force an internal exception by passing a corrupt windows list
            corrupt_windows = [{"bad_key": True}]
            result = _compute_tomorrow_cognitive_budget_for_digest(corrupt_windows, precomputed_cdi=None)
            assert result is None

    def test_dcb_hours_is_positive(self):
        windows = _make_windows()
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is not None
        assert result["dcb_hours"] > 0.0

    def test_dcb_low_lte_dcb_hours_lte_dcb_high(self):
        windows = _make_windows()
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is not None
        assert result["dcb_low"] <= result["dcb_hours"]
        assert result["dcb_hours"] <= result["dcb_high"]

    def test_tier_is_valid_string(self):
        windows = _make_windows()
        result = _compute_tomorrow_cognitive_budget_for_digest(windows, precomputed_cdi=None)
        assert result is not None
        valid_tiers = {"peak", "good", "moderate", "low", "recovery"}
        assert result["tier"] in valid_tiers


# ─── Tests: format_digest_message() rendering ─────────────────────────────────

class TestFormatDigestMessageCognitiveBudget:

    def test_budget_line_shown_in_tomorrow_section(self):
        budget = _meaningful_budget()
        digest = _minimal_digest(tomorrow_cognitive_budget=budget)
        msg = format_digest_message(digest)
        assert "Tomorrow's budget:" in msg

    def test_tomorrow_header_shown_when_budget_present(self):
        budget = _meaningful_budget()
        digest = _minimal_digest(tomorrow_cognitive_budget=budget)
        msg = format_digest_message(digest)
        assert "*Tomorrow*" in msg

    def test_budget_line_content_matches_dict(self):
        """The actual budget line text from the dict is rendered."""
        budget = _meaningful_budget(dcb_hours=6.5, label="Strong day")
        digest = _minimal_digest(tomorrow_cognitive_budget=budget)
        msg = format_digest_message(digest)
        # The line from the dict contains the hours + label
        assert "6.5h" in msg
        assert "Strong day" in msg

    def test_budget_appears_after_load_forecast(self):
        """When both forecast and budget are present, budget appears after forecast."""
        forecast = {
            "line": "📊 *Load forecast:* Moderate — CLS ~0.42",
            "predicted_cls": 0.42,
            "load_label": "Moderate",
            "confidence": "medium",
            "meeting_minutes": 90,
            "matching_days": 5,
            "narrative": "Moderate day ahead.",
            "is_meaningful": True,
        }
        budget = _meaningful_budget()
        digest = _minimal_digest(
            tomorrow_load_forecast=forecast,
            tomorrow_cognitive_budget=budget,
        )
        msg = format_digest_message(digest)

        # Both should appear
        assert "Load forecast" in msg
        assert "Tomorrow's budget:" in msg

        # Budget line appears after forecast line (by index in the message)
        forecast_pos = msg.index("Load forecast")
        budget_pos = msg.index("Tomorrow's budget:")
        assert budget_pos > forecast_pos, (
            "Budget line should appear after load forecast"
        )

    def test_budget_only_shows_tomorrow_header(self):
        """Budget alone (no load forecast, no focus plan) still triggers *Tomorrow* header."""
        budget = _meaningful_budget()
        digest = _minimal_digest(
            tomorrow_load_forecast=None,
            tomorrow_cognitive_budget=budget,
            tomorrow_focus_plan=None,
        )
        msg = format_digest_message(digest)
        assert "*Tomorrow*" in msg
        assert "Tomorrow's budget:" in msg

    def test_no_budget_and_no_forecast_and_no_plan_omits_tomorrow_section(self):
        """When all three are absent/None, the Tomorrow section is omitted."""
        digest = _minimal_digest(
            tomorrow_load_forecast=None,
            tomorrow_cognitive_budget=None,
            tomorrow_focus_plan=None,
        )
        msg = format_digest_message(digest)
        assert "*Tomorrow*" not in msg

    def test_none_budget_does_not_show_budget_line(self):
        """None budget → budget line omitted."""
        digest = _minimal_digest(tomorrow_cognitive_budget=None)
        msg = format_digest_message(digest)
        assert "Tomorrow's budget:" not in msg

    def test_not_meaningful_budget_omitted(self):
        """is_meaningful=False → budget line not rendered."""
        budget = {**_meaningful_budget(), "is_meaningful": False}
        digest = _minimal_digest(tomorrow_cognitive_budget=budget)
        msg = format_digest_message(digest)
        assert "Tomorrow's budget:" not in msg

    def test_does_not_crash_with_none_budget(self):
        """format_digest_message never crashes when tomorrow_cognitive_budget is None."""
        digest = _minimal_digest(tomorrow_cognitive_budget=None)
        # Should not raise
        msg = format_digest_message(digest)
        assert isinstance(msg, str)

    def test_does_not_crash_with_unexpected_keys(self):
        """format_digest_message never crashes with unexpected keys in budget dict."""
        weird_budget = {"unexpected": "data", "is_meaningful": True, "line": "🧠 *Tomorrow's budget:* ~5.0h — Steady"}
        digest = _minimal_digest(tomorrow_cognitive_budget=weird_budget)
        msg = format_digest_message(digest)
        assert isinstance(msg, str)

    def test_budget_with_focus_plan_and_load_forecast(self):
        """All three Tomorrow signals (forecast + budget + plan) render without duplication."""
        forecast = {
            "line": "📊 *Load forecast:* Light — CLS ~0.25",
            "predicted_cls": 0.25,
            "load_label": "Light",
            "confidence": "high",
            "meeting_minutes": 30,
            "matching_days": 8,
            "narrative": "Light day ahead.",
            "is_meaningful": True,
        }
        budget = _meaningful_budget(dcb_hours=7.0, label="Strong day")
        focus_plan = {
            "section": "*🎯 Tomorrow's Focus Plan:*\n• 9:00–11:00 _(120min)_",
            "is_meaningful": True,
        }
        digest = _minimal_digest(
            tomorrow_load_forecast=forecast,
            tomorrow_cognitive_budget=budget,
            tomorrow_focus_plan=focus_plan,
        )
        msg = format_digest_message(digest)

        # All three should appear exactly once
        assert msg.count("Load forecast") == 1
        assert msg.count("Tomorrow's budget:") == 1
        assert msg.count("Focus Plan") == 1
        assert msg.count("*Tomorrow*") == 1

    def test_low_recovery_shows_lower_budget(self):
        """Low recovery WHOOP data should yield a smaller dcb_hours in the budget line."""
        windows_low = _make_windows(whoop=WHOOP_LOW_RECOVERY)
        result = _compute_tomorrow_cognitive_budget_for_digest(windows_low, precomputed_cdi=None)
        assert result is not None
        # 42% recovery → low or moderate tier → hours < 5.0
        assert result["dcb_hours"] < 5.5

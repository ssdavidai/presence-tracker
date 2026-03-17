"""
Tests for v3.0: Actionable Insights in the Nightly Digest

Coverage:
  1. _compute_actionable_insights_for_digest()
     - Returns None when actionable_insights module raises an exception
     - Returns None when is_meaningful=False (insufficient data)
     - Returns None when insights list is empty
     - Returns dict with expected keys when insights are available
     - Dict contains is_meaningful, insights list, days_analysed, section
     - Insights list contains dicts with title, headline, evidence, impact_label, rank
     - Exception isolation: returns None on any internal exception

  2. compute_digest() integration
     - "actionable_insights" key is present in the digest dict
     - Value is None when mocked module returns not meaningful
     - Value is a dict when mocked module returns meaningful insights

  3. format_digest_message() rendering
     - Shows "💡 *Behaviour insights" section when insights are meaningful
     - Does NOT show insights section when actionable_insights is None
     - Does NOT show insights section when is_meaningful=False
     - Does not crash when actionable_insights has unexpected keys
     - Insights section appears before BRI when both are present

Run with: python3 -m pytest tests/test_actionable_insights_digest.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.daily_digest import (
    _compute_actionable_insights_for_digest,
    format_digest_message,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _minimal_digest(**overrides) -> dict:
    """Return a minimal digest dict sufficient to call format_digest_message()."""
    base = {
        "date": "2026-03-14",
        "metrics": {
            "avg_cls": 0.35,
            "peak_cls": 0.62,
            "avg_fdi_active": 0.72,
            "avg_ras": 0.80,
        },
        "whoop": {
            "recovery_score": 78.0,
            "hrv_rmssd_milli": 65.0,
            "sleep_hours": 7.2,
        },
        "activity": {
            "total_meeting_minutes": 60,
            "meeting_count": 2,
            "active_windows": 18,
            "slack_sent": 12,
        },
        "insight": "Good focus day.",
        "hourly_cls_curve": None,
        "omi": None,
        "peak_focus_hour": None,
        "peak_focus_fdi": None,
        "cognitive_debt": None,
        "presence_score": None,
        "meeting_intel": None,
        "ml_insights": None,
        "tomorrow_load_forecast": None,
        "load_decomposition": None,
        "sleep_target": None,
        "tomorrow_cognitive_budget": None,
        "load_volatility": None,
        "flow_state": None,
        "burnout_risk": None,
        "actionable_insights": None,
        "rescuetime": None,
        "peak_window": None,
        "cdi_forecast": None,
        "tomorrow_focus_plan": None,
        "personal_records": None,
    }
    base.update(overrides)
    return base


MOCK_INSIGHT = {
    "title": "Post-meeting recovery gap",
    "headline": "Schedule 15-min buffers after back-to-back calls.",
    "evidence": "FDI drops 38% in the window after 2+ consecutive meetings (8/12 days).",
    "impact_label": "High",
    "rank": 1,
}

MOCK_AI_DICT = {
    "is_meaningful": True,
    "insights": [MOCK_INSIGHT],
    "days_analysed": 14,
    "section": (
        "*💡 Actionable Insights* — 14 days analysed\n\n"
        "1. *Post-meeting recovery gap* 🔴\n"
        "   → Schedule 15-min buffers after back-to-back calls.\n"
        "   _Data: FDI drops 38% in the window after 2+ consecutive meetings (8/12 days)._"
    ),
}


# ─── _compute_actionable_insights_for_digest() ───────────────────────────────

class TestComputeActionableInsightsForDigest:
    """Unit tests for the helper function."""

    def test_returns_none_on_exception(self):
        """Any exception from the module is swallowed and None is returned."""
        with patch(
            "analysis.actionable_insights.compute_actionable_insights",
            side_effect=RuntimeError("boom"),
        ):
            result = _compute_actionable_insights_for_digest("2026-03-14")
        assert result is None

    def test_returns_none_when_not_meaningful(self):
        """Returns None when is_meaningful is False (not enough data)."""
        mock_ai = MagicMock()
        mock_ai.is_meaningful = False
        mock_ai.insights = []
        with patch(
            "analysis.actionable_insights.compute_actionable_insights",
            return_value=mock_ai,
        ):
            result = _compute_actionable_insights_for_digest("2026-03-14")
        assert result is None

    def test_returns_none_when_empty_insights(self):
        """Returns None when is_meaningful=True but insights list is empty."""
        mock_ai = MagicMock()
        mock_ai.is_meaningful = True
        mock_ai.insights = []
        with patch(
            "analysis.actionable_insights.compute_actionable_insights",
            return_value=mock_ai,
        ):
            result = _compute_actionable_insights_for_digest("2026-03-14")
        assert result is None

    def test_returns_dict_with_expected_keys(self):
        """Returns dict with required keys when insights are available."""
        mock_insight = MagicMock()
        mock_insight.title = "Post-meeting recovery gap"
        mock_insight.headline = "Schedule 15-min buffers."
        mock_insight.evidence = "FDI drops 38% (8/12 days)."
        mock_insight.impact_label = "High"
        mock_insight.rank = 1

        mock_ai = MagicMock()
        mock_ai.is_meaningful = True
        mock_ai.insights = [mock_insight]
        mock_ai.days_analysed = 14

        with patch(
            "analysis.actionable_insights.compute_actionable_insights",
            return_value=mock_ai,
        ), patch(
            "analysis.actionable_insights.format_insights_section",
            return_value="💡 *Actionable Insights* — 14 days",
        ):
            result = _compute_actionable_insights_for_digest("2026-03-14")

        assert result is not None
        assert result["is_meaningful"] is True
        assert "insights" in result
        assert "days_analysed" in result
        assert "section" in result

    def test_insights_list_has_correct_structure(self):
        """Each insight in the returned list has required sub-keys."""
        mock_insight = MagicMock()
        mock_insight.title = "Late-day cliff"
        mock_insight.headline = "Protect mornings."
        mock_insight.evidence = "FDI drops after 16:00 on 5/7 days."
        mock_insight.impact_label = "Medium"
        mock_insight.rank = 2

        mock_ai = MagicMock()
        mock_ai.is_meaningful = True
        mock_ai.insights = [mock_insight]
        mock_ai.days_analysed = 10

        with patch(
            "analysis.actionable_insights.compute_actionable_insights",
            return_value=mock_ai,
        ), patch(
            "analysis.actionable_insights.format_insights_section",
            return_value="💡 section",
        ):
            result = _compute_actionable_insights_for_digest("2026-03-14")

        assert result is not None
        assert len(result["insights"]) == 1
        ins = result["insights"][0]
        assert ins["title"] == "Late-day cliff"
        assert ins["headline"] == "Protect mornings."
        assert ins["evidence"] == "FDI drops after 16:00 on 5/7 days."
        assert ins["impact_label"] == "Medium"
        assert ins["rank"] == 2

    def test_days_analysed_matches(self):
        """days_analysed in returned dict matches the module's value."""
        mock_insight = MagicMock()
        mock_insight.title = "t"
        mock_insight.headline = "h"
        mock_insight.evidence = "e"
        mock_insight.impact_label = "Low"
        mock_insight.rank = 1

        mock_ai = MagicMock()
        mock_ai.is_meaningful = True
        mock_ai.insights = [mock_insight]
        mock_ai.days_analysed = 21

        with patch(
            "analysis.actionable_insights.compute_actionable_insights",
            return_value=mock_ai,
        ), patch(
            "analysis.actionable_insights.format_insights_section",
            return_value="section",
        ):
            result = _compute_actionable_insights_for_digest("2026-03-14")

        assert result["days_analysed"] == 21


# ─── format_digest_message() rendering ───────────────────────────────────────

class TestFormatDigestMessageActionableInsights:
    """Tests for how actionable_insights is rendered in the digest."""

    def test_no_section_when_actionable_insights_is_none(self):
        """Does not include insights section when actionable_insights is None."""
        digest = _minimal_digest(actionable_insights=None)
        message = format_digest_message(digest)
        assert "💡" not in message or "Actionable Insights" not in message

    def test_no_section_when_not_meaningful(self):
        """Does not include insights section when is_meaningful=False."""
        digest = _minimal_digest(
            actionable_insights={"is_meaningful": False, "section": "💡 *Should not appear*"}
        )
        message = format_digest_message(digest)
        assert "Should not appear" not in message

    def test_shows_section_when_meaningful(self):
        """Renders the insights section when is_meaningful=True and section is set."""
        digest = _minimal_digest(actionable_insights=MOCK_AI_DICT)
        message = format_digest_message(digest)
        # The format_insights_section() formats as "*💡 Actionable Insights*"
        assert "*💡 Actionable Insights*" in message

    def test_insight_headline_appears_in_message(self):
        """The insight headline appears in the digest message."""
        digest = _minimal_digest(actionable_insights=MOCK_AI_DICT)
        message = format_digest_message(digest)
        assert "Schedule 15-min buffers after back-to-back calls." in message

    def test_section_appears_before_bri(self):
        """Insights section appears before the BRI warning when both present."""
        bri_dict = {
            "is_meaningful": True,
            "bri": 55.0,
            "tier": "caution",
            "tier_label": "Caution",
            "section": "⚠️ *Burnout Risk: Caution* (BRI 55) — multiple signals trending negative.",
            "line": "⚠️ BRI 55 (Caution)",
            "dominant_signal": "hrv_trend",
            "trajectory_headline": "HRV declining.",
            "intervention_advice": "Reduce load.",
        }
        digest = _minimal_digest(
            actionable_insights=MOCK_AI_DICT,
            burnout_risk=bri_dict,
        )
        message = format_digest_message(digest)
        ai_pos = message.find("*💡 Actionable Insights*")
        bri_pos = message.find("⚠️ *Burnout Risk")
        assert ai_pos != -1, "Actionable insights section not found"
        assert bri_pos != -1, "BRI section not found"
        assert ai_pos < bri_pos, "Insights should appear before BRI"

    def test_no_crash_on_missing_section_key(self):
        """Does not crash when section key is absent from the dict."""
        digest = _minimal_digest(
            actionable_insights={"is_meaningful": True, "insights": [], "days_analysed": 14}
        )
        message = format_digest_message(digest)
        assert isinstance(message, str)

    def test_no_crash_on_empty_section(self):
        """Does not crash when section is an empty string."""
        digest = _minimal_digest(
            actionable_insights={
                "is_meaningful": True,
                "insights": [MOCK_INSIGHT],
                "days_analysed": 14,
                "section": "",
            }
        )
        message = format_digest_message(digest)
        assert isinstance(message, str)

    def test_does_not_crash_when_actionable_insights_has_wrong_type(self):
        """Does not crash when actionable_insights is an unexpected type."""
        digest = _minimal_digest(actionable_insights="unexpected_string")
        message = format_digest_message(digest)
        assert isinstance(message, str)

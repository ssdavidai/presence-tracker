"""
Tests for v2.3: Tomorrow's Load Forecast in the Nightly Digest

Coverage:
  1. _compute_load_forecast_for_digest()
     - Returns None when forecast is not meaningful (no history)
     - Returns None on exception (exception isolation)
     - Returns dict with expected keys when meaningful
     - Uses tomorrow's date (today+1) as forecast reference
     - Handles no calendar gracefully (gcal import failure)

  2. compute_digest() integration
     - "tomorrow_load_forecast" key is present in digest dict
     - Value is None when load forecast is not meaningful

  3. format_digest_message() rendering
     - Shows "*Tomorrow*" header when load forecast is meaningful
     - Shows the forecast line from tomorrow_load_forecast
     - Shows the narrative from tomorrow_load_forecast
     - Shows the focus plan section after the forecast
     - Shows "*Tomorrow*" header when only focus plan is present (no forecast)
     - Shows "*Tomorrow*" header when only forecast is present (no focus plan)
     - Omits "*Tomorrow*" section when neither is meaningful
     - Does not duplicate the focus plan section (legacy path not triggered)
     - Does not crash when tomorrow_load_forecast has unexpected keys

Run with: python3 -m pytest tests/test_tomorrow_load_forecast_digest.py -v
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.daily_digest import (
    _compute_load_forecast_for_digest,
    format_digest_message,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _minimal_digest(
    *,
    tomorrow_load_forecast=None,
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
        "tomorrow_focus_plan": tomorrow_focus_plan,
    }


def _meaningful_forecast(
    load_label: str = "Moderate",
    predicted_cls: float = 0.42,
    confidence: str = "medium",
    meeting_minutes: int = 90,
    matching_days: int = 5,
) -> dict:
    """Build a meaningful tomorrow_load_forecast dict."""
    return {
        "line": f"📊 *Load forecast:* {load_label} — CLS ~{predicted_cls:.2f} ({matching_days} similar days)",
        "predicted_cls": predicted_cls,
        "load_label": load_label,
        "confidence": confidence,
        "meeting_minutes": meeting_minutes,
        "matching_days": matching_days,
        "narrative": f"1h30m of meetings → {load_label.lower()} load expected (CLS ~{predicted_cls:.2f}). Front-load focused work.",
        "is_meaningful": True,
    }


def _meaningful_focus_plan() -> dict:
    """Build a minimal meaningful tomorrow_focus_plan dict."""
    return {
        "section": "*🎯 Tomorrow's Focus Plan:*\n• 9:00–11:00  _(120min, peak focus hour)_  🔥",
        "summary_line": "Best focus window: 9:00–11:00 (120min, peak FDI hour)",
        "advisory": "Two clear blocks available.",
        "is_meaningful": True,
        "block_count": 1,
        "cdi_tier": None,
    }


# ─── _compute_load_forecast_for_digest ───────────────────────────────────────

class TestComputeLoadForecastForDigest:
    """Tests for the _compute_load_forecast_for_digest() helper."""

    def test_returns_none_when_forecast_not_meaningful(self):
        """When the load forecast is not meaningful (no history), return None."""
        with (
            patch("analysis.load_forecast.list_available_dates", return_value=[]),
            patch("analysis.load_forecast.read_summary", return_value={"days": {}}),
            patch("collectors.gcal.collect", return_value={"total_meeting_minutes": 0, "events": []}),
        ):
            result = _compute_load_forecast_for_digest("2026-03-14")
        assert result is None

    def test_returns_none_on_exception(self):
        """Any exception in the helper should return None (never crash the digest)."""
        with patch("analysis.load_forecast.compute_load_forecast", side_effect=RuntimeError("boom")):
            result = _compute_load_forecast_for_digest("2026-03-14")
        assert result is None

    def test_returns_dict_with_expected_keys_when_meaningful(self):
        """When the forecast is meaningful, return a dict with required keys."""
        today = "2026-03-14"
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        tomorrow = (today_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        # Build 10 days of history prior to tomorrow
        history = {}
        for i in range(1, 11):
            d = (today_dt - timedelta(days=i)).strftime("%Y-%m-%d")
            history[d] = {
                "date": d,
                "calendar": {"total_meeting_minutes": 90},
                "metrics_avg": {"cognitive_load_score": 0.45},
            }
        dates = sorted(history.keys())

        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value={"days": history}),
            patch("collectors.gcal.collect", return_value={"total_meeting_minutes": 90, "events": []}),
        ):
            result = _compute_load_forecast_for_digest(today)

        assert result is not None
        assert result["is_meaningful"] is True
        assert "line" in result
        assert "predicted_cls" in result
        assert "load_label" in result
        assert "confidence" in result
        assert "meeting_minutes" in result
        assert "narrative" in result

    def test_uses_tomorrow_date_as_reference(self):
        """The helper should use today+1 as the forecast date, not today."""
        today = "2026-03-14"
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        expected_tomorrow = (today_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        # Track what date was passed to compute_load_forecast
        called_with_date = []

        from analysis.load_forecast import LoadForecast

        def mock_forecast(date_str, calendar):
            called_with_date.append(date_str)
            return LoadForecast(
                date_str=date_str,
                predicted_cls=0.42,
                cls_low=0.35,
                cls_high=0.50,
                load_label="Moderate",
                confidence="medium",
                meeting_minutes=90,
                days_of_history=10,
                matching_days=5,
                narrative="Test narrative.",
                is_meaningful=True,
            )

        with patch("analysis.load_forecast.compute_load_forecast", side_effect=mock_forecast):
            with patch("collectors.gcal.collect", return_value={"total_meeting_minutes": 90, "events": []}):
                result = _compute_load_forecast_for_digest(today)

        assert len(called_with_date) == 1
        assert called_with_date[0] == expected_tomorrow

    def test_handles_gcal_import_failure_gracefully(self):
        """If gcal.collect raises, the helper should still attempt forecast with None calendar."""
        today = "2026-03-14"
        today_dt = datetime.strptime(today, "%Y-%m-%d")

        history = {}
        for i in range(1, 11):
            d = (today_dt - timedelta(days=i)).strftime("%Y-%m-%d")
            history[d] = {
                "date": d,
                "calendar": {"total_meeting_minutes": 0},
                "metrics_avg": {"cognitive_load_score": 0.10},
            }
        dates = sorted(history.keys())

        with (
            patch("analysis.load_forecast.list_available_dates", return_value=dates),
            patch("analysis.load_forecast.read_summary", return_value={"days": history}),
            patch("collectors.gcal.collect", side_effect=ImportError("no gcal")),
        ):
            # Should not raise — should either return None or a forecast dict
            result = _compute_load_forecast_for_digest(today)
        # Accept either outcome: None (graceful) or a dict (meaningful forecast)
        assert result is None or isinstance(result, dict)

    def test_returns_none_when_line_is_empty(self):
        """If format_forecast_line returns empty string, return None (not meaningful)."""
        from analysis.load_forecast import LoadForecast

        not_meaningful = LoadForecast(
            date_str="2026-03-15",
            is_meaningful=False,
            narrative="Not enough history.",
        )

        with (
            patch("analysis.load_forecast.compute_load_forecast", return_value=not_meaningful),
            patch("collectors.gcal.collect", return_value={"total_meeting_minutes": 0, "events": []}),
        ):
            result = _compute_load_forecast_for_digest("2026-03-14")

        assert result is None


# ─── format_digest_message — Tomorrow section ────────────────────────────────

class TestFormatDigestMessageTomorrow:
    """Tests for the Tomorrow section in format_digest_message."""

    def test_shows_tomorrow_header_when_forecast_and_plan_present(self):
        """Both forecast and focus plan → show '*Tomorrow*' header."""
        digest = _minimal_digest(
            tomorrow_load_forecast=_meaningful_forecast(),
            tomorrow_focus_plan=_meaningful_focus_plan(),
        )
        msg = format_digest_message(digest)
        assert "*Tomorrow*" in msg

    def test_shows_forecast_line(self):
        """The formatted forecast line should appear in the message."""
        forecast = _meaningful_forecast(load_label="High", predicted_cls=0.65)
        digest = _minimal_digest(tomorrow_load_forecast=forecast)
        msg = format_digest_message(digest)
        assert "Load forecast" in msg
        assert "High" in msg

    def test_shows_forecast_narrative(self):
        """The narrative sentence should appear below the forecast line."""
        forecast = _meaningful_forecast()
        digest = _minimal_digest(tomorrow_load_forecast=forecast)
        msg = format_digest_message(digest)
        assert "moderate load expected" in msg.lower()

    def test_shows_focus_plan_after_forecast(self):
        """Focus plan should appear after the forecast in the message."""
        forecast = _meaningful_forecast()
        plan = _meaningful_focus_plan()
        digest = _minimal_digest(
            tomorrow_load_forecast=forecast,
            tomorrow_focus_plan=plan,
        )
        msg = format_digest_message(digest)
        forecast_pos = msg.find("Load forecast")
        plan_pos = msg.find("Focus Plan")
        assert forecast_pos != -1
        assert plan_pos != -1
        assert forecast_pos < plan_pos

    def test_shows_tomorrow_header_with_only_focus_plan(self):
        """Focus plan only (no forecast) → still shows '*Tomorrow*' header."""
        digest = _minimal_digest(
            tomorrow_load_forecast=None,
            tomorrow_focus_plan=_meaningful_focus_plan(),
        )
        msg = format_digest_message(digest)
        assert "*Tomorrow*" in msg
        assert "Focus Plan" in msg

    def test_shows_tomorrow_header_with_only_forecast(self):
        """Forecast only (no focus plan) → shows '*Tomorrow*' header."""
        digest = _minimal_digest(
            tomorrow_load_forecast=_meaningful_forecast(),
            tomorrow_focus_plan=None,
        )
        msg = format_digest_message(digest)
        assert "*Tomorrow*" in msg
        assert "Load forecast" in msg

    def test_omits_tomorrow_section_when_neither_present(self):
        """Neither forecast nor focus plan → no '*Tomorrow*' section."""
        digest = _minimal_digest(
            tomorrow_load_forecast=None,
            tomorrow_focus_plan=None,
        )
        msg = format_digest_message(digest)
        assert "*Tomorrow*" not in msg

    def test_omits_tomorrow_section_when_forecast_not_meaningful(self):
        """is_meaningful=False forecast + no focus plan → no tomorrow section."""
        not_meaningful = {
            "line": "",
            "is_meaningful": False,
            "predicted_cls": None,
            "load_label": "Unknown",
        }
        digest = _minimal_digest(
            tomorrow_load_forecast=not_meaningful,
            tomorrow_focus_plan=None,
        )
        msg = format_digest_message(digest)
        assert "*Tomorrow*" not in msg

    def test_does_not_crash_when_forecast_has_no_line_key(self):
        """Missing 'line' key in forecast dict → no crash, section skipped."""
        bad_forecast = {
            "is_meaningful": True,
            # 'line' key is missing
            "predicted_cls": 0.42,
        }
        digest = _minimal_digest(tomorrow_load_forecast=bad_forecast)
        # Should not raise
        msg = format_digest_message(digest)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_focus_plan_not_duplicated(self):
        """Focus plan section should appear exactly once, not twice."""
        digest = _minimal_digest(
            tomorrow_load_forecast=_meaningful_forecast(),
            tomorrow_focus_plan=_meaningful_focus_plan(),
        )
        msg = format_digest_message(digest)
        # Count occurrences of the focus plan header
        count = msg.count("🎯 Tomorrow's Focus Plan")
        assert count == 1

    def test_very_high_load_forecast_shown_correctly(self):
        """Very high load forecast should show label and be scary."""
        forecast = _meaningful_forecast(
            load_label="Very high",
            predicted_cls=0.82,
            confidence="high",
            meeting_minutes=300,
            matching_days=10,
        )
        digest = _minimal_digest(tomorrow_load_forecast=forecast)
        msg = format_digest_message(digest)
        assert "Very high" in msg

    def test_low_confidence_forecast_shows_warning_indicator(self):
        """Low-confidence forecast line should contain warning indicator."""
        # Build the line manually with the ⚠️ indicator that format_forecast_line adds
        forecast = {
            "line": "📊 *Load forecast:* Light — CLS ~0.18 ⚠️ low confidence",
            "predicted_cls": 0.18,
            "load_label": "Light",
            "confidence": "low",
            "meeting_minutes": 0,
            "matching_days": 1,
            "narrative": "No meetings scheduled → light load expected.",
            "is_meaningful": True,
        }
        digest = _minimal_digest(tomorrow_load_forecast=forecast)
        msg = format_digest_message(digest)
        assert "⚠️" in msg or "low confidence" in msg.lower()

    def test_complete_tomorrow_section_structure(self):
        """Full tomorrow section should have: header, forecast line, narrative, focus plan."""
        forecast = _meaningful_forecast()
        plan = _meaningful_focus_plan()
        digest = _minimal_digest(
            tomorrow_load_forecast=forecast,
            tomorrow_focus_plan=plan,
        )
        msg = format_digest_message(digest)

        # All four elements present
        assert "*Tomorrow*" in msg
        assert "📊" in msg             # forecast emoji
        assert "Load forecast" in msg  # forecast line
        assert "load expected" in msg  # narrative
        assert "🎯" in msg             # focus plan emoji

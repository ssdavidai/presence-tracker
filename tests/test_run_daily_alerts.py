"""
Tests for anomaly alert integration in scripts/run_daily.py (v1.7).

These tests verify that:
1. run() calls send_anomaly_alerts() when alerts=True (default) and the
   anomaly engine reports a triggered condition.
2. run() does NOT call send_anomaly_alerts() when alerts=False.
3. run() does NOT call send_anomaly_alerts() when no anomalies are triggered.
4. An exception in anomaly alerts does not crash run() (non-fatal).
5. backfill.py passes alerts=False to run() so historical days don't spam Slack.

All external I/O (collectors, store writes, Slack) is mocked so these tests
run with no credentials and no side-effects.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Shared mock data ─────────────────────────────────────────────────────────

_WHOOP = {
    "recovery_score": 78.0,
    "hrv_rmssd_milli": 65.0,
    "resting_heart_rate": 54.0,
    "sleep_performance": 82.0,
    "sleep_hours": 7.8,
    "strain": 13.0,
    "spo2_percentage": 95.0,
}

_CALENDAR = {
    "events": [],
    "event_count": 0,
    "total_meeting_minutes": 0,
    "max_concurrent_attendees": 0,
}

_SUMMARY = {
    "date": "2026-01-01",
    "whoop": {"recovery_score": 78.0, "hrv_rmssd_milli": 65.0},
    "metrics_avg": {"cognitive_load_score": 0.35, "focus_depth_index": 0.65},
    "calendar": {"total_meeting_minutes": 30},
}

_ANOMALY_TRIGGERED = {
    "date": "2026-01-01",
    "any_triggered": True,
    "alerts": {
        "cls_spike": {"triggered": True, "today_cls": 0.85, "baseline_mean": 0.30, "baseline_std": 0.10},
        "fdi_collapse": False,
        "recovery_streak": False,
    },
}

_ANOMALY_CLEAN = {
    "date": "2026-01-01",
    "any_triggered": False,
    "alerts": {
        "cls_spike": False,
        "fdi_collapse": False,
        "recovery_streak": False,
    },
}


def _make_windows():
    """Return a minimal list of 96 mock windows."""
    from engine.chunker import build_windows
    return build_windows(
        date_str="2026-01-01",
        whoop_data=_WHOOP,
        calendar_data=_CALENDAR,
        slack_windows={},
        rescuetime_windows={},
        omi_windows={},
    )


# ─── Fixtures / patchers ──────────────────────────────────────────────────────

def _base_patches():
    """Return a dict of patch targets used across multiple tests."""
    return {
        "whoop": "collectors.whoop.collect",
        "gcal": "collectors.gcal.collect",
        "slack": "collectors.slack.collect",
        "rt": "collectors.rescuetime.collect",
        "omi": "collectors.omi.collect",
        "write_day": "engine.store.write_day",
        "update_summary": "engine.store.update_summary",
        "day_exists": "engine.store.day_exists",
        "slack_log": "scripts.run_daily._slack_log",
        "digest": "analysis.daily_digest.send_daily_digest",
        "dashboard": "analysis.dashboard.generate_dashboard",
    }


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestAnomalyAlertsCalled:
    """send_anomaly_alerts IS called when alerts=True and a trigger fires."""

    def test_alerts_called_when_triggered(self):
        """With alerts=True (default) and a triggered anomaly, send_anomaly_alerts is called."""
        windows = _make_windows()

        with patch("collectors.whoop.collect", return_value=_WHOOP), \
             patch("collectors.gcal.collect", return_value=_CALENDAR), \
             patch("collectors.slack.collect", return_value={}), \
             patch("collectors.rescuetime.collect", return_value={}), \
             patch("collectors.omi.collect", return_value={}), \
             patch("engine.store.day_exists", return_value=False), \
             patch("engine.store.write_day", return_value=Path("/tmp/2026-01-01.jsonl")), \
             patch("engine.store.update_summary"), \
             patch("scripts.run_daily._slack_log"), \
             patch("analysis.daily_digest.send_daily_digest", return_value=True), \
             patch("analysis.dashboard.generate_dashboard", return_value=Path("/tmp/dash.html")), \
             patch("analysis.anomaly_alerts.check_anomalies", return_value=_ANOMALY_TRIGGERED) as mock_check, \
             patch("analysis.anomaly_alerts.send_anomaly_alerts", return_value=1) as mock_send, \
             patch("config.RESCUETIME_API_KEY", ""):

            from scripts.run_daily import run
            run("2026-01-01", alerts=True)

        mock_check.assert_called_once_with("2026-01-01")
        mock_send.assert_called_once_with("2026-01-01")

    def test_alerts_called_by_default(self):
        """alerts defaults to True — send_anomaly_alerts fires without explicit flag."""
        windows = _make_windows()

        with patch("collectors.whoop.collect", return_value=_WHOOP), \
             patch("collectors.gcal.collect", return_value=_CALENDAR), \
             patch("collectors.slack.collect", return_value={}), \
             patch("collectors.rescuetime.collect", return_value={}), \
             patch("collectors.omi.collect", return_value={}), \
             patch("engine.store.day_exists", return_value=False), \
             patch("engine.store.write_day", return_value=Path("/tmp/2026-01-01.jsonl")), \
             patch("engine.store.update_summary"), \
             patch("scripts.run_daily._slack_log"), \
             patch("analysis.daily_digest.send_daily_digest", return_value=True), \
             patch("analysis.dashboard.generate_dashboard", return_value=Path("/tmp/dash.html")), \
             patch("analysis.anomaly_alerts.check_anomalies", return_value=_ANOMALY_TRIGGERED), \
             patch("analysis.anomaly_alerts.send_anomaly_alerts", return_value=1) as mock_send, \
             patch("config.RESCUETIME_API_KEY", ""):

            from scripts.run_daily import run
            run("2026-01-01")  # no alerts kwarg → defaults to True

        mock_send.assert_called_once_with("2026-01-01")


class TestAnomalyAlertsNotCalled:
    """send_anomaly_alerts is NOT called in these cases."""

    def test_alerts_skipped_when_alerts_false(self):
        """With alerts=False, neither check_anomalies nor send_anomaly_alerts is called."""
        with patch("collectors.whoop.collect", return_value=_WHOOP), \
             patch("collectors.gcal.collect", return_value=_CALENDAR), \
             patch("collectors.slack.collect", return_value={}), \
             patch("collectors.rescuetime.collect", return_value={}), \
             patch("collectors.omi.collect", return_value={}), \
             patch("engine.store.day_exists", return_value=False), \
             patch("engine.store.write_day", return_value=Path("/tmp/2026-01-01.jsonl")), \
             patch("engine.store.update_summary"), \
             patch("scripts.run_daily._slack_log"), \
             patch("analysis.daily_digest.send_daily_digest", return_value=True), \
             patch("analysis.dashboard.generate_dashboard", return_value=Path("/tmp/dash.html")), \
             patch("analysis.anomaly_alerts.check_anomalies") as mock_check, \
             patch("analysis.anomaly_alerts.send_anomaly_alerts") as mock_send, \
             patch("config.RESCUETIME_API_KEY", ""):

            from scripts.run_daily import run
            run("2026-01-01", alerts=False)

        mock_check.assert_not_called()
        mock_send.assert_not_called()

    def test_send_not_called_when_no_trigger(self):
        """check_anomalies is called but send_anomaly_alerts is NOT called when nothing triggered."""
        with patch("collectors.whoop.collect", return_value=_WHOOP), \
             patch("collectors.gcal.collect", return_value=_CALENDAR), \
             patch("collectors.slack.collect", return_value={}), \
             patch("collectors.rescuetime.collect", return_value={}), \
             patch("collectors.omi.collect", return_value={}), \
             patch("engine.store.day_exists", return_value=False), \
             patch("engine.store.write_day", return_value=Path("/tmp/2026-01-01.jsonl")), \
             patch("engine.store.update_summary"), \
             patch("scripts.run_daily._slack_log"), \
             patch("analysis.daily_digest.send_daily_digest", return_value=True), \
             patch("analysis.dashboard.generate_dashboard", return_value=Path("/tmp/dash.html")), \
             patch("analysis.anomaly_alerts.check_anomalies", return_value=_ANOMALY_CLEAN) as mock_check, \
             patch("analysis.anomaly_alerts.send_anomaly_alerts") as mock_send, \
             patch("config.RESCUETIME_API_KEY", ""):

            from scripts.run_daily import run
            run("2026-01-01", alerts=True)

        mock_check.assert_called_once_with("2026-01-01")
        mock_send.assert_not_called()


class TestAnomalyAlertNonFatal:
    """An exception in the anomaly engine must not crash run()."""

    def test_anomaly_exception_is_non_fatal(self):
        """If check_anomalies raises, run() still completes and returns the summary."""
        with patch("collectors.whoop.collect", return_value=_WHOOP), \
             patch("collectors.gcal.collect", return_value=_CALENDAR), \
             patch("collectors.slack.collect", return_value={}), \
             patch("collectors.rescuetime.collect", return_value={}), \
             patch("collectors.omi.collect", return_value={}), \
             patch("engine.store.day_exists", return_value=False), \
             patch("engine.store.write_day", return_value=Path("/tmp/2026-01-01.jsonl")), \
             patch("engine.store.update_summary"), \
             patch("scripts.run_daily._slack_log"), \
             patch("analysis.daily_digest.send_daily_digest", return_value=True), \
             patch("analysis.dashboard.generate_dashboard", return_value=Path("/tmp/dash.html")), \
             patch("analysis.anomaly_alerts.check_anomalies", side_effect=RuntimeError("store unavailable")), \
             patch("config.RESCUETIME_API_KEY", ""):

            from scripts.run_daily import run
            summary = run("2026-01-01", alerts=True, quiet=True)

        # run() should still return a valid summary dict
        assert isinstance(summary, dict)
        assert "metrics_avg" in summary

    def test_send_exception_is_non_fatal(self):
        """If send_anomaly_alerts raises after check passes, run() still completes."""
        with patch("collectors.whoop.collect", return_value=_WHOOP), \
             patch("collectors.gcal.collect", return_value=_CALENDAR), \
             patch("collectors.slack.collect", return_value={}), \
             patch("collectors.rescuetime.collect", return_value={}), \
             patch("collectors.omi.collect", return_value={}), \
             patch("engine.store.day_exists", return_value=False), \
             patch("engine.store.write_day", return_value=Path("/tmp/2026-01-01.jsonl")), \
             patch("engine.store.update_summary"), \
             patch("scripts.run_daily._slack_log"), \
             patch("analysis.daily_digest.send_daily_digest", return_value=True), \
             patch("analysis.dashboard.generate_dashboard", return_value=Path("/tmp/dash.html")), \
             patch("analysis.anomaly_alerts.check_anomalies", return_value=_ANOMALY_TRIGGERED), \
             patch("analysis.anomaly_alerts.send_anomaly_alerts", side_effect=ConnectionError("gateway down")), \
             patch("config.RESCUETIME_API_KEY", ""):

            from scripts.run_daily import run
            summary = run("2026-01-01", alerts=True, quiet=True)

        assert isinstance(summary, dict)
        assert "metrics_avg" in summary


class TestBackfillDisablesAlerts:
    """scripts/backfill.py must pass alerts=False to run() on each date."""

    def test_backfill_passes_alerts_false(self):
        """backfill.run() is called with alerts=False so no Slack DMs fire during historical fill."""
        with patch("scripts.run_daily.run") as mock_run, \
             patch("scripts.backfill.run", wraps=None):

            # Patch sys.argv so argparse uses --days 1
            import sys
            old_argv = sys.argv
            sys.argv = ["backfill.py", "--days", "1"]
            try:
                # Import with fresh state
                import importlib
                import scripts.backfill as backfill_mod
                importlib.reload(backfill_mod)
                # Call main() with patched run
                with patch("scripts.backfill.run") as mock_bf_run:
                    mock_bf_run.return_value = {}
                    backfill_mod.main()
                    # Every call must have alerts=False
                    for c in mock_bf_run.call_args_list:
                        kwargs = c.kwargs if c.kwargs else {}
                        # alerts=False can come as positional or keyword arg
                        # run(date_str, force=..., alerts=False)
                        assert kwargs.get("alerts") is False, \
                            f"backfill called run() without alerts=False: {c}"
            finally:
                sys.argv = old_argv


class TestRunSignature:
    """run() accepts the new alerts kwarg cleanly."""

    def test_run_accepts_alerts_true(self):
        """run(alerts=True) does not raise a TypeError."""
        with patch("collectors.whoop.collect", return_value=_WHOOP), \
             patch("collectors.gcal.collect", return_value=_CALENDAR), \
             patch("collectors.slack.collect", return_value={}), \
             patch("collectors.rescuetime.collect", return_value={}), \
             patch("collectors.omi.collect", return_value={}), \
             patch("engine.store.day_exists", return_value=False), \
             patch("engine.store.write_day", return_value=Path("/tmp/2026-01-01.jsonl")), \
             patch("engine.store.update_summary"), \
             patch("scripts.run_daily._slack_log"), \
             patch("analysis.daily_digest.send_daily_digest", return_value=True), \
             patch("analysis.dashboard.generate_dashboard", return_value=Path("/tmp/dash.html")), \
             patch("analysis.anomaly_alerts.check_anomalies", return_value=_ANOMALY_CLEAN), \
             patch("analysis.anomaly_alerts.send_anomaly_alerts", return_value=0), \
             patch("config.RESCUETIME_API_KEY", ""):

            from scripts.run_daily import run
            result = run("2026-01-01", force=True, quiet=True, alerts=True)
            assert isinstance(result, dict)

    def test_run_accepts_alerts_false(self):
        """run(alerts=False) does not raise a TypeError."""
        with patch("collectors.whoop.collect", return_value=_WHOOP), \
             patch("collectors.gcal.collect", return_value=_CALENDAR), \
             patch("collectors.slack.collect", return_value={}), \
             patch("collectors.rescuetime.collect", return_value={}), \
             patch("collectors.omi.collect", return_value={}), \
             patch("engine.store.day_exists", return_value=False), \
             patch("engine.store.write_day", return_value=Path("/tmp/2026-01-01.jsonl")), \
             patch("engine.store.update_summary"), \
             patch("scripts.run_daily._slack_log"), \
             patch("analysis.daily_digest.send_daily_digest", return_value=True), \
             patch("analysis.dashboard.generate_dashboard", return_value=Path("/tmp/dash.html")), \
             patch("analysis.anomaly_alerts.check_anomalies") as mock_check, \
             patch("config.RESCUETIME_API_KEY", ""):

            from scripts.run_daily import run
            result = run("2026-01-01", force=True, quiet=True, alerts=False)
            assert isinstance(result, dict)
            mock_check.assert_not_called()

"""
Tests for temporal wiring completeness — verifies that all workflows and
activities are properly registered in worker.py and schedules.py.

v29 — wire MidDayCheckInWorkflow into the Temporal worker and schedule it.

These tests are static/import-level: they verify code structure without
connecting to a live Temporal server or any external APIs.
"""

import importlib
import sys
from pathlib import Path

import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Workflow registration completeness ──────────────────────────────────────

class TestWorkerRegistration:
    """Verify temporal/worker.py registers all expected workflows and activities."""

    def _get_worker_source(self) -> str:
        worker_path = Path(__file__).parent.parent / "temporal" / "worker.py"
        return worker_path.read_text()

    def test_midday_checkin_workflow_imported(self):
        """MidDayCheckInWorkflow must be imported in worker.py."""
        src = self._get_worker_source()
        assert "MidDayCheckInWorkflow" in src, (
            "worker.py must import MidDayCheckInWorkflow from temporal.workflows"
        )

    def test_midday_checkin_workflow_in_worker_list(self):
        """MidDayCheckInWorkflow must appear in the Worker(workflows=[...]) list."""
        src = self._get_worker_source()
        # The workflow must appear inside the Worker() constructor call
        assert "MidDayCheckInWorkflow," in src or "MidDayCheckInWorkflow\n" in src, (
            "MidDayCheckInWorkflow must be in the Worker workflows list"
        )

    def test_send_midday_checkin_activity_imported(self):
        """send_midday_checkin activity must be imported in worker.py."""
        src = self._get_worker_source()
        assert "send_midday_checkin" in src, (
            "worker.py must import send_midday_checkin from temporal.activities"
        )

    def test_generate_weekly_dashboard_activity_imported(self):
        """generate_weekly_dashboard activity must be imported in worker.py."""
        src = self._get_worker_source()
        assert "generate_weekly_dashboard" in src, (
            "worker.py must import generate_weekly_dashboard from temporal.activities"
        )

    def test_all_five_workflows_registered(self):
        """All five core workflows must appear in worker.py."""
        src = self._get_worker_source()
        expected = [
            "MorningBriefWorkflow",
            "MidDayCheckInWorkflow",
            "DailyIngestionWorkflow",
            "WeeklyAnalysisWorkflow",
            "MonthlyMLRetrainWorkflow",
        ]
        for wf in expected:
            assert wf in src, f"worker.py must reference {wf}"

    def test_all_core_activities_registered(self):
        """All core activities must appear in worker.py."""
        src = self._get_worker_source()
        expected = [
            "ingest_day",
            "send_morning_readiness_brief",
            "send_midday_checkin",
            "generate_daily_dashboard",
            "generate_weekly_dashboard",
            "run_anomaly_alerts",
            "run_weekly_intuition",
            "run_weekly_summary",
            "retrain_ml_models",
            "notify_slack_presence",
        ]
        for act in expected:
            assert act in src, f"worker.py must reference activity: {act}"


# ─── Schedule registration completeness ──────────────────────────────────────

class TestScheduleRegistration:
    """Verify temporal/schedules.py defines all expected schedules."""

    def _get_schedules_source(self) -> str:
        schedules_path = Path(__file__).parent.parent / "temporal" / "schedules.py"
        return schedules_path.read_text()

    def test_midday_schedule_defined(self):
        """schedules.py must define a schedule for the midday check-in."""
        src = self._get_schedules_source()
        assert "presence-midday-checkin" in src, (
            "schedules.py must define the 'presence-midday-checkin' schedule"
        )

    def test_midday_checkin_workflow_imported_in_schedules(self):
        """MidDayCheckInWorkflow must be imported in schedules.py."""
        src = self._get_schedules_source()
        assert "MidDayCheckInWorkflow" in src, (
            "schedules.py must import MidDayCheckInWorkflow"
        )

    def test_midday_cron_expression(self):
        """Midday schedule must use 12:00 UTC (= 13:00 Budapest CET)."""
        src = self._get_schedules_source()
        # The midday cron should fire at 12:00 UTC
        assert "0 12" in src, (
            "Midday schedule cron must be '0 12 ...' (12:00 UTC = 13:00 Budapest)"
        )

    def test_midday_weekdays_only(self):
        """Midday check-in should only run on weekdays (Mon–Fri = 1-5)."""
        src = self._get_schedules_source()
        assert "1-5" in src, (
            "Midday schedule should run Mon–Fri only (cron field '1-5')"
        )

    def test_all_four_base_schedules_present(self):
        """All four existing schedules must still be present."""
        src = self._get_schedules_source()
        expected_ids = [
            "presence-morning-brief",
            "presence-daily-ingestion",
            "presence-weekly-analysis",
            "presence-monthly-ml-retrain",
        ]
        for sid in expected_ids:
            assert sid in src, f"schedules.py must still define schedule: {sid}"


# ─── run_midday.py script existence ──────────────────────────────────────────

class TestRunMiddayScript:
    """Verify scripts/run_midday.py exists and is importable."""

    def test_script_exists(self):
        """scripts/run_midday.py must exist."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_midday.py"
        assert script_path.exists(), "scripts/run_midday.py must exist"

    def test_script_has_argparse(self):
        """run_midday.py must use argparse for CLI interface."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_midday.py"
        src = script_path.read_text()
        assert "argparse" in src, "run_midday.py must use argparse"

    def test_script_has_dry_run_flag(self):
        """run_midday.py must support --dry-run flag."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_midday.py"
        src = script_path.read_text()
        assert "--dry-run" in src, "run_midday.py must support --dry-run flag"

    def test_script_has_json_flag(self):
        """run_midday.py must support --json flag for machine-readable output."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_midday.py"
        src = script_path.read_text()
        assert "--json" in src, "run_midday.py must support --json flag"

    def test_script_imports_midday_checkin(self):
        """run_midday.py must import from analysis.midday_checkin."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_midday.py"
        src = script_path.read_text()
        assert "midday_checkin" in src, (
            "run_midday.py must import from analysis.midday_checkin"
        )


# ─── Activities module completeness ──────────────────────────────────────────

class TestActivitiesModule:
    """Verify temporal/activities.py defines all expected activities."""

    def _get_activities_source(self) -> str:
        activities_path = Path(__file__).parent.parent / "temporal" / "activities.py"
        return activities_path.read_text()

    def test_send_midday_checkin_defined(self):
        """send_midday_checkin must be defined as a Temporal activity."""
        src = self._get_activities_source()
        assert "async def send_midday_checkin" in src, (
            "temporal/activities.py must define send_midday_checkin"
        )

    def test_send_midday_checkin_has_activity_defn(self):
        """send_midday_checkin must be decorated with @activity.defn."""
        src = self._get_activities_source()
        # Find the decorator before the function definition
        idx = src.find("async def send_midday_checkin")
        assert idx > 0
        preceding = src[max(0, idx - 200):idx]
        assert "@activity.defn" in preceding, (
            "send_midday_checkin must be decorated with @activity.defn"
        )

    def test_generate_weekly_dashboard_defined(self):
        """generate_weekly_dashboard must be defined as a Temporal activity."""
        src = self._get_activities_source()
        assert "async def generate_weekly_dashboard" in src, (
            "temporal/activities.py must define generate_weekly_dashboard"
        )


# ─── Workflow module completeness ─────────────────────────────────────────────

class TestWorkflowsModule:
    """Verify temporal/workflows.py defines MidDayCheckInWorkflow correctly."""

    def _get_workflows_source(self) -> str:
        workflows_path = Path(__file__).parent.parent / "temporal" / "workflows.py"
        return workflows_path.read_text()

    def test_midday_workflow_defined(self):
        """MidDayCheckInWorkflow must be defined in temporal/workflows.py."""
        src = self._get_workflows_source()
        assert "class MidDayCheckInWorkflow" in src, (
            "temporal/workflows.py must define MidDayCheckInWorkflow"
        )

    def test_midday_workflow_has_defn_decorator(self):
        """MidDayCheckInWorkflow must be decorated with @workflow.defn."""
        src = self._get_workflows_source()
        idx = src.find("class MidDayCheckInWorkflow")
        assert idx > 0
        preceding = src[max(0, idx - 100):idx]
        assert "@workflow.defn" in preceding, (
            "MidDayCheckInWorkflow must be decorated with @workflow.defn"
        )

    def test_midday_workflow_imports_send_midday_checkin(self):
        """MidDayCheckInWorkflow must use the send_midday_checkin activity."""
        src = self._get_workflows_source()
        assert "send_midday_checkin" in src, (
            "temporal/workflows.py must reference send_midday_checkin"
        )

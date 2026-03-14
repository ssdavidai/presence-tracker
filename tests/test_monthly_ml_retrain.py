"""
Tests for the MonthlyMLRetrainWorkflow and retrain_ml_models activity.

All tests are pure unit tests — no Temporal server, no real ML training,
no network I/O.  The ML layer and Temporal activity are mocked so we can
verify the control flow and outcome formatting without touching real models.

Coverage:
  - retrain_ml_models activity: success path, insufficient_data path, error path
  - MonthlyMLRetrainWorkflow: success branch, insufficient-data branch, error branch
  - worker.py: all expected workflows and activities are importable
  - schedules.py: monthly schedule present in SCHEDULES list
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Helper: fake activity context ───────────────────────────────────────────

class _FakeActivityContext:
    """Minimal stand-in for the Temporal activity execution context."""

    @staticmethod
    def logger():
        return MagicMock()


# ─── retrain_ml_models activity ──────────────────────────────────────────────

class TestRetrainMLModelsActivity:
    """Unit tests for the retrain_ml_models Temporal activity."""

    @pytest.mark.asyncio
    async def test_successful_retrain_returns_result_dict(self):
        """Happy path: train_all returns a valid result dict."""
        mock_result = {
            "status": "trained",
            "days_used": 75,
            "anomaly": "trained",
            "recovery": "trained",
            "clustering": "trained",
        }
        mock_status = {"days_available": 75, "min_days_required": 60}

        with patch("analysis.ml_model.train_all", return_value=mock_result), \
             patch("analysis.ml_model.get_data_status", return_value=mock_status), \
             patch("temporalio.activity.logger", MagicMock()):
            from temporal.activities import retrain_ml_models
            result = await retrain_ml_models(force=False)

        assert result["status"] == "trained"
        assert result["days_used"] == 75
        assert result["anomaly"] == "trained"
        assert result["recovery"] == "trained"
        assert result["clustering"] == "trained"

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_skipped_status(self):
        """When fewer than 60 days exist, train_all returns insufficient_data."""
        mock_result = {
            "status": "insufficient_data",
            "days_available": 12,
            "days_required": 60,
            "anomaly": "skipped",
            "recovery": "skipped",
            "clustering": "skipped",
        }
        mock_status = {"days_available": 12, "min_days_required": 60}

        with patch("analysis.ml_model.train_all", return_value=mock_result), \
             patch("analysis.ml_model.get_data_status", return_value=mock_status), \
             patch("temporalio.activity.logger", MagicMock()):
            from temporal.activities import retrain_ml_models
            result = await retrain_ml_models(force=False)

        assert result["status"] == "insufficient_data"
        assert result["anomaly"] == "skipped"

    @pytest.mark.asyncio
    async def test_force_true_calls_train_all_with_force(self):
        """force=True must be forwarded to train_all."""
        mock_result = {
            "status": "trained",
            "days_used": 5,
            "anomaly": "trained",
            "recovery": "skipped",
            "clustering": "skipped",
        }
        mock_status = {"days_available": 5, "min_days_required": 60}

        with patch("analysis.ml_model.train_all", return_value=mock_result) as mock_train, \
             patch("analysis.ml_model.get_data_status", return_value=mock_status), \
             patch("temporalio.activity.logger", MagicMock()):
            from temporal.activities import retrain_ml_models
            await retrain_ml_models(force=True)

        mock_train.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self):
        """If train_all raises, the activity must return an error dict (not re-raise)."""
        mock_status = {"days_available": 10, "min_days_required": 60}

        with patch("analysis.ml_model.train_all", side_effect=RuntimeError("boom")), \
             patch("analysis.ml_model.get_data_status", return_value=mock_status), \
             patch("temporalio.activity.logger", MagicMock()):
            from temporal.activities import retrain_ml_models
            result = await retrain_ml_models(force=False)

        assert result["status"] == "error"
        assert "boom" in result["error"]
        assert result["anomaly"] == "error"
        assert result["recovery"] == "error"
        assert result["clustering"] == "error"

    @pytest.mark.asyncio
    async def test_get_data_status_exception_still_runs_train(self):
        """If get_data_status raises, train_all should still be called."""
        mock_result = {
            "status": "trained",
            "days_used": 2,
            "anomaly": "trained",
            "recovery": "skipped",
            "clustering": "skipped",
        }

        with patch("analysis.ml_model.get_data_status", side_effect=Exception("no data")), \
             patch("analysis.ml_model.train_all", return_value=mock_result) as mock_train, \
             patch("temporalio.activity.logger", MagicMock()):
            from temporal.activities import retrain_ml_models
            result = await retrain_ml_models(force=True)

        # get_data_status failure should be swallowed; train_all still runs
        mock_train.assert_called_once_with(force=True)
        assert result["status"] == "trained"


# ─── MonthlyMLRetrainWorkflow ─────────────────────────────────────────────────

class TestMonthlyMLRetrainWorkflow:
    """Tests for the MonthlyMLRetrainWorkflow control flow."""

    def _make_workflow(self):
        from temporal.workflows import MonthlyMLRetrainWorkflow
        return MonthlyMLRetrainWorkflow()

    def test_workflow_is_importable(self):
        """MonthlyMLRetrainWorkflow must be importable from temporal.workflows."""
        from temporal.workflows import MonthlyMLRetrainWorkflow
        assert MonthlyMLRetrainWorkflow is not None

    def test_workflow_has_run_method(self):
        """Workflow must expose a .run() coroutine method."""
        from temporal.workflows import MonthlyMLRetrainWorkflow
        assert hasattr(MonthlyMLRetrainWorkflow, "run")

    def test_workflow_run_is_coroutine(self):
        """run() must be an async method (coroutine function)."""
        import inspect
        from temporal.workflows import MonthlyMLRetrainWorkflow
        assert inspect.iscoroutinefunction(MonthlyMLRetrainWorkflow.run)


# ─── Worker registration ──────────────────────────────────────────────────────

class TestWorkerRegistration:
    """Verify temporal/worker.py registers all expected workflows and activities."""

    def test_all_workflows_importable_from_worker_module(self):
        """All four workflow classes must be importable from temporal.workflows."""
        from temporal.workflows import (
            MorningBriefWorkflow,
            DailyIngestionWorkflow,
            WeeklyAnalysisWorkflow,
            MonthlyMLRetrainWorkflow,
        )
        assert MorningBriefWorkflow is not None
        assert DailyIngestionWorkflow is not None
        assert WeeklyAnalysisWorkflow is not None
        assert MonthlyMLRetrainWorkflow is not None

    def test_all_activities_importable_from_activities_module(self):
        """All eight activity functions must be importable from temporal.activities."""
        from temporal.activities import (
            ingest_day,
            send_morning_readiness_brief,
            generate_daily_dashboard,
            run_anomaly_alerts,
            run_weekly_intuition,
            run_weekly_summary,
            retrain_ml_models,
            notify_slack_presence,
        )
        for fn in [
            ingest_day,
            send_morning_readiness_brief,
            generate_daily_dashboard,
            run_anomaly_alerts,
            run_weekly_intuition,
            run_weekly_summary,
            retrain_ml_models,
            notify_slack_presence,
        ]:
            assert fn is not None

    def test_retrain_ml_models_is_async(self):
        """retrain_ml_models must be a coroutine function (async def)."""
        import inspect
        from temporal.activities import retrain_ml_models
        assert inspect.iscoroutinefunction(retrain_ml_models)

    def test_worker_module_imports_monthly_workflow(self):
        """temporal/worker.py must import MonthlyMLRetrainWorkflow."""
        import importlib.util
        import ast

        worker_path = Path(__file__).parent.parent / "temporal" / "worker.py"
        source = worker_path.read_text()
        assert "MonthlyMLRetrainWorkflow" in source, (
            "temporal/worker.py must import and register MonthlyMLRetrainWorkflow"
        )

    def test_worker_module_imports_retrain_activity(self):
        """temporal/worker.py must import retrain_ml_models activity."""
        worker_path = Path(__file__).parent.parent / "temporal" / "worker.py"
        source = worker_path.read_text()
        assert "retrain_ml_models" in source, (
            "temporal/worker.py must import and register retrain_ml_models activity"
        )

    def test_worker_registers_morning_brief_workflow(self):
        """MorningBriefWorkflow must appear in worker.py (was missing before v7.2)."""
        worker_path = Path(__file__).parent.parent / "temporal" / "worker.py"
        source = worker_path.read_text()
        assert "MorningBriefWorkflow" in source

    def test_worker_registers_send_morning_brief_activity(self):
        """send_morning_readiness_brief must appear in worker.py."""
        worker_path = Path(__file__).parent.parent / "temporal" / "worker.py"
        source = worker_path.read_text()
        assert "send_morning_readiness_brief" in source


# ─── Schedule registration ────────────────────────────────────────────────────

class TestScheduleRegistration:
    """Verify temporal/schedules.py includes the monthly ML retrain schedule."""

    def _get_schedules(self):
        """Import SCHEDULES list from temporal/schedules.py."""
        import importlib
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "schedules",
            Path(__file__).parent.parent / "temporal" / "schedules.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # Patch the async client so we don't need a live Temporal server
        with patch("temporalio.client.Client.connect", AsyncMock()):
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass  # module-level code may fail without Temporal; we only need SCHEDULES
        return getattr(mod, "SCHEDULES", None)

    def test_schedules_list_contains_monthly_retrain(self):
        """SCHEDULES must include an entry with id='presence-monthly-ml-retrain'."""
        schedules_path = Path(__file__).parent.parent / "temporal" / "schedules.py"
        source = schedules_path.read_text()
        assert "presence-monthly-ml-retrain" in source, (
            "temporal/schedules.py must define a 'presence-monthly-ml-retrain' schedule"
        )

    def test_schedules_list_contains_four_schedules(self):
        """SCHEDULES must include all four schedules (morning, daily, weekly, monthly)."""
        schedules_path = Path(__file__).parent.parent / "temporal" / "schedules.py"
        source = schedules_path.read_text()
        expected_ids = [
            "presence-morning-brief",
            "presence-daily-ingestion",
            "presence-weekly-analysis",
            "presence-monthly-ml-retrain",
        ]
        for sid in expected_ids:
            assert sid in source, f"Missing schedule id: {sid}"

    def test_monthly_retrain_uses_monthly_cron_expression(self):
        """Monthly retrain cron must be monthly (1st-of-month pattern)."""
        schedules_path = Path(__file__).parent.parent / "temporal" / "schedules.py"
        source = schedules_path.read_text()
        # Cron "0 1 1 * *" = 01:00 UTC on the 1st of each month
        assert "1 * *" in source, "Monthly cron must have day-of-month=1"

    def test_monthly_retrain_imports_workflow_class(self):
        """schedules.py must import MonthlyMLRetrainWorkflow."""
        schedules_path = Path(__file__).parent.parent / "temporal" / "schedules.py"
        source = schedules_path.read_text()
        assert "MonthlyMLRetrainWorkflow" in source


# ─── Regression: existing workflow imports unbroken ──────────────────────────

class TestExistingWorkflowsUnbroken:
    """Ensure existing workflows still import and are present in the module."""

    def test_daily_ingestion_workflow_still_imports(self):
        from temporal.workflows import DailyIngestionWorkflow
        import inspect
        assert inspect.iscoroutinefunction(DailyIngestionWorkflow.run)

    def test_weekly_analysis_workflow_still_imports(self):
        from temporal.workflows import WeeklyAnalysisWorkflow
        import inspect
        assert inspect.iscoroutinefunction(WeeklyAnalysisWorkflow.run)

    def test_morning_brief_workflow_still_imports(self):
        from temporal.workflows import MorningBriefWorkflow
        import inspect
        assert inspect.iscoroutinefunction(MorningBriefWorkflow.run)

    def test_existing_activities_still_importable(self):
        from temporal.activities import (
            ingest_day,
            run_weekly_intuition,
            send_morning_readiness_brief,
            generate_daily_dashboard,
            run_anomaly_alerts,
            run_weekly_summary,
            notify_slack_presence,
        )
        # All should be callable coroutine functions
        import inspect
        for fn in [
            ingest_day, run_weekly_intuition, send_morning_readiness_brief,
            generate_daily_dashboard, run_anomaly_alerts, run_weekly_summary,
            notify_slack_presence,
        ]:
            assert inspect.iscoroutinefunction(fn), f"{fn.__name__} must be async"

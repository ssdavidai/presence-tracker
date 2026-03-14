"""
Presence Tracker — Temporal Workflows

DailyIngestionWorkflow: runs nightly at 23:45 Budapest time
WeeklyAnalysisWorkflow: runs Sunday at 21:00 Budapest time
"""

from datetime import timedelta, datetime
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from temporal.activities import (
        ingest_day,
        run_weekly_intuition,
        run_weekly_summary,
        send_morning_readiness_brief,
        notify_slack_presence,
        generate_daily_dashboard,
        run_anomaly_alerts,
    )

RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=30),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
)

DEFAULT_TIMEOUT = timedelta(minutes=30)


@workflow.defn
class DailyIngestionWorkflow:
    """
    Nightly presence data ingestion.

    Schedule: 23:45 Budapest time, daily
    """

    @workflow.run
    async def run(self, date_str: str = None) -> str:
        if not date_str:
            # Use current date in Budapest time
            date_str = datetime.now().strftime("%Y-%m-%d")

        workflow.logger.info(f"Starting daily ingestion for {date_str}")

        try:
            summary = await workflow.execute_activity(
                ingest_day,
                args=[date_str, False],
                start_to_close_timeout=DEFAULT_TIMEOUT,
                retry_policy=RETRY_POLICY,
            )

            recovery = summary.get("whoop", {}).get("recovery_score", "?")
            avg_cls = summary.get("metrics_avg", {}).get("cognitive_load_score", 0)
            meeting_mins = summary.get("calendar", {}).get("total_meeting_minutes", 0)

            # Generate HTML dashboard after data is ingested
            dashboard_path = await workflow.execute_activity(
                generate_daily_dashboard,
                args=[date_str],
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )

            # Run anomaly alerts — sends Slack DM if thresholds exceeded
            anomaly_result = await workflow.execute_activity(
                run_anomaly_alerts,
                args=[date_str],
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            triggered = [
                k for k, v in anomaly_result.get("alerts", {}).items() if v
            ]

            msg = (
                f"[PRESENCE] Daily ingestion complete — {date_str}\n"
                f"Recovery: {recovery}% | Avg CLS: {avg_cls:.2f} | Meetings: {meeting_mins}min"
                + (f"\nDashboard: {dashboard_path}" if dashboard_path else "")
                + (f"\nAnomalies: {', '.join(triggered)}" if triggered else "")
            )

            await workflow.execute_activity(
                notify_slack_presence,
                args=[msg],
                start_to_close_timeout=timedelta(seconds=30),
            )

            return f"OK: {date_str}"

        except Exception as e:
            error_msg = f"[PRESENCE] Daily ingestion FAILED for {date_str}: {str(e)}"
            workflow.logger.error(error_msg)
            await workflow.execute_activity(
                notify_slack_presence,
                args=[error_msg],
                start_to_close_timeout=timedelta(seconds=30),
            )
            raise


@workflow.defn
class MorningBriefWorkflow:
    """
    Morning readiness brief — send David a Slack DM with WHOOP readiness.

    Schedule: 07:00 Budapest time, daily
    """

    @workflow.run
    async def run(self, date_str: str = None) -> str:
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        workflow.logger.info(f"Starting morning brief for {date_str}")

        try:
            ok = await workflow.execute_activity(
                send_morning_readiness_brief,
                args=[date_str],
                start_to_close_timeout=DEFAULT_TIMEOUT,
                retry_policy=RETRY_POLICY,
            )

            status = "sent" if ok else "failed"
            await workflow.execute_activity(
                notify_slack_presence,
                args=[f"[PRESENCE] Morning brief {status} — {date_str}"],
                start_to_close_timeout=timedelta(seconds=30),
            )

            return f"Morning brief {status}: {date_str}"

        except Exception as e:
            error_msg = f"[PRESENCE] Morning brief FAILED for {date_str}: {str(e)}"
            workflow.logger.error(error_msg)
            await workflow.execute_activity(
                notify_slack_presence,
                args=[error_msg],
                start_to_close_timeout=timedelta(seconds=30),
            )
            raise


@workflow.defn
class WeeklyAnalysisWorkflow:
    """
    Weekly presence analysis.

    Runs two parallel tracks:
    1. Deterministic weekly summary — week-over-week metric deltas (no LLM)
    2. Alfred Intuition report — LLM-powered pattern interpretation

    Both are sent as Slack DMs to David.  The summary runs first so David
    gets the numbers even if the AI report is slow or fails.

    Schedule: Sunday 21:00 Budapest time
    """

    @workflow.run
    async def run(self, date_str: str = None) -> str:
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        workflow.logger.info(f"Starting weekly analysis for week ending {date_str}")

        # ── Step 1: Deterministic weekly summary (fast, no LLM) ──────────
        summary_ok = await workflow.execute_activity(
            run_weekly_summary,
            args=[date_str],
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # ── Step 2: AI Intuition report (LLM, slower) ────────────────────
        intuition_ok = await workflow.execute_activity(
            run_weekly_intuition,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        status = "complete" if (summary_ok or intuition_ok) else "failed"
        detail_parts = []
        if summary_ok:
            detail_parts.append("summary ✓")
        else:
            detail_parts.append("summary ✗")
        if intuition_ok:
            detail_parts.append("intuition ✓")
        else:
            detail_parts.append("intuition ✗")

        log_msg = f"[PRESENCE] Weekly analysis {status} — {', '.join(detail_parts)}"
        await workflow.execute_activity(
            notify_slack_presence,
            args=[log_msg],
            start_to_close_timeout=timedelta(seconds=30),
        )

        return f"Weekly analysis {status}: {date_str}"

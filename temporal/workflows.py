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
        send_morning_readiness_brief,
        notify_slack_presence,
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

            msg = (
                f"[PRESENCE] Daily ingestion complete — {date_str}\n"
                f"Recovery: {recovery}% | Avg CLS: {avg_cls:.2f} | Meetings: {meeting_mins}min"
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
    Weekly Alfred Intuition pattern analysis.

    Schedule: Sunday 21:00 Budapest time
    """

    @workflow.run
    async def run(self) -> str:
        workflow.logger.info("Starting weekly intuition analysis")

        success = await workflow.execute_activity(
            run_weekly_intuition,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        status = "complete" if success else "failed"
        await workflow.execute_activity(
            notify_slack_presence,
            args=[f"[PRESENCE] Weekly analysis {status}"],
            start_to_close_timeout=timedelta(seconds=30),
        )

        return f"Weekly analysis {status}"

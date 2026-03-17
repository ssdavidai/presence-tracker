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
        send_midday_checkin,
        send_evening_winddown,
        notify_slack_presence,
        generate_daily_dashboard,
        generate_weekly_dashboard,
        run_anomaly_alerts,
        retrain_ml_models,
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
class MidDayCheckInWorkflow:
    """
    Midday cognitive check-in — brief Slack pulse at 13:00 Budapest.

    Fills the 6-hour gap between the morning brief (07:00) and the nightly
    digest (23:45).  Ingests partial-day data (00:00–13:00), computes morning
    load signals (CLS, FDI, SDI, meetings) and an afternoon recommendation,
    then sends a 4–6 line Slack DM to David.

    The workflow silently skips when fewer than 3 active morning windows exist
    (e.g. very quiet mornings or days with no JSONL data yet) — this is not
    treated as an error.

    Schedule: 13:00 Budapest time, daily (Mon–Fri)
    """

    @workflow.run
    async def run(self, date_str: str = None) -> str:
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        workflow.logger.info(f"Starting midday check-in for {date_str}")

        try:
            ok = await workflow.execute_activity(
                send_midday_checkin,
                args=[date_str],
                start_to_close_timeout=DEFAULT_TIMEOUT,
                retry_policy=RETRY_POLICY,
            )

            status = "sent" if ok else "skipped"
            await workflow.execute_activity(
                notify_slack_presence,
                args=[f"[PRESENCE] Midday check-in {status} — {date_str}"],
                start_to_close_timeout=timedelta(seconds=30),
            )

            return f"Midday check-in {status}: {date_str}"

        except Exception as e:
            error_msg = f"[PRESENCE] Midday check-in FAILED for {date_str}: {str(e)}"
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

        # ── Step 2: Weekly HTML dashboard ─────────────────────────────────
        weekly_dashboard_path = await workflow.execute_activity(
            generate_weekly_dashboard,
            args=[date_str],
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # ── Step 3: AI Intuition report (LLM, slower) ────────────────────
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
        if weekly_dashboard_path:
            detail_parts.append("dashboard ✓")

        log_msg = f"[PRESENCE] Weekly analysis {status} — {', '.join(detail_parts)}"
        await workflow.execute_activity(
            notify_slack_presence,
            args=[log_msg],
            start_to_close_timeout=timedelta(seconds=30),
        )

        return f"Weekly analysis {status}: {date_str}"


@workflow.defn
class MonthlyMLRetrainWorkflow:
    """
    Monthly ML model retraining.

    Retrains the three scikit-learn models (Isolation Forest, Random Forest,
    KMeans) on all accumulated JSONL data.  Runs on the 1st of each month at
    02:00 Budapest time — low-traffic window, after the previous month's data
    has been fully ingested.

    Design notes:
    - Always passes force=True so the retrain runs even before 60 days have
      accumulated.  Early models are noisy but still useful as baselines;
      they are replaced each month as data grows.
    - A Slack log is sent regardless of outcome so David knows whether the
      monthly retrain succeeded or failed.
    - The workflow is idempotent: re-triggering it manually is safe.

    Schedule: 1st of month, 01:00 UTC = 02:00 Budapest (CET/UTC+1).
    """

    @workflow.run
    async def run(self, force: bool = True) -> str:
        workflow.logger.info("Starting monthly ML model retraining")

        try:
            result = await workflow.execute_activity(
                retrain_ml_models,
                args=[force],
                start_to_close_timeout=timedelta(minutes=15),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(minutes=1),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(minutes=10),
                    maximum_attempts=3,
                ),
            )

            status = result.get("status", "unknown")
            days = result.get("days_used", result.get("days_available", "?"))
            anomaly_st = result.get("anomaly", "?")
            recovery_st = result.get("recovery", "?")
            cluster_st = result.get("clustering", "?")

            if status == "insufficient_data":
                msg = (
                    f"[PRESENCE] Monthly ML retrain — insufficient data "
                    f"({days} days, need 60). Skipped."
                )
            elif status == "error":
                msg = (
                    f"[PRESENCE] Monthly ML retrain FAILED — "
                    f"{result.get('error', 'unknown error')}"
                )
            else:
                msg = (
                    f"[PRESENCE] Monthly ML retrain complete — {days} days of data\n"
                    f"anomaly={anomaly_st} | recovery={recovery_st} | clusters={cluster_st}"
                )

            await workflow.execute_activity(
                notify_slack_presence,
                args=[msg],
                start_to_close_timeout=timedelta(seconds=30),
            )

            workflow.logger.info(f"ML retrain done: {status}")
            return f"ML retrain {status}: {days} days"

        except Exception as e:
            error_msg = f"[PRESENCE] Monthly ML retrain FAILED: {str(e)}"
            workflow.logger.error(error_msg)
            await workflow.execute_activity(
                notify_slack_presence,
                args=[error_msg],
                start_to_close_timeout=timedelta(seconds=30),
            )
            raise


@workflow.defn
class EveningWindDownWorkflow:
    """
    Evening wind-down signal — end-of-workday cognitive wrap-up.

    Fills the gap between the midday check-in (13:00) and the nightly
    digest (23:45). Fires at end-of-business and answers:
      1. How did today go? (day type: PRODUCTIVE / DEEP / REACTIVE / etc.)
      2. Was it balanced? (load arc: front-loaded / back-loaded / even)
      3. What should I do now? (concrete wind-down recommendation)

    The workflow silently skips when fewer than 3 active workday windows
    exist (e.g. weekend, public holiday, or very light day) — not an error.

    Schedule: 17:00 UTC = 18:00 Budapest (CET/UTC+1), daily
    """

    @workflow.run
    async def run(self, date_str: str = None) -> str:
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        workflow.logger.info(f"Starting evening wind-down for {date_str}")

        try:
            ok = await workflow.execute_activity(
                send_evening_winddown,
                args=[date_str],
                start_to_close_timeout=DEFAULT_TIMEOUT,
                retry_policy=RETRY_POLICY,
            )

            status = "sent" if ok else "skipped"
            await workflow.execute_activity(
                notify_slack_presence,
                args=[f"[PRESENCE] Evening wind-down {status} — {date_str}"],
                start_to_close_timeout=timedelta(seconds=30),
            )

            return f"Evening wind-down {status}: {date_str}"

        except Exception as e:
            error_msg = f"[PRESENCE] Evening wind-down FAILED for {date_str}: {str(e)}"
            workflow.logger.error(error_msg)
            await workflow.execute_activity(
                notify_slack_presence,
                args=[error_msg],
                start_to_close_timeout=timedelta(seconds=30),
            )
            raise

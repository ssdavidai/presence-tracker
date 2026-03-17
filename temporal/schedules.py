#!/usr/bin/env python3
"""
Presence Tracker — Register Temporal Schedules

Run once to create (or update) all Temporal schedules.
Safe to re-run — will update existing schedules.

Usage:
    # From project root with Alfred's venv:
    source ~/clawd/temporal-workflows/.venv/bin/activate
    python3 temporal/schedules.py
"""

import asyncio
import logging
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleSpec,
    ScheduleState,
    SchedulePolicy,
    ScheduleOverlapPolicy,
)

from config import TEMPORAL_ADDRESS, TASK_QUEUE
from temporal.workflows import (
    DailyIngestionWorkflow,
    WeeklyAnalysisWorkflow,
    MorningBriefWorkflow,
    MidDayCheckInWorkflow,
    EveningWindDownWorkflow,
    MonthlyMLRetrainWorkflow,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCHEDULES = [
    {
        "id": "presence-morning-brief",
        # 06:00 UTC = 07:00 Budapest (CET/UTC+1).
        # WHOOP posts overnight data by ~6am local time, so 7am is safe.
        "cron": "0 6 * * *",
        "workflow": MorningBriefWorkflow,
        "workflow_id": "presence-morning-brief-workflow",
        "description": "Morning readiness brief — WHOOP readiness + day planning recommendation",
    },
    {
        "id": "presence-midday-checkin",
        # 12:00 UTC = 13:00 Budapest (CET/UTC+1).
        # Fires mid-afternoon after 4–5 hours of workday data are in JSONL.
        # Gracefully skips quiet mornings (< 3 active windows).
        "cron": "0 12 * * 1-5",
        "workflow": MidDayCheckInWorkflow,
        "workflow_id": "presence-midday-checkin-workflow",
        "description": "Midday cognitive check-in — morning load pulse + afternoon recommendation (Mon–Fri)",
    },
    {
        "id": "presence-evening-winddown",
        # 17:00 UTC = 18:00 Budapest (CET/UTC+1).
        # Fires at end-of-workday after 8–10 hours of data are available.
        # Classifies the day type and sends a wind-down recommendation.
        # Gracefully skips quiet days (< 3 active workday windows).
        "cron": "0 17 * * *",
        "workflow": EveningWindDownWorkflow,
        "workflow_id": "presence-evening-winddown-workflow",
        "description": "Evening wind-down — day type classification + wind-down recommendation",
    },
    {
        "id": "presence-daily-ingestion",
        # 22:45 UTC = 23:45 Budapest (CET/UTC+1). In CEST (UTC+2): adjust manually.
        "cron": "45 22 * * *",
        "workflow": DailyIngestionWorkflow,
        "workflow_id": "presence-daily-ingestion-workflow",
        "description": "Daily presence data ingestion (WHOOP + Calendar + Slack)",
    },
    {
        "id": "presence-weekly-analysis",
        # 20:00 UTC = 21:00 Budapest (CET). Sunday only.
        "cron": "0 20 * * 0",
        "workflow": WeeklyAnalysisWorkflow,
        "workflow_id": "presence-weekly-analysis-workflow",
        "description": "Weekly Alfred Intuition pattern analysis",
    },
    {
        "id": "presence-monthly-ml-retrain",
        # 01:00 UTC = 02:00 Budapest (CET/UTC+1). 1st of each month.
        # Low-traffic window; runs after the previous month is fully ingested.
        # force=True so the retrain always runs even before 60 days have accumulated.
        "cron": "0 1 1 * *",
        "workflow": MonthlyMLRetrainWorkflow,
        "workflow_id": "presence-monthly-ml-retrain-workflow",
        "description": "Monthly ML model retraining (Isolation Forest + Random Forest + KMeans)",
    },
]


async def register_schedules():
    client = await Client.connect(TEMPORAL_ADDRESS)

    for cfg in SCHEDULES:
        schedule_id = cfg["id"]
        logger.info(f"Registering schedule: {schedule_id} ({cfg['cron']})")

        schedule = Schedule(
            action=ScheduleActionStartWorkflow(
                cfg["workflow"].run,
                id=cfg["workflow_id"],
                task_queue=TASK_QUEUE,
            ),
            spec=ScheduleSpec(
                cron_expressions=[cfg["cron"]],
            ),
            state=ScheduleState(
                note=cfg.get("description", ""),
            ),
            policy=SchedulePolicy(
                overlap=ScheduleOverlapPolicy.SKIP,
            ),
        )

        try:
            await client.create_schedule(schedule_id, schedule)
            logger.info(f"  Created: {schedule_id}")
        except Exception as e:
            err = str(e).lower()
            if "already exists" in err or "already started" in err:
                handle = client.get_schedule_handle(schedule_id)
                await handle.update(lambda s, new=schedule: new)
                logger.info(f"  Updated: {schedule_id}")
            else:
                logger.error(f"  Failed to register {schedule_id}: {e}")
                raise

    logger.info("All schedules registered.")


if __name__ == "__main__":
    asyncio.run(register_schedules())

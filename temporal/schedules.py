#!/usr/bin/env python3
"""
Presence Tracker — Register Temporal Schedules

Run once to create (or update) all Temporal schedules.
Safe to re-run — will update existing schedules.

Usage:
    python3 temporal/schedules.py
"""

import asyncio
import logging
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from temporalio.client import Client, Schedule, ScheduleActionStartWorkflow
from temporalio.client import ScheduleSpec, ScheduleCronSpec, ScheduleState
from temporalio.common import RetryPolicy

from config import TEMPORAL_ADDRESS, TASK_QUEUE, DAILY_CRON, WEEKLY_CRON
from temporal.workflows import DailyIngestionWorkflow, WeeklyAnalysisWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCHEDULES = [
    {
        "id": "presence-daily-ingestion",
        "cron": DAILY_CRON,         # 23:45 daily Budapest time
        "workflow": DailyIngestionWorkflow,
        "args": [],
        "description": "Daily presence data ingestion (WHOOP + Calendar + Slack)",
        "timezone": "Europe/Budapest",
    },
    {
        "id": "presence-weekly-analysis",
        "cron": WEEKLY_CRON,        # Sunday 21:00 Budapest time
        "workflow": WeeklyAnalysisWorkflow,
        "args": [],
        "description": "Weekly Alfred Intuition pattern analysis",
        "timezone": "Europe/Budapest",
    },
]


async def register_schedules():
    client = await Client.connect(TEMPORAL_ADDRESS)

    for sched_config in SCHEDULES:
        schedule_id = sched_config["id"]
        logger.info(f"Registering schedule: {schedule_id}")

        schedule = Schedule(
            action=ScheduleActionStartWorkflow(
                sched_config["workflow"].run,
                *sched_config["args"],
                id=f"{schedule_id}-workflow",
                task_queue=TASK_QUEUE,
            ),
            spec=ScheduleSpec(
                cron_expressions=[sched_config["cron"]],
                timezone=sched_config.get("timezone", "Europe/Budapest"),
            ),
            state=ScheduleState(
                note=sched_config.get("description", ""),
            ),
        )

        try:
            await client.create_schedule(schedule_id, schedule)
            logger.info(f"  Created: {schedule_id}")
        except Exception as e:
            if "already exists" in str(e).lower() or "already registered" in str(e).lower():
                # Update existing
                handle = client.get_schedule_handle(schedule_id)
                await handle.update(lambda _: schedule)
                logger.info(f"  Updated: {schedule_id}")
            else:
                logger.error(f"  Failed to register {schedule_id}: {e}")
                raise

    logger.info("All schedules registered.")


if __name__ == "__main__":
    asyncio.run(register_schedules())

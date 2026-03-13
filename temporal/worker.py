#!/usr/bin/env python3
"""
Presence Tracker — Temporal Worker

Run this to start processing presence tracking workflows.
Connects to the local Temporal server and polls the 'presence-tracker' task queue.

Usage:
    python3 temporal/worker.py

Or with the run script:
    ./run_worker.sh
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from temporalio.client import Client
from temporalio.worker import Worker

from config import TEMPORAL_ADDRESS, TASK_QUEUE
from temporal.activities import ingest_day, run_weekly_intuition, notify_slack_presence
from temporal.workflows import DailyIngestionWorkflow, WeeklyAnalysisWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info(f"Connecting to Temporal at {TEMPORAL_ADDRESS}")
    client = await Client.connect(TEMPORAL_ADDRESS)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DailyIngestionWorkflow, WeeklyAnalysisWorkflow],
        activities=[ingest_day, run_weekly_intuition, notify_slack_presence],
    )

    logger.info(f"Starting worker on task queue: {TASK_QUEUE}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Presence Tracker — Temporal Worker

Run this to start processing presence tracking workflows.
Connects to the local Temporal server and polls the 'presence-tracker' task queue.

Usage:
    python3 temporal/worker.py

Or with the run script:
    ./run_worker.sh

Registered workflows:
  - MorningBriefWorkflow        (07:00 Budapest — WHOOP readiness brief)
  - DailyIngestionWorkflow      (23:45 Budapest — collect + metrics + digest)
  - WeeklyAnalysisWorkflow      (Sunday 21:00 Budapest — intuition + summary)
  - MonthlyMLRetrainWorkflow    (1st of month 02:00 Budapest — ML model update)

Registered activities:
  - ingest_day                  (daily data collection + metrics + JSONL write)
  - send_morning_readiness_brief (morning Slack DM)
  - generate_daily_dashboard    (HTML dashboard generation)
  - run_anomaly_alerts          (CLS spike / FDI collapse / RAS streak checks)
  - run_weekly_intuition        (LLM weekly pattern report)
  - run_weekly_summary          (deterministic week-over-week summary)
  - retrain_ml_models           (monthly scikit-learn model retraining)
  - notify_slack_presence       (log message to #alfred-logs)
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from temporalio.client import Client
from temporalio.worker import Worker

from config import TEMPORAL_ADDRESS, TASK_QUEUE
from temporal.activities import (
    ingest_day,
    run_weekly_intuition,
    run_weekly_summary,
    send_morning_readiness_brief,
    notify_slack_presence,
    generate_daily_dashboard,
    run_anomaly_alerts,
    retrain_ml_models,
)
from temporal.workflows import (
    DailyIngestionWorkflow,
    WeeklyAnalysisWorkflow,
    MorningBriefWorkflow,
    MonthlyMLRetrainWorkflow,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info(f"Connecting to Temporal at {TEMPORAL_ADDRESS}")
    client = await Client.connect(TEMPORAL_ADDRESS)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[
            MorningBriefWorkflow,
            DailyIngestionWorkflow,
            WeeklyAnalysisWorkflow,
            MonthlyMLRetrainWorkflow,
        ],
        activities=[
            ingest_day,
            send_morning_readiness_brief,
            generate_daily_dashboard,
            run_anomaly_alerts,
            run_weekly_intuition,
            run_weekly_summary,
            retrain_ml_models,
            notify_slack_presence,
        ],
    )

    logger.info(f"Starting worker on task queue: {TASK_QUEUE}")
    logger.info(
        f"Workflows: {[w.__name__ for w in worker._workflows.values()]}"
        if hasattr(worker, '_workflows') else "Worker ready."
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

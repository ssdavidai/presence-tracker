"""
Presence Tracker — Temporal Activities

Each activity does exactly one thing and is safe to retry.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from temporalio import activity

# Add project root to path (needed when run by Temporal worker)
sys.path.insert(0, str(Path(__file__).parent.parent))


@activity.defn
async def ingest_day(date_str: str, force: bool = False) -> dict:
    """Run the full daily ingestion pipeline."""
    from scripts.run_daily import run
    return run(date_str, force=force, quiet=True)


@activity.defn
async def run_weekly_intuition() -> bool:
    """Run the weekly Alfred Intuition analysis."""
    from analysis.intuition import run_weekly_analysis
    return run_weekly_analysis()


@activity.defn
async def notify_slack_presence(message: str) -> bool:
    """Send a message to #alfred-logs."""
    import urllib.request
    from config import GATEWAY_URL, GATEWAY_TOKEN, SLACK_LOGS_CHANNEL

    try:
        headers = {
            "Authorization": f"Bearer {GATEWAY_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "tool": "message",
            "args": {
                "action": "send",
                "channel": "slack",
                "target": SLACK_LOGS_CHANNEL,
                "message": message,
            }
        }).encode()
        req = urllib.request.Request(
            f"{GATEWAY_URL}/tools/invoke",
            data=payload,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        activity.logger.error(f"Slack notify failed: {e}")
        return False

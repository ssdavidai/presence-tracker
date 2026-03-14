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
async def send_morning_readiness_brief(date_str: str) -> bool:
    """Send the morning readiness brief to David's Slack DM."""
    from analysis.morning_brief import send_morning_brief
    return send_morning_brief(date_str)


@activity.defn
async def generate_daily_dashboard(date_str: str) -> str:
    """Generate the daily HTML presence dashboard.

    Returns the path to the generated HTML file.
    Runs after ingest_day so the data is already written.
    """
    from analysis.dashboard import generate_dashboard
    try:
        path = generate_dashboard(date_str)
        activity.logger.info(f"Dashboard generated: {path}")
        return str(path)
    except Exception as e:
        activity.logger.error(f"Dashboard generation failed: {e}")
        return ""


@activity.defn
async def run_anomaly_alerts(date_str: str) -> dict:
    """Run multi-source anomaly checks and send Slack DM if anything triggered.

    Returns the check_anomalies() result dict (safe to log / inspect).
    """
    from analysis.anomaly_alerts import check_anomalies, send_anomaly_alerts
    try:
        result = check_anomalies(date_str)
        if result["any_triggered"]:
            n = send_anomaly_alerts(date_str)
            activity.logger.info(
                f"Anomaly alerts: {n} DM sent — "
                + ", ".join(k for k, v in result["alerts"].items() if v)
            )
        else:
            activity.logger.info(f"Anomaly alerts: no triggers for {date_str}")
        return result
    except Exception as e:
        activity.logger.error(f"Anomaly alert check failed: {e}")
        return {"date": date_str, "alerts": {}, "any_triggered": False, "error": str(e)}


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

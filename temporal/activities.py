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
async def run_weekly_summary(date_str: str) -> bool:
    """Send the deterministic weekly presence summary DM to David.

    Runs alongside the AI Intuition report as a lightweight, non-LLM complement.
    Computes week-over-week metric deltas, best/worst days, and source coverage.

    Returns True if the DM was sent successfully.
    """
    from scripts.weekly_summary import send_weekly_summary
    try:
        ok = send_weekly_summary(date_str)
        activity.logger.info(f"Weekly summary DM {'sent' if ok else 'failed'} for {date_str}")
        return ok
    except Exception as e:
        activity.logger.error(f"Weekly summary failed: {e}")
        return False


@activity.defn
async def generate_weekly_dashboard(date_str: str) -> str:
    """Generate the weekly HTML presence dashboard for the 7 days ending on date_str.

    Returns the path to the generated HTML file, or empty string on failure.
    """
    from analysis.weekly_dashboard import generate_weekly_dashboard as _gen
    try:
        path = _gen(date_str)
        activity.logger.info(f"Weekly dashboard generated: {path}")
        return str(path)
    except Exception as e:
        activity.logger.error(f"Weekly dashboard generation failed: {e}")
        return ""


@activity.defn
async def retrain_ml_models(force: bool = False) -> dict:
    """Train (or retrain) the three scikit-learn ML models.

    Wraps analysis.ml_model.train_all().  Returns the training result dict
    with per-model status ('trained' | 'skipped' | 'error') so the workflow
    can log outcomes without knowing ML internals.

    Args:
        force: Train even when fewer than ML_MIN_DAYS (60) days of data are
               available.  Useful for monthly retrains on early-deployment
               systems; the caller (MonthlyMLRetrainWorkflow) passes True so
               the monthly cycle always produces a fresh model.

    Returns a dict:
        {
            "status": "trained" | "insufficient_data" | "error",
            "days_used": int,
            "anomaly": "trained" | "skipped" | "error",
            "recovery": "trained" | "skipped" | "error",
            "clustering": "trained" | "skipped" | "error",
        }
    """
    from analysis.ml_model import train_all, get_data_status
    try:
        # Log data status before training — failures here are non-fatal.
        try:
            status = get_data_status()
            days_available = status.get("days_available", 0)
            activity.logger.info(
                f"ML retrain: {days_available} days available "
                f"(min={status.get('min_days_required', 60)}, force={force})"
            )
        except Exception as status_err:
            activity.logger.warning(f"ML retrain: get_data_status failed ({status_err}); proceeding with train_all")

        result = train_all(force=force)
        activity.logger.info(
            f"ML retrain complete: {result.get('status')} — "
            f"anomaly={result.get('anomaly')}, "
            f"recovery={result.get('recovery')}, "
            f"clustering={result.get('clustering')}"
        )
        return result
    except Exception as e:
        activity.logger.error(f"ML retrain failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "anomaly": "error",
            "recovery": "error",
            "clustering": "error",
        }


@activity.defn
async def send_midday_checkin(date_str: str) -> bool:
    """Send the midday cognitive check-in DM to David.

    Computes morning-so-far signals (CLS, FDI, SDI, meetings) from the partial
    day JSONL, determines afternoon pace and recommendation, and sends a brief
    Slack DM.  Runs at 13:00 Budapest via MidDayCheckInWorkflow.

    Returns True if the DM was sent successfully (or not meaningful — graceful skip).
    """
    from analysis.midday_checkin import compute_midday_checkin, send_midday_checkin as _send
    try:
        checkin = compute_midday_checkin(date_str)
        if not checkin.is_meaningful:
            activity.logger.info(
                f"Midday check-in not meaningful for {date_str} "
                f"({checkin.active_windows} active morning windows). Skipping."
            )
            return True  # Not an error — just a quiet morning
        ok = _send(date_str)
        status = "sent" if ok else "failed"
        activity.logger.info(
            f"Midday check-in {status} for {date_str} — "
            f"CLS {checkin.morning_cls:.3f}, pace={checkin.pace_label}"
        )
        return ok
    except Exception as e:
        activity.logger.error(f"Midday check-in failed: {e}")
        return False


@activity.defn
async def send_evening_winddown(date_str: str) -> bool:
    """Send the evening wind-down DM to David.

    Computes the full workday data (08:00–18:00), classifies the day type
    (PRODUCTIVE / DEEP / REACTIVE / FRAGMENTED / RECOVERY / MIXED), computes
    the load arc (front-loaded / back-loaded / even), and sends a ~6-line
    Slack DM with a concrete wind-down recommendation.

    Runs at 17:00 UTC = 18:00 Budapest via EveningWindDownWorkflow.
    Gracefully skips when fewer than 3 active workday windows exist.

    Returns True if the DM was sent (or gracefully skipped — not an error).
    """
    from analysis.evening_winddown import compute_evening_winddown, send_evening_winddown as _send
    try:
        winddown = compute_evening_winddown(date_str)
        if not winddown.is_meaningful:
            activity.logger.info(
                f"Evening wind-down not meaningful for {date_str} "
                f"({winddown.active_windows_count} active workday windows). Skipping."
            )
            return True  # Not an error — quiet day
        ok = _send(date_str)
        status = "sent" if ok else "failed"
        activity.logger.info(
            f"Evening wind-down {status} for {date_str} — "
            f"day_type={winddown.day_type}, arc={winddown.load_arc}"
        )
        return ok
    except Exception as e:
        activity.logger.error(f"Evening wind-down failed: {e}")
        return False


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

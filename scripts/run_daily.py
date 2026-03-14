#!/usr/bin/env python3
"""
Presence Tracker — Daily Ingestion Runner

Usage:
    python3 scripts/run_daily.py                  # Run for today
    python3 scripts/run_daily.py 2026-03-13       # Run for a specific date
    python3 scripts/run_daily.py --force           # Re-run even if data exists
    python3 scripts/run_daily.py --no-alerts      # Skip anomaly alert DMs

This script:
1. Collects WHOOP data for the date
2. Collects Google Calendar events
3. Collects Slack activity
4. Collects RescueTime activity (if API key configured)
5. Collects Omi transcripts (if available)
6. Builds 96 × 15-min windows
7. Computes derived metrics
8. Writes YYYY-MM-DD.jsonl to data/chunks/
9. Updates rolling summary stats
10. Sends a digest DM to David
11. Generates HTML dashboard
12. Runs v5 multi-source anomaly alerts (CLS spike, FDI collapse, RAS streak)

v1.7 — Anomaly alerts wired into manual runner:
  Previously the v5 multi-source anomaly engine (analysis/anomaly_alerts.py)
  was only called via the Temporal DailyIngestionWorkflow.  When David or Alfred
  ran `python3 scripts/run_daily.py` directly — for backfills, re-runs, or manual
  ingestion — the proper CLS spike / FDI collapse / RAS streak checks were
  completely skipped.  A weak single-condition fallback (recovery < 50 AND
  CLS > 0.70) from the intuition module was the only guard.

  This release wires send_anomaly_alerts() directly into run_daily.py so that
  both the Temporal workflow path and the manual-runner path execute identical
  alert logic.  The old weak fallback is replaced entirely.

  --no-alerts flag added for tests and backfills that should not spam Slack.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import SLACK_LOGS_CHANNEL, GATEWAY_URL, GATEWAY_TOKEN
from collectors import whoop, gcal, slack, rescuetime, omi
from engine.chunker import build_windows
from engine.chunker import summarize_day as _summarize_day
from engine.store import write_day, update_summary, day_exists

import urllib.request


def _slack_log(message: str) -> None:
    """Send a message to #alfred-logs via gateway."""
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
            pass
    except Exception as e:
        print(f"[slack-log] Warning: {e}", file=sys.stderr)


def run(date_str: str, force: bool = False, quiet: bool = False, alerts: bool = True) -> dict:
    """
    Run the full daily ingestion pipeline for a given date.

    Args:
        date_str: Date to process (YYYY-MM-DD).
        force: Re-run even if data already exists for this date.
        quiet: Suppress progress output (for cron / backfill).
        alerts: Run v5 anomaly alerts after ingestion (default True).
                Set to False for backfills or testing to avoid Slack spam.

    Returns the day summary dict.
    Raises on critical failures.
    """
    if not quiet:
        print(f"[presence] Running daily ingestion for {date_str}")

    if day_exists(date_str) and not force:
        if not quiet:
            print(f"[presence] Data already exists for {date_str}, skipping (use --force to override)")
        from engine.store import read_day
        existing = read_day(date_str)
        return _summarize_day(existing) if existing else {}

    # ── Step 1: Collect WHOOP ─────────────────────────────────────────────
    if not quiet:
        print("[presence] Collecting WHOOP data...")
    try:
        whoop_data = whoop.collect(date_str)
        if not quiet:
            print(f"  Recovery: {whoop_data.get('recovery_score')}%  "
                  f"HRV: {whoop_data.get('hrv_rmssd_milli')}ms  "
                  f"Sleep: {whoop_data.get('sleep_hours')}h")
    except Exception as e:
        print(f"[presence] WHOOP collection failed: {e}", file=sys.stderr)
        whoop_data = {}

    # ── Step 2: Collect Calendar ──────────────────────────────────────────
    if not quiet:
        print("[presence] Collecting Calendar events...")
    try:
        calendar_data = gcal.collect(date_str)
        if not quiet:
            print(f"  Events: {calendar_data.get('event_count', 0)}  "
                  f"Meeting time: {calendar_data.get('total_meeting_minutes', 0)} min")
    except Exception as e:
        print(f"[presence] Calendar collection failed: {e}", file=sys.stderr)
        calendar_data = {"events": [], "event_count": 0, "total_meeting_minutes": 0, "max_concurrent_attendees": 0}

    # ── Step 3: Collect Slack ─────────────────────────────────────────────
    if not quiet:
        print("[presence] Collecting Slack activity...")
    try:
        slack_windows = slack.collect(date_str)
        total_msgs = sum(w.get("total_messages", 0) for w in slack_windows.values())
        if not quiet:
            print(f"  Total messages: {total_msgs}")
    except Exception as e:
        print(f"[presence] Slack collection failed: {e}", file=sys.stderr)
        slack_windows = {}

    # ── Step 3.5: Collect RescueTime (optional) ───────────────────────────
    rescuetime_windows = {}
    try:
        from config import RESCUETIME_API_KEY
        if RESCUETIME_API_KEY:
            if not quiet:
                print("[presence] Collecting RescueTime data...")
            rescuetime_windows = rescuetime.collect(date_str)
            active_rt = sum(1 for w in rescuetime_windows.values() if w.get("active_seconds", 0) > 0)
            if not quiet:
                print(f"  RescueTime: {active_rt} active windows")
        else:
            if not quiet:
                print("[presence] RescueTime: skipped (no API key configured)")
    except Exception as e:
        if not quiet:
            print(f"[presence] RescueTime collection failed (non-fatal): {e}", file=sys.stderr)
        rescuetime_windows = {}

    # ── Step 3.6: Collect Omi transcripts (optional) ──────────────────────
    omi_windows = {}
    try:
        omi_windows = omi.collect(date_str)
        active_omi = sum(1 for w in omi_windows.values() if w.get("conversation_active", False))
        if active_omi > 0:
            if not quiet:
                print(f"[presence] Omi: {active_omi} windows with conversation")
        else:
            if not quiet:
                print("[presence] Omi: no transcripts found for this date")
    except Exception as e:
        if not quiet:
            print(f"[presence] Omi collection failed (non-fatal): {e}", file=sys.stderr)
        omi_windows = {}

    # ── Step 4: Build windows ─────────────────────────────────────────────
    if not quiet:
        print("[presence] Building 15-minute windows...")
    windows = build_windows(
        date_str=date_str,
        whoop_data=whoop_data,
        calendar_data=calendar_data,
        slack_windows=slack_windows,
        rescuetime_windows=rescuetime_windows,
        omi_windows=omi_windows,
    )
    if not quiet:
        print(f"  Built {len(windows)} windows")

    # ── Step 5: Compute summary ───────────────────────────────────────────
    summary = _summarize_day(windows)

    # ── Step 6: Write JSONL ───────────────────────────────────────────────
    output_path = write_day(date_str, windows)
    if not quiet:
        print(f"[presence] Written to {output_path}")

    # ── Step 7: Update rolling summary ───────────────────────────────────
    update_summary(summary)

    # ── Step 8: Log digest to #alfred-logs ───────────────────────────────
    recovery = whoop_data.get("recovery_score")
    hrv = whoop_data.get("hrv_rmssd_milli")
    avg_cls = summary.get("metrics_avg", {}).get("cognitive_load_score")
    avg_fdi = summary.get("metrics_avg", {}).get("focus_depth_index")
    meeting_mins = summary.get("calendar", {}).get("total_meeting_minutes", 0)

    log_msg = (
        f"[PRESENCE] Daily ingestion complete — {date_str}\n"
        f"Recovery: {recovery}% | HRV: {hrv}ms\n"
        f"Avg CLS: {avg_cls:.2f} | Avg FDI: {avg_fdi:.2f}\n"
        f"Meetings: {meeting_mins} min | Windows: {len(windows)}"
    )
    _slack_log(log_msg)

    # ── Step 9: Send personal daily digest DM to David ────────────────────
    try:
        from analysis.daily_digest import send_daily_digest
        send_daily_digest(windows)
    except Exception as e:
        print(f"[presence] Daily digest failed (non-fatal): {e}", file=sys.stderr)

    # ── Step 10: Generate HTML dashboard ─────────────────────────────────
    try:
        from analysis.dashboard import generate_dashboard
        dashboard_path = generate_dashboard(date_str)
        if not quiet:
            print(f"[presence] Dashboard: {dashboard_path}")
    except Exception as e:
        print(f"[presence] Dashboard generation failed (non-fatal): {e}", file=sys.stderr)

    # ── Step 11: Run v5 multi-source anomaly alerts ───────────────────────
    # Uses the same engine as DailyIngestionWorkflow in Temporal.
    # Checks CLS spike (> 2 std devs above baseline), FDI collapse (>30% drop),
    # and RAS misalignment streak (3 consecutive days below 0.45).
    # Sends a batched Slack DM to David if any condition fires.
    # Skipped when alerts=False (backfills, tests, --no-alerts flag).
    if alerts:
        try:
            from analysis.anomaly_alerts import check_anomalies, send_anomaly_alerts
            anomaly_result = check_anomalies(date_str)
            triggered = [k for k, v in anomaly_result.get("alerts", {}).items() if v]
            if triggered:
                send_anomaly_alerts(date_str)
                if not quiet:
                    print(f"[presence] Anomaly alerts fired: {', '.join(triggered)}")
            else:
                if not quiet:
                    print("[presence] Anomaly alerts: no triggers")
        except Exception as e:
            print(f"[presence] Anomaly alerts failed (non-fatal): {e}", file=sys.stderr)

    if not quiet:
        print(f"[presence] Done. Summary: {json.dumps(summary.get('metrics_avg', {}), indent=2)}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Presence Tracker — Daily Ingestion")
    parser.add_argument("date", nargs="?", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date to process (YYYY-MM-DD, default: today)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Re-run even if data already exists")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output (for cron)")
    parser.add_argument("--no-alerts", action="store_true",
                        help="Skip anomaly alert DMs (useful for backfills)")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    summary = run(args.date, force=args.force, quiet=args.quiet, alerts=not args.no_alerts)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

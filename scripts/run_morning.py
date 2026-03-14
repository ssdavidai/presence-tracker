#!/usr/bin/env python3
"""
Presence Tracker — Morning Brief Runner

Usage:
    python3 scripts/run_morning.py                  # Run for today
    python3 scripts/run_morning.py 2026-03-14       # Run for a specific date
    python3 scripts/run_morning.py --dry-run        # Preview without sending

This script:
1. Collects WHOOP data for today (available by ~6am after overnight sync)
2. Loads yesterday's daily summary for context
3. Computes a 7-day HRV baseline for relative context
4. Sends a morning readiness brief to David's Slack DM

Run at 07:00 Budapest time via Temporal schedule.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.morning_brief import send_morning_brief


def main():
    parser = argparse.ArgumentParser(description="Presence Tracker — Morning Brief")
    parser.add_argument(
        "date",
        nargs="?",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date to run for (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview without sending to Slack",
    )
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        # Import and run the dry-run mode directly
        from analysis.morning_brief import (
            compute_morning_brief,
            format_morning_brief_message,
        )
        from datetime import timedelta

        date_str = args.date
        yesterday_date = (
            datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        try:
            from collectors.whoop import collect as whoop_collect
            whoop_data = whoop_collect(date_str)
        except Exception as e:
            print(f"WHOOP failed: {e}", file=sys.stderr)
            whoop_data = {}

        yesterday_summary = None
        try:
            from engine.store import read_summary
            yesterday_summary = read_summary().get("days", {}).get(yesterday_date)
        except Exception:
            pass

        hrv_baseline = None
        try:
            from engine.store import get_recent_summaries
            recent = get_recent_summaries(days=7)
            vals = [s["whoop"]["hrv_rmssd_milli"] for s in recent
                    if s.get("whoop", {}).get("hrv_rmssd_milli")]
            if len(vals) >= 3:
                hrv_baseline = sum(vals) / len(vals)
        except Exception:
            pass

        # Today's calendar (v7.0)
        today_calendar = None
        try:
            from collectors.gcal import collect as gcal_collect
            today_calendar = gcal_collect(date_str)
        except Exception as e:
            print(f"Calendar failed: {e}", file=sys.stderr)

        brief = compute_morning_brief(
            date_str, whoop_data, yesterday_summary, hrv_baseline,
            today_calendar=today_calendar,
        )
        message = format_morning_brief_message(brief)
        print("=" * 60)
        print(message)
        print("=" * 60)
        print("\n[dry-run] Not sent.")
    else:
        ok = send_morning_brief(args.date)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

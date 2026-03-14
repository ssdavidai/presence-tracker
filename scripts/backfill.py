#!/usr/bin/env python3
"""
Presence Tracker — Historical Backfill

Usage:
    python3 scripts/backfill.py --days 30           # Backfill last 30 days
    python3 scripts/backfill.py --start 2026-02-01  # Backfill from date to today
    python3 scripts/backfill.py --start 2026-02-01 --end 2026-02-28

Note: WHOOP data is available for all historical dates.
Calendar data goes back as far as Google Calendar retains it.
Slack history depends on your workspace plan.

Anomaly alerts are automatically disabled during backfill runs to avoid
sending multiple historical alerts to Slack.  Each day runs with alerts=False.
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_daily import run


def main():
    parser = argparse.ArgumentParser(description="Presence Tracker — Backfill historical data")
    parser.add_argument("--days", type=int, help="Number of days to backfill (from today back)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD, default: yesterday)")
    parser.add_argument("--force", "-f", action="store_true", help="Re-run existing dates")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds between days (default: 2)")
    args = parser.parse_args()

    today = datetime.now().date()

    if args.days:
        start = today - timedelta(days=args.days)
        end = today - timedelta(days=1)
    elif args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else today - timedelta(days=1)
    else:
        parser.error("Provide --days or --start")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(f"[backfill] Processing {len(dates)} days from {dates[0]} to {dates[-1]}")

    success = 0
    failed = 0
    for i, date_str in enumerate(dates):
        print(f"[backfill] [{i+1}/{len(dates)}] {date_str}")
        try:
            # alerts=False: backfilling historical dates must not fire Slack DMs
            run(date_str, force=args.force, alerts=False)
            success += 1
        except Exception as e:
            print(f"[backfill] FAILED {date_str}: {e}", file=sys.stderr)
            failed += 1
        if args.sleep > 0 and i < len(dates) - 1:
            time.sleep(args.sleep)

    print(f"[backfill] Done. Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()

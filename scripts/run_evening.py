#!/usr/bin/env python3
"""
Presence Tracker — Evening Wind-Down Runner (v30)

Sends the evening wind-down signal for today (or a specified date).

Fires at 17:00 UTC = 18:00 Budapest (CET/UTC+1), end-of-workday.

Classifies the day type (PRODUCTIVE / DEEP / REACTIVE / FRAGMENTED /
RECOVERY / MIXED), shows the load arc (front-loaded / back-loaded / even),
and sends a concrete wind-down recommendation to David's Slack DM.

Usage:
    python3 scripts/run_evening.py                 # Today, send to Slack
    python3 scripts/run_evening.py 2026-03-17      # Specific date
    python3 scripts/run_evening.py --dry-run       # Print without sending
    python3 scripts/run_evening.py --json          # JSON output
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path so modules resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evening Wind-Down — end-of-workday cognitive signal",
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="Date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message without sending to Slack.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON data.",
    )
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    from analysis.evening_winddown import (
        compute_evening_winddown,
        format_winddown_message,
        send_evening_winddown,
    )

    winddown = compute_evening_winddown(date_str)

    if args.json:
        print(json.dumps(winddown.to_dict(), indent=2))
        return 0

    if not winddown.is_meaningful:
        print(f"[run_evening] No meaningful data for {date_str} — skipping.")
        return 0

    message = format_winddown_message(winddown)

    if args.dry_run:
        print(message)
        return 0

    # Live send
    success = send_evening_winddown(date_str)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Presence Tracker — Midday Check-In Runner

Usage:
    python3 scripts/run_midday.py                   # Run for today
    python3 scripts/run_midday.py 2026-03-14        # Run for a specific date
    python3 scripts/run_midday.py --dry-run         # Preview without sending

This script:
1. Loads the current day's partial JSONL data (windows up to MIDDAY_HOUR=13:00)
2. Computes morning CLS, FDI, SDI and meeting load
3. Determines the pace label (Running hot / On track / Light morning)
4. Estimates remaining cognitive budget for the afternoon
5. Sends a brief Slack DM to David (or prints in dry-run mode)

Runs at 13:00 Budapest time via Temporal MidDayCheckInWorkflow schedule.
Can also be triggered manually for testing or catch-up.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.midday_checkin import (
    compute_midday_checkin,
    format_checkin_message,
    send_midday_checkin as _send,
    MIN_ACTIVE_WINDOWS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Presence Tracker — Midday Cognitive Check-In"
    )
    parser.add_argument(
        "date",
        nargs="?",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date to run for (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview check-in without sending to Slack",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw check-in data as JSON",
    )
    args = parser.parse_args()

    # Validate date
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    date_str = args.date
    checkin = compute_midday_checkin(date_str)

    if args.json:
        import json
        print(json.dumps(checkin.to_dict(), indent=2))
        return

    # Always print in dry-run or verbose mode
    print()
    print(f"Midday Check-In — {date_str}")
    print("=" * 50)

    if not checkin.is_meaningful:
        print(
            f"  Not meaningful — only {checkin.active_windows} active morning windows "
            f"(need ≥ {MIN_ACTIVE_WINDOWS})."
        )
        print()
        if not args.dry_run:
            print("[skipped] Insufficient morning data — not sending.")
        sys.exit(0)

    print(f"  Morning CLS:    {checkin.morning_cls:.3f}  ({_cls_label(checkin.morning_cls)})")
    print(f"  Morning FDI:    {checkin.morning_fdi:.3f}  ({_fdi_label(checkin.morning_fdi)})")
    if checkin.morning_sdi is not None:
        print(f"  Morning SDI:    {checkin.morning_sdi:.3f}")
    if checkin.meeting_minutes:
        print(f"  Meetings:       {_fmt_minutes(checkin.meeting_minutes)}")
    pace_str = checkin.pace_label
    if checkin.pace_ratio is not None:
        pace_str += f"  (×{checkin.pace_ratio:.2f} baseline)"
    print(f"  Pace:           {pace_str}")
    if checkin.remaining_budget_hours is not None:
        print(f"  Budget left:    ~{checkin.remaining_budget_hours:.1f}h")
    print()
    print(f"  → {checkin.afternoon_nudge}")
    print()

    if args.dry_run:
        print("-" * 50)
        print("Slack message preview:")
        print("-" * 50)
        print(format_checkin_message(checkin))
        print("-" * 50)
        print("\n[dry-run] Not sent.")
    else:
        ok = _send(date_str)
        if ok:
            print("✓ Midday check-in sent to David's DM")
            sys.exit(0)
        else:
            print("✗ Failed to send midday check-in", file=sys.stderr)
            sys.exit(1)


# ─── Label helpers (mirrors midday_checkin.py) ────────────────────────────────

def _cls_label(cls: float | None) -> str:
    if cls is None:
        return "N/A"
    if cls < 0.10:
        return "very light"
    if cls < 0.25:
        return "light"
    if cls < 0.50:
        return "moderate"
    if cls < 0.75:
        return "high"
    return "very high"


def _fdi_label(fdi: float | None) -> str:
    if fdi is None:
        return "N/A"
    if fdi >= 0.80:
        return "deep focus"
    if fdi >= 0.60:
        return "solid focus"
    if fdi >= 0.40:
        return "moderate focus"
    return "fragmented"


def _fmt_minutes(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}min"
    h = minutes // 60
    m = minutes % 60
    return f"{h}h{m:02d}min" if m else f"{h}h"


if __name__ == "__main__":
    main()

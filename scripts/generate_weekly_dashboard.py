#!/usr/bin/env python3
"""
Presence Tracker — Weekly Dashboard Generator

Usage:
    python3 scripts/generate_weekly_dashboard.py                  # Latest 7 days
    python3 scripts/generate_weekly_dashboard.py 2026-03-14       # Week ending on date
    python3 scripts/generate_weekly_dashboard.py --open           # Open in browser after

Generates: data/dashboard/week-YYYY-MM-DD.html
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.weekly_dashboard import generate_weekly_dashboard
from engine.store import list_available_dates


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate weekly HTML presence dashboard"
    )
    parser.add_argument("date", nargs="?", help="End date YYYY-MM-DD (default: latest available)")
    parser.add_argument("--open", action="store_true", help="Open in browser after generation")
    args = parser.parse_args()

    if args.date:
        end_date_str = args.date
    else:
        dates = list_available_dates()
        if not dates:
            print("No data available. Run scripts/run_daily.py first.", file=sys.stderr)
            sys.exit(1)
        end_date_str = sorted(dates)[-1]

    out = generate_weekly_dashboard(end_date_str)
    print(f"✅ Weekly dashboard: {out}")

    if args.open:
        import subprocess
        subprocess.run(["open", str(out)])


if __name__ == "__main__":
    main()

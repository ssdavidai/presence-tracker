#!/usr/bin/env python3
"""
Presence Tracker — Dashboard Generator

Generate (or regenerate) the daily HTML presence dashboard for any date.

Usage:
    python3 scripts/generate_dashboard.py              # Latest date
    python3 scripts/generate_dashboard.py 2026-03-13   # Specific date
    python3 scripts/generate_dashboard.py --open       # Open in browser after
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.dashboard import generate_dashboard
from engine.store import list_available_dates


def main():
    parser = argparse.ArgumentParser(description="Generate daily HTML presence dashboard")
    parser.add_argument("date", nargs="?", help="Date YYYY-MM-DD (default: latest available)")
    parser.add_argument("--open", action="store_true", help="Open in browser after generation")
    parser.add_argument("--output", help="Override output file path")
    args = parser.parse_args()

    if args.date:
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)
        date_str = args.date
    else:
        dates = list_available_dates()
        if not dates:
            print("No data available. Run run_daily.py first.", file=sys.stderr)
            sys.exit(1)
        date_str = sorted(dates)[-1]
        print(f"Using latest available date: {date_str}")

    output_path = Path(args.output) if args.output else None

    try:
        path = generate_dashboard(date_str, output_path=output_path)
        print(f"✓ Dashboard generated: {path}")
        if args.open:
            subprocess.run(["open", str(path)])
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Dashboard generation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

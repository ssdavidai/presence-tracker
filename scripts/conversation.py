#!/usr/bin/env python3
"""
Presence Tracker — Conversation Intelligence CLI

Usage:
    python3 scripts/conversation.py                  # Last 14 days
    python3 scripts/conversation.py --days 28        # Last 28 days
    python3 scripts/conversation.py --days 7         # Last week
    python3 scripts/conversation.py --json           # Machine-readable JSON

Analyses raw Omi transcript history directly — independent of the JSONL store.
Works even when full daily ingestion hasn't been run, because it reads
~/omi/transcripts/ directly.

Shows:
  - Total and average daily speech load
  - Peak conversation hour
  - Language distribution (English / Hungarian)
  - Topic category breakdown
  - Cognitive density of conversations
  - Hourly conversation activity sparkline
  - Load trend and actionable insights
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.conversation_intelligence import (
    analyse_conversation_history,
    format_conversation_terminal,
    to_dict,
)
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Presence Tracker — Conversation Intelligence"
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Days to look back (default: 14)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON",
    )
    args = parser.parse_args()

    ci = analyse_conversation_history(days=args.days, end_date_str=args.date)

    if args.json:
        print(json.dumps(to_dict(ci), indent=2))
    else:
        print(format_conversation_terminal(ci))


if __name__ == "__main__":
    main()

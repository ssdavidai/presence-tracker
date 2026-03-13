#!/usr/bin/env python3
"""
Presence Tracker — Weekly Analysis Runner

Usage:
    python3 scripts/run_analysis.py          # Run weekly intuition analysis

Spawns an Alfred subagent to analyze the past 7 days and deliver
a Presence Report to David's Slack DM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.intuition import run_weekly_analysis

if __name__ == "__main__":
    success = run_weekly_analysis()
    sys.exit(0 if success else 1)

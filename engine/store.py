"""
Presence Tracker — Storage Layer

Reads and writes daily JSONL chunk files.
Each line is a single 15-minute window JSON object.

Data lives in: data/chunks/YYYY-MM-DD.jsonl
Summary stats: data/summary/rolling.json
"""

import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

from config import CHUNKS_DIR, SUMMARY_DIR


# ─── JSONL Read/Write ─────────────────────────────────────────────────────────

def write_day(date_str: str, windows: list[dict]) -> Path:
    """Write 96 windows for a day to a JSONL file."""
    path = CHUNKS_DIR / f"{date_str}.jsonl"
    with open(path, "w") as f:
        for window in windows:
            f.write(json.dumps(window, default=str) + "\n")
    return path


def read_day(date_str: str) -> list[dict]:
    """Read all windows for a day from JSONL file."""
    path = CHUNKS_DIR / f"{date_str}.jsonl"
    if not path.exists():
        return []
    windows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    windows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return windows


def read_range(start_str: str, end_str: str) -> list[dict]:
    """Read all windows across a date range (inclusive)."""
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    all_windows = []
    current = start
    while current <= end:
        windows = read_day(current.strftime("%Y-%m-%d"))
        all_windows.extend(windows)
        current += timedelta(days=1)
    return all_windows


def list_available_dates() -> list[str]:
    """List all dates that have chunk files."""
    files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    return [f.stem for f in files]


def day_exists(date_str: str) -> bool:
    """Check if a day's chunk file exists."""
    return (CHUNKS_DIR / f"{date_str}.jsonl").exists()


# ─── Summary Stats ────────────────────────────────────────────────────────────

def update_summary(day_summary: dict) -> None:
    """Append or update a day's summary in the rolling stats file."""
    rolling_path = SUMMARY_DIR / "rolling.json"

    if rolling_path.exists():
        with open(rolling_path) as f:
            rolling = json.load(f)
    else:
        rolling = {"days": {}}

    date_key = day_summary.get("date", "unknown")
    rolling["days"][date_key] = day_summary
    rolling["last_updated"] = datetime.now().isoformat()
    rolling["total_days"] = len(rolling["days"])

    with open(rolling_path, "w") as f:
        json.dump(rolling, f, indent=2, default=str)


def read_summary() -> dict:
    """Read the rolling summary stats."""
    rolling_path = SUMMARY_DIR / "rolling.json"
    if not rolling_path.exists():
        return {"days": {}, "total_days": 0}
    with open(rolling_path) as f:
        return json.load(f)


def get_recent_summaries(days: int = 7) -> list[dict]:
    """Get day summaries for the last N days (most recent first)."""
    summary = read_summary()
    all_days = sorted(summary.get("days", {}).keys(), reverse=True)
    recent = all_days[:days]
    return [summary["days"][d] for d in recent]


# ─── Stats Helpers ────────────────────────────────────────────────────────────

def get_data_age_days() -> int:
    """Return how many days of data we have."""
    dates = list_available_dates()
    return len(dates)


def get_date_range() -> tuple[Optional[str], Optional[str]]:
    """Return (oldest_date, newest_date) or (None, None) if no data."""
    dates = list_available_dates()
    if not dates:
        return None, None
    return dates[0], dates[-1]

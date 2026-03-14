#!/usr/bin/env python3
"""
Presence Tracker — Data Export

Export daily presence metrics to CSV or JSON for external analysis,
spreadsheets, or data visualisation tools.

Usage:
    python3 scripts/export.py                        # Export all days to stdout (CSV)
    python3 scripts/export.py --format csv           # Explicit CSV (default)
    python3 scripts/export.py --format json          # JSON array of day objects
    python3 scripts/export.py --output metrics.csv   # Write to file instead of stdout
    python3 scripts/export.py --days 30              # Last 30 days only
    python3 scripts/export.py --start 2026-02-01     # From date
    python3 scripts/export.py --start 2026-02-01 --end 2026-03-01

Output columns (CSV):
    date                  YYYY-MM-DD
    recovery_score        WHOOP recovery % (0–100)
    hrv_rmssd_ms          Heart rate variability in ms
    resting_hr            Resting heart rate (bpm)
    sleep_hours           Hours of sleep
    sleep_performance     WHOOP sleep performance % (0–100)
    avg_cls               Average cognitive load score (0–1)
    peak_cls              Peak CLS in the day (0–1)
    avg_fdi               Focus depth index — active windows only (0–1)
    avg_sdi               Social drain index — active windows (0–1)
    avg_csc               Context switch cost — active windows (0–1)
    avg_ras               Recovery alignment score (0–1)
    active_fdi            FDI over active windows (identical to avg_fdi, for clarity)
    active_windows        Working-hour windows with measurable activity
    peak_focus_hour       Hour of day with best focus (0–23, or blank)
    peak_focus_fdi        FDI at peak focus hour (0–1, or blank)
    total_meeting_min     Total meeting minutes in the day
    meeting_windows       Number of 15-min meeting windows
    slack_sent            Slack messages sent
    slack_received        Slack messages received
    rt_focus_min          RescueTime focused minutes (blank if not configured)
    rt_distraction_min    RescueTime distraction minutes (blank if not configured)
    rt_productive_pct     RescueTime % productive time (blank if not configured)
    sources               Comma-separated list of data sources present

JSON format produces an array of objects with the same keys plus raw nested
data (whoop, metrics_avg, focus_quality, calendar, rescuetime) as sub-dicts.

Examples:
    # Weekly overview in terminal
    python3 scripts/export.py --days 7

    # Full history for Jupyter analysis
    python3 scripts/export.py --format json --output data/export/full_history.json

    # Copy last 30 days to clipboard (macOS)
    python3 scripts/export.py --days 30 | pbcopy
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.store import list_available_dates, read_summary, read_day


# ─── Column definitions ───────────────────────────────────────────────────────

# Ordered list of (csv_column_name, description) for the header
CSV_COLUMNS = [
    "date",
    "recovery_score",
    "hrv_rmssd_ms",
    "resting_hr",
    "sleep_hours",
    "sleep_performance",
    "avg_cls",
    "peak_cls",
    "avg_fdi",
    "avg_sdi",
    "avg_csc",
    "avg_ras",
    "active_fdi",
    "active_windows",
    "peak_focus_hour",
    "peak_focus_fdi",
    "total_meeting_min",
    "meeting_windows",
    "slack_sent",
    "slack_received",
    "rt_focus_min",
    "rt_distraction_min",
    "rt_productive_pct",
    "sources",
]


# ─── Row extraction ────────────────────────────────────────────────────────────

def _safe(value: Any, default: str = "") -> str:
    """Convert a value to CSV-safe string, using default for None."""
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _sources_for_date(date_str: str) -> str:
    """
    Return comma-separated list of data sources present for a given date.

    Reads the first window's metadata.sources_available.  Falls back to
    inferring from the summary dict if windows are unavailable.
    """
    windows = read_day(date_str)
    if windows:
        # Use sources from first window — these are day-level sources (whoop,
        # calendar, slack are always checked; rescuetime/omi are conditional).
        # Note: sources_available reflects what was available at ingestion time.
        sources = windows[0].get("metadata", {}).get("sources_available", [])
        # Supplement: check if any window has rescuetime/omi data, since
        # sources_available only lists them when active in that specific window.
        has_rt = any(w.get("rescuetime") is not None for w in windows)
        has_omi = any(w.get("omi") is not None for w in windows)
        full_sources = list(sources)
        if has_rt and "rescuetime" not in full_sources:
            full_sources.append("rescuetime")
        if has_omi and "omi" not in full_sources:
            full_sources.append("omi")
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for s in full_sources:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return "|".join(deduped)
    return ""


def build_row(date_str: str, day_data: dict) -> dict:
    """
    Build a flat dict from a day's rolling summary entry.

    Args:
        date_str: "YYYY-MM-DD"
        day_data: dict from rolling.json["days"][date_str]

    Returns dict with all CSV_COLUMNS as keys.
    """
    w = day_data.get("whoop") or {}
    m = day_data.get("metrics_avg") or {}
    mp = day_data.get("metrics_peak") or {}
    fq = day_data.get("focus_quality") or {}
    cal = day_data.get("calendar") or {}
    slack = day_data.get("slack") or {}
    rt = day_data.get("rescuetime") or {}

    return {
        "date": date_str,
        # WHOOP
        "recovery_score": _safe(w.get("recovery_score")),
        "hrv_rmssd_ms": _safe(w.get("hrv_rmssd_milli")),
        "resting_hr": _safe(w.get("resting_heart_rate")),
        "sleep_hours": _safe(w.get("sleep_hours")),
        "sleep_performance": _safe(w.get("sleep_performance")),
        # Metrics (averages over working hours)
        "avg_cls": _safe(m.get("cognitive_load_score")),
        "peak_cls": _safe(mp.get("cognitive_load_score")),
        "avg_fdi": _safe(fq.get("active_fdi") or m.get("focus_depth_index")),
        "avg_sdi": _safe(m.get("social_drain_index")),
        "avg_csc": _safe(m.get("context_switch_cost")),
        "avg_ras": _safe(m.get("recovery_alignment_score")),
        # Focus quality
        "active_fdi": _safe(fq.get("active_fdi")),
        "active_windows": _safe(fq.get("active_windows")),
        "peak_focus_hour": _safe(fq.get("peak_focus_hour")),
        "peak_focus_fdi": _safe(fq.get("peak_focus_fdi")),
        # Calendar
        "total_meeting_min": _safe(cal.get("total_meeting_minutes")),
        "meeting_windows": _safe(cal.get("meeting_windows")),
        # Slack
        "slack_sent": _safe(slack.get("messages_sent")),
        "slack_received": _safe(slack.get("messages_received")),
        # RescueTime (optional — blank when not configured)
        "rt_focus_min": _safe(rt.get("focus_minutes")),
        "rt_distraction_min": _safe(rt.get("distraction_minutes")),
        "rt_productive_pct": _safe(rt.get("productive_pct")),
        # Source coverage
        "sources": _sources_for_date(date_str),
    }


def build_json_row(date_str: str, day_data: dict) -> dict:
    """
    Build a rich dict for JSON export — includes nested sub-dicts.

    The flat CSV row is merged with the raw sub-dicts from the summary
    so the JSON output is useful for programmatic analysis.
    """
    flat = build_row(date_str, day_data)
    # Re-attach original nested data for richer JSON
    flat["_raw"] = {
        "whoop": day_data.get("whoop"),
        "metrics_avg": day_data.get("metrics_avg"),
        "metrics_peak": day_data.get("metrics_peak"),
        "focus_quality": day_data.get("focus_quality"),
        "calendar": day_data.get("calendar"),
        "slack": day_data.get("slack"),
        "rescuetime": day_data.get("rescuetime"),
    }
    return flat


# ─── Date filtering ────────────────────────────────────────────────────────────

def filter_dates(all_dates: list[str], days: Optional[int], start: Optional[str], end: Optional[str]) -> list[str]:
    """Apply date range filters to a sorted list of available dates."""
    if not all_dates:
        return []

    today = datetime.now().date()

    if days is not None:
        cutoff = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        return [d for d in all_dates if d >= cutoff]

    if start or end:
        start_str = start or all_dates[0]
        end_str = end or today.strftime("%Y-%m-%d")
        return [d for d in all_dates if start_str <= d <= end_str]

    return all_dates


# ─── Export functions ──────────────────────────────────────────────────────────

def export_csv(rows: list[dict], output=None) -> None:
    """Write rows as CSV to output (file object or stdout)."""
    dest = output or sys.stdout
    writer = csv.DictWriter(dest, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def export_json(rows: list[dict], output=None) -> None:
    """Write rows as JSON array to output (file object or stdout)."""
    dest = output or sys.stdout
    json.dump(rows, dest, indent=2, default=str)
    dest.write("\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Presence Tracker daily metrics to CSV or JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--format", "-f",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Export only the last N days",
    )
    parser.add_argument(
        "--start",
        help="Start date for range export (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        help="End date for range export (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress status messages (only data to stdout)",
    )
    args = parser.parse_args()

    # ── Validate date args ─────────────────────────────────────────────────
    for arg_name, arg_val in [("--start", args.start), ("--end", args.end)]:
        if arg_val:
            try:
                datetime.strptime(arg_val, "%Y-%m-%d")
            except ValueError:
                print(f"Invalid date format for {arg_name}: {arg_val}. Use YYYY-MM-DD.", file=sys.stderr)
                return 1

    # ── Load data ──────────────────────────────────────────────────────────
    all_dates = list_available_dates()
    if not all_dates:
        if not args.quiet:
            print("No data available. Run daily ingestion first.", file=sys.stderr)
        return 1

    target_dates = filter_dates(all_dates, args.days, args.start, args.end)
    if not target_dates:
        if not args.quiet:
            print("No dates match the specified filter.", file=sys.stderr)
        return 1

    summary = read_summary()
    all_day_data = summary.get("days", {})

    # ── Build rows ─────────────────────────────────────────────────────────
    if args.format == "json":
        rows = []
        for date_str in sorted(target_dates):
            day_data = all_day_data.get(date_str, {})
            rows.append(build_json_row(date_str, day_data))
    else:
        rows = []
        for date_str in sorted(target_dates):
            day_data = all_day_data.get(date_str, {})
            rows.append(build_row(date_str, day_data))

    if not args.quiet:
        date_range = f"{target_dates[0]} → {target_dates[-1]}" if len(target_dates) > 1 else target_dates[0]
        print(
            f"[export] {len(rows)} day(s) | {date_range} | format={args.format}",
            file=sys.stderr,
        )

    # ── Write output ───────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="" if args.format == "csv" else None) as f:
            if args.format == "csv":
                export_csv(rows, f)
            else:
                export_json(rows, f)
        if not args.quiet:
            print(f"[export] Written to {output_path}", file=sys.stderr)
    else:
        if args.format == "csv":
            export_csv(rows)
        else:
            export_json(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())

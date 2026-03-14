#!/usr/bin/env python3
"""
Presence Tracker — Metric Recomputation

Re-reads existing JSONL chunk files and recomputes all 5 derived metrics
using the current formula engine, without re-fetching data from APIs.

Why this matters:
  Metric formulas evolve across versions (v1.1 → v1.2 → v1.4 ...).  When a
  formula is improved, historical JSONL files on disk retain stale values.
  This script replays every stored window through the current engine, ensuring
  the full history uses the latest formulas.

  Example fixes this script applies retroactively:
  - v1.1: HRV and sleep_performance now feed into CLS and RAS (not just recovery_score)
  - v1.2: RescueTime signals upgrade FDI, CSC, CLS when present
  - v1.4: Solo calendar blocks (attendees ≤ 1) no longer inflate SDI/FDI
  - v1.3: is_active_window metadata flag (may be absent/None in older files)
  - v2.0: Omi conversation_active now counts as an active window signal
  - v5.8: sources_available refreshed from window signals on recompute

Usage:
    python3 scripts/recompute_metrics.py                   # Recompute all days
    python3 scripts/recompute_metrics.py --date 2026-03-13 # One specific day
    python3 scripts/recompute_metrics.py --days 30         # Last 30 days
    python3 scripts/recompute_metrics.py --start 2026-03-01 --end 2026-03-13
    python3 scripts/recompute_metrics.py --dry-run          # Show changes without writing
    python3 scripts/recompute_metrics.py --regenerate-dashboards  # Rebuild HTML dashboards too

The script preserves all raw signals (whoop, calendar, slack, rescuetime, omi)
and only replaces the metrics block, is_active_window flag, and sources_available.
Rolling summary stats are rebuilt after recomputation.

Output:
  - Updated YYYY-MM-DD.jsonl files (metrics block replaced, metadata refreshed)
  - Updated data/summary/rolling.json
  - (Optional) Regenerated data/dashboard/YYYY-MM-DD.html files
  - Per-day diff summary printed to stdout
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.metrics import compute_metrics
from engine.store import (
    read_day,
    write_day,
    list_available_dates,
    update_summary,
)
from engine.chunker import summarize_day


# ─── Active window detection (mirrors chunker.py logic — keep in sync!) ───────

def _is_active_window(window: dict) -> bool:
    """
    Determine whether a window had any measurable behavioral signal.

    Mirrors the logic in engine/chunker.py build_windows().
    A window is active if at least one of:
      - A calendar event was active (in_meeting = True)
      - Slack messages were sent or received (total_messages > 0)
      - RescueTime detected computer activity (active_seconds > 0)
      - Omi detected conversation activity (conversation_active = True)

    This flag lets downstream analytics filter out idle periods (sleep,
    away-from-keyboard) so FDI averages are computed only when David was
    genuinely engaged — not counting quiet hours where FDI=1.0 because
    there were zero interruptions.

    v2.0 fix: Omi conversation_active is now included as an activity signal.
    Previously, windows with Omi speech but no Slack/meetings were incorrectly
    marked inactive, causing FDI/active_fdi to be wrong for voice-heavy periods.
    """
    if window.get("calendar", {}).get("in_meeting", False):
        return True
    if window.get("slack", {}).get("total_messages", 0) > 0:
        return True
    rt = window.get("rescuetime") or {}
    if rt.get("active_seconds", 0) > 0:
        return True
    omi = window.get("omi") or {}
    if omi.get("conversation_active", False):
        return True
    return False


# ─── Sources available refresh (mirrors chunker.py logic) ────────────────────

def _compute_sources_available(window: dict) -> list[str]:
    """
    Recompute the sources_available list from window signals.

    Mirrors chunker.py: whoop and calendar are always present (they provide
    baseline signals even when empty), slack is always present, rescuetime
    and omi are added only when their data is present.

    This corrects older JSONL files that may have been written before a
    data source was active — e.g. a file written pre-v2.0 won't have
    'omi' in sources_available even if Omi data was later backfilled.
    """
    sources = ["whoop", "calendar", "slack"]

    rt = window.get("rescuetime") or {}
    if rt.get("active_seconds", 0) > 0:
        if "rescuetime" not in sources:
            sources.append("rescuetime")

    omi = window.get("omi") or {}
    if omi.get("conversation_active", False):
        if "omi" not in sources:
            sources.append("omi")

    return sources


# ─── Per-window recomputation ─────────────────────────────────────────────────

def recompute_window(window: dict) -> tuple[dict, dict]:
    """
    Recompute metrics for a single window.

    Returns (updated_window, diff_dict) where diff_dict maps metric name
    to (old_value, new_value) for any metric that changed.

    Updates:
    - metrics block (all 5 derived metrics via current formula engine)
    - metadata.is_active_window (Omi-aware, mirrors chunker.py)
    - metadata.sources_available (refreshed from window signals)
    """
    old_metrics = window.get("metrics", {})

    # Recompute using current engine
    new_metrics = compute_metrics(window)

    # Recompute is_active_window (now Omi-aware)
    new_is_active = _is_active_window(window)
    old_is_active = window.get("metadata", {}).get("is_active_window")

    # Recompute sources_available
    new_sources = _compute_sources_available(window)
    old_sources = window.get("metadata", {}).get("sources_available", [])

    # Build diff
    diff: dict[str, tuple] = {}
    for key in new_metrics:
        old_val = old_metrics.get(key)
        new_val = new_metrics[key]
        # Report change if value changed meaningfully (> 0.0001 for floats)
        if old_val is None or abs(float(new_val) - float(old_val)) > 0.0001:
            diff[key] = (old_val, new_val)

    if old_is_active != new_is_active:
        diff["is_active_window"] = (old_is_active, new_is_active)

    if sorted(old_sources) != sorted(new_sources):
        diff["sources_available"] = (old_sources, new_sources)

    # Apply updates
    updated = dict(window)
    updated["metrics"] = new_metrics
    updated.setdefault("metadata", {})
    updated["metadata"]["is_active_window"] = new_is_active
    updated["metadata"]["sources_available"] = new_sources

    return updated, diff


# ─── Per-day recomputation ────────────────────────────────────────────────────

def recompute_day(
    date_str: str,
    dry_run: bool = False,
    quiet: bool = False,
    regenerate_dashboard: bool = False,
) -> dict:
    """
    Recompute metrics for all windows in a day.

    Args:
        date_str: "YYYY-MM-DD"
        dry_run: if True, report changes without writing to disk
        quiet: suppress per-window output
        regenerate_dashboard: if True, regenerate the HTML dashboard after writing

    Returns:
        Summary dict with counts of changed windows and changed metrics.
    """
    windows = read_day(date_str)
    if not windows:
        if not quiet:
            print(f"  [{date_str}] No data — skipping")
        return {"date": date_str, "windows": 0, "changed": 0}

    updated_windows = []
    changed_count = 0
    metric_change_counts: dict[str, int] = {}
    total_diffs: list[dict] = []

    for window in windows:
        updated, diff = recompute_window(window)
        updated_windows.append(updated)
        if diff:
            changed_count += 1
            total_diffs.append({"window_id": window["window_id"], "diff": diff})
            for key in diff:
                metric_change_counts[key] = metric_change_counts.get(key, 0) + 1

    result = {
        "date": date_str,
        "windows": len(windows),
        "changed": changed_count,
        "metric_changes": metric_change_counts,
    }

    if not quiet:
        if changed_count == 0:
            print(f"  [{date_str}] {len(windows)} windows — no changes needed")
        else:
            change_parts = [f"{k}: {v}w" for k, v in sorted(metric_change_counts.items())]
            print(f"  [{date_str}] {len(windows)} windows — {changed_count} changed "
                  f"({', '.join(change_parts)})")

    if not dry_run and changed_count > 0:
        # Write updated JSONL
        write_day(date_str, updated_windows)
        # Rebuild summary stats
        summary = summarize_day(updated_windows)
        update_summary(summary)
        if not quiet:
            print(f"    ✓ Written and summary updated")

        # Optionally regenerate HTML dashboard from updated JSONL
        if regenerate_dashboard:
            try:
                from analysis.dashboard import generate_dashboard
                dashboard_path = generate_dashboard(date_str)
                if not quiet:
                    print(f"    ✓ Dashboard regenerated: {dashboard_path}")
                result["dashboard_regenerated"] = str(dashboard_path)
            except Exception as e:
                if not quiet:
                    print(f"    ⚠ Dashboard regeneration failed: {e}", file=sys.stderr)
                result["dashboard_error"] = str(e)

    elif dry_run and changed_count > 0 and not quiet:
        # Show first few diffs for inspection
        for item in total_diffs[:3]:
            print(f"    [{item['window_id']}]")
            for metric, (old_val, new_val) in item["diff"].items():
                if isinstance(old_val, float) and isinstance(new_val, float):
                    old_str = f"{old_val:.4f}"
                    new_str = f"{new_val:.4f}"
                else:
                    old_str = str(old_val)
                    new_str = str(new_val)
                print(f"      {metric}: {old_str} → {new_str}")
        if len(total_diffs) > 3:
            print(f"    ... and {len(total_diffs) - 3} more windows")

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Presence Tracker — Recompute metrics from stored JSONL data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date",
        help="Recompute a single day (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Recompute the last N days",
    )
    parser.add_argument(
        "--start",
        help="Start date for range recomputation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        help="End date for range recomputation (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing to disk",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-day output",
    )
    parser.add_argument(
        "--regenerate-dashboards",
        action="store_true",
        help="Regenerate HTML dashboards after recomputing metrics (days with changes only)",
    )
    args = parser.parse_args()

    # ── Determine dates to process ────────────────────────────────────────
    all_available = list_available_dates()

    if not all_available:
        print("No chunk data found. Run the daily ingestion first.")
        sys.exit(0)

    if args.date:
        target_dates = [args.date]
    elif args.days:
        today = datetime.now().date()
        cutoff = today - timedelta(days=args.days)
        target_dates = [d for d in all_available if d >= cutoff.strftime("%Y-%m-%d")]
    elif args.start:
        start_str = args.start
        end_str = args.end or datetime.now().strftime("%Y-%m-%d")
        target_dates = [d for d in all_available if start_str <= d <= end_str]
    else:
        # Default: all available dates
        target_dates = all_available

    if not target_dates:
        print("No matching dates found.")
        sys.exit(0)

    mode = "[dry-run] " if args.dry_run else ""
    regen_note = " [+dashboards]" if args.regenerate_dashboards and not args.dry_run else ""
    print(f"{mode}Recomputing metrics{regen_note} for {len(target_dates)} day(s): "
          f"{target_dates[0]} → {target_dates[-1]}")
    print()

    # ── Process each day ──────────────────────────────────────────────────
    total_changed_windows = 0
    total_windows = 0
    dashboards_regenerated = 0
    results = []

    for date_str in sorted(target_dates):
        result = recompute_day(
            date_str,
            dry_run=args.dry_run,
            quiet=args.quiet,
            regenerate_dashboard=args.regenerate_dashboards,
        )
        results.append(result)
        total_windows += result["windows"]
        total_changed_windows += result["changed"]
        if result.get("dashboard_regenerated"):
            dashboards_regenerated += 1

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    if args.dry_run:
        print(f"[dry-run] Would update {total_changed_windows} / {total_windows} windows "
              f"across {len(target_dates)} days.")
    else:
        days_with_changes = sum(1 for r in results if r["changed"] > 0)
        print(f"Done. Updated {total_changed_windows} / {total_windows} windows "
              f"across {days_with_changes} / {len(target_dates)} days.")
        if dashboards_regenerated:
            print(f"Regenerated {dashboards_regenerated} HTML dashboard(s).")

    if total_changed_windows == 0:
        print("All metrics are already up to date.")


if __name__ == "__main__":
    main()

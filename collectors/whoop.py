"""
WHOOP Collector

Fetches recovery, sleep, and strain data from the WHOOP API.
Delegates to the existing whoop-health-analysis skill scripts,
which handle token management correctly.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Path to the existing whoop skill scripts
WHOOP_SKILL_DIR = Path.home() / "clawd" / "skills" / "whoop-health-analysis"
WHOOP_DATA_SCRIPT = WHOOP_SKILL_DIR / "scripts" / "whoop_data.py"


def _run_whoop_script(command: list[str]) -> dict:
    """Run a whoop_data.py command and return parsed JSON."""
    result = subprocess.run(
        [sys.executable, str(WHOOP_DATA_SCRIPT)] + command,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(WHOOP_SKILL_DIR),
    )
    if result.returncode != 0:
        raise RuntimeError(f"whoop_data.py failed: {result.stderr[:200]}")
    return json.loads(result.stdout)


def collect(date_str: str) -> dict:
    """
    Collect WHOOP data for a given date (YYYY-MM-DD).

    Returns normalized dict with recovery, sleep, and strain data.
    Uses date-range filtering to get the right day's records.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    # Look back 1 day to catch the sleep that ended on this day
    start = (date - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (date + timedelta(days=1)).strftime("%Y-%m-%d")

    recovery_score = None
    hrv_rmssd_milli = None
    resting_heart_rate = None
    spo2_percentage = None
    skin_temp_celsius = None
    sleep_performance = None
    sleep_hours = None
    sleep_efficiency = None
    strain = None
    average_heart_rate = None
    max_heart_rate = None
    kilojoule = None

    # ── Recovery ──────────────────────────────────────────────────────────
    try:
        data = _run_whoop_script(["recovery", "--start", start, "--end", end])
        records = data.get("records", [])
        if records:
            # Find the record for this date
            target = None
            for r in records:
                updated = r.get("updated_at", "")
                created = r.get("created_at", "")
                if date_str in updated or date_str in created:
                    target = r
                    break
            if not target:
                target = records[0]
            score = target.get("score", {})
            recovery_score = score.get("recovery_score")
            hrv_rmssd_milli = score.get("hrv_rmssd_milli")
            resting_heart_rate = score.get("resting_heart_rate")
            spo2_percentage = score.get("spo2_percentage")
            skin_temp_celsius = score.get("skin_temp_celsius")
    except Exception as e:
        print(f"[whoop] Recovery fetch failed: {e}", file=sys.stderr)

    # ── Sleep ─────────────────────────────────────────────────────────────
    try:
        data = _run_whoop_script(["sleep", "--start", start, "--end", end])
        records = data.get("records", [])
        # Filter out naps, find main sleep
        main_sleeps = [r for r in records if not r.get("nap", False)]
        if main_sleeps:
            # Sort by end time descending, take most recent main sleep
            main_sleeps.sort(key=lambda r: r.get("end", ""), reverse=True)
            r = main_sleeps[0]
            score = r.get("score", {})
            sleep_performance = score.get("sleep_performance_percentage")
            sleep_efficiency = score.get("sleep_efficiency_percentage")
            # Compute hours from stage summary
            stages = score.get("stage_summary", {})
            total_ms = stages.get("total_in_bed_time_milli", 0)
            awake_ms = stages.get("total_awake_time_milli", 0)
            if total_ms:
                sleep_hours = round((total_ms - awake_ms) / 3_600_000, 2)
    except Exception as e:
        print(f"[whoop] Sleep fetch failed: {e}", file=sys.stderr)

    # ── Cycles (Strain) ───────────────────────────────────────────────────
    try:
        data = _run_whoop_script(["cycles", "--start", start, "--end", end])
        records = data.get("records", [])
        if records:
            # Find cycle for this date
            target = None
            for r in records:
                if date_str in r.get("start", ""):
                    target = r
                    break
            if not target:
                target = records[0]
            score = target.get("score", {})
            strain = score.get("strain")
            average_heart_rate = score.get("average_heart_rate")
            max_heart_rate = score.get("max_heart_rate")
            kilojoule = score.get("kilojoule")
    except Exception as e:
        print(f"[whoop] Cycle fetch failed: {e}", file=sys.stderr)

    return {
        "recovery_score": recovery_score,
        "hrv_rmssd_milli": hrv_rmssd_milli,
        "resting_heart_rate": resting_heart_rate,
        "spo2_percentage": spo2_percentage,
        "skin_temp_celsius": skin_temp_celsius,
        "sleep_performance": sleep_performance,
        "sleep_hours": sleep_hours,
        "sleep_efficiency": sleep_efficiency,
        "strain": strain,
        "average_heart_rate": average_heart_rate,
        "max_heart_rate": max_heart_rate,
        "kilojoule": kilojoule,
    }


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    result = collect(date)
    print(json.dumps(result, indent=2))

#!/usr/bin/env python3
"""
Presence Tracker — System Status

Quick health check and data overview for the full Presence Tracker system.
Runs entirely from local data — no API calls, no credentials required.

Usage:
    python3 scripts/status.py            # Full status report
    python3 scripts/status.py --json     # Machine-readable JSON output
    python3 scripts/status.py --brief    # One-line summary only

Exit codes:
    0 — system healthy (data fresh, no anomalies)
    1 — warning (data stale > 1 day, or previous anomaly alerts triggered)
    2 — error (no data collected yet, or critical gap)

Output covers:
  ① Data collection — how many days, date range, source coverage
  ② Recent metrics — 7-day average CLS, FDI, SDI, RAS trends
  ③ Cognitive Debt Index — 14-day accumulated fatigue level
  ④ WHOOP snapshot — latest recovery, HRV, sleep
  ⑤ Data source coverage — WHOOP, Calendar, Slack, RescueTime, Omi
  ⑥ ML model layer — training status, days until ready
  ⑦ Anomaly history — last triggered alert (if any)
  ⑧ Dashboard files — presence of HTML reports
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.store import (
    list_available_dates,
    read_day,
    read_summary,
    get_recent_summaries,
    get_data_age_days,
    get_data_staleness_days,
    get_date_range,
)
from analysis.ml_model import get_data_status
from analysis.anomaly_alerts import check_anomalies
from config import DATA_DIR


# ─── ANSI colour helpers ──────────────────────────────────────────────────────

def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m"


def _colour_cls(val: float) -> str:
    """Colour-code a CLS value: green → yellow → red."""
    s = f"{val:.2f}"
    if val < 0.30:
        return _green(s)
    elif val < 0.60:
        return _yellow(s)
    else:
        return _red(s)


def _colour_fdi(val: float) -> str:
    """Colour-code a FDI value: red → yellow → green (inverted)."""
    s = f"{val:.2f}"
    if val >= 0.70:
        return _green(s)
    elif val >= 0.40:
        return _yellow(s)
    else:
        return _red(s)


def _colour_recovery(val: float) -> str:
    """WHOOP-style recovery colours."""
    s = f"{val:.0f}%"
    if val >= 67:
        return _green(s)
    elif val >= 34:
        return _yellow(s)
    else:
        return _red(s)


def _bar(val: float, width: int = 12) -> str:
    """Compact bar using block characters: ▓▓▓░░░"""
    val = max(0.0, min(1.0, val))  # clamp to [0, 1]
    filled = round(val * width)
    return "▓" * filled + "░" * (width - filled)


def _trend_arrow(values: list[float]) -> str:
    """Return ↑ ↓ or → for the trend direction of the last N values."""
    if len(values) < 2:
        return "·"
    delta = values[-1] - values[-2]
    if delta > 0.03:
        return "↑"
    elif delta < -0.03:
        return "↓"
    return "→"


# ─── Data gathering ───────────────────────────────────────────────────────────

def _gather_status() -> dict:
    """Collect all status information from local files."""
    dates = list_available_dates()
    n_days = len(dates)

    # Date range
    oldest, newest = get_date_range() if n_days > 0 else (None, None)
    age_days = get_data_age_days()          # count of collected days (for ML progress)
    staleness_days = get_data_staleness_days()  # days since last ingestion (for freshness)

    # Rolling summary → recent trends
    recent = get_recent_summaries(days=7)
    recent_cls = [
        s["metrics_avg"]["cognitive_load_score"]
        for s in recent
        if s.get("metrics_avg", {}).get("cognitive_load_score") is not None
    ]
    recent_fdi = [
        s.get("focus_quality", {}).get("active_fdi") or s["metrics_avg"]["focus_depth_index"]
        for s in recent
        if s.get("metrics_avg", {}).get("focus_depth_index") is not None
    ]
    recent_ras = [
        s["metrics_avg"]["recovery_alignment_score"]
        for s in recent
        if s.get("metrics_avg", {}).get("recovery_alignment_score") is not None
    ]

    # WHOOP snapshot from most recent day
    whoop_latest: dict = {}
    latest_date = dates[-1] if dates else None
    if latest_date:
        latest_windows = read_day(latest_date)
        if latest_windows:
            w = latest_windows[0].get("whoop", {})
            whoop_latest = {
                "recovery_score": w.get("recovery_score"),
                "hrv_rmssd_milli": w.get("hrv_rmssd_milli"),
                "resting_heart_rate": w.get("resting_heart_rate"),
                "sleep_performance": w.get("sleep_performance"),
                "sleep_hours": w.get("sleep_hours"),
                "strain": w.get("strain"),
            }

    # Source coverage for recent days
    source_coverage: dict[str, int] = {}
    for day_date in dates[-7:]:
        day_windows = read_day(day_date)
        if day_windows:
            sources = day_windows[0].get("metadata", {}).get("sources_available", [])
            for src in sources:
                source_coverage[src] = source_coverage.get(src, 0) + 1

    # ML model status
    ml_status = get_data_status()

    # Dashboard files
    dashboard_dir = DATA_DIR / "dashboard"
    dashboard_files = sorted(dashboard_dir.glob("*.html")) if dashboard_dir.exists() else []

    # Anomaly history — check most recent anomaly alerts log if it exists
    # (check last 7 days of data for any triggered anomalies via anomaly_alerts.py)
    last_anomaly: dict = {}
    try:
        if latest_date:
            result = check_anomalies(latest_date)
            if result.get("any_triggered"):
                triggered = [k for k, v in result.get("alerts", {}).items() if v]
                last_anomaly = {
                    "date": latest_date,
                    "triggered": triggered,
                    "count": len(triggered),
                }
    except Exception:
        pass

    # Active sources across last 7 days
    source_days = len(dates[-7:]) if dates else 0

    # Cognitive Debt Index (v8.0)
    cdi_status: dict = {}
    try:
        from analysis.cognitive_debt import compute_cdi, format_cdi_line
        if latest_date:
            debt = compute_cdi(latest_date)
            if debt.is_meaningful:
                cdi_status = {
                    "cdi": debt.cdi,
                    "tier": debt.tier,
                    "days_in_deficit": debt.days_in_deficit,
                    "days_used": debt.days_used,
                    "trend_5d": debt.trend_5d,
                    "line": format_cdi_line(debt),
                }
    except Exception:
        pass

    return {
        "data": {
            "n_days": n_days,
            "oldest": oldest,
            "newest": newest,
            "age_days": age_days,
            "staleness_days": staleness_days,
            "dates": dates,
        },
        "metrics": {
            "recent_cls": recent_cls,
            "recent_fdi": recent_fdi,
            "recent_ras": recent_ras,
            "avg_cls_7d": round(sum(recent_cls) / len(recent_cls), 3) if recent_cls else None,
            "avg_fdi_7d": round(sum(recent_fdi) / len(recent_fdi), 3) if recent_fdi else None,
            "avg_ras_7d": round(sum(recent_ras) / len(recent_ras), 3) if recent_ras else None,
        },
        "whoop": whoop_latest,
        "sources": {
            "coverage": source_coverage,
            "source_days": source_days,
        },
        "ml": ml_status,
        "dashboard": {
            "count": len(dashboard_files),
            "latest": dashboard_files[-1].name if dashboard_files else None,
        },
        "anomalies": last_anomaly,
        "cognitive_debt": cdi_status,
    }


# ─── Health determination ─────────────────────────────────────────────────────

def _health_code(status: dict) -> int:
    """
    Determine overall system health.

    0 = healthy  (data fresh, no recent anomalies)
    1 = warning  (data stale, or anomalies triggered)
    2 = error    (no data, or very large gap)
    """
    n_days = status["data"]["n_days"]
    # Use staleness_days (calendar days since last ingestion) for freshness checks.
    # age_days is the *count* of collected days and must not be used here.
    staleness = status["data"].get("staleness_days", status["data"]["age_days"])

    if n_days == 0:
        return 2

    if staleness > 3:
        return 2  # Data collection has stopped

    if staleness > 1:
        return 1  # Missed a day

    if status["anomalies"].get("count", 0) > 0:
        return 1

    return 0


# ─── Formatters ───────────────────────────────────────────────────────────────

def _fmt_data_section(data: dict) -> list[str]:
    """Format the data collection section."""
    lines = [_bold("① Data Collection")]
    n = data["n_days"]
    # staleness_days = calendar days since last ingestion (0 = today, 1 = yesterday …)
    # age_days = total count of collected days (used for ML progress only)
    staleness = data.get("staleness_days", data["age_days"])

    if n == 0:
        lines.append(f"  {_red('✗ No data collected yet')}")
        lines.append(f"  Run: python3 scripts/run_daily.py")
        return lines

    # Date range
    lines.append(f"  Days collected:  {_green(str(n))} days  ({data['oldest']} → {data['newest']})")

    # Freshness — use staleness (days since last ingestion), not total day count
    if staleness == 0:
        freshness = _green("today")
    elif staleness == 1:
        freshness = _yellow("yesterday")
    else:
        freshness = _red(f"{staleness} days ago")
    lines.append(f"  Last ingested:   {freshness}")

    # ML progress — use total collected days (age_days = count)
    ml_min = 60
    progress_pct = min(100, round(n / ml_min * 100))
    bar = _bar(min(1.0, n / ml_min), width=16)
    lines.append(f"  ML progress:     {bar} {progress_pct}% ({n}/{ml_min} days needed)")

    return lines


def _fmt_metrics_section(metrics: dict) -> list[str]:
    """Format recent metric trends."""
    lines = [_bold("② Recent Metrics (7-day)")]

    avg_cls = metrics["avg_cls_7d"]
    avg_fdi = metrics["avg_fdi_7d"]
    avg_ras = metrics["avg_ras_7d"]

    if avg_cls is None:
        lines.append(f"  {_dim('No metric history yet')}")
        return lines

    cls_trend = _trend_arrow(metrics["recent_cls"])
    fdi_trend = _trend_arrow(metrics["recent_fdi"])
    ras_trend = _trend_arrow(metrics["recent_ras"])

    lines.append(f"  CLS (cognitive load):  {_colour_cls(avg_cls)}  {cls_trend}  {_bar(avg_cls)}")
    lines.append(f"  FDI (focus depth):     {_colour_fdi(avg_fdi)}  {fdi_trend}  {_bar(avg_fdi)}")
    lines.append(f"  RAS (alignment):       {_colour_fdi(avg_ras)}  {ras_trend}  {_bar(avg_ras)}")

    return lines


def _fmt_whoop_section(whoop: dict, date: str) -> list[str]:
    """Format WHOOP snapshot."""
    lines = [_bold(f"④ WHOOP Snapshot ({date or 'n/a'})")]

    recovery = whoop.get("recovery_score")
    hrv = whoop.get("hrv_rmssd_milli")
    rhr = whoop.get("resting_heart_rate")
    sleep_h = whoop.get("sleep_hours")
    sleep_p = whoop.get("sleep_performance")

    if recovery is None:
        lines.append(f"  {_dim('No WHOOP data')}")
        return lines

    rec_str = _colour_recovery(recovery) if recovery else _dim("n/a")
    hrv_str = f"{hrv:.0f}ms" if hrv else _dim("n/a")
    rhr_str = f"{rhr:.0f}bpm" if rhr else _dim("n/a")
    sleep_str = f"{sleep_h:.1f}h ({sleep_p:.0f}%)" if sleep_h and sleep_p else _dim("n/a")

    lines.append(f"  Recovery:  {rec_str}   HRV: {hrv_str}   RHR: {rhr_str}")
    lines.append(f"  Sleep:     {sleep_str}")

    return lines


def _fmt_sources_section(sources: dict) -> list[str]:
    """Format data source coverage."""
    lines = [_bold("⑤ Data Source Coverage (last 7 days)")]

    coverage = sources.get("coverage", {})
    days = sources.get("source_days", 0)

    if not coverage:
        lines.append(f"  {_dim('No source data')}")
        return lines

    all_sources = ["whoop", "calendar", "slack", "rescuetime", "omi"]
    for src in all_sources:
        count = coverage.get(src, 0)
        pct = round(count / days * 100) if days else 0
        bar = _bar(pct / 100.0, width=10)
        if count == days:
            indicator = _green("✓")
        elif count > 0:
            indicator = _yellow("~")
        else:
            indicator = _dim("·")
        lines.append(f"  {indicator} {src:<12}  {bar}  {count}/{days} days")

    return lines


def _fmt_ml_section(ml: dict) -> list[str]:
    """Format ML model status."""
    lines = [_bold("⑥ ML Model Layer")]

    n = ml.get("days_of_data", 0)
    ml_min = ml.get("min_days_required", 60)
    ready = ml.get("ready_to_train", False)
    models = ml.get("models_trained", {})
    last_trained = ml.get("last_trained")

    if ready:
        lines.append(f"  {_green('✓ Sufficient data for training')}  ({n}/{ml_min} days)")
    else:
        remaining = ml_min - n
        lines.append(f"  {_yellow(f'⧗ {remaining} more days needed')}  ({n}/{ml_min} days)")

    trained_models = [k for k, v in models.items() if v]
    if trained_models:
        lines.append(f"  Trained models:  {', '.join(trained_models)}")
        if last_trained:
            lines.append(f"  Last trained:    {last_trained}")
    else:
        lines.append(f"  Models:          {_dim('not yet trained')}")
        lines.append(f"  Train with:      python3 analysis/ml_model.py --train")

    return lines


def _fmt_anomaly_section(anomaly: dict) -> list[str]:
    """Format anomaly alert status."""
    lines = [_bold("⑦ Anomaly Alerts")]

    if not anomaly:
        lines.append(f"  {_green('✓ No anomalies triggered (latest day)')} ")
    else:
        triggered = anomaly.get("triggered", [])
        date = anomaly.get("date", "")
        lines.append(f"  {_yellow('⚠ Alerts triggered')} on {date}:")
        for t in triggered:
            alert_labels = {
                "cls_spike": "CLS spike (cognitive overload)",
                "fdi_collapse": "FDI collapse (focus fragmentation)",
                "recovery_streak": "Recovery misalignment streak",
            }
            lines.append(f"    · {alert_labels.get(t, t)}")

    return lines


def _fmt_cognitive_debt_section(cdi: dict) -> list[str]:
    """Format Cognitive Debt Index status."""
    lines = [_bold("③ Cognitive Debt Index (14-day)")]

    if not cdi:
        lines.append(f"  {_dim('Not yet meaningful (need ≥ 3 days of data)')}")
        return lines

    score = cdi.get("cdi", 50.0)
    tier = cdi.get("tier", "balanced")
    days_deficit = cdi.get("days_in_deficit", 0)
    days_used = cdi.get("days_used", 0)
    trend_5d = cdi.get("trend_5d")

    # Colour by tier
    tier_colour = {
        "surplus":  _green,
        "balanced": _green,
        "loading":  _yellow,
        "fatigued": _red,
        "critical": _red,
    }.get(tier, _dim)

    tier_emoji = {
        "surplus":  "🟢",
        "balanced": "🟡",
        "loading":  "🟠",
        "fatigued": "🔴",
        "critical": "🚨",
    }.get(tier, "⚪")

    score_str = tier_colour(f"{score:.0f}/100")
    lines.append(f"  CDI: {score_str}  {tier_emoji} {tier.capitalize()}")
    lines.append(f"  Deficit days: {days_deficit} of last {days_used} days")

    if trend_5d is not None:
        if trend_5d > 0.02:
            lines.append(f"  5d trend: {_red('↑ fatigue accumulating')}")
        elif trend_5d < -0.02:
            lines.append(f"  5d trend: {_green('↓ recovering')}")
        else:
            lines.append(f"  5d trend: stable")

    return lines


def _fmt_dashboard_section(dashboard: dict) -> list[str]:
    """Format dashboard file status."""
    lines = [_bold("⑧ HTML Dashboards")]

    count = dashboard.get("count", 0)
    latest = dashboard.get("latest")

    if count == 0:
        lines.append(f"  {_dim('No dashboards generated yet')}")
        lines.append(f"  Generate: python3 scripts/generate_dashboard.py")
    else:
        lines.append(f"  {_green(f'✓ {count} dashboards')}  (latest: {latest})")
        dashboard_dir = DATA_DIR / "dashboard"
        lines.append(f"  Path: {dashboard_dir}/")

    return lines


# ─── Main output ──────────────────────────────────────────────────────────────

def print_status(no_colour: bool = False) -> int:
    """
    Print the full status report. Returns health exit code.
    """
    status = _gather_status()
    health = _health_code(status)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    latest_date = status["data"].get("newest")

    # Header
    health_icons = {0: _green("● HEALTHY"), 1: _yellow("◐ WARNING"), 2: _red("○ ERROR")}
    print(f"\n{_bold('Presence Tracker — System Status')}  {_dim(now)}")
    print(f"Status: {health_icons[health]}")
    print()

    # All sections
    sections = [
        _fmt_data_section(status["data"]),
        _fmt_metrics_section(status["metrics"]),
        _fmt_cognitive_debt_section(status.get("cognitive_debt", {})),
        _fmt_whoop_section(status["whoop"], latest_date or ""),
        _fmt_sources_section(status["sources"]),
        _fmt_ml_section(status["ml"]),
        _fmt_anomaly_section(status["anomalies"]),
        _fmt_dashboard_section(status["dashboard"]),
    ]

    for section in sections:
        for line in section:
            print(line)
        print()

    # Footer hints
    if status["data"]["n_days"] > 0:
        staleness = status["data"].get("staleness_days", status["data"]["age_days"])
        if staleness > 0:
            print(_dim(f"  → Run: python3 scripts/run_daily.py  (last ingestion: {staleness}d ago)"))

    return health


def print_brief() -> int:
    """One-line summary for scripting / notifications."""
    status = _gather_status()
    health = _health_code(status)

    n = status["data"]["n_days"]
    staleness = status["data"].get("staleness_days", status["data"]["age_days"])
    cls = status["metrics"].get("avg_cls_7d")
    fdi = status["metrics"].get("avg_fdi_7d")
    ml_days = status["ml"].get("days_of_data", 0)
    ml_min = status["ml"].get("min_days_required", 60)
    newest = status["data"].get("newest", "—")

    parts = [f"data={n}d", f"latest={newest}", f"age={staleness}d"]
    if cls is not None:
        parts.append(f"cls={cls:.2f}")
    if fdi is not None:
        parts.append(f"fdi={fdi:.2f}")
    parts.append(f"ml={ml_days}/{ml_min}")

    health_label = {0: "OK", 1: "WARN", 2: "ERROR"}[health]
    print(f"[{health_label}] {' | '.join(parts)}")

    return health


def print_json() -> int:
    """Machine-readable JSON output."""
    status = _gather_status()
    health = _health_code(status)
    status["health"] = {0: "healthy", 1: "warning", 2: "error"}[health]
    status["generated_at"] = datetime.now().isoformat()
    print(json.dumps(status, indent=2, default=str))
    return health


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Presence Tracker — System Status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0  healthy — data fresh, no anomalies
  1  warning — data stale or anomalies triggered
  2  error   — no data or critical gap
""",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output machine-readable JSON",
    )
    parser.add_argument(
        "--brief", "-b",
        action="store_true",
        help="One-line summary only",
    )
    parser.add_argument(
        "--no-colour",
        action="store_true",
        help="Disable ANSI colour output",
    )
    args = parser.parse_args()

    if args.json:
        code = print_json()
    elif args.brief:
        code = print_brief()
    else:
        code = print_status(no_colour=args.no_colour)

    sys.exit(code)


if __name__ == "__main__":
    main()

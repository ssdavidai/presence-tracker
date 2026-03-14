"""
Presence Tracker — Multi-Source Anomaly Alerts (v5)

After each daily ingestion this module compares today's metrics against a
7-day rolling baseline and fires Slack DM alerts to David when thresholds
are exceeded.

Alert conditions (all configurable via config.py):

  1. CLS spike      — today's avg CLS > baseline_mean + 2 × baseline_std
                      (requires ≥ 3 baseline days to be meaningful)

  2. FDI collapse   — today's active FDI dropped > 30 % vs 7-day avg
                      (only fires when today and baseline both have data)

  3. Recovery misalignment streak — RAS < RECOVERY_MISALIGN_THRESHOLD for
                                     3 or more consecutive days ending today

Design principles:
  - Pure detection functions (no I/O) — safe to unit test
  - send_anomaly_alerts() is the single I/O entry point
  - Graceful degradation: missing data → alert skipped, never crashes
  - No alert spam: each condition fires at most once per day

Usage:
    # Called automatically by DailyIngestionWorkflow after ingest_day
    from analysis.anomaly_alerts import check_anomalies, send_anomaly_alerts

    # Check-only (returns structured dict, no Slack)
    result = check_anomalies(date_str)

    # Check + send Slack DM if anything triggered
    alerts_sent = send_anomaly_alerts(date_str)

    # CLI: python3 analysis/anomaly_alerts.py [DATE] [--check-only]
"""

import json
import math
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root so we can import config and engine modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    GATEWAY_URL,
    GATEWAY_TOKEN,
    SLACK_DM_CHANNEL,
)
from engine.store import list_available_dates, read_day, read_summary


# ─── Thresholds (mirror config.py values; import from there when added) ───────

# Minimum days of baseline history before CLS spike alert fires
CLS_SPIKE_MIN_BASELINE_DAYS = 3

# CLS spike: today > mean + N × std
CLS_SPIKE_STD_MULTIPLIER = 2.0

# FDI collapse: today's active_fdi dropped more than this fraction vs baseline
FDI_COLLAPSE_THRESHOLD = 0.30   # 30 %

# Minimum baseline FDI to compare against (avoids noise on near-zero days)
FDI_MIN_BASELINE = 0.05

# Recovery misalignment: RAS below this is "misaligned"
RECOVERY_MISALIGN_RAS = 0.45

# How many consecutive misaligned days trigger the streak alert
RECOVERY_MISALIGN_STREAK = 3

# Baseline window (days before today used to compute mean/std)
BASELINE_WINDOW_DAYS = 7


# ─── Pure detection helpers ───────────────────────────────────────────────────

def _mean(values: list[float]) -> Optional[float]:
    """Return arithmetic mean or None for empty list."""
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float:
    """Population standard deviation (returns 0 for < 2 samples)."""
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _get_daily_cls(date_str: str) -> Optional[float]:
    """Return avg CLS for working-hour windows on date_str, or None."""
    windows = read_day(date_str)
    cls_vals = [
        w["metrics"]["cognitive_load_score"]
        for w in windows
        if w.get("metadata", {}).get("is_working_hours")
        and "metrics" in w
    ]
    return _mean(cls_vals)


def _get_daily_active_fdi(date_str: str) -> Optional[float]:
    """
    Return active FDI for date_str.

    Reads from summarize_day output stored in rolling.json when available,
    otherwise computes from raw windows.  Active FDI only counts windows
    where at least one behavioral signal was present.
    """
    # Try rolling summary first (fast path — already computed)
    summary = read_summary()
    day_data = summary.get("days", {}).get(date_str)
    if day_data:
        fdi = day_data.get("focus_quality", {}).get("active_fdi")
        if fdi is not None:
            return fdi

    # Fallback: compute from raw windows
    windows = read_day(date_str)
    active_fdi_vals = [
        w["metrics"]["focus_depth_index"]
        for w in windows
        if w.get("metadata", {}).get("is_active_window")
        and w.get("metadata", {}).get("is_working_hours")
        and "metrics" in w
    ]
    return _mean(active_fdi_vals)


def _get_daily_ras(date_str: str) -> Optional[float]:
    """Return avg RAS for date_str (all windows), or None."""
    # Try rolling summary first
    summary = read_summary()
    day_data = summary.get("days", {}).get(date_str)
    if day_data:
        ras = day_data.get("metrics_avg", {}).get("recovery_alignment_score")
        if ras is not None:
            return ras

    # Fallback: raw windows
    windows = read_day(date_str)
    ras_vals = [
        w["metrics"]["recovery_alignment_score"]
        for w in windows
        if "metrics" in w
    ]
    return _mean(ras_vals)


def _baseline_dates(today: str, n: int = BASELINE_WINDOW_DAYS) -> list[str]:
    """Return up to N available dates strictly before today."""
    available = set(list_available_dates())
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    dates = []
    for i in range(1, n + 30):   # look back further if sparse
        candidate = (today_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        if candidate in available:
            dates.append(candidate)
        if len(dates) == n:
            break
    return dates


# ─── Detection functions (pure — no I/O) ─────────────────────────────────────

def detect_cls_spike(today: str) -> Optional[dict]:
    """
    Detect if today's CLS is > baseline_mean + 2σ.

    Returns None if no alert, or a dict with keys:
      today_cls, baseline_mean, baseline_std, threshold, days_used
    """
    baseline_dates = _baseline_dates(today)
    if len(baseline_dates) < CLS_SPIKE_MIN_BASELINE_DAYS:
        return None   # not enough history

    today_cls = _get_daily_cls(today)
    if today_cls is None:
        return None

    baseline_vals = [v for d in baseline_dates if (v := _get_daily_cls(d)) is not None]
    if len(baseline_vals) < CLS_SPIKE_MIN_BASELINE_DAYS:
        return None

    mean_cls = sum(baseline_vals) / len(baseline_vals)
    std_cls = _std(baseline_vals)
    threshold = mean_cls + CLS_SPIKE_STD_MULTIPLIER * std_cls

    if today_cls > threshold:
        return {
            "today_cls": today_cls,
            "baseline_mean": mean_cls,
            "baseline_std": std_cls,
            "threshold": threshold,
            "days_used": len(baseline_vals),
        }
    return None


def detect_fdi_collapse(today: str) -> Optional[dict]:
    """
    Detect if today's active FDI dropped > 30 % vs 7-day average.

    Returns None if no alert, or a dict with:
      today_fdi, baseline_fdi, drop_pct
    """
    baseline_dates = _baseline_dates(today)
    if not baseline_dates:
        return None

    today_fdi = _get_daily_active_fdi(today)
    if today_fdi is None:
        return None

    baseline_vals = [v for d in baseline_dates if (v := _get_daily_active_fdi(d)) is not None]
    if not baseline_vals:
        return None

    baseline_fdi = sum(baseline_vals) / len(baseline_vals)
    if baseline_fdi < FDI_MIN_BASELINE:
        return None   # too noisy to compare

    drop_pct = (baseline_fdi - today_fdi) / baseline_fdi
    if drop_pct > FDI_COLLAPSE_THRESHOLD:
        return {
            "today_fdi": today_fdi,
            "baseline_fdi": baseline_fdi,
            "drop_pct": drop_pct,
            "days_used": len(baseline_vals),
        }
    return None


def detect_recovery_misalignment_streak(today: str) -> Optional[dict]:
    """
    Detect a streak of ≥ 3 consecutive days (ending today) with RAS < 0.45.

    Returns None if no alert, or a dict with:
      streak_days, ras_values (list, newest first)
    """
    available = set(list_available_dates())
    today_dt = datetime.strptime(today, "%Y-%m-%d")

    streak_days = []
    ras_values = []
    for i in range(RECOVERY_MISALIGN_STREAK + BASELINE_WINDOW_DAYS):
        candidate = (today_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        if candidate not in available:
            break   # gap in data — reset streak
        ras = _get_daily_ras(candidate)
        if ras is None or ras >= RECOVERY_MISALIGN_RAS:
            break   # streak broken
        streak_days.append(candidate)
        ras_values.append(ras)

    if len(streak_days) >= RECOVERY_MISALIGN_STREAK:
        return {
            "streak_days": len(streak_days),
            "ras_values": ras_values,
            "avg_ras": sum(ras_values) / len(ras_values),
        }
    return None


# ─── Aggregated check ─────────────────────────────────────────────────────────

def detect_cognitive_debt_alert(date_str: str) -> Optional[dict]:
    """
    Check whether the Cognitive Debt Index is in a fatigued/critical state.

    Returns a dict when CDI is fatigued (≥ 70) or critical (≥ 85) and the
    result is meaningful (≥ 3 days of data).  Returns None otherwise.

    This is a fourth anomaly check — complementary to the existing CLS spike,
    FDI collapse, and recovery streak alerts.  While those detect single-day
    events or streaks of one metric, CDI captures sustained multi-day
    accumulation across all signals.  A week of moderately elevated CLS with
    mediocre WHOOP recovery might not trigger any single-metric alert but will
    push the CDI into fatigued territory.

    Returns:
    {
      "cdi": float,
      "tier": "fatigued" | "critical",
      "days_in_deficit": int,
      "days_used": int,
      "trend_5d": float or None,
      "alert_message": str,   ← formatted by format_cdi_alert()
    }
    """
    try:
        from analysis.cognitive_debt import compute_cdi, format_cdi_alert
        debt = compute_cdi(date_str)
        if not debt.is_meaningful:
            return None
        if debt.tier not in ("fatigued", "critical"):
            return None
        alert_message = format_cdi_alert(debt)
        return {
            "cdi": debt.cdi,
            "tier": debt.tier,
            "days_in_deficit": debt.days_in_deficit,
            "days_used": debt.days_used,
            "trend_5d": debt.trend_5d,
            "alert_message": alert_message,
        }
    except Exception:
        return None


def check_anomalies(date_str: str = None) -> dict:
    """
    Run all anomaly checks for date_str (defaults to today).

    Returns:
    {
      "date": "2026-03-14",
      "alerts": {
        "cls_spike": {...} or None,
        "fdi_collapse": {...} or None,
        "recovery_streak": {...} or None,
        "cognitive_debt": {...} or None,   ← v5.1: CDI fatigue alert
      },
      "any_triggered": bool,
    }
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    cls_spike = detect_cls_spike(date_str)
    fdi_collapse = detect_fdi_collapse(date_str)
    recovery_streak = detect_recovery_misalignment_streak(date_str)
    cognitive_debt = detect_cognitive_debt_alert(date_str)

    alerts = {
        "cls_spike": cls_spike,
        "fdi_collapse": fdi_collapse,
        "recovery_streak": recovery_streak,
        "cognitive_debt": cognitive_debt,
    }

    return {
        "date": date_str,
        "alerts": alerts,
        "any_triggered": any(v is not None for v in alerts.values()),
    }


# ─── Message formatting ───────────────────────────────────────────────────────

def _fmt_pct(value: float) -> str:
    return f"{value:.0%}"


def format_alert_message(result: dict) -> str:
    """
    Build a human-readable Slack DM alert message from check_anomalies() output.

    Returns empty string if nothing triggered.
    """
    alerts = result.get("alerts", {})
    date = result.get("date", "today")
    lines = []

    cls_alert = alerts.get("cls_spike")
    if cls_alert:
        lines.append(
            f"⚡ *Cognitive load spike* on {date}\n"
            f"   Today's CLS: *{cls_alert['today_cls']:.2f}* "
            f"(baseline {cls_alert['baseline_mean']:.2f} ± {cls_alert['baseline_std']:.2f}, "
            f"threshold {cls_alert['threshold']:.2f})\n"
            f"   Your brain had an unusually demanding day — well above your personal norm."
        )

    fdi_alert = alerts.get("fdi_collapse")
    if fdi_alert:
        drop = fdi_alert["drop_pct"]
        lines.append(
            f"📉 *Focus quality collapsed* on {date}\n"
            f"   Active FDI: *{fdi_alert['today_fdi']:.2f}* "
            f"vs 7-day avg *{fdi_alert['baseline_fdi']:.2f}* "
            f"(↓ {_fmt_pct(drop)})\n"
            f"   Deep focus was significantly harder today than your recent baseline."
        )

    streak_alert = alerts.get("recovery_streak")
    if streak_alert:
        n = streak_alert["streak_days"]
        avg = streak_alert["avg_ras"]
        lines.append(
            f"🔴 *Recovery misalignment streak — {n} days running*\n"
            f"   Avg RAS: *{avg:.2f}* (below 0.45 threshold)\n"
            f"   You've been operating above your physiological capacity for {n} consecutive days. "
            f"Consider a recovery day."
        )

    cdi_alert = alerts.get("cognitive_debt")
    if cdi_alert:
        alert_msg = cdi_alert.get("alert_message", "")
        if alert_msg:
            lines.append(alert_msg)

    if not lines:
        return ""

    header = f"🧠 *Presence Tracker — Anomaly Alert ({date})*\n"
    return header + "\n\n".join(lines)


# ─── I/O entry point ─────────────────────────────────────────────────────────

def _send_slack_dm(message: str) -> bool:
    """Send a message to David's Slack DM via the OpenClaw gateway."""
    try:
        headers = {
            "Authorization": f"Bearer {GATEWAY_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "tool": "message",
            "args": {
                "action": "send",
                "channel": "slack",
                "target": SLACK_DM_CHANNEL,
                "message": message,
            }
        }).encode()
        req = urllib.request.Request(
            f"{GATEWAY_URL}/tools/invoke",
            data=payload,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"[anomaly_alerts] Slack DM failed: {e}", file=sys.stderr)
        return False


def send_anomaly_alerts(date_str: str = None) -> int:
    """
    Run all anomaly checks and send a Slack DM if anything triggered.

    Returns the number of alerts sent (0 or 1 — we send one batched DM).
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    result = check_anomalies(date_str)
    if not result["any_triggered"]:
        return 0

    message = format_alert_message(result)
    if not message:
        return 0

    ok = _send_slack_dm(message)
    return 1 if ok else 0


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check presence tracker anomaly alerts")
    parser.add_argument("date", nargs="?", help="Date to check (YYYY-MM-DD, default: today)")
    parser.add_argument("--check-only", action="store_true", help="Print without sending Slack")
    parser.add_argument("--json", action="store_true", help="Output raw JSON result")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    result = check_anomalies(date_str)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)

    msg = format_alert_message(result)
    if msg:
        print(msg)
        if not args.check_only:
            n = send_anomaly_alerts(date_str)
            print(f"\n[{n} alert(s) sent to Slack]")
        else:
            print("\n[--check-only: not sent to Slack]")
    else:
        print(f"No anomalies detected for {date_str}")

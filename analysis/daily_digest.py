"""
Presence Tracker — Daily Digest

Sends David a personal Slack DM at the end of each day with a
concise cognitive load summary: how he spent his mental energy,
whether he was within physiological capacity, and one insight.

This is the primary human-facing output of the Presence Tracker —
the difference between data sitting in JSONL files and David actually
knowing how his day went cognitively.

v1.2 — Multi-day trend context:
  The digest now computes a trend context from recent history (up to 7 days).
  This detects:
  - HRV decline or improvement streaks (3+ consecutive days)
  - Consecutive above-capacity days (RAS < 0.45 for 3+ days)
  - CLS vs personal baseline (today vs 7-day average)
  - HRV vs personal baseline (today vs 7-day average)
  The most significant trend is surfaced as the insight, replacing generic
  single-day observations with multi-day pattern detection.

v1.3 — Hourly CLS sparkline:
  The digest now includes a compact hourly cognitive load curve covering
  all working hours (7am–10pm).  Each hour maps to one character:
    ░ = very light (< 0.10)   — idle / no demand
    ▒ = light     (0.10–0.25) — low engagement
    ▓ = moderate  (0.25–0.50) — meaningful load
    █ = heavy     (≥ 0.50)    — high cognitive demand

  The sparkline gives an at-a-glance picture of when load was heavy or
  light across the full day — something no single average number conveys.
  It also pinpoints where peak effort was concentrated (e.g. morning
  vs afternoon) and shows idle blocks between active periods.

  Implementation:
  - compute_hourly_cls_curve() aggregates per-window CLS into hourly means
    across the 7am–10pm working window (15 hours = 15 chars)
  - _format_hourly_sparkline() renders the array to a Unicode block string
  - Both are pure functions with no external dependencies, fully testable
  - The sparkline is added to the digest dict as "hourly_cls_curve" and
    rendered in format_digest_message() as a single compact line
"""

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GATEWAY_URL, GATEWAY_TOKEN, SLACK_DM_CHANNEL


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _score_bar(value: float, width: int = 10) -> str:
    """Convert a 0-1 score to a visual bar: ▓▓▓▓▓░░░░░"""
    filled = round(value * width)
    return "▓" * filled + "░" * (width - filled)


def _fmt_score(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.0%}"


def _cls_label(cls: float) -> str:
    """Human label for a CLS value."""
    if cls < 0.20:
        return "light"
    elif cls < 0.40:
        return "moderate"
    elif cls < 0.60:
        return "heavy"
    elif cls < 0.80:
        return "intense"
    else:
        return "maximal"


def _ras_label(ras: float) -> str:
    """Human label for a RAS value."""
    if ras >= 0.80:
        return "well within capacity"
    elif ras >= 0.60:
        return "within capacity"
    elif ras >= 0.40:
        return "slightly stretched"
    elif ras >= 0.20:
        return "over capacity"
    else:
        return "significantly over capacity"


def _fdi_label(fdi: float) -> str:
    """Human label for a FDI value (active hours only)."""
    if fdi >= 0.80:
        return "deep"
    elif fdi >= 0.60:
        return "reasonable"
    elif fdi >= 0.40:
        return "fragmented"
    else:
        return "highly fragmented"


# ─── Hourly CLS sparkline ─────────────────────────────────────────────────────

# Working hours covered by the sparkline (inclusive start, exclusive end)
_SPARKLINE_START_HOUR = 7
_SPARKLINE_END_HOUR = 22  # up to but not including 22:00

# Unicode block characters ordered from light → heavy load
# Each threshold is the *lower bound* for that character
_SPARKLINE_THRESHOLDS = [
    (0.50, "█"),   # heavy
    (0.25, "▓"),   # moderate
    (0.10, "▒"),   # light
    (0.0,  "░"),   # very light / idle
]


def compute_hourly_cls_curve(windows: list[dict]) -> list[Optional[float]]:
    """
    Compute mean CLS per working hour for the sparkline.

    Returns a list of length (_SPARKLINE_END_HOUR - _SPARKLINE_START_HOUR),
    i.e. one value per hour from 7am to 9pm (15 values for 7–21 inclusive).
    Each entry is the mean CLS across the four 15-min windows in that hour,
    or None if no windows existed for that hour (should not happen in practice).

    Uses *all* working-hour windows (not just active ones) because a quiet
    hour with CLS=0.02 is meaningfully different from no data at all, and
    the sparkline is intended to show the full shape of the day.

    Args:
        windows: list of 96 window dicts for a single day

    Returns:
        list[Optional[float]]: hourly mean CLS values, length 15
    """
    n_hours = _SPARKLINE_END_HOUR - _SPARKLINE_START_HOUR
    hourly: list[list[float]] = [[] for _ in range(n_hours)]

    for w in windows:
        h = w["metadata"]["hour_of_day"]
        if _SPARKLINE_START_HOUR <= h < _SPARKLINE_END_HOUR:
            idx = h - _SPARKLINE_START_HOUR
            hourly[idx].append(w["metrics"]["cognitive_load_score"])

    result: list[Optional[float]] = []
    for vals in hourly:
        if vals:
            result.append(round(sum(vals) / len(vals), 4))
        else:
            result.append(None)
    return result


def _format_hourly_sparkline(hourly_cls: list[Optional[float]]) -> str:
    """
    Render a list of hourly CLS means as a Unicode block sparkline string.

    Each value maps to one character based on its magnitude:
      ░ < 0.10  (very light — idle or minimal engagement)
      ▒ 0.10–0.25  (light — some activity)
      ▓ 0.25–0.50  (moderate — meaningful cognitive load)
      █ ≥ 0.50  (heavy — high demand)

    None values (missing data) render as a dash character.

    Args:
        hourly_cls: list of Optional[float] from compute_hourly_cls_curve()

    Returns:
        str: e.g. "░░░▒▓▓█▓▒░░░░░░"
    """
    chars = []
    for val in hourly_cls:
        if val is None:
            chars.append("·")
            continue
        char = "░"  # default: very light
        for threshold, symbol in _SPARKLINE_THRESHOLDS:
            if val >= threshold:
                char = symbol
                break
        chars.append(char)
    return "".join(chars)


# ─── Multi-day trend context ─────────────────────────────────────────────────

def compute_trend_context(today_date: str, lookback_days: int = 7) -> dict:
    """
    Build a multi-day trend context from recent daily summaries.

    Reads the rolling summary store and computes:
    - hrv_trend: direction and streak of HRV change ('declining', 'improving', 'stable')
    - hrv_streak_days: how many consecutive days HRV has been declining/improving
    - hrv_vs_baseline: today's HRV relative to 7-day average (pct difference)
    - cls_vs_baseline: today's CLS relative to 7-day average (pct difference)
    - overcapacity_streak: how many consecutive days RAS was < 0.45 (over capacity)
    - recovery_trend: 'declining', 'improving', 'stable' for recovery score streak
    - recovery_streak_days: consecutive days of recovery decline/improvement
    - days_of_data: how many days are in the lookback window
    - note: human-readable summary of the most significant trend

    Returns an empty dict if fewer than 2 days of history are available.
    All computations are robust to missing values (None fields in summaries).
    """
    try:
        from engine.store import get_recent_summaries
    except ImportError:
        return {}

    # Fetch recent days, most-recent-first; skip today (not yet written)
    all_summaries = get_recent_summaries(days=lookback_days + 1)

    # Exclude today from the historical baseline
    historical = [s for s in all_summaries if s.get("date") != today_date]
    today_summary = next((s for s in all_summaries if s.get("date") == today_date), None)

    if len(historical) < 1:
        return {"days_of_data": 0}

    # ── Helper ────────────────────────────────────────────────────────────
    def _safe(val) -> Optional[float]:
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    # Build chronological list of (date, hrv, recovery, avg_cls, avg_ras)
    # historical is most-recent-first, so reverse for chronological order
    chrono = list(reversed(historical))

    hrv_series: list[Optional[float]] = [
        _safe(s.get("whoop", {}).get("hrv_rmssd_milli")) for s in chrono
    ]
    recovery_series: list[Optional[float]] = [
        _safe(s.get("whoop", {}).get("recovery_score")) for s in chrono
    ]
    cls_series: list[Optional[float]] = [
        _safe(s.get("metrics_avg", {}).get("cognitive_load_score")) for s in chrono
    ]
    ras_series: list[Optional[float]] = [
        _safe(s.get("metrics_avg", {}).get("recovery_alignment_score")) for s in chrono
    ]

    # Today's values (from the window-based digest, not the rolling summary)
    today_hrv = _safe(
        today_summary.get("whoop", {}).get("hrv_rmssd_milli") if today_summary else None
    )
    today_recovery = _safe(
        today_summary.get("whoop", {}).get("recovery_score") if today_summary else None
    )
    today_cls = _safe(
        today_summary.get("metrics_avg", {}).get("cognitive_load_score") if today_summary else None
    )
    today_ras = _safe(
        today_summary.get("metrics_avg", {}).get("recovery_alignment_score") if today_summary else None
    )

    # ── Streak detection ──────────────────────────────────────────────────
    def _streak(series: list[Optional[float]], today_val: Optional[float],
                direction: str, threshold: float = 0.02) -> int:
        """
        Count consecutive days (ending with today) where the value
        moved in `direction` ('up' or 'down') by at least `threshold`.

        Returns the streak length (1 = only today vs yesterday, etc.).
        Stops on the first day where the change reversed or data is missing.
        """
        full = series + ([today_val] if today_val is not None else [])
        count = 0
        for i in range(len(full) - 1, 0, -1):
            curr = full[i]
            prev = full[i - 1]
            if curr is None or prev is None:
                break
            delta = curr - prev
            if direction == "down" and delta < -threshold:
                count += 1
            elif direction == "up" and delta > threshold:
                count += 1
            else:
                break
        return count

    def _baseline(series: list[Optional[float]]) -> Optional[float]:
        vals = [v for v in series if v is not None]
        return sum(vals) / len(vals) if vals else None

    # ── HRV trend ─────────────────────────────────────────────────────────
    hrv_decline_streak = _streak(hrv_series, today_hrv, "down", threshold=2.0)  # 2ms threshold
    hrv_improve_streak = _streak(hrv_series, today_hrv, "up", threshold=2.0)

    hrv_baseline = _baseline(hrv_series)
    hrv_vs_baseline: Optional[float] = None
    if today_hrv is not None and hrv_baseline is not None and hrv_baseline > 0:
        hrv_vs_baseline = round((today_hrv - hrv_baseline) / hrv_baseline * 100, 1)

    if hrv_decline_streak >= 2:
        hrv_trend = "declining"
        hrv_streak_days = hrv_decline_streak
    elif hrv_improve_streak >= 2:
        hrv_trend = "improving"
        hrv_streak_days = hrv_improve_streak
    else:
        hrv_trend = "stable"
        hrv_streak_days = 0

    # ── Recovery trend ────────────────────────────────────────────────────
    rec_decline_streak = _streak(recovery_series, today_recovery, "down", threshold=3.0)
    rec_improve_streak = _streak(recovery_series, today_recovery, "up", threshold=3.0)

    recovery_baseline = _baseline(recovery_series)
    recovery_vs_baseline: Optional[float] = None
    if today_recovery is not None and recovery_baseline is not None and recovery_baseline > 0:
        recovery_vs_baseline = round((today_recovery - recovery_baseline) / recovery_baseline * 100, 1)

    if rec_decline_streak >= 2:
        recovery_trend = "declining"
        recovery_streak_days = rec_decline_streak
    elif rec_improve_streak >= 2:
        recovery_trend = "improving"
        recovery_streak_days = rec_improve_streak
    else:
        recovery_trend = "stable"
        recovery_streak_days = 0

    # ── CLS vs baseline ───────────────────────────────────────────────────
    cls_baseline = _baseline(cls_series)
    cls_vs_baseline: Optional[float] = None
    if today_cls is not None and cls_baseline is not None and cls_baseline > 0:
        cls_vs_baseline = round((today_cls - cls_baseline) / cls_baseline * 100, 1)

    # ── Over-capacity streak ──────────────────────────────────────────────
    # RAS < 0.45 = over capacity
    overcapacity_streak = 0
    full_ras = ras_series + ([today_ras] if today_ras is not None else [])
    for ras_val in reversed(full_ras):
        if ras_val is not None and ras_val < 0.45:
            overcapacity_streak += 1
        else:
            break

    # ── Build human note ──────────────────────────────────────────────────
    note = ""

    if hrv_trend == "declining" and hrv_streak_days >= 3:
        note = (
            f"HRV has declined for {hrv_streak_days} consecutive days — "
            f"autonomic stress is accumulating. Protect recovery tonight."
        )
    elif hrv_trend == "declining" and hrv_streak_days == 2:
        note = "HRV has dropped two days in a row — watch for early fatigue signs."

    elif hrv_trend == "improving" and hrv_streak_days >= 3:
        note = (
            f"HRV has improved for {hrv_streak_days} consecutive days — "
            f"physiological recovery is trending well."
        )

    elif overcapacity_streak >= 3:
        note = (
            f"{overcapacity_streak} consecutive days over physiological capacity. "
            f"Accumulated strain — a genuine recovery day is needed."
        )
    elif overcapacity_streak == 2:
        note = "Second consecutive day over capacity — monitor RAS tomorrow."

    elif recovery_trend == "declining" and recovery_streak_days >= 3:
        note = (
            f"Recovery score has declined for {recovery_streak_days} days in a row. "
            f"Sleep or stress management may need attention."
        )

    elif cls_vs_baseline is not None and cls_vs_baseline > 40:
        note = (
            f"Today's cognitive load was {cls_vs_baseline:.0f}% above your recent baseline "
            f"({cls_baseline:.0%} avg). Higher-than-usual demand."
        )
    elif cls_vs_baseline is not None and cls_vs_baseline < -35:
        note = (
            f"Today's load was {abs(cls_vs_baseline):.0f}% below your recent baseline — "
            f"well-paced day."
        )

    elif hrv_vs_baseline is not None and hrv_vs_baseline < -15:
        note = (
            f"HRV is {abs(hrv_vs_baseline):.0f}% below your recent average "
            f"({hrv_baseline:.0f}ms baseline) — autonomic system under pressure."
        )
    elif hrv_vs_baseline is not None and hrv_vs_baseline > 15:
        note = (
            f"HRV is {hrv_vs_baseline:.0f}% above your recent average — "
            f"strong autonomic readiness today."
        )

    return {
        "days_of_data": len(historical),
        "hrv_trend": hrv_trend,
        "hrv_streak_days": hrv_streak_days,
        "hrv_vs_baseline": hrv_vs_baseline,
        "hrv_baseline_ms": round(hrv_baseline, 1) if hrv_baseline is not None else None,
        "recovery_trend": recovery_trend,
        "recovery_streak_days": recovery_streak_days,
        "recovery_vs_baseline": recovery_vs_baseline,
        "cls_vs_baseline": cls_vs_baseline,
        "cls_baseline": round(cls_baseline, 3) if cls_baseline is not None else None,
        "overcapacity_streak": overcapacity_streak,
        "note": note,
    }


# ─── Digest computation ───────────────────────────────────────────────────────

def compute_digest(windows: list[dict]) -> dict:
    """
    Compute the digest data from a day's windows.

    Returns a structured dict with all the numbers needed for the DM.
    Works only on working-hours windows, and only on active windows
    (meeting or Slack activity) for focus quality.
    """
    if not windows:
        return {}

    date_str = windows[0]["date"]
    whoop = windows[0]["whoop"]  # Same for all windows (daily data)

    # Working hours: 7am-10pm
    working = [w for w in windows if w["metadata"]["is_working_hours"]]

    # Active windows: in a meeting or had Slack messages
    active = [w for w in working if w["calendar"]["in_meeting"] or w["slack"]["total_messages"] > 0]

    # Idle working windows: no meeting, no Slack — pure quiet time
    idle = [w for w in working if not w["calendar"]["in_meeting"] and w["slack"]["total_messages"] == 0]

    def _avg(vals: list) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _peak(vals: list) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    # CLS: computed over all working hours (includes idle, which is legit—low CLS is good)
    cls_vals_working = [w["metrics"]["cognitive_load_score"] for w in working]
    avg_cls = _avg(cls_vals_working)
    peak_cls = _peak(cls_vals_working)

    # FDI: only meaningful over active windows — idle windows trivially = 1.0
    fdi_vals_active = [w["metrics"]["focus_depth_index"] for w in active]
    avg_fdi_active = _avg(fdi_vals_active)

    # SDI: active windows only
    sdi_vals_active = [w["metrics"]["social_drain_index"] for w in active]
    avg_sdi_active = _avg(sdi_vals_active)

    # CSC: active windows only
    csc_vals_active = [w["metrics"]["context_switch_cost"] for w in active]
    avg_csc_active = _avg(csc_vals_active)

    # RAS: all windows (recovery alignment is meaningful throughout the day)
    ras_vals = [w["metrics"]["recovery_alignment_score"] for w in windows]
    avg_ras = _avg(ras_vals)

    # Peak load window: when was CLS highest?
    peak_window = None
    if working:
        peak_window = max(working, key=lambda w: w["metrics"]["cognitive_load_score"])

    # Meeting stats
    meeting_windows = [w for w in working if w["calendar"]["in_meeting"]]
    total_meeting_minutes = len(meeting_windows) * 15
    meeting_count = len(set(
        w["calendar"]["meeting_title"]
        for w in meeting_windows
        if w["calendar"]["meeting_title"]
    ))

    # Slack stats
    total_sent = sum(w["slack"]["messages_sent"] for w in windows)
    total_received = sum(w["slack"]["messages_received"] for w in windows)

    # Recovery alignment insight
    recovery = whoop.get("recovery_score")
    hrv = whoop.get("hrv_rmssd_milli")
    sleep_h = whoop.get("sleep_hours")

    # ── RescueTime stats (v1.5) ───────────────────────────────────────────
    # Aggregate focus/distraction computer time from working-hour windows.
    # Only populated when RescueTime data is actually present in the windows.
    # None means RT is not configured — the digest section is skipped entirely.
    rt_working_windows = [
        w for w in working
        if w.get("rescuetime") is not None
        and w["rescuetime"].get("active_seconds", 0) > 0
    ]

    if rt_working_windows:
        rt_focus_secs = sum(w["rescuetime"]["focus_seconds"] for w in rt_working_windows)
        rt_distraction_secs = sum(w["rescuetime"]["distraction_seconds"] for w in rt_working_windows)
        rt_active_secs = sum(w["rescuetime"]["active_seconds"] for w in rt_working_windows)
        rt_productive_pct = (
            round(100.0 * rt_focus_secs / rt_active_secs, 1)
            if rt_active_secs > 0 else None
        )
        # Most common top_activity across RT windows (best proxy for "main app today")
        from collections import Counter as _Counter
        _acts = [
            w["rescuetime"]["top_activity"]
            for w in rt_working_windows
            if w["rescuetime"].get("top_activity")
        ]
        rt_top_activity = _Counter(_acts).most_common(1)[0][0] if _acts else None

        rescuetime_digest: Optional[dict] = {
            "focus_minutes": round(rt_focus_secs / 60, 1),
            "distraction_minutes": round(rt_distraction_secs / 60, 1),
            "active_minutes": round(rt_active_secs / 60, 1),
            "productive_pct": rt_productive_pct,
            "top_activity": rt_top_activity,
        }
    else:
        rescuetime_digest = None

    # ── Multi-day trend context ────────────────────────────────────────────
    # Loads recent history to detect streaks and baseline deviations.
    # Gracefully returns {} if no history is available yet.
    trend = compute_trend_context(date_str)

    # Generate one key insight (trend-aware)
    insight = _generate_insight(
        recovery=recovery,
        avg_cls=avg_cls,
        avg_fdi_active=avg_fdi_active,
        avg_ras=avg_ras,
        total_meeting_minutes=total_meeting_minutes,
        total_sent=total_sent,
        peak_window=peak_window,
        working_count=len(working),
        active_count=len(active),
        trend=trend,
    )

    # ── Hourly CLS curve (v1.3) ────────────────────────────────────────────
    # Compact per-hour cognitive load breakdown covering 7am–10pm.
    # Provides the temporal shape of the day — not just average and peak.
    hourly_cls_curve = compute_hourly_cls_curve(windows)

    return {
        "date": date_str,
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "sleep_hours": sleep_h,
            "sleep_performance": whoop.get("sleep_performance"),
        },
        "metrics": {
            "avg_cls": avg_cls,
            "peak_cls": peak_cls,
            "avg_fdi_active": avg_fdi_active,  # FDI over active windows only
            "avg_sdi_active": avg_sdi_active,
            "avg_csc_active": avg_csc_active,
            "avg_ras": avg_ras,
        },
        "activity": {
            "working_windows": len(working),
            "active_windows": len(active),
            "idle_windows": len(idle),
            "total_meeting_minutes": total_meeting_minutes,
            "meeting_count": meeting_count,
            "slack_sent": total_sent,
            "slack_received": total_received,
        },
        "peak_window": peak_window,
        "trend": trend,
        "insight": insight,
        # v1.3: hourly CLS sparkline data (list of 15 floats, 7am–9pm)
        "hourly_cls_curve": hourly_cls_curve,
        # v1.5: RescueTime computer-time breakdown (None when RT not configured)
        "rescuetime": rescuetime_digest,
    }


def _generate_insight(
    recovery: Optional[float],
    avg_cls: Optional[float],
    avg_fdi_active: Optional[float],
    avg_ras: Optional[float],
    total_meeting_minutes: int,
    total_sent: int,
    peak_window: Optional[dict],
    working_count: int,
    active_count: int,
    trend: Optional[dict] = None,
) -> str:
    """
    Generate one data-driven insight for today.

    Priority order:
    1. Multi-day trend signals (streaks beat single-day observations)
    2. Today's alignment issues (recovery vs load mismatch)
    3. Focus fragmentation
    4. Meeting load
    5. Quiet day fallback

    v1.2: trend parameter enables streak-based and baseline-relative insights.
    When a significant multi-day pattern is detected, it takes precedence over
    single-day observations because it's more actionable and less obvious.
    """
    trend = trend or {}

    # ── Tier 1: Multi-day trend signals ───────────────────────────────────
    # Streak-based insights are the most valuable — they surface patterns
    # that are invisible when looking at one day in isolation.

    overcapacity_streak = trend.get("overcapacity_streak", 0)
    hrv_trend = trend.get("hrv_trend", "stable")
    hrv_streak = trend.get("hrv_streak_days", 0)
    recovery_trend = trend.get("recovery_trend", "stable")
    recovery_streak = trend.get("recovery_streak_days", 0)
    cls_vs_baseline = trend.get("cls_vs_baseline")
    hrv_vs_baseline = trend.get("hrv_vs_baseline")
    hrv_baseline_ms = trend.get("hrv_baseline_ms")

    # Longest/most concerning streak takes top priority
    if overcapacity_streak >= 3:
        return (
            f"{overcapacity_streak} consecutive days over physiological capacity. "
            f"This is accumulated strain — schedule a genuine recovery day soon."
        )

    if hrv_trend == "declining" and hrv_streak >= 3:
        return (
            f"HRV has declined for {hrv_streak} consecutive days — autonomic fatigue is "
            f"building. Tonight's sleep quality is critical."
        )

    if overcapacity_streak == 2:
        if recovery is not None and recovery < 55:
            return (
                f"Two days over capacity, recovery now at {recovery:.0f}%. "
                f"Tomorrow needs to be lighter."
            )

    if hrv_trend == "declining" and hrv_streak == 2:
        if recovery is not None and recovery < 60:
            return (
                f"HRV has dropped two days in a row (recovery {recovery:.0f}%). "
                f"Consider protecting tomorrow morning."
            )

    if hrv_trend == "improving" and hrv_streak >= 3:
        return (
            f"HRV has improved for {hrv_streak} consecutive days — recovery trending well. "
            f"Good conditions for a demanding day if needed."
        )

    if recovery_trend == "declining" and recovery_streak >= 3:
        return (
            f"Recovery score has declined {recovery_streak} days in a row. "
            f"Check sleep consistency and evening wind-down."
        )

    # CLS vs baseline: today was notably different from normal
    if cls_vs_baseline is not None and cls_vs_baseline > 40 and trend.get("days_of_data", 0) >= 3:
        baseline_str = f"{trend['cls_baseline']:.0%}" if trend.get("cls_baseline") else "baseline"
        return (
            f"Today's cognitive load was {cls_vs_baseline:.0f}% above your recent {baseline_str} average. "
            f"Higher than usual demand — worth monitoring recovery tomorrow."
        )

    if cls_vs_baseline is not None and cls_vs_baseline < -35 and trend.get("days_of_data", 0) >= 3:
        return (
            f"Today was {abs(cls_vs_baseline):.0f}% lighter than your recent average — "
            f"good pacing relative to your baseline."
        )

    # HRV notably below baseline even if no streak
    if (hrv_vs_baseline is not None and hrv_vs_baseline < -15
            and hrv_baseline_ms is not None and trend.get("days_of_data", 0) >= 3):
        return (
            f"HRV is {abs(hrv_vs_baseline):.0f}% below your {hrv_baseline_ms:.0f}ms baseline — "
            f"autonomic system under more pressure than usual."
        )

    # ── Tier 2: Today's alignment issues ─────────────────────────────────
    if recovery is not None and avg_cls is not None:
        if recovery < 50 and avg_cls > 0.50:
            return (
                f"You pushed hard ({avg_cls:.0%} avg load) on a {recovery:.0f}% recovery day. "
                f"Consider a lighter schedule tomorrow."
            )
        if recovery < 50 and avg_cls <= 0.30:
            return (
                f"Good self-management: recovery was low ({recovery:.0f}%) and you kept load light. "
                f"HRV should bounce back tomorrow."
            )
        if recovery >= 80 and avg_cls < 0.20 and active_count < 5:
            return (
                f"High recovery ({recovery:.0f}%) but very light cognitive load today. "
                f"You have capacity to take on more if needed."
            )

    # ── Tier 3: Focus fragmentation ───────────────────────────────────────
    if avg_fdi_active is not None and avg_fdi_active < 0.50 and active_count >= 4:
        return (
            f"Focus was fragmented during active work (FDI {avg_fdi_active:.0%}). "
            f"Try protecting at least one uninterrupted 90-minute block tomorrow."
        )

    # ── Tier 4: Meeting load ──────────────────────────────────────────────
    if total_meeting_minutes >= 240:
        hours = total_meeting_minutes // 60
        return (
            f"{hours}+ hours in meetings today. "
            f"Heavy meeting load reduces recovery and deep work — consider blocking tomorrow morning."
        )

    # ── Tier 5: Fallbacks ─────────────────────────────────────────────────
    if active_count == 0:
        return "No significant cognitive activity detected today — rest day or data gap."

    if avg_cls is not None and avg_cls < 0.15:
        return "Light cognitive day. Good for recovery — HRV should hold or improve."

    return "Load within normal range. No anomalies detected."


# ─── Slack message builder ────────────────────────────────────────────────────

def format_digest_message(digest: dict) -> str:
    """
    Format the digest data into a Slack DM message.

    Designed to be readable in Slack without markdown rendering issues.
    """
    if not digest:
        return "Presence Tracker: no data available for today."

    date_str = digest.get("date", "today")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_label = dt.strftime("%A, %B %-d")
    except ValueError:
        date_label = date_str

    m = digest.get("metrics", {})
    w = digest.get("whoop", {})
    act = digest.get("activity", {})

    avg_cls = m.get("avg_cls")
    peak_cls = m.get("peak_cls")
    avg_fdi = m.get("avg_fdi_active")
    avg_ras = m.get("avg_ras")
    recovery = w.get("recovery_score")
    hrv = w.get("hrv_rmssd_milli")
    sleep_h = w.get("sleep_hours")

    meeting_mins = act.get("total_meeting_minutes", 0)
    meeting_count = act.get("meeting_count", 0)
    active_windows = act.get("active_windows", 0)
    slack_sent = act.get("slack_sent", 0)

    peak_window = digest.get("peak_window")
    insight = digest.get("insight", "")
    hourly_cls_curve = digest.get("hourly_cls_curve")
    rescuetime = digest.get("rescuetime")  # None when RT not configured

    lines = [
        f"*Presence Report — {date_label}*",
        "",
    ]

    # ── Health baseline ──
    if recovery is not None:
        lines.append(
            f"*Recovery* {_score_bar(recovery / 100)} {recovery:.0f}%"
            + (f"  ·  HRV {hrv:.0f}ms" if hrv else "")
            + (f"  ·  Sleep {sleep_h:.1f}h" if sleep_h else "")
        )
    else:
        lines.append("*Recovery* — unavailable")

    lines.append("")

    # ── Cognitive load ──
    if avg_cls is not None:
        cls_bar = _score_bar(avg_cls)
        label = _cls_label(avg_cls)
        lines.append(f"*Cognitive Load* {cls_bar} {avg_cls:.0%} avg ({label})")
        if peak_cls and peak_cls > avg_cls + 0.10:
            peak_info = ""
            if peak_window:
                h = peak_window["metadata"]["hour_of_day"]
                mi = peak_window["metadata"]["minute_of_hour"]
                peak_info = f" at {h:02d}:{mi:02d}"
            lines.append(f"  Peak: {peak_cls:.0%}{peak_info}")
    else:
        lines.append("*Cognitive Load* — no data")

    # ── Hourly CLS sparkline (v1.3) ──
    # Compact temporal view of cognitive load from 7am to 10pm.
    # Shows the shape of the day — where effort was concentrated.
    # Legend: ░ idle  ▒ light  ▓ moderate  █ heavy
    if hourly_cls_curve:
        sparkline = _format_hourly_sparkline(hourly_cls_curve)
        lines.append(f"  `7am {sparkline} 10pm`")

    # ── Focus quality (active windows only) ──
    if avg_fdi is not None and active_windows > 0:
        fdi_bar = _score_bar(avg_fdi)
        fdi_label = _fdi_label(avg_fdi)
        lines.append(f"*Focus Quality* {fdi_bar} {avg_fdi:.0%} ({fdi_label}, active windows)")
    elif active_windows == 0:
        lines.append("*Focus Quality* — no active work detected")

    # ── Recovery alignment ──
    if avg_ras is not None:
        ras_bar = _score_bar(avg_ras)
        ras_label = _ras_label(avg_ras)
        lines.append(f"*Alignment* {ras_bar} {avg_ras:.0%} ({ras_label})")

    lines.append("")

    # ── Activity summary ──
    activity_parts = []
    if meeting_count > 0:
        activity_parts.append(f"{meeting_count} meeting{'s' if meeting_count != 1 else ''} ({meeting_mins} min)")
    if slack_sent > 0:
        activity_parts.append(f"{slack_sent} messages sent")
    if active_windows > 0:
        activity_parts.append(f"{active_windows} active windows")

    if activity_parts:
        lines.append("_" + "  ·  ".join(activity_parts) + "_")
    else:
        lines.append("_No significant activity detected_")

    # ── RescueTime computer-time breakdown (v1.5) ─────────────────────────
    # Only shown when RT data was collected.  Gives David a concrete picture
    # of how his computer time broke down: focus vs distraction vs total.
    # Example: "Computer: 4.2h active  ·  2.8h focus (67%)  ·  0.4h distraction"
    if rescuetime:
        rt_active = rescuetime.get("active_minutes", 0)
        rt_focus = rescuetime.get("focus_minutes", 0)
        rt_distraction = rescuetime.get("distraction_minutes", 0)
        rt_pct = rescuetime.get("productive_pct")
        rt_top = rescuetime.get("top_activity")

        rt_parts = []
        if rt_active > 0:
            rt_parts.append(f"{rt_active / 60:.1f}h on computer")
        if rt_focus > 0:
            pct_str = f" ({rt_pct:.0f}%)" if rt_pct is not None else ""
            rt_parts.append(f"{rt_focus / 60:.1f}h focused{pct_str}")
        if rt_distraction > 0:
            rt_parts.append(f"{rt_distraction / 60:.1f}h distracted")
        if rt_top:
            rt_parts.append(f"mostly {rt_top}")

        if rt_parts:
            lines.append("_💻 " + "  ·  ".join(rt_parts) + "_")

    # ── Trend indicator (multi-day pattern, if detected) ──
    trend = digest.get("trend", {})
    if trend:
        trend_parts = []
        hrv_trend = trend.get("hrv_trend", "stable")
        hrv_streak = trend.get("hrv_streak_days", 0)
        overcapacity = trend.get("overcapacity_streak", 0)
        hrv_vs_baseline = trend.get("hrv_vs_baseline")
        cls_vs_baseline = trend.get("cls_vs_baseline")

        if hrv_trend == "declining" and hrv_streak >= 2:
            trend_parts.append(f"HRV ↓ {hrv_streak}d")
        elif hrv_trend == "improving" and hrv_streak >= 2:
            trend_parts.append(f"HRV ↑ {hrv_streak}d")

        if overcapacity >= 2:
            trend_parts.append(f"over-capacity {overcapacity}d")

        if hrv_vs_baseline is not None and abs(hrv_vs_baseline) >= 10:
            sign = "+" if hrv_vs_baseline > 0 else ""
            trend_parts.append(f"HRV {sign}{hrv_vs_baseline:.0f}% vs baseline")

        if cls_vs_baseline is not None and abs(cls_vs_baseline) >= 25:
            sign = "+" if cls_vs_baseline > 0 else ""
            trend_parts.append(f"Load {sign}{cls_vs_baseline:.0f}% vs baseline")

        if trend_parts:
            lines.append("")
            lines.append("_Trends: " + "  ·  ".join(trend_parts) + "_")

    # ── Insight ──
    if insight:
        lines.append("")
        lines.append(f"💡 {insight}")

    return "\n".join(lines)


# ─── Send digest ──────────────────────────────────────────────────────────────

def _send_slack_dm(message: str, target: str = SLACK_DM_CHANNEL) -> bool:
    """Send a message to David's Slack DM via the gateway."""
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
                "target": target,
                "message": message,
            }
        }).encode()
        req = urllib.request.Request(
            f"{GATEWAY_URL}/tools/invoke",
            data=payload,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"[digest] Failed to send DM: {e}", file=sys.stderr)
        return False


def send_daily_digest(windows: list[dict]) -> bool:
    """
    Compute and send the daily digest DM to David.

    Args:
        windows: List of 96 window dicts for the day.

    Returns:
        True if the DM was sent successfully.
    """
    if not windows:
        print("[digest] No windows to digest", file=sys.stderr)
        return False

    digest = compute_digest(windows)
    message = format_digest_message(digest)

    print(f"[digest] Sending daily DM for {digest.get('date')}")
    return _send_slack_dm(message)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send today's Presence Digest to David")
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD, default: today's chunk)")
    parser.add_argument("--dry-run", action="store_true", help="Print the message without sending")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.store import read_day, list_available_dates

    if args.date:
        date_str = args.date
    else:
        dates = list_available_dates()
        if not dates:
            print("No data available.", file=sys.stderr)
            sys.exit(1)
        date_str = sorted(dates)[-1]

    windows = read_day(date_str)
    if not windows:
        print(f"No data for {date_str}", file=sys.stderr)
        sys.exit(1)

    digest = compute_digest(windows)
    message = format_digest_message(digest)

    print("=" * 60)
    print(message)
    print("=" * 60)

    if not args.dry_run:
        ok = _send_slack_dm(message)
        print(f"\n{'✓ Sent' if ok else '✗ Failed to send'} to David's DM")
    else:
        print("\n[dry-run] Not sent.")

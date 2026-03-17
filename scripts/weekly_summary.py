#!/usr/bin/env python3
"""
Presence Tracker — Weekly Summary

Computes a deterministic week-over-week comparison report from stored JSONL data.
No API calls, no LLM — pure Python analytics over the rolling summary.

This is the lightweight complement to the AI-powered Alfred Intuition report.
Where Intuition interprets patterns, this script *computes* them: concrete numbers,
week-vs-week deltas, best/worst days, source coverage, and focus time breakdown.

Runs automatically as part of the weekly analysis pipeline (alongside Intuition),
or can be triggered manually for any 7-day window.

Usage:
    python3 scripts/weekly_summary.py                    # Last 7 days
    python3 scripts/weekly_summary.py --date 2026-03-10  # Week ending on date
    python3 scripts/weekly_summary.py --dry-run          # Print without sending
    python3 scripts/weekly_summary.py --json             # Machine-readable output

Output:
    A Slack DM to David summarising the week's presence metrics vs the prior week.

Architecture:
    1. Load daily summaries for the current and prior week from rolling.json
    2. Compute aggregates: mean CLS, FDI, SDI, CSC, RAS + WHOOP stats, DPS
    3. Compute week-over-week deltas (Δ with direction arrows)
    4. Identify best/worst day by DPS, CLS and by FDI
    5. Compute source coverage (how many days had WHOOP, Omi, RescueTime)
    6. Format a Slack-ready message
    7. Send as DM to David

v1.1 — DPS integration:
    Daily Presence Score (DPS) is now tracked in the weekly summary.
    DPS is the single composite 0–100 score that distils the full week
    into one number — analogous to WHOOP's weekly strain trend.

    Changes:
    - load_week_data() extracts DPS from rolling.json (presence_score.dps)
      with a real-time fallback: if DPS is absent from the summary (older
      ingestion runs pre-DPS, or DPS computation failed), it recomputes
      from the JSONL windows on the fly so the weekly summary always has it.
    - dps_per_day list (one DPS per date, None for missing days) enables
      a 7-char sparkline in the Slack message showing the week's arc.
    - dps_extremes: best and worst day by DPS (distinct from CLS extremes).
    - compute_weekly_summary() adds DPS avg and week-over-week DPS delta.
    - format_weekly_message() surfaces DPS as the headline composite with
      sparkline, week avg, and week-over-week trend direction.

v2.4 — Weekly load source attribution (load drivers):
    The weekly summary now includes a "Load Drivers" section that shows
    David *what caused* his cognitive load this week — not just the aggregate
    CLS number.  This surfaces the load_decomposer's week-aggregation logic
    (which existed but was never wired into the weekly Slack output).

    What it adds:
    - Dominant load driver for the week (e.g. "Meetings 43% of CLS")
    - Per-source breakdown: meetings / slack / physiology / rescuetime / omi
    - Week-over-week source shifts (e.g. "Slack ↑ 8 pts vs last week")
    - Only surfaces when ≥ 2 meaningful days in the week window

    This answers the question: "My CLS was 38% this week — but why?"
    Previously the weekly summary described outcomes (CLS up) without causes.
    Now David can see if meetings are the culprit vs Slack pressure vs
    physiological deficit — and take targeted action.

v2.3 — Personal Records milestones + Next-week cognitive pacing:
    Two previously built modules now surface in the Sunday weekly Slack DM:

    1. Personal Records milestones — highlights any new all-time bests or
       meaningful streaks that were set during the past 7 days.  This gives
       David a weekly achievement summary beyond just averages and deltas.
       Example: "🏆 New personal best: FDI · DPS  🔥 Deep focus streak: 4 days"

    2. Next-week cognitive pacing — the WeeklyPacingPlan is now embedded in
       the Sunday summary as a forward-looking section, giving David a
       day-by-day PUSH / STEADY / PROTECT view of the week ahead before
       Monday morning arrives.  On Sunday he can adjust the calendar; by
       Monday morning the brief just confirms the plan.

       The pacing plan is omitted when:
       - Computed on a non-Sunday AND the next-week dates overlap the current
         weekday (to avoid confusing mid-week summaries).
       - The plan is not meaningful (< 3 days of history).
       - Any exception occurs (never blocks the summary).

v2.5 — Weekly Flow State + Load Volatility in weekly summary:
    Two previously built daily-digest modules now aggregate across the week
    and surface in the Sunday Slack DM, closing the gap between per-day
    detail and weekly strategic view.

    1. Flow State weekly summary — compute_weekly_flow_summary() was already
       implemented in analysis/flow_detector.py but never wired into the
       weekly output.  Now the weekly summary shows:
         🌊 Flow: 4/7 days · 6.2h total · peak Wednesday (112min)
       David can see at a glance how much deep-flow work the week contained
       — not just average FDI (which blurs together active and idle windows).

    2. Load Volatility weekly summary — compute_weekly_lvi_summary() (new,
       added to analysis/load_volatility.py) aggregates per-day LVI across
       the week and surfaces noteworthy patterns:
         ⚡ Load rhythm: 2 volatile days this week — high cognitive switching cost
         〰️ Load rhythm: Smooth week (LVI avg 0.84) — consistent cognitive demand
       Only shown when noteworthy (≥ 2 volatile days, or ≥ 4 smooth days);
       average/steady weeks are silently skipped to avoid noise.

    Together these answer: "Did I actually do deep work this week, or just
    stay busy?" and "Was my cognitive load predictable or erratic?"

v2.7 — Burnout Risk Index (BRI) in weekly summary:
    The new Burnout Risk Index module surfaces here each Sunday, giving David
    an early-warning strategic signal that complements the day-level CDI.

    CDI (14-day) answers: "How much cognitive debt am I carrying right now?"
    BRI (28-day) answers: "Where is my multi-signal trajectory heading over
    the next 2–4 weeks if this pattern continues?"

    BRI is a composite of five independently-weighted trend signals:
      - HRV trend        (30%) — the most reliable physiological burnout predictor
      - Sleep degradation (20%) — declining sleep quality precedes burnout by 2–4 weeks
      - Cognitive load creep (20%) — gradually rising CLS signals demand exceeding recovery
      - Focus erosion    (15%) — declining FDI while load stays high = depletion pattern
      - Social drain     (15%) — rising SDI without recovery = social energy burnout

    BRI tiers: healthy (<20) / watch (20–40) / caution (40–60) / high_risk (60–80) / critical (>80)

    Only shown in the weekly summary when BRI ≥ watch (> 20); healthy trajectories
    need no action.  In the nightly digest it surfaces at caution (≥ 40) or above.
    In the morning brief it appears at caution or above with a specific intervention tip.

    The intervention advice is tier- and dominant-signal-specific:
      e.g. "HRV declining — this is a physiological stress signal. Take a light day."
"""

import argparse
import json
import math
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import GATEWAY_URL, GATEWAY_TOKEN, SLACK_DM_CHANNEL
from engine.store import read_summary, list_available_dates, read_day


# ─── DPS helpers ──────────────────────────────────────────────────────────────

def _extract_dps(day_record: dict) -> Optional[float]:
    """
    Extract DPS from a rolling-summary day record.

    Primary: presence_score.dps (written by enrich_summary_with_dps() since v1.6)
    Fallback: recompute from JSONL windows so that older days (pre-DPS) still
              show a score in the weekly summary without requiring a re-ingest.

    Returns a float 0–100, or None if computation fails.
    """
    # Primary path: already computed and stored in rolling.json
    stored_dps = day_record.get("presence_score", {}).get("dps")
    if stored_dps is not None:
        return round(stored_dps, 1)

    # Fallback: read JSONL and recompute
    date_str = day_record.get("date")
    if not date_str:
        return None
    try:
        from engine.store import read_day as _read_day
        from analysis.presence_score import compute_presence_score
        windows = _read_day(date_str)
        if not windows:
            return None
        score = compute_presence_score(windows)
        return round(score.dps, 1) if score.dps is not None else None
    except Exception:
        return None


def _dps_sparkline(dps_per_day: list[Optional[float]]) -> str:
    """
    Build a 7-char Unicode sparkline from a list of DPS values (0–100).

    Higher DPS = better cognitive day.  Uses block chars ▁▂▃▄▅▆▇█.
    Missing days (None) shown as '·'.

    Returns a string like "▅▆▇▃▄▅▆" — one char per day.
    """
    BLOCKS = " ▁▂▃▄▅▆▇█"
    chars = []
    for val in dps_per_day:
        if val is None:
            chars.append("·")
        else:
            # Map 0–100 → 1–8 (index into BLOCKS[1:9])
            idx = max(1, min(8, int(val / 100 * 8) + 1))
            chars.append(BLOCKS[idx])
    return "".join(chars)


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def _mean(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _delta(this_week: Optional[float], last_week: Optional[float]) -> Optional[float]:
    """Signed delta: this_week - last_week."""
    if this_week is None or last_week is None:
        return None
    return round(this_week - last_week, 3)


def _arrow(delta: Optional[float], good_direction: str = "up") -> str:
    """
    Return an arrow emoji for a delta value.

    good_direction:
        "up"   — higher is better (FDI, RAS, recovery)
        "down" — lower is better (CLS, SDI, CSC)
    """
    if delta is None or abs(delta) < 0.01:
        return "→"
    if good_direction == "up":
        return "↑" if delta > 0 else "↓"
    else:  # "down"
        return "↑" if delta < 0 else "↓"


def _pct_change(delta: Optional[float], base: Optional[float]) -> Optional[float]:
    """Percentage change from base."""
    if delta is None or base is None or base == 0:
        return None
    return round(100.0 * delta / abs(base), 1)


# ─── Week data loading ────────────────────────────────────────────────────────

def _week_dates(end_date_str: str) -> list[str]:
    """Return 7 date strings ending on end_date (inclusive), oldest first."""
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    return [(end - timedelta(days=6 - i)).strftime("%Y-%m-%d") for i in range(7)]


def load_week_data(end_date_str: str) -> dict:
    """
    Load and aggregate daily summaries for the 7-day window ending on end_date.

    Returns a structured dict with:
    - dates: list of date strings in the window
    - days_with_data: count of days that have JSONL data
    - metrics: averaged metrics across the week
    - whoop: averaged WHOOP signals
    - source_coverage: {source: days_present}
    - best_day: {cls|fdi: date_str, value}
    - worst_day: {cls|fdi: date_str, value}
    - focus_peak_hours: most common peak focus hour
    - omi_stats: Omi aggregate for the week (conversation windows, word counts)
    - rt_stats: RescueTime aggregate (focus minutes, productive pct)
    - calendar_stats: total meeting minutes
    - slack_stats: total messages
    - raw_days: list of per-day dicts for detailed rendering
    """
    dates = _week_dates(end_date_str)
    rolling = read_summary()
    all_day_data = rolling.get("days", {})

    # Build per-day stats list, only for days with data
    day_records = []
    for d in dates:
        if d in all_day_data:
            day_records.append({"date": d, **all_day_data[d]})

    if not day_records:
        return {"dates": dates, "days_with_data": 0, "metrics": None}

    # ── Metric averages ───────────────────────────────────────────────────────
    def _avg_metric(key: str) -> Optional[float]:
        vals = [r.get("metrics_avg", {}).get(key) for r in day_records]
        return _mean(vals)

    metrics = {
        "cls": _avg_metric("cognitive_load_score"),
        "fdi": _avg_metric("focus_depth_index"),
        "sdi": _avg_metric("social_drain_index"),
        "csc": _avg_metric("context_switch_cost"),
        "ras": _avg_metric("recovery_alignment_score"),
        "active_fdi": _mean([r.get("focus_quality", {}).get("active_fdi") for r in day_records]),
    }

    # ── DPS (Daily Presence Score) per day and weekly aggregate ──────────────
    # One DPS per date in the window — None for days without data.
    # Extracted from rolling summary (or recomputed from JSONL as fallback).
    date_to_record = {r["date"]: r for r in day_records}
    dps_per_day: list[Optional[float]] = []
    for d in dates:
        rec = date_to_record.get(d)
        dps_per_day.append(_extract_dps(rec) if rec else None)

    dps_values = [v for v in dps_per_day if v is not None]
    dps_avg = round(sum(dps_values) / len(dps_values), 1) if dps_values else None

    # Best / worst day by DPS (higher = better)
    dps_pairs = [(dates[i], dps_per_day[i]) for i in range(len(dates)) if dps_per_day[i] is not None]
    dps_extremes: dict = {"best": None, "worst": None}
    if dps_pairs:
        best_d, best_v = max(dps_pairs, key=lambda x: x[1])
        worst_d, worst_v = min(dps_pairs, key=lambda x: x[1])
        dps_extremes = {
            "best": {"date": best_d, "value": best_v},
            "worst": {"date": worst_d, "value": worst_v},
        }

    # ── WHOOP averages ────────────────────────────────────────────────────────
    def _avg_whoop(key: str) -> Optional[float]:
        vals = [r.get("whoop", {}).get(key) for r in day_records]
        return _mean(vals)

    whoop = {
        "recovery": _avg_whoop("recovery_score"),
        "hrv": _avg_whoop("hrv_rmssd_milli"),
        "sleep_hours": _avg_whoop("sleep_hours"),
        "sleep_performance": _avg_whoop("sleep_performance"),
    }

    # ── Best / worst days ─────────────────────────────────────────────────────
    def _extremes(key: str, metric_section: str = "metrics_avg") -> dict:
        pairs = [
            (r["date"], r.get(metric_section, {}).get(key))
            for r in day_records
            if r.get(metric_section, {}).get(key) is not None
        ]
        if not pairs:
            return {"best": None, "worst": None}
        best_date, best_val = max(pairs, key=lambda x: x[1])
        worst_date, worst_val = min(pairs, key=lambda x: x[1])
        return {
            "best": {"date": best_date, "value": round(best_val, 3)},
            "worst": {"date": worst_date, "value": round(worst_val, 3)},
        }

    cls_extremes = _extremes("cognitive_load_score")
    fdi_extremes = _extremes("active_fdi", metric_section="focus_quality")

    # ── Peak focus hours ─────────────────────────────────────────────────────
    peak_hours = [
        r.get("focus_quality", {}).get("peak_focus_hour")
        for r in day_records
        if r.get("focus_quality", {}).get("peak_focus_hour") is not None
    ]
    # Mode: most common peak focus hour across the week
    if peak_hours:
        from collections import Counter
        peak_focus_hour_mode = Counter(peak_hours).most_common(1)[0][0]
    else:
        peak_focus_hour_mode = None

    # ── Source coverage ───────────────────────────────────────────────────────
    # Count days where each source was present (any window with that source)
    source_days: dict[str, int] = {"whoop": 0, "calendar": 0, "slack": 0, "omi": 0, "rescuetime": 0}
    for r in day_records:
        if r.get("whoop", {}).get("recovery_score") is not None:
            source_days["whoop"] += 1
        # Calendar and Slack always collected
        source_days["calendar"] += 1
        source_days["slack"] += 1
        if r.get("rescuetime") is not None:
            source_days["rescuetime"] += 1

    # Omi coverage: requires reading the JSONL files directly (not in rolling summary yet)
    omi_total_words = 0
    omi_total_conversation_windows = 0
    omi_total_sessions = 0
    omi_days = 0
    for r in day_records:
        # Read actual JSONL to get Omi aggregate
        windows = read_day(r["date"])
        omi_windows = [w for w in windows if w.get("omi") and w["omi"].get("conversation_active")]
        if omi_windows:
            omi_days += 1
            source_days["omi"] += 1
            omi_total_conversation_windows += len(omi_windows)
            omi_total_words += sum(w["omi"].get("word_count", 0) for w in omi_windows)
            omi_total_sessions += sum(w["omi"].get("sessions_count", 0) for w in omi_windows)

    omi_stats = {
        "days_active": omi_days,
        "total_conversation_windows": omi_total_conversation_windows,
        "total_words": omi_total_words,
        "total_sessions": omi_total_sessions,
        "avg_words_per_day": round(omi_total_words / omi_days, 0) if omi_days > 0 else 0,
    }

    # ── RescueTime aggregate ──────────────────────────────────────────────────
    rt_days = [r for r in day_records if r.get("rescuetime") is not None]
    if rt_days:
        rt_focus_mins = sum(r["rescuetime"].get("focus_minutes", 0) for r in rt_days)
        rt_distraction_mins = sum(r["rescuetime"].get("distraction_minutes", 0) for r in rt_days)
        rt_active_mins = sum(r["rescuetime"].get("active_minutes", 0) for r in rt_days)
        rt_productive_pcts = [r["rescuetime"].get("productive_pct") for r in rt_days if r["rescuetime"].get("productive_pct") is not None]
        rt_stats = {
            "days_tracked": len(rt_days),
            "total_focus_minutes": round(rt_focus_mins, 0),
            "total_distraction_minutes": round(rt_distraction_mins, 0),
            "avg_productive_pct": _mean(rt_productive_pcts),
        }
    else:
        rt_stats = {"days_tracked": 0}

    # ── Calendar aggregate ────────────────────────────────────────────────────
    total_meeting_mins = sum(
        r.get("calendar", {}).get("total_meeting_minutes", 0)
        for r in day_records
    )
    calendar_stats = {"total_meeting_minutes": total_meeting_mins}

    # ── Slack aggregate ───────────────────────────────────────────────────────
    total_sent = sum(r.get("slack", {}).get("total_messages_sent", 0) for r in day_records)
    total_received = sum(r.get("slack", {}).get("total_messages_received", 0) for r in day_records)
    slack_stats = {"total_sent": total_sent, "total_received": total_received}

    return {
        "dates": dates,
        "days_with_data": len(day_records),
        "metrics": metrics,
        "whoop": whoop,
        "cls_extremes": cls_extremes,
        "fdi_extremes": fdi_extremes,
        "dps_avg": dps_avg,
        "dps_per_day": dps_per_day,
        "dps_extremes": dps_extremes,
        "peak_focus_hour": peak_focus_hour_mode,
        "source_coverage": source_days,
        "omi_stats": omi_stats,
        "rt_stats": rt_stats,
        "calendar_stats": calendar_stats,
        "slack_stats": slack_stats,
        "raw_days": day_records,
    }


# ─── Report computation ───────────────────────────────────────────────────────

def compute_weekly_summary(end_date_str: str) -> dict:
    """
    Compute the full weekly summary with week-over-week comparison.

    Args:
        end_date_str: The last day of the current week (YYYY-MM-DD).

    Returns:
        Dict with this_week, last_week, deltas, and formatted sections.
    """
    # Determine prior week end date (7 days before this week's end)
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    prior_end = (end_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    this_week = load_week_data(end_date_str)
    last_week = load_week_data(prior_end)

    # ── Week-over-week metric deltas ──────────────────────────────────────────
    deltas: dict[str, Optional[float]] = {}
    if this_week.get("metrics") and last_week.get("metrics"):
        tw_m = this_week["metrics"]
        lw_m = last_week["metrics"]
        for k in ["cls", "fdi", "active_fdi", "sdi", "csc", "ras"]:
            deltas[k] = _delta(tw_m.get(k), lw_m.get(k))

    # ── WHOOP deltas ──────────────────────────────────────────────────────────
    whoop_deltas: dict[str, Optional[float]] = {}
    if this_week.get("whoop") and last_week.get("whoop"):
        tw_w = this_week["whoop"]
        lw_w = last_week["whoop"]
        for k in ["recovery", "hrv", "sleep_hours", "sleep_performance"]:
            whoop_deltas[k] = _delta(tw_w.get(k), lw_w.get(k))

    # ── DPS delta (week-over-week composite score) ────────────────────────────
    dps_delta = _delta(this_week.get("dps_avg"), last_week.get("dps_avg"))

    # ── Load source attribution (v2.4) ────────────────────────────────────────
    # Aggregate per-source CLS shares for both weeks so we can show what drove
    # the load this week and how it shifted vs the prior week.
    this_week_drivers = compute_week_load_drivers(end_date_str, days=7)
    last_week_drivers = compute_week_load_drivers(prior_end, days=7)

    return {
        "end_date": end_date_str,
        "this_week": this_week,
        "last_week": last_week,
        "deltas": deltas,
        "whoop_deltas": whoop_deltas,
        "dps_delta": dps_delta,
        "this_week_drivers": this_week_drivers,
        "last_week_drivers": last_week_drivers,
    }


# ─── Load drivers computation ────────────────────────────────────────────────

_SOURCE_LABELS = {
    "meetings":    "Meetings",
    "slack":       "Slack",
    "physiology":  "Physiology",
    "rescuetime":  "RescueTime",
    "omi":         "Conversations",
}

_SOURCE_EMOJIS = {
    "meetings":    "📅",
    "slack":       "💬",
    "physiology":  "💓",
    "rescuetime":  "🖥",
    "omi":         "🎙",
}


def compute_week_load_drivers(end_date_str: str, days: int = 7) -> dict:
    """
    Compute the weekly load source attribution using the load_decomposer.

    Aggregates per-source CLS shares across the week and returns:
      - shares:         dict[source → float]  (avg fraction across days)
      - dominant:       str  (source with highest share)
      - days_meaningful: int
      - error:          str | None  (set when the decomposer is unavailable)

    Returns a dict with shares={} and days_meaningful=0 on error or no data,
    so callers can always safely test `days_meaningful > 0`.
    """
    try:
        from analysis.load_decomposer import compute_week_decomposition
        result = compute_week_decomposition(end_date_str=end_date_str, days=days)
        return {
            "shares":          result.get("weekly_shares", {}),
            "dominant":        result.get("dominant_source", "unknown"),
            "days_meaningful": result.get("days_meaningful", 0),
            "error":           None,
        }
    except Exception as exc:
        return {
            "shares":          {},
            "dominant":        "unknown",
            "days_meaningful": 0,
            "error":           str(exc),
        }


# ─── Message formatter ────────────────────────────────────────────────────────

def _fmt_val(val: Optional[float], scale: float = 100.0, suffix: str = "%", decimals: int = 0) -> str:
    """Format a 0-1 metric as percentage or other scaled value."""
    if val is None:
        return "—"
    scaled = val * scale
    if decimals == 0:
        return f"{round(scaled)}{suffix}"
    return f"{scaled:.{decimals}f}{suffix}"


def _fmt_delta(delta: Optional[float], scale: float = 100.0, good_direction: str = "up") -> str:
    """Format a delta value with direction arrow and sign."""
    if delta is None or abs(delta * scale) < 1.0:
        return ""
    arrow = _arrow(delta, good_direction=good_direction)
    scaled = abs(delta * scale)
    sign = "+" if delta > 0 else "−"
    if scaled >= 1.0:
        return f" {arrow} {sign}{round(scaled)}%"
    return f" {arrow} {sign}{scaled:.1f}%"


def _fmt_ms(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{round(val)}ms"


def _fmt_ms_delta(delta: Optional[float]) -> str:
    if delta is None or abs(delta) < 1.0:
        return ""
    arrow = _arrow(delta, good_direction="up")
    sign = "+" if delta > 0 else "−"
    return f" {arrow} {sign}{abs(round(delta))}ms"


def _day_label(date_str: Optional[str]) -> str:
    if not date_str:
        return "—"
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%a %-d")  # e.g. "Mon 10"
    except Exception:
        return date_str


def _hour_label(hour: Optional[int]) -> str:
    if hour is None:
        return "—"
    suffix = "am" if hour < 12 else "pm"
    h12 = hour if hour <= 12 else hour - 12
    if h12 == 0:
        h12 = 12
    return f"{h12}{suffix}"


def _dps_tier_label(dps: float) -> str:
    """Human-readable tier label for a DPS score (0–100)."""
    if dps >= 80:
        return "peak presence 🟢"
    elif dps >= 65:
        return "strong 🔵"
    elif dps >= 50:
        return "moderate 🟡"
    elif dps >= 35:
        return "stretched 🟠"
    else:
        return "recovery needed 🔴"


def format_weekly_message(summary: dict) -> str:
    """Format the weekly summary as a Slack-ready DM message."""
    tw = summary.get("this_week", {})
    lw = summary.get("last_week", {})
    deltas = summary.get("deltas", {})
    whoop_deltas = summary.get("whoop_deltas", {})
    end_date = summary.get("end_date", "")

    # Header
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=6)
        week_label = f"{start_dt.strftime('%b %-d')}–{end_dt.strftime('%b %-d')}"
    except Exception:
        week_label = end_date

    tw_n = tw.get("days_with_data", 0)
    lines = [
        f"*📊 Weekly Presence Summary — {week_label}*",
        f"_{tw_n} day{'s' if tw_n != 1 else ''} tracked_",
        "",
    ]

    # No data guard
    tw_m = tw.get("metrics")
    if not tw_m:
        lines.append("No presence data for this week.")
        return "\n".join(lines)

    # ── DPS headline ──────────────────────────────────────────────────────────
    # DPS is the primary composite metric — surfaces here as the weekly headline.
    tw_dps = tw.get("dps_avg")
    dps_delta = summary.get("dps_delta")
    dps_per_day = tw.get("dps_per_day", [])
    dps_spark = _dps_sparkline(dps_per_day) if dps_per_day else ""

    if tw_dps is not None:
        dps_tier = _dps_tier_label(tw_dps)
        dps_delta_str = ""
        if dps_delta is not None and abs(dps_delta) >= 1.0:
            arrow = "↑" if dps_delta > 0 else "↓"
            sign = "+" if dps_delta > 0 else "−"
            dps_delta_str = f"  {arrow} {sign}{abs(round(dps_delta, 1))} vs last week"
        spark_str = f"  `{dps_spark}`" if dps_spark else ""
        lines.append(f"*DPS: {round(tw_dps)}/100* — {dps_tier}{dps_delta_str}{spark_str}")
        lines.append("")

    # ── Core metrics ──────────────────────────────────────────────────────────
    lines.append("*Core Metrics (working hours avg)*")

    cls_val = _fmt_val(tw_m.get("cls"), scale=100, suffix="%")
    cls_d = _fmt_delta(deltas.get("cls"), scale=100, good_direction="down")
    lines.append(f"• CLS (Cognitive Load):  {cls_val}{cls_d}")

    # Use active_fdi as the accurate focus signal; fall back to fdi
    fdi_src = tw_m.get("active_fdi") or tw_m.get("fdi")
    fdi_delta_src = deltas.get("active_fdi") or deltas.get("fdi")
    fdi_val = _fmt_val(fdi_src, scale=100, suffix="%")
    fdi_d = _fmt_delta(fdi_delta_src, scale=100, good_direction="up")
    lines.append(f"• FDI (Focus Depth):     {fdi_val}{fdi_d}")

    sdi_val = _fmt_val(tw_m.get("sdi"), scale=100, suffix="%")
    sdi_d = _fmt_delta(deltas.get("sdi"), scale=100, good_direction="down")
    lines.append(f"• SDI (Social Drain):    {sdi_val}{sdi_d}")

    ras_val = _fmt_val(tw_m.get("ras"), scale=100, suffix="%")
    ras_d = _fmt_delta(deltas.get("ras"), scale=100, good_direction="up")
    lines.append(f"• RAS (Recovery Align):  {ras_val}{ras_d}")

    # ── WHOOP ─────────────────────────────────────────────────────────────────
    tw_w = tw.get("whoop", {})
    if tw_w.get("recovery") is not None:
        lines.append("")
        lines.append("*WHOOP (weekly avg)*")

        rec_val = tw_w.get("recovery")
        rec_d = whoop_deltas.get("recovery")
        rec_arrow = _arrow(rec_d, good_direction="up") if rec_d and abs(rec_d) >= 1 else "→"
        rec_sign = f" {rec_arrow} {'+' if rec_d and rec_d > 0 else '−'}{abs(round(rec_d or 0))}%" if rec_d and abs(rec_d) >= 1 else ""
        lines.append(f"• Recovery:   {round(rec_val or 0)}%{rec_sign}")

        hrv_d = whoop_deltas.get("hrv")
        lines.append(f"• HRV:        {_fmt_ms(tw_w.get('hrv'))}{_fmt_ms_delta(hrv_d)}")

        sleep_h = tw_w.get("sleep_hours")
        if sleep_h is not None:
            sleep_d = whoop_deltas.get("sleep_hours")
            sleep_arrow = ""
            if sleep_d is not None and abs(sleep_d) >= 0.1:
                arrow_s = "↑" if sleep_d > 0 else "↓"
                sign_s = "+" if sleep_d > 0 else "−"
                sleep_arrow = f" {arrow_s} {sign_s}{abs(sleep_d):.1f}h"
            lines.append(f"• Sleep:      {sleep_h:.1f}h{sleep_arrow}")

    # ── Load Drivers (source attribution) — v2.4 ─────────────────────────────
    # Shows what *caused* the week's cognitive load, not just the CLS number.
    # Sources: meetings, slack, physiology, rescuetime, omi.
    # Only shown when the decomposer has ≥ 2 meaningful days.
    this_drivers = summary.get("this_week_drivers", {})
    last_drivers = summary.get("last_week_drivers", {})

    if this_drivers.get("days_meaningful", 0) >= 2:
        shares = this_drivers.get("shares", {})
        dominant = this_drivers.get("dominant", "unknown")
        last_shares = last_drivers.get("shares", {}) if last_drivers.get("days_meaningful", 0) >= 2 else {}

        # Sort sources by share (descending), skip zero-share sources
        ranked = sorted(
            [(src, sh) for src, sh in shares.items() if sh > 0.005],
            key=lambda x: x[1],
            reverse=True,
        )

        if ranked:
            lines.append("")
            dom_emoji = _SOURCE_EMOJIS.get(dominant, "•")
            dom_label = _SOURCE_LABELS.get(dominant, dominant.capitalize())
            dom_pct = round(shares.get(dominant, 0) * 100)
            lines.append(f"*Load Drivers* — {dom_emoji} {dom_label} led ({dom_pct}%)")

            for src, sh in ranked:
                emoji = _SOURCE_EMOJIS.get(src, "•")
                label = _SOURCE_LABELS.get(src, src.capitalize())
                pct = round(sh * 100)

                # Week-over-week shift for this source
                shift_str = ""
                if last_shares and src in last_shares:
                    delta_pct = round(sh * 100) - round(last_shares[src] * 100)
                    if abs(delta_pct) >= 3:
                        arrow_s = "↑" if delta_pct > 0 else "↓"
                        sign_s = "+" if delta_pct > 0 else "−"
                        shift_str = f"  {arrow_s} {sign_s}{abs(delta_pct)}pp vs last week"

                # Mark dominant source
                marker = " ←" if src == dominant else ""
                lines.append(f"  • {emoji} {label}: {pct}%{shift_str}{marker}")

    # ── Best / worst days ─────────────────────────────────────────────────────
    cls_ext = tw.get("cls_extremes", {})
    fdi_ext = tw.get("fdi_extremes", {})
    dps_ext = tw.get("dps_extremes", {})
    has_extremes = cls_ext.get("best") or fdi_ext.get("best") or dps_ext.get("best")
    if has_extremes:
        lines.append("")
        lines.append("*Best & Worst Days*")

        # DPS extremes shown first — it's the primary composite signal
        if dps_ext.get("best"):
            best_dps = dps_ext["best"]
            lines.append(
                f"• Best day (DPS):     {_day_label(best_dps['date'])} "
                f"({round(best_dps['value'])}/100)"
            )
        if dps_ext.get("worst") and dps_ext.get("best") and \
                dps_ext["worst"]["date"] != dps_ext["best"]["date"]:
            worst_dps = dps_ext["worst"]
            lines.append(
                f"• Toughest day (DPS): {_day_label(worst_dps['date'])} "
                f"({round(worst_dps['value'])}/100)"
            )

        if cls_ext.get("worst") and cls_ext.get("best"):
            worst_cls = cls_ext["worst"]
            best_cls_by_low = cls_ext["worst"]  # lowest CLS = lightest day
            lightest = cls_ext["worst"]
            heaviest = cls_ext["best"]
            lines.append(
                f"• Heaviest day (CLS): {_day_label(heaviest['date'])} "
                f"({round(heaviest['value'] * 100)}%)"
            )
            lines.append(
                f"• Lightest day (CLS): {_day_label(lightest['date'])} "
                f"({round(lightest['value'] * 100)}%)"
            )

        if fdi_ext.get("best"):
            best_fdi = fdi_ext["best"]
            lines.append(
                f"• Best focus day (FDI): {_day_label(best_fdi['date'])} "
                f"({round(best_fdi['value'] * 100)}%)"
            )

    # ── Focus peak ────────────────────────────────────────────────────────────
    peak_hour = tw.get("peak_focus_hour")
    if peak_hour is not None:
        lines.append(f"• Peak focus window: {_hour_label(peak_hour)}")

    # ── Meetings ──────────────────────────────────────────────────────────────
    cal = tw.get("calendar_stats", {})
    total_meeting_mins = cal.get("total_meeting_minutes", 0)
    if total_meeting_mins > 0:
        meeting_hours = total_meeting_mins / 60
        lines.append("")
        lines.append("*Time Breakdown*")
        lines.append(f"• Meetings:    {meeting_hours:.1f}h ({total_meeting_mins} min)")

    # ── Omi stats ─────────────────────────────────────────────────────────────
    omi = tw.get("omi_stats", {})
    if omi.get("days_active", 0) > 0:
        omi_days = omi["days_active"]
        omi_words = omi["total_words"]
        omi_sessions = omi["total_sessions"]
        omi_line = f"• Conversations: {omi_sessions} sessions across {omi_days}d"
        if omi_words > 0:
            omi_line += f" ({omi_words:,} words)"
        lines.append(omi_line)

    # ── RescueTime ────────────────────────────────────────────────────────────
    rt = tw.get("rt_stats", {})
    if rt.get("days_tracked", 0) > 0:
        focus_h = (rt.get("total_focus_minutes") or 0) / 60
        productive_pct = rt.get("avg_productive_pct")
        rt_line = f"• Deep work: {focus_h:.1f}h focused"
        if productive_pct is not None:
            rt_line += f" ({round(productive_pct)}% productive)"
        lines.append(rt_line)

    # ── Slack totals ──────────────────────────────────────────────────────────
    slack = tw.get("slack_stats", {})
    total_msgs = slack.get("total_sent", 0) + slack.get("total_received", 0)
    if total_msgs > 0:
        lines.append(f"• Slack:       {slack.get('total_sent', 0)} sent / {slack.get('total_received', 0)} received")

    # ── Week-over-week comparison headline ────────────────────────────────────
    lw_n = lw.get("days_with_data", 0)
    if lw_n > 0 and deltas:
        lines.append("")
        lines.append("*vs Prior Week*")
        trend_parts = []

        # DPS delta first — the headline composite
        dps_d = summary.get("dps_delta")
        if dps_d is not None and abs(dps_d) >= 2.0:
            direction = "up" if dps_d > 0 else "down"
            sign = "+" if dps_d > 0 else "−"
            trend_parts.append(f"DPS {sign}{abs(round(dps_d, 1))} pts ({direction})")

        cls_d = deltas.get("cls")
        if cls_d is not None and abs(cls_d) >= 0.02:
            direction = "lighter" if cls_d < 0 else "heavier"
            trend_parts.append(f"CLS {abs(round(cls_d * 100))}% {direction}")

        fdi_d = deltas.get("active_fdi") or deltas.get("fdi")
        if fdi_d is not None and abs(fdi_d) >= 0.02:
            direction = "sharper" if fdi_d > 0 else "more fragmented"
            trend_parts.append(f"focus {direction}")

        rec_d = whoop_deltas.get("recovery")
        if rec_d is not None and abs(rec_d) >= 2:
            direction = "better" if rec_d > 0 else "lower"
            trend_parts.append(f"recovery {abs(round(rec_d))}% {direction}")

        hrv_d = whoop_deltas.get("hrv")
        if hrv_d is not None and abs(hrv_d) >= 3:
            direction = "improved" if hrv_d > 0 else "dipped"
            trend_parts.append(f"HRV {_fmt_ms(abs(hrv_d))} {direction}")

        if trend_parts:
            lines.append("  " + " · ".join(trend_parts))
        else:
            lines.append("  Stable week — no significant changes vs last week")

    # ── Cognitive Rhythm insight ──────────────────────────────────────────────
    try:
        from analysis.cognitive_rhythm import compute_cognitive_rhythm, format_rhythm_line
        rhythm = compute_cognitive_rhythm(as_of_date_str=end_date)
        rhythm_line = format_rhythm_line(rhythm)
        if rhythm_line:
            lines.append("")
            lines.append(rhythm_line)
    except Exception:
        pass  # rhythm is non-critical — never block the weekly summary

    # ── Flow State weekly summary (v2.5) ─────────────────────────────────────
    # Aggregates per-day flow state detection across the week.
    # compute_weekly_flow_summary() loads each day's JSONL and detects
    # contiguous flow sessions (high FDI + moderate CLS + low CSC ≥ 30 min).
    # Only surfaced when there are ≥ 2 days with meaningful flow data.
    # Shows: total flow minutes, days with flow, best flow day.
    # Degrades silently — never blocks the weekly summary.
    try:
        from analysis.flow_detector import compute_weekly_flow_summary
        # Build the per-day structure the aggregator expects
        week_dates_for_flow = _week_dates(end_date)
        flow_day_entries = [{"date": d} for d in week_dates_for_flow]
        flow_weekly = compute_weekly_flow_summary(flow_day_entries)

        flow_days = flow_weekly.get("flow_days", 0)
        total_flow_min = flow_weekly.get("total_flow_minutes", 0)
        avg_flow_min = flow_weekly.get("avg_flow_minutes_per_day", 0.0)
        best_flow_day = flow_weekly.get("best_flow_day")
        best_flow_min = flow_weekly.get("best_flow_minutes", 0)

        # Only surface when there's meaningful data (≥ 1 day with flow)
        if flow_days >= 1 and total_flow_min >= 30:
            total_flow_h = total_flow_min / 60
            lines.append("")
            flow_line = f"🌊 Flow: {flow_days}/7 days · {total_flow_h:.1f}h total"
            if best_flow_day:
                best_flow_h = best_flow_min / 60
                flow_line += f" · peak {_day_label(best_flow_day)} ({best_flow_min}min)"
            lines.append(flow_line)
    except Exception:
        pass  # flow is non-critical — never block the weekly summary

    # ── Load Volatility weekly summary (v2.5) ────────────────────────────────
    # Shows whether the week had a smooth, steady, or erratic cognitive rhythm.
    # LVI (Load Volatility Index) measures CLS consistency across active windows.
    # Only surfaced when noteworthy: volatile/variable days dominate, or
    # the week was notably smooth (≥ 4 smooth days).
    # Degrades silently — never blocks the weekly summary.
    try:
        from analysis.load_volatility import (
            compute_weekly_lvi_summary,
            format_weekly_lvi_line,
        )
        week_dates_for_lvi = _week_dates(end_date)
        weekly_lvi = compute_weekly_lvi_summary(week_dates_for_lvi)
        lvi_line = format_weekly_lvi_line(weekly_lvi)
        if lvi_line:
            lines.append("")
            lines.append(lvi_line)
    except Exception:
        pass  # LVI is non-critical — never block the weekly summary

    # ── Sleep → Focus correlation insight (v2.2) ──────────────────────────────
    # Shows David the empirical relationship between his sleep quality and
    # next-day cognitive performance, computed from all available JSONL history.
    # Only surfaces when there are ≥ MIN_PAIRS (5) paired days of data.
    # Degrades silently on error — never blocks the weekly summary.
    try:
        from analysis.sleep_focus_correlator import (
            compute_sleep_focus_correlation,
            format_sleep_insight_line,
        )
        sleep_corr = compute_sleep_focus_correlation(as_of_date_str=end_date)
        sleep_line = format_sleep_insight_line(sleep_corr)
        if sleep_line:
            lines.append("")
            lines.append(sleep_line)
    except Exception:
        pass  # sleep correlator is non-critical — never block the weekly summary

    # ── Personal Records milestones (v2.3) ────────────────────────────────────
    # Surfaces any new all-time bests or meaningful streaks set during the
    # past 7 days.  Checks each day in the week window for new records so
    # that a mid-week personal best is not missed just because it wasn't
    # set on the last day of the summary period.
    # Degrades silently — never blocks the weekly summary.
    try:
        from analysis.personal_records import (
            compute_personal_records,
            check_today_records,
            format_records_line,
        )
        # Use end_date as the reference so all-time bests include the full week
        all_records = compute_personal_records(as_of_date_str=end_date)
        if all_records.is_meaningful():
            week_record_lines = []
            # Check every day in the week window (oldest → newest)
            try:
                end_dt_rec = datetime.strptime(end_date, "%Y-%m-%d")
                week_days_to_check = [
                    (end_dt_rec - timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(6, -1, -1)
                ]
            except Exception:
                week_days_to_check = []

            seen_bests: set = set()  # avoid duplicate lines if same record hit twice
            for day_str in week_days_to_check:
                try:
                    # Compute records as-of this day so the check is accurate
                    day_records = compute_personal_records(as_of_date_str=day_str)
                    today_rec = check_today_records(day_str, day_records)
                    if today_rec.has_records:
                        rec_line = format_records_line(today_rec)
                        if rec_line and rec_line not in seen_bests:
                            week_record_lines.append(rec_line)
                            seen_bests.add(rec_line)
                except Exception:
                    continue

            if week_record_lines:
                lines.append("")
                lines.append("*🏆 This week's milestones*")
                for rl in week_record_lines:
                    lines.append(f"  {rl}")
    except Exception:
        pass  # records is non-critical — never block the weekly summary

    # ── Next-week cognitive pacing plan (v2.3) ────────────────────────────────
    # Embeds the WeeklyPacingPlan in the Sunday Slack summary so David sees
    # next week's PUSH / STEADY / PROTECT strategy 24 hours before the
    # Monday morning brief confirms it.
    # Only emitted when: plan is meaningful AND we are looking at a recent
    # week-end (not a historical backfill more than 2 days stale).
    # Degrades silently — never blocks the weekly summary.
    try:
        from analysis.weekly_pacing import (
            compute_weekly_pacing,
            format_weekly_pacing_section,
        )
        # Use the day *after* end_date as the reference so the pacing plan
        # targets the upcoming week, not the week we just reviewed.
        end_dt_for_pacing = datetime.strptime(end_date, "%Y-%m-%d")
        next_week_ref = (end_dt_for_pacing + timedelta(days=1)).strftime("%Y-%m-%d")

        # Only show next-week pacing when the summary is recent (≤ 2 days old)
        today_dt = datetime.now()
        summary_age_days = (today_dt - end_dt_for_pacing).days
        if summary_age_days <= 2:
            pacing_plan = compute_weekly_pacing(next_week_ref, fetch_calendar=False)
            if pacing_plan.is_meaningful:
                pacing_section = format_weekly_pacing_section(pacing_plan)
                if pacing_section.strip():
                    lines.append("")
                    lines.append(pacing_section.strip())
    except Exception:
        pass  # pacing is non-critical — never block the weekly summary

    # ── Conversation Intelligence (v2.6) ─────────────────────────────────────
    # Analyses raw Omi transcript history directly — independent of JSONL store.
    # Works even when full daily ingestion hasn't been run yet, because it reads
    # ~/omi/transcripts/ directly.  Surfaces: speech load trend, peak conversation
    # hour, language split, cognitive density, and actionable insights.
    # Only shown when ≥ 3 days of Omi transcript data exist in the window.
    # Degrades silently — never blocks the weekly summary.
    try:
        from analysis.conversation_intelligence import (
            analyse_conversation_history,
            format_conversation_intelligence_section,
        )
        # Use 14 days ending on the summary end_date to give a richer picture
        ci = analyse_conversation_history(days=14, end_date_str=end_date)
        if ci.is_meaningful:
            ci_section = format_conversation_intelligence_section(ci)
            if ci_section.strip():
                lines.append("")
                lines.append(ci_section.strip())
    except Exception:
        pass  # conversation intelligence is non-critical — never block the weekly summary

    # ── Burnout Risk Index (v2.7) ─────────────────────────────────────────────
    # BRI analyses 4-week trends across HRV, sleep quality, cognitive load,
    # focus depth, and social drain to detect early burnout trajectory.
    # Unlike CDI (current 14-day debt), BRI is a leading indicator:
    # "If this pattern continues, where is the trajectory heading?"
    # Only surfaced in the weekly summary when BRI ≥ watch tier (> 20),
    # since healthy BRI needs no action.
    # Degrades silently — never blocks the weekly summary.
    try:
        from analysis.burnout_risk import (
            compute_burnout_risk,
            format_bri_section,
        )
        bri = compute_burnout_risk(as_of_date_str=end_date, days=28)
        if bri.is_meaningful and bri.tier != "healthy":
            bri_section = format_bri_section(bri)
            if bri_section.strip():
                lines.append("")
                lines.append(bri_section.strip())
    except Exception:
        pass  # BRI is non-critical — never block the weekly summary

    # ── Actionable Insights (v42) ─────────────────────────────────────────
    # Surfaces the top 3 evidence-backed, data-driven behavioural recommendations
    # derived from the past 14 days of JSONL history.  Deterministic — no LLM.
    # Only included when at least one insight fires (is_meaningful=True).
    try:
        from analysis.actionable_insights import (
            compute_actionable_insights,
            format_insights_section,
        )
        ai = compute_actionable_insights(as_of_date_str=end_date, days=14)
        if ai.is_meaningful:
            insights_section = format_insights_section(ai)
            if insights_section.strip():
                lines.append("")
                lines.append(insights_section.strip())
    except Exception:
        pass  # Actionable insights is non-critical — never block the weekly summary

    lines.append("")
    lines.append("_Presence Tracker · weekly summary_")

    return "\n".join(lines)


# ─── Send DM ──────────────────────────────────────────────────────────────────

def send_weekly_summary(end_date_str: str) -> bool:
    """Compute and send the weekly summary DM to David."""
    summary = compute_weekly_summary(end_date_str)
    message = format_weekly_message(summary)

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
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"[weekly-summary] Failed to send DM: {e}", file=sys.stderr)
        return False


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Presence Tracker — Weekly Summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date of the week to summarise (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message without sending",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_out",
        help="Output raw summary as JSON instead of formatted message",
    )
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    summary = compute_weekly_summary(args.date)

    if args.json_out:
        print(json.dumps(summary, indent=2, default=str))
        return

    message = format_weekly_message(summary)

    print("=" * 60)
    print(message)
    print("=" * 60)

    if not args.dry_run:
        ok = send_weekly_summary(args.date)
        print(f"\n{'✓ Sent' if ok else '✗ Failed to send'} to David's DM")
    else:
        print("\n[dry-run] Not sent.")


if __name__ == "__main__":
    main()

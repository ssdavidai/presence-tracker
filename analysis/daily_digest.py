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

v1.6 — Omi conversation digest + peak focus hour:
  The digest now surfaces two previously invisible signals:

  1. Omi conversation activity — Omi ambient audio data has been wired into
     CLS, SDI, and FDI since v2.0, but never shown explicitly in the message.
     David could see its effect on metrics but not the cause.  Now when Omi
     detects speech on a given day the digest shows:
       - Session count (how many distinct conversations)
       - Total words spoken (engagement depth proxy)
       - Total speaking time in minutes
     Example: "🎙 3 conversations  ·  847 words  ·  14 min speaking"
     Shown only when at least one Omi conversation was detected.

  2. Peak focus hour — the working hour with the highest active FDI is now
     surfaced when it's above a meaningful threshold (≥ 0.70).  This answers
     "when am I most focused?" and can inform future scheduling.
     Example: "🏆 Best focus: 09:00–10:00 (FDI 84%)"
     Computed from active working-hour windows (same as active_fdi).
     Only shown when at least one active window exists and peak FDI ≥ 0.70.

  Both additions use data already collected — no new API calls or collectors.
  Both degrade gracefully: if Omi has no data, the section is omitted;
  if no active windows exist, the peak focus line is omitted.

v1.9 — Tomorrow's focus plan in the nightly digest:
  The nightly digest now includes tomorrow's recommended focus blocks, mirroring
  what the morning brief already shows.  Placing the plan in the *evening* digest
  lets David mentally prepare the night before — he can adjust tomorrow's calendar
  or set intentions before sleep, rather than discovering the plan in the morning.

  The section is produced by the same focus_planner.plan_tomorrow_focus() call
  used in the morning brief, formatted via format_focus_plan_section().

  Example output in the digest:
      🎯 *Tomorrow's Focus Plan:*
      • 9:00–11:00  _(120min, peak focus hour)_  🔥
      • 14:00–15:30  _(90min, strong focus hour)_  ✅
      _Two clear blocks available — front-load the harder task in the earlier one._

  Degrades gracefully: if calendar data or focus planner fails, the section is
  silently omitted — it is never load-bearing for the rest of the digest.

v2.2 — ML model insights in the nightly digest:
  The ML model layer (Isolation Forest anomaly detector + Random Forest recovery
  predictor) has been trained since v3 but its outputs were never surfaced to
  David.  This release wires both into the nightly digest.

  1. Recovery prediction — when the Random Forest recovery predictor is trained
     (requires ≥ 14 day-pairs), the digest includes:
       "🤖 Tomorrow's recovery: ~72% (high confidence)"
     This forward-looking signal lets David adjust tonight's wind-down routine.
     Degrades silently when the model isn't trained (< 14 data days).

  2. Anomaly detection — the Isolation Forest flags unusual 15-min windows
     (atypical combinations of CLS, FDI, RAS, meetings, Slack).  When ≥ 2
     anomalous windows are found in a cluster (consecutive hours), the digest
     surfaces a compact note:
       "🔍 Unusual pattern detected at 14:00–15:30 (3 anomalous windows)"
     This draws David's attention to atypical cognitive states — periods that
     don't fit his normal behavioral patterns and may deserve reflection.
     Shown only when the model is trained and ≥ 2 anomalies are detected.

  Both additions use only the existing ml_model.py infrastructure — no new
  collectors, no new training pipelines.  Both degrade gracefully when models
  are not yet trained or features cannot be extracted.

  Implementation:
  - _compute_ml_insights_for_digest(windows) — exception-isolated helper
    that calls detect_anomalies() and predict_recovery() and returns a dict
  - compute_digest() adds "ml_insights" key to the return dict
  - format_digest_message() renders the ML insights section after CDI

v2.3 — Tomorrow's load forecast in the nightly digest:
  The nightly digest already shows tomorrow's focus plan (since v1.9) but gave
  no context about *how cognitively demanding* tomorrow will be.  That context
  is exactly what the load forecast provides.

  This release pairs the two into a unified "Tomorrow" section:
    *Tomorrow*
    📊 Load forecast: Moderate — CLS ~0.42 (8 similar days)
    _2h30m of meetings → moderate load expected (CLS ~0.42). Front-load focused work._

    *🎯 Tomorrow's Focus Plan:*
    • 9:00–10:30  _(90min, peak focus hour)_  🔥
    ...

  The forecast is computed for *tomorrow's date* (today+1), using tomorrow's
  calendar and the same historical-bucket logic as the morning brief.

  Graceful degradation:
  - If the forecast is not meaningful (insufficient history, no calendar), the
    "Tomorrow" header and narrative are omitted — only the focus plan shows.
  - If *neither* forecast nor focus plan is meaningful, the section is omitted.
  - All compute paths are exception-isolated; the digest never crashes over this.

  Implementation:
  - _compute_load_forecast_for_digest(today_date_str) — new helper that
    fetches tomorrow's calendar, calls compute_load_forecast(tomorrow_str, ...),
    and returns a serialised dict
  - compute_digest() adds "tomorrow_load_forecast" key to the return dict
  - format_digest_message() renders both in a combined "Tomorrow" section
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


# ─── DPS helper ───────────────────────────────────────────────────────────────

def _compute_dps_for_digest(windows: list[dict]) -> Optional[dict]:
    """
    Compute Daily Presence Score for the digest.

    Returns a dict {"dps": float, "tier": str, "line": str} or None.
    Wraps the DPS module with full exception isolation so the digest
    never crashes due to a DPS computation error.
    """
    try:
        from analysis.presence_score import compute_presence_score, format_presence_score_line
        score = compute_presence_score(windows)
        if not score.is_meaningful:
            return None
        return {
            "dps": score.dps,
            "tier": score.tier,
            "line": format_presence_score_line(score),
        }
    except Exception:
        return None


def _compute_focus_plan_for_digest(today_date_str: str) -> Optional[dict]:
    """
    Compute tomorrow's focus plan for the nightly digest (v1.9).

    Mirrors the same call used by the morning brief, but placed in the
    evening digest so David can mentally prepare the night before.

    Returns a dict with serialised FocusPlan data, or None on error / no data.
    Fully exception-isolated — never crashes the digest.
    """
    try:
        from analysis.focus_planner import plan_tomorrow_focus, format_focus_plan_section
        plan = plan_tomorrow_focus(today_date_str)
        if plan is None:
            return None
        return {
            "section": format_focus_plan_section(plan),
            "summary_line": plan.summary_line,
            "advisory": plan.advisory,
            "is_meaningful": plan.is_meaningful,
            "block_count": len(plan.recommended_blocks),
            "cdi_tier": plan.cdi_tier,
        }
    except Exception:
        return None


def _compute_meeting_intel_for_digest(windows: list[dict], date_str: str) -> Optional[dict]:
    """
    Compute Meeting Intelligence for the digest.

    Returns a dict with MIS, FFS, CMC, SDR, meeting_recovery_fit, headline,
    advisory, and a pre-formatted Slack section — or None when no meetings
    or the meeting_intel module fails.
    """
    try:
        from analysis.meeting_intel import compute_meeting_intel, format_meeting_intel_section

        # Extract WHOOP data (same across all windows — take first hit)
        whoop_data: dict = {}
        for w in windows:
            wd = w.get("whoop") or {}
            if wd.get("recovery_score") is not None:
                whoop_data = wd
                break

        intel = compute_meeting_intel(windows, whoop_data, date_str)
        if not intel.is_meaningful:
            return None

        return {
            "mis": intel.meeting_intelligence_score,
            "ffs": intel.focus_fragmentation_score,
            "cmc": intel.cognitive_meeting_cost,
            "sdr": intel.social_drain_rate,
            "meeting_recovery_fit": intel.meeting_recovery_fit,
            "meeting_count": intel.meeting_count,
            "total_meeting_minutes": intel.total_meeting_minutes,
            "free_gap_minutes": intel.free_gap_minutes,
            "peak_focus_threats": intel.peak_focus_threats,
            "headline": intel.headline,
            "advisory": intel.advisory,
            # Pre-formatted Slack section so format_digest_message doesn't
            # need to re-import the module.
            "section": format_meeting_intel_section(intel),
        }
    except Exception:
        return None  # Never crash the digest over meeting intel


def _compute_ml_insights_for_digest(windows: list[dict]) -> Optional[dict]:
    """
    Compute ML model insights for the nightly digest (v2.2).

    Runs two ML inference passes:
      1. Isolation Forest anomaly detection — finds unusual 15-min windows
      2. Random Forest recovery prediction — predicts tomorrow's WHOOP recovery

    Both calls are fully exception-isolated; if either model is not trained
    or inference fails, the result is silently omitted (None returned for
    that sub-key).

    Returns a dict:
        {
            "recovery_prediction": {
                "predicted_recovery": float,
                "confidence": str,         # "high" | "medium" | "low"
                "prediction_std": float,
            } | None,
            "anomalies": [
                {
                    "window_id": str,
                    "hour_of_day": int,
                    "anomaly_score": float,
                    "features": dict,
                }
            ],
            "anomaly_clusters": [
                {
                    "start_hour": int,
                    "end_hour": int,
                    "count": int,
                    "description": str,   # e.g. "3 anomalous windows, 14:00–15:00"
                }
            ],
            "is_meaningful": bool,        # True when at least one signal is present
        }

    Returns None if both models are unavailable or all windows are empty.
    """
    try:
        from analysis.ml_model import detect_anomalies, predict_recovery

        # ── Recovery prediction ────────────────────────────────────────────
        recovery_pred: Optional[dict] = None
        try:
            recovery_pred = predict_recovery(windows)
        except Exception:
            pass  # Model not trained — silently skip

        # ── Anomaly detection ──────────────────────────────────────────────
        raw_anomalies: list[dict] = []
        try:
            raw_anomalies = detect_anomalies(windows)
        except Exception:
            pass  # Model not trained — silently skip

        # ── Cluster anomalies by consecutive hour ─────────────────────────
        # Group anomalous windows that fall in the same or adjacent hour into
        # a single cluster so the digest shows "14:00–15:30 (3 windows)"
        # instead of three separate lines.
        anomaly_clusters: list[dict] = []
        if raw_anomalies:
            # Sort by hour then by window_id for stable ordering
            sorted_anoms = sorted(raw_anomalies, key=lambda a: (a["hour_of_day"], a["window_id"]))

            # Group: a new cluster starts when the hour jumps by > 1
            current_cluster: list[dict] = [sorted_anoms[0]]
            for anm in sorted_anoms[1:]:
                last_hour = current_cluster[-1]["hour_of_day"]
                if anm["hour_of_day"] <= last_hour + 1:
                    current_cluster.append(anm)
                else:
                    anomaly_clusters.append(current_cluster)
                    current_cluster = [anm]
            anomaly_clusters.append(current_cluster)

            # Convert to dicts with human-readable descriptions
            cluster_dicts = []
            for cluster in anomaly_clusters:
                start_h = cluster[0]["hour_of_day"]
                end_h = cluster[-1]["hour_of_day"] + 1
                n = len(cluster)
                cluster_dicts.append({
                    "start_hour": start_h,
                    "end_hour": end_h,
                    "count": n,
                    "description": (
                        f"{n} anomalous window{'s' if n != 1 else ''} "
                        f"at {start_h:02d}:00–{end_h:02d}:00"
                    ),
                })
            anomaly_clusters = cluster_dicts

        # ── Determine meaningfulness ───────────────────────────────────────
        has_recovery = recovery_pred is not None
        # Only surface anomaly clusters with ≥ 2 windows to avoid noise
        significant_clusters = [c for c in anomaly_clusters if c["count"] >= 2]
        has_anomalies = len(significant_clusters) >= 1

        is_meaningful = has_recovery or has_anomalies

        if not is_meaningful:
            return None

        return {
            "recovery_prediction": recovery_pred,
            "anomalies": raw_anomalies,
            "anomaly_clusters": significant_clusters,
            "is_meaningful": is_meaningful,
        }

    except Exception:
        return None  # Never crash the digest over ML insights


def _compute_load_forecast_for_digest(today_date_str: str) -> Optional[dict]:
    """
    Compute tomorrow's predicted cognitive load for the nightly digest (v2.3).

    Mirrors the logic used in the morning brief, but placed in the *evening*
    digest so David can see tomorrow's expected load at the same time as the
    focus plan — giving a complete tomorrow preview in one glance.

    The morning brief already shows today's load forecast; the nightly digest
    now shows *tomorrow's* predicted load (i.e. the forecast for date+1).

    Returns a dict with:
      - line:           Slack-ready formatted line, e.g. "📊 Load forecast: Moderate..."
      - predicted_cls:  float
      - load_label:     str ("Very light" | "Light" | "Moderate" | "High" | "Very high")
      - confidence:     str ("high" | "medium" | "low")
      - meeting_minutes: int
      - narrative:      str — one actionable sentence
      - is_meaningful:  bool

    Returns None on error or when the forecast is not meaningful.
    Fully exception-isolated — never crashes the digest.
    """
    try:
        from datetime import timedelta

        tomorrow_dt = datetime.strptime(today_date_str, "%Y-%m-%d") + timedelta(days=1)
        tomorrow_str = tomorrow_dt.strftime("%Y-%m-%d")

        # Try to load tomorrow's calendar for meeting load context
        tomorrow_calendar = None
        try:
            from collectors.gcal import collect as gcal_collect
            tomorrow_calendar = gcal_collect(tomorrow_str)
        except Exception:
            tomorrow_calendar = None

        from analysis.load_forecast import compute_load_forecast, format_forecast_line
        # Pass tomorrow's date as the reference so the forecast is for tomorrow's load
        forecast = compute_load_forecast(tomorrow_str, tomorrow_calendar)

        if not forecast.is_meaningful:
            return None

        line = format_forecast_line(forecast)
        if not line:
            return None

        return {
            "line": line,
            "predicted_cls": forecast.predicted_cls,
            "load_label": forecast.load_label,
            "confidence": forecast.confidence,
            "meeting_minutes": forecast.meeting_minutes,
            "matching_days": forecast.matching_days,
            "narrative": forecast.narrative,
            "is_meaningful": True,
        }
    except Exception:
        return None  # Never crash the digest over load forecast


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

    # ── Omi conversation stats (v1.6) ─────────────────────────────────────
    # Aggregate spoken-conversation activity from working-hour windows.
    # Omi signals have been wired into CLS/SDI/FDI since v2.0, but were never
    # surfaced explicitly in the digest.  This shows David what drove those
    # metric shifts — how many conversations he had, how many words he spoke,
    # and how long he was actively talking.
    # Only populated when at least one Omi conversation was detected.
    omi_active_windows = [
        w for w in working
        if (w.get("omi") or {}).get("conversation_active", False)
    ]

    if omi_active_windows:
        omi_total_words = sum(
            (w.get("omi") or {}).get("word_count", 0) for w in omi_active_windows
        )
        omi_total_speech_secs = sum(
            (w.get("omi") or {}).get("speech_seconds", 0.0) for w in omi_active_windows
        )
        omi_total_sessions = sum(
            (w.get("omi") or {}).get("sessions_count", 0) for w in omi_active_windows
        )
        # v10.1: topic breakdown from v10.0 Omi topic classifier.
        # Aggregates category counts and mean cognitive density across
        # all active conversation windows.  Gracefully absent when topic
        # fields are not present in the JSONL (older data or no classifier).
        from collections import Counter as _Counter
        _cat_counts: _Counter = _Counter()
        _density_vals: list[float] = []
        for _w in omi_active_windows:
            _omi = _w.get("omi", {})
            _cat = _omi.get("topic_category") or _omi.get("category")
            if _cat and _cat not in ("unknown", None):
                _cat_counts[_cat] += 1
            _d = _omi.get("cognitive_density")
            if _d is not None:
                _density_vals.append(float(_d))

        _dominant = _cat_counts.most_common(1)[0][0] if _cat_counts else None
        _mean_density = round(sum(_density_vals) / len(_density_vals), 3) if _density_vals else None

        omi_digest: Optional[dict] = {
            "conversation_windows": len(omi_active_windows),
            "total_sessions": omi_total_sessions,
            "total_words": omi_total_words,
            "total_speech_minutes": round(omi_total_speech_secs / 60.0, 1),
            # v10.1 topic fields (None when topic classifier data unavailable)
            "dominant_topic": _dominant,
            "category_counts": dict(_cat_counts) if _cat_counts else None,
            "mean_cognitive_density": _mean_density,
        }
    else:
        omi_digest = None

    # ── Peak focus hour (v1.6) ─────────────────────────────────────────────
    # Find the working hour with the highest average active FDI.
    # Only considers active windows (meeting or Slack activity present) so that
    # idle hours (trivially FDI=1.0) don't crowd out genuinely productive periods.
    # Shows David when they're sharpest, useful for scheduling high-demand work.
    peak_focus_hour: Optional[int] = None
    peak_focus_fdi: Optional[float] = None
    if active:
        from collections import defaultdict as _defaultdict
        hour_fdi_vals: dict[int, list[float]] = _defaultdict(list)
        for w in active:
            hour_fdi_vals[w["metadata"]["hour_of_day"]].append(
                w["metrics"]["focus_depth_index"]
            )
        for h, vals in hour_fdi_vals.items():
            h_avg = sum(vals) / len(vals)
            if peak_focus_fdi is None or h_avg > peak_focus_fdi:
                peak_focus_fdi = round(h_avg, 4)
                peak_focus_hour = h

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

    # ── Personal records (v2.1) ───────────────────────────────────────────
    # Checks whether today set any all-time personal bests or notable streaks.
    # Gracefully returns None when insufficient history (< 2 days).
    personal_records_today = None
    try:
        from analysis.personal_records import (
            compute_personal_records,
            check_today_records,
            format_records_line,
        )
        _all_records = compute_personal_records(date_str)
        _today_summary = check_today_records(date_str, _all_records)
        if _today_summary.has_records:
            personal_records_today = {
                "new_bests": _today_summary.all_new_bests(),
                "active_streaks": [
                    {"name": n, "days": d}
                    for n, d in _today_summary.active_streaks()
                ],
                "new_streak_records": _today_summary.new_streak_records,
                "line": format_records_line(_today_summary),
            }
    except Exception:
        pass  # Never crash the digest over personal records

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

    # ── Cognitive Debt Index (v1.7) ─────────────────────────────────────────
    # Multi-day accumulated fatigue index.  Computes CDI for today using the
    # rolling summary store.  Gracefully returns None when insufficient data
    # (< 3 days) — the digest section is omitted in that case.
    cognitive_debt = None
    try:
        from analysis.cognitive_debt import compute_cdi, format_cdi_line
        _cdi = compute_cdi(date_str)
        if _cdi.is_meaningful:
            cognitive_debt = {
                "cdi": _cdi.cdi,
                "tier": _cdi.tier,
                "trend_5d": _cdi.trend_5d,
                "days_in_deficit": _cdi.days_in_deficit,
                "days_used": _cdi.days_used,
                "line": format_cdi_line(_cdi),
            }
    except Exception:
        pass  # CDI is non-critical; never crash the digest

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
        # v1.6: Omi conversation activity (None when no Omi data for the day)
        "omi": omi_digest,
        # v1.6: Peak focus hour (None when no active windows or FDI < threshold)
        "peak_focus_hour": peak_focus_hour,
        "peak_focus_fdi": peak_focus_fdi,
        # v1.7: Cognitive Debt Index (None when < 3 days of history)
        "cognitive_debt": cognitive_debt,
        # v1.8: Daily Presence Score — single 0–100 composite score
        # (None when insufficient working-hour windows)
        "presence_score": _compute_dps_for_digest(windows),
        # v1.9: Tomorrow's focus plan — specific deep-work blocks for tomorrow
        # (None when calendar unavailable or focus planner fails)
        "tomorrow_focus_plan": _compute_focus_plan_for_digest(date_str),
        # v2.0: Meeting Intelligence — FFS, CMC, SDR, MIS, recovery fit
        # (None when no meetings or meeting_intel fails)
        "meeting_intel": _compute_meeting_intel_for_digest(windows, date_str),
        # v2.1: Personal Records — new all-time bests and streak milestones
        # (None when no records set today or insufficient history)
        "personal_records": personal_records_today,
        # v2.2: ML model insights — recovery prediction + anomaly detection
        # (None when models not trained or both signals are absent)
        "ml_insights": _compute_ml_insights_for_digest(windows),
        # v2.3: Tomorrow's load forecast — predicted cognitive load for tomorrow
        # based on tomorrow's calendar and historical CLS patterns.
        # Shown alongside tomorrow's focus plan for a complete tomorrow preview.
        # (None when insufficient history or no calendar data)
        "tomorrow_load_forecast": _compute_load_forecast_for_digest(date_str),
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
    omi = digest.get("omi")               # None when no Omi conversations (v1.6)
    peak_focus_hour = digest.get("peak_focus_hour")   # v1.6
    peak_focus_fdi = digest.get("peak_focus_fdi")     # v1.6
    cognitive_debt = digest.get("cognitive_debt")     # v1.7: CDI dict or None
    presence_score = digest.get("presence_score")     # v1.8: DPS dict or None
    meeting_intel = digest.get("meeting_intel")       # v2.0: Meeting Intel dict or None
    ml_insights = digest.get("ml_insights")           # v2.2: ML model insights or None
    tomorrow_load_forecast = digest.get("tomorrow_load_forecast")  # v2.3: tomorrow load forecast or None

    lines = [
        f"*Presence Report — {date_label}*",
        "",
    ]

    # ── Daily Presence Score headline (v1.8) ──
    # One composite number before all other metrics — the cognitive equivalent
    # of WHOOP's daily strain score. "How was your cognitive day?" in one line.
    if presence_score:
        dps_line = presence_score.get("line", "")
        if dps_line:
            lines.append(dps_line)
            lines.append("")

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

    # ── Omi conversation activity (v1.6) ──────────────────────────────────
    # Show spoken conversation stats when Omi detected activity.
    # This surfaces the signal that's been affecting CLS/SDI/FDI since v2.0
    # but was never explicitly shown.  Sessions = distinct voice conversations.
    # Only shown when at least one conversation was detected today.
    if omi:
        omi_sessions = omi.get("total_sessions", 0)
        omi_words = omi.get("total_words", 0)
        omi_speech_min = omi.get("total_speech_minutes", 0.0)

        omi_parts = []
        if omi_sessions > 0:
            omi_parts.append(
                f"{omi_sessions} conversation{'s' if omi_sessions != 1 else ''}"
            )
        if omi_words > 0:
            omi_parts.append(f"{omi_words:,} words")
        if omi_speech_min > 0:
            omi_parts.append(f"{omi_speech_min:.0f} min speaking")

        if omi_parts:
            lines.append("_🎙 " + "  ·  ".join(omi_parts) + "_")

        # v10.1: Topic context line — shows what type of conversations happened.
        # Surfaces the topic classifier output that's been stored since v10.0.
        # Example: "_Topic: Technical (density 72%)_"
        # Only shown when topic data is available (post-v10.0 JSONL files).
        _TOPIC_EMOJI = {
            "work_technical": "⚙️",
            "work_strategic": "🎯",
            "personal":        "🏠",
            "operational":     "📦",
            "mixed":           "🔀",
        }
        _TOPIC_LABEL = {
            "work_technical": "Technical",
            "work_strategic": "Strategic",
            "personal":        "Personal",
            "operational":     "Operational",
            "mixed":           "Mixed",
        }
        dominant_topic = omi.get("dominant_topic")
        mean_density = omi.get("mean_cognitive_density")
        category_counts = omi.get("category_counts")

        if dominant_topic:
            emoji = _TOPIC_EMOJI.get(dominant_topic, "💬")
            label = _TOPIC_LABEL.get(dominant_topic, dominant_topic.replace("_", " ").title())
            topic_parts = [f"{emoji} {label}"]
            if mean_density is not None:
                topic_parts.append(f"density {mean_density:.0%}")
            # If multiple categories, show the split
            if category_counts and len(category_counts) > 1:
                total_cat = sum(category_counts.values())
                dominant_pct = round(100.0 * category_counts[dominant_topic] / total_cat)
                if dominant_pct < 80:  # Only mention split when meaningful
                    other_cats = [
                        _TOPIC_LABEL.get(c, c.replace("_", " ").title())
                        for c in category_counts
                        if c != dominant_topic
                    ]
                    if other_cats:
                        topic_parts.append(f"+ {', '.join(other_cats[:2]).lower()}")
            lines.append("_" + "  ·  ".join(topic_parts) + "_")

    # ── Peak focus hour (v1.6) ─────────────────────────────────────────────
    # Surface the single best focus hour of the day — useful for scheduling.
    # Only shown when there was meaningful focused work (FDI ≥ 0.70 peak).
    # Threshold avoids showing a peak when the whole day was fragmented.
    _PEAK_FOCUS_THRESHOLD = 0.70
    if peak_focus_hour is not None and peak_focus_fdi is not None and peak_focus_fdi >= _PEAK_FOCUS_THRESHOLD:
        end_hour = peak_focus_hour + 1
        lines.append(
            f"_🏆 Best focus: {peak_focus_hour:02d}:00–{end_hour:02d}:00 "
            f"(FDI {peak_focus_fdi:.0%})_"
        )

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

    # ── Cognitive Debt Index (v1.7) ───────────────────────────────────────
    # Multi-day accumulated fatigue indicator — shown when meaningful (≥ 3 days).
    # Surfaces the CDI line computed in compute_digest().
    # Example: "🟠 CDI 63/100 — Loading (5 deficit days in 14d, trend ↑ fatigue)"
    if cognitive_debt:
        cdi_line = cognitive_debt.get("line", "")
        if cdi_line:
            lines.append("")
            lines.append(f"_{cdi_line}_")

    # ── ML Model Insights (v2.2) ─────────────────────────────────────────
    # Surface ML-derived signals: recovery prediction and behavioral anomalies.
    # Both are hidden by default (model not trained early on) — surfaced once
    # enough data has been collected.
    #
    # Recovery prediction: "🤖 Tomorrow's recovery: ~72% (high confidence)"
    #   Shown when the Random Forest recovery predictor is trained.
    #   Lets David adjust tonight's wind-down based on predicted tomorrow.
    #
    # Anomaly clusters: "🔍 Unusual pattern at 14:00–15:00 (2 anomalous windows)"
    #   Shown when ≥ 2 anomalous windows form a cluster.
    #   Draws attention to periods that don't fit David's behavioral fingerprint.
    if ml_insights and ml_insights.get("is_meaningful"):
        ml_lines = []

        recovery_pred = ml_insights.get("recovery_prediction")
        if recovery_pred and recovery_pred.get("predicted_recovery") is not None:
            predicted = round(recovery_pred["predicted_recovery"])
            confidence = recovery_pred.get("confidence", "")
            confidence_str = f" ({confidence} confidence)" if confidence else ""
            ml_lines.append(f"🤖 Tomorrow's recovery: ~{predicted}%{confidence_str}")

        for cluster in (ml_insights.get("anomaly_clusters") or []):
            desc = cluster.get("description", "")
            if desc:
                ml_lines.append(f"🔍 {desc}")

        if ml_lines:
            lines.append("")
            for ml_line in ml_lines:
                lines.append(f"_{ml_line}_")

    # ── Meeting Intelligence (v2.0) ───────────────────────────────────────
    # Only rendered when the day had meetings AND the module produced meaningful
    # output (MIS, FFS, CMC, SDR).  The pre-formatted section is used directly
    # so we keep all rendering logic inside meeting_intel.py.
    if meeting_intel and meeting_intel.get("section"):
        lines.append("")
        lines.append(meeting_intel["section"])

    # ── Personal Records (v2.1) ───────────────────────────────────────────
    # Celebrate new personal bests and notable streaks.
    # Only shown when at least one record was set or a streak is active.
    personal_records = digest.get("personal_records")
    if personal_records and personal_records.get("line"):
        lines.append("")
        lines.append(personal_records["line"])

    # ── Insight ──
    if insight:
        lines.append("")
        lines.append(f"💡 {insight}")

    # ── Tomorrow Preview (v2.3) ───────────────────────────────────────────
    # Combine load forecast + focus plan into a unified tomorrow preview section.
    # Placed at the bottom of the digest so David ends with forward-looking info.
    #
    # Load forecast (new in v2.3): "Tomorrow looks like a Moderate day (CLS ~0.45)"
    # Focus plan (v1.9): specific deep-work blocks for tomorrow
    #
    # Both are individually optional — shown only when meaningful data exists.
    tomorrow_focus_plan = digest.get("tomorrow_focus_plan")
    has_tomorrow_load = tomorrow_load_forecast and tomorrow_load_forecast.get("is_meaningful")
    has_tomorrow_plan = tomorrow_focus_plan and tomorrow_focus_plan.get("is_meaningful") and tomorrow_focus_plan.get("section")

    if has_tomorrow_load or has_tomorrow_plan:
        lines.append("")
        lines.append("*Tomorrow*")

        if has_tomorrow_load:
            forecast_line = tomorrow_load_forecast.get("line", "")
            if forecast_line:
                lines.append(forecast_line)
            # Show the narrative as additional context if present and different from the line
            narrative = tomorrow_load_forecast.get("narrative", "")
            if narrative:
                lines.append(f"_{narrative}_")

        if has_tomorrow_plan:
            if has_tomorrow_load:
                lines.append("")
            lines.append(tomorrow_focus_plan["section"])
    elif tomorrow_focus_plan and tomorrow_focus_plan.get("is_meaningful") and tomorrow_focus_plan.get("section"):
        # Legacy path: focus plan without load forecast
        lines.append("")
        lines.append(tomorrow_focus_plan["section"])

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

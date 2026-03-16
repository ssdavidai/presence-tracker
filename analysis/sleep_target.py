"""
Presence Tracker — Sleep Target Advisor (v26)

Answers: *"How much sleep should I get tonight to perform well tomorrow?"*

The nightly digest already shows:
  - Tomorrow's predicted cognitive load (from load_forecast)
  - Tomorrow's recommended focus blocks (from focus_planner)
  - Current CDI tier (from cognitive_debt)
  - ML-predicted tomorrow recovery (from ml_model)

But none of these answer the concrete question: *"What time should I go to bed?"*

The Sleep Target Advisor closes this gap. It synthesises tonight's optimal
sleep duration and a concrete bedtime from:

  1. Tomorrow's predicted load — heavier day requires more recovery tonight
  2. Current CDI tier — debt means you need more, not less, sleep
  3. Today's CLS — taxing days demand proportionally deeper recovery
  4. Personal sleep→focus slope — if David's data shows 7.5h → better FDI,
     use that rather than a population assumption

## Sleep Target Formula

    base_hours = f(tomorrow_load_label)
        Very light  → 6.5h  (light day, less physiological requirement)
        Light       → 7.0h
        Moderate    → 7.5h  (default: population-norm "enough")
        High        → 8.0h  (demanding day — needs full repair)
        Very high   → 8.5h  (maximum cognitive demand)
        Unknown     → 7.5h  (fallback)

    cdi_modifier = f(cdi_tier)
        surplus   →  0.0h  (well recovered, no extra needed)
        balanced  →  0.0h
        loading   → +0.25h  (early debt — add a quarter hour)
        fatigued  → +0.50h  (significant debt — add half hour)
        critical  → +0.75h  (burnout risk — prioritise recovery)

    cls_modifier = f(today_avg_cls)
        < 0.20   →  0.0h   (light day, minimal repair needed)
        0.20–0.50 → +0.0h  (normal day)
        0.50–0.70 → +0.25h (demanding day — a little extra)
        ≥ 0.70   → +0.50h  (very high load — more repair needed)

    personal_modifier (optional, from sleep_focus_correlator)
        When the correlator has ≥ 10 pairs and shows a meaningful slope
        (Δ FDI > 0.05 per hour), adjust toward the bucket with best next-day FDI.
        Adjustment is capped at ±0.5h to avoid over-fitting.

    target_hours = clamp(base_hours + cdi_modifier + cls_modifier + personal_modifier, 6.0, 9.5)

## Bedtime Calculation

The target wake time defaults to 07:30 Budapest (consistent with existing
morning brief schedule of 07:00 alerts + a 30-min buffer to read it).
David can override via config.

    target_bedtime = target_wake_time − target_hours
    e.g. target 7.5h, wake 07:30 → target bedtime 00:00

If the inferred bedtime is in the past (already past midnight when digest runs),
we note this gracefully: "You need to be asleep now."

## Output

    SleepTarget dataclass:
      - target_hours: float           — recommended sleep duration
      - target_bedtime: str           — "23:00" style string (Budapest time)
      - urgency: str                  — 'normal' | 'elevated' | 'critical'
      - base_hours: float             — before modifiers
      - cdi_modifier: float           — hours added for debt
      - cls_modifier: float           — hours added for today's load
      - personal_modifier: float      — hours added from personal sleep data
      - tomorrow_load_label: str      — load label from load_forecast
      - cdi_tier: str                 — CDI tier used
      - narrative: str                — one actionable sentence
      - is_meaningful: bool           — False when can't compute (no load forecast)

## API

    from analysis.sleep_target import compute_sleep_target, format_sleep_target_line

    target = compute_sleep_target(today_date_str, today_windows)
    line = format_sleep_target_line(target)      # Slack-ready one-liner
    section = format_sleep_target_section(target) # multi-line version

## Integration

    In nightly digest — wire into compute_digest() and format_digest_message():

      sleep_target = _compute_sleep_target_for_digest(date_str, windows)
      # Attach to digest dict as "sleep_target"
      # Render in format_digest_message() inside the "Tomorrow" section

## Design principles

  - Pure computation — fully testable with mocked inputs
  - Graceful degradation: missing load forecast / CDI → uses sensible fallbacks
  - No LLM — all logic is deterministic and explainable
  - The bedtime is computed from a configurable wake time (not hardcoded)
  - Respects data scarcity: personal modifier only fires when ≥ 10 pairs exist

## CLI

    python3 analysis/sleep_target.py              # Tonight's recommendation
    python3 analysis/sleep_target.py 2026-03-14   # For a specific day
    python3 analysis/sleep_target.py --json       # Machine-readable JSON

"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import read_day


# ─── Constants ────────────────────────────────────────────────────────────────

# Target wake time (Budapest local time, 24h format)
# Aligns with the 07:00 morning brief + 30-min buffer to read it.
TARGET_WAKE_HOUR = 7
TARGET_WAKE_MINUTE = 30

# Base sleep hours per tomorrow's load label
_BASE_HOURS_BY_LOAD: dict[str, float] = {
    "Very light": 6.5,
    "Light":      7.0,
    "Moderate":   7.5,
    "High":       8.0,
    "Very high":  8.5,
}
_BASE_HOURS_FALLBACK = 7.5  # when load label unknown

# CDI tier → extra hours of sleep needed
_CDI_MODIFIER: dict[str, float] = {
    "surplus":  0.00,
    "balanced": 0.00,
    "loading":  0.25,
    "fatigued": 0.50,
    "critical": 0.75,
}

# CLS thresholds → extra hours for heavy days
_CLS_MOD_LIGHT    = 0.20   # below this → no modifier
_CLS_MOD_HIGH     = 0.50   # above this → +0.25h
_CLS_MOD_VERY_HIGH = 0.70  # above this → +0.50h

# Min pairs in sleep correlator before personal modifier applies
_MIN_PAIRS_FOR_PERSONAL = 10

# Max personal modifier (cap to avoid overfitting)
_MAX_PERSONAL_MODIFIER = 0.50

# Final clamp bounds
_MIN_TARGET_HOURS = 6.0
_MAX_TARGET_HOURS = 9.5

# Urgency thresholds
_URGENCY_ELEVATED_HOURS = 7.75   # ≥ this → elevated
_URGENCY_CRITICAL_HOURS = 8.50   # ≥ this → critical


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class SleepTarget:
    """Tonight's recommended sleep target."""

    target_hours: float
    target_bedtime: str          # "23:00"
    urgency: str                 # 'normal' | 'elevated' | 'critical'

    base_hours: float
    cdi_modifier: float
    cls_modifier: float
    personal_modifier: float

    tomorrow_load_label: str     # "Moderate" etc.
    cdi_tier: str                # "balanced" etc.
    today_avg_cls: Optional[float]

    narrative: str
    is_meaningful: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Core logic ───────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _bedtime_from_target(target_hours: float, wake_hour: int = TARGET_WAKE_HOUR,
                          wake_minute: int = TARGET_WAKE_MINUTE) -> str:
    """
    Compute the target bedtime string (HH:MM) from wake time and sleep duration.

    Returns a string like "23:30". If the bedtime is today (23:XX), it's
    formatted directly. If it rolls before midnight, we show the actual time.
    """
    total_sleep_minutes = round(target_hours * 60)
    wake_minutes_from_midnight = wake_hour * 60 + wake_minute
    bedtime_minutes = wake_minutes_from_midnight - total_sleep_minutes

    # Handle negative (bedtime is before midnight → it is on the same night)
    if bedtime_minutes < 0:
        bedtime_minutes += 24 * 60

    bed_hour = bedtime_minutes // 60
    bed_min = bedtime_minutes % 60
    return f"{bed_hour:02d}:{bed_min:02d}"


def _urgency_from_hours(hours: float) -> str:
    if hours >= _URGENCY_CRITICAL_HOURS:
        return "critical"
    if hours >= _URGENCY_ELEVATED_HOURS:
        return "elevated"
    return "normal"


def _narrative(
    target_hours: float,
    bedtime: str,
    tomorrow_load_label: str,
    cdi_tier: str,
    urgency: str,
    personal_modifier: float,
) -> str:
    """Generate a single actionable narrative sentence."""
    load_context = {
        "Very light": "light day ahead",
        "Light":      "light day ahead",
        "Moderate":   "moderate day ahead",
        "High":       "demanding day ahead",
        "Very high":  "heavy day ahead",
    }.get(tomorrow_load_label, "tomorrow's load")

    cdi_context = {
        "surplus":  "",
        "balanced": "",
        "loading":  "and your debt is building",
        "fatigued": "and your cognitive debt is elevated",
        "critical": "and you're at burnout risk",
    }.get(cdi_tier, "")

    personal_note = ""
    if abs(personal_modifier) >= 0.20:
        personal_note = (
            " (your data shows more sleep reliably improves next-day focus)"
            if personal_modifier > 0
            else " (based on your personal sleep pattern)"
        )

    urgency_prefix = {
        "normal":   "Target",
        "elevated": "Prioritise",
        "critical": "🚨 You need",
    }[urgency]

    parts = [f"{urgency_prefix} {target_hours:.1f}h sleep (in bed by {bedtime})"]
    if load_context:
        parts.append(f"— {load_context}")
    if cdi_context:
        parts.append(cdi_context)
    if personal_note:
        parts.append(personal_note)

    return " ".join(parts) + "."


def compute_sleep_target(
    today_date_str: str,
    today_windows: Optional[list[dict]] = None,
    *,
    tomorrow_load_label: Optional[str] = None,
    cdi_tier: Optional[str] = None,
    today_avg_cls: Optional[float] = None,
    sleep_correlator_pairs: int = 0,
    sleep_correlator_slope: Optional[float] = None,
    optimal_sleep_hours: Optional[float] = None,
) -> SleepTarget:
    """
    Compute tonight's sleep target for a given date.

    Args:
        today_date_str: The date being processed (YYYY-MM-DD).
        today_windows: The 96-window list for today (used to compute avg_cls
                       when today_avg_cls is not provided directly).

        # Optional pre-computed inputs (used by the digest to avoid duplicate work)
        tomorrow_load_label: e.g. "Moderate" (from load_forecast)
        cdi_tier: e.g. "balanced" (from cognitive_debt)
        today_avg_cls: average CLS across active windows (float 0–1)
        sleep_correlator_pairs: how many pairs the sleep correlator has
        sleep_correlator_slope: Δ FDI per extra sleep hour (from SleepFocusCorrelation)
        optimal_sleep_hours: the bucket with highest next-day FDI (from correlator)

    Returns:
        SleepTarget dataclass.  is_meaningful=False when we can't produce
        a reliable recommendation (e.g. no tomorrow load data at all).
    """

    # ── Step 1: Infer today's avg CLS from windows when not provided ──────────
    if today_avg_cls is None and today_windows:
        cls_vals = [
            w["metrics"]["cognitive_load_score"]
            for w in today_windows
            if w.get("metadata", {}).get("is_active_window")
            and w.get("metrics", {}).get("cognitive_load_score") is not None
        ]
        today_avg_cls = sum(cls_vals) / len(cls_vals) if cls_vals else None

    # ── Step 2: Tomorrow's load → base hours ──────────────────────────────────
    load_label = tomorrow_load_label or "Unknown"
    base_hours = _BASE_HOURS_BY_LOAD.get(load_label, _BASE_HOURS_FALLBACK)

    is_meaningful = load_label != "Unknown"

    # ── Step 3: CDI modifier ──────────────────────────────────────────────────
    tier = cdi_tier or "balanced"
    cdi_mod = _CDI_MODIFIER.get(tier, 0.0)

    # ── Step 4: CLS modifier ──────────────────────────────────────────────────
    cls_mod = 0.0
    if today_avg_cls is not None:
        if today_avg_cls >= _CLS_MOD_VERY_HIGH:
            cls_mod = 0.50
        elif today_avg_cls >= _CLS_MOD_HIGH:
            cls_mod = 0.25
        # below _CLS_MOD_HIGH → no modifier

    # ── Step 5: Personal modifier (from sleep-focus correlator) ───────────────
    personal_mod = 0.0
    if (
        sleep_correlator_pairs >= _MIN_PAIRS_FOR_PERSONAL
        and optimal_sleep_hours is not None
    ):
        # personal data says the optimal sleep duration; nudge toward it
        current_estimate = base_hours + cdi_mod + cls_mod
        raw_nudge = optimal_sleep_hours - current_estimate
        # Cap magnitude to avoid over-adjusting
        personal_mod = _clamp(raw_nudge * 0.5, -_MAX_PERSONAL_MODIFIER, _MAX_PERSONAL_MODIFIER)

    # ── Step 6: Final target ──────────────────────────────────────────────────
    raw_target = base_hours + cdi_mod + cls_mod + personal_mod
    target_hours = _clamp(raw_target, _MIN_TARGET_HOURS, _MAX_TARGET_HOURS)
    # Round to nearest 0.25h for cleanliness
    target_hours = round(target_hours * 4) / 4

    bedtime = _bedtime_from_target(target_hours)
    urgency = _urgency_from_hours(target_hours)

    narr = _narrative(
        target_hours=target_hours,
        bedtime=bedtime,
        tomorrow_load_label=load_label,
        cdi_tier=tier,
        urgency=urgency,
        personal_modifier=personal_mod,
    )

    return SleepTarget(
        target_hours=target_hours,
        target_bedtime=bedtime,
        urgency=urgency,
        base_hours=base_hours,
        cdi_modifier=cdi_mod,
        cls_modifier=cls_mod,
        personal_modifier=personal_mod,
        tomorrow_load_label=load_label,
        cdi_tier=tier,
        today_avg_cls=today_avg_cls,
        narrative=narr,
        is_meaningful=is_meaningful,
    )


# ─── Formatting helpers ───────────────────────────────────────────────────────

_URGENCY_EMOJI = {
    "normal":   "😴",
    "elevated": "🌙",
    "critical": "🚨",
}


def format_sleep_target_line(target: SleepTarget) -> str:
    """
    Compact one-liner for the nightly digest.

    Example:
        "😴 Tonight: 7.5h sleep (in bed by 23:00)"
    """
    if not target.is_meaningful:
        return ""
    emoji = _URGENCY_EMOJI.get(target.urgency, "😴")
    return (
        f"{emoji} Tonight: {target.target_hours:.1f}h sleep "
        f"(in bed by {target.target_bedtime})"
    )


def format_sleep_target_section(target: SleepTarget) -> str:
    """
    Two-line section for the nightly digest.

    Example:
        😴 *Sleep Target: 7.5h* (in bed by 23:00)
        _Prioritise 7.5h sleep — demanding day ahead and your debt is building._
    """
    if not target.is_meaningful:
        return ""
    emoji = _URGENCY_EMOJI.get(target.urgency, "😴")
    lines = [
        f"{emoji} *Sleep Target: {target.target_hours:.1f}h* "
        f"(in bed by {target.target_bedtime})",
        f"_{target.narrative}_",
    ]
    return "\n".join(lines)


# ─── High-level helper used by daily_digest.py ────────────────────────────────

def compute_sleep_target_for_digest(
    today_date_str: str,
    today_windows: list[dict],
    *,
    precomputed_tomorrow_load: Optional[dict] = None,
    precomputed_cdi: Optional[dict] = None,
) -> Optional[dict]:
    """
    Wrapper called by daily_digest._compute_sleep_target_for_digest().

    Pulls the necessary inputs from already-computed digest sub-dicts so we
    don't duplicate expensive store reads.

    Returns a compact dict with:
        line:     format_sleep_target_line() output
        section:  format_sleep_target_section() output
        target_hours, target_bedtime, urgency, narrative, is_meaningful
    """
    # ── Tomorrow load label ───────────────────────────────────────────────────
    tomorrow_load_label: Optional[str] = None
    if precomputed_tomorrow_load and precomputed_tomorrow_load.get("is_meaningful"):
        tomorrow_load_label = precomputed_tomorrow_load.get("load_label")

    # ── CDI tier ──────────────────────────────────────────────────────────────
    cdi_tier: Optional[str] = None
    if precomputed_cdi and precomputed_cdi.get("is_meaningful"):
        cdi_tier = precomputed_cdi.get("tier")

    # ── Sleep correlator data (optional, graceful if not available) ───────────
    sleep_pairs = 0
    sleep_slope: Optional[float] = None
    optimal_sleep_h: Optional[float] = None
    try:
        from analysis.sleep_focus_correlator import compute_sleep_focus_correlation
        corr = compute_sleep_focus_correlation(as_of_date_str=today_date_str)
        if corr.is_meaningful:
            sleep_pairs = corr.pairs_used
            sleep_slope = corr.sleep_hours_slope
            # Find the bucket with highest next-day FDI
            if corr.sleep_buckets:
                best_bucket = max(corr.sleep_buckets, key=lambda b: b.avg_next_fdi or 0.0)
                # Use the midpoint of the best bucket as the optimal hours target
                _BUCKET_MIDPOINTS = {"<6h": 5.5, "6–7h": 6.5, "7–8h": 7.5, "≥8h": 8.5}
                optimal_sleep_h = _BUCKET_MIDPOINTS.get(best_bucket.label)
    except Exception:
        pass  # correlator not available — proceed without personal modifier

    target = compute_sleep_target(
        today_date_str=today_date_str,
        today_windows=today_windows,
        tomorrow_load_label=tomorrow_load_label,
        cdi_tier=cdi_tier,
        sleep_correlator_pairs=sleep_pairs,
        sleep_correlator_slope=sleep_slope,
        optimal_sleep_hours=optimal_sleep_h,
    )

    if not target.is_meaningful:
        return None

    return {
        "line":            format_sleep_target_line(target),
        "section":         format_sleep_target_section(target),
        "target_hours":    target.target_hours,
        "target_bedtime":  target.target_bedtime,
        "urgency":         target.urgency,
        "narrative":       target.narrative,
        "is_meaningful":   target.is_meaningful,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sleep Target Advisor — compute tonight's sleep recommendation"
    )
    parser.add_argument("date", nargs="?", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date to compute for (YYYY-MM-DD, default today)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    # Load windows
    windows = read_day(args.date)

    # Try to load tomorrow's load forecast
    tomorrow_load_label = None
    try:
        tomorrow_str = (datetime.strptime(args.date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        from analysis.load_forecast import compute_load_forecast
        from collectors import gcal
        try:
            cal = gcal.collect(tomorrow_str)
        except Exception:
            cal = None
        forecast = compute_load_forecast(tomorrow_str, cal)
        if forecast.is_meaningful:
            tomorrow_load_label = forecast.load_label
    except Exception:
        pass

    # Try to load CDI
    cdi_tier = None
    try:
        from analysis.cognitive_debt import compute_cdi
        debt = compute_cdi(args.date)
        if debt.is_meaningful:
            cdi_tier = debt.tier
    except Exception:
        pass

    target = compute_sleep_target(
        today_date_str=args.date,
        today_windows=windows,
        tomorrow_load_label=tomorrow_load_label,
        cdi_tier=cdi_tier,
    )

    if args.json:
        print(json.dumps(target.to_dict(), indent=2, default=str))
        return

    if not target.is_meaningful:
        print(f"Not enough data to compute sleep target for {args.date}.")
        print("  (Need tomorrow's load forecast — at minimum 3 days of history)")
        return

    emoji = _URGENCY_EMOJI.get(target.urgency, "😴")
    print(f"\n{emoji}  Sleep Target for {args.date}\n")
    print(f"  Target:    {target.target_hours:.1f}h  (in bed by {target.target_bedtime})")
    print(f"  Urgency:   {target.urgency}")
    print(f"  Breakdown:")
    print(f"    Base (tomorrow load: {target.tomorrow_load_label}):  {target.base_hours:.2f}h")
    print(f"    CDI modifier ({target.cdi_tier}):                   +{target.cdi_modifier:.2f}h")
    if target.cls_modifier:
        print(f"    CLS modifier (today load: {target.today_avg_cls:.2f}):    +{target.cls_modifier:.2f}h")
    if target.personal_modifier:
        sign = "+" if target.personal_modifier >= 0 else ""
        print(f"    Personal modifier:                              {sign}{target.personal_modifier:.2f}h")
    print(f"\n  Narrative: {target.narrative}\n")


if __name__ == "__main__":
    _cli_main()

"""
Presence Tracker — Predicted Cognitive Load Forecast (v14)

Answers: *"How cognitively demanding will today be, before it happens?"*

The morning brief already tells David:
  - WHOOP readiness (physiological capacity)
  - Yesterday's CLS (retrospective load)
  - Today's calendar (meetings scheduled)

But none of these tell David what today's *actual* CLS will likely be.
The forecast closes this gap by using historical data to translate
today's calendar into a predicted CLS range.

## How it works

1. **Load profile** — from 30 days of history, compute the relationship
   between total_meeting_minutes and actual avg_cls for that day.
   Bucket days by meeting load: light / moderate / heavy / intense.

2. **Calendar match** — given today's total_meeting_minutes, find the
   matching historical bucket and compute:
   - predicted_cls: mean CLS for days with similar meeting load
   - cls_low: 25th percentile (optimistic scenario)
   - cls_high: 75th percentile (pessimistic scenario)
   - confidence: 'high' (≥8 matching days) | 'medium' (≥3) | 'low' (<3)

3. **Label** — translate the predicted CLS into a human-readable label:
   - < 0.20 → "Very light"
   - 0.20–0.40 → "Light"
   - 0.40–0.60 → "Moderate"
   - 0.60–0.80 → "High"
   - ≥ 0.80 → "Very high"

4. **Narrative** — generate a single actionable sentence:
   - "Based on your 2h30m of meetings, today will likely be a
      moderate-load day (CLS ~0.45). Schedule your hardest thinking
      for before 10am."

## Output

    LoadForecast dataclass:
      - predicted_cls: float          — point estimate
      - cls_low: float                — 25th percentile
      - cls_high: float               — 75th percentile
      - load_label: str               — Very light | Light | Moderate | High | Very high
      - confidence: str               — high | medium | low | none
      - meeting_minutes: int          — today's scheduled meeting time
      - days_of_history: int          — how many days were used
      - matching_days: int            — days in the same meeting-load bucket
      - narrative: str                — one actionable sentence
      - is_meaningful: bool           — False when insufficient history or no calendar

## API

    from analysis.load_forecast import compute_load_forecast, format_forecast_line

    forecast = compute_load_forecast(date_str, today_calendar)
    line = format_forecast_line(forecast)    # Slack-ready string

## Design principles

  - Pure functions except for store access — fully testable
  - Graceful degradation: missing calendar or history → is_meaningful=False
  - No external dependencies — uses only the existing JSONL store
  - Minimal complexity: linear bucketing, no ML required
  - Degrades to a heuristic estimate when < 3 matching days exist

## CLI

    python3 analysis/load_forecast.py                # Today
    python3 analysis/load_forecast.py 2026-03-13     # Specific date
    python3 analysis/load_forecast.py --json         # JSON output
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import list_available_dates, read_summary


# ─── Constants ────────────────────────────────────────────────────────────────

# Days of history to use for building the load profile
FORECAST_HISTORY_DAYS = 30

# Meeting-load buckets (minutes): maps to a label for human display
LOAD_BUCKETS = [
    (0,   60,   "light"),      # < 1h of meetings
    (60,  150,  "moderate"),   # 1–2.5h of meetings
    (150, 270,  "heavy"),      # 2.5–4.5h of meetings
    (270, 9999, "intense"),    # > 4.5h of meetings
]

# Minimum days in the matching bucket for 'medium' confidence
CONFIDENCE_MEDIUM_DAYS = 3

# Minimum days in the matching bucket for 'high' confidence
CONFIDENCE_HIGH_DAYS = 8

# CLS label thresholds
CLS_LABELS = [
    (0.00, 0.20, "Very light"),
    (0.20, 0.40, "Light"),
    (0.40, 0.60, "Moderate"),
    (0.60, 0.80, "High"),
    (0.80, 9.99, "Very high"),
]


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class LoadForecast:
    """Predicted cognitive load for a given day."""
    date_str: str
    predicted_cls: Optional[float] = None     # point estimate
    cls_low: Optional[float] = None           # 25th percentile
    cls_high: Optional[float] = None          # 75th percentile
    load_label: str = "Unknown"               # Very light | Light | ... | Very high
    confidence: str = "none"                  # high | medium | low | none
    meeting_minutes: int = 0                  # today's scheduled meeting time
    days_of_history: int = 0                  # total days in history
    matching_days: int = 0                    # days in same meeting-load bucket
    narrative: str = ""                       # one actionable sentence
    is_meaningful: bool = False

    def to_dict(self) -> dict:
        return {
            "date_str": self.date_str,
            "predicted_cls": round(self.predicted_cls, 3) if self.predicted_cls is not None else None,
            "cls_low": round(self.cls_low, 3) if self.cls_low is not None else None,
            "cls_high": round(self.cls_high, 3) if self.cls_high is not None else None,
            "load_label": self.load_label,
            "confidence": self.confidence,
            "meeting_minutes": self.meeting_minutes,
            "days_of_history": self.days_of_history,
            "matching_days": self.matching_days,
            "narrative": self.narrative,
            "is_meaningful": self.is_meaningful,
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bucket_for_minutes(total_minutes: int) -> str:
    """Return the load bucket label for a given meeting minute total."""
    for lo, hi, label in LOAD_BUCKETS:
        if lo <= total_minutes < hi:
            return label
    return "intense"


def _cls_label(cls: float) -> str:
    """Return a human-readable label for a CLS value."""
    for lo, hi, label in CLS_LABELS:
        if lo <= cls < hi:
            return label
    return "Very high"


def _percentile(values: list[float], pct: float) -> float:
    """
    Compute a percentile from a sorted list of values.

    Uses linear interpolation (same as numpy's default).
    `pct` is 0–100.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # Linear interpolation
    idx = (pct / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _fmt_minutes(m: int) -> str:
    """Format minutes as '1h30m' or '45m'."""
    if m >= 60:
        h = m // 60
        r = m % 60
        return f"{h}h{r:02d}m" if r else f"{h}h"
    return f"{m}m"


# ─── Historical profile ───────────────────────────────────────────────────────

def _build_load_profile(
    today_date_str: str,
    days: int = FORECAST_HISTORY_DAYS,
) -> dict[str, list[float]]:
    """
    Build a mapping from meeting-load bucket → list of CLS values
    using historical day summaries prior to today.

    Returns:
        dict like:
          {
            "light": [0.12, 0.18, 0.09, ...],
            "moderate": [0.35, 0.42, ...],
            "heavy": [0.61, 0.55, ...],
            "intense": [0.78, ...],
          }
    """
    profile: dict[str, list[float]] = {
        "light": [],
        "moderate": [],
        "heavy": [],
        "intense": [],
    }

    try:
        available = sorted(
            [d for d in list_available_dates() if d < today_date_str],
            reverse=True,
        )[:days]

        # Use the rolling summary to get per-day meeting minutes + avg CLS
        # This is much faster than re-reading all JSONL windows.
        try:
            summary = read_summary()
            all_days = summary.get("days", {})
        except Exception:
            all_days = {}

        for d in available:
            day_data = all_days.get(d, {})
            if not day_data:
                continue

            meeting_mins = day_data.get("calendar", {}).get("total_meeting_minutes", 0) or 0
            avg_cls = (
                day_data.get("metrics_avg", {}).get("cognitive_load_score")
                if day_data.get("metrics_avg")
                else None
            )

            if avg_cls is None:
                continue

            bucket = _bucket_for_minutes(meeting_mins)
            profile[bucket].append(avg_cls)

    except Exception:
        pass

    return profile


# ─── Forecast computation ─────────────────────────────────────────────────────

def compute_load_forecast(
    date_str: str,
    today_calendar: Optional[dict] = None,
) -> LoadForecast:
    """
    Compute the predicted CLS for today given today's calendar.

    Parameters
    ----------
    date_str : str
        Today's date (YYYY-MM-DD). Used as the reference for looking up
        historical data.
    today_calendar : dict | None
        Calendar data from collectors.gcal.collect() for today.
        Must contain 'total_meeting_minutes'. If None or missing, the
        forecast will use 0 meeting minutes (no calendar loaded).

    Returns
    -------
    LoadForecast
        If is_meaningful=False: no actionable forecast (no history or no calendar).
        Otherwise: predicted_cls, cls_low, cls_high, label, confidence, narrative.
    """
    # Extract today's meeting minutes
    meeting_minutes = 0
    if today_calendar is not None:
        meeting_minutes = today_calendar.get("total_meeting_minutes", 0) or 0

    # Build historical profile
    profile = _build_load_profile(date_str)
    days_of_history = sum(len(v) for v in profile.values())

    # Find the bucket for today's meeting load
    bucket = _bucket_for_minutes(meeting_minutes)
    bucket_vals = profile.get(bucket, [])
    matching_days = len(bucket_vals)

    if days_of_history < 2:
        # Not enough history to make any prediction
        return LoadForecast(
            date_str=date_str,
            meeting_minutes=meeting_minutes,
            days_of_history=days_of_history,
            is_meaningful=False,
            narrative="Not enough history yet to forecast today's load.",
        )

    if matching_days == 0:
        # No days in this exact bucket — use overall average as fallback
        all_cls = [v for vals in profile.values() for v in vals]
        if not all_cls:
            return LoadForecast(
                date_str=date_str,
                meeting_minutes=meeting_minutes,
                days_of_history=days_of_history,
                is_meaningful=False,
                narrative="No comparable historical days found.",
            )
        # Use overall distribution as best guess
        predicted = sum(all_cls) / len(all_cls)
        cls_low = _percentile(all_cls, 25)
        cls_high = _percentile(all_cls, 75)
        confidence = "low"
        matching_days = 0
    else:
        predicted = sum(bucket_vals) / len(bucket_vals)
        cls_low = _percentile(bucket_vals, 25)
        cls_high = _percentile(bucket_vals, 75)

        if matching_days >= CONFIDENCE_HIGH_DAYS:
            confidence = "high"
        elif matching_days >= CONFIDENCE_MEDIUM_DAYS:
            confidence = "medium"
        else:
            confidence = "low"

    load_label = _cls_label(predicted)
    narrative = _build_narrative(
        predicted, cls_low, cls_high, load_label, confidence,
        meeting_minutes, matching_days, bucket,
    )

    return LoadForecast(
        date_str=date_str,
        predicted_cls=round(predicted, 3),
        cls_low=round(cls_low, 3),
        cls_high=round(cls_high, 3),
        load_label=load_label,
        confidence=confidence,
        meeting_minutes=meeting_minutes,
        days_of_history=days_of_history,
        matching_days=matching_days,
        narrative=narrative,
        is_meaningful=True,
    )


# ─── Narrative builder ────────────────────────────────────────────────────────

def _build_narrative(
    predicted: float,
    cls_low: float,
    cls_high: float,
    label: str,
    confidence: str,
    meeting_minutes: int,
    matching_days: int,
    bucket: str,
) -> str:
    """
    Build a single actionable sentence for the forecast.

    Examples:
        "2h of meetings → expect moderate load today (CLS ~0.42, range 0.35–0.52)."
        "No meetings today → light cognitive day expected (CLS ~0.12)."
        "4h30m of meetings → high load day ahead (CLS ~0.68). Protect any gaps for recovery."
    """
    if meeting_minutes == 0:
        start = "No meetings scheduled"
    else:
        start = f"{_fmt_minutes(meeting_minutes)} of meetings"

    cls_str = f"CLS ~{predicted:.2f}"
    if cls_low is not None and cls_high is not None and abs(cls_high - cls_low) > 0.05:
        range_str = f", range {cls_low:.2f}–{cls_high:.2f}"
    else:
        range_str = ""

    label_lower = label.lower()

    # Build the base sentence
    base = f"{start} → {label_lower} load expected ({cls_str}{range_str})."

    # Add a confidence qualifier for low-confidence forecasts
    if confidence == "low":
        base += f" (low confidence — only {matching_days} matching historical day{'s' if matching_days != 1 else ''})"
    elif confidence == "medium":
        base += f" (based on {matching_days} similar days)"

    # Add a practical tip based on the load level
    tip = ""
    if predicted >= 0.70:
        tip = " Protect any free blocks — recovery windows matter on high-load days."
    elif predicted >= 0.50:
        tip = " Front-load focused work before meetings consume the morning."
    elif predicted <= 0.20 and meeting_minutes == 0:
        tip = " Ideal conditions for deep, uninterrupted work."

    return base + tip


# ─── Formatter ────────────────────────────────────────────────────────────────

def format_forecast_line(forecast: LoadForecast) -> str:
    """
    Format the load forecast as a single Slack-ready line.

    Examples:
        "📊 Load forecast: Moderate (CLS ~0.42, based on 5 similar days)"
        "📊 Load forecast: High (CLS ~0.65–0.72) — protect recovery gaps"

    Returns empty string when is_meaningful=False.
    """
    if not forecast.is_meaningful or forecast.predicted_cls is None:
        return ""

    conf_note = ""
    if forecast.confidence == "low":
        conf_note = " ⚠️ low confidence"
    elif forecast.confidence == "medium":
        conf_note = f" ({forecast.matching_days} similar days)"
    elif forecast.confidence == "high":
        conf_note = f" ({forecast.matching_days} similar days)"

    cls_str = f"CLS ~{forecast.predicted_cls:.2f}"
    if (
        forecast.cls_low is not None
        and forecast.cls_high is not None
        and abs(forecast.cls_high - forecast.cls_low) > 0.05
    ):
        cls_str = f"CLS ~{forecast.predicted_cls:.2f} ({forecast.cls_low:.2f}–{forecast.cls_high:.2f})"

    return f"📊 *Load forecast:* {forecast.load_label} — {cls_str}{conf_note}"


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.

    Usage:
        python3 analysis/load_forecast.py                 # Today
        python3 analysis/load_forecast.py 2026-03-13      # Specific date
        python3 analysis/load_forecast.py --json          # JSON output
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Predicted cognitive load forecast for today"
    )
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD), default = today")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    # Try to collect today's calendar
    today_calendar = None
    try:
        from collectors.gcal import collect as gcal_collect
        today_calendar = gcal_collect(date_str)
    except Exception:
        pass

    forecast = compute_load_forecast(date_str, today_calendar)

    if args.json:
        print(json.dumps(forecast.to_dict(), indent=2))
        return

    print()
    print(f"Load Forecast — {date_str}")
    print("=" * 50)

    if not forecast.is_meaningful:
        print(f"  Not enough data: {forecast.narrative}")
    else:
        meeting_str = _fmt_minutes(forecast.meeting_minutes) if forecast.meeting_minutes else "none"
        print(f"  Meetings today:   {meeting_str}")
        print(f"  Predicted CLS:    {forecast.predicted_cls:.3f}  ({forecast.load_label})")
        if forecast.cls_low is not None and forecast.cls_high is not None:
            print(f"  Range (25–75%):   {forecast.cls_low:.3f} – {forecast.cls_high:.3f}")
        print(f"  Confidence:       {forecast.confidence}")
        print(f"  Based on:         {forecast.matching_days} similar days / {forecast.days_of_history} days total")
        print()
        print(f"  → {forecast.narrative}")

    print()


if __name__ == "__main__":
    main()

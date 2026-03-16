"""
Presence Tracker — Cognitive Rhythm Analysis (v17)

Answers: *"When in the day — and week — does David actually do his best thinking?"*

Every person has a cognitive rhythm: predictable patterns of mental energy that
repeat across hours and days. This module uncovers David's personal rhythm from
historical JSONL data and surfaces it as actionable scheduling intelligence.

## What it computes

### Hourly profile (across all tracked days)
- avg_cls[hour]   — mean cognitive load per hour (lower = lighter)
- avg_fdi[hour]   — mean focus depth per hour (higher = better focus)
- window_count[hour] — number of windows in the sample (confidence indicator)

### Day-of-week profile (across all tracked weeks)
- avg_cls[dow]    — mean CLS per weekday (Mon=0 … Sun=6)
- avg_fdi[dow]    — mean FDI per weekday
- avg_recovery[dow] — mean WHOOP recovery per weekday
- avg_meetings[dow] — mean total meeting minutes per weekday

### Rhythm insights (deterministic, rule-based)
- peak_focus_hours:  top-3 hours by historical FDI (best hours for deep work)
- low_load_hours:    top-3 hours by lowest average CLS (mentally lightest)
- best_dow:          weekday with best FDI (ideal day for deep projects)
- worst_dow:         weekday with most meetings or highest CLS (most demanding)
- morning_vs_afternoon: whether David is historically stronger AM or PM

### Sparkline renderers (Slack-ready ASCII)
- hourly_fdi_sparkline:   12-char Unicode block chart, 8am–8pm
- hourly_cls_sparkline:   12-char Unicode block chart, 8am–8pm
- dow_fdi_sparkline:      7-char sparkline, Mon–Sun

## Output

    CognitiveRhythm dataclass
      → to_dict()      (JSON serialisable)
      → is_meaningful  (False when < MIN_DAYS_FOR_RHYTHM)

## API

    from analysis.cognitive_rhythm import compute_cognitive_rhythm, format_rhythm_section

    rhythm = compute_cognitive_rhythm(as_of_date_str)
    section = format_rhythm_section(rhythm)   # Slack markdown section
    line    = format_rhythm_line(rhythm)       # compact one-liner

## CLI

    python3 analysis/cognitive_rhythm.py            # Full rhythm report
    python3 analysis/cognitive_rhythm.py --json     # Machine-readable JSON
    python3 analysis/cognitive_rhythm.py 2026-03-14 # As of specific date

## Design

- Pure functions — fully testable, no live API calls
- Only reads from the local JSONL store (engine.store)
- Graceful degradation: returns is_meaningful=False with < MIN_DAYS_FOR_RHYTHM days
- No ML, no external dependencies — deterministic pattern extraction
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import list_available_dates, read_day, read_summary


# ─── Constants ────────────────────────────────────────────────────────────────

# Minimum days before rhythm analysis is considered meaningful
MIN_DAYS_FOR_RHYTHM = 3

# Working hours window for hourly profile
WORK_START_HOUR = 7
WORK_END_HOUR = 21

# Minimum windows in an hour to include in the profile (reliability threshold)
MIN_WINDOWS_PER_HOUR = 2

# Minimum windows per day-of-week to include in the profile
MIN_WINDOWS_PER_DOW = 4

# Max days of history to use (keep it to the recent 60 for relevance)
RHYTHM_HISTORY_DAYS = 60

# Unicode block characters for sparklines (8 levels, ascending)
SPARK_CHARS = " ▁▂▃▄▅▆▇█"

# Day-of-week labels
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DOW_LABELS_SHORT = ["M", "T", "W", "T", "F", "S", "S"]


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class HourlyProfile:
    """Per-hour cognitive averages across all tracked days."""
    hour: int
    avg_cls: Optional[float]           # mean cognitive load (0–1, lower = lighter)
    avg_fdi: Optional[float]           # mean focus depth (0–1, higher = better)
    window_count: int                  # number of data windows (confidence)


@dataclass
class DowProfile:
    """Per-day-of-week cognitive averages."""
    dow: int                           # 0=Monday … 6=Sunday
    label: str                         # "Mon" … "Sun"
    avg_cls: Optional[float]
    avg_fdi: Optional[float]
    avg_recovery: Optional[float]      # WHOOP recovery %
    avg_meetings: Optional[float]      # mean total meeting minutes
    day_count: int                     # number of days included


@dataclass
class CognitiveRhythm:
    """Full cognitive rhythm profile for a tracked period."""

    # Hourly profiles (list, indexed 0–23 but only working hours populated)
    hourly: list[HourlyProfile] = field(default_factory=list)

    # Day-of-week profiles (Mon–Sun)
    dow: list[DowProfile] = field(default_factory=list)

    # Top-level insights
    peak_focus_hours: list[int] = field(default_factory=list)    # best hours for deep work
    low_load_hours: list[int] = field(default_factory=list)       # lowest cognitive cost hours
    best_focus_dow: Optional[int] = None                          # 0=Mon … 6=Sun
    heaviest_dow: Optional[int] = None                            # most meeting-heavy / CLS
    morning_bias: Optional[str] = None                            # "morning" | "afternoon" | "balanced"

    # Sparklines (Slack-ready ASCII)
    hourly_fdi_sparkline: str = ""                                # 8am–8pm FDI bar
    hourly_cls_sparkline: str = ""                                # 8am–8pm CLS bar (inverted)
    dow_fdi_sparkline: str = ""                                   # Mon–Sun FDI sparkline

    # Meta
    days_analyzed: int = 0
    date_range: str = ""                                          # "2026-03-01 → 2026-03-14"
    is_meaningful: bool = False

    def to_dict(self) -> dict:
        return {
            "hourly": [
                {
                    "hour": h.hour,
                    "avg_cls": round(h.avg_cls, 3) if h.avg_cls is not None else None,
                    "avg_fdi": round(h.avg_fdi, 3) if h.avg_fdi is not None else None,
                    "window_count": h.window_count,
                }
                for h in self.hourly
            ],
            "dow": [
                {
                    "dow": d.dow,
                    "label": d.label,
                    "avg_cls": round(d.avg_cls, 3) if d.avg_cls is not None else None,
                    "avg_fdi": round(d.avg_fdi, 3) if d.avg_fdi is not None else None,
                    "avg_recovery": round(d.avg_recovery, 1) if d.avg_recovery is not None else None,
                    "avg_meetings": round(d.avg_meetings, 0) if d.avg_meetings is not None else None,
                    "day_count": d.day_count,
                }
                for d in self.dow
            ],
            "peak_focus_hours": self.peak_focus_hours,
            "low_load_hours": self.low_load_hours,
            "best_focus_dow": self.best_focus_dow,
            "heaviest_dow": self.heaviest_dow,
            "morning_bias": self.morning_bias,
            "hourly_fdi_sparkline": self.hourly_fdi_sparkline,
            "hourly_cls_sparkline": self.hourly_cls_sparkline,
            "dow_fdi_sparkline": self.dow_fdi_sparkline,
            "days_analyzed": self.days_analyzed,
            "date_range": self.date_range,
            "is_meaningful": self.is_meaningful,
        }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _mean(vals: list[float]) -> Optional[float]:
    """Mean of a non-empty list; None for empty."""
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def _spark_bar(value: float, min_val: float, max_val: float) -> str:
    """Map a value in [min_val, max_val] to a single Unicode block character."""
    if max_val <= min_val:
        return SPARK_CHARS[4]  # middle block
    t = (value - min_val) / (max_val - min_val)
    t = max(0.0, min(1.0, t))
    idx = int(t * (len(SPARK_CHARS) - 1))
    return SPARK_CHARS[idx]


def _build_sparkline(
    values: list[Optional[float]],
    invert: bool = False,
) -> str:
    """
    Build a sparkline string from a list of values.

    Parameters
    ----------
    values : list of float | None
        Ordered list of values (None = missing data, rendered as '·').
    invert : bool
        If True, high values map to low blocks (useful for CLS where lower = better).
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return "·" * len(values)
    min_v = min(clean)
    max_v = max(clean)
    result = []
    for v in values:
        if v is None:
            result.append("·")
        else:
            effective = (max_v - v + min_v) if invert else v
            result.append(_spark_bar(effective, min_v, max_v))
    return "".join(result)


# ─── Core builders ────────────────────────────────────────────────────────────

def _build_hourly_profile(dates: list[str]) -> list[HourlyProfile]:
    """
    Aggregate per-window data across all dates into an hourly profile.

    Returns a list of HourlyProfile objects for WORK_START_HOUR–WORK_END_HOUR.
    Hours with fewer than MIN_WINDOWS_PER_HOUR windows are included but flagged
    (window_count will be low — callers can filter these out).
    """
    # Collect raw values per hour
    hour_cls: dict[int, list[float]] = {h: [] for h in range(WORK_START_HOUR, WORK_END_HOUR + 1)}
    hour_fdi: dict[int, list[float]] = {h: [] for h in range(WORK_START_HOUR, WORK_END_HOUR + 1)}

    for date_str in dates:
        try:
            windows = read_day(date_str)
        except Exception:
            continue

        for w in windows:
            meta = w.get("metadata", {})
            h = meta.get("hour_of_day")
            if h is None or not (WORK_START_HOUR <= h <= WORK_END_HOUR):
                continue

            # Only include windows during working hours
            if not meta.get("is_working_hours", False):
                continue

            metrics = w.get("metrics", {})
            cls_val = metrics.get("cognitive_load_score")
            fdi_val = metrics.get("focus_depth_index")

            if cls_val is not None:
                hour_cls[h].append(cls_val)
            if fdi_val is not None:
                hour_fdi[h].append(fdi_val)

    profiles = []
    for h in range(WORK_START_HOUR, WORK_END_HOUR + 1):
        cls_list = hour_cls[h]
        fdi_list = hour_fdi[h]
        count = max(len(cls_list), len(fdi_list))
        profiles.append(HourlyProfile(
            hour=h,
            avg_cls=_mean(cls_list) if cls_list else None,
            avg_fdi=_mean(fdi_list) if fdi_list else None,
            window_count=count,
        ))

    return profiles


def _build_dow_profile(dates: list[str], summary: dict) -> list[DowProfile]:
    """
    Compute per-day-of-week averages across all tracked days.

    Uses the rolling summary for speed (avoids re-reading all JSONL windows
    for per-day aggregates).

    Returns a list of 7 DowProfile objects (Mon–Sun, dow 0–6).
    """
    all_days = summary.get("days", {})

    # Collect raw values per DOW
    dow_cls: dict[int, list[float]] = {i: [] for i in range(7)}
    dow_fdi: dict[int, list[float]] = {i: [] for i in range(7)}
    dow_recovery: dict[int, list[float]] = {i: [] for i in range(7)}
    dow_meetings: dict[int, list[float]] = {i: [] for i in range(7)}
    dow_count: dict[int, int] = {i: 0 for i in range(7)}

    for date_str in dates:
        # Determine day-of-week
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            dow = dt.weekday()  # 0=Monday … 6=Sunday
        except ValueError:
            continue

        day_data = all_days.get(date_str, {})

        avg_cls = (
            day_data.get("metrics_avg", {}).get("cognitive_load_score")
            if day_data.get("metrics_avg")
            else None
        )
        avg_fdi_active = (
            day_data.get("metrics_avg", {}).get("focus_depth_index")
            if day_data.get("metrics_avg")
            else None
        )
        recovery = (
            day_data.get("whoop", {}).get("recovery_score")
            if day_data.get("whoop")
            else None
        )
        meetings = (
            day_data.get("calendar", {}).get("total_meeting_minutes", 0)
            if day_data.get("calendar")
            else 0
        )

        if avg_cls is not None:
            dow_cls[dow].append(avg_cls)
        if avg_fdi_active is not None:
            dow_fdi[dow].append(avg_fdi_active)
        if recovery is not None:
            dow_recovery[dow].append(recovery)
        if meetings is not None:
            dow_meetings[dow].append(meetings)
        dow_count[dow] += 1

    profiles = []
    for i in range(7):
        profiles.append(DowProfile(
            dow=i,
            label=DOW_LABELS[i],
            avg_cls=_mean(dow_cls[i]) if dow_cls[i] else None,
            avg_fdi=_mean(dow_fdi[i]) if dow_fdi[i] else None,
            avg_recovery=_mean(dow_recovery[i]) if dow_recovery[i] else None,
            avg_meetings=_mean(dow_meetings[i]) if dow_meetings[i] else None,
            day_count=dow_count[i],
        ))

    return profiles


def _extract_peak_focus_hours(
    hourly: list[HourlyProfile],
    n: int = 3,
) -> list[int]:
    """
    Return the top-n hours by avg_fdi within working hours, with sufficient data.
    """
    eligible = [
        h for h in hourly
        if h.avg_fdi is not None and h.window_count >= MIN_WINDOWS_PER_HOUR
        and WORK_START_HOUR <= h.hour < WORK_END_HOUR
    ]
    eligible.sort(key=lambda h: h.avg_fdi, reverse=True)  # type: ignore[arg-type]
    return [h.hour for h in eligible[:n]]


def _extract_low_load_hours(
    hourly: list[HourlyProfile],
    n: int = 3,
) -> list[int]:
    """
    Return the top-n hours with lowest avg_cls (lightest cognitive load).
    """
    eligible = [
        h for h in hourly
        if h.avg_cls is not None and h.window_count >= MIN_WINDOWS_PER_HOUR
        and WORK_START_HOUR <= h.hour < WORK_END_HOUR
    ]
    eligible.sort(key=lambda h: h.avg_cls)  # type: ignore[arg-type]
    return [h.hour for h in eligible[:n]]


def _morning_vs_afternoon(hourly: list[HourlyProfile]) -> Optional[str]:
    """
    Compare mean FDI for morning (8–12) vs afternoon (13–17).

    Returns "morning" | "afternoon" | "balanced".
    Returns None when insufficient data.
    """
    morning_fdi = [
        h.avg_fdi for h in hourly
        if h.avg_fdi is not None and 8 <= h.hour < 13
        and h.window_count >= MIN_WINDOWS_PER_HOUR
    ]
    afternoon_fdi = [
        h.avg_fdi for h in hourly
        if h.avg_fdi is not None and 13 <= h.hour < 18
        and h.window_count >= MIN_WINDOWS_PER_HOUR
    ]

    am_mean = _mean(morning_fdi)
    pm_mean = _mean(afternoon_fdi)

    if am_mean is None or pm_mean is None:
        return None

    diff = am_mean - pm_mean
    if diff > 0.05:
        return "morning"
    elif diff < -0.05:
        return "afternoon"
    else:
        return "balanced"


def _hourly_sparkline(
    hourly: list[HourlyProfile],
    metric: str,     # "fdi" or "cls"
    start_h: int = 8,
    end_h: int = 20,  # exclusive
) -> str:
    """
    Build a sparkline for the given metric across the working day.

    Parameters
    ----------
    hourly : list[HourlyProfile]
    metric : "fdi" (higher = better) or "cls" (lower = better, so we invert)
    start_h, end_h : hour range for the sparkline
    """
    hour_map = {h.hour: h for h in hourly}
    values: list[Optional[float]] = []

    for h in range(start_h, end_h):
        prof = hour_map.get(h)
        if prof is None or (metric == "fdi" and prof.avg_fdi is None):
            values.append(None)
        elif metric == "fdi":
            values.append(prof.avg_fdi)
        elif prof.avg_cls is None:
            values.append(None)
        else:
            values.append(prof.avg_cls)

    return _build_sparkline(values, invert=(metric == "cls"))


def _dow_sparkline(dow_profiles: list[DowProfile], metric: str = "fdi") -> str:
    """Build a Mon–Sun FDI or CLS sparkline."""
    values: list[Optional[float]] = []
    for d in dow_profiles:  # already in Mon–Sun order
        if metric == "fdi":
            values.append(d.avg_fdi if d.day_count > 0 else None)
        else:
            values.append(d.avg_cls if d.day_count > 0 else None)
    return _build_sparkline(values, invert=(metric == "cls"))


# ─── Main function ────────────────────────────────────────────────────────────

def compute_cognitive_rhythm(
    as_of_date_str: Optional[str] = None,
    days: int = RHYTHM_HISTORY_DAYS,
) -> CognitiveRhythm:
    """
    Compute the cognitive rhythm profile from all available historical data.

    Parameters
    ----------
    as_of_date_str : str | None
        Upper bound date (YYYY-MM-DD). Defaults to today. Data after this
        date is excluded.
    days : int
        Maximum number of historical days to include.

    Returns
    -------
    CognitiveRhythm
        is_meaningful=False when fewer than MIN_DAYS_FOR_RHYTHM days are available.
    """
    today_str = as_of_date_str or datetime.now().strftime("%Y-%m-%d")

    # Load available dates up to as_of_date_str
    all_dates = sorted(list_available_dates())
    dates = sorted(
        [d for d in all_dates if d <= today_str],
        reverse=True,
    )[:days]
    dates = sorted(dates)  # chronological for profile building

    n_days = len(dates)
    date_range = f"{dates[0]} → {dates[-1]}" if dates else ""

    if n_days < MIN_DAYS_FOR_RHYTHM:
        return CognitiveRhythm(
            days_analyzed=n_days,
            date_range=date_range,
            is_meaningful=False,
        )

    # Load rolling summary for day-level aggregates
    try:
        summary = read_summary()
    except Exception:
        summary = {}

    # Build profiles
    hourly = _build_hourly_profile(dates)
    dow = _build_dow_profile(dates, summary)

    # Derive insights
    peak_focus_hours = _extract_peak_focus_hours(hourly)
    low_load_hours = _extract_low_load_hours(hourly)
    morning_bias = _morning_vs_afternoon(hourly)

    # Best FDI day-of-week (only from days with enough data)
    eligible_dow = [d for d in dow if d.avg_fdi is not None and d.day_count >= 1]
    best_focus_dow = max(eligible_dow, key=lambda d: d.avg_fdi, default=None)  # type: ignore[arg-type]
    best_focus_dow_idx = best_focus_dow.dow if best_focus_dow else None

    # Heaviest day-of-week by meeting minutes (fallback to avg_cls)
    heaviest_dow_obj = max(
        [d for d in dow if d.avg_meetings is not None and d.day_count >= 1],
        key=lambda d: d.avg_meetings,  # type: ignore[arg-type]
        default=None,
    )
    heaviest_dow_idx = heaviest_dow_obj.dow if heaviest_dow_obj else None

    # Build sparklines
    hourly_fdi_spark = _hourly_sparkline(hourly, "fdi", start_h=8, end_h=21)
    hourly_cls_spark = _hourly_sparkline(hourly, "cls", start_h=8, end_h=21)
    dow_fdi_spark = _dow_sparkline(dow, "fdi")

    return CognitiveRhythm(
        hourly=hourly,
        dow=dow,
        peak_focus_hours=peak_focus_hours,
        low_load_hours=low_load_hours,
        best_focus_dow=best_focus_dow_idx,
        heaviest_dow=heaviest_dow_idx,
        morning_bias=morning_bias,
        hourly_fdi_sparkline=hourly_fdi_spark,
        hourly_cls_sparkline=hourly_cls_spark,
        dow_fdi_sparkline=dow_fdi_spark,
        days_analyzed=n_days,
        date_range=date_range,
        is_meaningful=True,
    )


# ─── Formatters ───────────────────────────────────────────────────────────────

def _hour_label(h: int) -> str:
    """Format an hour (0–23) as e.g. '9am', '2pm'."""
    if h == 0:
        return "12am"
    if h < 12:
        return f"{h}am"
    if h == 12:
        return "12pm"
    return f"{h - 12}pm"


def format_rhythm_line(rhythm: CognitiveRhythm) -> str:
    """
    One-line Slack-ready summary of the cognitive rhythm.

    Example:
        "⏱ *Rhythm:* Peak focus 9am–11am · Lightest load 2pm · Morning-biased"
    """
    if not rhythm.is_meaningful:
        return ""

    parts = []

    if rhythm.peak_focus_hours:
        hours_str = "–".join(_hour_label(h) for h in rhythm.peak_focus_hours[:2])
        parts.append(f"Peak focus {hours_str}")

    if rhythm.morning_bias:
        bias_map = {
            "morning": "morning-biased",
            "afternoon": "afternoon-biased",
            "balanced": "balanced AM/PM",
        }
        parts.append(bias_map.get(rhythm.morning_bias, ""))

    if rhythm.best_focus_dow is not None:
        parts.append(f"Best day: {DOW_LABELS[rhythm.best_focus_dow]}")

    body = " · ".join(p for p in parts if p)
    if not body:
        return ""

    return f"⏱ *Rhythm:* {body}"


def format_rhythm_section(rhythm: CognitiveRhythm, compact: bool = False) -> str:
    """
    Multi-line Slack/terminal section showing the full cognitive rhythm.

    Example (compact=False):
        ⏱ *Cognitive Rhythm* _(14 days)_

        *Hourly FDI* (8am → 8pm):
        ▂▄▇█▇▆▅▄▃▃▄▅▆▆▄▄▃▃▂▂▂▂▂▂
        8  9  10 11 12 1  2  3  4  5  6  7  8pm

        Peak focus: 9am, 10am, 11am
        Lightest load: 2pm, 3pm, 4pm

        *Day-of-Week*  M  T  W  T  F  S  S
        FDI:           ▅  ▇  █  ▇  ▆  ▁  ▁
        Best focus day: Wednesday · Heaviest day: Thursday
    """
    if not rhythm.is_meaningful:
        return ""

    lines = []

    # Header
    lines.append(f"⏱ *Cognitive Rhythm* _(based on {rhythm.days_analyzed} days)_")
    lines.append("")

    # Hourly FDI chart
    hour_labels = "  ".join(_hour_label(h)[:-2] if len(_hour_label(h)) > 2 else _hour_label(h) for h in range(8, 21))
    lines.append("*Hourly Focus Depth* (8am → 8pm):")
    lines.append(f"`{rhythm.hourly_fdi_sparkline}`")

    if not compact:
        lines.append(f"`8 9 10 11 12 1  2  3  4  5  6  7  8pm`")

    # Peak focus hours
    if rhythm.peak_focus_hours:
        peak_str = ", ".join(_hour_label(h) for h in rhythm.peak_focus_hours)
        lines.append(f"Peak focus hours: *{peak_str}*")

    # Low-load hours
    if rhythm.low_load_hours and not compact:
        low_str = ", ".join(_hour_label(h) for h in rhythm.low_load_hours)
        lines.append(f"Lightest cognitive load: {low_str}")

    # Morning/afternoon bias
    if rhythm.morning_bias:
        bias_text = {
            "morning": "You're a *morning thinker* — front-load your hard work.",
            "afternoon": "You're an *afternoon thinker* — protect post-lunch blocks.",
            "balanced": "You're evenly balanced across morning and afternoon.",
        }.get(rhythm.morning_bias, "")
        if bias_text:
            lines.append(f"→ {bias_text}")

    if not compact:
        lines.append("")

        # Day-of-week profile
        dow_data = rhythm.dow
        lines.append("*Day-of-Week Profile:*")

        # FDI row
        fdi_vals = []
        for d in dow_data:
            if d.avg_fdi is not None and d.day_count > 0:
                fdi_vals.append(f"{d.avg_fdi:.0%}")
            else:
                fdi_vals.append("  —  ")

        header = "  ".join(DOW_LABELS)
        lines.append(f"         {header}")
        lines.append(f"FDI:     {'  '.join(fdi_vals)}")
        lines.append(f"Rhythm:  `{rhythm.dow_fdi_sparkline}`")

        # Meeting load per day
        mtg_vals = []
        for d in dow_data:
            if d.avg_meetings is not None and d.day_count > 0:
                mins = int(d.avg_meetings)
                if mins >= 60:
                    mtg_vals.append(f" {mins//60}h{mins%60:02d}m" if mins % 60 else f"  {mins//60}h   ")
                else:
                    mtg_vals.append(f" {mins}m  ")
            else:
                mtg_vals.append("  —  ")

        lines.append(f"Mtgs:    {'  '.join(mtg_vals)}")

        # Best / heaviest day callouts
        callouts = []
        if rhythm.best_focus_dow is not None:
            callouts.append(f"Best focus day: *{DOW_LABELS[rhythm.best_focus_dow]}*")
        if rhythm.heaviest_dow is not None:
            callouts.append(f"Most meeting-heavy: *{DOW_LABELS[rhythm.heaviest_dow]}*")
        if callouts:
            lines.append("→ " + " · ".join(callouts))

    return "\n".join(lines)


def format_rhythm_terminal(rhythm: CognitiveRhythm) -> str:
    """
    Terminal-formatted (ANSI) cognitive rhythm report.
    Used by scripts/report.py --rhythm.
    """
    if not rhythm.is_meaningful:
        min_days = MIN_DAYS_FOR_RHYTHM
        n = rhythm.days_analyzed
        return (
            f"\n  Cognitive Rhythm\n"
            f"  Not enough data yet ({n} day{'s' if n != 1 else ''} collected, "
            f"need {min_days}).\n"
        )

    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    CYAN  = "\033[96m"
    DIM   = "\033[2m"
    RESET = "\033[0m"

    lines = [
        "",
        f"{BOLD}Cognitive Rhythm{RESET}  {DIM}({rhythm.days_analyzed} days · {rhythm.date_range}){RESET}",
        "=" * 60,
    ]

    # Hourly FDI heatmap
    lines.append(f"\n{BOLD}Hourly Focus Depth — 8am to 8pm{RESET}")
    lines.append(f"  {CYAN}{rhythm.hourly_fdi_sparkline}{RESET}")
    lines.append(f"  {DIM}8 9 10 11 12  1  2  3  4  5  6  7  8pm{RESET}")

    # Hourly CLS heatmap (inverted = lighter = bigger block)
    lines.append(f"\n{BOLD}Hourly Cognitive Load — 8am to 8pm{RESET}  {DIM}(inverted: bigger = lighter){RESET}")
    lines.append(f"  {GREEN}{rhythm.hourly_cls_sparkline}{RESET}")
    lines.append(f"  {DIM}8 9 10 11 12  1  2  3  4  5  6  7  8pm{RESET}")

    # Insights
    if rhythm.peak_focus_hours:
        peak_str = ", ".join(_hour_label(h) for h in rhythm.peak_focus_hours)
        lines.append(f"\n  Peak focus hours:   {BOLD}{peak_str}{RESET}")
    if rhythm.low_load_hours:
        low_str = ", ".join(_hour_label(h) for h in rhythm.low_load_hours)
        lines.append(f"  Lightest load:      {low_str}")

    if rhythm.morning_bias:
        bias_text = {
            "morning": f"{GREEN}Morning thinker{RESET} — front-load your hardest work",
            "afternoon": f"{CYAN}Afternoon thinker{RESET} — protect post-lunch blocks",
            "balanced": "Balanced AM/PM — flexible scheduling",
        }.get(rhythm.morning_bias, "")
        lines.append(f"  Focus pattern:      {bias_text}")

    # Day-of-week table
    lines.append(f"\n{BOLD}Day-of-Week Profile{RESET}")
    header_row = "  " + "   ".join(DOW_LABELS)
    lines.append(header_row)

    fdi_row_vals = []
    for d in rhythm.dow:
        if d.avg_fdi is not None and d.day_count > 0:
            fdi_row_vals.append(f"{d.avg_fdi:.0%}".rjust(4))
        else:
            fdi_row_vals.append("  — ")
    lines.append(f"FDI  {'  '.join(fdi_row_vals)}")

    mtg_row_vals = []
    for d in rhythm.dow:
        if d.avg_meetings is not None and d.day_count > 0:
            mins = int(d.avg_meetings)
            s = f"{mins}m" if mins < 60 else f"{mins//60}h"
            mtg_row_vals.append(s.rjust(4))
        else:
            mtg_row_vals.append("  — ")
    lines.append(f"Mtg  {'  '.join(mtg_row_vals)}")

    lines.append(f"     {DIM}{rhythm.dow_fdi_sparkline}   (Mon–Sun FDI){RESET}")

    if rhythm.best_focus_dow is not None:
        lines.append(f"\n  Best focus day:     {BOLD}{DOW_LABELS[rhythm.best_focus_dow]}{RESET}")
    if rhythm.heaviest_dow is not None:
        lines.append(f"  Most meeting-heavy: {DOW_LABELS[rhythm.heaviest_dow]}")

    lines.append("")
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.

    Usage:
        python3 analysis/cognitive_rhythm.py             # Full rhythm report
        python3 analysis/cognitive_rhythm.py --json      # JSON output
        python3 analysis/cognitive_rhythm.py 2026-03-14  # As of specific date
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Cognitive Rhythm — hourly and day-of-week focus pattern analysis"
    )
    parser.add_argument("date", nargs="?", help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    rhythm = compute_cognitive_rhythm(date_str)

    if args.json:
        print(json.dumps(rhythm.to_dict(), indent=2))
        return

    print(format_rhythm_terminal(rhythm))


if __name__ == "__main__":
    main()

"""
Presence Tracker — Actionable Insights Generator (v42)

Answers: *"Based on my data, what are the top 3 things I should do differently?"*

## The gap this fills

Every other module in the system is **descriptive** or **predictive**:
  - "Your CLS was 0.62 yesterday." (descriptive)
  - "Tomorrow will be a high-load day." (predictive)
  - "Your BRI is rising." (trend-based warning)

None of them answer: *"What specific behavioural change would most improve my
cognitive performance, based on my actual data?"*

The Actionable Insights Generator closes this gap. It analyses 7–30 days of
JSONL history and surfaces the **3 highest-impact, evidence-backed behavioural
recommendations** — completely deterministic, no LLM required.

## How it works

The module runs 6 independent insight detectors, each searching for a specific
behavioural pattern with a known cognitive cost. Each detector produces either
an insight (with supporting evidence) or None.

The 6 detectors (in priority order):

### Detector 1: Post-Meeting Recovery Gap
Measures: does David's FDI recover after back-to-back meetings?

For each day, find pairs of adjacent meeting windows (w₁, w₂) where:
  w₁.in_meeting=True and w₂.in_meeting=True (consecutive or 1-window gap)
Then measure FDI in the 2 windows immediately after the meeting block ends.

If post-meeting FDI is systematically < pre-meeting FDI by > 0.15:
  → "Your FDI drops {delta:.0%} in the window after back-to-back meetings.
     A 15-min buffer between calls restores focus in your data."

### Detector 2: Late-Day Performance Cliff
Measures: when does David's actual cognitive performance tail off?

Per day, compute active-window FDI by hour. Find the hour where FDI first
drops permanently (remains lower for 2+ consecutive hours vs the daily peak).

If cliff hour < 17:00 and occurs on ≥ 3 of the last 7 days:
  → "Your focus quality drops sharply after {cliff_hour:02d}:00 on {days}/7 days.
     Protect mornings for deep work; use afternoons for admin and meetings."

### Detector 3: Slack Fragmentation Correlation
Measures: does high Slack volume predict lower FDI?

Bucket windows by messages_received: 0, 1–5, 6–15, >15.
Compute mean FDI per bucket. If FDI drops > 0.15 from the 0-message bucket
to the >15-message bucket:
  → "High Slack volume correlates with {delta:.0%} lower focus in your data.
     Slack batching (checking every 60–90min) would protect deep work windows."

### Detector 4: Meeting Load Threshold
Measures: at what total daily meeting load does David's CLS exceed sustainable?

Group days by total_meeting_minutes buckets: <60, 60–120, 120–180, >180.
Find the bucket where avg_cls first crosses 0.55 (high load).
Also find where avg_fdi drops below 0.55.

If threshold exists at < 180 min:
  → "Days with >{threshold_hours:.0f}h of meetings average CLS {cls:.2f} — your
     sustainable threshold is {threshold_hours-0.5:.1f}h. Cap daily meeting load there."

### Detector 5: Sleep-Recovery Leverage
Measures: which sleep inputs most strongly predict next-day FDI/recovery?

Compare next-day FDI for days where sleep_hours ≥ 7.5 vs < 7.5.
Compare next-day FDI for days where sleep_performance ≥ 80 vs < 80.
Surface the one with the largest delta.

If delta > 0.10 with ≥ 3 pairs:
  → "On nights with {predictor} ≥ {threshold}, your next-day FDI is {delta:.0%} higher.
     This is your highest-leverage sleep target."

### Detector 6: Cognitive Load Arc Optimisation
Measures: is David front- or back-loading his day, and does it matter?

Compute morning CLS (08–12) and afternoon CLS (13–18) for each day.
Compare next-day-equivalent FDI (that afternoon's FDI) for front-loaded vs
back-loaded days.

If front-loaded days show higher afternoon FDI by > 0.12 (or vice versa):
  → "Your front-loaded days (heavier morning) have {delta:.0%} higher afternoon
     focus than back-loaded days. Schedule hard meetings before noon."

---

## Ranking and deduplication

All detectors that return an insight are ranked by:
  1. Evidence strength (n_supporting_days — more days = stronger signal)
  2. Impact magnitude (delta size — bigger difference = more actionable)
  3. Detector priority (tie-break: lower detector number = higher priority)

Top 3 insights by score are returned.

## Output

    ActionableInsights dataclass:
      - insights: list[Insight]       — top 3 (or fewer with limited data)
      - is_meaningful: bool           — False when < MIN_DAYS
      - days_analysed: int
      - date_range: str               — "YYYY-MM-DD → YYYY-MM-DD"
      - generated_at: str             — ISO timestamp

    Insight dataclass:
      - rank: int                     — 1, 2, or 3
      - title: str                    — short label (e.g. "Post-Meeting Recovery Gap")
      - headline: str                 — one concrete action sentence
      - evidence: str                 — the data that supports this recommendation
      - impact_label: str             — "High" | "Medium" | "Low"
      - n_supporting_days: int        — how many days showed this pattern
      - magnitude: float              — size of the measured effect (0–1)
      - detector: str                 — which detector found this

## API

    from analysis.actionable_insights import (
        compute_actionable_insights,
        format_insights_section,
        format_insights_brief,
    )

    insights = compute_actionable_insights(as_of_date_str, days=14)
    if insights.is_meaningful:
        section = format_insights_section(insights)   # full Slack section
        brief = format_insights_brief(insights)       # compact morning-brief version

## CLI

    python3 analysis/actionable_insights.py                   # Last 14 days
    python3 analysis/actionable_insights.py --days 30         # Last 30 days
    python3 analysis/actionable_insights.py 2026-03-14        # As of date
    python3 analysis/actionable_insights.py --json            # JSON output

## Design principles

  - Fully deterministic: same data → same output every time
  - No LLM, no external APIs
  - Each recommendation is backed by a concrete delta (not a heuristic)
  - Degrades gracefully: fewer days → fewer insights, never crashes
  - Min 3 supporting data points required before surfacing an insight
  - Complementary to Alfred Intuition (deterministic here; narrative there)

"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import list_available_dates, read_day

# ─── Constants ────────────────────────────────────────────────────────────────

MIN_DAYS = 3          # need at least this many days to surface any insight
DEFAULT_LOOKBACK = 14 # default history window in days

# FDI drop threshold to flag post-meeting recovery gap
MEETING_FDI_DROP_THRESHOLD = 0.12

# Hours of the day where "late-day cliff" can be detected
CLIFF_SEARCH_START = 13
CLIFF_SEARCH_END = 19

# FDI cliff threshold: FDI must drop by this amount vs daily peak
CLIFF_FDI_DROP = 0.15

# Slack fragmentation threshold: FDI drop from 0-msg to high-msg bucket
SLACK_FDI_DROP_THRESHOLD = 0.12

# Meeting load threshold CLS crossing
MEETING_CLS_THRESHOLD = 0.55

# Sleep leverage minimum delta
SLEEP_DELTA_THRESHOLD = 0.08

# Load arc minimum delta
ARC_FDI_THRESHOLD = 0.10

# Minimum supporting data points per insight
MIN_SUPPORT_POINTS = 3


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class Insight:
    """A single actionable, evidence-backed recommendation."""
    rank: int
    title: str
    headline: str            # one concrete action sentence
    evidence: str            # data backing (e.g. "6/9 days showed this pattern")
    impact_label: str        # "High" | "Medium" | "Low"
    n_supporting_days: int
    magnitude: float         # effect size 0–1
    detector: str            # which detector produced this


@dataclass
class ActionableInsights:
    """Top-3 (or fewer) actionable insights derived from JSONL history."""
    insights: list[Insight] = field(default_factory=list)
    is_meaningful: bool = False
    days_analysed: int = 0
    date_range: str = ""
    generated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "is_meaningful": self.is_meaningful,
            "days_analysed": self.days_analysed,
            "date_range": self.date_range,
            "generated_at": self.generated_at,
            "insights": [
                {
                    "rank": i.rank,
                    "title": i.title,
                    "headline": i.headline,
                    "evidence": i.evidence,
                    "impact_label": i.impact_label,
                    "n_supporting_days": i.n_supporting_days,
                    "magnitude": round(i.magnitude, 3),
                    "detector": i.detector,
                }
                for i in self.insights
            ],
        }


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _safe_mean(vals: list[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _load_days_data(dates: list[str]) -> dict[str, list[dict]]:
    """Load JSONL windows for each date in the list."""
    return {d: read_day(d) for d in dates if read_day(d)}


def _active_working_windows(windows: list[dict]) -> list[dict]:
    """Return windows that are both active and in working hours."""
    return [
        w for w in windows
        if w.get("metadata", {}).get("is_active_window", False)
        and w.get("metadata", {}).get("is_working_hours", False)
    ]


def _fdi(w: dict) -> Optional[float]:
    return w.get("metrics", {}).get("focus_depth_index")


def _cls(w: dict) -> Optional[float]:
    return w.get("metrics", {}).get("cognitive_load_score")


def _in_meeting(w: dict) -> bool:
    return bool(w.get("calendar", {}).get("in_meeting", False))


def _attendees(w: dict) -> int:
    return w.get("calendar", {}).get("meeting_attendees", 0) or 0


def _messages_received(w: dict) -> int:
    return w.get("slack", {}).get("messages_received", 0) or 0


def _hour(w: dict) -> int:
    return w.get("metadata", {}).get("hour_of_day", 0)


def _sleep_hours(windows: list[dict]) -> Optional[float]:
    for w in windows:
        v = w.get("whoop", {}).get("sleep_hours")
        if v is not None:
            return float(v)
    return None


def _sleep_perf(windows: list[dict]) -> Optional[float]:
    for w in windows:
        v = w.get("whoop", {}).get("sleep_performance")
        if v is not None:
            return float(v)
    return None


def _recovery(windows: list[dict]) -> Optional[float]:
    for w in windows:
        v = w.get("whoop", {}).get("recovery_score")
        if v is not None:
            return float(v)
    return None


def _daily_avg_cls(windows: list[dict]) -> Optional[float]:
    active = _active_working_windows(windows)
    vals = [_cls(w) for w in active if _cls(w) is not None]
    return _safe_mean(vals)


def _daily_avg_fdi(windows: list[dict]) -> Optional[float]:
    active = _active_working_windows(windows)
    vals = [_fdi(w) for w in active if _fdi(w) is not None]
    return _safe_mean(vals)


def _total_meeting_minutes(windows: list[dict]) -> int:
    """Total calendar minutes in meetings (social meetings only)."""
    count = 0
    for w in windows:
        if _in_meeting(w) and _attendees(w) > 1:
            count += 15  # each window = 15 minutes
    return count


# ─── Detector 1: Post-Meeting Recovery Gap ────────────────────────────────────

def _detect_meeting_recovery_gap(
    days_data: dict[str, list[dict]],
) -> Optional[Insight]:
    """
    Detect whether FDI systematically drops after back-to-back social meetings.
    """
    pre_meeting_fdis: list[float] = []
    post_meeting_fdis: list[float] = []

    for date_str, windows in days_data.items():
        # Sort windows by index
        sorted_windows = sorted(windows, key=lambda w: w.get("window_index", 0))
        n = len(sorted_windows)

        for i, w in enumerate(sorted_windows):
            # Look for end of a meeting block (in_meeting → not in_meeting)
            if not (_in_meeting(w) and _attendees(w) > 1):
                continue
            # Check this is the LAST window of a meeting block
            if i + 1 < n and _in_meeting(sorted_windows[i + 1]) and _attendees(sorted_windows[i + 1]) > 1:
                continue  # meeting continues — not the end yet

            # Find the window just before the meeting block started
            block_start = i
            while block_start > 0 and _in_meeting(sorted_windows[block_start - 1]) and _attendees(sorted_windows[block_start - 1]) > 1:
                block_start -= 1

            # Pre-meeting FDI: window before the meeting block
            if block_start > 0:
                pre_w = sorted_windows[block_start - 1]
                pre_fdi = _fdi(pre_w)
                if pre_fdi is not None and _hour(pre_w) >= 8:
                    pre_meeting_fdis.append(pre_fdi)

            # Post-meeting FDI: up to 2 windows after the meeting ends
            post_window_fdis = []
            for j in range(i + 1, min(i + 3, n)):
                post_w = sorted_windows[j]
                pf = _fdi(post_w)
                if pf is not None and _hour(post_w) >= 8:
                    post_window_fdis.append(pf)
            if post_window_fdis:
                post_meeting_fdis.append(sum(post_window_fdis) / len(post_window_fdis))

    if len(pre_meeting_fdis) < MIN_SUPPORT_POINTS or len(post_meeting_fdis) < MIN_SUPPORT_POINTS:
        return None

    avg_pre = _safe_mean(pre_meeting_fdis)
    avg_post = _safe_mean(post_meeting_fdis)

    if avg_pre is None or avg_post is None:
        return None

    delta = avg_pre - avg_post
    if delta < MEETING_FDI_DROP_THRESHOLD:
        return None

    pct_drop = delta / max(avg_pre, 0.01)
    impact = "High" if pct_drop > 0.25 else ("Medium" if pct_drop > 0.15 else "Low")

    return Insight(
        rank=0,
        title="Post-Meeting Recovery Gap",
        headline=(
            f"Your focus drops {pct_drop:.0%} after meetings end. "
            f"Add a 15-min buffer between back-to-back calls to restore focus before the next task."
        ),
        evidence=(
            f"Pre-meeting FDI avg {avg_pre:.2f} → post-meeting {avg_post:.2f} "
            f"across {min(len(pre_meeting_fdis), len(post_meeting_fdis))} transitions"
        ),
        impact_label=impact,
        n_supporting_days=min(len(pre_meeting_fdis), len(post_meeting_fdis)),
        magnitude=pct_drop,
        detector="post_meeting_recovery_gap",
    )


# ─── Detector 2: Late-Day Performance Cliff ───────────────────────────────────

def _detect_late_day_cliff(
    days_data: dict[str, list[dict]],
) -> Optional[Insight]:
    """
    Find the hour where FDI first drops and stays low — the cognitive wall.
    """
    cliff_hours: list[int] = []
    daily_peak_fdis: list[float] = []

    for date_str, windows in days_data.items():
        # Build hourly average FDI for working-hour windows
        hour_fdis: dict[int, list[float]] = {}
        for w in windows:
            if not w.get("metadata", {}).get("is_working_hours", False):
                continue
            h = _hour(w)
            fv = _fdi(w)
            if fv is not None and 8 <= h < CLIFF_SEARCH_END:
                hour_fdis.setdefault(h, []).append(fv)

        if not hour_fdis:
            continue

        # Compute mean FDI per hour
        hourly_avg = {h: sum(vals) / len(vals) for h, vals in hour_fdis.items()}
        if not hourly_avg:
            continue

        peak_fdi = max(hourly_avg.values())
        daily_peak_fdis.append(peak_fdi)

        # Find first hour in CLIFF_SEARCH range where FDI drops > CLIFF_FDI_DROP vs peak
        # AND the drop is sustained (next hour also low)
        sorted_hours = sorted(h for h in hourly_avg if h >= CLIFF_SEARCH_START)
        for i, h in enumerate(sorted_hours):
            avg_fdi_here = hourly_avg[h]
            if peak_fdi - avg_fdi_here >= CLIFF_FDI_DROP:
                # Check if sustained (next 1+ hours also low)
                next_hours_low = all(
                    hourly_avg.get(sorted_hours[j], avg_fdi_here) <= avg_fdi_here + 0.05
                    for j in range(i + 1, min(i + 2, len(sorted_hours)))
                )
                if next_hours_low or i == len(sorted_hours) - 1:
                    cliff_hours.append(h)
                    break

    if len(cliff_hours) < MIN_SUPPORT_POINTS:
        return None

    # Compute the modal cliff hour
    from collections import Counter
    cliff_counter = Counter(cliff_hours)
    modal_cliff, modal_count = cliff_counter.most_common(1)[0]

    consistency_pct = modal_count / len(days_data)
    if consistency_pct < 0.35:  # less than 35% of days → not a reliable pattern
        return None

    avg_peak = _safe_mean(daily_peak_fdis) or 0.70
    impact = "High" if consistency_pct > 0.60 else ("Medium" if consistency_pct > 0.40 else "Low")

    return Insight(
        rank=0,
        title="Late-Day Performance Cliff",
        headline=(
            f"Your focus quality falls off after {modal_cliff:02d}:00 on most days. "
            f"Move your hardest cognitive work to the morning and use afternoons for lower-stakes tasks."
        ),
        evidence=(
            f"FDI drops {CLIFF_FDI_DROP:.0%}+ from daily peak after {modal_cliff:02d}:00 "
            f"on {modal_count}/{len(days_data)} analysed days"
        ),
        impact_label=impact,
        n_supporting_days=modal_count,
        magnitude=consistency_pct,
        detector="late_day_cliff",
    )


# ─── Detector 3: Slack Fragmentation Correlation ──────────────────────────────

def _detect_slack_fragmentation(
    days_data: dict[str, list[dict]],
) -> Optional[Insight]:
    """
    Measure FDI degradation as incoming Slack message volume increases.
    """
    buckets: dict[str, list[float]] = {
        "none":   [],   # 0 messages
        "low":    [],   # 1–4
        "medium": [],   # 5–14
        "high":   [],   # 15+
    }

    for date_str, windows in days_data.items():
        for w in windows:
            if not w.get("metadata", {}).get("is_working_hours", False):
                continue
            msgs = _messages_received(w)
            fv = _fdi(w)
            if fv is None:
                continue
            if msgs == 0:
                buckets["none"].append(fv)
            elif msgs <= 4:
                buckets["low"].append(fv)
            elif msgs <= 14:
                buckets["medium"].append(fv)
            else:
                buckets["high"].append(fv)

    # Need both quiet and high-volume buckets to have meaningful data
    quiet_fdis = buckets["none"]
    high_fdis = buckets["high"]

    if len(quiet_fdis) < MIN_SUPPORT_POINTS or len(high_fdis) < MIN_SUPPORT_POINTS:
        return None

    avg_quiet = _safe_mean(quiet_fdis)
    avg_high = _safe_mean(high_fdis)

    if avg_quiet is None or avg_high is None:
        return None

    delta = avg_quiet - avg_high
    if delta < SLACK_FDI_DROP_THRESHOLD:
        return None

    pct_drop = delta / max(avg_quiet, 0.01)
    impact = "High" if pct_drop > 0.25 else ("Medium" if pct_drop > 0.15 else "Low")

    return Insight(
        rank=0,
        title="Slack Fragmentation Impact",
        headline=(
            f"High Slack volume correlates with {pct_drop:.0%} lower focus depth. "
            f"Batch Slack to specific windows (e.g. 09:30, 12:00, 16:00) to protect deep work."
        ),
        evidence=(
            f"FDI in quiet windows (0 msgs): {avg_quiet:.2f} · "
            f"FDI in high-volume windows (15+ msgs): {avg_high:.2f} "
            f"(n={len(quiet_fdis)} quiet, {len(high_fdis)} high)"
        ),
        impact_label=impact,
        n_supporting_days=min(len(quiet_fdis), len(high_fdis)),
        magnitude=pct_drop,
        detector="slack_fragmentation",
    )


# ─── Detector 4: Meeting Load Threshold ───────────────────────────────────────

def _detect_meeting_load_threshold(
    days_data: dict[str, list[dict]],
) -> Optional[Insight]:
    """
    Find the daily meeting load level where CLS exceeds a sustainable threshold.
    """
    bucket_data: dict[str, dict[str, list[float]]] = {
        "light":    {"cls": [], "fdi": []},  # < 60 min
        "moderate": {"cls": [], "fdi": []},  # 60–120 min
        "heavy":    {"cls": [], "fdi": []},  # 120–180 min
        "intense":  {"cls": [], "fdi": []},  # > 180 min
    }

    def _bucket(total_mins: int) -> str:
        if total_mins < 60:
            return "light"
        elif total_mins < 120:
            return "moderate"
        elif total_mins < 180:
            return "heavy"
        else:
            return "intense"

    for date_str, windows in days_data.items():
        total_mins = _total_meeting_minutes(windows)
        avg_cls = _daily_avg_cls(windows)
        avg_fdi = _daily_avg_fdi(windows)
        b = _bucket(total_mins)
        if avg_cls is not None:
            bucket_data[b]["cls"].append(avg_cls)
        if avg_fdi is not None:
            bucket_data[b]["fdi"].append(avg_fdi)

    # Find the first bucket where CLS exceeds the threshold
    threshold_bucket = None
    threshold_cls = None
    threshold_mins = None

    bucket_order = [("light", 30), ("moderate", 90), ("heavy", 150), ("intense", 210)]
    light_cls = _safe_mean(bucket_data["light"]["cls"])

    for bucket_name, mid_mins in bucket_order:
        cls_vals = bucket_data[bucket_name]["cls"]
        if len(cls_vals) < MIN_SUPPORT_POINTS:
            continue
        avg_cls = _safe_mean(cls_vals)
        if avg_cls is not None and avg_cls >= MEETING_CLS_THRESHOLD:
            if threshold_bucket is None:
                threshold_bucket = bucket_name
                threshold_cls = avg_cls
                threshold_mins = mid_mins
                break

    if threshold_bucket is None or threshold_mins is None or threshold_cls is None:
        return None

    # Only report if there are lighter buckets to compare against
    if light_cls is None or threshold_cls - light_cls < 0.15:
        return None

    # How many days crossed this threshold?
    threshold_mins_lower = {"moderate": 60, "heavy": 120, "intense": 180}.get(threshold_bucket, 180)
    over_threshold_days = sum(
        1 for d, w in days_data.items()
        if _total_meeting_minutes(w) >= threshold_mins_lower
    )

    impact = "High" if threshold_mins_lower <= 120 else "Medium"

    return Insight(
        rank=0,
        title="Meeting Load Threshold",
        headline=(
            f"Days with {threshold_mins_lower // 60}+ hours of meetings average "
            f"CLS {threshold_cls:.2f} — above your sustainable zone. "
            f"Cap daily meeting load at {max(threshold_mins_lower - 60, 30) // 60}h{max(threshold_mins_lower - 60, 30) % 60:02d}min."
        ),
        evidence=(
            f"Light-meeting days (CLS {light_cls:.2f}) vs "
            f"{threshold_bucket}-meeting days (CLS {threshold_cls:.2f}) — "
            f"{over_threshold_days}/{len(days_data)} days exceeded threshold"
        ),
        impact_label=impact,
        n_supporting_days=over_threshold_days,
        magnitude=min((threshold_cls - light_cls) / max(1 - light_cls, 0.01), 1.0),
        detector="meeting_load_threshold",
    )


# ─── Detector 5: Sleep Leverage ───────────────────────────────────────────────

def _detect_sleep_leverage(
    days_data: dict[str, list[dict]],
) -> Optional[Insight]:
    """
    Find the sleep input (hours or quality) that most strongly predicts next-day FDI.
    """
    dates = sorted(days_data.keys())
    pairs_hours: list[tuple[float, float]] = []   # (sleep_hours, next_day_fdi)
    pairs_perf: list[tuple[float, float]] = []    # (sleep_perf, next_day_fdi)

    for i, date_str in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        sleep_h = _sleep_hours(days_data[date_str])
        sleep_p = _sleep_perf(days_data[date_str])
        next_fdi = _daily_avg_fdi(days_data[next_date])

        if next_fdi is None:
            continue
        if sleep_h is not None:
            pairs_hours.append((sleep_h, next_fdi))
        if sleep_p is not None:
            pairs_perf.append((sleep_p, next_fdi))

    best_insight = None
    best_delta = 0.0

    # Sleep hours: threshold 7.5h
    if len(pairs_hours) >= MIN_SUPPORT_POINTS:
        good = [fdi for sh, fdi in pairs_hours if sh >= 7.5]
        poor = [fdi for sh, fdi in pairs_hours if sh < 7.5]
        if good and poor:
            delta = (_safe_mean(good) or 0) - (_safe_mean(poor) or 0)
            if delta >= SLEEP_DELTA_THRESHOLD and delta > best_delta:
                best_delta = delta
                avg_good = _safe_mean(good) or 0
                avg_poor = _safe_mean(poor) or 0
                impact = "High" if delta > 0.15 else "Medium"
                best_insight = Insight(
                    rank=0,
                    title="Sleep Hours Leverage",
                    headline=(
                        f"Getting 7.5+ hours of sleep is associated with {delta:.0%} higher next-day focus. "
                        f"This is your highest-return sleep investment."
                    ),
                    evidence=(
                        f"Next-day FDI with ≥7.5h sleep: {avg_good:.2f} "
                        f"vs <7.5h: {avg_poor:.2f} "
                        f"(n={len(good)} good nights, {len(poor)} short nights)"
                    ),
                    impact_label=impact,
                    n_supporting_days=len(pairs_hours),
                    magnitude=delta,
                    detector="sleep_leverage_hours",
                )

    # Sleep performance: threshold 80%
    if len(pairs_perf) >= MIN_SUPPORT_POINTS:
        good = [fdi for sp, fdi in pairs_perf if sp >= 80]
        poor = [fdi for sp, fdi in pairs_perf if sp < 80]
        if good and poor:
            delta = (_safe_mean(good) or 0) - (_safe_mean(poor) or 0)
            if delta >= SLEEP_DELTA_THRESHOLD and delta > best_delta:
                best_delta = delta
                avg_good = _safe_mean(good) or 0
                avg_poor = _safe_mean(poor) or 0
                impact = "High" if delta > 0.15 else "Medium"
                best_insight = Insight(
                    rank=0,
                    title="Sleep Quality Leverage",
                    headline=(
                        f"Sleep quality ≥80% correlates with {delta:.0%} higher next-day focus. "
                        f"Prioritise sleep consistency and cool/dark conditions."
                    ),
                    evidence=(
                        f"Next-day FDI with sleep perf ≥80%: {avg_good:.2f} "
                        f"vs <80%: {avg_poor:.2f} "
                        f"(n={len(good)} quality nights, {len(poor)} poor nights)"
                    ),
                    impact_label=impact,
                    n_supporting_days=len(pairs_perf),
                    magnitude=delta,
                    detector="sleep_leverage_quality",
                )

    return best_insight


# ─── Detector 6: Load Arc Optimisation ────────────────────────────────────────

def _detect_load_arc(
    days_data: dict[str, list[dict]],
) -> Optional[Insight]:
    """
    Determine if front-loading (heavy morning) vs back-loading (heavy afternoon)
    correlates with better focus quality in the second half of the day.
    """
    front_loaded_pm_fdis: list[float] = []   # morning CLS > afternoon CLS → afternoon FDI
    back_loaded_pm_fdis: list[float] = []    # morning CLS < afternoon CLS → afternoon FDI

    for date_str, windows in days_data.items():
        # Morning: 08:00–12:59 (hours 8–12)
        am_cls_vals = [
            _cls(w) for w in windows
            if _cls(w) is not None
            and w.get("metadata", {}).get("is_active_window", False)
            and 8 <= _hour(w) <= 12
        ]
        # Afternoon: 13:00–17:59 (hours 13–17)
        pm_windows = [
            w for w in windows
            if w.get("metadata", {}).get("is_active_window", False)
            and 13 <= _hour(w) <= 17
        ]
        pm_cls_vals = [_cls(w) for w in pm_windows if _cls(w) is not None]
        pm_fdi_vals = [_fdi(w) for w in pm_windows if _fdi(w) is not None]

        if len(am_cls_vals) < 2 or len(pm_cls_vals) < 2 or len(pm_fdi_vals) < 2:
            continue

        avg_am = _safe_mean(am_cls_vals)
        avg_pm_cls = _safe_mean(pm_cls_vals)
        avg_pm_fdi = _safe_mean(pm_fdi_vals)

        if avg_am is None or avg_pm_cls is None or avg_pm_fdi is None:
            continue

        arc_delta = avg_am - avg_pm_cls
        if arc_delta > 0.10:    # front-loaded (morning heavier)
            front_loaded_pm_fdis.append(avg_pm_fdi)
        elif arc_delta < -0.10:  # back-loaded (afternoon heavier)
            back_loaded_pm_fdis.append(avg_pm_fdi)

    if len(front_loaded_pm_fdis) < MIN_SUPPORT_POINTS or len(back_loaded_pm_fdis) < MIN_SUPPORT_POINTS:
        return None

    avg_front_pm_fdi = _safe_mean(front_loaded_pm_fdis) or 0
    avg_back_pm_fdi = _safe_mean(back_loaded_pm_fdis) or 0
    delta = abs(avg_front_pm_fdi - avg_back_pm_fdi)

    if delta < ARC_FDI_THRESHOLD:
        return None

    if avg_front_pm_fdi > avg_back_pm_fdi:
        better = "front-loading"
        action = "Schedule heavier meetings and collaborative work in the morning to protect afternoon focus."
    else:
        better = "back-loading"
        action = "Morning is your deep-work prime time — protect it from meetings and save collaborations for afternoon."

    impact = "High" if delta > 0.20 else "Medium"

    return Insight(
        rank=0,
        title="Daily Load Arc Optimisation",
        headline=(
            f"{better.capitalize()} correlates with {delta:.0%} better afternoon focus in your data. "
            f"{action}"
        ),
        evidence=(
            f"Afternoon FDI on front-loaded days: {avg_front_pm_fdi:.2f} "
            f"vs back-loaded days: {avg_back_pm_fdi:.2f} "
            f"(n={len(front_loaded_pm_fdis)} front, {len(back_loaded_pm_fdis)} back)"
        ),
        impact_label=impact,
        n_supporting_days=min(len(front_loaded_pm_fdis), len(back_loaded_pm_fdis)),
        magnitude=delta,
        detector="load_arc",
    )


# ─── Main compute function ────────────────────────────────────────────────────

def compute_actionable_insights(
    as_of_date_str: Optional[str] = None,
    days: int = DEFAULT_LOOKBACK,
) -> ActionableInsights:
    """
    Compute top-3 actionable, evidence-backed insights from JSONL history.

    Args:
        as_of_date_str: date to anchor analysis at (defaults to latest available)
        days: lookback window in days (default 14)

    Returns:
        ActionableInsights with .insights list (0–3 items), .is_meaningful bool
    """
    all_dates = list_available_dates()

    if not all_dates:
        return ActionableInsights(is_meaningful=False)

    # Anchor date
    if as_of_date_str:
        available = [d for d in all_dates if d <= as_of_date_str]
    else:
        available = all_dates

    if len(available) < MIN_DAYS:
        return ActionableInsights(
            is_meaningful=False,
            days_analysed=len(available),
        )

    # Select lookback window
    selected_dates = available[-days:] if len(available) > days else available

    # Load all days' data
    days_data: dict[str, list[dict]] = {}
    for d in selected_dates:
        w = read_day(d)
        if w:
            days_data[d] = w

    if len(days_data) < MIN_DAYS:
        return ActionableInsights(is_meaningful=False, days_analysed=len(days_data))

    # Run all 6 detectors
    candidate_insights: list[Insight] = []

    for detector_fn in [
        _detect_meeting_recovery_gap,
        _detect_late_day_cliff,
        _detect_slack_fragmentation,
        _detect_meeting_load_threshold,
        _detect_sleep_leverage,
        _detect_load_arc,
    ]:
        try:
            result = detector_fn(days_data)
            if result is not None:
                candidate_insights.append(result)
        except Exception:
            # Individual detector failures should never crash the pipeline
            pass

    # Sort by impact then magnitude: High first, then within tier by magnitude desc
    impact_order = {"High": 0, "Medium": 1, "Low": 2}
    candidate_insights.sort(
        key=lambda i: (impact_order.get(i.impact_label, 3), -i.magnitude)
    )

    # Take top 3, assign ranks
    top_insights = candidate_insights[:3]
    for rank, insight in enumerate(top_insights, start=1):
        insight.rank = rank

    date_range = (
        f"{min(days_data.keys())} → {max(days_data.keys())}"
        if days_data else ""
    )

    return ActionableInsights(
        insights=top_insights,
        is_meaningful=len(top_insights) > 0,
        days_analysed=len(days_data),
        date_range=date_range,
        generated_at=datetime.now().isoformat(),
    )


# ─── Formatting ───────────────────────────────────────────────────────────────

def format_insights_section(ai: ActionableInsights) -> str:
    """
    Full Slack markdown section with all insights.
    Suitable for weekly summary or on-demand query.
    """
    if not ai.is_meaningful or not ai.insights:
        return ""

    lines = [
        f"*💡 Actionable Insights* — {ai.days_analysed} days analysed",
        "",
    ]

    for insight in ai.insights:
        impact_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(
            insight.impact_label, "•"
        )
        lines.append(f"{insight.rank}. *{insight.title}* {impact_emoji}")
        lines.append(f"   → {insight.headline}")
        lines.append(f"   _Data: {insight.evidence}_")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_insights_brief(ai: ActionableInsights) -> str:
    """
    Compact one-liner for morning brief — just the top insight headline.
    """
    if not ai.is_meaningful or not ai.insights:
        return ""

    top = ai.insights[0]
    return f"💡 *Top insight:* {top.headline}"


def format_insights_terminal(ai: ActionableInsights) -> str:
    """
    Terminal-friendly formatted output with ANSI colours.
    """
    if not ai.is_meaningful:
        return f"Not enough data to generate insights (need ≥{MIN_DAYS} days, have {ai.days_analysed})"

    lines = [
        "\033[1mActionable Insights\033[0m  "
        f"\033[2m{ai.days_analysed} days · {ai.date_range}\033[0m",
        "",
    ]

    impact_colours = {
        "High":   "\033[91m",  # red
        "Medium": "\033[93m",  # yellow
        "Low":    "\033[92m",  # green
    }
    reset = "\033[0m"

    for insight in ai.insights:
        colour = impact_colours.get(insight.impact_label, "")
        lines.append(
            f"  {colour}[{insight.impact_label}]{reset} "
            f"\033[1m{insight.rank}. {insight.title}\033[0m"
        )
        lines.append(f"  → {insight.headline}")
        lines.append(f"  \033[2m{insight.evidence}\033[0m")
        lines.append("")

    return "\n".join(lines).rstrip()


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Presence Tracker — Actionable Insights Generator"
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="Anchor date YYYY-MM-DD (default: latest available)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f"Lookback days (default {DEFAULT_LOOKBACK})",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    insights = compute_actionable_insights(
        as_of_date_str=args.date,
        days=args.days,
    )

    if args.json:
        print(json.dumps(insights.to_dict(), indent=2))
    else:
        print(format_insights_terminal(insights))

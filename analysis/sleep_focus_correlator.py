"""
Presence Tracker — Sleep-to-Focus Correlator (v21)

Answers: *"How does David's sleep quality predict his next-day cognitive performance?"*

WHOOP tells David his recovery score each morning. The CDI tells him his
accumulated fatigue. But neither answers the core planning question:

  "If I sleep an extra hour tonight, how much better will I focus tomorrow?"

The Sleep-Focus Correlator closes this gap. It analyses the relationship
between sleep inputs (hours, HRV, sleep performance, recovery score) and
the following day's cognitive outputs (FDI, CLS, DPS) across all available
JSONL history.

## What it computes

### Per-predictor correlations (Pearson r)
For each sleep input vs each cognitive output:
  - sleep_hours → next_day_fdi
  - sleep_hours → next_day_cls
  - sleep_hours → next_day_dps
  - hrv_rmssd_milli → next_day_fdi
  - sleep_performance → next_day_fdi
  - recovery_score → next_day_fdi

Pearson r ranges from -1 (perfect inverse) to +1 (perfect positive).
A positive r for sleep_hours→fdi means "more sleep → better focus".

### Sleep threshold analysis
Computes the mean next-day FDI and CLS for days bucketed by:
  - sleep_hours: <6h, 6–7h, 7–8h, ≥8h
  - recovery: <50 (red), 50–67 (yellow), ≥67 (green)

These buckets show David exactly how much each hour of extra sleep
is worth in focus quality, using his own historical data.

### Actionable insight generation
From the correlation data, generates one to three actionable sentences:
  - "Your data shows 7+ hours of sleep → 23% better next-day focus."
  - "HRV below 65ms predicts fragmented focus days — protect sleep tonight."
  - "Every extra hour of sleep adds ~0.08 to your next-day FDI."

These are computed deterministically — no LLM required.

### The SleepFocusCorrelation dataclass
  - pairs: list[SleepFocusPair]         — all (sleep, next_day_metrics) pairs
  - correlations: dict[str, float]       — Pearson r for each predictor→outcome
  - sleep_buckets: dict[str, BucketStats] — per-bucket averages (FDI, CLS, DPS)
  - recovery_buckets: dict[str, BucketStats]
  - sleep_hours_slope: float | None      — Δ FDI per extra sleep hour (linear fit)
  - insight_lines: list[str]             — 1-3 actionable sentences
  - is_meaningful: bool                  — False when < MIN_PAIRS pairs
  - pairs_used: int
  - date_range: str

## API

    from analysis.sleep_focus_correlator import (
        compute_sleep_focus_correlation,
        format_sleep_insight_line,
        format_sleep_insight_section,
    )

    corr = compute_sleep_focus_correlation(as_of_date_str)

    # One-liner for morning brief / digest
    line = format_sleep_insight_line(corr)

    # Full section for weekly summary
    section = format_sleep_insight_section(corr)

## CLI

    python3 analysis/sleep_focus_correlator.py             # Full analysis
    python3 analysis/sleep_focus_correlator.py --json      # Machine-readable JSON
    python3 analysis/sleep_focus_correlator.py 2026-04-01  # As of specific date

## Design

- Pure functions — fully testable, no live API calls
- Only reads from the local JSONL store (engine.store)
- Graceful degradation: returns is_meaningful=False with < MIN_PAIRS pairs
- No ML/scipy: uses stdlib math for Pearson correlation and linear fit
- Minimal dependencies — pure Python 3.11 stdlib

## Why this matters

Sleep is the single biggest lever for next-day cognitive performance.
WHOOP quantifies the physiological output (HRV, recovery %).
This module connects the dots: it tells David, from his own data, exactly
how much his sleep choices affect his cognitive capacity — making the
abstract concrete and the actionable personal.
"""

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import list_available_dates, read_summary


# ─── Constants ────────────────────────────────────────────────────────────────

# Minimum day-pairs needed for meaningful correlation analysis
MIN_PAIRS = 5

# Maximum history days to use (recent 90 days for relevance)
HISTORY_DAYS = 90

# Sleep hour bucket boundaries
SLEEP_BUCKETS = [
    ("< 6h", 0.0, 6.0),
    ("6–7h", 6.0, 7.0),
    ("7–8h", 7.0, 8.0),
    ("≥ 8h", 8.0, 99.0),
]

# Recovery bucket labels (WHOOP green/yellow/red)
RECOVERY_BUCKETS = [
    ("Red (<50)", 0, 50),
    ("Yellow (50–67)", 50, 67),
    ("Green (≥67)", 67, 101),
]

# Minimum absolute Pearson r to call a correlation "meaningful"
MEANINGFUL_R = 0.20

# Minimum Δ FDI across bucket bands to report a threshold effect
MIN_BUCKET_DELTA = 0.03


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SleepFocusPair:
    """One day's sleep data paired with the following day's cognitive metrics."""
    sleep_date: str               # YYYY-MM-DD — the night we're measuring
    focus_date: str               # YYYY-MM-DD — the day we're predicting
    sleep_hours: Optional[float]
    sleep_performance: Optional[float]   # 0–100 (WHOOP sleep performance %)
    hrv_rmssd: Optional[float]           # HRV from WHOOP that night
    recovery_score: Optional[float]      # WHOOP recovery % (next morning)
    next_day_fdi: Optional[float]        # avg active FDI on focus_date
    next_day_cls: Optional[float]        # avg CLS on focus_date
    next_day_dps: Optional[float]        # DPS on focus_date

    def to_dict(self) -> dict:
        return {
            "sleep_date": self.sleep_date,
            "focus_date": self.focus_date,
            "sleep_hours": self.sleep_hours,
            "sleep_performance": self.sleep_performance,
            "hrv_rmssd": self.hrv_rmssd,
            "recovery_score": self.recovery_score,
            "next_day_fdi": self.next_day_fdi,
            "next_day_cls": self.next_day_cls,
            "next_day_dps": self.next_day_dps,
        }


@dataclass
class BucketStats:
    """Aggregate cognitive metrics for a sleep/recovery bucket."""
    label: str
    count: int
    avg_fdi: Optional[float]
    avg_cls: Optional[float]
    avg_dps: Optional[float]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "count": self.count,
            "avg_fdi": round(self.avg_fdi, 3) if self.avg_fdi is not None else None,
            "avg_cls": round(self.avg_cls, 3) if self.avg_cls is not None else None,
            "avg_dps": round(self.avg_dps, 1) if self.avg_dps is not None else None,
        }


@dataclass
class SleepFocusCorrelation:
    """Full sleep-to-focus correlation analysis."""
    pairs: list[SleepFocusPair] = field(default_factory=list)
    correlations: dict[str, Optional[float]] = field(default_factory=dict)
    sleep_buckets: list[BucketStats] = field(default_factory=list)
    recovery_buckets: list[BucketStats] = field(default_factory=list)
    sleep_hours_slope: Optional[float] = None    # Δ FDI per extra sleep hour
    sleep_hours_intercept: Optional[float] = None
    insight_lines: list[str] = field(default_factory=list)
    is_meaningful: bool = False
    pairs_used: int = 0
    date_range: str = ""

    def to_dict(self) -> dict:
        return {
            "pairs": [p.to_dict() for p in self.pairs],
            "correlations": {
                k: round(v, 4) if v is not None else None
                for k, v in self.correlations.items()
            },
            "sleep_buckets": [b.to_dict() for b in self.sleep_buckets],
            "recovery_buckets": [b.to_dict() for b in self.recovery_buckets],
            "sleep_hours_slope": round(self.sleep_hours_slope, 4) if self.sleep_hours_slope is not None else None,
            "sleep_hours_intercept": round(self.sleep_hours_intercept, 4) if self.sleep_hours_intercept is not None else None,
            "insight_lines": self.insight_lines,
            "is_meaningful": self.is_meaningful,
            "pairs_used": self.pairs_used,
            "date_range": self.date_range,
        }


# ─── Statistical helpers ──────────────────────────────────────────────────────

def _mean(vals: list[float]) -> Optional[float]:
    """Mean of a non-empty list, None for empty."""
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def _pearson_r(xs: list[float], ys: list[float]) -> Optional[float]:
    """
    Compute Pearson correlation coefficient from paired lists.

    Returns None when:
    - Lists have fewer than 3 pairs (not statistically useful)
    - Standard deviation of either series is zero (constant series)
    - Lists have different lengths

    Pure stdlib — no numpy/scipy required.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return None

    r = cov / denom
    # Clamp to [-1, 1] to handle floating point rounding
    return max(-1.0, min(1.0, r))


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[Optional[float], Optional[float]]:
    """
    Fit a line y = slope * x + intercept via ordinary least squares.

    Returns (slope, intercept) or (None, None) on failure.
    Pure stdlib — no numpy required.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None, None

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)

    if abs(den) < 1e-12:
        return None, None

    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _paired(pairs: list[SleepFocusPair], x_attr: str, y_attr: str) -> tuple[list[float], list[float]]:
    """
    Extract paired (x, y) lists from SleepFocusPair list, dropping rows
    where either value is None.
    """
    xs, ys = [], []
    for p in pairs:
        x = getattr(p, x_attr, None)
        y = getattr(p, y_attr, None)
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_pairs(
    as_of_date_str: str,
    days: int = HISTORY_DAYS,
) -> list[SleepFocusPair]:
    """
    Build a list of (night_sleep, next_day_focus) pairs from rolling summary.

    Each pair represents:
      - sleep_date: the date whose WHOOP sleep data is used as predictors
      - focus_date: sleep_date + 1 day (the day we're predicting)

    Only pairs where both dates have data are included.
    Days after as_of_date_str are excluded to prevent data leakage.
    """
    try:
        summary = read_summary()
        all_days = summary.get("days", {})
    except Exception:
        return []

    # All available dates up to as_of_date_str, sorted chronologically
    available = sorted(
        [d for d in list_available_dates() if d <= as_of_date_str],
    )

    # Limit to recent HISTORY_DAYS
    available = available[-days:]

    pairs = []

    for i, sleep_date in enumerate(available[:-1]):
        # Next calendar date (not just next available)
        try:
            sleep_dt = datetime.strptime(sleep_date, "%Y-%m-%d")
            from datetime import timedelta
            focus_dt = sleep_dt + timedelta(days=1)
            focus_date = focus_dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

        # We need the focus date to be in available data
        if focus_date not in available:
            continue

        sleep_day = all_days.get(sleep_date, {})
        focus_day = all_days.get(focus_date, {})

        if not sleep_day or not focus_day:
            continue

        # Extract sleep predictors from the sleep date's WHOOP data
        sleep_whoop = sleep_day.get("whoop", {}) or {}
        sleep_hours = sleep_whoop.get("sleep_hours")
        sleep_perf = sleep_whoop.get("sleep_performance")
        hrv = sleep_whoop.get("hrv_rmssd_milli")
        recovery = sleep_whoop.get("recovery_score")

        # Extract cognitive outcomes from the focus date
        focus_metrics = (focus_day.get("metrics_avg") or {})
        next_fdi = focus_metrics.get("focus_depth_index")
        next_cls = focus_metrics.get("cognitive_load_score")

        # DPS from presence_score section (set by enrich_summary_with_dps)
        next_dps = (focus_day.get("presence_score") or {}).get("dps")

        # Skip pairs with no meaningful data on either end
        if (sleep_hours is None and hrv is None and recovery is None):
            continue
        if next_fdi is None and next_cls is None:
            continue

        pairs.append(SleepFocusPair(
            sleep_date=sleep_date,
            focus_date=focus_date,
            sleep_hours=sleep_hours,
            sleep_performance=sleep_perf,
            hrv_rmssd=hrv,
            recovery_score=recovery,
            next_day_fdi=next_fdi,
            next_day_cls=next_cls,
            next_day_dps=next_dps,
        ))

    return pairs


# ─── Bucket analysis ──────────────────────────────────────────────────────────

def _compute_sleep_buckets(pairs: list[SleepFocusPair]) -> list[BucketStats]:
    """Compute next-day cognitive averages bucketed by sleep hours."""
    buckets = []
    for label, lo, hi in SLEEP_BUCKETS:
        members = [
            p for p in pairs
            if p.sleep_hours is not None and lo <= p.sleep_hours < hi
        ]
        fdi_vals = [p.next_day_fdi for p in members if p.next_day_fdi is not None]
        cls_vals = [p.next_day_cls for p in members if p.next_day_cls is not None]
        dps_vals = [p.next_day_dps for p in members if p.next_day_dps is not None]
        buckets.append(BucketStats(
            label=label,
            count=len(members),
            avg_fdi=_mean(fdi_vals),
            avg_cls=_mean(cls_vals),
            avg_dps=_mean(dps_vals),
        ))
    return buckets


def _compute_recovery_buckets(pairs: list[SleepFocusPair]) -> list[BucketStats]:
    """Compute next-day cognitive averages bucketed by WHOOP recovery score."""
    buckets = []
    for label, lo, hi in RECOVERY_BUCKETS:
        members = [
            p for p in pairs
            if p.recovery_score is not None and lo <= p.recovery_score < hi
        ]
        fdi_vals = [p.next_day_fdi for p in members if p.next_day_fdi is not None]
        cls_vals = [p.next_day_cls for p in members if p.next_day_cls is not None]
        dps_vals = [p.next_day_dps for p in members if p.next_day_dps is not None]
        buckets.append(BucketStats(
            label=label,
            count=len(members),
            avg_fdi=_mean(fdi_vals),
            avg_cls=_mean(cls_vals),
            avg_dps=_mean(dps_vals),
        ))
    return buckets


# ─── Insight generation ───────────────────────────────────────────────────────

def _generate_insights(
    correlations: dict[str, Optional[float]],
    sleep_buckets: list[BucketStats],
    recovery_buckets: list[BucketStats],
    slope: Optional[float],
    pairs_used: int,
) -> list[str]:
    """
    Generate 1–3 actionable insight sentences from the correlation data.

    Insights are deterministic — derived purely from the computed statistics.
    The most impactful insight comes first.

    Rules (in priority order):
    1. If sleep_hours→fdi correlation is meaningful (+r ≥ 0.20):
       Report the slope in concrete terms ("each extra hour adds X to FDI").
    2. If sleep bucket data shows a clear inflection point (best bucket vs worst):
       Report the threshold effect in % terms.
    3. If HRV→fdi correlation is meaningful:
       Report the HRV signal.
    4. If recovery→fdi correlation is meaningful:
       Report the recovery score signal.
    5. If no correlations are meaningful but data exists:
       Report "insufficient data" with days count.
    """
    lines = []

    r_sleep_fdi = correlations.get("sleep_hours__next_day_fdi")
    r_hrv_fdi = correlations.get("hrv_rmssd__next_day_fdi")
    r_recovery_fdi = correlations.get("recovery_score__next_day_fdi")
    r_sleep_cls = correlations.get("sleep_hours__next_day_cls")

    # ── Insight 1: Sleep hours → FDI slope ───────────────────────────────────
    if r_sleep_fdi is not None and abs(r_sleep_fdi) >= MEANINGFUL_R and slope is not None:
        slope_pct = round(slope * 100, 1)
        direction = "+" if slope > 0 else ""
        if abs(slope) >= 0.01:
            lines.append(
                f"Each extra hour of sleep {'improves' if slope > 0 else 'changes'} "
                f"next-day focus (FDI) by {direction}{slope_pct:.1f} points on average "
                f"(r={r_sleep_fdi:+.2f}, {pairs_used} nights)."
            )
        elif r_sleep_fdi >= MEANINGFUL_R:
            lines.append(
                f"Sleep duration correlates positively with next-day focus "
                f"(r={r_sleep_fdi:+.2f}, {pairs_used} nights)."
            )

    # ── Insight 2: Sleep bucket threshold effect ──────────────────────────────
    meaningful_buckets = [b for b in sleep_buckets if b.count >= 2 and b.avg_fdi is not None]
    if len(meaningful_buckets) >= 2:
        best_b = max(meaningful_buckets, key=lambda b: b.avg_fdi)
        worst_b = min(meaningful_buckets, key=lambda b: b.avg_fdi)
        delta = best_b.avg_fdi - worst_b.avg_fdi
        if delta >= MIN_BUCKET_DELTA:
            pct_delta = round(delta * 100, 0)
            lines.append(
                f"Sleep {best_b.label} → avg FDI {best_b.avg_fdi:.0%} vs "
                f"{worst_b.label} → {worst_b.avg_fdi:.0%} "
                f"({'+' if pct_delta >= 0 else ''}{pct_delta:.0f}pp difference)."
            )

    # ── Insight 3: HRV signal ─────────────────────────────────────────────────
    if (
        r_hrv_fdi is not None
        and abs(r_hrv_fdi) >= MEANINGFUL_R
        and len(lines) < 3
    ):
        direction = "higher" if r_hrv_fdi > 0 else "lower"
        lines.append(
            f"Higher HRV → {'better' if r_hrv_fdi > 0 else 'worse'} next-day focus "
            f"(r={r_hrv_fdi:+.2f}) — HRV is a {direction} predictor of your focus quality."
        )

    # ── Insight 4: Recovery score signal ─────────────────────────────────────
    if (
        r_recovery_fdi is not None
        and abs(r_recovery_fdi) >= MEANINGFUL_R
        and len(lines) < 3
    ):
        lines.append(
            f"WHOOP recovery score correlates {'positively' if r_recovery_fdi > 0 else 'negatively'} "
            f"with next-day focus depth (r={r_recovery_fdi:+.2f})."
        )

    # ── Fallback: Not enough data yet ────────────────────────────────────────
    if not lines:
        need_more = max(0, MIN_PAIRS - pairs_used)
        if need_more > 0:
            lines.append(
                f"Need {need_more} more day{'s' if need_more != 1 else ''} of data "
                f"to surface sleep-focus correlations."
            )
        else:
            lines.append(
                f"No strong sleep-focus correlations detected in {pairs_used} nights — "
                "your focus appears resilient across sleep conditions."
            )

    return lines[:3]  # Max 3 insight lines


# ─── Main function ────────────────────────────────────────────────────────────

def compute_sleep_focus_correlation(
    as_of_date_str: Optional[str] = None,
    days: int = HISTORY_DAYS,
) -> SleepFocusCorrelation:
    """
    Compute the full sleep-to-focus correlation analysis from historical data.

    Parameters
    ----------
    as_of_date_str : str | None
        Upper bound date (YYYY-MM-DD). Defaults to today.
    days : int
        Maximum historical days to include.

    Returns
    -------
    SleepFocusCorrelation
        is_meaningful=False when fewer than MIN_PAIRS pairs available.
    """
    today_str = as_of_date_str or datetime.now().strftime("%Y-%m-%d")

    # Load day pairs
    pairs = _load_pairs(today_str, days)

    date_range = ""
    if pairs:
        date_range = f"{pairs[0].sleep_date} → {pairs[-1].focus_date}"

    pairs_used = len(pairs)

    if pairs_used < MIN_PAIRS:
        # Not enough data — return graceful empty result
        return SleepFocusCorrelation(
            pairs=pairs,
            is_meaningful=False,
            pairs_used=pairs_used,
            date_range=date_range,
            insight_lines=_generate_insights({}, [], [], None, pairs_used),
        )

    # ── Compute correlations ─────────────────────────────────────────────────

    correlations: dict[str, Optional[float]] = {}

    # sleep_hours as predictor
    for x_attr in ("sleep_hours", "sleep_performance", "hrv_rmssd", "recovery_score"):
        for y_attr in ("next_day_fdi", "next_day_cls", "next_day_dps"):
            xs, ys = _paired(pairs, x_attr, y_attr)
            key = f"{x_attr}__{y_attr}"
            correlations[key] = _pearson_r(xs, ys)

    # ── Linear fit: sleep_hours → next_day_fdi (most actionable) ────────────
    sleep_xs, fdi_ys = _paired(pairs, "sleep_hours", "next_day_fdi")
    slope, intercept = _linear_fit(sleep_xs, fdi_ys)

    # ── Sleep bucket stats ───────────────────────────────────────────────────
    sleep_buckets = _compute_sleep_buckets(pairs)
    recovery_buckets = _compute_recovery_buckets(pairs)

    # ── Generate insights ────────────────────────────────────────────────────
    insights = _generate_insights(correlations, sleep_buckets, recovery_buckets, slope, pairs_used)

    return SleepFocusCorrelation(
        pairs=pairs,
        correlations=correlations,
        sleep_buckets=sleep_buckets,
        recovery_buckets=recovery_buckets,
        sleep_hours_slope=slope,
        sleep_hours_intercept=intercept,
        insight_lines=insights,
        is_meaningful=True,
        pairs_used=pairs_used,
        date_range=date_range,
    )


# ─── Formatters ───────────────────────────────────────────────────────────────

def format_sleep_insight_line(corr: SleepFocusCorrelation) -> str:
    """
    Format the top sleep-focus insight as a single Slack-ready line.

    Returns empty string when not meaningful.

    Example:
        "😴 *Sleep insight:* Each extra hour adds +6.2 FDI points (r=+0.41, 21 nights)."
    """
    if not corr.is_meaningful or not corr.insight_lines:
        return ""
    return f"😴 *Sleep insight:* {corr.insight_lines[0]}"


def format_sleep_insight_section(corr: SleepFocusCorrelation) -> str:
    """
    Format the full sleep-focus analysis as a multi-line Slack/terminal section.

    Returns empty string when not meaningful.

    Example:
        😴 *Sleep → Focus Correlation* (21 nights)

        Each extra hour of sleep improves next-day FDI by +6.2 points (r=+0.41).
        Sleep 7–8h → avg FDI 87% vs <6h → 71% (+16pp difference).

        *Sleep buckets:*
        < 6h  (3 nights)  FDI 71%  CLS 0.42
        6–7h  (8 nights)  FDI 79%  CLS 0.31
        7–8h  (7 nights)  FDI 87%  CLS 0.22
        ≥ 8h  (3 nights)  FDI 89%  CLS 0.18
    """
    if not corr.is_meaningful:
        if corr.insight_lines:
            return f"😴 *Sleep → Focus:* _{corr.insight_lines[0]}_"
        return ""

    lines = [f"😴 *Sleep → Focus Correlation* _({corr.pairs_used} nights · {corr.date_range})_"]
    lines.append("")

    # Insight sentences
    for insight in corr.insight_lines:
        lines.append(insight)

    # Bucket table (only buckets with ≥ 1 pair)
    meaningful_buckets = [b for b in corr.sleep_buckets if b.count >= 1]
    if meaningful_buckets:
        lines.append("")
        lines.append("*Sleep duration buckets:*")
        for b in meaningful_buckets:
            fdi_str = f"FDI {b.avg_fdi:.0%}" if b.avg_fdi is not None else "—"
            cls_str = f"CLS {b.avg_cls:.2f}" if b.avg_cls is not None else "—"
            dps_str = f"DPS {b.avg_dps:.0f}" if b.avg_dps is not None else ""
            night_str = f"{b.count} night{'s' if b.count != 1 else ''}"
            parts = [p for p in [fdi_str, cls_str, dps_str] if p]
            lines.append(f"  {b.label:<8}  ({night_str})  {' · '.join(parts)}")

    # Recovery bucket table
    meaningful_rec = [b for b in corr.recovery_buckets if b.count >= 1]
    if meaningful_rec and len(meaningful_rec) > 1:
        lines.append("")
        lines.append("*WHOOP recovery buckets:*")
        for b in meaningful_rec:
            fdi_str = f"FDI {b.avg_fdi:.0%}" if b.avg_fdi is not None else "—"
            cls_str = f"CLS {b.avg_cls:.2f}" if b.avg_cls is not None else "—"
            night_str = f"{b.count} night{'s' if b.count != 1 else ''}"
            lines.append(f"  {b.label:<18}  ({night_str})  {fdi_str} · {cls_str}")

    return "\n".join(lines)


def format_sleep_insight_terminal(corr: SleepFocusCorrelation) -> str:
    """
    Terminal-formatted sleep-focus correlation report (ANSI colours).
    """
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    CYAN  = "\033[96m"
    DIM   = "\033[2m"
    RESET = "\033[0m"

    if not corr.is_meaningful:
        msg = corr.insight_lines[0] if corr.insight_lines else "Not enough data."
        return f"\n  {BOLD}Sleep → Focus Correlator{RESET}\n  {DIM}{msg}{RESET}\n"

    lines = [
        "",
        f"{BOLD}Sleep → Focus Correlation{RESET}  {DIM}({corr.pairs_used} nights · {corr.date_range}){RESET}",
        "=" * 60,
    ]

    for insight in corr.insight_lines:
        lines.append(f"\n  {insight}")

    # Correlation table
    r_sleep = corr.correlations.get("sleep_hours__next_day_fdi")
    r_hrv = corr.correlations.get("hrv_rmssd__next_day_fdi")
    r_recovery = corr.correlations.get("recovery_score__next_day_fdi")

    lines.append(f"\n{BOLD}Pearson correlations → next-day FDI{RESET}")
    for label, r in [("sleep_hours", r_sleep), ("hrv_rmssd", r_hrv), ("recovery_score", r_recovery)]:
        if r is not None:
            colour = GREEN if r >= MEANINGFUL_R else (CYAN if r >= 0 else "")
            bar = "▓" * int(abs(r) * 10) + "░" * (10 - int(abs(r) * 10))
            lines.append(f"  {label:<18} r={r:+.3f}  {colour}{bar}{RESET}")

    # Bucket table
    lines.append(f"\n{BOLD}Sleep buckets (next-day FDI){RESET}")
    header = f"  {'Bucket':<10}  {'Nights':>6}  {'FDI':>6}  {'CLS':>6}"
    lines.append(header)
    for b in corr.sleep_buckets:
        if b.count == 0:
            continue
        fdi_str = f"{b.avg_fdi:.0%}" if b.avg_fdi is not None else "  —  "
        cls_str = f"{b.avg_cls:.3f}" if b.avg_cls is not None else "  —  "
        lines.append(f"  {b.label:<10}  {b.count:>6}  {fdi_str:>6}  {cls_str:>6}")

    lines.append("")
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.

    Usage:
        python3 analysis/sleep_focus_correlator.py             # Full analysis
        python3 analysis/sleep_focus_correlator.py --json      # JSON output
        python3 analysis/sleep_focus_correlator.py 2026-04-01  # As of date
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Sleep-to-Focus Correlator — how sleep predicts next-day cognitive performance"
    )
    parser.add_argument("date", nargs="?", help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    corr = compute_sleep_focus_correlation(date_str)

    if args.json:
        print(json.dumps(corr.to_dict(), indent=2))
        return

    print(format_sleep_insight_terminal(corr))


if __name__ == "__main__":
    main()

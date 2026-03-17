"""
Presence Tracker — Burnout Risk Index (BRI)

Answers: *"Am I on a trajectory toward burnout — and how urgent is the warning?"*

## Why BRI is different from CDI

The Cognitive Debt Index (CDI) measures the *current* load/recovery imbalance
over a 14-day rolling window.  CDI answers: "How much debt am I carrying right now?"

The Burnout Risk Index answers a different question: "Where is my *trajectory* heading
over the next 2–4 weeks if this pattern continues?"

CDI can be high from a single hard week.  BRI rises only when *multiple signals*
are simultaneously trending in the wrong direction over 4 weeks — the pattern
that historically precedes burnout in knowledge workers.

## What drives BRI

BRI is a composite of five signal trends, each weighted by clinical relevance:

    1. HRV Trend (30%)
       HRV is the most reliable physiological burnout predictor.  A declining
       HRV trend over 3+ weeks signals autonomic nervous system overload.
       Signal: slope of daily avg hrv_rmssd_milli over the lookback window.
       Normalized: 0.0 (strong positive trend) → 1.0 (strong negative trend)

    2. Sleep Degradation (20%)
       Sustained sleep quality decline precedes burnout by 2–4 weeks.
       Signal: slope of sleep_performance + sleep_hours over lookback window.
       Normalized: 0.0 (improving) → 1.0 (strongly degrading)

    3. Cognitive Load Creep (20%)
       Gradually rising CLS over weeks signals increasing demand without
       proportionate recovery.  A single spike is noise; a slope is a signal.
       Signal: slope of daily avg_cls over lookback window.
       Normalized: 0.0 (stable/declining) → 1.0 (strong upward creep)

    4. Focus Erosion (15%)
       FDI declining over weeks while CLS rises (or stays high) is a
       classic burnout precursor — cognitive capacity being consumed
       faster than it's being restored.
       Signal: slope of daily active_fdi over lookback window (inverted).
       Normalized: 0.0 (improving) → 1.0 (strong downward trend)

    5. Social Drain Accumulation (15%)
       Rising SDI over weeks — sustained high social energy expenditure
       without recovery time — compounds physical fatigue signals.
       Signal: slope of daily avg_sdi over lookback window.
       Normalized: 0.0 (stable/declining) → 1.0 (strongly rising)

## Formula

    raw_bri = (
        0.30 × hrv_trend_component
      + 0.20 × sleep_degradation_component
      + 0.20 × load_creep_component
      + 0.15 × focus_erosion_component
      + 0.15 × social_drain_component
    )

    BRI = clamp(raw_bri × 100, 0, 100)

## BRI Tiers

    BRI < 20   → 'healthy'     — all trends favourable or neutral; low risk
    BRI 20–40  → 'watch'       — mild negative trends; monitor closely
    BRI 40–60  → 'caution'     — multiple concerning trends; intervene soon
    BRI 60–80  → 'high_risk'   — strong multi-signal burnout trajectory
    BRI > 80   → 'critical'    — immediate pattern change required

## Minimum data requirements

BRI requires ≥ 14 days of data to compute meaningful slopes.
With fewer days the signal is too short for trend detection.
Returns is_meaningful=False when insufficient data.

## Interpretation principle

BRI is *not* a diagnostic — it's a leading indicator.  A BRI of 55
does not mean David is burned out; it means the current trajectory,
if unchanged for 2–4 more weeks, leads to a high-risk state.
Early intervention (a lighter week, a rest day, sleep protection)
resets the trajectory.

## Output

    BurnoutRisk dataclass:
      - bri: float                    — 0–100 (higher = more risk)
      - tier: str                     — healthy | watch | caution | high_risk | critical
      - tier_label: str               — human label with emoji
      - days_used: int                — actual days in analysis
      - days_requested: int           — lookback window requested
      - is_meaningful: bool           — False when < MIN_DAYS
      - components: dict[str, float]  — per-signal component (0.0–1.0 each)
      - component_labels: dict[str, str] — human label for each component
      - dominant_signal: str          — which signal is driving the risk most
      - trend_direction: str          — 'worsening' | 'stable' | 'improving'
      - trajectory_headline: str      — one-sentence summary
      - intervention_advice: str      — specific action David can take
      - hrv_slope: float | None       — HRV change per day (ms/day, negative=declining)
      - sleep_slope: float | None     — sleep performance change per day (%/day)
      - cls_slope: float | None       — CLS change per day
      - fdi_slope: float | None       — FDI change per day (negative=declining)
      - sdi_slope: float | None       — SDI change per day

## API

    from analysis.burnout_risk import compute_burnout_risk, format_bri_line
    from analysis.burnout_risk import format_bri_section

    bri = compute_burnout_risk(as_of_date_str, days=28)

    # One-liner for morning brief or nightly digest
    line = format_bri_line(bri)

    # Full section for weekly summary
    section = format_bri_section(bri)

## CLI

    python3 analysis/burnout_risk.py             # Last 28 days
    python3 analysis/burnout_risk.py --days 21   # Last 21 days
    python3 analysis/burnout_risk.py --json      # Machine-readable JSON
    python3 analysis/burnout_risk.py 2026-04-01  # As of specific date

## Design

  - Pure functions except for store access — fully testable
  - Slope computation uses scipy linregress if available, else a simple
    linear regression fallback (no hard dependency)
  - All inputs are drawn from existing JSONL data — no new API calls
  - Graceful degradation at every step — any missing signal → that
    component returns 0.0 (neutral, not missing)
  - The intervention advice is tier- and dominant-signal-specific,
    not generic
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import read_day, list_available_dates

# ─── Constants ────────────────────────────────────────────────────────────────

# Minimum days of data for meaningful BRI
BRI_MIN_DAYS = 14

# Default lookback window
BRI_DEFAULT_DAYS = 28

# Component weights (must sum to 1.0)
WEIGHT_HRV          = 0.30
WEIGHT_SLEEP        = 0.20
WEIGHT_LOAD_CREEP   = 0.20
WEIGHT_FOCUS_EROSION = 0.15
WEIGHT_SOCIAL_DRAIN  = 0.15

# Slope normalisation reference points
# These represent the slope value that maps to component = 1.0 (max risk)
HRV_SLOPE_SCALE      = -0.5   # ms/day — losing 0.5ms HRV per day maps to 1.0
SLEEP_SLOPE_SCALE    = -1.0   # %/day  — losing 1% sleep performance per day → 1.0
CLS_SLOPE_SCALE      =  0.005 # CLS/day — gaining 0.005 CLS per day → 1.0
FDI_SLOPE_SCALE      = -0.005 # FDI/day — losing 0.005 FDI per day → 1.0
SDI_SLOPE_SCALE      =  0.005 # SDI/day — gaining 0.005 SDI per day → 1.0

# BRI tier thresholds
BRI_HEALTHY_MAX    = 20
BRI_WATCH_MAX      = 40
BRI_CAUTION_MAX    = 60
BRI_HIGH_RISK_MAX  = 80
# > BRI_HIGH_RISK_MAX → 'critical'


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class BurnoutRisk:
    """Burnout Risk Index for a given analysis period."""

    # Core score
    bri: float                              # 0–100
    tier: str                               # healthy | watch | caution | high_risk | critical
    tier_label: str                         # human label with emoji
    is_meaningful: bool                     # False when insufficient data

    # Data coverage
    days_used: int
    days_requested: int
    date_range: str                         # "YYYY-MM-DD → YYYY-MM-DD"

    # Component scores (each 0.0–1.0, higher = more risk)
    components: dict = field(default_factory=dict)
    component_labels: dict = field(default_factory=dict)

    # Dominant driver
    dominant_signal: str = "unknown"        # which component drove BRI highest

    # Trajectory
    trend_direction: str = "stable"         # worsening | stable | improving
    trajectory_headline: str = ""
    intervention_advice: str = ""

    # Raw slopes (for debugging / display)
    hrv_slope: Optional[float] = None       # ms/day
    sleep_slope: Optional[float] = None     # %/day
    cls_slope: Optional[float] = None       # CLS/day
    fdi_slope: Optional[float] = None       # FDI/day
    sdi_slope: Optional[float] = None       # SDI/day

    def to_dict(self) -> dict:
        return {
            "bri": self.bri,
            "tier": self.tier,
            "tier_label": self.tier_label,
            "is_meaningful": self.is_meaningful,
            "days_used": self.days_used,
            "days_requested": self.days_requested,
            "date_range": self.date_range,
            "components": self.components,
            "component_labels": self.component_labels,
            "dominant_signal": self.dominant_signal,
            "trend_direction": self.trend_direction,
            "trajectory_headline": self.trajectory_headline,
            "intervention_advice": self.intervention_advice,
            "hrv_slope": self.hrv_slope,
            "sleep_slope": self.sleep_slope,
            "cls_slope": self.cls_slope,
            "fdi_slope": self.fdi_slope,
            "sdi_slope": self.sdi_slope,
        }


# ─── Internal helpers ────────────────────────────────────────────────────────

def _linear_slope(xs: list[float], ys: list[float]) -> Optional[float]:
    """
    Compute slope (dy/dx) of a linear regression through the given points.

    Uses scipy if available; falls back to a pure-Python implementation.
    Returns None if fewer than 2 points.
    """
    if len(xs) < 2 or len(ys) < 2:
        return None

    n = len(xs)
    assert n == len(ys)

    try:
        from scipy import stats as sp_stats
        result = sp_stats.linregress(xs, ys)
        return float(result.slope)
    except ImportError:
        pass

    # Pure-Python fallback: ordinary least squares
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_mean(vals: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in vals if v is not None]
    return sum(valid) / len(valid) if valid else None


def _extract_daily_series(dates: list[str]) -> dict:
    """
    For each date in the list, extract per-day averages of key metrics.
    Returns a dict of lists (one per signal), in date order.
    """
    hrv_vals: list[Optional[float]] = []
    sleep_vals: list[Optional[float]] = []
    cls_vals: list[Optional[float]] = []
    fdi_vals: list[Optional[float]] = []
    sdi_vals: list[Optional[float]] = []

    for date_str in dates:
        windows = read_day(date_str)
        if not windows:
            hrv_vals.append(None)
            sleep_vals.append(None)
            cls_vals.append(None)
            fdi_vals.append(None)
            sdi_vals.append(None)
            continue

        # WHOOP signals — same value per day; take from any non-null window
        hrv = None
        sleep_perf = None
        for w in windows:
            whoop = w.get("whoop") or {}
            if hrv is None and whoop.get("hrv_rmssd_milli"):
                hrv = float(whoop["hrv_rmssd_milli"])
            if sleep_perf is None and whoop.get("sleep_performance"):
                sleep_perf = float(whoop["sleep_performance"])
            if hrv and sleep_perf:
                break

        hrv_vals.append(hrv)
        sleep_vals.append(sleep_perf)

        # Per-window metrics — average over active working-hours windows
        active = [
            w for w in windows
            if w.get("metadata", {}).get("is_active_window")
            and w.get("metadata", {}).get("is_working_hours")
        ]
        if not active:
            # Fall back to any window with metrics
            active = [w for w in windows if w.get("metrics")]

        if active:
            m_cls = _safe_mean([w["metrics"].get("cognitive_load_score") for w in active])
            m_fdi = _safe_mean([w["metrics"].get("focus_depth_index") for w in active])
            m_sdi = _safe_mean([w["metrics"].get("social_drain_index") for w in active])
        else:
            m_cls = m_fdi = m_sdi = None

        cls_vals.append(m_cls)
        fdi_vals.append(m_fdi)
        sdi_vals.append(m_sdi)

    return {
        "hrv": hrv_vals,
        "sleep": sleep_vals,
        "cls": cls_vals,
        "fdi": fdi_vals,
        "sdi": sdi_vals,
    }


def _slope_to_component(slope: Optional[float], scale: float) -> float:
    """
    Convert a raw slope value to a component score in [0.0, 1.0].

    The `scale` parameter represents the slope value that corresponds to
    maximum risk (component = 1.0).  The scaling is:
      - If slope is in the direction of risk (same sign as scale), maps to [0, 1]
      - If slope is in the opposite direction (improving), clamps to 0.0

    For declining signals (scale < 0): slope < scale → 1.0
    For rising signals (scale > 0): slope > scale → 1.0
    """
    if slope is None:
        return 0.0

    if scale < 0:
        # Signal where decline (negative slope) = risk
        # slope < scale → component near 1.0
        if slope >= 0:
            return 0.0  # improving, no risk
        return _clamp(slope / scale)
    else:
        # Signal where increase (positive slope) = risk
        if slope <= 0:
            return 0.0  # improving, no risk
        return _clamp(slope / scale)


def _tier_from_bri(bri: float) -> tuple[str, str]:
    """Return (tier_key, tier_label_with_emoji) for a BRI score."""
    if bri < BRI_HEALTHY_MAX:
        return "healthy", "🟢 Healthy"
    elif bri < BRI_WATCH_MAX:
        return "watch", "🟡 Watch"
    elif bri < BRI_CAUTION_MAX:
        return "caution", "🟠 Caution"
    elif bri < BRI_HIGH_RISK_MAX:
        return "high_risk", "🔴 High Risk"
    else:
        return "critical", "🚨 Critical"


def _intervention_advice(tier: str, dominant_signal: str) -> str:
    """Generate specific intervention advice based on tier and dominant signal."""
    if tier == "healthy":
        return "Current trajectory is sustainable. Keep protecting sleep and pacing load."

    advice_matrix = {
        "watch": {
            "hrv":         "HRV trending down — prioritise 8h sleep for the next 5 days.",
            "sleep":       "Sleep quality dipping — set a consistent wind-down time this week.",
            "load_creep":  "Load creeping up gradually — protect one half-day per week as no-meetings.",
            "focus_erosion": "Focus depth fading — reduce multitasking; block 90min focused work daily.",
            "social_drain": "Social load rising — schedule solitary recovery time each afternoon.",
        },
        "caution": {
            "hrv":         "HRV declining — this is a physiological stress signal. Take a light day now, not later.",
            "sleep":       "Sleep is degrading under sustained load — sleep must become non-negotiable for 1 week.",
            "load_creep":  "CLS has been climbing for weeks — reschedule or delegate tasks that aren't essential.",
            "focus_erosion": "Focus is eroding — this is the early sign of cognitive depletion. Reduce meeting load.",
            "social_drain": "Social drain is accumulating — protect at least 2 quiet mornings this week.",
        },
        "high_risk": {
            "hrv":         "HRV trajectory is a clear autonomic stress signal — consider a recovery day this week.",
            "sleep":       "Sleep quality has been declining for weeks — this is the primary driver; fix sleep first.",
            "load_creep":  "Cognitive load has been steadily rising — an unscheduled buffer day is needed this week.",
            "focus_erosion": "Focus depth has collapsed — this indicates significant cognitive fatigue. Protect recovery.",
            "social_drain": "Social drain is high and sustained — cancel or delegate high-SDI commitments this week.",
        },
        "critical": {
            "hrv":         "URGENT: HRV trend indicates autonomic overload. A full recovery day is needed immediately.",
            "sleep":       "URGENT: Sleep quality has been severely degraded. Prioritise 9h sleep tonight.",
            "load_creep":  "URGENT: Cognitive load has climbed to unsustainable levels. Reschedule everything non-critical.",
            "focus_erosion": "URGENT: Focus has collapsed to near-zero depth. Immediate cognitive rest required.",
            "social_drain": "URGENT: Social drain is maxed. Clear your calendar for tomorrow.",
        },
    }

    tier_advice = advice_matrix.get(tier, {})
    return tier_advice.get(dominant_signal, tier_advice.get("load_creep", "Protect recovery time this week."))


def _component_label(signal: str, component: float) -> str:
    """Return a short human label for a component score."""
    labels = {
        "hrv": ("HRV trending up", "HRV stable", "HRV slowly declining", "HRV declining"),
        "sleep": ("Sleep improving", "Sleep stable", "Sleep degrading", "Sleep declining sharply"),
        "load_creep": ("Load stable/dropping", "Load mild creep", "Load creeping up", "Load rising sharply"),
        "focus_erosion": ("Focus improving", "Focus stable", "Focus fading", "Focus eroding"),
        "social_drain": ("Social load stable", "Social load rising", "Social drain high", "Social drain critical"),
    }
    thresholds = [0.0, 0.25, 0.5, 0.75]
    idx = 0
    for t in thresholds:
        if component >= t:
            idx = thresholds.index(t)
    bucket = min(idx, 3)
    return labels.get(signal, ("fine", "mild", "moderate", "high"))[bucket]


def _trajectory_headline(bri: float, tier: str, dominant_signal: str,
                         trend_direction: str, days_used: int) -> str:
    """Generate a one-sentence trajectory summary."""
    signal_names = {
        "hrv": "HRV decline",
        "sleep": "sleep degradation",
        "load_creep": "rising cognitive load",
        "focus_erosion": "focus erosion",
        "social_drain": "social drain accumulation",
    }
    signal_label = signal_names.get(dominant_signal, "load pattern")

    if tier == "healthy":
        return f"All burnout indicators are in normal range over the last {days_used} days."
    elif tier == "watch":
        return f"Mild {signal_label} detected over {days_used} days — worth monitoring but not urgent."
    elif tier == "caution":
        return f"Multiple concerning trends over {days_used} days, led by {signal_label}. Intervene soon."
    elif tier == "high_risk":
        return f"Strong burnout trajectory over {days_used} days. {signal_label.capitalize()} is the primary driver."
    else:
        return f"Critical burnout risk. {days_used}-day pattern shows {signal_label} requiring immediate change."


# ─── Core computation ─────────────────────────────────────────────────────────

def compute_burnout_risk(
    as_of_date_str: Optional[str] = None,
    days: int = BRI_DEFAULT_DAYS,
) -> "BurnoutRisk":
    """
    Compute the Burnout Risk Index for the given date and lookback window.

    Args:
        as_of_date_str: End date for analysis (default: most recent data day).
        days: Lookback window in days (default: 28).

    Returns:
        BurnoutRisk dataclass.
    """
    available = list_available_dates()

    if not available:
        return BurnoutRisk(
            bri=0.0,
            tier="healthy",
            tier_label="🟢 Healthy",
            is_meaningful=False,
            days_used=0,
            days_requested=days,
            date_range="no data",
            trajectory_headline="No data available.",
            intervention_advice="",
        )

    if as_of_date_str is None:
        as_of_date_str = available[-1]

    end_dt = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()
    start_dt = end_dt - timedelta(days=days - 1)

    # Build list of dates to analyse (only dates that exist in store)
    all_dates_in_range = []
    d = start_dt
    while d <= end_dt:
        ds = d.strftime("%Y-%m-%d")
        if ds in available:
            all_dates_in_range.append(ds)
        d += timedelta(days=1)

    days_used = len(all_dates_in_range)
    date_range = (
        f"{all_dates_in_range[0]} → {all_dates_in_range[-1]}"
        if all_dates_in_range
        else "no data"
    )

    if days_used < BRI_MIN_DAYS:
        return BurnoutRisk(
            bri=0.0,
            tier="healthy",
            tier_label="🟢 Healthy",
            is_meaningful=False,
            days_used=days_used,
            days_requested=days,
            date_range=date_range,
            trajectory_headline=f"Need ≥ {BRI_MIN_DAYS} days of data (have {days_used}).",
            intervention_advice="",
        )

    # Extract time-series data
    series = _extract_daily_series(all_dates_in_range)
    xs = list(range(days_used))  # [0, 1, 2, ..., N-1]

    def _xs_for(vals: list) -> list[int]:
        """Return only the x positions where the corresponding val is not None."""
        return [xs[i] for i, v in enumerate(vals) if v is not None]

    def _valid(vals: list) -> list[float]:
        return [v for v in vals if v is not None]

    # Compute slopes
    hrv_xs   = _xs_for(series["hrv"])
    hrv_ys   = _valid(series["hrv"])
    hrv_slope = _linear_slope(hrv_xs, hrv_ys)

    sleep_xs  = _xs_for(series["sleep"])
    sleep_ys  = _valid(series["sleep"])
    sleep_slope = _linear_slope(sleep_xs, sleep_ys)

    cls_xs   = _xs_for(series["cls"])
    cls_ys   = _valid(series["cls"])
    cls_slope = _linear_slope(cls_xs, cls_ys)

    fdi_xs   = _xs_for(series["fdi"])
    fdi_ys   = _valid(series["fdi"])
    fdi_slope = _linear_slope(fdi_xs, fdi_ys)

    sdi_xs   = _xs_for(series["sdi"])
    sdi_ys   = _valid(series["sdi"])
    sdi_slope = _linear_slope(sdi_xs, sdi_ys)

    # Convert slopes to components
    comp_hrv   = _slope_to_component(hrv_slope,   HRV_SLOPE_SCALE)
    comp_sleep = _slope_to_component(sleep_slope,  SLEEP_SLOPE_SCALE)
    comp_load  = _slope_to_component(cls_slope,    CLS_SLOPE_SCALE)
    comp_focus = _slope_to_component(fdi_slope,    FDI_SLOPE_SCALE)
    comp_sdi   = _slope_to_component(sdi_slope,    SDI_SLOPE_SCALE)

    components = {
        "hrv":           comp_hrv,
        "sleep":         comp_sleep,
        "load_creep":    comp_load,
        "focus_erosion": comp_focus,
        "social_drain":  comp_sdi,
    }

    component_labels = {
        sig: _component_label(sig, val) for sig, val in components.items()
    }

    # Compute raw BRI
    raw_bri = (
        WEIGHT_HRV          * comp_hrv
        + WEIGHT_SLEEP      * comp_sleep
        + WEIGHT_LOAD_CREEP * comp_load
        + WEIGHT_FOCUS_EROSION * comp_focus
        + WEIGHT_SOCIAL_DRAIN  * comp_sdi
    )
    bri = round(_clamp(raw_bri * 100, 0.0, 100.0), 1)

    tier, tier_label = _tier_from_bri(bri)

    # Find dominant signal
    dominant_signal = max(components, key=lambda k: components[k])

    # Compute trend direction (compare first half vs second half)
    half = days_used // 2
    if half >= 2:
        first_half_indices = list(range(0, half))
        second_half_indices = list(range(half, days_used))

        def _half_bri(indices: list[int]) -> float:
            sub_xs = [xs[i] for i in indices]
            sub_ys_hrv = [series["hrv"][i] for i in indices if series["hrv"][i] is not None]
            if not sub_ys_hrv:
                return bri  # can't determine
            # Just use a simplified approach: compare mean metrics
            sub_cls = [series["cls"][i] for i in indices if series["cls"][i] is not None]
            sub_fdi = [series["fdi"][i] for i in indices if series["fdi"][i] is not None]
            sub_hrv = [series["hrv"][i] for i in indices if series["hrv"][i] is not None]
            cls_avg = sum(sub_cls) / len(sub_cls) if sub_cls else 0
            fdi_avg = sum(sub_fdi) / len(sub_fdi) if sub_fdi else 1
            hrv_avg = sum(sub_hrv) / len(sub_hrv) if sub_hrv else 70
            # Simplified risk score: higher CLS + lower FDI + lower HRV = more risk
            return cls_avg * 0.5 + (1 - fdi_avg) * 0.3 + (1 - min(hrv_avg / 100, 1.0)) * 0.2

        first_risk = _half_bri(first_half_indices)
        second_risk = _half_bri(second_half_indices)
        delta = second_risk - first_risk
        if delta > 0.05:
            trend_direction = "worsening"
        elif delta < -0.05:
            trend_direction = "improving"
        else:
            trend_direction = "stable"
    else:
        trend_direction = "stable"

    headline = _trajectory_headline(bri, tier, dominant_signal, trend_direction, days_used)
    advice = _intervention_advice(tier, dominant_signal)

    return BurnoutRisk(
        bri=bri,
        tier=tier,
        tier_label=tier_label,
        is_meaningful=True,
        days_used=days_used,
        days_requested=days,
        date_range=date_range,
        components=components,
        component_labels=component_labels,
        dominant_signal=dominant_signal,
        trend_direction=trend_direction,
        trajectory_headline=headline,
        intervention_advice=advice,
        hrv_slope=hrv_slope,
        sleep_slope=sleep_slope,
        cls_slope=cls_slope,
        fdi_slope=fdi_slope,
        sdi_slope=sdi_slope,
    )


# ─── Formatting ───────────────────────────────────────────────────────────────

def format_bri_line(bri: "BurnoutRisk") -> str:
    """
    Compact one-liner for morning brief or nightly digest.
    Example: "🔥 BRI 42/100 (Caution) — load creep driving risk"
    """
    if not bri.is_meaningful:
        return ""

    signal_names = {
        "hrv":           "HRV decline",
        "sleep":         "sleep degradation",
        "load_creep":    "load creep",
        "focus_erosion": "focus erosion",
        "social_drain":  "social drain",
    }

    tier_emoji = {
        "healthy":   "✅",
        "watch":     "👀",
        "caution":   "⚠️",
        "high_risk": "🔴",
        "critical":  "🚨",
    }.get(bri.tier, "⚠️")

    signal_label = signal_names.get(bri.dominant_signal, bri.dominant_signal)
    tier_label_clean = bri.tier_label.split(" ", 1)[1]  # strip emoji from tier_label

    if bri.tier == "healthy":
        return f"{tier_emoji} Burnout Risk: {bri.bri:.0f}/100 ({tier_label_clean}) — all trends healthy"
    else:
        return (
            f"{tier_emoji} Burnout Risk: {bri.bri:.0f}/100 ({tier_label_clean})"
            f" — {signal_label} is primary driver"
        )


def format_bri_section(bri: "BurnoutRisk") -> str:
    """
    Full Slack section for weekly summary.
    Returns multi-line markdown with all signal components.
    """
    if not bri.is_meaningful:
        return ""

    tier_emoji = {
        "healthy":   "✅",
        "watch":     "👀",
        "caution":   "⚠️",
        "high_risk": "🔴",
        "critical":  "🚨",
    }.get(bri.tier, "⚠️")

    lines = [
        f"{tier_emoji} *Burnout Risk Index — {bri.bri:.0f}/100 ({bri.tier_label.split(' ', 1)[1]})*",
        f"_{bri.trajectory_headline}_",
        "",
    ]

    # Signal breakdown
    signal_display = [
        ("HRV trend",      "hrv",           f"{bri.hrv_slope:+.2f} ms/day"   if bri.hrv_slope   is not None else "—"),
        ("Sleep quality",  "sleep",         f"{bri.sleep_slope:+.1f} %/day"  if bri.sleep_slope is not None else "—"),
        ("Load creep",     "load_creep",    f"{bri.cls_slope:+.4f} CLS/day"  if bri.cls_slope   is not None else "—"),
        ("Focus erosion",  "focus_erosion", f"{bri.fdi_slope:+.4f} FDI/day"  if bri.fdi_slope   is not None else "—"),
        ("Social drain",   "social_drain",  f"{bri.sdi_slope:+.4f} SDI/day"  if bri.sdi_slope   is not None else "—"),
    ]

    risk_bars = {
        "healthy":   "░",
        "watch":     "▒",
        "caution":   "▓",
        "high_risk": "█",
        "critical":  "█",
    }

    def _comp_bar(c: float) -> str:
        filled = round(c * 10)
        return "█" * filled + "░" * (10 - filled)

    for display_name, key, slope_str in signal_display:
        comp = bri.components.get(key, 0.0)
        bar = _comp_bar(comp)
        label = bri.component_labels.get(key, "—")
        is_dominant = "  ← primary" if key == bri.dominant_signal and bri.tier != "healthy" else ""
        lines.append(f"  {display_name:<15} {bar}  {slope_str:<18}  _{label}_{is_dominant}")

    lines.append("")
    lines.append(f"💡 {bri.intervention_advice}")
    lines.append(f"_Analysis: {bri.date_range} ({bri.days_used} days)_")

    return "\n".join(lines)


def format_bri_terminal(bri: "BurnoutRisk") -> str:
    """
    Colour-formatted terminal output for scripts/report.py.
    """
    if not bri.is_meaningful:
        need = BRI_MIN_DAYS - bri.days_used
        return f"  Burnout Risk Index:  \033[2m(need {need} more days of data)\033[0m"

    colour = {
        "healthy":   "\033[92m",  # green
        "watch":     "\033[93m",  # yellow
        "caution":   "\033[33m",  # orange-ish
        "high_risk": "\033[91m",  # red
        "critical":  "\033[31m",  # bright red
    }.get(bri.tier, "")
    reset = "\033[0m"

    lines = [
        f"\033[1mBurnout Risk Index (BRI)\033[0m",
        f"  Score:   {colour}{bri.bri:.0f}/100{reset}  {bri.tier_label}",
        f"  Trend:   {bri.trend_direction.title()}",
        f"  Driver:  {bri.dominant_signal.replace('_', ' ').title()}",
        f"  Insight: {bri.trajectory_headline}",
        f"  Action:  {bri.intervention_advice}",
        "",
        "  Signal breakdown:",
    ]

    signal_names = {
        "hrv":           "HRV trend",
        "sleep":         "Sleep quality",
        "load_creep":    "Load creep",
        "focus_erosion": "Focus erosion",
        "social_drain":  "Social drain",
    }
    for key, display in signal_names.items():
        comp = bri.components.get(key, 0.0)
        filled = round(comp * 12)
        bar = "█" * filled + "░" * (12 - filled)
        label = bri.component_labels.get(key, "")
        lines.append(f"    {display:<16} {bar}  {label}")

    return "\n".join(lines)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Presence Tracker — Burnout Risk Index")
    parser.add_argument("date", nargs="?", help="As-of date (YYYY-MM-DD, default: latest)")
    parser.add_argument("--days", type=int, default=BRI_DEFAULT_DAYS, help="Lookback days (default: 28)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    bri = compute_burnout_risk(args.date, days=args.days)

    if args.json:
        print(json.dumps(bri.to_dict(), indent=2))
    else:
        print(format_bri_terminal(bri))

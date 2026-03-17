"""
Presence Tracker — CDI Trajectory Forecast (v1)

Answers: *"If I keep this pace, how will my cognitive debt evolve over the next 5 days?"*

The CDI tells David where he is right now — surplus, balanced, loading, fatigued, or critical.
But it says nothing about trajectory: is the debt accumulating, stabilising, or recovering?

The CDI Forecast closes this gap. It projects the CDI forward 3–5 days and answers:
  - "At this pace, you'll reach fatigued tier in ~3 days."
  - "If tomorrow is light, you recover to balanced by Thursday."
  - "You're on track for a stable week — CDI stays in balanced range."

## How it works

### Step 1: Anchor on current state

Read the current CDI and its constituent series from `compute_cdi()`.
The last value in `debt_series` is the current running sum that CDI is derived from.

    running_sum = debt_series[-1]   (from compute_cdi)

### Step 2: Estimate future daily debt deltas

For each upcoming day d in [1 … FORECAST_DAYS]:

    estimated_recovery_signal = mean WHOOP recovery over last N days / 100.0
                                 (clamped to [0.35, 0.95])

    estimated_load_signal = predicted_cls × DEFAULT_ACTIVE_FRACTION
        predicted_cls comes from:
          a) load_forecast.compute_load_forecast(day_str) if calendar is available
          b) avg CLS over last 7 days (personal baseline) when no calendar
          c) CDI trend_5d back-calculated as a fallback

    estimated_debt_delta = estimated_load_signal − estimated_recovery_signal

### Step 3: Project running sum and CDI

For each future day:
    running_sum = clamp(running_sum + estimated_debt_delta, -14, +14)
    projected_cdi = 50 + (running_sum / 14) × 50

### Step 4: Detect tier crossings

Scan the projected series for the first day a tier boundary is crossed:
  - When CDI is currently < 70 (not fatigued): "days until fatigued" (CDI ≥ 70)
  - When CDI is currently > 50 (in debt): "days until balanced" (CDI ≤ 50)
  - When CDI is currently ≤ 50 (balanced/surplus): "days until loading" (CDI ≥ 50)

## Output

    CDIForecast dataclass:
      - today_cdi: float              — current CDI score
      - today_tier: str               — current tier
      - projected_cdis: list[float]   — CDI for days [+1, +2, +3, +4, +5]
      - projected_dates: list[str]    — corresponding dates (YYYY-MM-DD)
      - projected_tiers: list[str]    — tier for each projected day
      - trend_direction: str          — 'worsening' | 'stable' | 'improving'
      - days_to_fatigued: int | None  — days until CDI hits fatigued (or None)
      - days_to_recovery: int | None  — days until CDI drops below 50 (or None)
      - headline: str                 — one-line human summary
      - is_meaningful: bool           — False when not enough history to project
      - recovery_signal_used: float   — average recovery used in projection
      - load_signal_used: float       — average load used in projection

## API

    from analysis.cdi_forecast import compute_cdi_forecast, format_cdi_forecast_line
    from analysis.cdi_forecast import format_cdi_forecast_section

    forecast = compute_cdi_forecast(date_str)
    line    = format_cdi_forecast_line(forecast)    # compact one-liner
    section = format_cdi_forecast_section(forecast) # full Slack section

## Integration

    In morning brief (after CDI block):
        forecast = compute_cdi_forecast(today_date)
        if forecast.is_meaningful:
            lines.append(format_cdi_forecast_line(forecast))

    In daily digest (nightly):
        forecast = compute_cdi_forecast(date_str)
        if forecast.is_meaningful:
            lines.append(format_cdi_forecast_section(forecast))

## CLI

    python3 analysis/cdi_forecast.py                # Today
    python3 analysis/cdi_forecast.py 2026-03-14     # Specific date
    python3 analysis/cdi_forecast.py --json         # JSON output

## Design principles

  - Pure projection — no ML, no black boxes; the formula is auditable
  - Graceful degradation: missing WHOOP / calendar → conservative fallbacks
  - Never raises — returns is_meaningful=False with neutral projections on error
  - Testable: all core logic in pure functions, IO separated

"""

import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import list_available_dates, read_day, read_summary

# ─── Constants ────────────────────────────────────────────────────────────────

# Number of days to project forward
FORECAST_DAYS = 5

# CDI formula constants (must stay in sync with cognitive_debt.py)
CDI_SERIES_CLAMP = 14.0
CDI_SURPLUS_THRESHOLD = 30.0
CDI_BALANCED_MAX = 50.0
CDI_LOADING_MAX = 70.0
CDI_FATIGUED_MAX = 85.0

# Tier-crossing thresholds for "days to X" computation
FATIGUED_THRESHOLD = CDI_LOADING_MAX     # CDI ≥ 70 = fatigued
RECOVERY_THRESHOLD = CDI_BALANCED_MAX   # CDI ≤ 50 = balanced (recovering)
LOADING_THRESHOLD  = CDI_BALANCED_MAX   # CDI ≥ 50 = loading (from surplus)

# Minimum recent days needed to make a meaningful projection
MIN_DAYS_FOR_FORECAST = 3

# How many recent days to average for recovery and load baselines
BASELINE_WINDOW = 7

# Default active fraction when calendar data is unavailable (fraction of
# working windows that are actually cognitively active on a typical day)
DEFAULT_ACTIVE_FRACTION = 0.40

# Minimum / maximum clamps for projected signals
RECOVERY_SIGNAL_MIN = 0.35
RECOVERY_SIGNAL_MAX = 0.95


# ─── Data class ───────────────────────────────────────────────────────────────

@dataclass
class CDIForecast:
    """
    Projected Cognitive Debt Index over the next FORECAST_DAYS days.

    Attributes:
        today_cdi:           Current CDI score (0–100)
        today_tier:          Current tier name
        projected_cdis:      CDI for each of the next FORECAST_DAYS days
        projected_dates:     Corresponding date strings (YYYY-MM-DD)
        projected_tiers:     Tier for each projected day
        trend_direction:     'worsening' | 'stable' | 'improving'
        days_to_fatigued:    Days until CDI first hits ≥70, or None
        days_to_recovery:    Days until CDI first drops to ≤50, or None
        headline:            One-line human summary
        is_meaningful:       False when insufficient history to project
        recovery_signal_used: Average recovery_score/100 used in projection
        load_signal_used:    Average CLS×active_fraction used in projection
        days_of_history:     History days that informed the projection
    """
    today_cdi: float
    today_tier: str
    projected_cdis: list
    projected_dates: list
    projected_tiers: list
    trend_direction: str
    days_to_fatigued: Optional[int]
    days_to_recovery: Optional[int]
    headline: str
    is_meaningful: bool
    recovery_signal_used: float
    load_signal_used: float
    days_of_history: int

    def to_dict(self) -> dict:
        return {
            "today_cdi": self.today_cdi,
            "today_tier": self.today_tier,
            "projected_cdis": self.projected_cdis,
            "projected_dates": self.projected_dates,
            "projected_tiers": self.projected_tiers,
            "trend_direction": self.trend_direction,
            "days_to_fatigued": self.days_to_fatigued,
            "days_to_recovery": self.days_to_recovery,
            "headline": self.headline,
            "is_meaningful": self.is_meaningful,
            "recovery_signal_used": self.recovery_signal_used,
            "load_signal_used": self.load_signal_used,
            "days_of_history": self.days_of_history,
        }


# ─── Tier helper ─────────────────────────────────────────────────────────────

def _cdi_tier(cdi: float) -> str:
    """Map CDI score to tier name."""
    if cdi < CDI_SURPLUS_THRESHOLD:
        return "surplus"
    elif cdi <= CDI_BALANCED_MAX:
        return "balanced"
    elif cdi <= CDI_LOADING_MAX:
        return "loading"
    elif cdi <= CDI_FATIGUED_MAX:
        return "fatigued"
    else:
        return "critical"


# ─── Signal helpers (pure functions) ─────────────────────────────────────────

def _mean(values: list) -> Optional[float]:
    """Return mean of a list, or None if empty."""
    if not values:
        return None
    return sum(values) / len(values)


def _extract_recovery_signal(rolling: dict, recent_dates: list[str]) -> float:
    """
    Compute the average recovery signal (WHOOP recovery / 100) from recent days.

    Falls back to 0.60 (neutral slightly-positive) when no WHOOP data.
    """
    values = []
    days_data = rolling.get("days", {})
    for d in recent_dates:
        day = days_data.get(d, {})
        whoop = day.get("whoop") or {}
        rec = whoop.get("recovery_score")
        if rec is not None:
            values.append(float(rec) / 100.0)

    if not values:
        return 0.60  # neutral-ish fallback

    mean_rec = sum(values) / len(values)
    return max(RECOVERY_SIGNAL_MIN, min(RECOVERY_SIGNAL_MAX, mean_rec))


def _extract_load_signal(rolling: dict, recent_dates: list[str]) -> float:
    """
    Compute the average load signal (avg_cls × active_fraction) from recent days.

    Mirrors the formula in cognitive_debt._debt_delta_for_day().
    Falls back to 0.15 (light-load default) when no data.
    """
    values = []
    days_data = rolling.get("days", {})
    _WORKING_WINDOWS = 60

    for d in recent_dates:
        day = days_data.get(d, {})
        metrics_avg = day.get("metrics_avg") or {}
        avg_cls = metrics_avg.get("cognitive_load_score")
        if avg_cls is None:
            continue

        focus_quality = day.get("focus_quality") or {}
        active_windows = focus_quality.get("active_windows")
        if active_windows is not None and active_windows > 0:
            active_fraction = min(1.0, active_windows / _WORKING_WINDOWS)
        else:
            active_fraction = DEFAULT_ACTIVE_FRACTION

        values.append(avg_cls * active_fraction)

    if not values:
        return 0.15  # conservative low-load fallback

    return sum(values) / len(values)


def _get_load_forecast_for_date(date_str: str) -> Optional[float]:
    """
    Try to get a predicted CLS for date_str from load_forecast.

    Returns predicted_cls, or None if unavailable.
    Only imports load_forecast here to avoid circular imports.
    """
    try:
        from analysis.load_forecast import compute_load_forecast
        forecast = compute_load_forecast(date_str)
        if forecast and forecast.is_meaningful:
            return forecast.predicted_cls
    except Exception:
        pass
    return None


def _project_cdi(
    current_running_sum: float,
    debt_delta_per_day: float,
    base_date: datetime,
    forecast_days: int = FORECAST_DAYS,
) -> tuple[list[float], list[str], list[str]]:
    """
    Project CDI forward by applying the estimated daily debt delta.

    Args:
        current_running_sum: The CDI running sum (debt_series[-1]) right now.
        debt_delta_per_day:  Estimated daily debt delta (positive = accumulating debt).
        base_date:           The date of the current running sum.
        forecast_days:       How many days ahead to project.

    Returns:
        (projected_cdis, projected_dates, projected_tiers)
    """
    cdis = []
    dates = []
    tiers = []

    running_sum = current_running_sum

    for i in range(1, forecast_days + 1):
        running_sum = max(-CDI_SERIES_CLAMP, min(CDI_SERIES_CLAMP, running_sum + debt_delta_per_day))
        cdi = 50.0 + (running_sum / CDI_SERIES_CLAMP) * 50.0
        cdi = round(max(0.0, min(100.0, cdi)), 1)

        day_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        cdis.append(cdi)
        dates.append(day_str)
        tiers.append(_cdi_tier(cdi))

    return cdis, dates, tiers


def _detect_days_to_fatigued(today_cdi: float, projected_cdis: list[float]) -> Optional[int]:
    """
    Return number of days until CDI first reaches 'fatigued' (≥70), or None.

    Only meaningful when today_cdi < 70 (i.e., not already fatigued).
    """
    if today_cdi >= FATIGUED_THRESHOLD:
        return None  # already fatigued
    for i, cdi in enumerate(projected_cdis, start=1):
        if cdi >= FATIGUED_THRESHOLD:
            return i
    return None


def _detect_days_to_recovery(today_cdi: float, projected_cdis: list[float]) -> Optional[int]:
    """
    Return number of days until CDI first drops to 'balanced' or below (≤50), or None.

    Only meaningful when today_cdi > 50 (i.e., currently in debt).
    """
    if today_cdi <= RECOVERY_THRESHOLD:
        return None  # already recovered
    for i, cdi in enumerate(projected_cdis, start=1):
        if cdi <= RECOVERY_THRESHOLD:
            return i
    return None


def _detect_trend(today_cdi: float, projected_cdis: list[float]) -> str:
    """
    Classify whether the CDI trajectory is worsening, stable, or improving.

    - worsening: CDI rises by ≥ 5 over the projection window
    - improving: CDI falls by ≥ 5 over the projection window
    - stable: within ±5
    """
    if not projected_cdis:
        return "stable"
    end_cdi = projected_cdis[-1]
    delta = end_cdi - today_cdi
    if delta >= 5.0:
        return "worsening"
    elif delta <= -5.0:
        return "improving"
    else:
        return "stable"


def _build_headline(
    today_cdi: float,
    today_tier: str,
    trend_direction: str,
    days_to_fatigued: Optional[int],
    days_to_recovery: Optional[int],
    projected_cdis: list[float],
) -> str:
    """Build a one-sentence human summary of the CDI forecast."""
    end_cdi = round(projected_cdis[-1]) if projected_cdis else round(today_cdi)
    horizon = len(projected_cdis)

    if today_tier in ("fatigued", "critical"):
        if days_to_recovery is not None:
            return (
                f"At this pace, CDI recovers to balanced in ~{days_to_recovery} day"
                + ("s" if days_to_recovery != 1 else "")
                + f" (today: {round(today_cdi)}/100)."
            )
        elif trend_direction == "improving":
            return (
                f"Debt is easing but still high — CDI drops from {round(today_cdi)} "
                f"to ~{end_cdi} over the next {horizon} days."
            )
        elif trend_direction == "stable":
            return (
                f"CDI stabilising at {round(today_cdi)}/100 ({today_tier}) — "
                f"needs a genuine recovery day to reset."
            )
        else:
            return (
                f"⚠️ CDI rising from {round(today_cdi)} to ~{end_cdi} over {horizon} days — "
                f"cognitive debt is compounding."
            )

    elif today_tier == "loading":
        if days_to_fatigued is not None:
            return (
                f"At this pace, CDI reaches fatigued in ~{days_to_fatigued} day"
                + ("s" if days_to_fatigued != 1 else "")
                + f" — consider lightening load soon."
            )
        elif trend_direction == "improving":
            return (
                f"CDI trending down from {round(today_cdi)} to ~{end_cdi} — "
                f"good recovery arc."
            )
        elif trend_direction == "stable":
            return (
                f"CDI holding steady at {round(today_cdi)}/100 (loading) — "
                f"sustainable but not recovering."
            )
        else:
            return (
                f"CDI drifting higher from {round(today_cdi)} toward ~{end_cdi} — "
                f"watch for accumulation."
            )

    else:  # surplus or balanced
        if trend_direction == "worsening":
            return (
                f"CDI is healthy now ({round(today_cdi)}/100) but trending up — "
                f"projected ~{end_cdi} in {horizon} days."
            )
        elif trend_direction == "stable":
            return f"CDI stable at {round(today_cdi)}/100 ({today_tier}) — sustainable pace."
        else:
            return (
                f"CDI improving from {round(today_cdi)} to ~{end_cdi} over {horizon} days — "
                f"building cognitive surplus."
            )


# ─── Main computation ─────────────────────────────────────────────────────────

def compute_cdi_forecast(date_str: str = None) -> CDIForecast:
    """
    Compute the CDI trajectory forecast for the next FORECAST_DAYS days.

    Args:
        date_str: The anchor date (YYYY-MM-DD). Defaults to today.

    Returns:
        CDIForecast dataclass. Never raises — returns is_meaningful=False on error.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    _NEUTRAL = CDIForecast(
        today_cdi=50.0,
        today_tier="balanced",
        projected_cdis=[50.0] * FORECAST_DAYS,
        projected_dates=[
            (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, FORECAST_DAYS + 1)
        ],
        projected_tiers=["balanced"] * FORECAST_DAYS,
        trend_direction="stable",
        days_to_fatigued=None,
        days_to_recovery=None,
        headline="Insufficient data for CDI trajectory forecast.",
        is_meaningful=False,
        recovery_signal_used=0.60,
        load_signal_used=0.15,
        days_of_history=0,
    )

    try:
        from analysis.cognitive_debt import compute_cdi

        # ── Step 1: Get current CDI ──────────────────────────────────────
        debt = compute_cdi(date_str)

        if not debt.is_meaningful:
            return _NEUTRAL

        today_cdi = debt.cdi
        today_tier = debt.tier

        # Current running sum drives the projection
        current_running_sum = debt.debt_series[-1] if debt.debt_series else 0.0

        # ── Step 2: Compute signal baselines from recent history ──────────
        rolling = read_summary()
        available_dates = list_available_dates()

        today_dt = datetime.strptime(date_str, "%Y-%m-%d")
        recent_dates = [
            d for d in available_dates
            if d < date_str
        ][-BASELINE_WINDOW:]

        days_of_history = len(recent_dates)
        if days_of_history < MIN_DAYS_FOR_FORECAST:
            return _NEUTRAL

        recovery_signal = _extract_recovery_signal(rolling, recent_dates)
        load_signal = _extract_load_signal(rolling, recent_dates)

        # ── Step 3: Try to get calendar-informed load estimates for each future day ──
        # We project a per-day delta, trying load_forecast for each day.
        # If load_forecast is unavailable, we use the historical average.
        future_deltas = []
        for i in range(1, FORECAST_DAYS + 1):
            future_date = (today_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            predicted_cls = _get_load_forecast_for_date(future_date)
            if predicted_cls is not None:
                # Use predicted CLS × DEFAULT_ACTIVE_FRACTION for load signal
                day_load = predicted_cls * DEFAULT_ACTIVE_FRACTION
            else:
                # Fall back to historical average load signal
                day_load = load_signal
            future_deltas.append(day_load - recovery_signal)

        # ── Step 4: Project CDI forward ───────────────────────────────────
        projected_cdis = []
        projected_dates = []
        projected_tiers = []

        running_sum = current_running_sum
        for i, delta in enumerate(future_deltas, start=1):
            running_sum = max(-CDI_SERIES_CLAMP, min(CDI_SERIES_CLAMP, running_sum + delta))
            cdi_val = 50.0 + (running_sum / CDI_SERIES_CLAMP) * 50.0
            cdi_val = round(max(0.0, min(100.0, cdi_val)), 1)
            day_str = (today_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            projected_cdis.append(cdi_val)
            projected_dates.append(day_str)
            projected_tiers.append(_cdi_tier(cdi_val))

        # ── Step 5: Detect crossings and build headline ───────────────────
        trend_direction = _detect_trend(today_cdi, projected_cdis)
        days_to_fatigued = _detect_days_to_fatigued(today_cdi, projected_cdis)
        days_to_recovery = _detect_days_to_recovery(today_cdi, projected_cdis)

        headline = _build_headline(
            today_cdi, today_tier, trend_direction,
            days_to_fatigued, days_to_recovery, projected_cdis,
        )

        return CDIForecast(
            today_cdi=today_cdi,
            today_tier=today_tier,
            projected_cdis=projected_cdis,
            projected_dates=projected_dates,
            projected_tiers=projected_tiers,
            trend_direction=trend_direction,
            days_to_fatigued=days_to_fatigued,
            days_to_recovery=days_to_recovery,
            headline=headline,
            is_meaningful=True,
            recovery_signal_used=round(recovery_signal, 3),
            load_signal_used=round(load_signal, 3),
            days_of_history=days_of_history,
        )

    except Exception:
        return _NEUTRAL


# ─── Formatting ──────────────────────────────────────────────────────────────

_TIER_EMOJI = {
    "surplus":  "🟢",
    "balanced": "🟡",
    "loading":  "🟠",
    "fatigued": "🔴",
    "critical": "🚨",
}

_TREND_ARROW = {
    "worsening": "↑",
    "stable":    "→",
    "improving": "↓",
}


def _sparkline(cdis: list[float]) -> str:
    """
    Render a 5-char Unicode sparkline for the projected CDI series.
    Each position: ▁ (low, good) → ▅ (mid) → █ (high, bad)
    Bounded to [0, 100].
    """
    bars = " ▁▂▃▄▅▆▇█"
    result = []
    for v in cdis:
        idx = int(v / 100.0 * (len(bars) - 1))
        idx = max(0, min(len(bars) - 1, idx))
        result.append(bars[idx])
    return "".join(result)


def format_cdi_forecast_line(forecast: CDIForecast) -> str:
    """
    Return a compact one-line CDI forecast for use in morning brief.

    Example:
        📈 CDI Forecast: worsening — reaches fatigued in ~3 days [▁▂▃▄▅]
        📉 CDI Forecast: improving — recovers to balanced in ~2 days [▅▄▃▂▁]
        ➡️ CDI Forecast: stable — CDI stays in loading range [▃▃▃▃▃]
    """
    if not forecast.is_meaningful:
        return ""

    arrow = _TREND_ARROW.get(forecast.trend_direction, "→")
    sparkle = _sparkline(forecast.projected_cdis)

    if forecast.days_to_fatigued is not None:
        urgency = f" — fatigued in ~{forecast.days_to_fatigued}d"
    elif forecast.days_to_recovery is not None:
        urgency = f" — balanced in ~{forecast.days_to_recovery}d"
    else:
        urgency = ""

    tier_emoji = _TIER_EMOJI.get(forecast.today_tier, "⬜")
    return (
        f"{tier_emoji} CDI trajectory: {forecast.trend_direction} {arrow}"
        f"{urgency} [{sparkle}]"
    )


def format_cdi_forecast_section(forecast: CDIForecast) -> str:
    """
    Return a multi-line Slack-formatted CDI forecast section.

    Example:
        *CDI Trajectory Forecast* (next 5 days)
        ├ Today:   🟠 Loading (CDI 63)
        ├ +1 day:  🟠 Loading (65)
        ├ +2 days: 🔴 Fatigued (71)  ← tier crossed
        ├ +3 days: 🔴 Fatigued (73)
        ├ +4 days: 🔴 Fatigued (75)
        └ +5 days: 🔴 Fatigued (77)
        At this pace, CDI reaches fatigued in ~2 days — consider lightening load soon.
    """
    if not forecast.is_meaningful:
        return ""

    lines = ["*CDI Trajectory Forecast* (next 5 days)"]
    prev_tier = forecast.today_tier

    tree_chars = ["├"] * (len(forecast.projected_cdis) - 1) + ["└"]

    for i, (cdi, date_str, tier, char) in enumerate(zip(
        forecast.projected_cdis, forecast.projected_dates,
        forecast.projected_tiers, tree_chars
    )):
        emoji = _TIER_EMOJI.get(tier, "⬜")
        crossed = " ← tier crossed" if tier != prev_tier and i > 0 else ""
        if crossed:
            prev_tier = tier
        label = tier.capitalize()
        label_day = f"+{i+1}d"
        lines.append(f"  {char} {label_day}: {emoji} {label} ({round(cdi)}){crossed}")

    lines.append(f"_{forecast.headline}_")
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CDI Trajectory Forecast")
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD, default: today)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    forecast = compute_cdi_forecast(date_str)

    if args.json:
        print(json.dumps(forecast.to_dict(), indent=2))
        return

    if not forecast.is_meaningful:
        print("⚠️  Insufficient data for CDI forecast (need ≥ 3 days of history).")
        return

    print(f"\n🧠 CDI Trajectory Forecast — {date_str}")
    print(f"   Today: {_TIER_EMOJI.get(forecast.today_tier, '')} {forecast.today_tier.capitalize()} (CDI {round(forecast.today_cdi)}/100)")
    print(f"   Trend: {forecast.trend_direction} {_TREND_ARROW.get(forecast.trend_direction, '')}")
    print(f"   History: {forecast.days_of_history} days | Avg recovery: {forecast.recovery_signal_used:.0%} | Avg load: {forecast.load_signal_used:.2f}")
    print()
    print(format_cdi_forecast_section(forecast))
    print()
    if forecast.days_to_fatigued is not None:
        print(f"⚠️  Reaches fatigued in ~{forecast.days_to_fatigued} day(s)")
    elif forecast.days_to_recovery is not None:
        print(f"✅  Recovers to balanced in ~{forecast.days_to_recovery} day(s)")
    print()
    print(f"One-liner: {format_cdi_forecast_line(forecast)}")


if __name__ == "__main__":
    main()

"""
Presence Tracker — Cognitive Debt Index (CDI)

The CDI answers: "How much accumulated cognitive fatigue is David carrying?"

This is the multi-day complement to single-day metrics.  CLS tells you how hard
today was.  RAS tells you if today was appropriate for your physiology.  CDI tells
you whether you've been running a sustained deficit — and by how much.

## Concept

Think of it like a battery:
  - Recovery day (high WHOOP recovery, low CLS) → charges the battery
  - Hard day (high CLS, low recovery) → drains the battery
  - The CDI is the charge level: 100 = fully charged, 0 = dangerously depleted

## Formula

For each day d in the lookback window:

    load_signal(d) = avg_cls(d)  × active_window_fraction(d)
                     ─────────────────────────────────────────
                     (fraction of working windows with activity)

    recovery_signal(d) = WHOOP recovery_score(d) / 100.0
                          (normalized 0→1)

    debt_delta(d) = load_signal(d) - recovery_signal(d)
                    positive = accumulated more debt
                    negative = paid off debt (recovery day)

    debt_series = running sum of debt_deltas, clamped to [-14, +14]
                  (series bounded so a single catastrophic day can't max out CDI)

    CDI = 50  -  (debt_series[-1] / 14) × 50
          range 0–100, centred at 50 (neutral baseline)
          CDI > 70 = significant debt (fatigue accumulating)
          CDI > 85 = high debt (burnout risk)
          CDI < 30 = well recovered

The centre (50) represents equilibrium: days where load matched recovery.
Above 50 = net load > net recovery → fatigue accumulating.
Below 50 = net recovery > net load → energy surplus.

## Why this matters

A single bad day is noise.  Four days of CLS > 0.65 with WHOOP < 60% means
something systemic is happening — but the existing metrics only show you
today's snapshot.  CDI makes the accumulation visible:

  "Your CDI is 78/100 (high debt).  You've been running above capacity for
   5 of the last 7 days.  Consider protecting tomorrow."

This is the cognitive equivalent of WHOOP's strain-vs-recovery balance.

## API

    compute_cdi(end_date_str, days=14) → CognitiveDebt
    format_cdi_line(debt) → str  (compact one-line summary for digest/brief)

## Tiers

    CDI < 30 : 'surplus'   — well recovered, can take on load
    CDI 30-50: 'balanced'  — sustainable pace
    CDI 50-70: 'loading'   — load exceeding recovery, watch for accumulation
    CDI 70-85: 'fatigued'  — significant debt, reduce load
    CDI > 85 : 'critical'  — high burnout risk, protect recovery aggressively

"""

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Constants ────────────────────────────────────────────────────────────────

# Lookback window (days)
CDI_DEFAULT_DAYS = 14

# Debt series bounds: clamped so one extreme day can't dominate
CDI_SERIES_CLAMP = 14.0

# Minimum days of data before CDI is considered meaningful
CDI_MIN_DAYS = 3

# Alert thresholds
CDI_SURPLUS_THRESHOLD = 30
CDI_BALANCED_MAX = 50
CDI_LOADING_MAX = 70
CDI_FATIGUED_MAX = 85
# > CDI_FATIGUED_MAX → 'critical'


# ─── Data class ───────────────────────────────────────────────────────────────

@dataclass
class CognitiveDebt:
    """
    Computed Cognitive Debt Index for a given date.

    Attributes:
        cdi: 0–100 score (50 = neutral, > 50 = more fatigue than recovery)
        tier: 'surplus' | 'balanced' | 'loading' | 'fatigued' | 'critical'
        debt_delta_today: today's contribution (positive = added debt, negative = paid off)
        trend_5d: average debt_delta over last 5 days (positive = trending toward fatigue)
        days_in_deficit: count of days in the lookback window where debt_delta > 0
        days_in_surplus: count of days in lookback window where debt_delta < 0
        days_used: actual number of days that contributed to this computation
        is_meaningful: True when days_used ≥ CDI_MIN_DAYS
        end_date: the date this was computed for
        debt_series: raw debt delta values (oldest first, up to days_used entries)
    """
    cdi: float
    tier: str
    debt_delta_today: Optional[float]
    trend_5d: Optional[float]
    days_in_deficit: int
    days_in_surplus: int
    days_used: int
    is_meaningful: bool
    end_date: str
    debt_series: list


# ─── CDI computation ──────────────────────────────────────────────────────────

def _debt_delta_for_day(summary: dict) -> Optional[float]:
    """
    Compute the debt delta for a single day from its daily summary.

    Returns positive values when load > recovery, negative when recovery > load.
    Returns None when insufficient data is available.

    debt_delta = load_signal - recovery_signal

    load_signal:
      - avg_cls × active_fraction (how much of the working day was cognitively loaded?)
      - active_fraction = active_windows / total_working_windows (from focus_quality)
      - Falls back to avg_cls alone when active_fraction unavailable

    recovery_signal:
      - WHOOP recovery_score / 100.0 (normalized 0→1)
      - Falls back to 0.5 (neutral) when WHOOP unavailable
    """
    if not summary:
        return None

    # ── Load signal ──────────────────────────────────────────────────────
    metrics_avg = summary.get("metrics_avg") or {}
    avg_cls = metrics_avg.get("cognitive_load_score")

    if avg_cls is None:
        return None

    # Weight by active fraction (idle hours shouldn't register as loaded)
    focus_quality = summary.get("focus_quality") or {}
    active_windows = focus_quality.get("active_windows")
    # Total working-hour windows = 15 hours × 4 per hour = 60
    _WORKING_WINDOWS = 60
    if active_windows is not None and active_windows > 0:
        active_fraction = min(1.0, active_windows / _WORKING_WINDOWS)
    else:
        # No focus quality data — fall back to unweighted CLS
        active_fraction = 1.0

    load_signal = avg_cls * active_fraction

    # ── Recovery signal ───────────────────────────────────────────────────
    whoop = summary.get("whoop") or {}
    recovery_score = whoop.get("recovery_score")

    if recovery_score is not None:
        recovery_signal = float(recovery_score) / 100.0
    else:
        # No WHOOP data — use neutral 0.5 so we don't bias the series
        recovery_signal = 0.5

    return round(load_signal - recovery_signal, 4)


def compute_cdi(end_date_str: str, days: int = CDI_DEFAULT_DAYS) -> CognitiveDebt:
    """
    Compute the Cognitive Debt Index for the given end date.

    Reads from the rolling summary store — no JSONL parsing required.
    Gracefully handles missing data and sparse history.

    Args:
        end_date_str: Date to compute CDI for (YYYY-MM-DD). Usually today.
        days: Lookback window in days (default: 14).

    Returns:
        CognitiveDebt with cdi score, tier, and diagnostics.
        Never raises — returns a safe 'balanced' CDI if anything fails.
    """
    try:
        from engine.store import read_summary as _read_summary

        rolling = _read_summary()
        all_days = rolling.get("days", {})

        # Build the date window (end_date inclusive, going back `days` days)
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        date_window = [
            (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(days - 1, -1, -1)
        ]

        # Compute per-day deltas (None for missing days)
        deltas: list[Optional[float]] = []
        for d in date_window:
            summary = all_days.get(d)
            if summary:
                delta = _debt_delta_for_day(summary)
                deltas.append(delta)
            # Missing days are skipped (not counted as zeros — absent data ≠ rest day)

        # Filter to actual values only
        valid_deltas = [d for d in deltas if d is not None]
        days_used = len(valid_deltas)

        if days_used == 0:
            return CognitiveDebt(
                cdi=50.0,
                tier="balanced",
                debt_delta_today=None,
                trend_5d=None,
                days_in_deficit=0,
                days_in_surplus=0,
                days_used=0,
                is_meaningful=False,
                end_date=end_date_str,
                debt_series=[],
            )

        # Running sum of debt deltas (represents accumulated fatigue)
        running_sum = 0.0
        debt_series = []
        for delta in valid_deltas:
            running_sum = max(-CDI_SERIES_CLAMP, min(CDI_SERIES_CLAMP, running_sum + delta))
            debt_series.append(round(running_sum, 4))

        # CDI: 50 (neutral) ± 50 (max), centred so 50 = balanced
        # running_sum > 0 → more debt (CDI > 50)
        # running_sum < 0 → more surplus (CDI < 50)
        final_debt = debt_series[-1] if debt_series else 0.0
        cdi = 50.0 + (final_debt / CDI_SERIES_CLAMP) * 50.0
        cdi = round(max(0.0, min(100.0, cdi)), 1)

        # Tier
        if cdi < CDI_SURPLUS_THRESHOLD:
            tier = "surplus"
        elif cdi <= CDI_BALANCED_MAX:
            tier = "balanced"
        elif cdi <= CDI_LOADING_MAX:
            tier = "loading"
        elif cdi <= CDI_FATIGUED_MAX:
            tier = "fatigued"
        else:
            tier = "critical"

        # Diagnostics
        debt_delta_today = valid_deltas[-1] if valid_deltas else None
        days_in_deficit = sum(1 for d in valid_deltas if d > 0)
        days_in_surplus = sum(1 for d in valid_deltas if d < 0)

        # 5-day trend (average delta over last 5 valid days)
        recent_5 = valid_deltas[-5:]
        trend_5d = round(sum(recent_5) / len(recent_5), 4) if recent_5 else None

        is_meaningful = days_used >= CDI_MIN_DAYS

        return CognitiveDebt(
            cdi=cdi,
            tier=tier,
            debt_delta_today=debt_delta_today,
            trend_5d=trend_5d,
            days_in_deficit=days_in_deficit,
            days_in_surplus=days_in_surplus,
            days_used=days_used,
            is_meaningful=is_meaningful,
            end_date=end_date_str,
            debt_series=debt_series,
        )

    except Exception:
        # Never crash — return a safe neutral value
        return CognitiveDebt(
            cdi=50.0,
            tier="balanced",
            debt_delta_today=None,
            trend_5d=None,
            days_in_deficit=0,
            days_in_surplus=0,
            days_used=0,
            is_meaningful=False,
            end_date=end_date_str,
            debt_series=[],
        )


# ─── Formatting ───────────────────────────────────────────────────────────────

_TIER_EMOJI = {
    "surplus":  "🟢",
    "balanced": "🟡",
    "loading":  "🟠",
    "fatigued": "🔴",
    "critical": "🚨",
}

_TIER_LABEL = {
    "surplus":  "Surplus",
    "balanced": "Balanced",
    "loading":  "Loading",
    "fatigued": "Fatigued",
    "critical": "Critical",
}


def format_cdi_line(debt: CognitiveDebt) -> str:
    """
    Format a compact one-line CDI summary for use in morning brief or daily digest.

    Examples:
        "🟢 CDI 24/100 — Surplus (well recovered, energy available)"
        "🟠 CDI 63/100 — Loading (5 deficit days in 14d, trend ↑)"
        "🔴 CDI 79/100 — Fatigued (reduce load, protect recovery)"

    Returns empty string if not meaningful (< MIN_DAYS days of data).
    """
    if not debt.is_meaningful:
        return ""

    emoji = _TIER_EMOJI.get(debt.tier, "⚪")
    label = _TIER_LABEL.get(debt.tier, debt.tier.capitalize())
    cdi_str = f"CDI {debt.cdi:.0f}/100"

    # Build detail clause
    detail_parts = []
    if debt.days_in_deficit > 0 and debt.days_used > 0:
        detail_parts.append(
            f"{debt.days_in_deficit} deficit day{'s' if debt.days_in_deficit != 1 else ''}"
            f" in {debt.days_used}d"
        )

    # Trend arrow
    if debt.trend_5d is not None and abs(debt.trend_5d) > 0.02:
        arrow = "↑ fatigue" if debt.trend_5d > 0 else "↓ recovering"
        detail_parts.append(f"trend {arrow}")

    # Tier-specific message
    tier_msg = {
        "surplus":  "energy available, good time to take on load",
        "balanced": "sustainable pace",
        "loading":  "load > recovery, watch accumulation",
        "fatigued": "reduce load, protect recovery",
        "critical": "high burnout risk, rest required",
    }.get(debt.tier, "")

    if detail_parts:
        detail = f"({', '.join(detail_parts)})"
    elif tier_msg:
        detail = f"({tier_msg})"
    else:
        detail = ""

    parts = [emoji, cdi_str, "—", label]
    if detail:
        parts.append(detail)
    return " ".join(parts)


def format_cdi_alert(debt: CognitiveDebt) -> str:
    """
    Format a standalone alert message for high-debt conditions.

    Returns empty string if debt is not fatigued/critical.
    Intended for use by the anomaly alerts module.
    """
    if not debt.is_meaningful or debt.tier not in ("fatigued", "critical"):
        return ""

    emoji = _TIER_EMOJI[debt.tier]
    deficit_days = debt.days_in_deficit

    msg_parts = [
        f"{emoji} *Cognitive Debt Alert* — CDI {debt.cdi:.0f}/100 ({debt.tier})",
        "",
    ]

    if debt.tier == "critical":
        msg_parts.append(
            f"You've accumulated significant cognitive debt over the last "
            f"{debt.days_used} days ({deficit_days} deficit days). "
            f"Physiological recovery is falling behind cognitive load. "
            f"This is the mental-load equivalent of a WHOOP red day: rest is required."
        )
    else:
        msg_parts.append(
            f"Cognitive load has exceeded recovery for {deficit_days} of the last "
            f"{debt.days_used} days. Fatigue is accumulating. "
            f"Protect tomorrow's recovery — consider reducing meeting density or "
            f"deferring high-demand work."
        )

    if debt.trend_5d is not None and debt.trend_5d > 0.05:
        msg_parts.append(
            f"\nThe 5-day trend is still moving toward higher debt — this is not "
            f"a one-day spike."
        )

    return "\n".join(msg_parts)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Show Cognitive Debt Index for a given date"
    )
    parser.add_argument(
        "date", nargs="?",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--days", type=int, default=CDI_DEFAULT_DAYS,
        help=f"Lookback window in days (default: {CDI_DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    debt = compute_cdi(args.date, days=args.days)

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(debt), indent=2))
        return

    print(f"\nCognitive Debt Index — {args.date}")
    print(f"  CDI:        {debt.cdi:.1f} / 100")
    print(f"  Tier:       {_TIER_EMOJI.get(debt.tier, '')} {debt.tier}")
    print(f"  Days used:  {debt.days_used}  (need {CDI_MIN_DAYS} for meaningful)")
    print(f"  Deficit days: {debt.days_in_deficit} / {debt.days_used}")
    print(f"  Surplus days: {debt.days_in_surplus} / {debt.days_used}")
    if debt.debt_delta_today is not None:
        sign = "+" if debt.debt_delta_today >= 0 else ""
        print(f"  Today's delta: {sign}{debt.debt_delta_today:.4f}")
    if debt.trend_5d is not None:
        sign = "+" if debt.trend_5d >= 0 else ""
        print(f"  5-day trend:   {sign}{debt.trend_5d:.4f}")
    print()
    line = format_cdi_line(debt)
    if line:
        print(f"  {line}")
    alert = format_cdi_alert(debt)
    if alert:
        print(f"\n{alert}")
    print()


if __name__ == "__main__":
    main()

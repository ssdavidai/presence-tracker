"""
Presence Tracker — Personal Baseline Module

Computes David's personal physiological and cognitive baselines from
accumulated JSONL data.  Used to replace hardcoded population-norm thresholds
with percentile-anchored personal thresholds.

## Why this matters

The morning brief's `_readiness_tier()` originally used:
  - HRV < 45ms → "stressed"  (population median; useless for David whose HRV is ~79ms)
  - Recovery ≥ 80% → "peak"  (WHOOP's fixed population threshold)

These work fine for a population study.  They're wrong for an individual.
If David's resting HRV is 79ms, then 60ms IS stressed for him even though it's
above the population "stressed" threshold.  Conversely, if his average recovery
is 72%, then 80% is a great day for him — but maybe his personal "peak" is 85%.

Personal thresholds are derived from percentiles of his own historical data:
  - HRV low threshold = 20th percentile of David's HRV values
  - HRV high threshold = 80th percentile
  - Recovery tiers at 20th / 40th / 60th / 80th percentiles
  - CLS baseline = mean + std of his cognitive load across all active windows

When fewer than MIN_DAYS_FOR_PERSONAL_THRESHOLDS days are available, the module
falls back gracefully to the original population-norm thresholds so early operation
is unaffected.

## API

    get_personal_baseline() → PersonalBaseline (named dataclass)
    is_hrv_low(hrv, baseline) → bool
    readiness_tier_personal(recovery, hrv, baseline) → str

    Baseline dataclass fields:
      hrv_mean, hrv_std, hrv_p20, hrv_p80        — from historical data
      recovery_mean, recovery_p20, recovery_p40,
      recovery_p60, recovery_p80                  — from historical data
      cls_mean, cls_std                            — all-time avg load
      days_of_data                                 — how many days contributed
      is_personal                                  — True when personal thresholds used

All functions degrade gracefully to population norms when data is insufficient.
"""

import sys
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Constants ─────────────────────────────────────────────────────────────────

# Minimum days of data before personal baselines are trusted over population norms.
# Below this threshold, population norms are used and is_personal=False.
MIN_DAYS_FOR_PERSONAL_THRESHOLDS = 14

# Population-norm fallbacks (same as original morning_brief.py values)
POPULATION_HRV_LOW = 45.0        # ms — well below population median
POPULATION_RECOVERY_PEAK = 80.0  # % — WHOOP's "green" threshold
POPULATION_RECOVERY_GOOD = 67.0
POPULATION_RECOVERY_MODERATE = 50.0
POPULATION_RECOVERY_LOW = 33.0


# ─── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class PersonalBaseline:
    """
    David's personal physiological and cognitive baselines.

    All threshold values are derived from his historical data when
    is_personal=True.  When is_personal=False, the values are population-norm
    fallbacks and should be treated as rough guidance only.

    HRV thresholds use the 20th and 80th percentiles:
      - hrv_p20 = low threshold: below this, HRV is "low" relative to personal norm
      - hrv_p80 = high threshold: above this, HRV is "elevated" (very good)

    Recovery tiers use quartile-style percentiles spread across the distribution:
      - p80 = personal "peak" — top quintile of recovery days
      - p60 = personal "good"
      - p40 = personal "moderate"
      - p20 = personal "low" — bottom quintile; below this is a recovery day

    When insufficient data exists (is_personal=False), these fall back to
    WHOOP/population norms.
    """

    # HRV
    hrv_mean: Optional[float] = None
    hrv_std: Optional[float] = None
    hrv_p20: float = POPULATION_HRV_LOW   # low threshold (default = population median)
    hrv_p80: Optional[float] = None        # high threshold (no population fallback needed)

    # Recovery
    recovery_mean: Optional[float] = None
    recovery_p20: float = POPULATION_RECOVERY_LOW
    recovery_p40: float = POPULATION_RECOVERY_MODERATE
    recovery_p60: float = POPULATION_RECOVERY_GOOD
    recovery_p80: float = POPULATION_RECOVERY_PEAK

    # Cognitive load
    cls_mean: Optional[float] = None
    cls_std: Optional[float] = None

    # Metadata
    days_of_data: int = 0
    is_personal: bool = False    # True when personal thresholds are used

    def __repr__(self) -> str:
        src = "personal" if self.is_personal else "population-norm"
        if self.hrv_mean is not None and self.hrv_std is not None:
            hrv_str = f"{self.hrv_mean:.1f}±{self.hrv_std:.1f}ms"
        elif self.hrv_mean is not None:
            hrv_str = f"{self.hrv_mean:.1f}ms"
        else:
            hrv_str = "N/A"
        rec_str = f"{self.recovery_mean:.1f}%" if self.recovery_mean is not None else "N/A"
        return (
            f"PersonalBaseline({src}, days={self.days_of_data}, "
            f"HRV={hrv_str}, recovery_avg={rec_str}, "
            f"recovery_tiers={self.recovery_p20:.0f}/{self.recovery_p40:.0f}"
            f"/{self.recovery_p60:.0f}/{self.recovery_p80:.0f})"
        )


# ─── Percentile helper ─────────────────────────────────────────────────────────

def _percentile(sorted_values: list[float], p: float) -> float:
    """
    Compute the p-th percentile of a sorted list using linear interpolation.

    Args:
        sorted_values: list of floats sorted in ascending order
        p: percentile in [0, 100]

    Returns:
        float: the interpolated percentile value

    Examples:
        _percentile([1, 2, 3, 4, 5], 50) → 3.0
        _percentile([1, 2, 3, 4, 5], 20) → 1.8
        _percentile([10, 20, 30], 0)  → 10.0
        _percentile([10, 20, 30], 100) → 30.0
    """
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]

    n = len(sorted_values)
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    fraction = idx - lo
    return sorted_values[lo] + fraction * (sorted_values[hi] - sorted_values[lo])


# ─── Core computation ──────────────────────────────────────────────────────────

def get_personal_baseline(days: int = 90) -> PersonalBaseline:
    """
    Compute David's personal baseline from accumulated daily summaries.

    Reads the rolling summary store (up to `days` days of history) and
    derives personalized thresholds.  Returns population-norm fallbacks
    when fewer than MIN_DAYS_FOR_PERSONAL_THRESHOLDS days are available.

    Args:
        days: lookback window in days (default: 90 for stable baselines)

    Returns:
        PersonalBaseline with either personal or population-norm thresholds.

    Note:
        This function never raises — it catches all exceptions and returns
        a safe population-norm baseline.  The caller should always check
        baseline.is_personal to know which mode is active.
    """
    try:
        from engine.store import get_recent_summaries
        summaries = get_recent_summaries(days=days)
    except Exception:
        return PersonalBaseline()

    if not summaries:
        return PersonalBaseline(days_of_data=0)

    # ── Collect HRV and recovery values ─────────────────────────────────
    hrv_vals: list[float] = []
    recovery_vals: list[float] = []
    cls_vals: list[float] = []

    for s in summaries:
        whoop = s.get("whoop") or {}
        hrv = whoop.get("hrv_rmssd_milli")
        recovery = whoop.get("recovery_score")

        if hrv is not None:
            try:
                hrv_vals.append(float(hrv))
            except (TypeError, ValueError):
                pass

        if recovery is not None:
            try:
                recovery_vals.append(float(recovery))
            except (TypeError, ValueError):
                pass

        metrics_avg = s.get("metrics_avg") or {}
        cls = metrics_avg.get("cognitive_load_score")
        if cls is not None:
            try:
                cls_vals.append(float(cls))
            except (TypeError, ValueError):
                pass

    days_of_data = len(summaries)
    is_personal = days_of_data >= MIN_DAYS_FOR_PERSONAL_THRESHOLDS

    # ── HRV stats ────────────────────────────────────────────────────────
    hrv_mean: Optional[float] = None
    hrv_std: Optional[float] = None
    hrv_p20: float = POPULATION_HRV_LOW
    hrv_p80: Optional[float] = None

    if hrv_vals:
        hrv_sorted = sorted(hrv_vals)
        hrv_mean = statistics.mean(hrv_vals)
        hrv_std = statistics.stdev(hrv_vals) if len(hrv_vals) >= 2 else 0.0
        if is_personal:
            hrv_p20 = _percentile(hrv_sorted, 20)
            hrv_p80 = _percentile(hrv_sorted, 80)

    # ── Recovery stats ───────────────────────────────────────────────────
    recovery_mean: Optional[float] = None
    recovery_p20 = POPULATION_RECOVERY_LOW
    recovery_p40 = POPULATION_RECOVERY_MODERATE
    recovery_p60 = POPULATION_RECOVERY_GOOD
    recovery_p80 = POPULATION_RECOVERY_PEAK

    if recovery_vals:
        rec_sorted = sorted(recovery_vals)
        recovery_mean = statistics.mean(recovery_vals)
        if is_personal:
            recovery_p20 = _percentile(rec_sorted, 20)
            recovery_p40 = _percentile(rec_sorted, 40)
            recovery_p60 = _percentile(rec_sorted, 60)
            recovery_p80 = _percentile(rec_sorted, 80)

    # ── CLS stats ────────────────────────────────────────────────────────
    cls_mean: Optional[float] = None
    cls_std: Optional[float] = None

    if cls_vals:
        cls_mean = statistics.mean(cls_vals)
        cls_std = statistics.stdev(cls_vals) if len(cls_vals) >= 2 else 0.0

    return PersonalBaseline(
        hrv_mean=round(hrv_mean, 2) if hrv_mean is not None else None,
        hrv_std=round(hrv_std, 2) if hrv_std is not None else None,
        hrv_p20=round(hrv_p20, 1),
        hrv_p80=round(hrv_p80, 1) if hrv_p80 is not None else None,
        recovery_mean=round(recovery_mean, 1) if recovery_mean is not None else None,
        recovery_p20=round(recovery_p20, 1),
        recovery_p40=round(recovery_p40, 1),
        recovery_p60=round(recovery_p60, 1),
        recovery_p80=round(recovery_p80, 1),
        cls_mean=round(cls_mean, 4) if cls_mean is not None else None,
        cls_std=round(cls_std, 4) if cls_std is not None else None,
        days_of_data=days_of_data,
        is_personal=is_personal,
    )


# ─── Readiness tier using personal thresholds ──────────────────────────────────

def is_hrv_low(hrv: Optional[float], baseline: Optional[PersonalBaseline] = None) -> bool:
    """
    Determine if today's HRV is "low" relative to personal or population norms.

    When baseline.is_personal=True, uses the 20th percentile of David's historical
    HRV as the "low" threshold.  This means: HRV is low on the bottom 20% of days.

    When baseline is None or not personal, falls back to the population-norm
    threshold of 45ms (well below the typical adult median of ~55-65ms).

    Args:
        hrv: today's HRV RMSSD in milliseconds (None = data unavailable)
        baseline: PersonalBaseline instance (None = use population norms)

    Returns:
        True if HRV is below the low threshold (stressed autonomic state).
        False if HRV is above threshold, or if hrv is None.

    Examples:
        # David's p20 is 65ms — 60ms is low for him even though > 45ms
        is_hrv_low(60, baseline_with_p20_65) → True

        # Population norm: 60ms is not low (> 45ms threshold)
        is_hrv_low(60, None) → False

        # Missing data: never flag as stressed
        is_hrv_low(None, baseline) → False
    """
    if hrv is None:
        return False
    threshold = POPULATION_HRV_LOW
    if baseline is not None and baseline.is_personal and baseline.hrv_p20 is not None:
        threshold = baseline.hrv_p20
    return hrv < threshold


def readiness_tier_personal(
    recovery: Optional[float],
    hrv: Optional[float],
    baseline: Optional[PersonalBaseline] = None,
) -> str:
    """
    Classify today's physiological readiness using personal (or population) thresholds.

    When baseline.is_personal=True, the five recovery tiers are anchored to David's
    own percentile distribution rather than fixed WHOOP population thresholds.

    Tier mapping:
      - 'peak'     : recovery ≥ p80  (top 20% of David's days)
      - 'good'     : recovery ≥ p60
      - 'moderate' : recovery ≥ p40
      - 'low'      : recovery ≥ p20
      - 'recovery' : recovery < p20  (bottom 20% — genuine rest needed)

    HRV modifier: when today's HRV is below the personal low threshold (p20),
    the tier is downgraded by one step to reflect autonomic stress that the
    recovery score composite may not fully capture.

    Args:
        recovery: WHOOP recovery score (0–100, None = unavailable)
        hrv: HRV RMSSD in milliseconds (None = unavailable)
        baseline: PersonalBaseline (None → population-norm thresholds)

    Returns:
        str: one of 'peak', 'good', 'moderate', 'low', 'recovery', 'unknown'

    Examples:
        # With personal thresholds (p80=85, p60=75, p40=65, p20=55)
        readiness_tier_personal(88, 79, personal_baseline) → 'peak'
        readiness_tier_personal(70, 79, personal_baseline) → 'good'
        readiness_tier_personal(60, 55, personal_baseline) → 'low' (downgraded from moderate)
        readiness_tier_personal(45, 50, personal_baseline) → 'recovery'

        # Without personal baseline (population norms)
        readiness_tier_personal(82, 79, None) → 'peak'
        readiness_tier_personal(72, 40, None) → 'moderate' (HRV stressed)
    """
    if recovery is None:
        return "unknown"

    # Determine thresholds
    if baseline is not None and baseline.is_personal:
        p80 = baseline.recovery_p80
        p60 = baseline.recovery_p60
        p40 = baseline.recovery_p40
        p20 = baseline.recovery_p20
    else:
        p80 = POPULATION_RECOVERY_PEAK
        p60 = POPULATION_RECOVERY_GOOD
        p40 = POPULATION_RECOVERY_MODERATE
        p20 = POPULATION_RECOVERY_LOW

    hrv_stressed = is_hrv_low(hrv, baseline)

    if recovery >= p80:
        return "peak" if not hrv_stressed else "good"
    elif recovery >= p60:
        return "good" if not hrv_stressed else "moderate"
    elif recovery >= p40:
        return "moderate" if not hrv_stressed else "low"
    elif recovery >= p20:
        return "low"
    else:
        return "recovery"


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _fmt_val(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def main() -> None:
    """Print the current personal baseline to stdout."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Show David's personal physiological baselines"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Lookback window in days (default: 90)"
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="Output as JSON"
    )
    args = parser.parse_args()

    baseline = get_personal_baseline(days=args.days)

    if args.json:
        import json
        import dataclasses
        print(json.dumps(dataclasses.asdict(baseline), indent=2))
        return

    src = "📊 personal data" if baseline.is_personal else "⚠️  population norms (insufficient data)"
    print(f"\nPersonal Baseline  [{src}]")
    print(f"  Days of data: {baseline.days_of_data}"
          f"  (need {MIN_DAYS_FOR_PERSONAL_THRESHOLDS} for personal thresholds)")
    print()
    print(f"  HRV")
    print(f"    Mean:          {_fmt_val(baseline.hrv_mean)} ms")
    print(f"    Std:           {_fmt_val(baseline.hrv_std)} ms")
    print(f"    Low (p20):     {_fmt_val(baseline.hrv_p20)} ms  ← 'stressed' threshold")
    print(f"    High (p80):    {_fmt_val(baseline.hrv_p80)} ms")
    print()
    print(f"  Recovery Score")
    print(f"    Mean:          {_fmt_val(baseline.recovery_mean)} %")
    print(f"    Tier thresholds (personal {'✓' if baseline.is_personal else '✗ fallback'}):")
    print(f"      peak   ≥ {_fmt_val(baseline.recovery_p80)} %  (p80)")
    print(f"      good   ≥ {_fmt_val(baseline.recovery_p60)} %  (p60)")
    print(f"      mod    ≥ {_fmt_val(baseline.recovery_p40)} %  (p40)")
    print(f"      low    ≥ {_fmt_val(baseline.recovery_p20)} %  (p20)")
    print(f"      rest   <  {_fmt_val(baseline.recovery_p20)} %")
    print()
    print(f"  Cognitive Load")
    print(f"    Mean:          {_fmt_val(baseline.cls_mean)}")
    print(f"    Std:           {_fmt_val(baseline.cls_std)}")
    print()


if __name__ == "__main__":
    main()

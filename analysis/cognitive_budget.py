"""
Presence Tracker — Daily Cognitive Budget (DCB)

Answers: *"How many quality cognitive hours do I actually have today?"*

WHOOP tells you your physiological recovery percentage.
CDI tells you your accumulated fatigue debt.
But neither translates directly into: "I have N hours of deep focus work available."

The Cognitive Budget bridges that gap. It computes a concrete, actionable estimate
of David's available quality cognitive hours for the day, accounting for:

  1. **WHOOP recovery tier** — physiological ceiling (base hours)
  2. **Sleep quality modifier** — sleep performance as a % adjustment
  3. **CDI tier modifier** — accumulated debt penalises available budget
  4. **HRV modifier** — HRV relative to personal baseline fine-tunes the estimate

## Formula

    base_hours = f(recovery_tier)
        surplus/peak  → 8.0h
        good          → 6.5h
        moderate      → 5.0h
        low           → 3.5h
        recovery      → 2.0h

    sleep_modifier = lerp(0.85, 1.05, sleep_performance / 100)
        (sleep < 50% → -15%; sleep > 90% → +5%)

    cdi_modifier = f(cdi_tier)
        surplus   → 1.10  (well-rested, can push)
        balanced  → 1.00
        loading   → 0.90  (early fatigue, -10%)
        fatigued  → 0.75  (significant debt, -25%)
        critical  → 0.60  (burnout risk, -40%)

    hrv_modifier = lerp(0.95, 1.05, clamp(hrv / hrv_baseline, 0.8, 1.2))
        (only applied when hrv_baseline is available from PersonalBaseline)

    dcb_hours = base_hours × sleep_modifier × cdi_modifier × hrv_modifier
    dcb_hours = clamp(dcb_hours, 1.0, 9.0)

## Output

    CognitiveBudget dataclass:
      - dcb_hours: float             — point estimate (e.g. 5.5)
      - dcb_low: float               — conservative estimate
      - dcb_high: float              — optimistic estimate
      - tier: str                    — peak | good | moderate | low | recovery
      - label: str                   — "Strong day" | "Steady" | "Conserve" | "Protect"
      - base_hours: float            — before modifiers (from WHOOP tier alone)
      - recovery_score: float | None
      - sleep_modifier: float
      - cdi_modifier: float
      - hrv_modifier: float
      - narrative: str               — one actionable sentence
      - guidance: str                — concrete allocation suggestion

## Integration (morning brief)

    from analysis.cognitive_budget import compute_cognitive_budget, format_budget_line
    budget = compute_cognitive_budget(date_str, whoop_data, cdi, baseline)
    if budget.is_meaningful:
        lines.append(format_budget_line(budget))

## Design principles

  - Pure functions + dataclass → fully testable with no live data
  - Graceful degradation: any missing input falls back to a conservative estimate
  - No ML, no black boxes — the formula is explainable and auditable
  - Minimal complexity: linear modifiers, no elaborate curve fitting

## CLI

    python3 analysis/cognitive_budget.py                # Today
    python3 analysis/cognitive_budget.py 2026-03-14     # Specific date
    python3 analysis/cognitive_budget.py --json         # JSON output

"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Constants ────────────────────────────────────────────────────────────────

# Base cognitive hours by WHOOP recovery tier
BASE_HOURS_BY_TIER: dict[str, float] = {
    "peak":     8.0,
    "good":     6.5,
    "moderate": 5.0,
    "low":      3.5,
    "recovery": 2.0,
    "unknown":  5.0,  # fall back to moderate
}

# CDI tier multipliers
CDI_TIER_MODIFIERS: dict[str, float] = {
    "surplus":  1.10,
    "balanced": 1.00,
    "loading":  0.90,
    "fatigued": 0.75,
    "critical": 0.60,
}

# Sleep performance → modifier (clamped to [0.85, 1.05])
SLEEP_MODIFIER_MIN  = 0.85
SLEEP_MODIFIER_MAX  = 1.05
SLEEP_PERF_LOW      = 50.0   # sleep performance % below which we apply minimum modifier
SLEEP_PERF_HIGH     = 90.0   # sleep performance % above which we apply maximum modifier

# HRV relative to baseline → modifier (clamped to [0.95, 1.05])
HRV_MODIFIER_MIN    = 0.95
HRV_MODIFIER_MAX    = 1.05
HRV_RATIO_LOW       = 0.80   # hrv/baseline below this → minimum modifier
HRV_RATIO_HIGH      = 1.20   # hrv/baseline above this → maximum modifier

# Hard bounds for the final DCB estimate
DCB_FLOOR = 1.0
DCB_CEILING = 9.0

# Budget tiers (hours)
# Used for the 'tier' field and human label
BUDGET_TIERS = [
    (7.5, 9.0,  "peak",     "Peak day"),
    (5.5, 7.5,  "good",     "Strong day"),
    (3.5, 5.5,  "moderate", "Steady"),
    (2.0, 3.5,  "low",      "Conserve"),
    (0.0, 2.0,  "recovery", "Protect"),
]

# Guidance by budget tier
GUIDANCE_BY_TIER = {
    "peak": (
        "3 deep focus blocks today — up to 3h each. "
        "Tackle your hardest, most creative work."
    ),
    "good": (
        "2 solid deep focus blocks (90–120min each). "
        "Front-load the demanding work before afternoon."
    ),
    "moderate": (
        "1–2 focused blocks, max 90min. "
        "Reserve willpower — avoid low-value context switching."
    ),
    "low": (
        "1 focused block (60–90min) in your best window. "
        "Let admin and meetings fill the rest."
    ),
    "recovery": (
        "Minimal cognitive load — process, review, async only. "
        "Guard energy; deep work today will cost tomorrow."
    ),
}


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class CognitiveBudget:
    """Daily Cognitive Budget estimate for a given date."""
    date_str: str
    dcb_hours: float              # point estimate
    dcb_low: float                # conservative (−0.5 variance)
    dcb_high: float               # optimistic (+0.5 variance)
    tier: str                     # peak | good | moderate | low | recovery
    label: str                    # human label
    base_hours: float             # before modifiers (WHOOP recovery tier alone)
    recovery_score: Optional[float]
    sleep_performance: Optional[float]
    sleep_modifier: float
    cdi_tier: Optional[str]
    cdi_modifier: float
    hrv_modifier: float
    narrative: str
    guidance: str
    is_meaningful: bool

    def to_dict(self) -> dict:
        return {
            "date_str": self.date_str,
            "dcb_hours": round(self.dcb_hours, 1),
            "dcb_low": round(self.dcb_low, 1),
            "dcb_high": round(self.dcb_high, 1),
            "tier": self.tier,
            "label": self.label,
            "base_hours": round(self.base_hours, 1),
            "recovery_score": self.recovery_score,
            "sleep_performance": self.sleep_performance,
            "sleep_modifier": round(self.sleep_modifier, 3),
            "cdi_tier": self.cdi_tier,
            "cdi_modifier": round(self.cdi_modifier, 3),
            "hrv_modifier": round(self.hrv_modifier, 3),
            "narrative": self.narrative,
            "guidance": self.guidance,
            "is_meaningful": self.is_meaningful,
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _lerp(lo: float, hi: float, t: float) -> float:
    """Linear interpolation between lo and hi; t is clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return lo + (hi - lo) * t


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _sleep_modifier(sleep_performance: Optional[float]) -> float:
    """
    Convert sleep_performance % (0–100) to a modifier in [0.85, 1.05].

    Below 50% → 0.85 (worst penalty)
    Above 90% → 1.05 (small bonus)
    Linear between those endpoints.
    """
    if sleep_performance is None:
        return 1.0  # neutral fallback
    if sleep_performance <= SLEEP_PERF_LOW:
        return SLEEP_MODIFIER_MIN
    if sleep_performance >= SLEEP_PERF_HIGH:
        return SLEEP_MODIFIER_MAX
    t = (sleep_performance - SLEEP_PERF_LOW) / (SLEEP_PERF_HIGH - SLEEP_PERF_LOW)
    return _lerp(SLEEP_MODIFIER_MIN, SLEEP_MODIFIER_MAX, t)


def _hrv_modifier(hrv: Optional[float], hrv_baseline: Optional[float]) -> float:
    """
    Compare today's HRV to personal baseline and produce a modifier in [0.95, 1.05].

    Only applied when both values are available and baseline > 0.
    Falls back to 1.0 (neutral) otherwise.
    """
    if hrv is None or hrv_baseline is None or hrv_baseline <= 0:
        return 1.0
    ratio = hrv / hrv_baseline
    # Map [0.80, 1.20] → [0.95, 1.05]
    if ratio <= HRV_RATIO_LOW:
        return HRV_MODIFIER_MIN
    if ratio >= HRV_RATIO_HIGH:
        return HRV_MODIFIER_MAX
    t = (ratio - HRV_RATIO_LOW) / (HRV_RATIO_HIGH - HRV_RATIO_LOW)
    return _lerp(HRV_MODIFIER_MIN, HRV_MODIFIER_MAX, t)


def _whoop_recovery_tier(recovery_score: Optional[float]) -> str:
    """
    Map WHOOP recovery % to a tier string.

    Uses population norms (same as personal_baseline fallbacks):
      ≥ 80 → peak
      ≥ 67 → good
      ≥ 50 → moderate
      ≥ 34 → low
      <  34 → recovery
    """
    if recovery_score is None:
        return "unknown"
    if recovery_score >= 80:
        return "peak"
    if recovery_score >= 67:
        return "good"
    if recovery_score >= 50:
        return "moderate"
    if recovery_score >= 34:
        return "low"
    return "recovery"


def _budget_tier(hours: float) -> tuple[str, str]:
    """Return (tier, label) for a given DCB hours value."""
    for lo, hi, tier, label in BUDGET_TIERS:
        if lo <= hours < hi:
            return tier, label
    # Fallback: >= 9h is peak
    return "peak", "Peak day"


def _narrative(
    dcb_hours: float,
    tier: str,
    recovery_score: Optional[float],
    cdi_tier: Optional[str],
    sleep_performance: Optional[float],
) -> str:
    """
    Build a one-sentence summary combining the key drivers.

    Examples:
        "86% recovery + balanced fatigue → ~6.0 quality hours today."
        "72% recovery + loading fatigue → ~4.5 hours — front-load your best work."
        "58% recovery + fatigued CDI → ~3.0h available — protect recovery windows."
    """
    hours_str = f"~{dcb_hours:.1f}h"

    # Recovery part
    if recovery_score is not None:
        rec_part = f"{recovery_score:.0f}% recovery"
    else:
        rec_part = "no WHOOP data"

    # CDI part
    cdi_labels = {
        "surplus":  "surplus energy",
        "balanced": "balanced fatigue",
        "loading":  "loading fatigue",
        "fatigued": "elevated fatigue",
        "critical": "critical fatigue",
    }
    if cdi_tier and cdi_tier in cdi_labels:
        cdi_part = cdi_labels[cdi_tier]
    else:
        cdi_part = None

    # Sleep flag
    sleep_note = ""
    if sleep_performance is not None and sleep_performance < 65:
        sleep_note = f" (sleep {sleep_performance:.0f}%)"

    # Build the sentence
    if cdi_part:
        body = f"{rec_part} + {cdi_part} → {hours_str} quality hours today"
    else:
        body = f"{rec_part} → {hours_str} quality hours today"

    body += sleep_note + "."

    # Add a tip for low tiers
    if tier == "recovery":
        body += " Protect your recovery — deep work today costs tomorrow."
    elif tier == "low":
        body += " Front-load your best work in your first available window."

    return body


# ─── Main computation ─────────────────────────────────────────────────────────

def compute_cognitive_budget(
    date_str: str,
    whoop_data: Optional[dict] = None,
    cdi_tier: Optional[str] = None,
    hrv_baseline: Optional[float] = None,
) -> "CognitiveBudget":
    """
    Compute the Daily Cognitive Budget for a given date.

    Parameters
    ----------
    date_str : str
        Date string (YYYY-MM-DD).
    whoop_data : dict | None
        WHOOP signals dict: {recovery_score, sleep_performance, hrv_rmssd_milli, ...}
        If None, falls back to neutral/conservative defaults.
    cdi_tier : str | None
        CDI tier from compute_cdi(): 'surplus' | 'balanced' | 'loading' | 'fatigued' | 'critical'
        If None, neutral modifier (1.0) is applied.
    hrv_baseline : float | None
        Personal HRV baseline from PersonalBaseline.hrv_mean.
        If None, HRV modifier is not applied (1.0).

    Returns
    -------
    CognitiveBudget
        Always returns a valid object. is_meaningful=False when no WHOOP data.
    """
    # Extract WHOOP signals
    recovery_score: Optional[float] = None
    sleep_performance: Optional[float] = None
    hrv: Optional[float] = None

    if whoop_data:
        recovery_score = whoop_data.get("recovery_score")
        sleep_performance = whoop_data.get("sleep_performance")
        hrv = whoop_data.get("hrv_rmssd_milli")

    # ── Base hours from WHOOP recovery tier ──────────────────────────────
    recovery_tier = _whoop_recovery_tier(recovery_score)
    base_hours = BASE_HOURS_BY_TIER[recovery_tier]

    # ── Modifiers ────────────────────────────────────────────────────────
    s_mod = _sleep_modifier(sleep_performance)
    c_mod = CDI_TIER_MODIFIERS.get(cdi_tier or "balanced", 1.0)
    h_mod = _hrv_modifier(hrv, hrv_baseline)

    # ── Combined estimate ────────────────────────────────────────────────
    dcb_raw = base_hours * s_mod * c_mod * h_mod
    dcb_hours = round(_clamp(dcb_raw, DCB_FLOOR, DCB_CEILING), 1)

    # ── Variance band ────────────────────────────────────────────────────
    # ±0.5h represents realistic daily variance not captured by the model
    dcb_low  = round(_clamp(dcb_hours - 0.5, DCB_FLOOR, DCB_CEILING), 1)
    dcb_high = round(_clamp(dcb_hours + 0.5, DCB_FLOOR, DCB_CEILING), 1)

    # ── Tier and label ───────────────────────────────────────────────────
    tier, label = _budget_tier(dcb_hours)

    # ── Narrative and guidance ───────────────────────────────────────────
    narrative = _narrative(dcb_hours, tier, recovery_score, cdi_tier, sleep_performance)
    guidance = GUIDANCE_BY_TIER.get(tier, "")

    # ── Meaningfulness ───────────────────────────────────────────────────
    is_meaningful = recovery_score is not None

    return CognitiveBudget(
        date_str=date_str,
        dcb_hours=dcb_hours,
        dcb_low=dcb_low,
        dcb_high=dcb_high,
        tier=tier,
        label=label,
        base_hours=base_hours,
        recovery_score=recovery_score,
        sleep_performance=sleep_performance,
        sleep_modifier=s_mod,
        cdi_tier=cdi_tier,
        cdi_modifier=c_mod,
        hrv_modifier=h_mod,
        narrative=narrative,
        guidance=guidance,
        is_meaningful=is_meaningful,
    )


# ─── Formatters ───────────────────────────────────────────────────────────────

def format_budget_line(budget: "CognitiveBudget") -> str:
    """
    One-line Slack-ready summary.

    Example:
        "🧠 *Cognitive budget:* ~6.0h — Strong day  _(86% recovery · balanced fatigue)_"
    """
    if not budget.is_meaningful:
        return ""

    modifiers = []
    if budget.recovery_score is not None:
        modifiers.append(f"{budget.recovery_score:.0f}% recovery")
    if budget.cdi_tier:
        cdi_short = {
            "surplus":  "surplus",
            "balanced": "balanced",
            "loading":  "loading",
            "fatigued": "fatigued",
            "critical": "critical",
        }.get(budget.cdi_tier, budget.cdi_tier)
        modifiers.append(f"{cdi_short} CDI")

    mod_str = " · ".join(modifiers)
    range_str = (
        f" ({budget.dcb_low:.1f}–{budget.dcb_high:.1f}h)"
        if budget.dcb_low != budget.dcb_high
        else ""
    )
    suffix = f"  _({mod_str})_" if mod_str else ""

    return (
        f"🧠 *Cognitive budget:* ~{budget.dcb_hours:.1f}h{range_str} — {budget.label}"
        + suffix
    )


def format_budget_section(budget: "CognitiveBudget") -> str:
    """
    Multi-line section for the morning brief — includes guidance.

    Example:
        🧠 *Cognitive budget: ~6.0h — Strong day*
        _86% recovery + balanced fatigue → ~6.0 quality hours today._
        2 solid deep focus blocks (90–120min each). Front-load the demanding work.
    """
    if not budget.is_meaningful:
        return ""

    lines = [
        f"🧠 *Cognitive budget: ~{budget.dcb_hours:.1f}h — {budget.label}*",
        f"_{budget.narrative}_",
    ]
    if budget.guidance:
        lines.append(budget.guidance)

    return "\n".join(lines)


# ─── Convenience loader ───────────────────────────────────────────────────────

def load_and_compute(date_str: str) -> "CognitiveBudget":
    """
    Load WHOOP data and CDI from the store and compute the budget.

    Convenience function for use in morning brief and CLI.
    """
    whoop_data: Optional[dict] = None
    cdi_tier: Optional[str] = None
    hrv_baseline: Optional[float] = None

    # Load WHOOP data from today's JSONL (first window)
    try:
        from engine.store import read_day
        windows = read_day(date_str)
        if windows:
            whoop_data = windows[0].get("whoop") or None
    except Exception:
        pass

    # Load CDI
    try:
        from analysis.cognitive_debt import compute_cdi
        debt = compute_cdi(date_str)
        if debt.is_meaningful:
            cdi_tier = debt.tier
    except Exception:
        pass

    # Load personal baseline for HRV
    try:
        from analysis.personal_baseline import get_personal_baseline
        baseline = get_personal_baseline()
        if baseline.hrv_mean is not None:
            hrv_baseline = baseline.hrv_mean
    except Exception:
        pass

    return compute_cognitive_budget(
        date_str=date_str,
        whoop_data=whoop_data,
        cdi_tier=cdi_tier,
        hrv_baseline=hrv_baseline,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.

    Usage:
        python3 analysis/cognitive_budget.py                  # Today
        python3 analysis/cognitive_budget.py 2026-03-14       # Specific date
        python3 analysis/cognitive_budget.py --json           # JSON output
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Daily Cognitive Budget — quality focus hours available today"
    )
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD), default = today")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    budget = load_and_compute(date_str)

    if args.json:
        print(json.dumps(budget.to_dict(), indent=2))
        return

    print()
    print(f"Cognitive Budget — {date_str}")
    print("=" * 50)

    if not budget.is_meaningful:
        print("  No WHOOP data available for this date.")
        print(f"  Conservative estimate: {budget.dcb_hours:.1f}h")
    else:
        print(f"  Budget:       ~{budget.dcb_hours:.1f}h  ({budget.dcb_low:.1f}–{budget.dcb_high:.1f}h range)")
        print(f"  Tier:         {budget.label}")
        print(f"  Base hours:   {budget.base_hours:.1f}h  (WHOOP {budget.recovery_score:.0f}% recovery)")
        if budget.sleep_performance is not None:
            print(f"  Sleep:        {budget.sleep_performance:.0f}% performance  (×{budget.sleep_modifier:.2f})")
        if budget.cdi_tier:
            print(f"  CDI:          {budget.cdi_tier}  (×{budget.cdi_modifier:.2f})")
        if abs(budget.hrv_modifier - 1.0) > 0.005:
            print(f"  HRV:          ×{budget.hrv_modifier:.3f}")
        print()
        print(f"  → {budget.narrative}")
        if budget.guidance:
            print()
            print(f"  Plan: {budget.guidance}")

    print()


if __name__ == "__main__":
    main()

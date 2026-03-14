"""
Presence Tracker — Daily Presence Score (DPS)

A single composite 0–100 score that answers: "How was your cognitive day?"

This is the mental-state equivalent of WHOOP's daily strain score — one number
that David can internalize at a glance, track over time, and use to spot trends
before individual metrics tell the story.

## Concept

Five metrics capture different dimensions of cognitive performance, but none
alone answers the gestalt question.  A day with high focus (FDI 0.85) and
excellent recovery alignment (RAS 0.92) but heavy load (CLS 0.75) is a very
different day from one with moderate focus (FDI 0.55) and poor alignment
(RAS 0.38).  The DPS captures this composite.

High DPS (≥ 75) = a cognitively quality day:
  focused work, load matched to capacity, low fragmentation, low social drain

Low DPS (< 40) = a poor cognitive day:
  high fragmentation, load exceeded capacity, heavy drain, unfocused

## Formula

    DPS = weighted blend of the five metric components
          where each component is scaled to reward beneficial ranges

    Component contributions (must sum to 1.0):

    1. Focus Quality (0.30 weight)
       Component = active_fdi × (1 − avg_csc)
       FDI tells us how deep focus was; CSC penalises fragmentation.
       Both being high is the "deep work" signal.

    2. Recovery Alignment (0.25 weight)
       Component = avg_ras
       Were today's demands appropriate for your physiology?
       This is the most direct "sustainable day" signal.

    3. Cognitive Load Quality (0.25 weight)
       Component = _load_quality(avg_cls, avg_ras)
       Load by itself is neither good nor bad — moderate CLS with high
       alignment is healthy; the same CLS with poor alignment is strain.
       This rewards days where load was proportionate to capacity.
       Formula: ras_weight × (1 − cls_penalty)
         - cls_penalty = max(0, avg_cls − sweet_spot) × excess_multiplier
         - sweet_spot = 0.55 (moderate engaged load — cognitively productive)
         - above sweet_spot → penalty scales linearly; very high CLS = low component

    4. Social Sustainability (0.10 weight)
       Component = 1 − avg_sdi_active
       Low social drain = sustainable engagement.
       High drain (lots of meetings, many attendees) reduces the score.

    5. Active Engagement (0.10 weight)
       Component = active_fraction (active_windows / working_windows)
       A day with no active windows (all idle) doesn't score high —
       the DPS rewards purposeful engaged time, not just absence of pressure.
       Capped at 1.0, floors at 0.05 so empty days still register.

    raw_dps = (
        0.30 × focus_quality +
        0.25 × recovery_alignment +
        0.25 × load_quality +
        0.10 × social_sustainability +
        0.10 × active_engagement
    )

    DPS = round(raw_dps × 100, 1)  — always 0–100

## Tier labels

    DPS ≥ 85 : "exceptional"  — peak cognitive performance day
    DPS 75–84: "strong"       — well-focused, sustainable
    DPS 60–74: "good"         — solid day, minor friction
    DPS 45–59: "moderate"     — mixed; one dimension pulled it down
    DPS 30–44: "low"          — fragmented, overloaded, or misaligned
    DPS < 30  : "poor"        — high cognitive cost, recovery needed

## API

    compute_presence_score(windows) → PresenceScore
    format_presence_score_line(score) → str   (e.g. "🧠 DPS 78/100 — Strong")
    get_historical_dps(end_date, days) → list[dict]  (for trend analysis)

## Design principles

- No new data required: uses the same windows that power the digest
- Consistent: same formula every time — no LLM randomness
- Robust: degrades gracefully with partial data (missing FDI, no WHOOP, etc.)
- Single responsibility: this module computes and formats the score only;
  integration into digest and morning brief is a thin import

"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Constants ────────────────────────────────────────────────────────────────

# Metric component weights (must sum to 1.0)
WEIGHT_FOCUS_QUALITY      = 0.30
WEIGHT_RECOVERY_ALIGNMENT = 0.25
WEIGHT_LOAD_QUALITY       = 0.25
WEIGHT_SOCIAL_SUSTAIN     = 0.10
WEIGHT_ACTIVE_ENGAGEMENT  = 0.10

# Cognitive load "sweet spot" — load above this starts reducing the component
# 0.55 = a meaningfully engaged working day without exceeding capacity
CLS_SWEET_SPOT = 0.55
# How fast the load quality penalty grows above the sweet spot
CLS_EXCESS_MULTIPLIER = 1.8

# Minimum active fraction to prevent gaming the metric with pure idle days
ACTIVE_FRACTION_FLOOR = 0.05

# Minimum working-hour windows required for a meaningful score
MIN_WORKING_WINDOWS = 4

# DPS tier thresholds
DPS_EXCEPTIONAL = 85
DPS_STRONG      = 75
DPS_GOOD        = 60
DPS_MODERATE    = 45
DPS_LOW         = 30
# < DPS_LOW → 'poor'


# ─── Data class ───────────────────────────────────────────────────────────────

@dataclass
class PresenceScore:
    """
    Daily Presence Score for one day.

    Attributes:
        dps: 0–100 composite score
        tier: 'exceptional' | 'strong' | 'good' | 'moderate' | 'low' | 'poor'
        components: dict with individual component values (0–1 each)
        metrics_used: dict with raw metric values that fed the computation
        is_meaningful: True when ≥ MIN_WORKING_WINDOWS data points existed
        date: date string this was computed for
    """
    dps: float
    tier: str
    components: dict
    metrics_used: dict
    is_meaningful: bool
    date: str


# ─── Core computation ─────────────────────────────────────────────────────────

def _safe_avg(vals: list) -> Optional[float]:
    """Mean of non-None values; None if empty."""
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _load_quality_component(avg_cls: float, avg_ras: float) -> float:
    """
    Compute the cognitive load quality component.

    Rewards days where load was proportionate to physiological capacity.
    Moderate load (CLS ≤ sweet_spot) + high RAS = good component.
    High load (CLS >> sweet_spot) always reduces the component regardless of RAS.

    Args:
        avg_cls: mean cognitive load score over working hours (0–1)
        avg_ras: mean recovery alignment score over all windows (0–1)

    Returns:
        float in [0, 1]
    """
    # CLS penalty: how far above the sweet spot is today's load?
    excess = max(0.0, avg_cls - CLS_SWEET_SPOT)
    cls_penalty = min(1.0, excess * CLS_EXCESS_MULTIPLIER)

    # Base quality = alignment × (1 - penalty)
    # On a low-RAS day (pushed too hard for recovery state), even low CLS scores less
    quality = avg_ras * (1.0 - cls_penalty)
    return round(max(0.0, min(1.0, quality)), 4)


def compute_presence_score(windows: list[dict]) -> PresenceScore:
    """
    Compute the Daily Presence Score from a day's 15-minute windows.

    Args:
        windows: list of window dicts for one day (all 96 windows, or subset)

    Returns:
        PresenceScore with dps, tier, components, and diagnostic metrics.
        Never raises — returns a 'poor'/not-meaningful score on failure.
    """
    try:
        date_str = windows[0].get("date") if windows else None
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")
    except Exception:
        date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        if not windows:
            return _empty_score(date_str)

        # ── Segment windows ──────────────────────────────────────────────
        working = [w for w in windows if w["metadata"]["is_working_hours"]]
        if len(working) < MIN_WORKING_WINDOWS:
            return _empty_score(date_str)

        active = [
            w for w in working
            if w["calendar"]["in_meeting"] or w["slack"]["total_messages"] > 0
        ]

        # ── Extract raw metric values ────────────────────────────────────
        # CLS: all working windows (idle = low CLS = appropriate)
        avg_cls = _safe_avg([w["metrics"]["cognitive_load_score"] for w in working])

        # FDI: active windows only (idle trivially = 1.0; not meaningful)
        avg_fdi = _safe_avg([w["metrics"]["focus_depth_index"] for w in active])

        # SDI: active windows only
        avg_sdi = _safe_avg([w["metrics"]["social_drain_index"] for w in active])

        # CSC: active windows only
        avg_csc = _safe_avg([w["metrics"]["context_switch_cost"] for w in active])

        # RAS: all windows (recovery alignment is meaningful throughout)
        avg_ras = _safe_avg([w["metrics"]["recovery_alignment_score"] for w in windows])

        # Active fraction: what share of working time had real activity?
        active_fraction = max(
            ACTIVE_FRACTION_FLOOR,
            min(1.0, len(active) / len(working)) if working else ACTIVE_FRACTION_FLOOR
        )

        # ── Handle missing data ─────────────────────────────────────────
        # Use neutral fallbacks so the score degrades gracefully when
        # one source is missing (e.g. no WHOOP data = RAS unknown)
        _cls = avg_cls if avg_cls is not None else 0.35   # neutral moderate
        _fdi = avg_fdi if avg_fdi is not None else 0.60   # neutral focus
        _sdi = avg_sdi if avg_sdi is not None else 0.30   # neutral social
        _csc = avg_csc if avg_csc is not None else 0.25   # neutral fragmentation
        _ras = avg_ras if avg_ras is not None else 0.50   # neutral alignment

        # ── Compute individual components ────────────────────────────────
        # 1. Focus Quality: how deep and unfragmented was the focused work?
        #    FDI × (1 − CSC) rewards both depth and coherence.
        focus_quality = round(_fdi * (1.0 - _csc), 4)

        # 2. Recovery Alignment: how well was load matched to physiology?
        recovery_alignment = round(_ras, 4)

        # 3. Cognitive Load Quality: appropriate load for the day's capacity?
        load_quality = _load_quality_component(_cls, _ras)

        # 4. Social Sustainability: low drain = sustainable engagement
        social_sustain = round(1.0 - _sdi, 4)

        # 5. Active Engagement: purposeful engaged time as fraction of working day
        active_engagement = round(active_fraction, 4)

        # ── Weighted composite ───────────────────────────────────────────
        raw_dps = (
            WEIGHT_FOCUS_QUALITY      * focus_quality      +
            WEIGHT_RECOVERY_ALIGNMENT * recovery_alignment +
            WEIGHT_LOAD_QUALITY       * load_quality       +
            WEIGHT_SOCIAL_SUSTAIN     * social_sustain     +
            WEIGHT_ACTIVE_ENGAGEMENT  * active_engagement
        )

        dps = round(max(0.0, min(100.0, raw_dps * 100.0)), 1)
        tier = _dps_tier(dps)

        return PresenceScore(
            dps=dps,
            tier=tier,
            components={
                "focus_quality":      round(focus_quality, 4),
                "recovery_alignment": round(recovery_alignment, 4),
                "load_quality":       round(load_quality, 4),
                "social_sustain":     round(social_sustain, 4),
                "active_engagement":  round(active_engagement, 4),
            },
            metrics_used={
                "avg_cls": avg_cls,
                "avg_fdi_active": avg_fdi,
                "avg_sdi_active": avg_sdi,
                "avg_csc_active": avg_csc,
                "avg_ras": avg_ras,
                "active_windows": len(active),
                "working_windows": len(working),
            },
            is_meaningful=True,
            date=date_str,
        )

    except Exception:
        # Never crash the pipeline — return a safe neutral score
        return _empty_score(date_str)


def _empty_score(date_str: str) -> PresenceScore:
    """Return a safe not-meaningful score for missing/insufficient data."""
    return PresenceScore(
        dps=50.0,
        tier="moderate",
        components={
            "focus_quality": None,
            "recovery_alignment": None,
            "load_quality": None,
            "social_sustain": None,
            "active_engagement": None,
        },
        metrics_used={},
        is_meaningful=False,
        date=date_str,
    )


def _dps_tier(dps: float) -> str:
    """Map a DPS value to a tier label."""
    if dps >= DPS_EXCEPTIONAL:
        return "exceptional"
    elif dps >= DPS_STRONG:
        return "strong"
    elif dps >= DPS_GOOD:
        return "good"
    elif dps >= DPS_MODERATE:
        return "moderate"
    elif dps >= DPS_LOW:
        return "low"
    else:
        return "poor"


# ─── Formatting ───────────────────────────────────────────────────────────────

_TIER_EMOJI = {
    "exceptional": "🌟",
    "strong":      "💪",
    "good":        "✅",
    "moderate":    "🟡",
    "low":         "🔴",
    "poor":        "🚨",
}

_TIER_LABEL = {
    "exceptional": "Exceptional",
    "strong":      "Strong",
    "good":        "Good",
    "moderate":    "Moderate",
    "low":         "Low",
    "poor":        "Poor",
}

_TIER_TAGLINE = {
    "exceptional": "peak cognitive day — bank the focus patterns",
    "strong":      "well-focused and sustainable",
    "good":        "solid work, minor friction",
    "moderate":    "mixed — one dimension pulled it down",
    "low":         "fragmented, overloaded, or misaligned",
    "poor":        "high cognitive cost, recovery needed tomorrow",
}


def format_presence_score_line(score: PresenceScore) -> str:
    """
    Format a compact one-line Presence Score for use in digest/brief.

    Examples:
        "🌟 DPS 91/100 — Exceptional (peak cognitive day — bank the focus patterns)"
        "✅ DPS 67/100 — Good (solid work, minor friction)"
        "🔴 DPS 38/100 — Low (fragmented, overloaded, or misaligned)"

    Returns empty string if score is not meaningful.
    """
    if not score.is_meaningful:
        return ""

    emoji = _TIER_EMOJI.get(score.tier, "⚪")
    label = _TIER_LABEL.get(score.tier, score.tier.capitalize())
    tagline = _TIER_TAGLINE.get(score.tier, "")

    parts = [f"{emoji} DPS {score.dps:.0f}/100 — {label}"]
    if tagline:
        parts.append(f"({tagline})")
    return " ".join(parts)


def format_presence_score_block(score: PresenceScore) -> str:
    """
    Format a multi-line component breakdown for use in terminal reports.

    Shows the score, tier, and each component with its contribution.

    Returns empty string if score is not meaningful.
    """
    if not score.is_meaningful:
        return "(DPS: insufficient data)"

    emoji = _TIER_EMOJI.get(score.tier, "⚪")
    label = _TIER_LABEL.get(score.tier, score.tier.capitalize())

    lines = [
        f"{emoji} Presence Score: {score.dps:.0f}/100 — {label}",
        f"  Focus quality:     {score.components.get('focus_quality', 0):.0%}  (weight 30%)",
        f"  Recovery alignment:{score.components.get('recovery_alignment', 0):.0%}  (weight 25%)",
        f"  Load quality:      {score.components.get('load_quality', 0):.0%}  (weight 25%)",
        f"  Social sustain:    {score.components.get('social_sustain', 0):.0%}  (weight 10%)",
        f"  Active engagement: {score.components.get('active_engagement', 0):.0%}  (weight 10%)",
    ]
    return "\n".join(lines)


# ─── Historical trend ─────────────────────────────────────────────────────────

def get_historical_dps(end_date_str: str, days: int = 14) -> list[dict]:
    """
    Compute DPS for each of the last N days ending on end_date_str.

    Uses the rolling summary store — reads each day's stored windows.
    Returns a list of dicts [{"date": ..., "dps": ..., "tier": ...}, ...]
    sorted oldest-first.  Missing days are skipped (not returned as zeros).

    Args:
        end_date_str: Last date to include (YYYY-MM-DD), inclusive.
        days: How many days back to compute (default: 14).

    Returns:
        List of {"date", "dps", "tier", "is_meaningful"} dicts, oldest first.
        Empty list if no data is available.
    """
    try:
        from engine.store import read_day, list_available_dates

        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        date_window = [
            (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(days - 1, -1, -1)
        ]
        available = set(list_available_dates())

        results = []
        for d in date_window:
            if d not in available:
                continue
            day_windows = read_day(d)
            if not day_windows:
                continue
            score = compute_presence_score(day_windows)
            results.append({
                "date": d,
                "dps": score.dps,
                "tier": score.tier,
                "is_meaningful": score.is_meaningful,
            })

        return results

    except Exception:
        return []


def compute_dps_trend(end_date_str: str, days: int = 7) -> Optional[dict]:
    """
    Compute a simple DPS trend summary for use in digests and briefs.

    Returns a dict with:
    - mean_dps: average over the window
    - delta_dps: difference between last 3d avg and prior 4d avg
    - trend_direction: 'improving' | 'declining' | 'stable'
    - days_used: number of days with data
    - best_day: date with highest DPS
    - worst_day: date with lowest DPS

    Returns None when insufficient data (< 3 days).
    """
    history = get_historical_dps(end_date_str, days=days)
    if len(history) < 3:
        return None

    dps_vals = [h["dps"] for h in history if h["is_meaningful"]]
    if len(dps_vals) < 3:
        return None

    mean_dps = round(sum(dps_vals) / len(dps_vals), 1)

    # Split into recent (last 3 days) vs earlier
    recent = dps_vals[-3:]
    prior = dps_vals[:-3]
    mean_recent = sum(recent) / len(recent)
    mean_prior = sum(prior) / len(prior) if prior else mean_recent
    delta = round(mean_recent - mean_prior, 1)

    if abs(delta) < 3.0:
        direction = "stable"
    elif delta > 0:
        direction = "improving"
    else:
        direction = "declining"

    best = max(history, key=lambda h: h["dps"])
    worst = min(history, key=lambda h: h["dps"])

    return {
        "mean_dps": mean_dps,
        "delta_dps": delta,
        "trend_direction": direction,
        "days_used": len(dps_vals),
        "best_day": best["date"],
        "best_dps": best["dps"],
        "worst_day": worst["date"],
        "worst_dps": worst["dps"],
    }


# ─── Store integration helper ─────────────────────────────────────────────────

def enrich_summary_with_dps(summary: dict, windows: list[dict]) -> dict:
    """
    Compute and inject DPS into a daily summary dict.

    Called by summarize_day() to persist the DPS alongside other metrics.
    The DPS is stored as summary["presence_score"] = {"dps": float, "tier": str}.

    Args:
        summary: the daily summary dict (mutated in place)
        windows: the full window list for the day

    Returns:
        The modified summary dict.
    """
    try:
        score = compute_presence_score(windows)
        if score.is_meaningful:
            summary["presence_score"] = {
                "dps": score.dps,
                "tier": score.tier,
                "components": score.components,
            }
    except Exception:
        pass  # Never crash the pipeline
    return summary


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Compute Daily Presence Score for a given date"
    )
    parser.add_argument(
        "date", nargs="?",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--history", type=int, default=0,
        help="Show DPS for the last N days ending on date",
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--breakdown", "-b", action="store_true",
        help="Show component breakdown",
    )
    args = parser.parse_args()

    if args.history > 0:
        history = get_historical_dps(args.date, days=args.history)
        trend = compute_dps_trend(args.date, days=args.history)

        if args.json:
            print(_json.dumps({"history": history, "trend": trend}, indent=2))
            return

        print(f"\nPresence Score — last {args.history} days ending {args.date}")
        for h in history:
            emoji = _TIER_EMOJI.get(h["tier"], "⚪")
            print(f"  {h['date']}  {emoji} {h['dps']:5.1f}  ({h['tier']})")

        if trend:
            arrow = {"improving": "↑", "declining": "↓", "stable": "→"}.get(
                trend["trend_direction"], ""
            )
            print(f"\n  7-day avg: {trend['mean_dps']:.1f}  trend: {arrow} {trend['trend_direction']}")
            print(f"  Best: {trend['best_day']} ({trend['best_dps']:.0f})")
            print(f"  Worst: {trend['worst_day']} ({trend['worst_dps']:.0f})")
        print()
        return

    # Single day
    from engine.store import read_day, list_available_dates
    dates = list_available_dates()
    date_str = args.date

    if date_str not in dates:
        print(f"No data for {date_str} (available: {', '.join(sorted(dates)[-5:])})")
        return

    windows = read_day(date_str)
    score = compute_presence_score(windows)

    if args.json:
        import dataclasses
        print(_json.dumps(dataclasses.asdict(score), indent=2))
        return

    print(f"\nDaily Presence Score — {date_str}")
    print(f"  DPS:    {score.dps:.1f} / 100")
    print(f"  Tier:   {_TIER_EMOJI.get(score.tier, '')} {score.tier}")
    print()

    if args.breakdown:
        print(format_presence_score_block(score))
    else:
        print(f"  {format_presence_score_line(score)}")

    if score.metrics_used:
        print(f"\n  Inputs:")
        print(f"    CLS (avg):  {(score.metrics_used.get('avg_cls') or 0):.3f}")
        print(f"    FDI (active): {(score.metrics_used.get('avg_fdi_active') or 0):.3f}")
        print(f"    SDI (active): {(score.metrics_used.get('avg_sdi_active') or 0):.3f}")
        print(f"    CSC (active): {(score.metrics_used.get('avg_csc_active') or 0):.3f}")
        print(f"    RAS (all):  {(score.metrics_used.get('avg_ras') or 0):.3f}")
        print(f"    Active windows: {score.metrics_used.get('active_windows', 0)} / "
              f"{score.metrics_used.get('working_windows', 0)} working")
    print()


if __name__ == "__main__":
    main()

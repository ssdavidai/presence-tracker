"""
Presence Tracker — Recovery Planner (v1)

Answers: *"What specific recovery actions would restore my cognitive balance fastest?"*

The CDI Forecast tells David: "At this pace, you'll reach fatigued in 3 days."
But it doesn't answer the follow-up question: "What should I DO about it?"

The Recovery Planner closes this gap with scenario planning:

  1. **Status-quo trajectory** — where CDI is heading if nothing changes
     (identical to CDI Forecast, used as the baseline)

  2. **Recovery scenario** — what happens if the next 1–2 days are light
     (CLS reduced to recovery_load_cls = 0.12 · active_fraction)

  3. **Aggressive recovery** — what happens if David takes a genuine rest day
     (CLS = 0, e.g. weekend / no-work day, with above-baseline WHOOP recovery)

  4. **Payback schedule** — how many light/rest days needed to return to balanced,
     and what's the optimal placement (weekday vs weekend)

  5. **Concrete recommendation** — one actionable sentence with a day-specific plan:
       "Take Thursday light (no deep work) + protect the weekend → balanced by Mon"

## When it fires

Recovery Planner is only meaningful when CDI ≥ loading (≥ 50).
For balanced/surplus CDI, it returns is_meaningful=False (nothing to plan).

## Scenarios modelled

### Scenario A: Status quo
  - Load: historical baseline load_signal (same as CDI Forecast)
  - Recovery: historical baseline recovery_signal

### Scenario B: One light day (today/tomorrow is protected)
  - Load: LIGHT_DAY_CLS × DEFAULT_ACTIVE_FRACTION for day +1
  - Load: historical baseline for days +2 onwards
  - Recovery: baseline

### Scenario C: Two consecutive light days
  - Load: LIGHT_DAY_CLS × DEFAULT_ACTIVE_FRACTION for days +1 and +2
  - Load: historical baseline afterwards
  - Recovery: baseline

### Scenario D: One full rest day (e.g. weekend / holiday)
  - Load: REST_DAY_CLS (≈ 0) for day +1
  - WHOOP recovery assumed to improve (REST_DAY_RECOVERY_BOOST applied)
  - Load: historical baseline afterwards

## Key outputs

    RecoveryPlan dataclass:
      - date_str: str                  — anchor date
      - today_cdi: float              — current CDI score
      - today_tier: str               — current tier
      - days_to_balanced_status_quo: int | None  — no intervention
      - days_to_balanced_light_day:  int | None  — 1 light day
      - days_to_balanced_two_light:  int | None  — 2 light days
      - days_to_balanced_rest_day:   int | None  — 1 full rest day
      - recommended_action: str      — one concrete action label
      - recommendation_detail: str   — one full sentence
      - scenarios: dict              — full per-day CDI for each scenario
      - is_meaningful: bool          — False when CDI < loading or insufficient data

## API

    from analysis.recovery_planner import compute_recovery_plan, format_recovery_section
    from analysis.recovery_planner import format_recovery_line

    plan = compute_recovery_plan(date_str)
    line    = format_recovery_line(plan)     # compact one-liner for digest
    section = format_recovery_section(plan)  # full section for weekly/report

## Integration

    In nightly digest (after CDI Forecast block):
        plan = compute_recovery_plan(date_str)
        if plan.is_meaningful:
            lines.append(format_recovery_line(plan))

    In weekly summary (recovery section):
        plan = compute_recovery_plan(end_date)
        if plan.is_meaningful:
            lines.append(format_recovery_section(plan))

    In report.py (CDI section):
        plan = compute_recovery_plan(date_str)
        if plan.is_meaningful:
            print(format_recovery_section(plan))

## CLI

    python3 analysis/recovery_planner.py              # Today
    python3 analysis/recovery_planner.py 2026-03-14   # Specific date
    python3 analysis/recovery_planner.py --json       # JSON output

## Design principles

  - Scenario planning, not prediction — explicit "what if X" reasoning
  - Conservative assumptions: recovery boost is modest, not magical
  - Pure functions — fully testable with no live API calls
  - Never raises — returns is_meaningful=False on any error
  - Only fires when CDI ≥ loading — avoids noise when things are fine
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import list_available_dates, read_summary

# ─── Constants ────────────────────────────────────────────────────────────────

# CDI formula constants (must stay in sync with cognitive_debt.py)
CDI_SERIES_CLAMP = 14.0
CDI_BALANCED_MAX = 50.0
CDI_LOADING_MIN = 50.0   # CDI ≥ 50 is "loading" (worth planning recovery)
CDI_LOADING_MAX = 70.0
CDI_FATIGUED_MAX = 85.0

# Tier thresholds for Recovery Planner
THRESHOLD_TO_PLAN = CDI_LOADING_MIN   # Only fire when CDI ≥ this
THRESHOLD_BALANCED = CDI_BALANCED_MAX # Target: reach CDI ≤ this

# Light day: reduced cognitive load (meetings cancelled / deep work avoided)
# Represents a conscious "protection" day — still working, just lighter
LIGHT_DAY_CLS = 0.12

# Rest day: minimal cognitive load (holiday / weekend day with genuine rest)
# Very low load, much better recovery signal expected
REST_DAY_CLS = 0.02
REST_DAY_RECOVERY_BOOST = 0.12  # Adds to historical recovery_signal (capped at 0.95)

# Default active fraction for load→signal conversion
DEFAULT_ACTIVE_FRACTION = 0.40

# How many recent days to use for baselines
BASELINE_WINDOW = 7

# Minimum days of history needed
MIN_DAYS = 3

# Projection horizon (days to look forward for "will it recover?")
HORIZON = 10

# Minimum CDI delta before we consider a scenario "meaningfully better"
MEANINGFUL_IMPROVEMENT_DELTA = 3.0

# Recovery signal clamps
RECOVERY_SIGNAL_MIN = 0.35
RECOVERY_SIGNAL_MAX = 0.95


# ─── Tier helpers ────────────────────────────────────────────────────────────

def _cdi_tier(cdi: float) -> str:
    """Map CDI score to tier name."""
    if cdi < 30.0:
        return "surplus"
    elif cdi <= CDI_BALANCED_MAX:
        return "balanced"
    elif cdi <= CDI_LOADING_MAX:
        return "loading"
    elif cdi <= CDI_FATIGUED_MAX:
        return "fatigued"
    else:
        return "critical"


def _cdi_from_running_sum(running_sum: float) -> float:
    """Convert CDI running sum to 0–100 CDI score."""
    return round(max(0.0, min(100.0, 50.0 + (running_sum / CDI_SERIES_CLAMP) * 50.0)), 1)


# ─── Pure projection functions ───────────────────────────────────────────────

def _project_scenario(
    current_running_sum: float,
    per_day_deltas: list[float],
) -> list[float]:
    """
    Project CDI values for each future day given a list of per-day debt deltas.

    Args:
        current_running_sum: CDI debt_series[-1] (anchor point).
        per_day_deltas:       Delta for each future day (positive = more debt).

    Returns:
        List of CDI scores (0–100) for each future day.
    """
    cdis = []
    running_sum = current_running_sum
    for delta in per_day_deltas:
        running_sum = max(-CDI_SERIES_CLAMP, min(CDI_SERIES_CLAMP, running_sum + delta))
        cdis.append(_cdi_from_running_sum(running_sum))
    return cdis


def _days_to_balanced(projected_cdis: list[float]) -> Optional[int]:
    """
    Return how many days until CDI first reaches ≤ THRESHOLD_BALANCED.

    Returns None if CDI never reaches balanced within the projection horizon.
    """
    for i, cdi in enumerate(projected_cdis, start=1):
        if cdi <= THRESHOLD_BALANCED:
            return i
    return None


def _build_status_quo_deltas(
    load_signal: float,
    recovery_signal: float,
    horizon: int = HORIZON,
) -> list[float]:
    """Status quo: same load and recovery each day."""
    delta = load_signal - recovery_signal
    return [delta] * horizon


def _build_light_day_deltas(
    load_signal: float,
    recovery_signal: float,
    light_days: int = 1,
    horizon: int = HORIZON,
) -> list[float]:
    """
    Scenario: the next `light_days` days are light (reduced CLS).
    Remaining days revert to historical baseline.
    """
    light_load = LIGHT_DAY_CLS * DEFAULT_ACTIVE_FRACTION
    light_delta = light_load - recovery_signal
    normal_delta = load_signal - recovery_signal
    deltas = []
    for i in range(horizon):
        if i < light_days:
            deltas.append(light_delta)
        else:
            deltas.append(normal_delta)
    return deltas


def _build_rest_day_deltas(
    load_signal: float,
    recovery_signal: float,
    horizon: int = HORIZON,
) -> list[float]:
    """
    Scenario: day +1 is a genuine rest day (holiday/weekend) with boosted recovery.
    Days +2 onwards revert to historical baseline.
    """
    rest_load = REST_DAY_CLS * DEFAULT_ACTIVE_FRACTION
    boosted_recovery = min(RECOVERY_SIGNAL_MAX, recovery_signal + REST_DAY_RECOVERY_BOOST)
    rest_delta = rest_load - boosted_recovery
    normal_delta = load_signal - recovery_signal
    deltas = [rest_delta] + [normal_delta] * (horizon - 1)
    return deltas


# ─── Signal extraction ────────────────────────────────────────────────────────

def _extract_signals(date_str: str) -> tuple[float, float, float, int]:
    """
    Extract recovery_signal, load_signal, current_running_sum, days_of_history
    from the rolling summary store.

    Returns conservative defaults on any failure.
    """
    try:
        rolling = read_summary()
        available_dates = list_available_dates()

        recent_dates = [d for d in available_dates if d < date_str][-BASELINE_WINDOW:]
        days_of_history = len(recent_dates)

        if days_of_history < MIN_DAYS:
            return 0.60, 0.15, 0.0, days_of_history

        # ── Recovery signal ────────────────────────────────────────────────
        rec_values = []
        days_data = rolling.get("days", {})
        for d in recent_dates:
            day = days_data.get(d, {})
            whoop = day.get("whoop") or {}
            rec = whoop.get("recovery_score")
            if rec is not None:
                rec_values.append(float(rec) / 100.0)
        recovery_signal = (
            max(RECOVERY_SIGNAL_MIN, min(RECOVERY_SIGNAL_MAX, sum(rec_values) / len(rec_values)))
            if rec_values else 0.60
        )

        # ── Load signal ────────────────────────────────────────────────────
        load_values = []
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
            load_values.append(avg_cls * active_fraction)
        load_signal = (
            sum(load_values) / len(load_values) if load_values else 0.15
        )

        # ── Current running sum from CDI debt series ───────────────────────
        current_running_sum = 0.0
        try:
            from analysis.cognitive_debt import compute_cdi
            debt = compute_cdi(date_str)
            if debt.is_meaningful and debt.debt_series:
                current_running_sum = debt.debt_series[-1]
        except Exception:
            pass

        return recovery_signal, load_signal, current_running_sum, days_of_history

    except Exception:
        return 0.60, 0.15, 0.0, 0


# ─── Recommendation logic ─────────────────────────────────────────────────────

def _pick_recommended_action(
    today_tier: str,
    days_sq: Optional[int],   # status quo
    days_ld: Optional[int],   # one light day
    days_2l: Optional[int],   # two light days
    days_rd: Optional[int],   # rest day
) -> tuple[str, str]:
    """
    Pick the most efficient recovery action and return (action_label, detail_sentence).

    Decision logic:
    - If rest day recovers fastest AND is materially better → recommend rest
    - If two light days recover significantly faster than one → recommend two light
    - If one light day is sufficient → recommend one light day
    - If nothing helps within horizon → recommend long-term load reduction
    """

    def _days_str(n: Optional[int]) -> str:
        if n is None:
            return "not within 10 days"
        return f"{n} day{'s' if n != 1 else ''}"

    # Nothing recovers within horizon
    if days_sq is None and days_ld is None and days_2l is None and days_rd is None:
        return (
            "reduce load long-term",
            "CDI won't recover within 10 days at current pace — a structural load reduction "
            "is needed: fewer meetings, protected deep-work blocks, or a multi-day break.",
        )

    # Find minimum recovery time across all scenarios
    options = {
        k: v for k, v in {
            "rest day": days_rd,
            "2 light days": days_2l,
            "1 light day": days_ld,
            "status quo": days_sq,
        }.items()
        if v is not None
    }
    best_option = min(options, key=lambda k: options[k])
    best_days = options[best_option]

    # Compare status quo vs interventions
    sq_days = days_sq if days_sq is not None else HORIZON + 1

    # Rest day: only recommend if materially better than light day and status quo
    if days_rd is not None and (days_rd < sq_days - 2):
        if days_rd == 1:
            return (
                "take a full rest day",
                f"A genuine rest day tomorrow brings CDI back to balanced in 1 day — "
                f"the fastest recovery option vs {_days_str(days_sq)} at current pace.",
            )
        return (
            "take a full rest day",
            f"A full rest day cuts recovery time to {_days_str(days_rd)} "
            f"vs {_days_str(days_sq)} without intervention.",
        )

    # Two light days better than one
    if days_2l is not None and days_ld is not None and days_2l < days_ld - 1:
        return (
            "protect 2 consecutive light days",
            f"Two protected (light) days recovers CDI in {_days_str(days_2l)} — "
            f"faster than one light day ({_days_str(days_ld)}) "
            f"and much better than status quo ({_days_str(days_sq)}).",
        )

    # One light day is sufficient and better than status quo
    if days_ld is not None and (days_sq is None or days_ld < sq_days - 1):
        if days_ld == 1:
            return (
                "make tomorrow a light day",
                f"One protected day tomorrow restores CDI to balanced — "
                f"cancel non-essential meetings and avoid reactive work.",
            )
        return (
            "protect one light day",
            f"One protected (light) workday recovers CDI in {_days_str(days_ld)} "
            f"vs {_days_str(days_sq)} without change.",
        )

    # Status quo recovers within horizon — no urgent action
    if days_sq is not None:
        return (
            "maintain current pace",
            f"CDI recovers to balanced in {_days_str(days_sq)} without intervention — "
            f"current recovery arc is sufficient.",
        )

    # Fallback
    return (
        "monitor",
        "CDI is elevated but no single-day intervention is sufficient — "
        "focus on consistent sleep and gradual load reduction.",
    )


# ─── Dataclass ───────────────────────────────────────────────────────────────

@dataclass
class RecoveryPlan:
    """
    CDI Recovery Plan — scenario-based payback schedule.

    Attributes:
        date_str:                    Anchor date (YYYY-MM-DD)
        today_cdi:                   Current CDI score (0–100)
        today_tier:                  Current CDI tier name
        days_to_balanced_status_quo: Days to balanced without intervention
        days_to_balanced_light_day:  Days to balanced with one protected day
        days_to_balanced_two_light:  Days to balanced with two protected days
        days_to_balanced_rest_day:   Days to balanced with a full rest day
        recommended_action:          Short action label
        recommendation_detail:       One-sentence concrete recommendation
        scenarios:                   Per-day CDI lists for each scenario (horizon days)
        recovery_signal_used:        Historical recovery signal (avg recovery/100)
        load_signal_used:            Historical load signal (avg CLS × active_fraction)
        days_of_history:             Days of history used to build baselines
        is_meaningful:               False when CDI < loading or insufficient data
    """
    date_str: str
    today_cdi: float
    today_tier: str
    days_to_balanced_status_quo: Optional[int]
    days_to_balanced_light_day: Optional[int]
    days_to_balanced_two_light: Optional[int]
    days_to_balanced_rest_day: Optional[int]
    recommended_action: str
    recommendation_detail: str
    scenarios: dict
    recovery_signal_used: float
    load_signal_used: float
    days_of_history: int
    is_meaningful: bool

    def to_dict(self) -> dict:
        return {
            "date_str": self.date_str,
            "today_cdi": self.today_cdi,
            "today_tier": self.today_tier,
            "days_to_balanced_status_quo": self.days_to_balanced_status_quo,
            "days_to_balanced_light_day": self.days_to_balanced_light_day,
            "days_to_balanced_two_light": self.days_to_balanced_two_light,
            "days_to_balanced_rest_day": self.days_to_balanced_rest_day,
            "recommended_action": self.recommended_action,
            "recommendation_detail": self.recommendation_detail,
            "scenarios": self.scenarios,
            "recovery_signal_used": self.recovery_signal_used,
            "load_signal_used": self.load_signal_used,
            "days_of_history": self.days_of_history,
            "is_meaningful": self.is_meaningful,
        }


# ─── Main computation ─────────────────────────────────────────────────────────

def compute_recovery_plan(date_str: str = None) -> RecoveryPlan:
    """
    Compute the CDI recovery plan for a given anchor date.

    Only meaningful when CDI ≥ loading (≥ 50).  For balanced/surplus CDI,
    returns is_meaningful=False — no recovery planning needed.

    Args:
        date_str: Anchor date (YYYY-MM-DD). Defaults to today.

    Returns:
        RecoveryPlan dataclass. Never raises.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    _EMPTY = RecoveryPlan(
        date_str=date_str,
        today_cdi=50.0,
        today_tier="balanced",
        days_to_balanced_status_quo=None,
        days_to_balanced_light_day=None,
        days_to_balanced_two_light=None,
        days_to_balanced_rest_day=None,
        recommended_action="monitor",
        recommendation_detail="Not enough data for recovery planning.",
        scenarios={},
        recovery_signal_used=0.60,
        load_signal_used=0.15,
        days_of_history=0,
        is_meaningful=False,
    )

    try:
        # ── Get current CDI ────────────────────────────────────────────────
        from analysis.cognitive_debt import compute_cdi

        debt = compute_cdi(date_str)
        if not debt.is_meaningful:
            return _EMPTY

        today_cdi = debt.cdi
        today_tier = debt.tier

        # Only fire when CDI is elevated (loading or above)
        if today_cdi < THRESHOLD_TO_PLAN:
            return RecoveryPlan(
                date_str=date_str,
                today_cdi=today_cdi,
                today_tier=today_tier,
                days_to_balanced_status_quo=None,
                days_to_balanced_light_day=None,
                days_to_balanced_two_light=None,
                days_to_balanced_rest_day=None,
                recommended_action="maintain",
                recommendation_detail="CDI is balanced — no recovery intervention needed.",
                scenarios={},
                recovery_signal_used=0.60,
                load_signal_used=0.15,
                days_of_history=0,
                is_meaningful=False,
            )

        current_running_sum = debt.debt_series[-1] if debt.debt_series else 0.0

        # ── Extract baselines ──────────────────────────────────────────────
        recovery_signal, load_signal, _, days_of_history = _extract_signals(date_str)
        # Reuse current_running_sum from debt object (more accurate than extract)

        if days_of_history < MIN_DAYS:
            return _EMPTY

        # ── Build per-day delta lists for each scenario ────────────────────
        sq_deltas = _build_status_quo_deltas(load_signal, recovery_signal)
        ld_deltas = _build_light_day_deltas(load_signal, recovery_signal, light_days=1)
        tl_deltas = _build_light_day_deltas(load_signal, recovery_signal, light_days=2)
        rd_deltas = _build_rest_day_deltas(load_signal, recovery_signal)

        # ── Project CDI for each scenario ──────────────────────────────────
        sq_cdis = _project_scenario(current_running_sum, sq_deltas)
        ld_cdis = _project_scenario(current_running_sum, ld_deltas)
        tl_cdis = _project_scenario(current_running_sum, tl_deltas)
        rd_cdis = _project_scenario(current_running_sum, rd_deltas)

        # ── Compute days to balanced ───────────────────────────────────────
        days_sq = _days_to_balanced(sq_cdis)
        days_ld = _days_to_balanced(ld_cdis)
        days_2l = _days_to_balanced(tl_cdis)
        days_rd = _days_to_balanced(rd_cdis)

        # ── Pick recommendation ────────────────────────────────────────────
        action, detail = _pick_recommended_action(
            today_tier, days_sq, days_ld, days_2l, days_rd
        )

        # ── Build date labels for scenarios ───────────────────────────────
        today_dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_labels = [
            (today_dt + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(HORIZON)
        ]

        return RecoveryPlan(
            date_str=date_str,
            today_cdi=today_cdi,
            today_tier=today_tier,
            days_to_balanced_status_quo=days_sq,
            days_to_balanced_light_day=days_ld,
            days_to_balanced_two_light=days_2l,
            days_to_balanced_rest_day=days_rd,
            recommended_action=action,
            recommendation_detail=detail,
            scenarios={
                "status_quo": dict(zip(date_labels, sq_cdis)),
                "one_light_day": dict(zip(date_labels, ld_cdis)),
                "two_light_days": dict(zip(date_labels, tl_cdis)),
                "rest_day": dict(zip(date_labels, rd_cdis)),
            },
            recovery_signal_used=round(recovery_signal, 3),
            load_signal_used=round(load_signal, 3),
            days_of_history=days_of_history,
            is_meaningful=True,
        )

    except Exception:
        return _EMPTY


# ─── Formatting ──────────────────────────────────────────────────────────────

_TIER_EMOJI = {
    "surplus":  "🟢",
    "balanced": "🟡",
    "loading":  "🟠",
    "fatigued": "🔴",
    "critical": "🚨",
}


def _days_label(n: Optional[int]) -> str:
    """Format days count as a human label, e.g. '3 days', '1 day', 'not within 10d'."""
    if n is None:
        return ">10d"
    return f"{n}d"


def format_recovery_line(plan: RecoveryPlan) -> str:
    """
    Compact one-liner for nightly digest insertion.

    Example:
        🔋 Recovery path: 1 light day → balanced in 2d (vs 6d at current pace)
    """
    if not plan.is_meaningful:
        return ""

    sq_label = _days_label(plan.days_to_balanced_status_quo)

    # Pick the best intervention scenario
    best_days = None
    best_label = None
    options = [
        (plan.days_to_balanced_rest_day, "rest day"),
        (plan.days_to_balanced_two_light, "2 light days"),
        (plan.days_to_balanced_light_day, "1 light day"),
    ]
    for days, label in options:
        if days is not None:
            if best_days is None or days < best_days:
                best_days = days
                best_label = label

    emoji = _TIER_EMOJI.get(plan.today_tier, "🔶")

    if best_days is not None and (
        plan.days_to_balanced_status_quo is None
        or best_days < plan.days_to_balanced_status_quo
    ):
        sq_part = f" (vs {sq_label} at current pace)" if plan.days_to_balanced_status_quo != best_days else ""
        return (
            f"{emoji} Recovery path: {best_label} → balanced in "
            f"{_days_label(best_days)}{sq_part}"
        )
    elif plan.days_to_balanced_status_quo is not None:
        return f"{emoji} Recovery path: balanced in {sq_label} at current pace"
    else:
        return f"{emoji} Recovery path: CDI won't recover within 10d without load reduction"


def format_recovery_section(plan: RecoveryPlan) -> str:
    """
    Full Slack-formatted section for weekly summary or standalone display.

    Example:
        *🔋 Recovery Planner* — CDI 72 (Fatigued)
        At this pace:      balanced in 6 days
        1 light day:       balanced in 4 days  ✓ recommended
        2 light days:      balanced in 2 days
        Full rest day:     balanced in 1 day

        → Protect 2 consecutive light days to recover in 2 days vs 6 days without.
    """
    if not plan.is_meaningful:
        return ""

    tier_emoji = _TIER_EMOJI.get(plan.today_tier, "🔶")
    tier_label = plan.today_tier.capitalize()

    lines = [
        f"*🔋 Recovery Planner* — CDI {round(plan.today_cdi)} ({tier_label})",
        "",
    ]

    # Scenario table
    sq_label = _days_label(plan.days_to_balanced_status_quo)
    ld_label = _days_label(plan.days_to_balanced_light_day)
    tl_label = _days_label(plan.days_to_balanced_two_light)
    rd_label = _days_label(plan.days_to_balanced_rest_day)

    recommended = plan.recommended_action

    def _marker(scenario_key: str) -> str:
        labels_map = {
            "maintain current pace": "status quo",
            "make tomorrow a light day": "1 light day",
            "protect one light day": "1 light day",
            "protect 2 consecutive light days": "2 light days",
            "take a full rest day": "rest day",
        }
        mapped = labels_map.get(recommended, "")
        if mapped == scenario_key:
            return "  ← recommended"
        return ""

    lines.append(f"  At this pace:      balanced in {sq_label}{_marker('status quo')}")
    lines.append(f"  1 light day:       balanced in {ld_label}{_marker('1 light day')}")
    lines.append(f"  2 light days:      balanced in {tl_label}{_marker('2 light days')}")
    lines.append(f"  Full rest day:     balanced in {rd_label}{_marker('rest day')}")
    lines.append("")
    lines.append(f"→ _{plan.recommendation_detail}_")

    return "\n".join(lines)


def format_recovery_terminal(plan: RecoveryPlan) -> str:
    """
    ANSI-coloured terminal output for scripts/report.py.
    """
    if not plan.is_meaningful:
        return ""

    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    tier_colour = {
        "surplus":  GREEN,
        "balanced": GREEN,
        "loading":  YELLOW,
        "fatigued": RED,
        "critical": RED,
    }.get(plan.today_tier, "")

    lines = [
        f"\n  {BOLD}🔋 Recovery Planner{RESET}  "
        f"{DIM}CDI {round(plan.today_cdi)} — "
        f"{tier_colour}{plan.today_tier.capitalize()}{RESET}",
        "",
    ]

    def _fmt_days(n: Optional[int]) -> str:
        if n is None:
            return f"{RED}>10d{RESET}"
        if n <= 2:
            return f"{GREEN}{n}d{RESET}"
        if n <= 5:
            return f"{YELLOW}{n}d{RESET}"
        return f"{RED}{n}d{RESET}"

    recommended = plan.recommended_action

    def _rec_marker(scenario_key: str) -> str:
        labels_map = {
            "maintain current pace": "status quo",
            "make tomorrow a light day": "1 light day",
            "protect one light day": "1 light day",
            "protect 2 consecutive light days": "2 light days",
            "take a full rest day": "rest day",
        }
        if labels_map.get(recommended, "") == scenario_key:
            return f"  {GREEN}← recommended{RESET}"
        return ""

    lines.append(f"  At this pace:    {_fmt_days(plan.days_to_balanced_status_quo)}{_rec_marker('status quo')}")
    lines.append(f"  1 light day:     {_fmt_days(plan.days_to_balanced_light_day)}{_rec_marker('1 light day')}")
    lines.append(f"  2 light days:    {_fmt_days(plan.days_to_balanced_two_light)}{_rec_marker('2 light days')}")
    lines.append(f"  Full rest day:   {_fmt_days(plan.days_to_balanced_rest_day)}{_rec_marker('rest day')}")
    lines.append("")
    lines.append(f"  {DIM}→ {plan.recommendation_detail}{RESET}")
    lines.append("")

    return "\n".join(lines)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Presence Tracker — CDI Recovery Planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 analysis/recovery_planner.py              # Today
  python3 analysis/recovery_planner.py 2026-03-14   # Specific date
  python3 analysis/recovery_planner.py --json       # Machine-readable JSON
        """,
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="Anchor date (YYYY-MM-DD). Defaults to latest available date.",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    # Default to latest available date
    if args.date:
        date_str = args.date
    else:
        from engine.store import list_available_dates
        dates = list_available_dates()
        if not dates:
            print("No data available.", file=sys.stderr)
            sys.exit(1)
        date_str = sorted(dates)[-1]

    plan = compute_recovery_plan(date_str)

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return

    if not plan.is_meaningful:
        print(f"\n  CDI is {plan.today_tier} ({round(plan.today_cdi)}/100) — no recovery planning needed.")
        return

    print(format_recovery_terminal(plan))


if __name__ == "__main__":
    main()

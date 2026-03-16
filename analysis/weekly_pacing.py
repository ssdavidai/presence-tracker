"""
Presence Tracker — Weekly Cognitive Pacing (v22)

Answers: *"How should David pace himself across the week ahead?"*

The daily FocusPlanner answers "what should I do tomorrow?".
The morning brief answers "how ready am I today?".
The CDI answers "how much fatigue am I carrying?".

None of these answer the strategic question:

  "Given my current state and what's on my calendar this week,
   how should I pace Monday through Friday to maximise output
   without accumulating cognitive debt?"

The Weekly Pacing module closes this gap. It:

  1. Looks at the next 5 working days (Mon–Fri of the current or upcoming week)
  2. Fetches each day's calendar events via the existing gcal collector
  3. Predicts each day's cognitive load from calendar meetings + historical patterns
  4. Classifies each day as PUSH / STEADY / PROTECT based on predicted load + CDI state
  5. Recommends focus blocks and cognitive strategy per day
  6. Computes a weekly pacing strategy: which days to go deep, which to hold back

## Day Classification

Each weekday gets a pacing label based on predicted CLS and current CDI:

| Predicted CLS | CDI tier      | Day type  | Meaning |
|---------------|---------------|-----------|---------|
| < 0.25        | any           | PUSH      | Light calendar — go deep, tackle hardest work |
| 0.25–0.45     | surplus/bal   | STEADY    | Moderate load — good sustainable output day |
| 0.25–0.45     | fatigued/crit | PROTECT   | Moderate load but debt is high — pace carefully |
| > 0.45        | surplus/bal   | STEADY    | Heavy calendar — manage energy, use gaps wisely |
| > 0.45        | fatigued/crit | PROTECT   | Heavy calendar + high debt — protect this day |

## Weekly Strategy

After classifying all days, the module computes an overall weekly strategy:

- PUSH WEEK: ≥3 PUSH days → go hard all week, accumulate positive debt
- BALANCED WEEK: mix of push/steady → good default; protect one recovery slot
- PROTECT WEEK: ≥3 PROTECT days → debt risk; reduce discretionary load
- TRANSITION WEEK: moving from fatigued to recovering; progressive overload

## Output

    WeeklyPacingPlan dataclass:
      - week_start: str                 — Monday's date (YYYY-MM-DD)
      - week_end: str                   — Friday's date
      - days: list[DayPacingProfile]    — one per working day
      - strategy: str                   — PUSH|BALANCED|PROTECT|TRANSITION
      - strategy_headline: str          — 1 sentence summary
      - weekly_load_forecast: float     — mean predicted CLS across the week
      - push_days: list[str]            — dates classified as PUSH
      - protect_days: list[str]         — dates classified as PROTECT
      - is_meaningful: bool             — False when insufficient data
      - cdi_context: str                — CDI tier used for classification
      - days_of_history: int            — history days used for load forecasts

    DayPacingProfile dataclass:
      - date_str: str
      - weekday: str                    — "Monday", "Tuesday", etc.
      - meeting_minutes: int            — scheduled meeting time
      - predicted_cls: float | None     — from load_forecast
      - cls_label: str                  — Very light | Light | Moderate | High
      - day_type: str                   — PUSH | STEADY | PROTECT
      - focus_hours: float              — estimated free focus hours
      - strategy_note: str              — 1 sentence recommendation
      - calendar_events: int            — number of events
      - calendar_available: bool        — False when calendar fetch failed

## API

    from analysis.weekly_pacing import compute_weekly_pacing, format_weekly_pacing_section

    plan = compute_weekly_pacing(today_date_str)
    section = format_weekly_pacing_section(plan)   # Slack markdown
    line    = format_weekly_pacing_line(plan)       # compact one-liner

## CLI

    python3 analysis/weekly_pacing.py              # This week's plan
    python3 analysis/weekly_pacing.py --json       # Machine-readable JSON
    python3 analysis/weekly_pacing.py 2026-03-17   # Plan starting from this date

## Design

- Pure functions where possible — testable without live calendar data
- Calendar fetch via existing collectors.gcal.collect() with graceful fallback
- Load prediction via analysis.load_forecast.compute_load_forecast()
- CDI via analysis.cognitive_debt.compute_cdi()
- No ML, no black boxes — rule-based classification, fully auditable
- Graceful degradation at every step: calendar unavailable → use meeting_minutes=0
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.cognitive_debt import compute_cdi
from analysis.load_forecast import compute_load_forecast
from engine.store import list_available_dates


# ─── Constants ────────────────────────────────────────────────────────────────

# CLS thresholds for day classification
CLS_LIGHT_THRESHOLD = 0.25       # Below → PUSH candidate
CLS_MODERATE_THRESHOLD = 0.45    # Above this → heavy day

# CDI tiers that indicate elevated fatigue (push harder with these = risky)
FATIGUED_CDI_TIERS = {"fatigued", "critical"}

# Minimum working hours detected as free focus time
# (based on 8h working day minus meetings, adjusted down 20% for overhead)
WORKING_HOURS = 8.0
OVERHEAD_FACTOR = 0.80  # 20% overhead on "free" hours


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class DayPacingProfile:
    """Cognitive pacing profile for a single working day."""
    date_str: str
    weekday: str                        # "Monday", "Tuesday", etc.
    meeting_minutes: int                # Total scheduled meeting time
    predicted_cls: Optional[float]      # From load_forecast
    cls_label: str                      # "Very light" | "Light" | "Moderate" | "High" | "Very high"
    day_type: str                       # "PUSH" | "STEADY" | "PROTECT"
    focus_hours: float                  # Estimated free focused hours
    strategy_note: str                  # 1-sentence recommendation
    calendar_events: int = 0            # Number of events fetched
    calendar_available: bool = True     # False when gcal fetch failed
    cls_confidence: str = "none"        # "high" | "medium" | "low" | "none"

    def to_dict(self) -> dict:
        return {
            "date_str": self.date_str,
            "weekday": self.weekday,
            "meeting_minutes": self.meeting_minutes,
            "predicted_cls": round(self.predicted_cls, 3) if self.predicted_cls is not None else None,
            "cls_label": self.cls_label,
            "day_type": self.day_type,
            "focus_hours": round(self.focus_hours, 1),
            "strategy_note": self.strategy_note,
            "calendar_events": self.calendar_events,
            "calendar_available": self.calendar_available,
            "cls_confidence": self.cls_confidence,
        }


@dataclass
class WeeklyPacingPlan:
    """Full weekly cognitive pacing plan."""
    week_start: str                         # Monday YYYY-MM-DD
    week_end: str                           # Friday YYYY-MM-DD
    days: list[DayPacingProfile] = field(default_factory=list)
    strategy: str = "BALANCED"              # PUSH | BALANCED | PROTECT | TRANSITION
    strategy_headline: str = ""             # 1-sentence strategy summary
    weekly_load_forecast: Optional[float] = None  # Mean predicted CLS
    push_days: list[str] = field(default_factory=list)      # Dates of PUSH days
    protect_days: list[str] = field(default_factory=list)   # Dates of PROTECT days
    is_meaningful: bool = False
    cdi_context: str = "balanced"           # CDI tier used for classification
    cdi_score: Optional[float] = None       # CDI 0–100 for context
    days_of_history: int = 0                # JSONL days available

    def to_dict(self) -> dict:
        return {
            "week_start": self.week_start,
            "week_end": self.week_end,
            "days": [d.to_dict() for d in self.days],
            "strategy": self.strategy,
            "strategy_headline": self.strategy_headline,
            "weekly_load_forecast": (
                round(self.weekly_load_forecast, 3)
                if self.weekly_load_forecast is not None
                else None
            ),
            "push_days": self.push_days,
            "protect_days": self.protect_days,
            "is_meaningful": self.is_meaningful,
            "cdi_context": self.cdi_context,
            "cdi_score": round(self.cdi_score, 1) if self.cdi_score is not None else None,
            "days_of_history": self.days_of_history,
        }


# ─── Week date helpers ────────────────────────────────────────────────────────

def _get_week_dates(from_date_str: str) -> list[str]:
    """
    Return the 5 working day (Mon–Fri) dates for the week containing or
    starting from from_date_str.

    If from_date_str is a Saturday or Sunday, returns the following week's
    Mon–Fri. If it's a weekday, returns that week's Mon–Fri.

    Returns a list of 5 YYYY-MM-DD strings.
    """
    dt = datetime.strptime(from_date_str, "%Y-%m-%d")
    weekday = dt.weekday()  # 0=Mon, 6=Sun

    # If Saturday (5) or Sunday (6), advance to next Monday
    if weekday >= 5:
        days_to_monday = 7 - weekday
        monday = dt + timedelta(days=days_to_monday)
    else:
        # Find the Monday of the current week
        monday = dt - timedelta(days=weekday)

    return [(monday + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]


def _weekday_name(date_str: str) -> str:
    """Return the full weekday name for a YYYY-MM-DD date string."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%A")


# ─── Calendar loading ─────────────────────────────────────────────────────────

def _fetch_calendar_safe(date_str: str) -> Optional[dict]:
    """
    Fetch calendar data for a single day with full error isolation.

    Returns the raw collect() dict on success, None on any failure.
    Never raises — used in hot path.
    """
    try:
        from collectors.gcal import collect
        cal = collect(date_str)
        return cal
    except Exception:
        return None


# ─── Load prediction ──────────────────────────────────────────────────────────

def _predict_day_load(
    date_str: str,
    calendar: Optional[dict],
    history_ref_date: str,
) -> tuple[Optional[float], str, str, str]:
    """
    Predict a day's CLS from its calendar.

    Returns (predicted_cls, cls_label, confidence, narrative).
    predicted_cls is None when load_forecast returns is_meaningful=False.
    """
    forecast = compute_load_forecast(history_ref_date, calendar)
    if not forecast.is_meaningful or forecast.predicted_cls is None:
        # Fallback: derive a rough estimate from meeting minutes alone
        meeting_minutes = (calendar or {}).get("total_meeting_minutes", 0) or 0
        cls_estimate = min(0.15 + (meeting_minutes / 600), 0.80)
        label = _cls_to_label(cls_estimate)
        return cls_estimate, label, "none", ""

    return (
        forecast.predicted_cls,
        forecast.load_label,
        forecast.confidence,
        forecast.narrative,
    )


def _cls_to_label(cls: float) -> str:
    """Map a CLS value to a human label."""
    if cls < 0.20:
        return "Very light"
    elif cls < 0.30:
        return "Light"
    elif cls < 0.50:
        return "Moderate"
    elif cls < 0.70:
        return "High"
    else:
        return "Very high"


# ─── Day classification ───────────────────────────────────────────────────────

def _classify_day(
    predicted_cls: Optional[float],
    cdi_tier: str,
    meeting_minutes: int,
) -> str:
    """
    Classify a day as PUSH / STEADY / PROTECT.

    Rules:
    - PUSH: low predicted load (< 0.25) regardless of CDI — light calendar day
    - PROTECT: high CDI (fatigued/critical) AND moderate+ load (≥ 0.25)
    - STEADY: everything else — sustainable output
    """
    if predicted_cls is None:
        # No forecast available: use meeting minutes heuristic
        if meeting_minutes < 60:
            return "PUSH"
        elif cdi_tier in FATIGUED_CDI_TIERS:
            return "PROTECT"
        return "STEADY"

    if predicted_cls < CLS_LIGHT_THRESHOLD:
        return "PUSH"
    elif predicted_cls >= CLS_MODERATE_THRESHOLD and cdi_tier in FATIGUED_CDI_TIERS:
        return "PROTECT"
    elif predicted_cls >= CLS_LIGHT_THRESHOLD and cdi_tier in FATIGUED_CDI_TIERS:
        return "PROTECT"
    else:
        return "STEADY"


def _estimate_focus_hours(meeting_minutes: int) -> float:
    """
    Estimate available focused work hours given meeting minutes.

    Assumes 8-hour working day; 20% overhead per free hour (context switching,
    admin, transitions); focus hours = (480 - meeting_minutes) / 60 × 0.80.
    Clamped to [0, 6.5].
    """
    free_minutes = max(0, WORKING_HOURS * 60 - meeting_minutes)
    focus_hours = (free_minutes / 60) * OVERHEAD_FACTOR
    return round(min(focus_hours, 6.5), 1)


def _strategy_note(
    day_type: str,
    weekday: str,
    predicted_cls: Optional[float],
    focus_hours: float,
    meeting_minutes: int,
    cdi_tier: str,
) -> str:
    """Generate a 1-sentence strategy note for the day."""
    meeting_h = meeting_minutes // 60
    meeting_m = meeting_minutes % 60
    meeting_str = (
        f"{meeting_h}h{meeting_m:02d}m" if meeting_h else f"{meeting_m}m"
    ) if meeting_minutes > 0 else "no meetings"

    if day_type == "PUSH":
        is_fatigued = cdi_tier in FATIGUED_CDI_TIERS
        if focus_hours >= 4.0:
            if is_fatigued:
                return (
                    f"{weekday}: Light calendar ({meeting_str}) despite elevated CDI — "
                    f"good day for moderate deep work. Avoid over-extending."
                )
            return (
                f"{weekday}: Light day ({meeting_str}) — "
                f"ideal for deep work. Front-load your hardest thinking."
            )
        else:
            return (
                f"{weekday}: Relatively open ({meeting_str}) — "
                f"use the ~{focus_hours:.0f}h of free time for focused output."
            )

    elif day_type == "PROTECT":
        if cdi_tier == "critical":
            return (
                f"{weekday}: High debt + busy calendar ({meeting_str}) — "
                f"protect energy. Do only what must be done today."
            )
        elif meeting_minutes >= 180:
            return (
                f"{weekday}: Heavy meetings ({meeting_str}) while in debt — "
                f"keep commitments, defer discretionary tasks."
            )
        else:
            return (
                f"{weekday}: Moderate load ({meeting_str}) but fatigue is elevated — "
                f"pace carefully, take real breaks."
            )

    else:  # STEADY
        if meeting_minutes >= 180:
            return (
                f"{weekday}: Meeting-heavy ({meeting_str}) — "
                f"protect your {focus_hours:.0f}h focus windows, batch admin."
            )
        elif meeting_minutes > 0:
            return (
                f"{weekday}: Balanced day ({meeting_str}) — "
                f"good output day. Mix deep work with meetings."
            )
        else:
            return (
                f"{weekday}: Clear calendar — "
                f"strong focus day. Set an ambitious goal and protect the morning."
            )


# ─── Weekly strategy ──────────────────────────────────────────────────────────

def _compute_weekly_strategy(
    days: list[DayPacingProfile],
    cdi_tier: str,
    cdi_score: Optional[float],
) -> tuple[str, str]:
    """
    Determine the overall weekly strategy label and headline sentence.

    Returns (strategy, headline).
    """
    push_count = sum(1 for d in days if d.day_type == "PUSH")
    protect_count = sum(1 for d in days if d.day_type == "PROTECT")
    steady_count = sum(1 for d in days if d.day_type == "STEADY")

    total = len(days) or 1

    cdi_score_str = f"{int(cdi_score)}/100" if cdi_score is not None else "?"

    if protect_count >= 3:
        strategy = "PROTECT"
        if cdi_tier in FATIGUED_CDI_TIERS:
            headline = (
                f"Protect week ahead — CDI is {cdi_tier} ({cdi_score_str}). "
                f"Three or more heavy or demanding days ahead; reduce discretionary load and "
                f"prioritise sleep."
            )
        else:
            headline = (
                "Demanding week ahead — most days have heavy meeting load. "
                "Batch admin tasks, protect your focus windows, and watch for fatigue build-up."
            )
    elif push_count >= 3:
        strategy = "PUSH"
        headline = (
            f"Strong week ahead — {push_count} light days and well-positioned CDI. "
            "Front-load your most cognitively demanding projects while energy is available."
        )
    elif cdi_tier in FATIGUED_CDI_TIERS and push_count > 0:
        strategy = "TRANSITION"
        headline = (
            f"Recovery week — CDI is {cdi_tier}. Use the {push_count} light day(s) "
            f"to rebuild capacity. Don't over-commit on {protect_count} heavier day(s)."
        )
    else:
        strategy = "BALANCED"
        push_str = f"{push_count} push" if push_count > 0 else "no dedicated push"
        protect_str = f"{protect_count} protect" if protect_count > 0 else "no protection"
        headline = (
            f"Balanced week — {push_str} day(s), {steady_count} steady, {protect_str} day(s). "
            "Sustainable pace: maintain deep work discipline on open days."
        )

    return strategy, headline


# ─── Main function ────────────────────────────────────────────────────────────

def compute_weekly_pacing(
    from_date_str: Optional[str] = None,
    fetch_calendar: bool = True,
) -> WeeklyPacingPlan:
    """
    Compute the weekly cognitive pacing plan for the week containing or
    starting from from_date_str.

    Parameters
    ----------
    from_date_str : str | None
        Reference date (YYYY-MM-DD). Defaults to today.
        If Saturday/Sunday, advances to the following Monday.
    fetch_calendar : bool
        Whether to call the real Google Calendar API. Set False in tests.

    Returns
    -------
    WeeklyPacingPlan
        is_meaningful=True when the plan has actionable content.
        is_meaningful=False only when data is critically insufficient.
    """
    today_str = from_date_str or datetime.now().strftime("%Y-%m-%d")

    # Get the 5 working days of the target week
    week_dates = _get_week_dates(today_str)
    week_start = week_dates[0]
    week_end = week_dates[-1]

    # Compute current CDI for pacing context
    cdi = compute_cdi(today_str)
    cdi_tier = cdi.tier
    cdi_score = cdi.cdi

    # How many days of history are available?
    available_dates = list_available_dates()
    days_of_history = len(available_dates)

    # Process each working day
    day_profiles: list[DayPacingProfile] = []

    for date_str in week_dates:
        weekday = _weekday_name(date_str)

        # Fetch calendar (with graceful fallback)
        calendar: Optional[dict] = None
        calendar_available = True
        calendar_events = 0
        meeting_minutes = 0

        if fetch_calendar:
            calendar = _fetch_calendar_safe(date_str)
            if calendar is None:
                calendar_available = False
            else:
                calendar_events = len(calendar.get("events", []))
                meeting_minutes = calendar.get("total_meeting_minutes", 0) or 0
        else:
            calendar_available = False

        # Predict load using today as the history reference point
        predicted_cls, cls_label, confidence, _narrative = _predict_day_load(
            date_str, calendar, today_str
        )

        # Classify the day
        day_type = _classify_day(predicted_cls, cdi_tier, meeting_minutes)

        # Estimate focus hours
        focus_hours = _estimate_focus_hours(meeting_minutes)

        # Strategy note
        note = _strategy_note(
            day_type, weekday, predicted_cls, focus_hours, meeting_minutes, cdi_tier
        )

        day_profiles.append(DayPacingProfile(
            date_str=date_str,
            weekday=weekday,
            meeting_minutes=meeting_minutes,
            predicted_cls=predicted_cls,
            cls_label=cls_label,
            day_type=day_type,
            focus_hours=focus_hours,
            strategy_note=note,
            calendar_events=calendar_events,
            calendar_available=calendar_available,
            cls_confidence=confidence,
        ))

    # Compute weekly aggregate
    cls_values = [
        d.predicted_cls for d in day_profiles if d.predicted_cls is not None
    ]
    weekly_load_forecast = sum(cls_values) / len(cls_values) if cls_values else None

    push_days = [d.date_str for d in day_profiles if d.day_type == "PUSH"]
    protect_days = [d.date_str for d in day_profiles if d.day_type == "PROTECT"]

    strategy, headline = _compute_weekly_strategy(day_profiles, cdi_tier, cdi_score)

    return WeeklyPacingPlan(
        week_start=week_start,
        week_end=week_end,
        days=day_profiles,
        strategy=strategy,
        strategy_headline=headline,
        weekly_load_forecast=weekly_load_forecast,
        push_days=push_days,
        protect_days=protect_days,
        is_meaningful=True,
        cdi_context=cdi_tier,
        cdi_score=cdi_score,
        days_of_history=days_of_history,
    )


# ─── Formatters ───────────────────────────────────────────────────────────────

# Day type emoji
_DAY_EMOJI = {
    "PUSH": "🟢",
    "STEADY": "🔵",
    "PROTECT": "🟡",
}

# Day type labels for display
_DAY_LABEL = {
    "PUSH": "PUSH",
    "STEADY": "STEADY",
    "PROTECT": "PROTECT",
}


def format_weekly_pacing_line(plan: WeeklyPacingPlan) -> str:
    """
    Format a compact one-line weekly pacing summary for Slack.

    Example:
        "📅 Week pacing: BALANCED — 2 push days, 1 protect. Avg CLS ~0.31"
    """
    if not plan.is_meaningful:
        return ""

    push_str = f"{len(plan.push_days)} push" if plan.push_days else "no push days"
    protect_str = f"{len(plan.protect_days)} protect" if plan.protect_days else "no protect days"
    cls_str = f" · avg CLS ~{plan.weekly_load_forecast:.2f}" if plan.weekly_load_forecast else ""

    return (
        f"📅 *Week pacing ({plan.week_start}):* {plan.strategy} — "
        f"{push_str}, {protect_str}{cls_str}"
    )


def format_weekly_pacing_section(plan: WeeklyPacingPlan) -> str:
    """
    Format the full weekly pacing plan as a Slack markdown section.

    Example:
        📅 *Weekly Pacing — Mon 17 Mar → Fri 21 Mar*

        Balanced week — 2 push day(s), 2 steady, 1 protect day(s).

        🟢 *Mon 17*  PUSH    · 0 meetings  · ~6.4h focus
           Front-load your hardest thinking.
        🔵 *Tue 18*  STEADY  · 1h30m mtgs  · ~5.2h focus
           Balanced day — mix deep work with meetings.
        🟡 *Wed 19*  PROTECT · 3h30m mtgs  · ~3.6h focus
           Heavy meetings while in debt — defer discretionary tasks.
        🔵 *Thu 20*  STEADY  · 1h mtgs     · ~5.6h focus
           Good output day.
        🟢 *Fri 21*  PUSH    · 30m mtgs    · ~6.0h focus
           Use Friday's open calendar to wrap up or get ahead.

        CDI: balanced (43/100) · 2 days history
    """
    if not plan.is_meaningful:
        return ""

    week_start_dt = datetime.strptime(plan.week_start, "%Y-%m-%d")
    week_end_dt = datetime.strptime(plan.week_end, "%Y-%m-%d")
    week_str = (
        f"Mon {week_start_dt.strftime('%-d %b')} → "
        f"Fri {week_end_dt.strftime('%-d %b')}"
    )

    lines = [
        f"📅 *Weekly Pacing — {week_str}*",
        "",
        plan.strategy_headline,
        "",
    ]

    for day in plan.days:
        emoji = _DAY_EMOJI.get(day.day_type, "⚪")
        label_padded = _DAY_LABEL.get(day.day_type, day.day_type).ljust(7)

        # Format meeting minutes
        if day.meeting_minutes == 0:
            mtg_str = "no meetings"
        else:
            h = day.meeting_minutes // 60
            m = day.meeting_minutes % 60
            mtg_str = (f"{h}h{m:02d}m" if h else f"{m}m") + " mtgs"

        # Date label
        day_dt = datetime.strptime(day.date_str, "%Y-%m-%d")
        day_label = day_dt.strftime("%a %-d")  # e.g. "Mon 17"

        # Calendar availability caveat
        cal_note = "" if day.calendar_available else " _(no calendar)_"

        lines.append(
            f"{emoji} *{day_label}*  {label_padded} · {mtg_str}  · ~{day.focus_hours:.0f}h focus{cal_note}"
        )
        lines.append(f"   _{day.strategy_note}_")

    # Footer
    lines.append("")
    cdi_str = f"{plan.cdi_context} ({int(plan.cdi_score)}/100)" if plan.cdi_score is not None else plan.cdi_context
    avg_cls_str = f" · avg CLS ~{plan.weekly_load_forecast:.2f}" if plan.weekly_load_forecast else ""
    lines.append(
        f"_CDI: {cdi_str}{avg_cls_str} · {plan.days_of_history} day{'s' if plan.days_of_history != 1 else ''} history_"
    )

    return "\n".join(lines)


def format_weekly_pacing_terminal(plan: WeeklyPacingPlan) -> str:
    """Terminal-formatted weekly pacing report with ANSI colours."""
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    if not plan.is_meaningful:
        return f"\n  {BOLD}Weekly Pacing{RESET}\n  {DIM}No data available.{RESET}\n"

    week_start_dt = datetime.strptime(plan.week_start, "%Y-%m-%d")
    week_end_dt = datetime.strptime(plan.week_end, "%Y-%m-%d")

    lines = [
        "",
        f"{BOLD}Weekly Pacing — "
        f"{week_start_dt.strftime('%a %-d %b')} → {week_end_dt.strftime('%a %-d %b %Y')}{RESET}",
        "=" * 65,
        "",
        f"  Strategy: {BOLD}{plan.strategy}{RESET}  —  {plan.strategy_headline}",
        "",
    ]

    type_colour = {
        "PUSH": GREEN,
        "STEADY": BLUE,
        "PROTECT": YELLOW,
    }

    for day in plan.days:
        colour = type_colour.get(day.day_type, "")
        day_dt = datetime.strptime(day.date_str, "%Y-%m-%d")
        day_label = day_dt.strftime("%a %-d %b")

        h = day.meeting_minutes // 60
        m = day.meeting_minutes % 60
        if day.meeting_minutes == 0:
            mtg_str = "no meetings"
        else:
            mtg_str = (f"{h}h{m:02d}m" if h else f"{m}m") + " meetings"

        cal_note = "" if day.calendar_available else " (no calendar)"

        lines.append(
            f"  {colour}{day.day_type:<8}{RESET}  "
            f"{BOLD}{day_label:<12}{RESET}  "
            f"{mtg_str:<18}  ~{day.focus_hours:.0f}h focus{cal_note}"
        )
        lines.append(f"            {DIM}{day.strategy_note}{RESET}")
        lines.append("")

    # Summary
    cls_str = f"avg CLS ~{plan.weekly_load_forecast:.2f}" if plan.weekly_load_forecast else "no CLS data"
    cdi_str = f"CDI {plan.cdi_context} ({int(plan.cdi_score) if plan.cdi_score is not None else '?'}/100)"
    lines.append(f"  {DIM}{cdi_str}  ·  {cls_str}  ·  {plan.days_of_history} days history{RESET}")
    lines.append("")

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.

    Usage:
        python3 analysis/weekly_pacing.py              # This week
        python3 analysis/weekly_pacing.py 2026-03-17   # Week starting from date
        python3 analysis/weekly_pacing.py --json        # JSON output
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Weekly Cognitive Pacing — how to pace the week ahead"
    )
    parser.add_argument("date", nargs="?", help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    print(f"[weekly_pacing] Computing weekly pacing plan from {date_str}...", flush=True)
    plan = compute_weekly_pacing(date_str)

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return

    print(format_weekly_pacing_terminal(plan))


if __name__ == "__main__":
    main()

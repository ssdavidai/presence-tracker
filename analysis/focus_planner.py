"""
Presence Tracker — Focus Planner

Answers: *"When should David block focus time tomorrow?"*

The Focus Planner closes the loop from descriptive analytics → actionable scheduling.
It takes three inputs:

  1. **Historical focus pattern** — David's actual peak-FDI hours from the JSONL store
  2. **Tomorrow's calendar** — free windows that aren't already blocked by meetings
  3. **Current state** — CDI (cognitive debt tier) + WHOOP recovery trend

And outputs specific, ranked focus block recommendations for the next workday.

## Why this matters

The morning brief already answers: "How ready am I today?"
The CDI answers: "How much fatigue am I carrying?"
The DPS trend answers: "Are my days getting better or worse?"

But none of these answer: "What should I actually *do* with tomorrow's schedule?"

The Focus Planner bridges analytics → action:
  - "Your peak focus window historically is 9–11am. Tomorrow it's free. Block it."
  - "You're in CDI 'fatigued' tier — only schedule one deep block, max 90 minutes."
  - "Tomorrow is meeting-heavy. Your only real focus window is 8:00–9:30am."

## Focus block criteria

A candidate focus block must:
  - Fall within working hours (8am–7pm, Budapest time)
  - Be at least 45 minutes of contiguous free time (no meetings)
  - Not overlap with any existing calendar event

A block is "quality" if it:
  - Aligns with a historically high-FDI hour (from stored data)
  - Starts before 13:00 (morning cognitive advantage)

## CDI modifier

CDI tier adjusts the recommended block count and max duration:

  | CDI Tier   | Max blocks | Max duration each |
  |------------|------------|-------------------|
  | surplus    | 3          | 3h                |
  | balanced   | 2          | 2h                |
  | loading    | 2          | 90min             |
  | fatigued   | 1          | 90min             |
  | critical   | 1          | 60min             |

## Output

    FocusPlan dataclass with:
      - recommended_blocks: list[FocusBlock]  — ordered best-first
      - peak_hours: list[int]                  — David's top historical hours (HH)
      - cdi_modifier: str                      — why blocks were limited/adjusted
      - summary_line: str                      — one-line Slack-ready summary
      - advisory: str                          — one sentence extra guidance

## API

    plan_tomorrow_focus(today_date_str) → FocusPlan | None
    format_focus_plan_section(plan) → str   (Slack markdown section)

"""

import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import list_available_dates, read_day

# ─── Constants ────────────────────────────────────────────────────────────────

# Working hours range: blocks are only scheduled here
WORK_START_HOUR = 8
WORK_END_HOUR = 19

# Minimum contiguous free minutes to consider as a focus block
MIN_BLOCK_MINUTES = 45

# Historical lookback for computing peak-FDI hour profile
FOCUS_HISTORY_DAYS = 30

# Minimum active windows in an hour to include in hourly profile
MIN_WINDOWS_FOR_HOUR_PROFILE = 2

# Timezone offset for Budapest (UTC+1 winter, UTC+2 summer)
# We'll detect the offset from stored window timestamps rather than hardcoding.

# CDI tier → (max_blocks, max_duration_minutes)
CDI_LIMITS: dict[str, tuple[int, int]] = {
    "surplus":  (3, 180),
    "balanced": (2, 120),
    "loading":  (2, 90),
    "fatigued": (1, 90),
    "critical": (1, 60),
}

# Default limits when CDI is unavailable
CDI_DEFAULT_LIMITS = (2, 120)

# Hours before 13:00 are considered "morning preference"
MORNING_CUTOFF = 13


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class FocusBlock:
    """A single recommended focus block."""
    start_hour: int      # 0-23
    start_minute: int    # 0, 15, 30, 45 (15-min resolution)
    duration_minutes: int
    label: str           # e.g. "9:00–10:30"
    quality: str         # 'peak' | 'good' | 'ok'
    reason: str          # e.g. "historically your best focus hour + free window"
    is_morning: bool     # True if before MORNING_CUTOFF
    fdi_score: Optional[float] = None   # historical avg FDI for this hour (0–1) if known

    def to_dict(self) -> dict:
        return {
            "start_hour": self.start_hour,
            "start_minute": self.start_minute,
            "duration_minutes": self.duration_minutes,
            "label": self.label,
            "quality": self.quality,
            "reason": self.reason,
            "is_morning": self.is_morning,
            "fdi_score": self.fdi_score,
        }


@dataclass
class FocusPlan:
    """Tomorrow's recommended focus schedule."""
    date_str: str
    recommended_blocks: list[FocusBlock] = field(default_factory=list)
    peak_hours: list[int] = field(default_factory=list)   # top historical focus hours
    cdi_tier: Optional[str] = None
    cdi_modifier: str = ""           # explanation of CDI-driven limit
    summary_line: str = ""           # one-liner for Slack
    advisory: str = ""               # extra guidance sentence
    is_meaningful: bool = True
    days_of_history: int = 0

    def to_dict(self) -> dict:
        return {
            "date_str": self.date_str,
            "recommended_blocks": [b.to_dict() for b in self.recommended_blocks],
            "peak_hours": self.peak_hours,
            "cdi_tier": self.cdi_tier,
            "cdi_modifier": self.cdi_modifier,
            "summary_line": self.summary_line,
            "advisory": self.advisory,
            "is_meaningful": self.is_meaningful,
            "days_of_history": self.days_of_history,
        }


# ─── Peak-hour profile ────────────────────────────────────────────────────────

def _build_hourly_fdi_profile(today_date_str: str, days: int = FOCUS_HISTORY_DAYS) -> dict[int, float]:
    """
    Compute average active-window FDI per hour of day using historical data.

    Returns a dict mapping hour (0–23) → mean FDI across all active windows
    in that hour, drawn from up to `days` days of JSONL history prior to
    today_date_str.

    Only hours with MIN_WINDOWS_FOR_HOUR_PROFILE or more active windows are
    included — we need statistical reliability before making recommendations.

    Returns empty dict when insufficient history exists.
    """
    try:
        today = datetime.strptime(today_date_str, "%Y-%m-%d").date()
    except ValueError:
        return {}

    available = list_available_dates()
    # Dates strictly before today, sorted descending, up to `days`
    history = sorted(
        [d for d in available if d < today_date_str],
        reverse=True,
    )[:days]

    hourly_fdi: dict[int, list[float]] = {}

    for d in history:
        try:
            windows = read_day(d)
        except Exception:
            continue

        for w in windows:
            meta = w.get("metadata", {}) or {}
            if not meta.get("is_active_window"):
                continue
            hour = meta.get("hour_of_day")
            if hour is None:
                continue
            fdi = (w.get("metrics") or {}).get("focus_depth_index")
            if fdi is None:
                continue
            hourly_fdi.setdefault(hour, []).append(fdi)

    profile = {}
    for hour, vals in hourly_fdi.items():
        if len(vals) >= MIN_WINDOWS_FOR_HOUR_PROFILE:
            profile[hour] = sum(vals) / len(vals)

    return profile


def _top_focus_hours(profile: dict[int, float], n: int = 3) -> list[int]:
    """Return the top-n hours by historical FDI, within working hours."""
    working = {h: v for h, v in profile.items() if WORK_START_HOUR <= h < WORK_END_HOUR}
    return [h for h, _ in sorted(working.items(), key=lambda x: x[1], reverse=True)[:n]]


# ─── Free-window detection ────────────────────────────────────────────────────

def _get_free_blocks(calendar_data: dict, target_date_str: str) -> list[dict]:
    """
    Find contiguous free blocks in tomorrow's working hours.

    Uses 15-minute resolution to find gaps between existing calendar events.
    Returns a list of dicts: {start_hour, start_minute, duration_minutes}
    sorted chronologically.

    Only blocks with duration >= MIN_BLOCK_MINUTES are returned.
    """
    events = calendar_data.get("events", []) or []

    # Build a set of occupied 15-minute slots (as (hour, quarter) tuples)
    occupied: set[tuple[int, int]] = set()

    for e in events:
        if e.get("is_all_day"):
            continue
        start_str = e.get("start")
        end_str = e.get("end")
        if not start_str or not end_str:
            continue
        try:
            start_dt = datetime.fromisoformat(start_str)
            end_dt = datetime.fromisoformat(end_str)
        except (ValueError, TypeError):
            continue

        # Walk through the event in 15-minute increments
        cursor = start_dt
        while cursor < end_dt:
            h = cursor.hour
            q = cursor.minute // 15
            if WORK_START_HOUR <= h < WORK_END_HOUR:
                occupied.add((h, q))
            cursor += timedelta(minutes=15)

    # Find all free 15-minute slots within working hours
    all_slots = []
    for h in range(WORK_START_HOUR, WORK_END_HOUR):
        for q in range(4):
            if (h, q) not in occupied:
                all_slots.append((h, q))

    if not all_slots:
        return []

    # Group contiguous free slots into blocks
    blocks = []
    block_start = all_slots[0]
    block_len = 1

    for i in range(1, len(all_slots)):
        prev_h, prev_q = all_slots[i - 1]
        curr_h, curr_q = all_slots[i]

        # Are these consecutive?
        prev_minutes = prev_h * 60 + prev_q * 15
        curr_minutes = curr_h * 60 + curr_q * 15

        if curr_minutes == prev_minutes + 15:
            block_len += 1
        else:
            # End of this block
            dur = block_len * 15
            if dur >= MIN_BLOCK_MINUTES:
                sh, sq = block_start
                blocks.append({
                    "start_hour": sh,
                    "start_minute": sq * 15,
                    "duration_minutes": dur,
                })
            block_start = curr_h, curr_q
            block_len = 1

    # Last block
    dur = block_len * 15
    if dur >= MIN_BLOCK_MINUTES:
        sh, sq = block_start
        blocks.append({
            "start_hour": sh,
            "start_minute": sq * 15,
            "duration_minutes": dur,
        })

    return blocks


# ─── Block scoring ────────────────────────────────────────────────────────────

def _score_block(
    block: dict,
    hourly_profile: dict[int, float],
    peak_hours: list[int],
) -> tuple[float, str, str, Optional[float]]:
    """
    Score a free block for focus quality.

    Returns (score, quality_label, reason_str, fdi_score).
    Higher score = better block.

    Scoring factors:
      - +3.0 if start hour is in top historical focus hours
      - +1.5 if start hour is in historical working-hours profile (any hour)
      - +2.0 if block starts before MORNING_CUTOFF (cognitive morning advantage)
      - +0.5 per 30 minutes of duration (longer is better, diminishing returns)
      - +0.0 base (so blocks always have a positive score)
    """
    start_h = block["start_hour"]
    dur = block["duration_minutes"]
    score = 0.0
    reasons = []
    quality = "ok"

    # Historical FDI for this hour
    fdi_score = hourly_profile.get(start_h)
    fdi_score_display = fdi_score

    if start_h in peak_hours:
        # Index = how top this hour is (0 = best)
        rank = peak_hours.index(start_h)
        bonus = 3.0 - rank * 0.5  # 3.0, 2.5, 2.0 for rank 0, 1, 2
        score += bonus
        if rank == 0:
            reasons.append("your #1 historical focus hour")
        elif rank == 1:
            reasons.append("your 2nd-best historical focus hour")
        else:
            reasons.append("historically a strong focus hour")
        quality = "peak" if rank == 0 else "good"
    elif start_h in hourly_profile:
        score += 1.5
        reasons.append(f"historically active (avg FDI {hourly_profile[start_h]:.0%})")
        quality = "good"
    else:
        # No historical data for this hour
        reasons.append("free window (no historical focus data yet)")

    # Morning preference
    if start_h < MORNING_CUTOFF:
        score += 2.0
        reasons.append("morning window")

    # Duration bonus: +0.5 per 30min up to 150min
    dur_bonus = min(0.5 * (dur // 30), 2.5)
    score += dur_bonus

    if not reasons:
        reasons = ["available window"]

    reason = ", ".join(reasons)
    return score, quality, reason, fdi_score_display


# ─── CDI modifier ─────────────────────────────────────────────────────────────

def _get_cdi_limits(today_date_str: str) -> tuple[int, int, Optional[str], str]:
    """
    Fetch CDI for today and return (max_blocks, max_duration_min, cdi_tier, modifier_note).
    Falls back to defaults if CDI is unavailable.
    """
    try:
        from analysis.cognitive_debt import compute_cdi
        debt = compute_cdi(today_date_str)
        if not debt.is_meaningful:
            return CDI_DEFAULT_LIMITS[0], CDI_DEFAULT_LIMITS[1], None, ""
        tier = debt.tier
        max_blocks, max_dur = CDI_LIMITS.get(tier, CDI_DEFAULT_LIMITS)
        notes = {
            "surplus":  "Energy surplus — room for up to 3 focus blocks.",
            "balanced": "Balanced load — 2 focus blocks recommended.",
            "loading":  "Load accumulating — limit blocks to 90min each.",
            "fatigued": "Cognitive fatigue detected — one focused block, then protect recovery.",
            "critical": "High debt — one short deep-work block only. Prioritize rest.",
        }
        return max_blocks, max_dur, tier, notes.get(tier, "")
    except Exception:
        return CDI_DEFAULT_LIMITS[0], CDI_DEFAULT_LIMITS[1], None, ""


# ─── Plan builder ─────────────────────────────────────────────────────────────

def plan_tomorrow_focus(
    today_date_str: str,
    tomorrow_calendar: Optional[dict] = None,
) -> Optional["FocusPlan"]:
    """
    Build a focus block plan for tomorrow.

    Parameters
    ----------
    today_date_str : str
        Today's date (YYYY-MM-DD).  Used to find tomorrow and as CDI reference.
    tomorrow_calendar : dict | None
        Raw calendar data from collectors.gcal.collect() for tomorrow.
        If None, the planner will attempt to fetch it live.

    Returns
    -------
    FocusPlan | None
        A plan with ranked focus blocks, or None on fatal error.
        An empty plan (no free windows found) returns a FocusPlan with
        is_meaningful=False.
    """
    # ── Determine tomorrow's date ──────────────────────────────────────────
    try:
        today_dt = datetime.strptime(today_date_str, "%Y-%m-%d")
        tomorrow_dt = today_dt + timedelta(days=1)
        tomorrow_str = tomorrow_dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

    # ── Fetch tomorrow's calendar if not provided ─────────────────────────
    if tomorrow_calendar is None:
        try:
            from collectors import gcal
            tomorrow_calendar = gcal.collect(tomorrow_str)
        except Exception:
            tomorrow_calendar = {"events": [], "event_count": 0, "total_meeting_minutes": 0}

    # ── Build historical hourly FDI profile ───────────────────────────────
    hourly_profile = _build_hourly_fdi_profile(today_date_str)
    peak_hours = _top_focus_hours(hourly_profile)
    days_of_history = len([
        d for d in list_available_dates()
        if d < today_date_str
    ])

    # ── Find free blocks in tomorrow's calendar ───────────────────────────
    free_blocks = _get_free_blocks(tomorrow_calendar, tomorrow_str)

    if not free_blocks:
        plan = FocusPlan(
            date_str=tomorrow_str,
            peak_hours=peak_hours,
            days_of_history=days_of_history,
            is_meaningful=False,
            summary_line="Tomorrow is fully booked — no free focus windows found.",
            advisory="Consider declining one meeting to protect at least 45 minutes of deep work.",
        )
        return plan

    # ── Get CDI limits ────────────────────────────────────────────────────
    max_blocks, max_dur, cdi_tier, cdi_modifier = _get_cdi_limits(today_date_str)

    # ── Score and rank blocks ─────────────────────────────────────────────
    scored = []
    for block in free_blocks:
        score, quality, reason, fdi_score = _score_block(block, hourly_profile, peak_hours)
        scored.append((score, quality, reason, fdi_score, block))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Build FocusBlock objects, respecting CDI limits ───────────────────
    recommended: list[FocusBlock] = []

    for score, quality, reason, fdi_score, block in scored:
        if len(recommended) >= max_blocks:
            break

        start_h = block["start_hour"]
        start_m = block["start_minute"]
        raw_dur = block["duration_minutes"]

        # Cap duration at CDI maximum
        dur = min(raw_dur, max_dur)

        # Compute label
        end_total_min = start_h * 60 + start_m + dur
        end_h = end_total_min // 60
        end_m = end_total_min % 60
        label = f"{start_h}:{start_m:02d}–{end_h}:{end_m:02d}"

        is_morning = start_h < MORNING_CUTOFF

        recommended.append(FocusBlock(
            start_hour=start_h,
            start_minute=start_m,
            duration_minutes=dur,
            label=label,
            quality=quality,
            reason=reason,
            is_morning=is_morning,
            fdi_score=fdi_score,
        ))

    # ── Build summary + advisory ──────────────────────────────────────────
    summary_line, advisory = _build_summary(recommended, cdi_tier, tomorrow_calendar)

    return FocusPlan(
        date_str=tomorrow_str,
        recommended_blocks=recommended,
        peak_hours=peak_hours,
        cdi_tier=cdi_tier,
        cdi_modifier=cdi_modifier,
        summary_line=summary_line,
        advisory=advisory,
        is_meaningful=bool(recommended),
        days_of_history=days_of_history,
    )


def _build_summary(
    blocks: list[FocusBlock],
    cdi_tier: Optional[str],
    tomorrow_calendar: dict,
) -> tuple[str, str]:
    """
    Build a one-line summary and one-sentence advisory for the plan.
    """
    if not blocks:
        return (
            "No focus windows found for tomorrow.",
            "Tomorrow is fully scheduled — consider protecting a morning block.",
        )

    total_mins = sum(b.duration_minutes for b in blocks)
    hours_str = f"{total_mins // 60}h{total_mins % 60:02d}min" if total_mins % 60 else f"{total_mins // 60}h"

    peak_blocks = [b for b in blocks if b.quality == "peak"]
    morning_blocks = [b for b in blocks if b.is_morning]

    if peak_blocks:
        best = peak_blocks[0]
        summary = f"Best focus window: {best.label} ({best.duration_minutes}min, peak FDI hour)"
    elif morning_blocks:
        best = morning_blocks[0]
        summary = f"Best focus window: {best.label} ({best.duration_minutes}min, morning window)"
    else:
        best = blocks[0]
        summary = f"Best focus window: {best.label} ({best.duration_minutes}min)"

    if len(blocks) > 1:
        summary += f" + {len(blocks)-1} more block{'s' if len(blocks)-1 > 1 else ''} ({hours_str} total)"
    else:
        summary += f" ({hours_str} deep work)"

    # Advisory
    total_meeting_mins = tomorrow_calendar.get("total_meeting_minutes", 0) or 0
    advisory = ""
    if cdi_tier in ("fatigued", "critical"):
        advisory = "Prioritise the single focus block — the rest of tomorrow should be lighter recovery time."
    elif cdi_tier == "loading":
        advisory = "Cap deep work at 90min per block; breaks between blocks are load management, not laziness."
    elif total_meeting_mins >= 240:
        advisory = "Heavy meeting day ahead — protect the focus block(s) by blocking calendar and setting Slack DND."
    elif not morning_blocks:
        advisory = "Your best windows are in the afternoon — try to minimise distraction in those slots."
    elif len(blocks) >= 2:
        advisory = "Two clear blocks available — front-load the harder task in the earlier one."
    else:
        advisory = "One clear window available — use it for your single most important task."

    return summary, advisory


# ─── Slack formatter ──────────────────────────────────────────────────────────

def format_focus_plan_section(plan: "FocusPlan") -> str:
    """
    Format the focus plan into a Slack DM section.

    Designed to slot into the morning brief or daily digest.

    Example output:
        *🎯 Tomorrow's Focus Plan:*
        • 9:00–11:00 _(120min, peak FDI hour)_  `PEAK`
        • 14:00–15:30 _(90min, afternoon window)_  `OK`
        _Cap each block at 90min (CDI: loading). One break between blocks._

    Returns empty string when plan is None or has no blocks.
    """
    if plan is None:
        return ""

    if not plan.is_meaningful or not plan.recommended_blocks:
        # Show the no-blocks advisory anyway
        if plan.summary_line:
            return f"*🎯 Tomorrow's Focus:* _{plan.summary_line}_"
        return ""

    lines = ["*🎯 Tomorrow's Focus Plan:*"]

    quality_emoji = {"peak": "🔥", "good": "✅", "ok": "🔵"}

    for block in plan.recommended_blocks:
        emoji = quality_emoji.get(block.quality, "🔵")
        dur_str = f"{block.duration_minutes}min"
        reason_short = _shorten_reason(block.reason)
        lines.append(f"• {block.label}  _({dur_str}{', ' + reason_short if reason_short else ''})_  {emoji}")

    # CDI modifier note
    if plan.cdi_modifier:
        lines.append(f"_{plan.cdi_modifier}_")

    # Advisory
    if plan.advisory:
        lines.append(f"_{plan.advisory}_")

    return "\n".join(lines)


def _shorten_reason(reason: str) -> str:
    """Shorten a reason string for compact display."""
    if "peak focus hour" in reason or "#1 historical" in reason:
        return "peak focus hour"
    if "2nd-best historical" in reason:
        return "strong focus hour"
    if "historically a strong" in reason:
        return "strong focus hour"
    if "historically active" in reason:
        return "historically active"
    if "morning window" in reason and "historical" not in reason:
        return "morning window"
    if "no historical focus data" in reason:
        return "morning window" if True else ""
    return ""


# ─── Standalone runner ────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point for manual testing.

    Usage:
        python3 analysis/focus_planner.py                  # Plan for tomorrow
        python3 analysis/focus_planner.py 2026-03-15       # Specific today date
        python3 analysis/focus_planner.py --json           # JSON output
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Focus Planner — tomorrow's deep work schedule")
    parser.add_argument("date", nargs="?", help="Today's date (YYYY-MM-DD), default = today")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    today_str = args.date or datetime.now().strftime("%Y-%m-%d")

    print(f"[focus_planner] Computing focus plan for tomorrow (today={today_str})...", flush=True)
    plan = plan_tomorrow_focus(today_str)

    if plan is None:
        print("Error: could not build focus plan.")
        sys.exit(1)

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return

    # Human-readable terminal output
    print()
    print(f"Focus Plan for {plan.date_str}")
    print("=" * 50)

    if not plan.is_meaningful or not plan.recommended_blocks:
        print(f"  {plan.summary_line}")
        if plan.advisory:
            print(f"  → {plan.advisory}")
    else:
        quality_labels = {"peak": "PEAK", "good": "GOOD", "ok": "OK"}
        for i, block in enumerate(plan.recommended_blocks, 1):
            dur_h = block.duration_minutes // 60
            dur_m = block.duration_minutes % 60
            dur_str = f"{dur_h}h{dur_m:02d}m" if dur_h else f"{dur_m}m"
            q_label = quality_labels.get(block.quality, "OK")
            fdi_str = f"  hist-FDI={block.fdi_score:.0%}" if block.fdi_score else ""
            print(f"  [{i}] {block.label}  ({dur_str})  [{q_label}]{fdi_str}")
            print(f"       {block.reason}")
            print()

        if plan.cdi_modifier:
            print(f"CDI note: {plan.cdi_modifier}")
        if plan.advisory:
            print(f"→ {plan.advisory}")

    if plan.peak_hours:
        print()
        print(f"Your peak historical focus hours: {', '.join(f'{h}:00' for h in plan.peak_hours)}")
        print(f"  (based on {plan.days_of_history} day{'s' if plan.days_of_history != 1 else ''} of history)")


if __name__ == "__main__":
    main()

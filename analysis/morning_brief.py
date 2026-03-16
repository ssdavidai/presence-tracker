"""
Presence Tracker — Morning Readiness Brief

Sends David a morning Slack DM at 07:00 Budapest time with:
- Today's WHOOP readiness (recovery score, HRV, sleep quality)
- A capacity label and day-planning recommendation
- Yesterday's cognitive load as context
- A scheduling suggestion based on physiological state
- Multi-day trend context (HRV streaks, overcapacity streaks, recovery momentum)
- Today's calendar preview with schedule-aware recommendations (v7.0)
- Cognitive Debt Index — accumulated fatigue signal (v8.0)
- DPS trend — cognitive day quality trajectory over last 7 days (v9.0)
- Predicted Cognitive Load Forecast — expected CLS based on today's calendar (v14.0)
- Cognitive Rhythm hint — peak focus hours, morning/afternoon bias, best day-of-week (v18.0)

This is the forward-looking complement to the end-of-day digest.
The digest tells you how the day went; the morning brief tells you
what kind of day to plan.

Architecture:
    1. Collect WHOOP data for today (WHOOP posts last night's data by ~6am)
    2. Load yesterday's daily summary from the store (if available)
    3. Collect today's Google Calendar events (v7.0)
    4. Compute readiness tier and calendar-aware recommendation
    5. Compute multi-day trend context from rolling summary
    6. Send DM to David

Design principle: actionable specificity.
Generic "rest today" advice is useless. The brief gives David one
concrete scheduling action based on the actual numbers — now combined
with what's actually on the calendar today.

v1.5 — Multi-day trend context in morning brief:
    The morning brief now incorporates the same trend-detection engine
    used by the daily digest.  When the rolling summary has ≥ 2 days of
    history, the brief surfaces the most significant pattern:
    - HRV decline/improvement streaks (signals accumulating stress or recovery)
    - Consecutive above-capacity days (accumulated strain warning)
    - Recovery score decline streaks (sleep/stress management signal)
    - CLS above/below personal baseline (load-pacing context)

    This closes the gap where the digest would flag "3 days of HRV decline"
    at 23:45 — too late to change the day — but the morning brief would
    show nothing.  Now David gets the same pattern intelligence at 07:00,
    when it can actually influence scheduling decisions.

v7.0 — Calendar-aware scheduling advice:
    The morning brief now fetches today's Google Calendar events and
    incorporates them into:

    1. Schedule-specific recommendations: instead of "front-load creative work
       this morning" (generic), the brief now says "You have 3h of meetings
       from 10am onwards — use the 8–9:45 window for deep work" (specific).

    2. Today's Schedule section: a compact list of today's meetings with
       times, attendee counts, and meeting-load summary so David can see the
       shape of his day at a glance.

    3. Calendar-load classification:
       - Free day (0 meetings) → full deep-work mode
       - Light day (< 90 min) → flex schedule with one anchor block
       - Moderate (90–240 min) → significant meetings, protect free blocks
       - Heavy (> 240 min) → meeting-dominant day, manage energy carefully

    The calendar data flows into _calendar_aware_recommendation(), which
    replaces _tier_recommendation() in the send path.  The original
    _tier_recommendation() is preserved for backward compatibility.

    When today's calendar cannot be fetched (network error, API timeout),
    the system falls back gracefully to the existing tier-based
    recommendation — no message is lost.
"""

import json
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GATEWAY_URL, GATEWAY_TOKEN, SLACK_DM_CHANNEL


# ─── Readiness tiers ──────────────────────────────────────────────────────────

def _readiness_tier(
    recovery: Optional[float],
    hrv: Optional[float],
    baseline=None,
) -> str:
    """
    Classify today's physiological readiness into a tier.

    Returns one of: 'peak', 'good', 'moderate', 'low', 'recovery'

    Tier logic uses WHOOP recovery as the primary signal and HRV as a
    secondary confirmation.  This mirrors the WHOOP UX but adds nuance:
    moderate recovery with low HRV is treated as 'low' because the
    autonomic nervous system is signalling stress even if WHOOP's composite
    is in the moderate range.

    v6.0 — Personal baseline integration:
        When a PersonalBaseline with is_personal=True is supplied, the tier
        thresholds are anchored to David's own percentile distribution rather
        than fixed WHOOP/population norms.  This makes the tiers meaningful
        relative to his own physiology:
          - "Peak" = top quintile of his days (not WHOOP's generic 80%)
          - "Low HRV" = below his personal 20th percentile (not population 45ms)

        Falls back to population norms when baseline is None or not personal
        (i.e., insufficient data — fewer than 14 days of history).
    """
    from analysis.personal_baseline import readiness_tier_personal
    return readiness_tier_personal(recovery, hrv, baseline)


def _tier_label(tier: str) -> str:
    """Human-readable label for the readiness tier."""
    return {
        "peak": "Peak",
        "good": "Good",
        "moderate": "Moderate",
        "low": "Low",
        "recovery": "Recovery Day",
        "unknown": "Unknown",
    }.get(tier, tier.title())


def _tier_recommendation(
    tier: str,
    recovery: Optional[float],
    hrv: Optional[float],
    yesterday_cls: Optional[float],
    yesterday_meeting_mins: Optional[int],
) -> str:
    """
    Generate a specific, actionable scheduling recommendation.

    Combines today's physiological state with yesterday's cognitive
    load to produce a concrete suggestion — not just "rest" or "go hard"
    but specifically what kind of work to front-load or protect.

    Parameters
    ----------
    tier : readiness tier string
    recovery : WHOOP recovery score (0–100)
    hrv : HRV RMSSD in milliseconds
    yesterday_cls : average cognitive load score from yesterday (0–1)
    yesterday_meeting_mins : total meeting time yesterday in minutes
    """
    # Context from yesterday
    heavy_yesterday = yesterday_cls is not None and yesterday_cls > 0.45
    meeting_heavy = yesterday_meeting_mins is not None and yesterday_meeting_mins >= 240

    if tier == "peak":
        if heavy_yesterday:
            return (
                "You're fully recovered from yesterday's demanding session. "
                "Good window for creative or strategic work that requires full cognitive bandwidth."
            )
        return (
            "High physiological readiness. Front-load complex, creative, or high-stakes work "
            "into this morning while capacity is at its peak."
        )

    elif tier == "good":
        return (
            "Solid readiness — capable of sustained demanding work. "
            "Normal scheduling is fine; protect at least one deep-work block of 90+ minutes."
        )

    elif tier == "moderate":
        if meeting_heavy:
            return (
                "Moderate readiness after a meeting-heavy day. "
                "Avoid stacking another heavy meeting block today — protect afternoon for recovery."
            )
        if heavy_yesterday:
            return (
                "Moderate readiness following a high-load day. "
                "Lighter cognitive work preferred; defer decisions requiring deep analysis to tomorrow."
            )
        return (
            "Moderate readiness. Manageable day, but avoid stacking meetings. "
            "One focused deep-work session is realistic; more than two demanding blocks will drain reserves."
        )

    elif tier == "low":
        hrv_note = (
            f" (HRV at {hrv:.0f}ms signals autonomic pressure)" if hrv else ""
        )
        return (
            f"Low readiness{hrv_note}. Keep today's schedule light: "
            "routine tasks, async communication, no major strategic decisions. "
            "Prioritise sleep hygiene tonight."
        )

    elif tier == "recovery":
        rec_note = f"Recovery at {recovery:.0f}%" if recovery else "Very low recovery"
        return (
            f"{rec_note} — this is a genuine recovery day. "
            "Cancel or reschedule anything cognitively demanding. "
            "Short walks, admin tasks, and protecting sleep tonight are the priority."
        )

    else:  # unknown
        return "WHOOP data unavailable — check your device charge and sync."


# ─── Calendar analysis (v7.0) ─────────────────────────────────────────────────

def analyse_today_calendar(calendar_data: dict) -> dict:
    """
    Summarise today's calendar into scheduling-relevant signals.

    Takes the raw output from collectors.gcal.collect() and produces a
    compact dict describing the shape of the day: total meeting load,
    first meeting time, largest meeting, free blocks in working hours,
    and a load classification.

    Parameters
    ----------
    calendar_data : dict
        Raw output from collectors.gcal.collect(date_str).
        Expected keys: 'events', 'total_meeting_minutes', 'event_count'.

    Returns
    -------
    dict with fields:
        event_count          : int   — number of meetings today
        total_minutes        : int   — total scheduled meeting time (minutes)
        load_class           : str   — 'free' | 'light' | 'moderate' | 'heavy'
        first_meeting_hour   : int | None   — hour of first meeting (0–23)
        first_meeting_label  : str | None   — e.g. "10:00"
        largest_meeting_mins : int          — duration of longest meeting
        largest_meeting_title: str          — title of longest meeting
        largest_attendees    : int          — attendee count of largest meeting
        social_meetings      : int          — meetings with > 1 attendee
        free_morning         : bool         — True if no meeting before 10am
        free_afternoon       : bool         — True if no meeting after 13:00
        events_summary       : list[dict]   — [{time, title, duration_min, attendees}]
        pre_first_free_mins  : int          — minutes of free time before first meeting

    Load classification:
        free      : 0 meetings
        light     : > 0 meetings, ≤ 90 total minutes
        moderate  : 91–240 total minutes
        heavy     : > 240 total minutes
    """
    events = calendar_data.get("events", []) or []
    total_minutes = calendar_data.get("total_meeting_minutes", 0) or 0

    # Filter to working-hours events (7am–10pm) and non-all-day
    working_events = []
    for e in events:
        start_str = e.get("start")
        if not start_str:
            continue
        if e.get("is_all_day"):
            continue
        try:
            start_dt = datetime.fromisoformat(start_str)
            hour = start_dt.hour
            if 7 <= hour < 22:
                working_events.append((start_dt, e))
        except (ValueError, TypeError):
            continue

    # Sort chronologically
    working_events.sort(key=lambda x: x[0])
    event_dicts = [e for _, e in working_events]

    # Recompute total minutes from working events only
    working_minutes = sum(e.get("duration_minutes", 0) for e in event_dicts)

    # Load class
    if not event_dicts:
        load_class = "free"
    elif working_minutes <= 90:
        load_class = "light"
    elif working_minutes <= 240:
        load_class = "moderate"
    else:
        load_class = "heavy"

    # First meeting
    first_meeting_hour: Optional[int] = None
    first_meeting_label: Optional[str] = None
    pre_first_free_mins: int = 0
    if working_events:
        first_dt, _ = working_events[0]
        first_meeting_hour = first_dt.hour
        first_meeting_label = first_dt.strftime("%-H:%M")
        # Minutes between 8am (brief fires at 7am; 8am is realistic start) and first meeting
        workday_start_h = 8
        if first_meeting_hour > workday_start_h:
            pre_first_free_mins = (first_meeting_hour - workday_start_h) * 60 + first_dt.minute
        elif first_meeting_hour == workday_start_h:
            pre_first_free_mins = first_dt.minute

    # Largest meeting
    largest_meeting_mins = 0
    largest_meeting_title = ""
    largest_attendees = 0
    if event_dicts:
        biggest = max(event_dicts, key=lambda e: e.get("duration_minutes", 0))
        largest_meeting_mins = biggest.get("duration_minutes", 0)
        largest_meeting_title = biggest.get("title", "")
        largest_attendees = biggest.get("attendee_count", 0)

    # Social meetings (those with other attendees)
    social_meetings = sum(1 for e in event_dicts if e.get("attendee_count", 0) > 1)

    # Free morning (no meeting before 10am)
    free_morning = not any(
        dt.hour < 10 for dt, _ in working_events
    )

    # Free afternoon (no meeting at 13:00 or later)
    free_afternoon = not any(
        dt.hour >= 13 for dt, _ in working_events
    )

    # Events summary for display
    events_summary = []
    for dt, e in working_events:
        events_summary.append({
            "time": dt.strftime("%-H:%M"),
            "title": e.get("title", ""),
            "duration_min": e.get("duration_minutes", 0),
            "attendees": e.get("attendee_count", 0),
        })

    return {
        "event_count": len(event_dicts),
        "total_minutes": working_minutes,
        "load_class": load_class,
        "first_meeting_hour": first_meeting_hour,
        "first_meeting_label": first_meeting_label,
        "largest_meeting_mins": largest_meeting_mins,
        "largest_meeting_title": largest_meeting_title,
        "largest_attendees": largest_attendees,
        "social_meetings": social_meetings,
        "free_morning": free_morning,
        "free_afternoon": free_afternoon,
        "events_summary": events_summary,
        "pre_first_free_mins": pre_first_free_mins,
    }


def _calendar_aware_recommendation(
    tier: str,
    recovery: Optional[float],
    hrv: Optional[float],
    yesterday_cls: Optional[float],
    yesterday_meeting_mins: Optional[int],
    cal: Optional[dict],
) -> str:
    """
    Generate a calendar-aware scheduling recommendation.

    When today's calendar data is available (cal is not None), the advice is
    specific to the actual meeting schedule — pointing at real free windows
    rather than generic guidance.  When cal is None, falls back to the
    original tier-based advice from _tier_recommendation().

    Parameters
    ----------
    tier : readiness tier ('peak', 'good', 'moderate', 'low', 'recovery')
    recovery : WHOOP recovery score (0–100), or None
    hrv : HRV RMSSD in ms, or None
    yesterday_cls : avg CLS yesterday (0–1), or None
    yesterday_meeting_mins : meeting minutes yesterday, or None
    cal : output from analyse_today_calendar(), or None

    Returns
    -------
    str — one to three sentence recommendation, specific to today.
    """
    # No calendar data → fall back to original logic
    if cal is None:
        return _tier_recommendation(tier, recovery, hrv, yesterday_cls, yesterday_meeting_mins)

    load_class = cal.get("load_class", "free")
    total_mins = cal.get("total_minutes", 0)
    first_label = cal.get("first_meeting_label")
    pre_free = cal.get("pre_first_free_mins", 0)
    free_morning = cal.get("free_morning", True)
    free_afternoon = cal.get("free_afternoon", True)
    event_count = cal.get("event_count", 0)
    largest_mins = cal.get("largest_meeting_mins", 0)
    social = cal.get("social_meetings", 0)

    # ── Recovery/Low tier: calendar doesn't change the core advice ────────
    if tier == "recovery":
        rec_note = f"Recovery at {recovery:.0f}%" if recovery else "Very low recovery"
        cal_note = ""
        if event_count > 0:
            cal_note = f" You have {event_count} meeting{'s' if event_count > 1 else ''} today ({total_mins}min) — consider rescheduling anything non-essential."
        return (
            f"{rec_note} — this is a genuine recovery day.{cal_note} "
            "Short walks, admin tasks, and protecting sleep tonight are the priority."
        )

    if tier == "low":
        hrv_note = (
            f" (HRV at {hrv:.0f}ms signals autonomic pressure)" if hrv else ""
        )
        cal_note = ""
        if load_class == "heavy":
            cal_note = f" With {total_mins//60}h{total_mins%60:02d}min of meetings ahead, pace yourself carefully — take breaks between blocks."
        elif load_class == "moderate":
            cal_note = f" {total_mins}min of meetings today — protect any free blocks for recovery."
        return (
            f"Low readiness{hrv_note}. Keep today light: routine tasks, async comms, no major decisions.{cal_note} "
            "Prioritise sleep hygiene tonight."
        )

    # ── Peak / Good / Moderate tier: give calendar-specific guidance ───────

    if load_class == "free":
        # No meetings — pure deep work day
        if tier in ("peak", "good"):
            return (
                "Calendar is clear today — ideal conditions for deep, uninterrupted work. "
                "Block a 2–3h focused session on your most complex current problem."
            )
        else:  # moderate
            return (
                "Calendar is clear today — use the freedom to pace yourself. "
                "One solid deep-work block in the morning, lighter work after lunch."
            )

    elif load_class == "light":
        # Short meetings, plenty of free time
        if first_label and pre_free >= 60:
            window_note = f"First meeting at {first_label} — use the {pre_free//60}h{pre_free%60:02d}min before it for focused work."
        else:
            window_note = f"{total_mins}min of meetings — plenty of free time for deep work."
        if tier == "peak":
            return f"High readiness + light meeting load. {window_note} Front-load your hardest problem."
        elif tier == "good":
            return f"Good readiness with a light schedule. {window_note}"
        else:
            return f"Manageable day. {window_note} One focused block is realistic."

    elif load_class == "moderate":
        # 90–240 min meetings — needs strategy
        if free_morning and first_label:
            return (
                f"Meetings start at {first_label} ({total_mins}min total). "
                f"Protect the {'morning' if free_morning else 'afternoon'} window for deep work — "
                f"{pre_free//60}h{pre_free%60:02d}min before first meeting is your focus runway."
                if pre_free >= 45 else
                f"Meetings from {first_label} ({total_mins}min total). "
                "Use gaps between meetings for shallow work; protect any 45min+ free blocks for focused tasks."
            )
        elif free_afternoon:
            return (
                f"Morning meetings ({total_mins}min). "
                "Afternoon is free — use it for the deep work that meetings will displace. "
                "Batch any quick decisions into the last meeting of the morning."
            )
        else:
            return (
                f"Moderate meeting load ({total_mins}min across {event_count} meetings). "
                "Identify the longest free gap and protect it for focused work. "
                "Set status to DND during that block."
            )

    else:  # heavy (> 240 min)
        hours = total_mins // 60
        mins_rem = total_mins % 60
        time_str = f"{hours}h{mins_rem:02d}min" if mins_rem else f"{hours}h"
        if tier == "peak":
            return (
                f"High readiness but heavy meeting load ({time_str}). "
                f"{'Use 8–' + first_label + ' for any focused work before the day gets consumed.' if first_label and pre_free >= 30 else 'Meetings dominate today — prioritise decision quality over volume.'} "
                "Energy manage: stay hydrated, take 5min breaks between calls."
            )
        elif tier == "good":
            return (
                f"Solid readiness heading into a {time_str} meeting day. "
                f"{'Pre-work any critical items before ' + first_label + '.' if first_label and pre_free >= 30 else 'Front-load any solo prep before the first meeting.'} "
                "Batch decisions into the meetings rather than deferring for async."
            )
        else:  # moderate readiness + heavy calendar = warning
            return (
                f"⚠️ Moderate readiness + heavy meeting load ({time_str}). "
                "This is a risk combination — protect breaks between meetings and "
                "defer any non-essential deep work to tomorrow when load is lighter."
            )


# ─── Score bar helper (shared style with daily digest) ───────────────────────

def _score_bar(value: float, width: int = 10) -> str:
    """Convert a 0–1 score to a visual progress bar."""
    filled = round(value * width)
    return "▓" * filled + "░" * (width - filled)


def _hrv_context(hrv: Optional[float], hrv_baseline: Optional[float]) -> str:
    """Format HRV with a relative context note if baseline is available."""
    if hrv is None:
        return "N/A"
    if hrv_baseline is None:
        return f"{hrv:.0f}ms"
    diff_pct = (hrv - hrv_baseline) / hrv_baseline * 100
    if abs(diff_pct) < 8:
        return f"{hrv:.0f}ms (baseline)"
    elif diff_pct > 0:
        return f"{hrv:.0f}ms (+{diff_pct:.0f}% vs baseline)"
    else:
        return f"{hrv:.0f}ms ({diff_pct:.0f}% vs baseline)"


# ─── Brief computation ────────────────────────────────────────────────────────

def compute_morning_brief(
    today_date: str,
    whoop_data: dict,
    yesterday_summary: Optional[dict] = None,
    hrv_baseline: Optional[float] = None,
    trend_context: Optional[dict] = None,
    personal_baseline=None,
    today_calendar: Optional[dict] = None,
) -> dict:
    """
    Compute the morning brief data structure.

    Parameters
    ----------
    today_date : "YYYY-MM-DD"
    whoop_data : raw output from collectors.whoop.collect(today_date)
    yesterday_summary : daily summary dict from store (optional)
    hrv_baseline : 7-day average HRV in ms (optional, for relative context)
    trend_context : multi-day trend dict from daily_digest.compute_trend_context()
        (optional, v1.5).  When provided, surfaces the most significant
        pattern (HRV streak, overcapacity streak, recovery decline) in the
        morning message so David sees it before planning his day — not at
        23:45 when the digest arrives.
    personal_baseline : PersonalBaseline instance from analysis.personal_baseline
        (optional, v6.0).  When supplied and is_personal=True, readiness tiers
        are anchored to David's own percentile distribution rather than
        population-norm WHOOP thresholds.  Falls back gracefully to population
        norms when None or when insufficient data exists.
    today_calendar : raw output from collectors.gcal.collect(today_date) (optional, v7.0).
        When supplied, used to generate calendar-aware scheduling recommendations
        and a "Today's Schedule" section in the brief.  When None, falls back
        to tier-only recommendations (backward compatible).

    Returns
    -------
    dict with all fields needed to format the morning brief message.
    """
    recovery = whoop_data.get("recovery_score")
    hrv = whoop_data.get("hrv_rmssd_milli")
    sleep_hours = whoop_data.get("sleep_hours")
    sleep_performance = whoop_data.get("sleep_performance")
    rhr = whoop_data.get("resting_heart_rate")

    # Yesterday's context
    yesterday_cls = None
    yesterday_meeting_mins = None
    yesterday_date = None
    if yesterday_summary:
        yesterday_date = yesterday_summary.get("date")
        yesterday_cls = yesterday_summary.get("metrics_avg", {}).get("cognitive_load_score")
        yesterday_meeting_mins = yesterday_summary.get("calendar", {}).get("total_meeting_minutes")

    tier = _readiness_tier(recovery, hrv, baseline=personal_baseline)

    # v7.0: calendar-aware recommendation
    cal_analysis: Optional[dict] = None
    if today_calendar is not None:
        try:
            cal_analysis = analyse_today_calendar(today_calendar)
        except Exception:
            cal_analysis = None

    recommendation = _calendar_aware_recommendation(
        tier, recovery, hrv, yesterday_cls, yesterday_meeting_mins, cal_analysis
    )

    return {
        "date": today_date,
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "sleep_hours": sleep_hours,
            "sleep_performance": sleep_performance,
            "resting_heart_rate": rhr,
        },
        "readiness": {
            "tier": tier,
            "label": _tier_label(tier),
            "recommendation": recommendation,
        },
        "yesterday": {
            "date": yesterday_date,
            "avg_cls": yesterday_cls,
            "meeting_minutes": yesterday_meeting_mins,
        },
        "hrv_baseline": hrv_baseline,
        "trend_context": trend_context or {},
        # v6.0: personal baseline metadata (for display/debugging)
        "personal_baseline": {
            "is_personal": getattr(personal_baseline, "is_personal", False),
            "days_of_data": getattr(personal_baseline, "days_of_data", 0),
            "recovery_p80": getattr(personal_baseline, "recovery_p80", None),
            "hrv_p20": getattr(personal_baseline, "hrv_p20", None),
        } if personal_baseline is not None else None,
        # v7.0: today's calendar analysis (for display)
        "today_calendar": cal_analysis,
        # v8.0: Cognitive Debt Index — multi-day accumulated fatigue
        # Computed here so the morning brief can warn David before the day starts.
        # Returns None when < 3 days of history exist (graceful degradation).
        "cognitive_debt": _compute_cdi_for_brief(today_date),
        # v9.0: Daily Presence Score trend — is cognitive day quality improving or declining?
        # CDI tracks load vs. recovery balance; DPS trend tracks *quality* of days.
        # A declining DPS trend (e.g. 78 → 65 → 52) means David's days are getting
        # worse even if CDI isn't yet signalling alarm — early warning before fatigue peaks.
        # Returns None when < 3 days of history exist (graceful degradation).
        "dps_trend": _compute_dps_trend_for_brief(today_date),
        # v10.0: Tomorrow's Focus Plan — specific time blocks for deep work tomorrow.
        # Uses historical hourly FDI pattern + tomorrow's calendar + CDI to recommend
        # when to protect focused work time.  Answers: "When should I block focus time?"
        # Returns None when tomorrow_calendar is unavailable (graceful degradation).
        "tomorrow_focus_plan": _compute_focus_plan_for_brief(today_date, today_calendar),
        # v14.0: Predicted Cognitive Load Forecast — estimate today's expected CLS
        # from today's calendar meeting load vs. historical CLS on similar-load days.
        # Answers: "Before the day starts, how demanding will it likely be?"
        # Complements WHOOP readiness (capacity) with a demand estimate (load forecast).
        # Returns None when < 2 days of history exist (graceful degradation).
        "load_forecast": _compute_load_forecast_for_brief(today_date, today_calendar),
        # v16.0: Daily Cognitive Budget — concrete quality hours available today.
        # Translates WHOOP recovery % + CDI tier + sleep into a single number:
        # "You have ~6.0h of quality cognitive work available today."
        # Bridges the gap between physiological capacity (WHOOP) and actionable
        # work planning.  Anchors CDI and load forecast in a single concrete figure.
        # _compute_cognitive_budget_for_brief fetches CDI from the store internally.
        # Returns None when WHOOP data is unavailable (graceful degradation).
        "cognitive_budget": _compute_cognitive_budget_for_brief(
            today_date,
            whoop_data=whoop_data,
        ),
        # v18.0: Cognitive Rhythm — when in the day does David do his best thinking?
        # Surfaces peak focus hours, morning/afternoon bias, and day-of-week pattern
        # in a compact one-liner each morning.  Answers: "When should I schedule hard
        # work today?"  Derived from accumulated JSONL history — becomes meaningful
        # after MIN_DAYS_FOR_RHYTHM (3) days.  Shown as a compact hint below the
        # schedule, not as a prominent section, so it doesn't overwhelm the brief.
        # Returns None when insufficient history (graceful degradation).
        "cognitive_rhythm": _compute_cognitive_rhythm_for_brief(today_date),
    }


def _compute_cdi_for_brief(date_str: str) -> Optional[dict]:
    """
    Compute CDI for the morning brief.  Returns None when not meaningful.
    Wraps the CDI module with full exception isolation.
    """
    try:
        from analysis.cognitive_debt import compute_cdi, format_cdi_line
        debt = compute_cdi(date_str)
        if not debt.is_meaningful:
            return None
        return {
            "cdi": debt.cdi,
            "tier": debt.tier,
            "trend_5d": debt.trend_5d,
            "days_in_deficit": debt.days_in_deficit,
            "days_used": debt.days_used,
            "line": format_cdi_line(debt),
        }
    except Exception:
        return None


def _compute_dps_trend_for_brief(date_str: str) -> Optional[dict]:
    """
    Compute DPS trend for the morning brief.  Returns None when insufficient data.

    The DPS trend answers: "Is the quality of David's cognitive days improving
    or declining over the past 7 days?"

    This is distinct from CDI:
    - CDI: load vs. recovery balance (fatigue accumulation signal)
    - DPS trend: composite day-quality trajectory (early warning before CDI peaks)

    A declining DPS trend with a healthy CDI still tells David that his days are
    becoming less focused/aligned even if he hasn't crossed fatigue thresholds yet.

    Wraps compute_dps_trend() with full exception isolation.

    Returns:
        dict with keys:
            mean_dps       — 7-day average DPS (0–100)
            delta_dps      — recent 3d avg minus prior 4d avg
            trend_direction— 'improving' | 'declining' | 'stable'
            days_used      — days with meaningful DPS data
            recent_scores  — last 3 DPS values for sparkline display
            line           — formatted one-line summary for Slack
        or None if < 3 meaningful days exist.
    """
    try:
        from analysis.presence_score import compute_dps_trend, get_historical_dps

        trend = compute_dps_trend(date_str, days=7)
        if trend is None:
            return None

        # Gather last 3 DPS scores for the sparkline
        history = get_historical_dps(date_str, days=7)
        meaningful = [h for h in history if h["is_meaningful"]]
        recent_scores = [round(h["dps"]) for h in meaningful[-3:]]

        line = _format_dps_trend_line(trend, recent_scores)
        if not line:
            return None

        return {
            "mean_dps": trend["mean_dps"],
            "delta_dps": trend["delta_dps"],
            "trend_direction": trend["trend_direction"],
            "days_used": trend["days_used"],
            "recent_scores": recent_scores,
            "line": line,
        }
    except Exception:
        return None


def _compute_focus_plan_for_brief(today_date_str: str, today_calendar: Optional[dict] = None) -> Optional[dict]:
    """
    Compute tomorrow's focus plan for the morning brief.

    Uses the Focus Planner to find the best available deep-work windows for
    tomorrow, factoring in David's historical peak-FDI hours, CDI tier, and
    tomorrow's calendar schedule.

    The today_calendar parameter is unused here (it's today's, not tomorrow's).
    The Focus Planner fetches tomorrow's calendar itself.

    Returns a dict with serialised FocusPlan data, or None on error / no data.

    v10.0: Added focus plan to morning brief.
    """
    try:
        from analysis.focus_planner import plan_tomorrow_focus, format_focus_plan_section
        plan = plan_tomorrow_focus(today_date_str)
        if plan is None:
            return None
        return {
            "section": format_focus_plan_section(plan),
            "summary_line": plan.summary_line,
            "advisory": plan.advisory,
            "is_meaningful": plan.is_meaningful,
            "block_count": len(plan.recommended_blocks),
            "cdi_tier": plan.cdi_tier,
        }
    except Exception:
        return None


def _compute_load_forecast_for_brief(
    date_str: str,
    today_calendar: Optional[dict] = None,
) -> Optional[dict]:
    """
    Compute the predicted CLS forecast for the morning brief.

    Uses today's meeting minutes (from today_calendar) + historical CLS
    patterns from similar-load days to estimate the day's expected CLS.

    Returns a dict with forecast data, or None when not meaningful.

    v14.0: Added load forecast to morning brief.
    """
    try:
        from analysis.load_forecast import compute_load_forecast, format_forecast_line
        forecast = compute_load_forecast(date_str, today_calendar)
        if not forecast.is_meaningful:
            return None
        return {
            "line": format_forecast_line(forecast),
            "predicted_cls": forecast.predicted_cls,
            "cls_low": forecast.cls_low,
            "cls_high": forecast.cls_high,
            "load_label": forecast.load_label,
            "confidence": forecast.confidence,
            "matching_days": forecast.matching_days,
            "narrative": forecast.narrative,
            "is_meaningful": True,
        }
    except Exception:
        return None


def _compute_cognitive_budget_for_brief(
    date_str: str,
    whoop_data: Optional[dict] = None,
) -> Optional[dict]:
    """
    Compute the Daily Cognitive Budget for the morning brief.

    Translates WHOOP recovery + CDI + sleep into a concrete estimate of
    quality cognitive hours available today.  Fetches CDI tier and personal
    HRV baseline from the store internally.

    Returns a dict with budget data, or None when not meaningful (no WHOOP data).

    v16.0: Added cognitive budget to morning brief.
    """
    try:
        from analysis.cognitive_budget import (
            compute_cognitive_budget,
            format_budget_line,
            format_budget_section,
        )
        from analysis.personal_baseline import get_personal_baseline

        hrv_baseline: Optional[float] = None
        try:
            baseline = get_personal_baseline()
            if baseline.hrv_mean is not None:
                hrv_baseline = baseline.hrv_mean
        except Exception:
            pass

        # Fetch CDI tier from store for accurate modifier
        cdi_tier: Optional[str] = None
        try:
            from analysis.cognitive_debt import compute_cdi
            debt = compute_cdi(date_str)
            if debt.is_meaningful:
                cdi_tier = debt.tier
        except Exception:
            pass

        budget = compute_cognitive_budget(
            date_str=date_str,
            whoop_data=whoop_data,
            cdi_tier=cdi_tier,
            hrv_baseline=hrv_baseline,
        )
        if not budget.is_meaningful:
            return None
        return {
            "dcb_hours": budget.dcb_hours,
            "dcb_low": budget.dcb_low,
            "dcb_high": budget.dcb_high,
            "tier": budget.tier,
            "label": budget.label,
            "narrative": budget.narrative,
            "guidance": budget.guidance,
            "line": format_budget_line(budget),
            "section": format_budget_section(budget),
            "is_meaningful": True,
        }
    except Exception:
        return None


def _compute_cognitive_rhythm_for_brief(date_str: str) -> Optional[dict]:
    """
    Compute the cognitive rhythm summary for the morning brief.

    Returns a compact dict with the one-liner and key rhythm signals,
    or None when not meaningful (insufficient data or error).

    v18.0: Added cognitive rhythm to morning brief.
    """
    try:
        from analysis.cognitive_rhythm import (
            compute_cognitive_rhythm,
            format_rhythm_line,
            DOW_LABELS,
        )

        rhythm = compute_cognitive_rhythm(date_str)
        if not rhythm.is_meaningful:
            return None

        line = format_rhythm_line(rhythm)
        if not line:
            return None

        return {
            "line": line,
            "peak_focus_hours": rhythm.peak_focus_hours,
            "morning_bias": rhythm.morning_bias,
            "best_focus_dow": DOW_LABELS[rhythm.best_focus_dow] if rhythm.best_focus_dow is not None else None,
            "hourly_fdi_sparkline": rhythm.hourly_fdi_sparkline,
            "days_analyzed": rhythm.days_analyzed,
            "is_meaningful": True,
        }
    except Exception:
        return None


def _format_dps_trend_line(trend: dict, recent_scores: list) -> str:
    """
    Format a compact DPS trend line for the morning brief.

    Examples:
        "📈 Day quality improving: 58 → 65 → 71 (+13 pts)"
        "📉 Day quality declining: 78 → 65 → 52 (−26 pts)"
        "➡️ Day quality stable: 68 → 72 → 70 (7-day avg 70)"

    Only shows when trend is meaningful (direction != stable OR delta is notable).
    Returns empty string for stable + no notable delta (caller skips it).
    """
    direction = trend.get("trend_direction", "stable")
    delta = trend.get("delta_dps", 0.0)
    mean_dps = trend.get("mean_dps", 0.0)
    days_used = trend.get("days_used", 0)

    # Build sparkline from recent scores
    sparkline = " → ".join(str(s) for s in recent_scores) if recent_scores else ""

    if direction == "improving":
        delta_str = f"+{delta:.0f} pts" if delta >= 0 else f"{delta:.0f} pts"
        return f"📈 Day quality improving: {sparkline} ({delta_str})"
    elif direction == "declining":
        delta_str = f"−{abs(delta):.0f} pts"
        return f"📉 Day quality declining: {sparkline} ({delta_str})"
    else:
        # Stable — only show when we have enough context (≥ 5 days)
        if days_used >= 5:
            return f"➡️ Day quality stable ({days_used}d avg: {mean_dps:.0f}/100)"
        return ""


# ─── Message formatter ────────────────────────────────────────────────────────

def format_morning_brief_message(brief: dict) -> str:
    """
    Format the morning brief into a Slack DM.

    Designed to be scannable in under 30 seconds at 7am.
    Key info is front-loaded; recommendation is prominent.
    """
    if not brief:
        return "Presence Tracker: no morning data available."

    date_str = brief.get("date", "today")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_label = dt.strftime("%A, %B %-d")
    except ValueError:
        date_label = date_str

    w = brief.get("whoop", {})
    r = brief.get("readiness", {})
    yesterday = brief.get("yesterday", {})
    hrv_baseline = brief.get("hrv_baseline")

    recovery = w.get("recovery_score")
    hrv = w.get("hrv_rmssd_milli")
    sleep_h = w.get("sleep_hours")
    sleep_perf = w.get("sleep_performance")
    rhr = w.get("resting_heart_rate")

    tier = r.get("tier", "unknown")
    tier_label = r.get("label", "Unknown")
    recommendation = r.get("recommendation", "")

    # Tier emoji
    tier_emoji = {
        "peak": "🟢",
        "good": "🔵",
        "moderate": "🟡",
        "low": "🟠",
        "recovery": "🔴",
        "unknown": "⚪",
    }.get(tier, "⚪")

    lines = [
        f"*Morning Readiness — {date_label}*",
        "",
    ]

    # ── Readiness headline ──
    if recovery is not None:
        rec_bar = _score_bar(recovery / 100)
        lines.append(f"{tier_emoji} *{tier_label}*  {rec_bar} Recovery {recovery:.0f}%")
    else:
        lines.append(f"{tier_emoji} *{tier_label}*  (no WHOOP data)")

    # ── WHOOP signals ──
    detail_parts = []
    if hrv is not None:
        detail_parts.append(_hrv_context(hrv, hrv_baseline))
    if rhr is not None:
        detail_parts.append(f"RHR {rhr:.0f}bpm")
    if sleep_h is not None:
        sleep_str = f"Sleep {sleep_h:.1f}h"
        if sleep_perf is not None:
            sleep_str += f" ({sleep_perf:.0f}%)"
        detail_parts.append(sleep_str)
    if detail_parts:
        lines.append("_" + "  ·  ".join(detail_parts) + "_")

    lines.append("")

    # ── Recommendation ──
    lines.append(f"*Today:* {recommendation}")

    # ── Yesterday context ──
    if yesterday.get("avg_cls") is not None:
        y_cls = yesterday["avg_cls"]
        y_mins = yesterday.get("meeting_minutes", 0) or 0
        y_date = yesterday.get("date", "yesterday")
        try:
            y_dt = datetime.strptime(y_date, "%Y-%m-%d")
            y_label = y_dt.strftime("%A")
        except ValueError:
            y_label = "Yesterday"

        cls_parts = [f"CLS {y_cls:.0%}"]
        if y_mins:
            cls_parts.append(f"{y_mins}min meetings")
        lines.append("")
        lines.append(f"_Yesterday ({y_label}): {' · '.join(cls_parts)}_")

    # ── Trend context (v1.5) ──
    # Surface the most significant multi-day pattern so David sees it
    # at 7am when it can influence scheduling, not at 23:45.
    trend = brief.get("trend_context", {})
    if trend and trend.get("days_of_data", 0) >= 2:
        trend_lines = []

        # HRV streak
        hrv_trend = trend.get("hrv_trend")
        hrv_streak = trend.get("hrv_streak_days", 0)
        if hrv_trend == "declining" and hrv_streak >= 2:
            trend_lines.append(f"⚠️ HRV declining {hrv_streak} days in a row")
        elif hrv_trend == "improving" and hrv_streak >= 2:
            trend_lines.append(f"✅ HRV improving {hrv_streak} days in a row")

        # Overcapacity streak
        oc_streak = trend.get("overcapacity_streak", 0)
        if oc_streak >= 2:
            trend_lines.append(f"⚠️ Above capacity {oc_streak} consecutive days")

        # Recovery decline
        rec_trend = trend.get("recovery_trend")
        rec_streak = trend.get("recovery_streak_days", 0)
        if rec_trend == "declining" and rec_streak >= 2:
            trend_lines.append(f"⚠️ Recovery declining {rec_streak} days in a row")

        # CLS vs baseline (cls_vs_baseline is a % value, e.g. 40.0 = 40% above)
        cls_vs = trend.get("cls_vs_baseline")
        if cls_vs is not None and abs(cls_vs) >= 25:
            direction = "above" if cls_vs > 0 else "below"
            trend_lines.append(f"Load {abs(cls_vs):.0f}% {direction} your 7-day baseline")

        if trend_lines:
            lines.append("")
            lines.append("*Pattern:*")
            for tl in trend_lines:
                lines.append(f"  {tl}")

    # ── Cognitive Debt Index (v8.0) ──────────────────────────────────────
    # Multi-day accumulated fatigue — shown when meaningful (≥ 3 days data).
    # Morning is the right time to see CDI: it can influence whether to protect
    # the day for recovery or push forward on high-demand work.
    cognitive_debt = brief.get("cognitive_debt")
    if cognitive_debt:
        cdi_line = cognitive_debt.get("line", "")
        if cdi_line:
            lines.append("")
            lines.append(f"_{cdi_line}_")

    # ── Daily Cognitive Budget (v16.0) ────────────────────────────────────
    # Concrete quality cognitive hours available today.
    # Translates WHOOP recovery + sleep + CDI into a single actionable figure:
    # "~6.0h of quality focus work available."  Positioned after CDI so the
    # two signals read together: "Here's your debt level — here's your budget."
    cognitive_budget = brief.get("cognitive_budget")
    if cognitive_budget and cognitive_budget.get("is_meaningful"):
        budget_line = cognitive_budget.get("line", "")
        if budget_line:
            lines.append("")
            lines.append(budget_line)
        budget_guidance = cognitive_budget.get("guidance", "")
        if budget_guidance:
            lines.append(f"_{budget_guidance}_")

    # ── DPS Trend (v9.0) ──────────────────────────────────────────────────
    # Is the quality of David's cognitive days improving or declining?
    # Shown after CDI so the two multi-day signals appear together.
    # CDI = fatigue accumulation (load vs. recovery balance).
    # DPS trend = cognitive day quality trajectory (focus, alignment, sustainability).
    # Together they answer: "Am I carrying debt AND getting worse days?"
    dps_trend = brief.get("dps_trend")
    if dps_trend:
        dps_trend_line = dps_trend.get("line", "")
        if dps_trend_line:
            lines.append("")
            lines.append(f"_{dps_trend_line}_")

    # ── Load Forecast (v14.0) ─────────────────────────────────────────────
    # Predicted CLS for today based on meeting load vs. historical patterns.
    # "Before the day starts, how cognitively demanding will it likely be?"
    # This bridges WHOOP capacity (what you have) with calendar demand (what's coming).
    # Shown between the multi-day trend signals and tomorrow's focus plan so it
    # flows naturally: "Here's your capacity — here's today's likely demand —
    # here's tomorrow's plan."
    load_forecast = brief.get("load_forecast")
    if load_forecast and load_forecast.get("is_meaningful"):
        forecast_line = load_forecast.get("line", "")
        if forecast_line:
            lines.append("")
            lines.append(forecast_line)

    # ── Tomorrow's Focus Plan (v10.0) ─────────────────────────────────────
    # Specific time blocks for deep work tomorrow, combining:
    # - David's historical peak-FDI hours (when does he actually focus best?)
    # - Tomorrow's calendar free windows
    # - CDI tier (how much cognitive debt is he carrying?)
    # This is the "what to do" complement to all the "what happened" signals above.
    tomorrow_plan = brief.get("tomorrow_focus_plan")
    if tomorrow_plan and tomorrow_plan.get("is_meaningful") and tomorrow_plan.get("section"):
        lines.append("")
        lines.append(tomorrow_plan["section"])

    # ── Today's Schedule (v7.0) ───────────────────────────────────────────
    # Show what's on the calendar today so David can see the shape of his day.
    cal = brief.get("today_calendar")
    if cal:
        events_summary = cal.get("events_summary", [])
        total_mins = cal.get("total_minutes", 0)
        load_class = cal.get("load_class", "free")

        if load_class == "free":
            lines.append("")
            lines.append("*📅 Today:* No meetings scheduled — full focus day")
        else:
            load_emoji = {
                "light": "🟢",
                "moderate": "🟡",
                "heavy": "🔴",
            }.get(load_class, "⚪")
            hours = total_mins // 60
            mins_rem = total_mins % 60
            if hours > 0:
                time_str = f"{hours}h{mins_rem:02d}min" if mins_rem else f"{hours}h"
            else:
                time_str = f"{mins_rem}min"

            lines.append("")
            lines.append(f"*📅 Today ({load_emoji} {time_str} meetings):*")
            for ev in events_summary[:5]:  # cap at 5 to keep the message scannable
                t = ev.get("time", "")
                title = ev.get("title", "") or "Meeting"
                dur = ev.get("duration_min", 0)
                att = ev.get("attendees", 0)
                att_str = f"  {att}p" if att > 1 else ""
                dur_str = f"{dur}min" if dur else ""
                # Truncate long titles
                if len(title) > 32:
                    title = title[:30] + "…"
                lines.append(f"  {t}  {title}  _{dur_str}{att_str}_")
            if len(events_summary) > 5:
                remaining = len(events_summary) - 5
                lines.append(f"  _(+{remaining} more)_")

    # ── Cognitive Rhythm hint (v18.0) ─────────────────────────────────────
    # One compact line at the bottom of the brief: "Peak focus 9am–11am ·
    # morning-biased · Best day: Wednesday" — derived from historical data.
    # Placed last so it doesn't compete with the actionable signals above,
    # but is available for the motivated reader who wants to plan optimally.
    cognitive_rhythm = brief.get("cognitive_rhythm")
    if cognitive_rhythm and cognitive_rhythm.get("is_meaningful"):
        rhythm_line = cognitive_rhythm.get("line", "")
        if rhythm_line:
            lines.append("")
            lines.append(f"_{rhythm_line}_")

    return "\n".join(lines)


# ─── Gateway helpers ──────────────────────────────────────────────────────────

def _gateway_invoke(tool: str, args: dict, timeout: int = 30) -> dict:
    """Call an OpenClaw tool via the gateway."""
    headers = {
        "Authorization": f"Bearer {GATEWAY_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({"tool": tool, "args": args}).encode()
    req = urllib.request.Request(
        f"{GATEWAY_URL}/tools/invoke",
        data=payload,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _send_slack_dm(message: str) -> bool:
    """Send the morning brief to David's Slack DM."""
    try:
        result = _gateway_invoke("message", {
            "action": "send",
            "channel": "slack",
            "target": SLACK_DM_CHANNEL,
            "message": message,
        })
        return result.get("ok", False)
    except Exception as e:
        print(f"[morning] Failed to send DM: {e}", file=sys.stderr)
        return False


# ─── Main entry point ─────────────────────────────────────────────────────────

def send_morning_brief(date_str: Optional[str] = None) -> bool:
    """
    Collect today's WHOOP data and send the morning readiness brief.

    Args:
        date_str: Date to run for (YYYY-MM-DD). Defaults to today.

    Returns:
        True if the DM was sent successfully.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"[morning] Running morning brief for {date_str}")

    # ── Collect WHOOP data ────────────────────────────────────────────────
    try:
        from collectors.whoop import collect as whoop_collect
        whoop_data = whoop_collect(date_str)
        print(
            f"[morning] WHOOP: recovery={whoop_data.get('recovery_score')}% "
            f"HRV={whoop_data.get('hrv_rmssd_milli')}ms"
        )
    except Exception as e:
        print(f"[morning] WHOOP collection failed: {e}", file=sys.stderr)
        whoop_data = {}

    # ── Load yesterday's summary ──────────────────────────────────────────
    yesterday_summary = None
    yesterday_date = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    try:
        from engine.store import read_summary
        rolling = read_summary()
        yesterday_summary = rolling.get("days", {}).get(yesterday_date)
        if yesterday_summary:
            print(
                f"[morning] Yesterday ({yesterday_date}): "
                f"CLS={yesterday_summary.get('metrics_avg', {}).get('cognitive_load_score')}"
            )
        else:
            print(f"[morning] No summary for yesterday ({yesterday_date})")
    except Exception as e:
        print(f"[morning] Could not load yesterday's summary: {e}", file=sys.stderr)

    # ── Compute 7-day HRV baseline for context ────────────────────────────
    hrv_baseline = None
    try:
        from engine.store import get_recent_summaries
        recent = get_recent_summaries(days=7)
        hrv_vals = [
            s.get("whoop", {}).get("hrv_rmssd_milli")
            for s in recent
            if s.get("whoop", {}).get("hrv_rmssd_milli") is not None
        ]
        if len(hrv_vals) >= 3:
            hrv_baseline = sum(hrv_vals) / len(hrv_vals)
            print(f"[morning] 7-day HRV baseline: {hrv_baseline:.1f}ms")
    except Exception as e:
        print(f"[morning] Could not compute HRV baseline: {e}", file=sys.stderr)

    # ── Compute trend context (v1.5) ─────────────────────────────────────
    trend_context = {}
    try:
        from analysis.daily_digest import compute_trend_context
        trend_context = compute_trend_context(date_str, lookback_days=7)
        days = trend_context.get("days_of_data", 0)
        print(f"[morning] Trend context: {days} days of history")
    except Exception as e:
        print(f"[morning] Could not compute trend context: {e}", file=sys.stderr)

    # ── Compute personal baseline (v6.0) ─────────────────────────────────
    # Derives David's personal HRV/recovery percentile thresholds from
    # accumulated data.  Falls back to population norms when < 14 days
    # of history exist.  The baseline is passed into compute_morning_brief
    # which uses it in _readiness_tier via readiness_tier_personal().
    personal_baseline = None
    try:
        from analysis.personal_baseline import get_personal_baseline
        personal_baseline = get_personal_baseline(days=90)
        if personal_baseline.is_personal:
            print(
                f"[morning] Personal baseline: {personal_baseline.days_of_data} days, "
                f"recovery_p80={personal_baseline.recovery_p80:.0f}%, "
                f"hrv_p20={personal_baseline.hrv_p20:.0f}ms"
            )
        else:
            print(
                f"[morning] Personal baseline: {personal_baseline.days_of_data} days "
                f"(population norms — need {14} more days)"
            )
    except Exception as e:
        print(f"[morning] Could not compute personal baseline: {e}", file=sys.stderr)

    # ── Collect today's calendar (v7.0) ──────────────────────────────────
    # Fetches today's meetings so the recommendation can reference actual
    # schedule events rather than giving generic advice.
    today_calendar = None
    try:
        from collectors.gcal import collect as gcal_collect
        today_calendar = gcal_collect(date_str)
        event_count = today_calendar.get("event_count", 0)
        total_mins = today_calendar.get("total_meeting_minutes", 0)
        print(f"[morning] Today's calendar: {event_count} events, {total_mins}min scheduled")
    except Exception as e:
        print(f"[morning] Could not collect today's calendar: {e}", file=sys.stderr)

    # ── Compute and send ──────────────────────────────────────────────────
    brief = compute_morning_brief(
        today_date=date_str,
        whoop_data=whoop_data,
        yesterday_summary=yesterday_summary,
        hrv_baseline=hrv_baseline,
        trend_context=trend_context,
        personal_baseline=personal_baseline,
        today_calendar=today_calendar,
    )
    message = format_morning_brief_message(brief)

    print(f"[morning] Sending brief to David...")
    ok = _send_slack_dm(message)
    print(f"[morning] {'Sent' if ok else 'Failed'}")
    return ok


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send morning readiness brief to David")
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD, default: today)")
    parser.add_argument("--dry-run", action="store_true", help="Print without sending")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    # Collect WHOOP
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from collectors.whoop import collect as whoop_collect
        whoop_data = whoop_collect(date_str)
    except Exception as e:
        print(f"WHOOP failed: {e}", file=sys.stderr)
        whoop_data = {}

    # Yesterday summary
    yesterday_date = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")
    yesterday_summary = None
    try:
        from engine.store import read_summary
        yesterday_summary = read_summary().get("days", {}).get(yesterday_date)
    except Exception:
        pass

    # HRV baseline
    hrv_baseline = None
    try:
        from engine.store import get_recent_summaries
        recent = get_recent_summaries(days=7)
        vals = [s["whoop"]["hrv_rmssd_milli"] for s in recent
                if s.get("whoop", {}).get("hrv_rmssd_milli")]
        if len(vals) >= 3:
            hrv_baseline = sum(vals) / len(vals)
    except Exception:
        pass

    # Trend context (v1.5)
    trend_context = {}
    try:
        from analysis.daily_digest import compute_trend_context
        trend_context = compute_trend_context(date_str, lookback_days=7)
    except Exception:
        pass

    # Today's calendar (v7.0)
    today_calendar = None
    try:
        from collectors.gcal import collect as gcal_collect
        today_calendar = gcal_collect(date_str)
    except Exception:
        pass

    brief = compute_morning_brief(
        date_str, whoop_data, yesterday_summary, hrv_baseline, trend_context,
        today_calendar=today_calendar,
    )
    message = format_morning_brief_message(brief)

    print("=" * 60)
    print(message)
    print("=" * 60)

    if not args.dry_run:
        ok = _send_slack_dm(message)
        print(f"\n{'✓ Sent' if ok else '✗ Failed'} to David's DM")
    else:
        print("\n[dry-run] Not sent.")

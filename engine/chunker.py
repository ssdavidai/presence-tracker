"""
Presence Tracker — 15-Minute Window Chunker

Assembles all collected signals into 96 × 15-minute observation windows
for a given day, then computes derived metrics for each window.

Output: list of window dicts, written to data/chunks/YYYY-MM-DD.jsonl

v1.3 — is_active_window + accurate FDI filtering:
  Added `is_active_window` to window metadata.  A window is active when
  at least one behavioral signal is present: in_meeting, Slack messages,
  or RescueTime computer activity.

  This flag is critical for meaningful FDI analytics.  Without it, idle
  windows (no meetings, no Slack, no computer use) all return FDI=1.0
  because there is literally no disruption — but that 1.0 does NOT mean
  deep focused work; it means nothing was measured.  Averaging FDI over
  all working-hour windows (most of which can be idle, especially in
  non-meeting blocks) produces a falsely inflated daily FDI score.

  summarize_day() now computes two FDI statistics:
  - metrics_avg.focus_depth_index: all working-hour windows (unchanged,
    for backward compatibility)
  - focus_quality.active_fdi: FDI averaged over active windows only
    (the accurate signal for "how deep was focus when actually working?")
  - focus_quality.active_windows: count of active working-hour windows
  - focus_quality.peak_focus_hour: hour of day with best active FDI
    (useful for "your best focus window" reporting in Intuition)
"""

import sys
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from engine.metrics import compute_metrics

TIMEZONE = ZoneInfo("Europe/Budapest")
WINDOW_MINUTES = 15
WINDOWS_PER_DAY = 96  # 24h × 4 windows/hour
WORKING_HOURS_START = 7   # 7am
WORKING_HOURS_END = 22    # 10pm


def build_windows(
    date_str: str,
    whoop_data: dict,
    calendar_data: dict,
    slack_windows: dict,
    rescuetime_windows: dict | None = None,
) -> list[dict]:
    """
    Build 96 observation windows for a given date.

    Args:
        date_str: "YYYY-MM-DD"
        whoop_data: output from collectors.whoop.collect()
        calendar_data: output from collectors.gcal.collect()
        slack_windows: output from collectors.slack.collect() (keyed by window_index)
        rescuetime_windows: output from collectors.rescuetime.collect() (keyed by
            window_index), optional. If None or empty, rescuetime signals are omitted.

    Returns:
        List of 96 window dicts, each conforming to the schema in SPEC.md
    """
    from collectors.gcal import get_events_in_window

    date = datetime.strptime(date_str, "%Y-%m-%d")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_of_week = day_names[date.weekday()]

    windows = []

    for i in range(WINDOWS_PER_DAY):
        hour = i // 4
        minute = (i % 4) * WINDOW_MINUTES

        # Window start/end as timezone-aware datetimes
        window_start = datetime(date.year, date.month, date.day, hour, minute, 0, tzinfo=TIMEZONE)
        window_end = window_start + timedelta(minutes=WINDOW_MINUTES)

        # ── Calendar signals ──────────────────────────────────────────────
        events_in_window = get_events_in_window(
            calendar_data.get("events", []),
            window_start,
            window_end,
        )

        if events_in_window:
            # Use the longest/largest event if multiple overlap
            primary = max(events_in_window, key=lambda e: e.get("duration_minutes", 0))
            cal_signals = {
                "in_meeting": True,
                "meeting_title": primary.get("title", ""),
                "meeting_attendees": primary.get("attendee_count", 1),
                "meeting_duration_minutes": primary.get("duration_minutes", 0),
                "meeting_organizer": primary.get("organizer_email", ""),
                "meetings_count": len(events_in_window),
            }
        else:
            cal_signals = {
                "in_meeting": False,
                "meeting_title": None,
                "meeting_attendees": 0,
                "meeting_duration_minutes": 0,
                "meeting_organizer": None,
                "meetings_count": 0,
            }

        # ── WHOOP signals (daily, not per-window) ─────────────────────────
        whoop_signals = {
            "recovery_score": whoop_data.get("recovery_score"),
            "hrv_rmssd_milli": whoop_data.get("hrv_rmssd_milli"),
            "resting_heart_rate": whoop_data.get("resting_heart_rate"),
            "sleep_performance": whoop_data.get("sleep_performance"),
            "sleep_hours": whoop_data.get("sleep_hours"),
            "strain": whoop_data.get("strain"),
            "spo2_percentage": whoop_data.get("spo2_percentage"),
        }

        # ── Slack signals ─────────────────────────────────────────────────
        window_slack = slack_windows.get(i, {})
        slack_signals = {
            "messages_sent": window_slack.get("messages_sent", 0),
            "messages_received": window_slack.get("messages_received", 0),
            "total_messages": window_slack.get("total_messages", 0),
            "channels_active": window_slack.get("channels_active", 0),
        }

        # ── RescueTime signals ────────────────────────────────────────────
        rt_windows = rescuetime_windows or {}
        window_rt = rt_windows.get(i, {})
        rescuetime_signals = {
            "focus_seconds": window_rt.get("focus_seconds", 0),
            "distraction_seconds": window_rt.get("distraction_seconds", 0),
            "neutral_seconds": window_rt.get("neutral_seconds", 0),
            "active_seconds": window_rt.get("active_seconds", 0),
            "app_switches": window_rt.get("app_switches", 0),
            "productivity_score": window_rt.get("productivity_score"),
            "top_activity": window_rt.get("top_activity"),
        } if rt_windows else None

        # ── Metric computation ────────────────────────────────────────────
        metrics = compute_metrics({
            "calendar": cal_signals,
            "whoop": whoop_signals,
            "slack": slack_signals,
            **({"rescuetime": rescuetime_signals} if rescuetime_signals else {}),
        })

        # ── Active window detection ───────────────────────────────────────
        # A window is "active" when at least one behavioral signal is present.
        # This distinguishes genuine idle periods (sleep, away-from-keyboard)
        # from deep-focus working periods that happen to have no interruptions.
        #
        # Why this matters:
        # FDI=1.0 for idle windows is mathematically correct (zero disruption)
        # but analytically misleading — it inflates daily average FDI.  The
        # `is_active_window` flag lets downstream code filter to only windows
        # where David was actually engaged, making FDI stats meaningful.
        rt_active_secs = (rescuetime_signals or {}).get("active_seconds", 0)
        is_active = (
            cal_signals["in_meeting"]
            or slack_signals["total_messages"] > 0
            or rt_active_secs > 0
        )

        # ── Metadata ──────────────────────────────────────────────────────
        sources = []
        if whoop_data.get("recovery_score") is not None:
            sources.append("whoop")
        if calendar_data.get("events") is not None:
            sources.append("calendar")
        if window_slack.get("total_messages", 0) > 0 or True:  # Always available
            sources.append("slack")
        if rescuetime_signals and rescuetime_signals.get("active_seconds", 0) > 0:
            sources.append("rescuetime")

        window = {
            "window_id": window_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "date": date_str,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "window_index": i,
            "calendar": cal_signals,
            "whoop": whoop_signals,
            "slack": slack_signals,
            **({"rescuetime": rescuetime_signals} if rescuetime_signals else {}),
            "metrics": metrics,
            "metadata": {
                "day_of_week": day_of_week,
                "hour_of_day": hour,
                "minute_of_hour": minute,
                "is_working_hours": WORKING_HOURS_START <= hour < WORKING_HOURS_END,
                "is_active_window": is_active,
                "sources_available": sources,
            },
        }
        windows.append(window)

    return windows


def summarize_day(windows: list[dict]) -> dict:
    """
    Compute daily summary statistics from the 96 windows.

    v1.3: adds focus_quality section with FDI computed over active windows only.
    Active windows are those where at least one behavioral signal is present
    (in_meeting, Slack messages, or RescueTime computer activity).

    This is the accurate FDI signal — averaging FDI over all working-hour
    windows including idle ones inflates the score because idle windows
    return FDI=1.0 (no disruption ≠ deep focus).
    """
    if not windows:
        return {}

    # Working hours only (filter for analysis relevance)
    working = [w for w in windows if w["metadata"]["is_working_hours"]]

    # Active working-hour windows: behavioral signal was present
    active_working = [
        w for w in working
        if w["metadata"].get("is_active_window", False)
    ]

    def avg(vals: list) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def maximum(vals: list) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    cls_vals = [w["metrics"]["cognitive_load_score"] for w in working]
    fdi_vals = [w["metrics"]["focus_depth_index"] for w in working]
    sdi_vals = [w["metrics"]["social_drain_index"] for w in working]
    csc_vals = [w["metrics"]["context_switch_cost"] for w in working]
    ras_vals = [w["metrics"]["recovery_alignment_score"] for w in windows]

    # Active-window FDI stats — the meaningful signal for "how focused was David?"
    active_fdi_vals = [w["metrics"]["focus_depth_index"] for w in active_working]

    # Best focus hour: hour of day with highest mean FDI over active working windows
    # (grouped by hour, minimum 2 active windows in that hour to be reliable)
    from collections import defaultdict
    hour_fdi: dict[int, list[float]] = defaultdict(list)
    for w in active_working:
        hour_fdi[w["metadata"]["hour_of_day"]].append(w["metrics"]["focus_depth_index"])

    peak_focus_hour: Optional[int] = None
    peak_focus_fdi: Optional[float] = None
    for h, fdi_list in hour_fdi.items():
        if len(fdi_list) >= 2:
            h_avg = sum(fdi_list) / len(fdi_list)
            if peak_focus_fdi is None or h_avg > peak_focus_fdi:
                peak_focus_fdi = round(h_avg, 4)
                peak_focus_hour = h

    return {
        "date": windows[0]["date"] if windows else None,
        "working_hours_analyzed": len(working),
        "total_windows": len(windows),
        "metrics_avg": {
            "cognitive_load_score": avg(cls_vals),
            "focus_depth_index": avg(fdi_vals),
            "social_drain_index": avg(sdi_vals),
            "context_switch_cost": avg(csc_vals),
            "recovery_alignment_score": avg(ras_vals),
        },
        "metrics_peak": {
            "cognitive_load_score": maximum(cls_vals),
            "focus_depth_index": maximum(fdi_vals),
        },
        # v1.3: accurate FDI over active (behaviorally engaged) windows only.
        # active_fdi is the signal to use for "how focused was David when working?"
        # The metrics_avg.focus_depth_index includes idle windows (returns 1.0)
        # and is kept for backward compatibility only.
        "focus_quality": {
            "active_fdi": avg(active_fdi_vals),          # True focus depth (active windows only)
            "active_windows": len(active_working),        # Windows with behavioral signal
            "peak_focus_hour": peak_focus_hour,           # Hour with best FDI (int, or None)
            "peak_focus_fdi": peak_focus_fdi,             # FDI at peak hour
        },
        "calendar": {
            # Count actual meeting time as windows × 15 min (avoids double-counting duration)
            "total_meeting_minutes": sum(1 for w in working if w["calendar"]["in_meeting"]) * 15,
            "meeting_windows": sum(1 for w in working if w["calendar"]["in_meeting"]),
        },
        "slack": {
            "total_messages_sent": sum(w["slack"]["messages_sent"] for w in windows),
            "total_messages_received": sum(w["slack"]["messages_received"] for w in windows),
        },
        "whoop": {
            "recovery_score": windows[0]["whoop"].get("recovery_score"),
            "hrv_rmssd_milli": windows[0]["whoop"].get("hrv_rmssd_milli"),
            "resting_heart_rate": windows[0]["whoop"].get("resting_heart_rate"),
            "sleep_hours": windows[0]["whoop"].get("sleep_hours"),
            "sleep_performance": windows[0]["whoop"].get("sleep_performance"),
        },
    }

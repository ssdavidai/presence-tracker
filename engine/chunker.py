"""
Presence Tracker — 15-Minute Window Chunker

Assembles all collected signals into 96 × 15-minute observation windows
for a given day, then computes derived metrics for each window.

Output: list of window dicts, written to data/chunks/YYYY-MM-DD.jsonl
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
) -> list[dict]:
    """
    Build 96 observation windows for a given date.

    Args:
        date_str: "YYYY-MM-DD"
        whoop_data: output from collectors.whoop.collect()
        calendar_data: output from collectors.gcal.collect()
        slack_windows: output from collectors.slack.collect() (keyed by window_index)

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

        # ── Metric computation ────────────────────────────────────────────
        metrics = compute_metrics({
            "calendar": cal_signals,
            "whoop": whoop_signals,
            "slack": slack_signals,
        })

        # ── Metadata ──────────────────────────────────────────────────────
        sources = []
        if whoop_data.get("recovery_score") is not None:
            sources.append("whoop")
        if calendar_data.get("events") is not None:
            sources.append("calendar")
        if window_slack.get("total_messages", 0) > 0 or True:  # Always available
            sources.append("slack")

        window = {
            "window_id": window_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "date": date_str,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "window_index": i,
            "calendar": cal_signals,
            "whoop": whoop_signals,
            "slack": slack_signals,
            "metrics": metrics,
            "metadata": {
                "day_of_week": day_of_week,
                "hour_of_day": hour,
                "minute_of_hour": minute,
                "is_working_hours": WORKING_HOURS_START <= hour < WORKING_HOURS_END,
                "sources_available": sources,
            },
        }
        windows.append(window)

    return windows


def summarize_day(windows: list[dict]) -> dict:
    """
    Compute daily summary statistics from the 96 windows.
    """
    if not windows:
        return {}

    # Working hours only (filter for analysis relevance)
    working = [w for w in windows if w["metadata"]["is_working_hours"]]

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

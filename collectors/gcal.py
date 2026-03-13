"""
Google Calendar Collector

Fetches events for a given date using the gog CLI.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from typing import List, Optional


def _parse_datetime(dt_dict: dict) -> Optional[datetime]:
    """Parse a Google Calendar datetime or date object."""
    if not dt_dict:
        return None
    dt_str = dt_dict.get("dateTime") or dt_dict.get("date")
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _parse_event(event: dict) -> dict:
    """Normalize a raw Google Calendar event into a simple dict."""
    start = _parse_datetime(event.get("start", {}))
    end = _parse_datetime(event.get("end", {}))
    duration_minutes = 0
    if start and end:
        duration_minutes = int((end - start).total_seconds() / 60)

    attendees = event.get("attendees", [])
    attendee_count = len(attendees) if attendees else 1  # At minimum, just David

    return {
        "id": event.get("id", ""),
        "title": event.get("summary", ""),
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "duration_minutes": duration_minutes,
        "attendee_count": attendee_count,
        "organizer_email": event.get("organizer", {}).get("email", ""),
        "is_all_day": "date" in event.get("start", {}),
        "location": event.get("location", ""),
        "status": event.get("status", "confirmed"),
    }


def collect(date_str: str) -> dict:
    """
    Collect calendar events for a given date (YYYY-MM-DD).

    Returns a dict with:
    - events: list of parsed events
    - total_meeting_minutes: total duration of confirmed meetings
    - max_concurrent_attendees: largest meeting by attendee count
    - event_count: number of events
    """
    # Build date range: full day
    date = datetime.strptime(date_str, "%Y-%m-%d")
    next_day = (date + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        result = subprocess.run(
            [
                "gog", "calendar", "events",
                "--account", "david@szabostuban.com",
                "--all",
                "--from", date_str,
                "--to", next_day,
                "--max", "50",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[calendar] gog error: {result.stderr}", file=sys.stderr)
            return _empty()

        data = json.loads(result.stdout)
        raw_events = data.get("events", [])

    except subprocess.TimeoutExpired:
        print("[calendar] gog timed out", file=sys.stderr)
        return _empty()
    except json.JSONDecodeError as e:
        print(f"[calendar] JSON parse error: {e}", file=sys.stderr)
        return _empty()
    except Exception as e:
        print(f"[calendar] Unexpected error: {e}", file=sys.stderr)
        return _empty()

    events = [_parse_event(e) for e in raw_events
              if e.get("status") != "cancelled"
              and not _parse_event(e)["is_all_day"]]

    total_minutes = sum(e["duration_minutes"] for e in events)
    max_attendees = max((e["attendee_count"] for e in events), default=0)

    return {
        "events": events,
        "event_count": len(events),
        "total_meeting_minutes": total_minutes,
        "max_concurrent_attendees": max_attendees,
    }


def _empty() -> dict:
    return {
        "events": [],
        "event_count": 0,
        "total_meeting_minutes": 0,
        "max_concurrent_attendees": 0,
    }


def get_events_in_window(events: list, window_start: datetime, window_end: datetime) -> list:
    """Filter events that overlap with a given 15-minute window."""
    result = []
    for event in events:
        if not event.get("start") or not event.get("end"):
            continue
        try:
            ev_start = datetime.fromisoformat(event["start"])
            ev_end = datetime.fromisoformat(event["end"])
        except Exception:
            continue
        # Remove timezone for comparison if needed
        if ev_start.tzinfo and window_start.tzinfo is None:
            ev_start = ev_start.replace(tzinfo=None)
            ev_end = ev_end.replace(tzinfo=None)
        elif ev_start.tzinfo is None and window_start.tzinfo:
            window_start = window_start.replace(tzinfo=None)
            window_end = window_end.replace(tzinfo=None)
        # Overlap condition
        if ev_start < window_end and ev_end > window_start:
            result.append(event)
    return result


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    result = collect(date)
    print(json.dumps(result, indent=2))

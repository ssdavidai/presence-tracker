"""
RescueTime Collector

Fetches app usage, focus depth, and distraction data from the RescueTime
Analytic Data API and maps it to 15-minute windows.

Data signals extracted per window:
- focus_seconds: time in productive/very_productive apps
- distraction_seconds: time in distracting/very_distracting apps
- neutral_seconds: neutral app time
- active_seconds: total computer-active time in this window
- app_switches: number of distinct activities (proxy for context switching)
- productivity_score: weighted productivity score (−2 to +2 scale, normalized)
- top_activity: most-used app/site in this window

RescueTime API key is read from environment variable RESCUETIME_API_KEY
or from ~/.clawdbot/clawdbot.json.

Productivity coding from RescueTime:
  +2 = Very Productive (focus work)
  +1 = Productive
   0 = Neutral
  -1 = Distracting
  -2 = Very Distracting (entertainment, social)

Context switch proxy: number of distinct activities in a 15-min window.
More app switches = more fragmentation = higher CSC.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────

RESCUETIME_API_BASE = "https://www.rescuetime.com/anapi/data"
RESCUETIME_DAILY_SUMMARY_URL = "https://www.rescuetime.com/anapi/daily_summary_feed"

# Productivity normalization: RescueTime uses -2 to +2; we map to 0.0 to 1.0
# where 1.0 = very productive and 0.0 = very distracting
_PRODUCTIVITY_LEVELS = {
    2: "very_productive",   # Very Productive (focus work)
    1: "productive",        # Productive
    0: "neutral",           # Neutral
    -1: "distracting",      # Distracting
    -2: "very_distracting", # Very Distracting
}

# Productivity codes that count as "focus time" (signal for FDI)
_FOCUS_PRODUCTIVITY_LEVELS = {2, 1}  # Very Productive + Productive

# Productivity codes that count as "distraction time"
_DISTRACTION_PRODUCTIVITY_LEVELS = {-1, -2}  # Distracting + Very Distracting

# Seconds per 15-minute window (for normalization)
_WINDOW_SECONDS = 900

# Maximum app switches per window before saturation (for normalization)
_MAX_APP_SWITCHES = 8


# ─── API Key loading ──────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """
    Load RescueTime API key.

    Priority order:
    1. RESCUETIME_API_KEY environment variable
    2. ~/.clawdbot/clawdbot.json (key: "rescuetime_api_key")
    3. ~/.clawdbot/ environment file (any file containing the key)
    """
    # 1. Environment variable
    key = os.environ.get("RESCUETIME_API_KEY", "")
    if key:
        return key

    # 2. clawdbot.json
    config_path = Path.home() / ".clawdbot" / "clawdbot.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            key = config.get("rescuetime_api_key", "")
            if key:
                return key
        except Exception:
            pass

    raise RuntimeError(
        "RescueTime API key not found. Set RESCUETIME_API_KEY environment variable "
        "or add 'rescuetime_api_key' to ~/.clawdbot/clawdbot.json"
    )


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def _rt_get(api_key: str, params: dict, base_url: str = RESCUETIME_API_BASE) -> dict:
    """
    Make a GET request to the RescueTime API with retry/backoff.

    Returns the parsed JSON response, or an empty dict on failure.
    """
    params["key"] = api_key
    params["format"] = "json"
    url = base_url + "?" + urllib.parse.urlencode(params)

    backoff = 1.0
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "presence-tracker/1.0"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited — back off
                time.sleep(backoff)
                backoff *= 2
                continue
            print(f"[rescuetime] HTTP error {e.code}: {e.reason}", file=sys.stderr)
            return {}
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            if attempt < 3:
                time.sleep(backoff)
                backoff *= 2
                continue
            print(f"[rescuetime] Request failed after {attempt+1} attempts: {e}", file=sys.stderr)
            return {}
    return {}


# ─── Data parsing helpers ─────────────────────────────────────────────────────

def _parse_activity_rows(response: dict) -> list[dict]:
    """
    Parse the rows array from the RescueTime analytic API response into
    a list of dicts keyed by row_headers.

    Activity-level response has headers:
        [Date, Time Spent (seconds), Number of People, Activity, Category, Productivity]

    Productivity-level response has headers:
        [Date, Time Spent (seconds), Number of People, Productivity]
    """
    headers = response.get("row_headers", [])
    rows = response.get("rows", [])
    if not headers or not rows:
        return []

    # Normalize header names to snake_case
    _HEADER_MAP = {
        "Date": "date",
        "Time Spent (seconds)": "seconds",
        "Number of People": "people",
        "Activity": "activity",
        "Category": "category",
        "Productivity": "productivity",
    }
    normalized_headers = [_HEADER_MAP.get(h, h.lower().replace(" ", "_")) for h in headers]

    result = []
    for row in rows:
        record = dict(zip(normalized_headers, row))
        # Parse the date string into a datetime object
        if "date" in record and isinstance(record["date"], str):
            try:
                record["dt"] = datetime.fromisoformat(record["date"])
            except ValueError:
                record["dt"] = None
        result.append(record)
    return result


def _window_index_for_dt(dt: datetime) -> int:
    """
    Map a datetime to a 15-minute window index (0–95).
    Window 0 = 00:00–00:15, Window 36 = 09:00–09:15, etc.
    """
    return (dt.hour * 4) + (dt.minute // 15)


# ─── Per-window aggregation ───────────────────────────────────────────────────

def _aggregate_to_windows(activity_rows: list[dict]) -> dict[int, dict]:
    """
    Aggregate 5-minute activity data into 96 × 15-minute windows.

    For each window:
    - focus_seconds: seconds in productive/very productive activities
    - distraction_seconds: seconds in distracting activities
    - neutral_seconds: seconds in neutral activities
    - active_seconds: total logged computer time
    - app_count: distinct activities seen (proxy for context switches)
    - productivity_weighted_seconds: sum of (seconds × productivity_score)
    - activities: list of activity names seen

    The 5-minute RescueTime buckets may span window boundaries.
    We do a simple assignment: the bucket timestamp's window gets all
    the seconds. This is a known approximation — RescueTime doesn't
    provide sub-minute precision.
    """
    windows: dict[int, dict] = {}
    for i in range(96):
        windows[i] = {
            "focus_seconds": 0,
            "distraction_seconds": 0,
            "neutral_seconds": 0,
            "active_seconds": 0,
            "app_count": 0,
            "_activities": set(),
            "_productivity_weighted": 0.0,
        }

    for row in activity_rows:
        dt: Optional[datetime] = row.get("dt")
        if dt is None:
            continue
        seconds: int = int(row.get("seconds", 0))
        if seconds <= 0:
            continue
        productivity: int = int(row.get("productivity", 0))
        activity_name: str = str(row.get("activity", "unknown"))
        idx = _window_index_for_dt(dt)

        if idx < 0 or idx >= 96:
            continue

        w = windows[idx]
        w["active_seconds"] += seconds
        w["_productivity_weighted"] += seconds * productivity
        w["_activities"].add(activity_name)

        if productivity in _FOCUS_PRODUCTIVITY_LEVELS:
            w["focus_seconds"] += seconds
        elif productivity in _DISTRACTION_PRODUCTIVITY_LEVELS:
            w["distraction_seconds"] += seconds
        else:
            w["neutral_seconds"] += seconds

    # Post-process: convert sets to counts, compute app_switches
    for idx, w in windows.items():
        activities = w.pop("_activities")
        w["app_count"] = len(activities)
        # App switches = number of distinct apps − 1 (minimum 0)
        w["app_switches"] = max(0, len(activities) - 1)
        # Top activity (not stored in output but useful internally)
        # Note: we don't have per-row duration by app here for aggregation,
        # so we just leave top_activity as None at window level.
        w["top_activity"] = None

        # Normalize productivity score: map from [-2*active, +2*active] to [0, 1]
        if w["active_seconds"] > 0:
            weighted = w.pop("_productivity_weighted")
            # raw score: -2 to +2
            raw = weighted / w["active_seconds"]
            # normalize to [0, 1]: (raw + 2) / 4
            w["productivity_score"] = round((raw + 2.0) / 4.0, 4)
        else:
            w.pop("_productivity_weighted")
            w["productivity_score"] = None

    return windows


# ─── Public collector ─────────────────────────────────────────────────────────

def collect(date_str: str) -> dict[int, dict]:
    """
    Collect RescueTime activity data for a given date (YYYY-MM-DD).

    Returns a dict keyed by window_index (0–95), each value:
    {
        "focus_seconds": int,          # seconds in productive apps
        "distraction_seconds": int,    # seconds in distracting apps
        "neutral_seconds": int,        # seconds in neutral apps
        "active_seconds": int,         # total active computer time
        "app_switches": int,           # proxy for context switch count
        "productivity_score": float|None,  # 0.0 (distracted) to 1.0 (focused)
        "top_activity": str|None,      # most active app (from hourly data)
    }

    Windows with no RescueTime data will have all zeros.
    This is normal for hours when the computer wasn't in use.
    """
    api_key = _get_api_key()

    # Fetch activity-level data at 5-minute resolution for context switches
    activity_response = _rt_get(api_key, {
        "perspective": "interval",
        "restrict_kind": "activity",
        "resolution_time": "minute",  # 5-minute buckets
        "restrict_begin": date_str,
        "restrict_end": date_str,
    })
    activity_rows = _parse_activity_rows(activity_response)

    # Aggregate into 15-minute windows
    windows = _aggregate_to_windows(activity_rows)

    # Fetch hourly-level data to get top activity per window
    # (hourly data maps directly to window groups without partial-window issues)
    hourly_response = _rt_get(api_key, {
        "perspective": "interval",
        "restrict_kind": "activity",
        "resolution_time": "hour",
        "restrict_begin": date_str,
        "restrict_end": date_str,
    })
    hourly_rows = _parse_activity_rows(hourly_response)

    # Map top activity per hour → broadcast to the 4 windows in that hour
    hour_top: dict[int, tuple[str, int]] = {}  # hour → (activity, seconds)
    hour_activities: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for row in hourly_rows:
        dt: Optional[datetime] = row.get("dt")
        if dt is None:
            continue
        hour_activities[dt.hour].append((
            str(row.get("activity", "unknown")),
            int(row.get("seconds", 0))
        ))

    for hour, acts in hour_activities.items():
        if acts:
            top_act = max(acts, key=lambda x: x[1])[0]
            for q in range(4):
                idx = hour * 4 + q
                if 0 <= idx < 96:
                    windows[idx]["top_activity"] = top_act

    return windows


def collect_daily_summary(date_str: str) -> Optional[dict]:
    """
    Fetch the daily summary for a specific date from RescueTime.

    Returns a dict with high-level productivity percentages and hours,
    or None if the date is not yet available (data lags by ~1 day).

    Useful for the daily digest and rolling stats enrichment.
    """
    api_key = _get_api_key()
    response = _rt_get(api_key, {}, base_url=RESCUETIME_DAILY_SUMMARY_URL)

    if not isinstance(response, list):
        return None

    for summary in response:
        if summary.get("date") == date_str:
            return summary
    return None


def get_window_data(rt_windows: dict, window_index: int) -> dict:
    """
    Get RescueTime data for a specific window index.
    Returns a zero-filled dict if the window is not present.
    """
    default = {
        "focus_seconds": 0,
        "distraction_seconds": 0,
        "neutral_seconds": 0,
        "active_seconds": 0,
        "app_switches": 0,
        "productivity_score": None,
        "top_activity": None,
    }
    return rt_windows.get(window_index, default)


# ─── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    print(f"[rescuetime] Collecting data for {date}...")
    result = collect(date)
    # Print non-empty windows only
    active = {k: v for k, v in result.items() if v["active_seconds"] > 0}
    if active:
        print(f"[rescuetime] Found data in {len(active)} windows:")
        print(json.dumps(active, indent=2))
    else:
        print("[rescuetime] No activity data found for this date.")

    summary = collect_daily_summary(date)
    if summary:
        print(f"\n[rescuetime] Daily summary:")
        print(json.dumps({
            k: v for k, v in summary.items()
            if "percentage" in k or "pulse" in k or k == "date"
        }, indent=2))
    else:
        print(f"\n[rescuetime] No daily summary available for {date} (may lag by 1 day).")

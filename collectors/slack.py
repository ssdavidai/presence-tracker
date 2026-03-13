"""
Slack Collector

Fetches David's Slack activity for a given day:
- Messages sent by David
- Messages received (in David's active channels)
- Channel activity breakdown

Uses the Slack Web API with the bot token from OpenClaw config.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ─── Token loading ───────────────────────────────────────────────────────────

OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
DAVID_USER_ID = "U08UGBQL5J5"
ALFRED_BOT_USER_ID = "U08U7C4DSBE"

# Channels to monitor for activity (David's key channels)
MONITORED_CHANNELS = [
    "D08U7C4G6TE",   # David's DM with Alfred
    "C08UGBR3PQ9",   # general
    "C08U9EB1PHD",   # ask_alfred
    "C0ACVH414JC",   # notifications
    "C08U44YTEBF",   # malik
    "C08U5FSQ62W",   # money
]


def _get_token() -> str:
    try:
        config = json.loads(OPENCLAW_CONFIG.read_text())
        return config["channels"]["slack"]["botToken"]
    except Exception as e:
        raise RuntimeError(f"Could not read Slack token from OpenClaw config: {e}")


def _slack_get(token: str, endpoint: str, params: dict) -> dict:
    """Make a Slack Web API GET request."""
    url = f"https://slack.com/api/{endpoint}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })
    backoff = 1
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
                if not data.get("ok"):
                    if data.get("error") == "ratelimited":
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    # Return empty but don't crash on permission errors
                    print(f"[slack] API error: {data.get('error')} for {endpoint}", file=sys.stderr)
                    return {}
                return data
        except urllib.error.URLError as e:
            if attempt < 3:
                time.sleep(backoff)
                backoff *= 2
                continue
            print(f"[slack] Request failed: {e}", file=sys.stderr)
            return {}
    return {}


def _fetch_channel_messages(token: str, channel_id: str, oldest: float, latest: float) -> list:
    """Fetch all messages in a channel within a time window."""
    messages = []
    cursor = None
    while True:
        params = {
            "channel": channel_id,
            "oldest": str(oldest),
            "latest": str(latest),
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        data = _slack_get(token, "conversations.history", params)
        if not data:
            break
        messages.extend(data.get("messages", []))
        next_cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(0.5)  # Rate limiting
    return messages


def collect(date_str: str) -> dict:
    """
    Collect Slack activity for a given date (YYYY-MM-DD).

    Returns per-15min-window activity summary.
    """
    token = _get_token()

    date = datetime.strptime(date_str, "%Y-%m-%d")
    oldest = date.timestamp()
    latest = (date + timedelta(days=1)).timestamp()

    # Initialize per-window buckets (96 windows = 24 hours × 4 per hour)
    # window_index 0 = 00:00-00:15, 1 = 00:15-00:30, etc.
    windows: dict[int, dict] = {}
    for i in range(96):
        windows[i] = {
            "messages_sent_by_david": 0,
            "messages_received": 0,
            "channels_active": set(),
            "total_messages": 0,
        }

    # Fetch messages from monitored channels
    for channel_id in MONITORED_CHANNELS:
        msgs = _fetch_channel_messages(token, channel_id, oldest, latest)
        for msg in msgs:
            # Skip bots/system messages (but keep David's messages)
            msg_type = msg.get("type", "")
            if msg_type != "message":
                continue
            subtype = msg.get("subtype", "")
            if subtype in ("bot_message", "channel_join", "channel_leave"):
                continue

            ts = float(msg.get("ts", 0))
            if ts < oldest or ts >= latest:
                continue

            msg_dt = datetime.fromtimestamp(ts)
            window_index = (msg_dt.hour * 4) + (msg_dt.minute // 15)

            if window_index not in windows:
                continue

            user = msg.get("user", "")
            windows[window_index]["total_messages"] += 1
            windows[window_index]["channels_active"].add(channel_id)

            if user == DAVID_USER_ID:
                windows[window_index]["messages_sent_by_david"] += 1
            else:
                windows[window_index]["messages_received"] += 1

        time.sleep(0.3)  # Be gentle with the API

    # Convert sets to counts and build output
    result = {}
    for i, w in windows.items():
        result[i] = {
            "messages_sent": w["messages_sent_by_david"],
            "messages_received": w["messages_received"],
            "total_messages": w["total_messages"],
            "channels_active": len(w["channels_active"]),
        }

    return result


def get_window_data(slack_windows: dict, window_index: int) -> dict:
    """Get Slack data for a specific window index."""
    default = {
        "messages_sent": 0,
        "messages_received": 0,
        "total_messages": 0,
        "channels_active": 0,
    }
    return slack_windows.get(window_index, default)


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    result = collect(date)
    # Print only non-zero windows
    print(json.dumps({k: v for k, v in result.items() if v["total_messages"] > 0}, indent=2))

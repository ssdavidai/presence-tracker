"""
Presence Tracker — Omi Transcript Collector

Reads Omi ambient audio transcripts from ~/omi/transcripts/YYYY-MM-DD/
and maps each session to 15-minute observation windows.

Data signals extracted per window:
- conversation_active: bool — at least one transcript session started in this window
- word_count: int — total words spoken across all sessions in this window
- speech_seconds: float — total speech duration (actual speaking time, not audio length)
- audio_seconds: float — total audio recording duration (including silences)
- sessions_count: int — number of distinct Omi sessions starting in this window
- speech_ratio: float — speech_seconds / audio_seconds (how much of recording was speech)

Metric impact:
- CLS: active conversation = cognitive load (processing + responding in real-time)
  Word-dense windows signal high cognitive engagement.
- SDI: spoken conversation = social energy expenditure — even more direct than Slack.
  Speech time in a window is a strong signal for social drain.
- FDI: active conversation interrupts deep focus; it's counted as disruption.
- sources_available: "omi" added when conversation data is present.

Transcript format (~/omi/transcripts/YYYY-MM-DD/HH-MM-SS_uid.json):
{
  "uid": "53g2vVx9...",
  "text": "full transcript text",
  "language": "en",
  "timestamp": "2026-03-12T09:12:53.979953",   ← naive local time (Europe/Budapest)
  "audio_duration_seconds": 251.9,
  "speech_duration_seconds": 171.26
}

The timestamp in the file is naive (no timezone).  We assume Europe/Budapest
because that's where David lives and where the device is used.

Window assignment:
  Each transcript is assigned to the 15-min window containing its start timestamp.
  A long session (e.g. 4 minutes) that starts in window N is assigned entirely to N.
  This is a deliberate simplification — the window marks "when was David engaged
  in conversation?" rather than "exactly how many seconds per window?"

Usage:
    from collectors.omi import collect
    omi_windows = collect("2026-03-12")
    # Returns dict: window_index → {conversation_active, word_count, ...}

    # Or from CLI:
    python3 collectors/omi.py 2026-03-12
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

# ─── Configuration ────────────────────────────────────────────────────────────

OMI_TRANSCRIPTS_DIR = Path.home() / "omi" / "transcripts"
TIMEZONE = ZoneInfo("Europe/Budapest")
WINDOW_MINUTES = 15
WINDOWS_PER_DAY = 96


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _window_index_for_time(dt: datetime) -> int:
    """
    Return the 0-based window index (0–95) for a given datetime.

    Index 0 = 00:00–00:15, index 36 = 09:00–09:15, etc.
    If the datetime falls outside 00:00–23:59 on its date, clamps to 0 or 95.
    """
    total_minutes = dt.hour * 60 + dt.minute
    idx = total_minutes // WINDOW_MINUTES
    return max(0, min(WINDOWS_PER_DAY - 1, idx))


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parse Omi timestamp string into a timezone-aware datetime.

    The Omi receiver writes naive local timestamps (Europe/Budapest).
    We attach the timezone so downstream window-assignment logic is consistent.

    Handles:
    - "2026-03-12T09:12:53.979953"      (no tz, assumed local)
    - "2026-03-12T09:12:53"             (no tz, no microseconds)
    - "2026-03-12T09:12:53+01:00"       (tz-aware, leave as-is → convert to local)
    """
    if not ts_str:
        return None
    try:
        # Try full ISO format first
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            # Naive timestamp → assume Budapest local time
            dt = dt.replace(tzinfo=TIMEZONE)
        else:
            # Tz-aware → convert to Budapest local time for consistency
            dt = dt.astimezone(TIMEZONE)
        return dt
    except (ValueError, TypeError):
        return None


def _count_words(text: str) -> int:
    """Count whitespace-separated words in a transcript text."""
    if not text:
        return 0
    return len(text.split())


# ─── Main collector ───────────────────────────────────────────────────────────

def collect(date_str: str) -> dict[int, dict]:
    """
    Collect Omi transcript signals for a given date and map to window indices.

    Args:
        date_str: "YYYY-MM-DD"

    Returns:
        Dict mapping window_index (0–95) → signal dict:
        {
            "conversation_active": bool,
            "word_count": int,
            "speech_seconds": float,
            "audio_seconds": float,
            "sessions_count": int,
            "speech_ratio": float,   # speech_seconds / audio_seconds (0.0–1.0)
        }

        Only windows that had at least one Omi session are included in the dict.
        Windows with no Omi data are absent (not None — caller should use .get(i, {})).
    """
    transcript_dir = OMI_TRANSCRIPTS_DIR / date_str

    if not transcript_dir.exists():
        # No Omi data for this date — completely normal, many days won't have it.
        return {}

    # Accumulate per-window stats
    window_data: dict[int, dict] = {}

    transcript_files = sorted(transcript_dir.glob("*.json"))
    if not transcript_files:
        return {}

    for tf in transcript_files:
        try:
            with tf.open() as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Skip corrupt/unreadable files silently — robustness over noise
            continue

        # Parse timestamp
        ts = _parse_timestamp(record.get("timestamp", ""))
        if ts is None:
            continue

        # Only process transcripts whose date matches the requested date
        # (guards against cross-midnight sessions ending up in the wrong dir)
        if ts.strftime("%Y-%m-%d") != date_str:
            continue

        # Assign to window
        idx = _window_index_for_time(ts)

        text = record.get("text", "") or ""
        audio_secs = float(record.get("audio_duration_seconds") or 0.0)
        speech_secs = float(record.get("speech_duration_seconds") or 0.0)
        words = _count_words(text)

        if idx not in window_data:
            window_data[idx] = {
                "conversation_active": False,
                "word_count": 0,
                "speech_seconds": 0.0,
                "audio_seconds": 0.0,
                "sessions_count": 0,
                "speech_ratio": 0.0,
            }

        w = window_data[idx]
        w["conversation_active"] = True
        w["word_count"] += words
        w["speech_seconds"] += speech_secs
        w["audio_seconds"] += audio_secs
        w["sessions_count"] += 1

    # Compute speech_ratio per window (avoid division by zero)
    for w in window_data.values():
        if w["audio_seconds"] > 0:
            w["speech_ratio"] = round(
                min(1.0, w["speech_seconds"] / w["audio_seconds"]), 4
            )
        else:
            w["speech_ratio"] = 0.0

        # Round floats for clean JSONL output
        w["speech_seconds"] = round(w["speech_seconds"], 1)
        w["audio_seconds"] = round(w["audio_seconds"], 1)

    return window_data


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Omi transcript signals for a date")
    parser.add_argument("date", help="Date in YYYY-MM-DD format")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    results = collect(args.date)

    if not results:
        print(f"No Omi transcripts found for {args.date}")
        sys.exit(0)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Omi transcripts for {args.date}:")
        print(f"  Windows with conversation: {len(results)}")
        total_words = sum(w["word_count"] for w in results.values())
        total_speech = sum(w["speech_seconds"] for w in results.values())
        total_audio = sum(w["audio_seconds"] for w in results.values())
        print(f"  Total words: {total_words:,}")
        print(f"  Total speech: {total_speech:.0f}s ({total_speech/60:.1f}min)")
        print(f"  Total audio: {total_audio:.0f}s ({total_audio/60:.1f}min)")
        print()
        for idx, w in sorted(results.items()):
            hour = (idx * 15) // 60
            minute = (idx * 15) % 60
            print(f"  {hour:02d}:{minute:02d} — {w['sessions_count']} session(s), "
                  f"{w['word_count']} words, "
                  f"speech={w['speech_seconds']:.0f}s, "
                  f"ratio={w['speech_ratio']:.0%}")

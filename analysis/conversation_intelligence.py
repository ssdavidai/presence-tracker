"""
Presence Tracker — Conversation Intelligence (v1)

Answers: *"What are David's conversation patterns, and how cognitively demanding are they?"*

The Omi transcript collector feeds real-time conversation signals into each 15-min
window (word_count, speech_seconds, topic_category).  But this only surfaces
when a full daily ingestion has been run — meaning the 28+ days of Omi transcript
history are largely invisible to the analytics layer.

The Conversation Intelligence module reads raw Omi transcript files directly,
independent of the JSONL store, and surfaces historical conversation patterns:

## What it computes

### Daily conversation profile
  - total_sessions: int         — Omi recording sessions started this day
  - total_words: int            — all words spoken (across all sessions)
  - speech_minutes: float       — total speech duration (not audio)
  - speech_ratio: float         — speech_minutes / audio_minutes (density)
  - primary_language: str       — "en" | "hu" | "mixed"
  - dominant_topic: str         — most frequent category across sessions
  - avg_cognitive_density: float — mean density across classified sessions

### Hourly conversation distribution (working hours 7am–10pm)
  - words_by_hour: dict[int, int]   — total words per hour of day
  - peak_hour: int                  — hour with most words (conversation peak)

### Cross-day trends (across requested window)
  - ConversationWeek dataclass:
      - date_range: str               — "YYYY-MM-DD → YYYY-MM-DD"
      - days_analysed: int
      - total_speech_minutes: float
      - total_words: int
      - avg_speech_minutes_per_day: float
      - avg_cognitive_density: float
      - peak_conversation_hour: int
      - heavy_days: list[str]         — days with > 90th percentile word count
      - light_days: list[str]         — days with < 10th percentile word count
      - language_split: dict          — {"en": N, "hu": N, "mixed": N}
      - topic_distribution: dict      — {category: proportion}
      - hourly_profile: dict[int, int] — words per hour aggregated across days
      - daily_summaries: list[dict]   — one per day, sorted by date

### Language intelligence
  English vs Hungarian conversation patterns matter because:
  - English = typically work conversations (higher cognitive load)
  - Hungarian = typically personal / family conversations (different cognitive profile)
  Days with heavy English conversation load correlate with higher CLS.

### Cognitive density trend
  cognitive_density from the topic classifier measures how lexically and
  conceptually demanding conversations are.  A trend of rising density
  = increasing conversation-driven cognitive load.

## API

    from analysis.conversation_intelligence import (
        analyse_conversation_history,
        format_conversation_intelligence_section,
        format_conversation_brief_line,
    )

    # Analyse last 14 days
    ci = analyse_conversation_history(days=14)

    # One-liner for morning brief
    line = format_conversation_brief_line(ci)

    # Full section for weekly summary
    section = format_conversation_intelligence_section(ci)

## CLI

    python3 analysis/conversation_intelligence.py              # Last 14 days
    python3 analysis/conversation_intelligence.py --days 28    # Last 28 days
    python3 analysis/conversation_intelligence.py --json       # JSON output
    python3 analysis/conversation_intelligence.py 2026-03-01   # From specific date

## Design principles

  - Zero dependency on JSONL store — reads Omi transcripts directly
  - Works with partial data (graceful when some days have no transcripts)
  - No external dependencies — pure stdlib + project omi_topics module
  - Degrades gracefully: min_days_for_meaningful=3 check before interpreting

## Integration

  Wire into weekly summary to surface "Conversation Intelligence" section.
  Wire into morning brief for a one-line conversation-load context.
"""

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Constants ────────────────────────────────────────────────────────────────

OMI_TRANSCRIPTS_DIR = Path.home() / "omi" / "transcripts"

# Minimum days with Omi data before drawing conclusions
MIN_DAYS_FOR_MEANINGFUL = 3

# Minimum words in a session to be worth classifying (filter noise)
MIN_WORDS_FOR_CLASSIFY = 20

# Words/day thresholds for heavy/light classification
HEAVY_DAY_PERCENTILE = 0.80  # top 20% by word count = heavy
LIGHT_DAY_PERCENTILE = 0.20  # bottom 20% by word count = light

# Working hours window for hourly profile
WORKING_HOURS_START = 7
WORKING_HOURS_END = 22


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class DailyConversationSummary:
    """Conversation profile for a single day."""
    date: str
    total_sessions: int
    total_words: int
    speech_minutes: float
    audio_minutes: float
    speech_ratio: float           # speech_minutes / audio_minutes
    primary_language: str         # "en" | "hu" | "mixed" | "unknown"
    dominant_topic: str           # most frequent category
    avg_cognitive_density: float
    words_by_hour: dict           # hour → word count
    peak_hour: Optional[int]      # hour with most words
    topic_distribution: dict      # category → count
    language_counts: dict         # language → session count
    is_meaningful: bool           # True when ≥ 1 classifiable session


@dataclass
class ConversationIntelligence:
    """Cross-day conversation intelligence summary."""
    date_range: str               # "YYYY-MM-DD → YYYY-MM-DD"
    days_requested: int
    days_with_data: int           # days that had ≥ 1 Omi session
    total_speech_minutes: float
    total_words: int
    avg_speech_minutes_per_day: float
    avg_words_per_day: float
    avg_cognitive_density: float
    peak_conversation_hour: int   # hour with most words across all days
    heavy_days: list              # dates above HEAVY_DAY_PERCENTILE
    light_days: list              # dates below LIGHT_DAY_PERCENTILE
    language_split: dict          # {"en": N, "hu": N, "mixed": N}
    dominant_language: str        # language with most session-days
    topic_distribution: dict      # {category: proportion}
    dominant_topic: str           # most frequent topic category
    hourly_profile: dict          # hour → total words (aggregated)
    daily_summaries: list         # DailyConversationSummary objects
    is_meaningful: bool           # False when < MIN_DAYS_FOR_MEANINGFUL days with data
    # Trend signals
    trend_direction: str          # "increasing" | "decreasing" | "stable"
    trend_description: str        # one-sentence trend summary
    insight_lines: list           # list of actionable insight strings


# ─── Core analysis functions ─────────────────────────────────────────────────

def _load_day_transcripts(date_str: str) -> list[dict]:
    """
    Load all Omi transcript JSON files for a given date.
    Returns list of raw transcript dicts.  Empty list if no data.
    """
    day_dir = OMI_TRANSCRIPTS_DIR / date_str
    if not day_dir.exists() or not day_dir.is_dir():
        return []

    transcripts = []
    for f in sorted(day_dir.iterdir()):
        if not f.suffix == ".json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            # Validate required fields
            if "text" in data and "timestamp" in data:
                transcripts.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return transcripts


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse naive ISO timestamp from Omi (assumed Europe/Budapest local time)."""
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def _classify_session(text: str, speech_seconds: float) -> dict:
    """
    Classify a single transcript session.
    Returns classify_transcript result or a minimal fallback dict.
    """
    try:
        from collectors.omi_topics import classify_transcript
        return classify_transcript(text, speech_seconds=speech_seconds)
    except Exception:
        return {
            "category": "unknown",
            "cognitive_density": 0.0,
            "confidence": 0.0,
            "topic_signals": [],
            "language": "unknown",
        }


def analyse_day(date_str: str) -> DailyConversationSummary:
    """
    Analyse Omi transcript data for a single day.
    Always returns a DailyConversationSummary (with is_meaningful=False when no data).
    """
    transcripts = _load_day_transcripts(date_str)

    if not transcripts:
        return DailyConversationSummary(
            date=date_str,
            total_sessions=0,
            total_words=0,
            speech_minutes=0.0,
            audio_minutes=0.0,
            speech_ratio=0.0,
            primary_language="unknown",
            dominant_topic="unknown",
            avg_cognitive_density=0.0,
            words_by_hour={},
            peak_hour=None,
            topic_distribution={},
            language_counts={},
            is_meaningful=False,
        )

    total_words = 0
    total_speech_secs = 0.0
    total_audio_secs = 0.0
    words_by_hour: dict[int, int] = {}
    topic_counts: dict[str, int] = {}
    lang_counts: dict[str, int] = {}
    density_vals: list[float] = []

    for t in transcripts:
        text = t.get("text", "") or ""
        words = len(text.split())
        speech_secs = t.get("speech_duration_seconds", 0.0) or 0.0
        audio_secs = t.get("audio_duration_seconds", 0.0) or 0.0

        total_words += words
        total_speech_secs += speech_secs
        total_audio_secs += audio_secs

        # Hour assignment
        ts = _parse_timestamp(t.get("timestamp", ""))
        if ts:
            h = ts.hour
            words_by_hour[h] = words_by_hour.get(h, 0) + words

        # Classify sessions with enough content
        if words >= MIN_WORDS_FOR_CLASSIFY:
            result = _classify_session(text, speech_secs)
            cat = result.get("category", "unknown")
            topic_counts[cat] = topic_counts.get(cat, 0) + 1
            lang = result.get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            density = result.get("cognitive_density", 0.0)
            if density > 0:
                density_vals.append(density)
        else:
            # Short sessions: track language only if detectable from raw data
            # Use a simple heuristic — transcripts with Hungarian chars
            if any(c in text for c in "áéíóöőúüűÁÉÍÓÖŐÚÜŰ"):
                lang_counts["hu"] = lang_counts.get("hu", 0) + 1
            elif text.strip():
                lang_counts["en"] = lang_counts.get("en", 0) + 1

    # Compute aggregates
    speech_minutes = total_speech_secs / 60.0
    audio_minutes = total_audio_secs / 60.0
    speech_ratio = speech_minutes / max(audio_minutes, 1.0)

    # Peak hour (working hours only if possible)
    working_hours_words = {h: w for h, w in words_by_hour.items()
                           if WORKING_HOURS_START <= h <= WORKING_HOURS_END}
    peak_hour = None
    if working_hours_words:
        peak_hour = max(working_hours_words, key=working_hours_words.get)
    elif words_by_hour:
        peak_hour = max(words_by_hour, key=words_by_hour.get)

    # Dominant topic
    dominant_topic = "unknown"
    if topic_counts:
        dominant_topic = max(topic_counts, key=topic_counts.get)

    # Primary language (most sessions)
    primary_language = "unknown"
    if lang_counts:
        top_lang = max(lang_counts, key=lang_counts.get)
        total_lang_sessions = sum(lang_counts.values())
        top_share = lang_counts[top_lang] / total_lang_sessions
        if top_share >= 0.6:
            primary_language = top_lang
        else:
            primary_language = "mixed"

    avg_density = sum(density_vals) / len(density_vals) if density_vals else 0.0

    return DailyConversationSummary(
        date=date_str,
        total_sessions=len(transcripts),
        total_words=total_words,
        speech_minutes=speech_minutes,
        audio_minutes=audio_minutes,
        speech_ratio=speech_ratio,
        primary_language=primary_language,
        dominant_topic=dominant_topic,
        avg_cognitive_density=avg_density,
        words_by_hour=words_by_hour,
        peak_hour=peak_hour,
        topic_distribution=topic_counts,
        language_counts=lang_counts,
        is_meaningful=(len(transcripts) > 0),
    )


def _compute_trend(word_counts: list[float]) -> str:
    """
    Simple linear trend over a list of daily word counts.
    Returns "increasing" | "decreasing" | "stable".
    """
    if len(word_counts) < 3:
        return "stable"

    n = len(word_counts)
    # Simple least-squares slope
    x_mean = (n - 1) / 2
    y_mean = sum(word_counts) / n
    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(word_counts))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return "stable"

    slope = numerator / denominator
    slope_pct = slope / max(y_mean, 1.0)

    if slope_pct > 0.05:
        return "increasing"
    elif slope_pct < -0.05:
        return "decreasing"
    return "stable"


def _generate_insights(
    ci_data: dict,
    daily_summaries: list[DailyConversationSummary],
) -> list[str]:
    """
    Generate 2–4 actionable insight strings from the conversation intelligence data.
    """
    insights = []

    avg_speech = ci_data["avg_speech_minutes_per_day"]
    peak_hour = ci_data["peak_conversation_hour"]
    dominant_lang = ci_data["dominant_language"]
    dominant_topic = ci_data["dominant_topic"]
    trend = ci_data["trend_direction"]
    avg_density = ci_data["avg_cognitive_density"]
    heavy_days = ci_data["heavy_days"]
    lang_split = ci_data["language_split"]
    days_with_data = ci_data["days_with_data"]

    # 1. Conversation load insight
    if avg_speech > 180:
        insights.append(
            f"📣 Heavy conversation load — averaging {avg_speech:.0f} min/day of speech. "
            "Consider protecting more silent focus time."
        )
    elif avg_speech > 90:
        insights.append(
            f"💬 Moderate conversation load — {avg_speech:.0f} min/day of speech. "
            "Typical for an active workday."
        )
    elif avg_speech > 0:
        insights.append(
            f"🤫 Light conversation load — {avg_speech:.0f} min/day of speech on tracked days."
        )

    # 2. Peak conversation hour insight
    if peak_hour is not None:
        hour_label = f"{peak_hour}:00"
        insights.append(
            f"⏰ Peak conversation time: {hour_label} — schedule deep focus before this block."
        )

    # 3. Language pattern insight
    total_lang = sum(lang_split.values())
    if total_lang > 0 and dominant_lang in ("en", "hu"):
        en_count = lang_split.get("en", 0)
        hu_count = lang_split.get("hu", 0)
        if dominant_lang == "en" and en_count > 0:
            en_pct = int(100 * en_count / total_lang)
            insights.append(
                f"🌐 {en_pct}% of tracked sessions in English — typically higher cognitive load "
                "(technical/work context)."
            )
        elif dominant_lang == "hu" and hu_count > 0:
            hu_pct = int(100 * hu_count / total_lang)
            insights.append(
                f"🗣️ {hu_pct}% of tracked sessions in Hungarian — predominantly personal/family context."
            )

    # 4. Trend insight
    if trend == "increasing":
        insights.append(
            "📈 Conversation load is trending up over this period — watch for accumulated social drain."
        )
    elif trend == "decreasing":
        insights.append(
            "📉 Conversation load is trending down — good if intentional (deep work mode)."
        )

    # 5. Cognitive density insight
    if avg_density > 0.55:
        insights.append(
            f"🧠 High cognitive conversation density ({avg_density:.2f}) — discussions are "
            "technically/strategically demanding. Factor this into load estimates."
        )
    elif avg_density > 0.40:
        insights.append(
            f"💡 Moderate cognitive conversation density ({avg_density:.2f}) — a healthy mix of "
            "demanding and lighter conversations."
        )

    # Limit to 3 most actionable
    return insights[:4]


def analyse_conversation_history(
    days: int = 14,
    end_date_str: Optional[str] = None,
) -> ConversationIntelligence:
    """
    Analyse Omi conversation history for the last N days.

    Args:
        days: number of days to look back (default: 14)
        end_date_str: end date (default: today). Format: "YYYY-MM-DD"

    Returns:
        ConversationIntelligence dataclass
    """
    if end_date_str:
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    else:
        end_date = datetime.now().date()

    start_date = end_date - timedelta(days=days - 1)

    # Collect daily summaries
    daily_summaries: list[DailyConversationSummary] = []
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        summary = analyse_day(date_str)
        daily_summaries.append(summary)
        current += timedelta(days=1)

    # Filter to days with data
    days_with_data = [s for s in daily_summaries if s.is_meaningful]

    if not days_with_data:
        return ConversationIntelligence(
            date_range=f"{start_date} → {end_date}",
            days_requested=days,
            days_with_data=0,
            total_speech_minutes=0.0,
            total_words=0,
            avg_speech_minutes_per_day=0.0,
            avg_words_per_day=0.0,
            avg_cognitive_density=0.0,
            peak_conversation_hour=9,
            heavy_days=[],
            light_days=[],
            language_split={},
            dominant_language="unknown",
            topic_distribution={},
            dominant_topic="unknown",
            hourly_profile={},
            daily_summaries=daily_summaries,
            is_meaningful=False,
            trend_direction="stable",
            trend_description="Insufficient data for trend analysis.",
            insight_lines=[],
        )

    # Aggregate across all days with data
    total_speech = sum(s.speech_minutes for s in days_with_data)
    total_words = sum(s.total_words for s in days_with_data)
    n = len(days_with_data)

    # Hourly profile (aggregate)
    hourly_profile: dict[int, int] = {}
    for s in days_with_data:
        for h, w in s.words_by_hour.items():
            hourly_profile[h] = hourly_profile.get(h, 0) + w

    # Peak conversation hour (working hours preferred)
    working_hours_profile = {h: w for h, w in hourly_profile.items()
                              if WORKING_HOURS_START <= h <= WORKING_HOURS_END}
    peak_hour = max(working_hours_profile, key=working_hours_profile.get) \
        if working_hours_profile else \
        (max(hourly_profile, key=hourly_profile.get) if hourly_profile else 9)

    # Language split (session-level)
    lang_split: dict[str, int] = {}
    for s in days_with_data:
        for lang, count in s.language_counts.items():
            lang_split[lang] = lang_split.get(lang, 0) + count

    dominant_language = "unknown"
    if lang_split:
        total_lang = sum(lang_split.values())
        top_lang = max(lang_split, key=lang_split.get)
        if lang_split[top_lang] / total_lang >= 0.55:
            dominant_language = top_lang
        else:
            dominant_language = "mixed"

    # Topic distribution
    topic_counts: dict[str, int] = {}
    for s in days_with_data:
        for cat, count in s.topic_distribution.items():
            topic_counts[cat] = topic_counts.get(cat, 0) + count

    total_classified = sum(topic_counts.values())
    topic_distribution = {cat: round(count / total_classified, 3)
                          for cat, count in topic_counts.items()} \
        if total_classified > 0 else {}
    dominant_topic = max(topic_counts, key=topic_counts.get) if topic_counts else "unknown"

    # Avg cognitive density
    density_vals = [s.avg_cognitive_density for s in days_with_data if s.avg_cognitive_density > 0]
    avg_density = sum(density_vals) / len(density_vals) if density_vals else 0.0

    # Heavy / light days (by word count percentile)
    word_counts_sorted = sorted(s.total_words for s in days_with_data)
    heavy_threshold = word_counts_sorted[max(0, int(len(word_counts_sorted) * HEAVY_DAY_PERCENTILE))]
    light_threshold = word_counts_sorted[min(len(word_counts_sorted) - 1,
                                             int(len(word_counts_sorted) * LIGHT_DAY_PERCENTILE))]

    heavy_days = [s.date for s in days_with_data if s.total_words >= heavy_threshold
                  and heavy_threshold > 0]
    light_days = [s.date for s in days_with_data if s.total_words <= light_threshold
                  and light_threshold >= 0 and s.total_words > 0]

    # Trend direction (word count over time)
    word_count_series = [s.total_words for s in days_with_data]
    trend_direction = _compute_trend(word_count_series)

    trend_descriptions = {
        "increasing": f"Conversation volume trending up over the last {n} days — social/verbal load rising.",
        "decreasing": f"Conversation volume trending down over the last {n} days — more quiet focus time.",
        "stable": f"Conversation volume stable over the last {n} days — consistent verbal pattern.",
    }
    trend_description = trend_descriptions[trend_direction]

    # Build CI object (data dict for insight generation)
    ci_data = {
        "avg_speech_minutes_per_day": total_speech / n if n > 0 else 0.0,
        "peak_conversation_hour": peak_hour,
        "dominant_language": dominant_language,
        "dominant_topic": dominant_topic,
        "trend_direction": trend_direction,
        "avg_cognitive_density": avg_density,
        "heavy_days": heavy_days,
        "light_days": light_days,
        "lang_split": lang_split,
        "days_with_data": n,
        "language_split": lang_split,
    }
    insight_lines = _generate_insights(ci_data, days_with_data)

    return ConversationIntelligence(
        date_range=f"{start_date} → {end_date}",
        days_requested=days,
        days_with_data=n,
        total_speech_minutes=total_speech,
        total_words=total_words,
        avg_speech_minutes_per_day=total_speech / n if n > 0 else 0.0,
        avg_words_per_day=total_words / n if n > 0 else 0.0,
        avg_cognitive_density=avg_density,
        peak_conversation_hour=peak_hour,
        heavy_days=heavy_days,
        light_days=light_days,
        language_split=lang_split,
        dominant_language=dominant_language,
        topic_distribution=topic_distribution,
        dominant_topic=dominant_topic,
        hourly_profile=hourly_profile,
        daily_summaries=daily_summaries,
        is_meaningful=n >= MIN_DAYS_FOR_MEANINGFUL,
        trend_direction=trend_direction,
        trend_description=trend_description,
        insight_lines=insight_lines,
    )


# ─── Formatting functions ─────────────────────────────────────────────────────

def _hourly_sparkline(hourly_profile: dict[int, int],
                       start: int = 8, end: int = 20,
                       width: int = 12) -> str:
    """
    Render an ASCII sparkline of conversation activity across the working day.
    Width = number of 1-hour buckets displayed.
    """
    blocks = "░▒▓█"
    hours = list(range(start, end + 1))
    vals = [hourly_profile.get(h, 0) for h in hours]
    max_val = max(vals) if vals else 1

    chars = []
    for v in vals:
        if max_val == 0:
            chars.append("·")
        else:
            idx = min(3, int(v / max_val * 4))
            chars.append(blocks[idx] if v > 0 else "·")

    return "".join(chars)


def _fmt_minutes(minutes: float) -> str:
    """Format minutes as '2h 15m' or '45m'."""
    m = int(round(minutes))
    if m >= 60:
        h, rem = divmod(m, 60)
        return f"{h}h {rem}m" if rem else f"{h}h"
    return f"{m}m"


def _lang_label(lang: str) -> str:
    return {"en": "English", "hu": "Hungarian", "mixed": "Mixed", "unknown": "Unknown"}.get(lang, lang)


def _topic_label(topic: str) -> str:
    return {
        "work_technical": "Technical",
        "work_strategic": "Strategic",
        "personal": "Personal",
        "operational": "Operational",
        "mixed": "Mixed",
        "unknown": "Uncategorised",
    }.get(topic, topic)


def format_conversation_brief_line(ci: ConversationIntelligence) -> str:
    """
    One-line summary for morning brief or midday check-in.
    Example: "🗣 Conversation: 127 min/day · peak 9am · English · trending up"
    """
    if not ci.is_meaningful:
        return ""

    avg = _fmt_minutes(ci.avg_speech_minutes_per_day)
    peak = f"{ci.peak_conversation_hour}:00"
    lang = _lang_label(ci.dominant_language)
    trend = {"increasing": "↑", "decreasing": "↓", "stable": "→"}.get(ci.trend_direction, "→")

    return (
        f"🗣 Conversation ({ci.days_with_data}d): {avg}/day · peak {peak} · "
        f"{lang} · {trend}"
    )


def format_conversation_intelligence_section(ci: ConversationIntelligence) -> str:
    """
    Full Slack-formatted section for the weekly summary.
    Returns empty string if not meaningful.
    """
    if not ci.is_meaningful:
        return ""

    lines = [
        "─" * 40,
        f"*🗣 Conversation Intelligence* ({ci.date_range})",
        "",
    ]

    # Overview stats
    avg = _fmt_minutes(ci.avg_speech_minutes_per_day)
    total = _fmt_minutes(ci.total_speech_minutes)
    lines.append(
        f"  Speech load:   {avg}/day  ·  {total} total across {ci.days_with_data} days"
    )
    lines.append(
        f"  Avg words/day: {ci.avg_words_per_day:,.0f}  ·  {ci.total_words:,} total"
    )

    # Language
    lang_label = _lang_label(ci.dominant_language)
    lang_detail = ""
    if ci.language_split:
        total_sessions = sum(ci.language_split.values())
        parts = []
        for lang in ("en", "hu", "mixed"):
            count = ci.language_split.get(lang, 0)
            if count > 0:
                pct = int(100 * count / total_sessions)
                parts.append(f"{_lang_label(lang)} {pct}%")
        if parts:
            lang_detail = f"  ({', '.join(parts)})"
    lines.append(f"  Language:      {lang_label}{lang_detail}")

    # Topic distribution
    if ci.topic_distribution:
        # Sort by proportion desc, show top 3
        sorted_topics = sorted(ci.topic_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        topic_str = "  ·  ".join(f"{_topic_label(t)} {int(p*100)}%" for t, p in sorted_topics)
        lines.append(f"  Topics:        {topic_str}")

    # Cognitive density
    if ci.avg_cognitive_density > 0:
        density_label = (
            "High" if ci.avg_cognitive_density > 0.55 else
            "Moderate" if ci.avg_cognitive_density > 0.40 else
            "Light"
        )
        lines.append(
            f"  Density:       {ci.avg_cognitive_density:.2f}  ({density_label} cognitive load in conversations)"
        )

    lines.append("")

    # Hourly sparkline
    sparkline = _hourly_sparkline(ci.hourly_profile, start=8, end=20)
    lines.append(f"  Hourly:  8am {'.' * 0}{sparkline} 8pm")
    lines.append(f"           Peak: {ci.peak_conversation_hour}:00")

    lines.append("")

    # Trend
    trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "〰️"}.get(ci.trend_direction, "〰️")
    lines.append(f"  {trend_emoji} {ci.trend_description}")

    # Heavy/light days
    if ci.heavy_days:
        lines.append(f"  🔴 Heavy days: {', '.join(ci.heavy_days[:3])}")
    if ci.light_days:
        lines.append(f"  🟢 Light days: {', '.join(ci.light_days[:3])}")

    lines.append("")

    # Insights
    if ci.insight_lines:
        lines.append("  *Insights:*")
        for line in ci.insight_lines:
            lines.append(f"  • {line}")

    return "\n".join(lines)


def format_conversation_terminal(ci: ConversationIntelligence) -> str:
    """
    Terminal-formatted output with ANSI colour and Unicode sparkline.
    """
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    if not ci.is_meaningful:
        return f"\n{DIM}  Conversation Intelligence: insufficient data (need ≥{MIN_DAYS_FOR_MEANINGFUL} days){RESET}\n"

    lines = [
        "",
        f"  {BOLD}Conversation Intelligence{RESET}  {DIM}{ci.date_range}{RESET}",
        "",
    ]

    avg = _fmt_minutes(ci.avg_speech_minutes_per_day)
    total = _fmt_minutes(ci.total_speech_minutes)

    def _c_load(minutes_per_day: float) -> str:
        if minutes_per_day > 180:
            return RED
        elif minutes_per_day > 90:
            return YELLOW
        return GREEN

    load_colour = _c_load(ci.avg_speech_minutes_per_day)
    lines.append(f"  Speech load    {load_colour}{avg}/day{RESET}  {DIM}({total} total, {ci.days_with_data} days){RESET}")
    lines.append(f"  Words/day      {ci.avg_words_per_day:,.0f}")
    lines.append(f"  Language       {_lang_label(ci.dominant_language)}")

    if ci.avg_cognitive_density > 0:
        dens_col = RED if ci.avg_cognitive_density > 0.55 else YELLOW if ci.avg_cognitive_density > 0.40 else GREEN
        lines.append(f"  Density        {dens_col}{ci.avg_cognitive_density:.2f}{RESET}")

    # Topic breakdown
    if ci.topic_distribution:
        sorted_topics = sorted(ci.topic_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        topic_str = "  ·  ".join(f"{_topic_label(t)} {int(p*100)}%" for t, p in sorted_topics)
        lines.append(f"  Topics         {DIM}{topic_str}{RESET}")

    lines.append("")

    # Hourly sparkline
    sparkline = _hourly_sparkline(ci.hourly_profile, start=8, end=20)
    lines.append(f"  8am  {sparkline}  8pm")
    lines.append(f"  {DIM}Peak conversation: {ci.peak_conversation_hour}:00{RESET}")

    lines.append("")

    # Trend
    trend_icons = {"increasing": "↑", "decreasing": "↓", "stable": "→"}
    trend_icon = trend_icons.get(ci.trend_direction, "→")
    lines.append(f"  Trend  {trend_icon}  {DIM}{ci.trend_description}{RESET}")

    # Insights
    if ci.insight_lines:
        lines.append("")
        for insight in ci.insight_lines:
            lines.append(f"  {DIM}→{RESET} {insight}")

    lines.append("")
    return "\n".join(lines)


def to_dict(ci: ConversationIntelligence) -> dict:
    """Serialize ConversationIntelligence to a JSON-compatible dict."""
    return {
        "date_range": ci.date_range,
        "days_requested": ci.days_requested,
        "days_with_data": ci.days_with_data,
        "total_speech_minutes": round(ci.total_speech_minutes, 1),
        "total_words": ci.total_words,
        "avg_speech_minutes_per_day": round(ci.avg_speech_minutes_per_day, 1),
        "avg_words_per_day": round(ci.avg_words_per_day, 1),
        "avg_cognitive_density": round(ci.avg_cognitive_density, 3),
        "peak_conversation_hour": ci.peak_conversation_hour,
        "heavy_days": ci.heavy_days,
        "light_days": ci.light_days,
        "language_split": ci.language_split,
        "dominant_language": ci.dominant_language,
        "topic_distribution": ci.topic_distribution,
        "dominant_topic": ci.dominant_topic,
        "hourly_profile": {str(k): v for k, v in ci.hourly_profile.items()},
        "is_meaningful": ci.is_meaningful,
        "trend_direction": ci.trend_direction,
        "trend_description": ci.trend_description,
        "insight_lines": ci.insight_lines,
        "daily_summaries": [
            {
                "date": s.date,
                "total_sessions": s.total_sessions,
                "total_words": s.total_words,
                "speech_minutes": round(s.speech_minutes, 1),
                "primary_language": s.primary_language,
                "dominant_topic": s.dominant_topic,
                "avg_cognitive_density": round(s.avg_cognitive_density, 3),
                "peak_hour": s.peak_hour,
                "is_meaningful": s.is_meaningful,
            }
            for s in ci.daily_summaries
        ],
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Presence Tracker — Conversation Intelligence"
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Days to look back (default: 14)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON",
    )
    args = parser.parse_args()

    ci = analyse_conversation_history(days=args.days, end_date_str=args.date)

    if args.json:
        print(json.dumps(to_dict(ci), indent=2))
    else:
        print(format_conversation_terminal(ci))

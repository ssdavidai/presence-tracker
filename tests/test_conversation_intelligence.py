"""
Tests for analysis/conversation_intelligence.py

Tests cover:
  - analyse_day() with mock transcript data
  - analyse_conversation_history() with multiple days
  - Language detection logic
  - Trend computation
  - Formatting functions
  - Graceful degradation on missing data
"""

import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.conversation_intelligence import (
    analyse_day,
    analyse_conversation_history,
    _compute_trend,
    _generate_insights,
    _hourly_sparkline,
    _fmt_minutes,
    format_conversation_brief_line,
    format_conversation_intelligence_section,
    format_conversation_terminal,
    to_dict,
    DailyConversationSummary,
    ConversationIntelligence,
    MIN_DAYS_FOR_MEANINGFUL,
    OMI_TRANSCRIPTS_DIR,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_transcript(
    text: str = "Hello world this is a test transcript about technology",
    timestamp: str = "2026-03-12T09:15:00",
    speech_seconds: float = 30.0,
    audio_seconds: float = 60.0,
    language: str = "en",
) -> dict:
    return {
        "uid": "test-uid",
        "text": text,
        "timestamp": timestamp,
        "speech_duration_seconds": speech_seconds,
        "audio_duration_seconds": audio_seconds,
        "language": language,
    }


def _write_day_transcripts(tmp_dir: Path, date_str: str, transcripts: list[dict]) -> Path:
    day_dir = tmp_dir / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(transcripts):
        fname = f"session_{i:03d}_uid.json"
        (day_dir / fname).write_text(json.dumps(t))
    return day_dir


# ─── analyse_day() tests ──────────────────────────────────────────────────────

class TestAnalyseDay:
    def test_no_data_returns_not_meaningful(self, tmp_path):
        """Days with no Omi data return is_meaningful=False."""
        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = analyse_day("2026-01-01")

        assert result.is_meaningful is False
        assert result.total_sessions == 0
        assert result.total_words == 0
        assert result.speech_minutes == 0.0
        assert result.primary_language == "unknown"

    def test_single_transcript_parsed_correctly(self, tmp_path):
        """Single transcript produces correct aggregates."""
        transcripts = [
            _make_transcript(
                text="This is a technical discussion about distributed systems and machine learning",
                timestamp="2026-03-12T09:30:00",
                speech_seconds=120.0,
                audio_seconds=180.0,
            )
        ]
        _write_day_transcripts(tmp_path, "2026-03-12", transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = analyse_day("2026-03-12")

        assert result.is_meaningful is True
        assert result.total_sessions == 1
        assert result.total_words > 10
        assert abs(result.speech_minutes - 2.0) < 0.1
        assert abs(result.audio_minutes - 3.0) < 0.1
        assert result.speech_ratio > 0

    def test_multiple_transcripts_aggregated(self, tmp_path):
        """Multiple transcripts aggregate correctly."""
        transcripts = [
            _make_transcript(
                text="First conversation about strategy and planning",
                timestamp="2026-03-12T09:00:00",
                speech_seconds=60.0,
                audio_seconds=90.0,
            ),
            _make_transcript(
                text="Second conversation about technical implementation details",
                timestamp="2026-03-12T11:30:00",
                speech_seconds=90.0,
                audio_seconds=120.0,
            ),
            _make_transcript(
                text="Short",  # Below MIN_WORDS_FOR_CLASSIFY
                timestamp="2026-03-12T14:00:00",
                speech_seconds=5.0,
                audio_seconds=10.0,
            ),
        ]
        _write_day_transcripts(tmp_path, "2026-03-12", transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = analyse_day("2026-03-12")

        assert result.total_sessions == 3
        assert result.total_words > 0
        assert result.speech_minutes > 0

    def test_peak_hour_assigned_correctly(self, tmp_path):
        """Peak hour reflects the hour with most words."""
        transcripts = [
            # Hour 9: few words
            _make_transcript(
                text="Hi there",
                timestamp="2026-03-12T09:00:00",
            ),
            # Hour 14: many words
            _make_transcript(
                text=" ".join(["word"] * 300),
                timestamp="2026-03-12T14:00:00",
                speech_seconds=180.0,
            ),
        ]
        _write_day_transcripts(tmp_path, "2026-03-12", transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = analyse_day("2026-03-12")

        assert result.peak_hour == 14

    def test_hungarian_text_detected_as_hu(self, tmp_path):
        """Text with Hungarian diacritics is counted as Hungarian."""
        transcripts = [
            _make_transcript(
                text="Jó reggelt, hogy vagy? Én jól vagyok köszönöm szépen.",
                timestamp="2026-03-12T08:00:00",
                speech_seconds=20.0,
            )
        ]
        _write_day_transcripts(tmp_path, "2026-03-12", transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = analyse_day("2026-03-12")

        # Hungarian should be detected
        assert result.primary_language in ("hu", "mixed", "unknown")
        assert result.language_counts.get("hu", 0) > 0

    def test_corrupt_json_skipped_gracefully(self, tmp_path):
        """Corrupt JSON files are silently skipped."""
        day_dir = tmp_path / "2026-03-12"
        day_dir.mkdir()
        (day_dir / "bad.json").write_text("this is not json {{{")
        (day_dir / "good.json").write_text(json.dumps(
            _make_transcript(text="A valid conversation about work projects")
        ))

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = analyse_day("2026-03-12")

        assert result.total_sessions == 1  # Only the valid file
        assert result.is_meaningful is True

    def test_missing_timestamp_handled(self, tmp_path):
        """Transcripts with missing/invalid timestamps still count toward totals."""
        transcripts = [
            {
                "uid": "test",
                "text": "Hello this is a test conversation",
                "timestamp": "invalid-timestamp",
                "speech_duration_seconds": 30.0,
                "audio_duration_seconds": 60.0,
            }
        ]
        _write_day_transcripts(tmp_path, "2026-03-12", transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = analyse_day("2026-03-12")

        assert result.total_sessions == 1
        assert result.total_words > 0
        assert result.peak_hour is None  # No valid timestamp

    def test_speech_ratio_computed(self, tmp_path):
        """speech_ratio is speech_minutes / audio_minutes."""
        transcripts = [
            _make_transcript(
                text="Testing the speech ratio calculation",
                speech_seconds=60.0,
                audio_seconds=120.0,
            )
        ]
        _write_day_transcripts(tmp_path, "2026-03-12", transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = analyse_day("2026-03-12")

        assert abs(result.speech_ratio - 0.5) < 0.01


# ─── _compute_trend() tests ───────────────────────────────────────────────────

class TestComputeTrend:
    def test_increasing_trend(self):
        """Monotonically increasing series returns 'increasing'."""
        assert _compute_trend([100, 200, 400, 800, 1200]) == "increasing"

    def test_decreasing_trend(self):
        """Monotonically decreasing series returns 'decreasing'."""
        assert _compute_trend([1200, 800, 400, 200, 100]) == "decreasing"

    def test_flat_returns_stable(self):
        """Flat series returns 'stable'."""
        assert _compute_trend([500, 500, 500, 500, 500]) == "stable"

    def test_short_series_returns_stable(self):
        """Series with < 3 points always returns 'stable'."""
        assert _compute_trend([]) == "stable"
        assert _compute_trend([100]) == "stable"
        assert _compute_trend([100, 200]) == "stable"

    def test_noisy_but_increasing(self):
        """Noisy but clearly increasing trend returns 'increasing'."""
        assert _compute_trend([100, 150, 130, 250, 220, 400]) == "increasing"


# ─── analyse_conversation_history() tests ─────────────────────────────────────

class TestAnalyseConversationHistory:
    def test_no_data_returns_not_meaningful(self, tmp_path):
        """No Omi data → is_meaningful=False."""
        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            ci = analyse_conversation_history(days=7)

        assert ci.is_meaningful is False
        assert ci.days_with_data == 0

    def test_one_day_of_data_not_meaningful(self, tmp_path):
        """Single day of data → is_meaningful=False (need MIN_DAYS_FOR_MEANINGFUL)."""
        transcripts = [
            _make_transcript(
                text=" ".join(["word"] * 100),
                timestamp="2026-03-12T09:00:00",
            )
        ]
        _write_day_transcripts(tmp_path, "2026-03-12", transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            ci = analyse_conversation_history(days=14, end_date_str="2026-03-12")

        assert ci.days_with_data == 1
        assert ci.is_meaningful is False  # need >= MIN_DAYS_FOR_MEANINGFUL

    def test_three_days_is_meaningful(self, tmp_path):
        """Three days with data → is_meaningful=True."""
        for day in ["2026-03-10", "2026-03-11", "2026-03-12"]:
            transcripts = [
                _make_transcript(
                    text=" ".join(["test", "conversation"] * 30),
                    timestamp=f"{day}T09:00:00",
                    speech_seconds=120.0,
                )
            ]
            _write_day_transcripts(tmp_path, day, transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            ci = analyse_conversation_history(days=7, end_date_str="2026-03-12")

        assert ci.days_with_data == 3
        assert ci.is_meaningful is True

    def test_aggregates_computed_correctly(self, tmp_path):
        """Totals and averages aggregate correctly across days."""
        # 3 days with known word counts
        day_data = {
            "2026-03-10": 100,
            "2026-03-11": 200,
            "2026-03-12": 300,
        }
        for day, n_words in day_data.items():
            transcripts = [
                _make_transcript(
                    text=" ".join(["word"] * n_words),
                    timestamp=f"{day}T09:00:00",
                    speech_seconds=60.0,
                    audio_seconds=120.0,
                )
            ]
            _write_day_transcripts(tmp_path, day, transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            ci = analyse_conversation_history(days=7, end_date_str="2026-03-12")

        assert ci.total_words == 600
        assert abs(ci.avg_words_per_day - 200.0) < 1.0  # 600 / 3
        assert ci.days_with_data == 3

    def test_peak_hour_determined_from_aggregate(self, tmp_path):
        """Peak conversation hour reflects aggregate across all days."""
        for day in ["2026-03-10", "2026-03-11", "2026-03-12"]:
            # All conversations at 15:00
            transcripts = [
                _make_transcript(
                    text=" ".join(["discussion"] * 200),
                    timestamp=f"{day}T15:00:00",
                    speech_seconds=180.0,
                ),
            ]
            _write_day_transcripts(tmp_path, day, transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            ci = analyse_conversation_history(days=7, end_date_str="2026-03-12")

        assert ci.peak_conversation_hour == 15

    def test_heavy_and_light_days_classified(self, tmp_path):
        """Heavy and light days are correctly identified by percentile."""
        # Create days with very different word counts
        day_words = {
            "2026-03-08": 1000,  # heavy
            "2026-03-09": 100,   # light
            "2026-03-10": 500,
            "2026-03-11": 600,
            "2026-03-12": 5000,  # heavy
        }
        for day, n_words in day_words.items():
            transcripts = [
                _make_transcript(
                    text=" ".join(["word"] * n_words),
                    timestamp=f"{day}T09:00:00",
                    speech_seconds=60.0,
                )
            ]
            _write_day_transcripts(tmp_path, day, transcripts)

        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            ci = analyse_conversation_history(days=10, end_date_str="2026-03-12")

        # At least one heavy day (5000-word day should be in heavy)
        assert len(ci.heavy_days) >= 1
        assert "2026-03-12" in ci.heavy_days

    def test_date_range_in_output(self, tmp_path):
        """date_range string is correctly set."""
        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            ci = analyse_conversation_history(days=7, end_date_str="2026-03-12")

        # Range should span 7 days ending 2026-03-12
        assert "2026-03-06" in ci.date_range
        assert "2026-03-12" in ci.date_range

    def test_days_requested_matches(self, tmp_path):
        """days_requested reflects the requested window."""
        with patch("analysis.conversation_intelligence.OMI_TRANSCRIPTS_DIR", tmp_path):
            ci = analyse_conversation_history(days=21, end_date_str="2026-03-12")

        assert ci.days_requested == 21


# ─── Formatting tests ─────────────────────────────────────────────────────────

class TestFormatting:
    def _make_ci(self, is_meaningful: bool = True) -> ConversationIntelligence:
        return ConversationIntelligence(
            date_range="2026-03-06 → 2026-03-12",
            days_requested=7,
            days_with_data=5 if is_meaningful else 1,
            total_speech_minutes=300.0,
            total_words=45000,
            avg_speech_minutes_per_day=60.0,
            avg_words_per_day=9000.0,
            avg_cognitive_density=0.42,
            peak_conversation_hour=9,
            heavy_days=["2026-03-10"],
            light_days=["2026-03-06"],
            language_split={"en": 20, "hu": 60, "mixed": 10},
            dominant_language="hu",
            topic_distribution={"work_technical": 0.3, "personal": 0.4, "mixed": 0.3},
            dominant_topic="personal",
            hourly_profile={8: 1000, 9: 5000, 10: 3000, 14: 4000, 16: 2000},
            daily_summaries=[],
            is_meaningful=is_meaningful,
            trend_direction="stable",
            trend_description="Conversation volume stable over the last 5 days.",
            insight_lines=["Test insight 1", "Test insight 2"],
        )

    def test_brief_line_not_meaningful_returns_empty(self):
        """format_conversation_brief_line returns '' when not meaningful."""
        ci = self._make_ci(is_meaningful=False)
        assert format_conversation_brief_line(ci) == ""

    def test_brief_line_contains_key_fields(self):
        """Brief line contains avg speech, peak hour, and language."""
        ci = self._make_ci()
        line = format_conversation_brief_line(ci)

        assert "1h" in line  # 60 min/day
        assert "9:00" in line  # peak hour
        assert "Hungarian" in line  # dominant language

    def test_slack_section_not_meaningful_returns_empty(self):
        """Slack section returns '' when not meaningful."""
        ci = self._make_ci(is_meaningful=False)
        assert format_conversation_intelligence_section(ci) == ""

    def test_slack_section_contains_key_sections(self):
        """Full Slack section contains expected content blocks."""
        ci = self._make_ci()
        section = format_conversation_intelligence_section(ci)

        assert "Conversation Intelligence" in section
        assert "Speech load" in section
        assert "Language" in section
        assert "Hourly" in section
        # Trend info appears as the trend_description text or the trend emoji
        assert any(s in section for s in ["〰️", "📈", "📉", "stable", "Conversation volume"])

    def test_terminal_format_not_meaningful(self):
        """Terminal format shows 'insufficient data' when not meaningful."""
        ci = self._make_ci(is_meaningful=False)
        output = format_conversation_terminal(ci)
        assert "insufficient" in output.lower() or "Insufficient" in output

    def test_terminal_format_contains_data(self):
        """Terminal format shows key metrics when meaningful."""
        ci = self._make_ci()
        output = format_conversation_terminal(ci)
        assert "1h" in output  # speech per day
        assert "9:00" in output  # peak hour

    def test_to_dict_serializable(self):
        """to_dict() produces a JSON-serializable dict."""
        ci = self._make_ci()
        d = to_dict(ci)
        # Should not raise
        json_str = json.dumps(d)
        # Should round-trip correctly
        parsed = json.loads(json_str)
        assert parsed["days_requested"] == 7
        assert parsed["peak_conversation_hour"] == 9
        assert parsed["dominant_language"] == "hu"

    def test_to_dict_has_required_keys(self):
        """to_dict() includes all required keys."""
        ci = self._make_ci()
        d = to_dict(ci)
        required_keys = [
            "date_range", "days_requested", "days_with_data",
            "total_speech_minutes", "total_words", "avg_speech_minutes_per_day",
            "avg_words_per_day", "avg_cognitive_density", "peak_conversation_hour",
            "heavy_days", "light_days", "language_split", "dominant_language",
            "topic_distribution", "dominant_topic", "hourly_profile",
            "is_meaningful", "trend_direction", "trend_description",
            "insight_lines", "daily_summaries",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"


# ─── Sparkline tests ──────────────────────────────────────────────────────────

class TestSparkline:
    def test_empty_profile_returns_dots(self):
        """Empty hourly profile returns all dots."""
        result = _hourly_sparkline({}, start=8, end=12, width=5)
        assert "·" in result
        assert "█" not in result

    def test_single_peak_shows_full_block(self):
        """Hour with all the words shows full block."""
        result = _hourly_sparkline({10: 1000}, start=8, end=12)
        assert "█" in result

    def test_sparkline_length(self):
        """Sparkline has correct number of characters for the hour range."""
        result = _hourly_sparkline({}, start=8, end=20)
        assert len(result) == (20 - 8 + 1)

    def test_relative_scaling(self):
        """Blocks are relative to the max value."""
        profile = {8: 100, 9: 500, 10: 1000}
        result = _hourly_sparkline(profile, start=8, end=10)
        # 10:00 should have the highest block
        assert result[2] == "█"


# ─── _fmt_minutes() tests ─────────────────────────────────────────────────────

class TestFmtMinutes:
    def test_under_60(self):
        assert _fmt_minutes(45) == "45m"
        assert _fmt_minutes(1) == "1m"

    def test_exactly_60(self):
        assert _fmt_minutes(60) == "1h"

    def test_over_60(self):
        assert _fmt_minutes(90) == "1h 30m"
        assert _fmt_minutes(125) == "2h 5m"

    def test_zero(self):
        assert _fmt_minutes(0) == "0m"


# ─── _generate_insights() tests ───────────────────────────────────────────────

class TestGenerateInsights:
    def _make_ci_data(self, **kwargs) -> dict:
        defaults = {
            "avg_speech_minutes_per_day": 60.0,
            "peak_conversation_hour": 9,
            "dominant_language": "en",
            "dominant_topic": "work_technical",
            "trend_direction": "stable",
            "avg_cognitive_density": 0.42,
            "heavy_days": [],
            "light_days": [],
            "language_split": {"en": 30, "hu": 10},
            "days_with_data": 7,
        }
        defaults.update(kwargs)
        return defaults

    def test_heavy_load_triggers_insight(self):
        """Very high speech load triggers load insight."""
        ci_data = self._make_ci_data(avg_speech_minutes_per_day=250)
        insights = _generate_insights(ci_data, [])
        assert any("Heavy" in i or "heavy" in i for i in insights)

    def test_peak_hour_always_in_insights(self):
        """Peak conversation hour always generates an insight."""
        ci_data = self._make_ci_data(peak_conversation_hour=15)
        insights = _generate_insights(ci_data, [])
        assert any("15:00" in i for i in insights)

    def test_english_dominant_triggers_insight(self):
        """English-dominant conversation mentions higher load."""
        ci_data = self._make_ci_data(
            dominant_language="en",
            language_split={"en": 80, "hu": 5},
        )
        insights = _generate_insights(ci_data, [])
        assert any("English" in i or "english" in i.lower() for i in insights)

    def test_increasing_trend_triggers_insight(self):
        """Increasing trend triggers an increase insight."""
        ci_data = self._make_ci_data(trend_direction="increasing")
        insights = _generate_insights(ci_data, [])
        assert any("↑" in i or "up" in i or "trending" in i.lower() for i in insights)

    def test_max_four_insights(self):
        """Never returns more than 4 insights."""
        ci_data = self._make_ci_data(
            avg_speech_minutes_per_day=250,
            trend_direction="increasing",
            dominant_language="en",
            avg_cognitive_density=0.65,
        )
        insights = _generate_insights(ci_data, [])
        assert len(insights) <= 4

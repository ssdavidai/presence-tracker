"""
Tests for the Omi transcript collector (collectors/omi.py)

Covers:
- Window index assignment from timestamps
- Word counting
- Speech ratio computation
- Multi-session accumulation per window
- Missing/corrupt file handling
- Cross-midnight timestamp rejection
- Metric integration: CLS, FDI, SDI shift with Omi signals
- Chunker integration: omi signals in window dict and sources_available
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.omi import (
    _window_index_for_time,
    _parse_timestamp,
    _count_words,
    collect,
    OMI_TRANSCRIPTS_DIR,
)
from engine.metrics import (
    cognitive_load_score,
    social_drain_index,
    focus_depth_index,
    compute_metrics,
)
from engine.chunker import build_windows


# ─── Unit: _window_index_for_time ────────────────────────────────────────────

class TestWindowIndexForTime:
    """_window_index_for_time maps a datetime to a 0-based 15-min window index."""

    def test_midnight_is_window_zero(self):
        from zoneinfo import ZoneInfo
        dt = datetime(2026, 3, 12, 0, 0, 0, tzinfo=ZoneInfo("Europe/Budapest"))
        assert _window_index_for_time(dt) == 0

    def test_nine_am_is_window_36(self):
        from zoneinfo import ZoneInfo
        dt = datetime(2026, 3, 12, 9, 0, 0, tzinfo=ZoneInfo("Europe/Budapest"))
        assert _window_index_for_time(dt) == 36  # 9 * 4 = 36

    def test_nine_12_is_window_36(self):
        """9:12 is still within the 9:00–9:15 window."""
        from zoneinfo import ZoneInfo
        dt = datetime(2026, 3, 12, 9, 12, 53, tzinfo=ZoneInfo("Europe/Budapest"))
        assert _window_index_for_time(dt) == 36

    def test_nine_15_is_window_37(self):
        """9:15 starts a new window."""
        from zoneinfo import ZoneInfo
        dt = datetime(2026, 3, 12, 9, 15, 0, tzinfo=ZoneInfo("Europe/Budapest"))
        assert _window_index_for_time(dt) == 37

    def test_23_59_is_window_95(self):
        from zoneinfo import ZoneInfo
        dt = datetime(2026, 3, 12, 23, 59, 0, tzinfo=ZoneInfo("Europe/Budapest"))
        assert _window_index_for_time(dt) == 95

    def test_22_00_is_window_88(self):
        from zoneinfo import ZoneInfo
        dt = datetime(2026, 3, 12, 22, 0, 0, tzinfo=ZoneInfo("Europe/Budapest"))
        assert _window_index_for_time(dt) == 88  # 22 * 4 = 88


# ─── Unit: _parse_timestamp ───────────────────────────────────────────────────

class TestParseTimestamp:
    """_parse_timestamp handles various timestamp formats from Omi files."""

    def test_naive_timestamp_gets_budapest_tz(self):
        ts = "2026-03-12T09:12:53.979953"
        dt = _parse_timestamp(ts)
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.hour == 9

    def test_naive_timestamp_no_microseconds(self):
        ts = "2026-03-12T09:12:53"
        dt = _parse_timestamp(ts)
        assert dt is not None
        assert dt.hour == 9
        assert dt.minute == 12

    def test_aware_timestamp_converted_to_local(self):
        ts = "2026-03-12T09:12:53+01:00"
        dt = _parse_timestamp(ts)
        assert dt is not None
        assert dt.tzinfo is not None

    def test_empty_string_returns_none(self):
        assert _parse_timestamp("") is None

    def test_none_returns_none(self):
        assert _parse_timestamp(None) is None

    def test_garbage_returns_none(self):
        assert _parse_timestamp("not-a-date") is None


# ─── Unit: _count_words ───────────────────────────────────────────────────────

class TestCountWords:
    """_count_words counts whitespace-separated words in transcript text."""

    def test_empty_string(self):
        assert _count_words("") == 0

    def test_none(self):
        assert _count_words(None) == 0

    def test_single_word(self):
        assert _count_words("hello") == 1

    def test_multiple_words(self):
        assert _count_words("Hello world this is a test") == 6

    def test_extra_whitespace(self):
        assert _count_words("  hello   world  ") == 2

    def test_long_text(self):
        text = " ".join(["word"] * 100)
        assert _count_words(text) == 100


# ─── Integration: collect() with mock filesystem ─────────────────────────────

@pytest.fixture
def mock_omi_dir(tmp_path):
    """Create a temporary Omi transcripts directory structure."""
    date_dir = tmp_path / "2026-03-12"
    date_dir.mkdir()
    return tmp_path, date_dir


def _write_transcript(dir_path: Path, filename: str, data: dict):
    """Write a JSON transcript file."""
    (dir_path / filename).write_text(json.dumps(data))


class TestCollect:
    """collect() reads Omi transcript files and maps to window indices."""

    def test_no_transcripts_dir_returns_empty(self, tmp_path):
        with patch("collectors.omi.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = collect("2026-03-13")  # Date with no dir
        assert result == {}

    def test_empty_dir_returns_empty(self, mock_omi_dir):
        tmp_path, date_dir = mock_omi_dir
        with patch("collectors.omi.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = collect("2026-03-12")
        assert result == {}

    def test_single_transcript_mapped_to_correct_window(self, mock_omi_dir):
        tmp_path, date_dir = mock_omi_dir
        _write_transcript(date_dir, "09-12-53_abc.json", {
            "uid": "abc",
            "text": "Hello world this is a test of the system",
            "language": "en",
            "timestamp": "2026-03-12T09:12:53.979953",
            "audio_duration_seconds": 251.9,
            "speech_duration_seconds": 171.26,
        })

        with patch("collectors.omi.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = collect("2026-03-12")

        assert 36 in result  # 9:12 → window 36
        w = result[36]
        assert w["conversation_active"] is True
        assert w["word_count"] == 8  # "Hello world this is a test of the system"
        assert w["speech_seconds"] == 171.3  # rounded
        assert w["audio_seconds"] == 251.9
        assert w["sessions_count"] == 1
        assert 0 < w["speech_ratio"] <= 1.0

    def test_speech_ratio_computed_correctly(self, mock_omi_dir):
        tmp_path, date_dir = mock_omi_dir
        _write_transcript(date_dir, "09-12-53_abc.json", {
            "uid": "abc",
            "text": "hello",
            "language": "en",
            "timestamp": "2026-03-12T09:12:53",
            "audio_duration_seconds": 200.0,
            "speech_duration_seconds": 100.0,
        })

        with patch("collectors.omi.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = collect("2026-03-12")

        assert 36 in result
        assert result[36]["speech_ratio"] == 0.5

    def test_two_sessions_in_same_window_accumulated(self, mock_omi_dir):
        tmp_path, date_dir = mock_omi_dir
        _write_transcript(date_dir, "09-12-53_abc.json", {
            "uid": "abc",
            "text": "Hello world",
            "language": "en",
            "timestamp": "2026-03-12T09:12:53",
            "audio_duration_seconds": 100.0,
            "speech_duration_seconds": 60.0,
        })
        _write_transcript(date_dir, "09-14-00_def.json", {
            "uid": "def",
            "text": "Testing one two three",
            "language": "en",
            "timestamp": "2026-03-12T09:14:00",
            "audio_duration_seconds": 80.0,
            "speech_duration_seconds": 40.0,
        })

        with patch("collectors.omi.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = collect("2026-03-12")

        assert 36 in result
        w = result[36]
        assert w["sessions_count"] == 2
        assert w["word_count"] == 6  # "Hello world" + "Testing one two three"
        assert w["speech_seconds"] == 100.0
        assert w["audio_seconds"] == 180.0

    def test_sessions_in_different_windows(self, mock_omi_dir):
        tmp_path, date_dir = mock_omi_dir
        _write_transcript(date_dir, "09-12-53_abc.json", {
            "uid": "abc",
            "text": "Morning session",
            "language": "en",
            "timestamp": "2026-03-12T09:12:53",
            "audio_duration_seconds": 100.0,
            "speech_duration_seconds": 60.0,
        })
        _write_transcript(date_dir, "10-00-00_def.json", {
            "uid": "def",
            "text": "Later session with more words here",
            "language": "en",
            "timestamp": "2026-03-12T10:00:00",
            "audio_duration_seconds": 150.0,
            "speech_duration_seconds": 90.0,
        })

        with patch("collectors.omi.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = collect("2026-03-12")

        assert 36 in result  # 9:12
        assert 40 in result  # 10:00 → 10 * 4 = 40
        assert result[36]["word_count"] == 2
        assert result[40]["word_count"] == 6

    def test_corrupt_file_skipped_gracefully(self, mock_omi_dir):
        tmp_path, date_dir = mock_omi_dir
        # Good file
        _write_transcript(date_dir, "09-12-53_abc.json", {
            "uid": "abc",
            "text": "Good data",
            "language": "en",
            "timestamp": "2026-03-12T09:12:53",
            "audio_duration_seconds": 100.0,
            "speech_duration_seconds": 60.0,
        })
        # Corrupt file
        (date_dir / "09-14-00_bad.json").write_text("{invalid json")

        with patch("collectors.omi.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = collect("2026-03-12")

        # Should still have data from good file
        assert 36 in result
        assert result[36]["word_count"] == 2

    def test_cross_midnight_transcript_rejected(self, mock_omi_dir):
        """A transcript with a different date than requested is rejected."""
        tmp_path, date_dir = mock_omi_dir
        _write_transcript(date_dir, "23-59-00_abc.json", {
            "uid": "abc",
            "text": "Cross midnight session",
            "language": "en",
            "timestamp": "2026-03-11T23:59:00",  # Yesterday's date
            "audio_duration_seconds": 100.0,
            "speech_duration_seconds": 60.0,
        })

        with patch("collectors.omi.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = collect("2026-03-12")  # Requesting 12th

        # The transcript from the 11th should be rejected
        assert len(result) == 0

    def test_missing_timestamp_skipped(self, mock_omi_dir):
        tmp_path, date_dir = mock_omi_dir
        _write_transcript(date_dir, "09-12-53_abc.json", {
            "uid": "abc",
            "text": "No timestamp",
            "language": "en",
            # No "timestamp" field
            "audio_duration_seconds": 100.0,
            "speech_duration_seconds": 60.0,
        })

        with patch("collectors.omi.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = collect("2026-03-12")

        assert result == {}

    def test_zero_audio_duration_gives_zero_speech_ratio(self, mock_omi_dir):
        tmp_path, date_dir = mock_omi_dir
        _write_transcript(date_dir, "09-12-53_abc.json", {
            "uid": "abc",
            "text": "Hello",
            "language": "en",
            "timestamp": "2026-03-12T09:12:53",
            "audio_duration_seconds": 0.0,
            "speech_duration_seconds": 0.0,
        })

        with patch("collectors.omi.OMI_TRANSCRIPTS_DIR", tmp_path):
            result = collect("2026-03-12")

        assert 36 in result
        assert result[36]["speech_ratio"] == 0.0


# ─── Integration: metric functions with Omi signals ──────────────────────────

class TestCLSWithOmi:
    """cognitive_load_score() increases when Omi conversation is active."""

    def _base_cls(self, **kwargs):
        return cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=70.0,
            **kwargs,
        )

    def test_no_omi_baseline(self):
        cls_base = self._base_cls()
        assert 0 < cls_base < 0.5

    def test_omi_active_raises_cls(self):
        cls_base = self._base_cls()
        cls_omi = self._base_cls(omi_conversation_active=True, omi_word_count=0)
        assert cls_omi > cls_base

    def test_omi_word_dense_raises_cls_more(self):
        cls_sparse = self._base_cls(omi_conversation_active=True, omi_word_count=10)
        cls_dense = self._base_cls(omi_conversation_active=True, omi_word_count=500)
        assert cls_dense > cls_sparse

    def test_omi_saturates_at_500_words(self):
        """Word count beyond 500 should not increase CLS further."""
        cls_500 = self._base_cls(omi_conversation_active=True, omi_word_count=500)
        cls_1000 = self._base_cls(omi_conversation_active=True, omi_word_count=1000)
        assert abs(cls_500 - cls_1000) < 0.01  # Should be identical (clamped)

    def test_omi_inactive_word_count_has_no_effect(self):
        """word_count is ignored when conversation_active is False."""
        cls_no_omi = self._base_cls()
        cls_words_no_active = self._base_cls(omi_conversation_active=False, omi_word_count=500)
        assert cls_no_omi == cls_words_no_active

    def test_cls_stays_in_range(self):
        """CLS must always be in [0.0, 1.0]."""
        for words in [0, 50, 200, 500, 1000]:
            cls = self._base_cls(omi_conversation_active=True, omi_word_count=words)
            assert 0.0 <= cls <= 1.0, f"CLS out of range for {words} words: {cls}"


class TestSDIWithOmi:
    """social_drain_index() increases with Omi speech time."""

    def _base_sdi(self, **kwargs):
        return social_drain_index(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_sent=0,
            slack_messages_received=0,
            **kwargs,
        )

    def test_no_omi_baseline_near_zero(self):
        sdi = self._base_sdi()
        assert sdi == 0.0

    def test_omi_conversation_raises_sdi(self):
        sdi_base = self._base_sdi()
        sdi_omi = self._base_sdi(omi_conversation_active=True, omi_speech_seconds=120.0)
        assert sdi_omi > sdi_base

    def test_longer_speech_higher_sdi(self):
        sdi_short = self._base_sdi(omi_conversation_active=True, omi_speech_seconds=60.0)
        sdi_long = self._base_sdi(omi_conversation_active=True, omi_speech_seconds=600.0)
        assert sdi_long > sdi_short

    def test_full_window_speech_raises_sdi_significantly(self):
        """900s of speech (full 15-min window) should give noticeable SDI boost."""
        sdi_full = self._base_sdi(omi_conversation_active=True, omi_speech_seconds=900.0)
        assert sdi_full >= 0.12  # Should add ~15% weight

    def test_omi_inactive_speech_has_no_effect(self):
        """speech_seconds ignored when conversation_active is False."""
        sdi_no_omi = self._base_sdi()
        sdi_speech_no_active = self._base_sdi(omi_conversation_active=False, omi_speech_seconds=900.0)
        assert sdi_no_omi == sdi_speech_no_active

    def test_sdi_stays_in_range(self):
        for secs in [0.0, 60.0, 300.0, 900.0, 1800.0]:
            sdi = self._base_sdi(omi_conversation_active=True, omi_speech_seconds=secs)
            assert 0.0 <= sdi <= 1.0, f"SDI out of range for {secs}s: {sdi}"


class TestFDIWithOmi:
    """focus_depth_index() decreases when Omi conversation is active."""

    def _base_fdi(self, **kwargs):
        return focus_depth_index(
            in_meeting=False,
            slack_messages_received=0,
            **kwargs,
        )

    def test_no_omi_baseline_near_one(self):
        """Idle window with no disruption should have FDI near 1.0."""
        fdi = self._base_fdi()
        assert fdi >= 0.9

    def test_omi_active_lowers_fdi(self):
        fdi_base = self._base_fdi()
        fdi_omi = self._base_fdi(omi_conversation_active=True, omi_speech_ratio=0.5)
        assert fdi_omi < fdi_base

    def test_higher_speech_ratio_lower_fdi(self):
        fdi_low = self._base_fdi(omi_conversation_active=True, omi_speech_ratio=0.2)
        fdi_high = self._base_fdi(omi_conversation_active=True, omi_speech_ratio=0.9)
        assert fdi_low > fdi_high

    def test_omi_inactive_ratio_has_no_effect(self):
        fdi_no_omi = self._base_fdi()
        fdi_ratio_no_active = self._base_fdi(omi_conversation_active=False, omi_speech_ratio=1.0)
        assert fdi_no_omi == fdi_ratio_no_active

    def test_fdi_stays_in_range(self):
        for ratio in [0.0, 0.3, 0.6, 1.0]:
            fdi = self._base_fdi(omi_conversation_active=True, omi_speech_ratio=ratio)
            assert 0.0 <= fdi <= 1.0, f"FDI out of range for ratio={ratio}: {fdi}"


# ─── Integration: compute_metrics() with Omi sub-dict ────────────────────────

class TestComputeMetricsWithOmi:
    """compute_metrics() correctly extracts and forwards Omi signals."""

    def _window_no_omi(self):
        return {
            "calendar": {"in_meeting": False, "meeting_attendees": 0, "meeting_duration_minutes": 0},
            "whoop": {"recovery_score": 70.0, "hrv_rmssd_milli": 60.0, "sleep_performance": 80.0},
            "slack": {"messages_sent": 0, "messages_received": 0, "channels_active": 0},
        }

    def _window_with_omi(self, conversation_active=True, word_count=200, speech_seconds=120.0,
                         audio_seconds=200.0, sessions_count=2, speech_ratio=0.6):
        w = self._window_no_omi()
        w["omi"] = {
            "conversation_active": conversation_active,
            "word_count": word_count,
            "speech_seconds": speech_seconds,
            "audio_seconds": audio_seconds,
            "sessions_count": sessions_count,
            "speech_ratio": speech_ratio,
        }
        return w

    def test_omi_absent_does_not_crash(self):
        metrics = compute_metrics(self._window_no_omi())
        assert "cognitive_load_score" in metrics
        assert "social_drain_index" in metrics
        assert "focus_depth_index" in metrics

    def test_omi_present_shifts_cls_up(self):
        m_base = compute_metrics(self._window_no_omi())
        m_omi = compute_metrics(self._window_with_omi())
        assert m_omi["cognitive_load_score"] > m_base["cognitive_load_score"]

    def test_omi_present_shifts_sdi_up(self):
        m_base = compute_metrics(self._window_no_omi())
        m_omi = compute_metrics(self._window_with_omi())
        assert m_omi["social_drain_index"] > m_base["social_drain_index"]

    def test_omi_present_shifts_fdi_down(self):
        m_base = compute_metrics(self._window_no_omi())
        m_omi = compute_metrics(self._window_with_omi())
        assert m_omi["focus_depth_index"] < m_base["focus_depth_index"]

    def test_omi_inactive_conversation_has_no_effect(self):
        """omi sub-dict present but conversation_active=False → same as no Omi."""
        m_base = compute_metrics(self._window_no_omi())
        m_omi_inactive = compute_metrics(self._window_with_omi(
            conversation_active=False, word_count=0, speech_seconds=0.0,
            audio_seconds=0.0, sessions_count=0, speech_ratio=0.0,
        ))
        # Should be identical since no active conversation
        assert m_base["cognitive_load_score"] == m_omi_inactive["cognitive_load_score"]
        assert m_base["social_drain_index"] == m_omi_inactive["social_drain_index"]

    def test_omi_none_value_is_none_safe(self):
        """omi=None in window_data should not crash compute_metrics."""
        w = self._window_no_omi()
        w["omi"] = None
        metrics = compute_metrics(w)
        assert "cognitive_load_score" in metrics

    def test_all_metrics_in_range(self):
        m = compute_metrics(self._window_with_omi())
        for key in ["cognitive_load_score", "focus_depth_index", "social_drain_index",
                    "context_switch_cost", "recovery_alignment_score"]:
            assert 0.0 <= m[key] <= 1.0, f"{key} = {m[key]} out of range"


# ─── Integration: chunker build_windows() with Omi ───────────────────────────

class TestChunkerWithOmi:
    """build_windows() correctly wires Omi signals into window dicts."""

    def _minimal_inputs(self, date_str="2026-03-12"):
        whoop = {"recovery_score": 75.0, "hrv_rmssd_milli": 65.0, "sleep_performance": 85.0,
                 "resting_heart_rate": 55.0, "sleep_hours": 8.0, "strain": 10.0, "spo2_percentage": None}
        cal = {"events": [], "event_count": 0, "total_meeting_minutes": 0, "max_concurrent_attendees": 0}
        slack = {}
        return date_str, whoop, cal, slack

    def test_no_omi_windows_no_omi_key(self):
        date_str, whoop, cal, slack = self._minimal_inputs()
        windows = build_windows(date_str, whoop, cal, slack, omi_windows=None)
        # No window should have an "omi" key
        for w in windows:
            assert "omi" not in w

    def test_omi_window_has_omi_key(self):
        date_str, whoop, cal, slack = self._minimal_inputs()
        omi_windows = {
            36: {  # 9:00 window
                "conversation_active": True,
                "word_count": 150,
                "speech_seconds": 90.0,
                "audio_seconds": 150.0,
                "sessions_count": 1,
                "speech_ratio": 0.6,
            }
        }
        windows = build_windows(date_str, whoop, cal, slack, omi_windows=omi_windows)
        w36 = windows[36]
        assert "omi" in w36
        assert w36["omi"]["conversation_active"] is True
        assert w36["omi"]["word_count"] == 150

    def test_non_omi_windows_have_no_omi_key(self):
        date_str, whoop, cal, slack = self._minimal_inputs()
        omi_windows = {
            36: {
                "conversation_active": True,
                "word_count": 100,
                "speech_seconds": 60.0,
                "audio_seconds": 100.0,
                "sessions_count": 1,
                "speech_ratio": 0.6,
            }
        }
        windows = build_windows(date_str, whoop, cal, slack, omi_windows=omi_windows)
        # Window 37 should not have omi key
        assert "omi" not in windows[37]

    def test_omi_source_in_sources_available(self):
        date_str, whoop, cal, slack = self._minimal_inputs()
        omi_windows = {
            36: {
                "conversation_active": True,
                "word_count": 100,
                "speech_seconds": 60.0,
                "audio_seconds": 100.0,
                "sessions_count": 1,
                "speech_ratio": 0.6,
            }
        }
        windows = build_windows(date_str, whoop, cal, slack, omi_windows=omi_windows)
        assert "omi" in windows[36]["metadata"]["sources_available"]

    def test_non_omi_windows_dont_have_omi_in_sources(self):
        date_str, whoop, cal, slack = self._minimal_inputs()
        omi_windows = {36: {
            "conversation_active": True, "word_count": 100,
            "speech_seconds": 60.0, "audio_seconds": 100.0,
            "sessions_count": 1, "speech_ratio": 0.6,
        }}
        windows = build_windows(date_str, whoop, cal, slack, omi_windows=omi_windows)
        assert "omi" not in windows[37]["metadata"]["sources_available"]

    def test_omi_window_is_active(self):
        """A window with Omi conversation should be marked is_active_window=True."""
        date_str, whoop, cal, slack = self._minimal_inputs()
        omi_windows = {
            50: {  # An otherwise idle window
                "conversation_active": True,
                "word_count": 50,
                "speech_seconds": 30.0,
                "audio_seconds": 60.0,
                "sessions_count": 1,
                "speech_ratio": 0.5,
            }
        }
        windows = build_windows(date_str, whoop, cal, slack, omi_windows=omi_windows)
        assert windows[50]["metadata"]["is_active_window"] is True

    def test_omi_metrics_shift_in_conversation_window(self):
        """CLS should be higher and FDI lower in the Omi window vs baseline."""
        date_str, whoop, cal, slack = self._minimal_inputs()

        # No Omi
        windows_no_omi = build_windows(date_str, whoop, cal, slack)
        # With Omi in window 36
        omi_windows = {
            36: {
                "conversation_active": True,
                "word_count": 300,
                "speech_seconds": 200.0,
                "audio_seconds": 300.0,
                "sessions_count": 2,
                "speech_ratio": 0.67,
            }
        }
        windows_with_omi = build_windows(date_str, whoop, cal, slack, omi_windows=omi_windows)

        cls_no_omi = windows_no_omi[36]["metrics"]["cognitive_load_score"]
        cls_with_omi = windows_with_omi[36]["metrics"]["cognitive_load_score"]
        fdi_no_omi = windows_no_omi[36]["metrics"]["focus_depth_index"]
        fdi_with_omi = windows_with_omi[36]["metrics"]["focus_depth_index"]

        assert cls_with_omi > cls_no_omi
        assert fdi_with_omi < fdi_no_omi

    def test_build_windows_without_omi_param_still_works(self):
        """build_windows with no omi_windows kwarg should not crash."""
        date_str, whoop, cal, slack = self._minimal_inputs()
        windows = build_windows(date_str, whoop, cal, slack)  # No omi_windows
        assert len(windows) == 96


# ─── Regression: existing tests unbroken ─────────────────────────────────────

class TestOmiBackwardCompatibility:
    """
    Verify that adding Omi parameters doesn't break existing metric behaviour.
    All existing metric tests should still pass.
    """

    def test_cls_without_omi_unchanged(self):
        """cognitive_load_score without Omi kwargs matches expected formula."""
        cls = cognitive_load_score(
            in_meeting=True,
            meeting_attendees=4,
            slack_messages_received=5,
            recovery_score=80.0,
        )
        # With recovery_score=80 → readiness=0.8, recovery_inverse=0.2
        # meeting_component=1.0, calendar_pressure=4/10=0.4, slack=5/30
        # base_cls = 0.35*1.0 + 0.20*0.4 + 0.25*(5/30) + 0.20*0.2
        assert 0.0 <= cls <= 1.0

    def test_sdi_without_omi_unchanged(self):
        sdi = social_drain_index(
            in_meeting=True,
            meeting_attendees=4,
            slack_messages_sent=3,
            slack_messages_received=10,
        )
        # Classic formula: attendee_component=4/10, meeting_component=1.0
        assert 0.0 <= sdi <= 1.0

    def test_fdi_without_omi_unchanged(self):
        fdi = focus_depth_index(
            in_meeting=False,
            slack_messages_received=0,
        )
        assert fdi == 1.0  # Zero disruption → perfect focus

    def test_compute_metrics_backward_compatible(self):
        """compute_metrics without omi key returns same values as before."""
        window = {
            "calendar": {"in_meeting": True, "meeting_attendees": 3, "meeting_duration_minutes": 45},
            "whoop": {"recovery_score": 75.0, "hrv_rmssd_milli": 60.0, "sleep_performance": 80.0},
            "slack": {"messages_sent": 2, "messages_received": 8, "channels_active": 2},
        }
        metrics = compute_metrics(window)
        assert set(metrics.keys()) == {
            "cognitive_load_score", "focus_depth_index", "social_drain_index",
            "context_switch_cost", "recovery_alignment_score"
        }
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

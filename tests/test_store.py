"""
Tests for the storage layer.

Run with: python3 -m pytest tests/test_store.py -v
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def make_sample_windows(date_str: str = "2026-01-01", count: int = 96) -> list:
    """Generate minimal valid windows for testing."""
    return [
        {
            "window_id": f"{date_str}T{(i // 4):02d}:{(i % 4) * 15:02d}:00",
            "date": date_str,
            "window_start": f"{date_str}T00:00:00+01:00",
            "window_end": f"{date_str}T00:15:00+01:00",
            "window_index": i,
            "calendar": {"in_meeting": False, "meeting_attendees": 0, "meeting_duration_minutes": 0},
            "whoop": {"recovery_score": 75.0},
            "slack": {"messages_sent": 0, "messages_received": 0, "total_messages": 0, "channels_active": 0},
            "metrics": {
                "cognitive_load_score": 0.1,
                "focus_depth_index": 0.9,
                "social_drain_index": 0.0,
                "context_switch_cost": 0.0,
                "recovery_alignment_score": 0.85,
            },
            "metadata": {
                "day_of_week": "Thursday",
                "hour_of_day": i // 4,
                "minute_of_hour": (i % 4) * 15,
                "is_working_hours": 7 <= (i // 4) < 22,
                "sources_available": ["whoop", "calendar", "slack"],
            },
        }
        for i in range(count)
    ]


class TestWriteAndReadDay:
    def test_write_creates_file(self, tmp_path, monkeypatch):
        # Patch CHUNKS_DIR to use tmp_path
        import config
        monkeypatch.setattr(config, "CHUNKS_DIR", tmp_path)
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        windows = make_sample_windows("2026-01-01")
        path = store.write_day("2026-01-01", windows)
        assert path.exists()

    def test_roundtrip_preserves_data(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        original = make_sample_windows("2026-01-15", count=10)
        store.write_day("2026-01-15", original)
        loaded = store.read_day("2026-01-15")

        assert len(loaded) == len(original)
        assert loaded[0]["window_index"] == 0
        assert loaded[5]["metrics"]["cognitive_load_score"] == 0.1

    def test_read_nonexistent_returns_empty(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        result = store.read_day("2000-01-01")
        assert result == []

    def test_write_96_lines(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        windows = make_sample_windows("2026-02-01")
        path = store.write_day("2026-02-01", windows)
        lines = [l for l in path.read_text().strip().split("\n") if l]
        assert len(lines) == 96

    def test_each_line_is_valid_json(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        windows = make_sample_windows("2026-02-05")
        path = store.write_day("2026-02-05", windows)
        for line in path.read_text().strip().split("\n"):
            if line:
                obj = json.loads(line)  # Should not raise
                assert "window_index" in obj


class TestDayExists:
    def test_exists_after_write(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        assert not store.day_exists("2026-03-01")
        store.write_day("2026-03-01", make_sample_windows("2026-03-01", 5))
        assert store.day_exists("2026-03-01")


class TestSummaryUpdate:
    def test_update_and_read(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        summary = {
            "date": "2026-03-13",
            "working_hours_analyzed": 60,
            "metrics_avg": {"cognitive_load_score": 0.45},
            "whoop": {"recovery_score": 85.0},
            "calendar": {"meeting_windows": 4},
            "slack": {"total_messages_sent": 10},
        }
        store.update_summary(summary)
        rolling = store.read_summary()

        assert "2026-03-13" in rolling["days"]
        assert rolling["total_days"] == 1
        assert rolling["days"]["2026-03-13"]["whoop"]["recovery_score"] == 85.0

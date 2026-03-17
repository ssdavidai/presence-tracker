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


def make_day_summary(date_str: str, recovery: float = 75.0, cls: float = 0.4) -> dict:
    """Generate a minimal day summary dict for testing."""
    return {
        "date": date_str,
        "working_hours_analyzed": 60,
        "metrics_avg": {"cognitive_load_score": cls, "focus_depth_index": 0.6},
        "whoop": {"recovery_score": recovery},
        "calendar": {"meeting_windows": 4, "total_meeting_minutes": 60},
        "slack": {"total_messages_sent": 10},
    }


# ─── Write / Read ─────────────────────────────────────────────────────────────

class TestWriteAndReadDay:
    def test_write_creates_file(self, tmp_path, monkeypatch):
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
                obj = json.loads(line)
                assert "window_index" in obj

    def test_overwrite_replaces_file(self, tmp_path, monkeypatch):
        """Writing a second time for the same date replaces the existing file."""
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        store.write_day("2026-03-01", make_sample_windows("2026-03-01", count=5))
        store.write_day("2026-03-01", make_sample_windows("2026-03-01", count=3))
        loaded = store.read_day("2026-03-01")
        assert len(loaded) == 3

    def test_read_skips_corrupt_lines(self, tmp_path, monkeypatch):
        """Corrupt JSONL lines are silently skipped."""
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        path = tmp_path / "2026-04-01.jsonl"
        path.write_text('{"window_index": 0}\nNOT JSON\n{"window_index": 1}\n')
        loaded = store.read_day("2026-04-01")
        assert len(loaded) == 2
        assert loaded[0]["window_index"] == 0
        assert loaded[1]["window_index"] == 1


# ─── Day Exists ───────────────────────────────────────────────────────────────

class TestDayExists:
    def test_exists_after_write(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        assert not store.day_exists("2026-03-01")
        store.write_day("2026-03-01", make_sample_windows("2026-03-01", 5))
        assert store.day_exists("2026-03-01")

    def test_not_exists_for_missing_date(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        assert not store.day_exists("1999-12-31")


# ─── Read Range ───────────────────────────────────────────────────────────────

class TestReadRange:
    def test_returns_windows_across_multiple_days(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        store.write_day("2026-03-10", make_sample_windows("2026-03-10", count=5))
        store.write_day("2026-03-11", make_sample_windows("2026-03-11", count=4))
        store.write_day("2026-03-12", make_sample_windows("2026-03-12", count=3))

        result = store.read_range("2026-03-10", "2026-03-12")
        assert len(result) == 12  # 5 + 4 + 3

    def test_single_day_range(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        store.write_day("2026-03-15", make_sample_windows("2026-03-15", count=6))
        result = store.read_range("2026-03-15", "2026-03-15")
        assert len(result) == 6

    def test_missing_day_in_range_is_skipped(self, tmp_path, monkeypatch):
        """A missing day in the range produces zero windows for that day."""
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        store.write_day("2026-03-01", make_sample_windows("2026-03-01", count=2))
        # 2026-03-02 intentionally missing
        store.write_day("2026-03-03", make_sample_windows("2026-03-03", count=3))

        result = store.read_range("2026-03-01", "2026-03-03")
        assert len(result) == 5  # 2 + 0 + 3

    def test_range_with_no_data_returns_empty(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        result = store.read_range("2020-01-01", "2020-01-07")
        assert result == []

    def test_ordering_preserved_oldest_first(self, tmp_path, monkeypatch):
        """Windows should appear in ascending date order."""
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        store.write_day("2026-03-05", make_sample_windows("2026-03-05", count=2))
        store.write_day("2026-03-06", make_sample_windows("2026-03-06", count=2))

        result = store.read_range("2026-03-05", "2026-03-06")
        # First two windows are from 2026-03-05
        assert result[0]["date"] == "2026-03-05"
        assert result[2]["date"] == "2026-03-06"


# ─── List Available Dates ─────────────────────────────────────────────────────

class TestListAvailableDates:
    def test_returns_empty_when_no_data(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        assert store.list_available_dates() == []

    def test_returns_sorted_dates(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        for d in ["2026-03-05", "2026-03-01", "2026-03-10"]:
            store.write_day(d, make_sample_windows(d, count=1))

        dates = store.list_available_dates()
        assert dates == ["2026-03-01", "2026-03-05", "2026-03-10"]

    def test_only_jsonl_files_counted(self, tmp_path, monkeypatch):
        """Non-.jsonl files in the chunks dir should be ignored."""
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        store.write_day("2026-03-01", make_sample_windows("2026-03-01", count=1))
        (tmp_path / "README.txt").write_text("not a chunk file")
        (tmp_path / "notes.json").write_text("{}")

        dates = store.list_available_dates()
        assert dates == ["2026-03-01"]

    def test_count_matches_written_days(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        for d in ["2026-01-01", "2026-01-02", "2026-01-03"]:
            store.write_day(d, make_sample_windows(d, count=1))

        assert len(store.list_available_dates()) == 3


# ─── Summary Update / Read ────────────────────────────────────────────────────

class TestSummaryUpdate:
    def test_update_and_read(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        summary = make_day_summary("2026-03-13")
        store.update_summary(summary)
        rolling = store.read_summary()

        assert "2026-03-13" in rolling["days"]
        assert rolling["total_days"] == 1
        assert rolling["days"]["2026-03-13"]["whoop"]["recovery_score"] == 75.0

    def test_update_multiple_days_accumulates(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        for d in ["2026-03-10", "2026-03-11", "2026-03-12"]:
            store.update_summary(make_day_summary(d))

        rolling = store.read_summary()
        assert rolling["total_days"] == 3
        assert "2026-03-10" in rolling["days"]
        assert "2026-03-11" in rolling["days"]
        assert "2026-03-12" in rolling["days"]

    def test_update_same_date_twice_replaces(self, tmp_path, monkeypatch):
        """Updating the same date twice should replace (not duplicate) the entry."""
        import engine.store as store
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        store.update_summary(make_day_summary("2026-03-01", recovery=60.0))
        store.update_summary(make_day_summary("2026-03-01", recovery=90.0))
        rolling = store.read_summary()

        assert rolling["total_days"] == 1
        assert rolling["days"]["2026-03-01"]["whoop"]["recovery_score"] == 90.0

    def test_read_summary_empty_returns_defaults(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)
        rolling = store.read_summary()
        assert rolling["days"] == {}
        assert rolling["total_days"] == 0


# ─── Get Recent Summaries ─────────────────────────────────────────────────────

class TestGetRecentSummaries:
    def test_returns_most_recent_first(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        for d in ["2026-03-10", "2026-03-11", "2026-03-12"]:
            store.update_summary(make_day_summary(d))

        recent = store.get_recent_summaries(days=3)
        assert len(recent) == 3
        assert recent[0]["date"] == "2026-03-12"
        assert recent[1]["date"] == "2026-03-11"
        assert recent[2]["date"] == "2026-03-10"

    def test_limits_to_requested_days(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        for d in ["2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13", "2026-03-14"]:
            store.update_summary(make_day_summary(d))

        recent = store.get_recent_summaries(days=2)
        assert len(recent) == 2
        assert recent[0]["date"] == "2026-03-14"

    def test_returns_all_when_fewer_than_n(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        store.update_summary(make_day_summary("2026-03-01"))
        recent = store.get_recent_summaries(days=7)
        assert len(recent) == 1

    def test_empty_store_returns_empty_list(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)
        assert store.get_recent_summaries(days=7) == []


# ─── Data Age / Date Range ────────────────────────────────────────────────────

class TestGetDataAgeDays:
    def test_zero_with_no_data(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        assert store.get_data_age_days() == 0

    def test_count_matches_written_days(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        for d in ["2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04"]:
            store.write_day(d, make_sample_windows(d, count=1))

        assert store.get_data_age_days() == 4


class TestGetDataStalenessDays:
    """get_data_staleness_days() returns calendar days since most recent ingestion."""

    def test_zero_with_no_data(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        assert store.get_data_staleness_days() == 0

    def test_zero_when_newest_is_today(self, tmp_path, monkeypatch):
        import engine.store as store
        from datetime import date
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        today_str = date.today().isoformat()
        store.write_day(today_str, make_sample_windows(today_str, count=1))
        assert store.get_data_staleness_days() == 0

    def test_correct_days_for_past_date(self, tmp_path, monkeypatch):
        import engine.store as store
        from datetime import date, timedelta
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        # Write data for 3 days ago
        three_days_ago = (date.today() - timedelta(days=3)).isoformat()
        store.write_day(three_days_ago, make_sample_windows(three_days_ago, count=1))
        assert store.get_data_staleness_days() == 3

    def test_uses_newest_date_not_count(self, tmp_path, monkeypatch):
        """Staleness reflects the newest date, not the number of files."""
        import engine.store as store
        from datetime import date, timedelta
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        # 6 days of data, but newest is yesterday
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        for i in range(6):
            d = (date.today() - timedelta(days=6 - i)).isoformat()
            store.write_day(d, make_sample_windows(d, count=1))
        # Newest is yesterday → staleness should be 1, not 6
        assert store.get_data_staleness_days() == 1

    def test_does_not_return_negative(self, tmp_path, monkeypatch):
        """Never returns negative even if newest date is in the future."""
        import engine.store as store
        from datetime import date, timedelta
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        future = (date.today() + timedelta(days=1)).isoformat()
        store.write_day(future, make_sample_windows(future, count=1))
        assert store.get_data_staleness_days() == 0


class TestGetDateRange:
    def test_none_none_with_no_data(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        oldest, newest = store.get_date_range()
        assert oldest is None
        assert newest is None

    def test_same_start_and_end_with_one_day(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        store.write_day("2026-03-01", make_sample_windows("2026-03-01", count=1))
        oldest, newest = store.get_date_range()
        assert oldest == "2026-03-01"
        assert newest == "2026-03-01"

    def test_correct_bounds_with_multiple_days(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        for d in ["2026-03-05", "2026-03-01", "2026-03-10"]:
            store.write_day(d, make_sample_windows(d, count=1))

        oldest, newest = store.get_date_range()
        assert oldest == "2026-03-01"
        assert newest == "2026-03-10"

    def test_range_spans_multiple_months(self, tmp_path, monkeypatch):
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)

        for d in ["2026-01-15", "2026-02-28", "2026-03-14"]:
            store.write_day(d, make_sample_windows(d, count=1))

        oldest, newest = store.get_date_range()
        assert oldest == "2026-01-15"
        assert newest == "2026-03-14"

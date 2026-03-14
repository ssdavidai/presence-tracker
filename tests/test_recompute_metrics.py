"""
Tests for the metric recomputation script.

Run with: python3 -m pytest tests/test_recompute_metrics.py -v
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ─── Test fixtures ────────────────────────────────────────────────────────────

def _make_window(
    index: int,
    date_str: str = "2026-01-01",
    in_meeting: bool = False,
    meeting_attendees: int = 0,
    meeting_duration: int = 0,
    recovery_score: float = 75.0,
    hrv_rmssd_milli: float = 65.0,
    sleep_performance: float = 80.0,
    slack_sent: int = 0,
    slack_received: int = 0,
    channels_active: int = 0,
    rescuetime: dict = None,
    is_active_window=None,
    cls_override: float = 0.1,
    sdi_override: float = 0.0,
    fdi_override: float = 0.9,
    ras_override: float = 0.85,
    csc_override: float = 0.0,
) -> dict:
    """Build a minimal valid window dict for testing."""
    hour = index // 4
    minute = (index % 4) * 15
    window = {
        "window_id": f"{date_str}T{hour:02d}:{minute:02d}:00",
        "date": date_str,
        "window_start": f"{date_str}T{hour:02d}:{minute:02d}:00+01:00",
        "window_end": f"{date_str}T{hour:02d}:{minute:02d}:00+01:00",
        "window_index": index,
        "calendar": {
            "in_meeting": in_meeting,
            "meeting_attendees": meeting_attendees,
            "meeting_duration_minutes": meeting_duration,
        },
        "whoop": {
            "recovery_score": recovery_score,
            "hrv_rmssd_milli": hrv_rmssd_milli,
            "sleep_performance": sleep_performance,
        },
        "slack": {
            "messages_sent": slack_sent,
            "messages_received": slack_received,
            "total_messages": slack_sent + slack_received,
            "channels_active": channels_active,
        },
        "metrics": {
            "cognitive_load_score": cls_override,
            "focus_depth_index": fdi_override,
            "social_drain_index": sdi_override,
            "context_switch_cost": csc_override,
            "recovery_alignment_score": ras_override,
        },
        "metadata": {
            "day_of_week": "Thursday",
            "hour_of_day": hour,
            "minute_of_hour": minute,
            "is_working_hours": 7 <= hour < 22,
            "sources_available": ["whoop", "calendar", "slack"],
        },
    }
    if is_active_window is not None:
        window["metadata"]["is_active_window"] = is_active_window
    if rescuetime is not None:
        window["rescuetime"] = rescuetime
    return window


# ─── Import helpers ───────────────────────────────────────────────────────────

def _import_recompute():
    """Import the recompute_metrics module from the scripts directory."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import recompute_metrics
    return recompute_metrics


# ─── Tests: _is_active_window ─────────────────────────────────────────────────

class TestIsActiveWindow:
    def setup_method(self):
        self.mod = _import_recompute()

    def test_idle_window_is_not_active(self):
        w = _make_window(0, in_meeting=False, slack_sent=0, slack_received=0)
        assert self.mod._is_active_window(w) is False

    def test_in_meeting_is_active(self):
        w = _make_window(36, in_meeting=True, meeting_attendees=2, meeting_duration=60)
        assert self.mod._is_active_window(w) is True

    def test_slack_messages_is_active(self):
        w = _make_window(36, slack_received=3)
        assert self.mod._is_active_window(w) is True

    def test_rescuetime_active_seconds_is_active(self):
        w = _make_window(36, rescuetime={"active_seconds": 300})
        assert self.mod._is_active_window(w) is True

    def test_zero_rescuetime_not_active(self):
        w = _make_window(36, rescuetime={"active_seconds": 0})
        assert self.mod._is_active_window(w) is False

    def test_no_rescuetime_key_not_active(self):
        w = _make_window(36)
        # No rescuetime key at all
        assert "rescuetime" not in w
        assert self.mod._is_active_window(w) is False

    def test_none_rescuetime_not_active(self):
        w = _make_window(36)
        w["rescuetime"] = None
        assert self.mod._is_active_window(w) is False

    def test_solo_meeting_is_active(self):
        """Solo focus blocks (no other attendees) are still 'active' windows."""
        w = _make_window(36, in_meeting=True, meeting_attendees=0, meeting_duration=60)
        assert self.mod._is_active_window(w) is True


# ─── Tests: recompute_window ──────────────────────────────────────────────────

class TestRecomputeWindow:
    def setup_method(self):
        self.mod = _import_recompute()

    def test_recompute_updates_metrics(self):
        """If stored metrics differ from current formula, diff is non-empty."""
        # Deliberately wrong stored metrics
        w = _make_window(36, in_meeting=False, recovery_score=80.0, hrv_rmssd_milli=70.0,
                         cls_override=0.999, ras_override=0.001)  # Obviously wrong
        updated, diff = self.mod.recompute_window(w)
        assert len(diff) > 0
        assert "cognitive_load_score" in diff
        assert "recovery_alignment_score" in diff

    def test_recompute_preserves_raw_signals(self):
        """Recomputation must not modify calendar/whoop/slack signals."""
        w = _make_window(36, in_meeting=True, meeting_attendees=3,
                         recovery_score=60.0, slack_received=5)
        updated, diff = self.mod.recompute_window(w)
        assert updated["calendar"] == w["calendar"]
        assert updated["whoop"] == w["whoop"]
        assert updated["slack"] == w["slack"]

    def test_recompute_sets_is_active_window(self):
        """is_active_window must be set in updated metadata."""
        w = _make_window(36, slack_received=2)
        # No is_active_window in metadata
        assert "is_active_window" not in w["metadata"]
        updated, diff = self.mod.recompute_window(w)
        assert updated["metadata"]["is_active_window"] is True
        assert "is_active_window" in diff  # Was None/absent → now True

    def test_recompute_corrects_none_active_flag(self):
        """Windows with is_active_window=None must be updated."""
        w = _make_window(36, in_meeting=True, slack_received=1)
        w["metadata"]["is_active_window"] = None
        updated, diff = self.mod.recompute_window(w)
        assert updated["metadata"]["is_active_window"] is True
        assert "is_active_window" in diff

    def test_recompute_no_diff_when_already_correct(self):
        """
        If stored metrics already match current formulas, diff should be empty
        (or only contain is_active_window if it was None).
        """
        from engine.metrics import compute_metrics
        # Build a window and pre-compute correct metrics
        w = _make_window(36, recovery_score=75.0, hrv_rmssd_milli=65.0,
                         sleep_performance=80.0)
        correct_metrics = compute_metrics(w)
        w["metrics"] = correct_metrics
        w["metadata"]["is_active_window"] = False  # Already correct (idle window)

        updated, diff = self.mod.recompute_window(w)
        # No metrics should differ
        metric_diffs = {k: v for k, v in diff.items() if k != "is_active_window"}
        assert len(metric_diffs) == 0, f"Unexpected metric changes: {metric_diffs}"

    def test_solo_meeting_fix_applied(self):
        """
        v1.4 fix: solo meetings (attendees=1) should get SDI=0 and FDI=1.0.
        If stored data has old values (SDI=0.35, FDI=0.60), they should be corrected.
        """
        # Simulate pre-v1.4 stored data with wrong SDI/FDI for solo meeting
        w = _make_window(
            44,  # 11:00am
            in_meeting=True,
            meeting_attendees=1,
            meeting_duration=180,
            recovery_score=92.0,
            sdi_override=0.35,  # Old wrong value
            fdi_override=0.60,  # Old wrong value
        )
        updated, diff = self.mod.recompute_window(w)

        # After recompute, solo meeting should have correct metrics
        assert updated["metrics"]["social_drain_index"] == 0.0, (
            f"Solo meeting SDI should be 0.0, got {updated['metrics']['social_drain_index']}"
        )
        assert updated["metrics"]["focus_depth_index"] == 1.0, (
            f"Solo meeting FDI should be 1.0, got {updated['metrics']['focus_depth_index']}"
        )
        assert "social_drain_index" in diff
        assert "focus_depth_index" in diff

    def test_social_meeting_unchanged_by_fix(self):
        """
        Social meetings (attendees > 1) should NOT be changed by the v1.4 fix.
        Their SDI and FDI logic was always correct.
        """
        from engine.metrics import compute_metrics
        w = _make_window(
            40,  # 10:00am
            in_meeting=True,
            meeting_attendees=5,
            meeting_duration=60,
            recovery_score=75.0,
        )
        # Pre-compute correct metrics
        correct = compute_metrics(w)
        w["metrics"] = correct
        w["metadata"]["is_active_window"] = True

        updated, diff = self.mod.recompute_window(w)
        metric_diffs = {k: v for k, v in diff.items() if k != "is_active_window"}
        assert len(metric_diffs) == 0, (
            f"Social meeting metrics should not change, but got diffs: {metric_diffs}"
        )

    def test_hrv_affects_cls_after_recompute(self):
        """
        v1.1 fix: HRV should affect CLS through physiological_readiness.
        Old data computed without HRV should be updated to include it.
        """
        from engine.metrics import cognitive_load_score, physiological_readiness

        # Window with HRV data stored but old CLS computed without HRV
        w = _make_window(36, recovery_score=75.0, hrv_rmssd_milli=30.0)  # Very low HRV
        # Old pre-v1.1 CLS would ignore HRV; new CLS uses composite readiness
        new_cls = cognitive_load_score(
            in_meeting=False, meeting_attendees=0, slack_messages_received=0,
            recovery_score=75.0, hrv_rmssd_milli=30.0
        )
        old_cls_no_hrv = cognitive_load_score(
            in_meeting=False, meeting_attendees=0, slack_messages_received=0,
            recovery_score=75.0, hrv_rmssd_milli=None
        )
        # With low HRV, CLS should be higher (more load, less capacity)
        assert new_cls > old_cls_no_hrv, (
            f"Low HRV should increase CLS; new={new_cls}, old_no_hrv={old_cls_no_hrv}"
        )


# ─── Tests: recompute_day (integration) ──────────────────────────────────────

class TestRecomputeDay:
    def setup_method(self):
        self.mod = _import_recompute()

    def test_recompute_day_no_data(self, tmp_path, monkeypatch):
        """If no data exists for a date, recompute_day returns gracefully."""
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        result = self.mod.recompute_day("2099-01-01", dry_run=True, quiet=True)
        assert result["windows"] == 0
        assert result["changed"] == 0

    def test_recompute_day_with_stale_data(self, tmp_path, monkeypatch):
        """Recompute a day with stale metrics and verify they are corrected."""
        import engine.store as store
        import config
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)
        monkeypatch.setattr(config, "SUMMARY_DIR", tmp_path)

        # Write windows with obviously wrong metrics
        windows = [
            _make_window(i, cls_override=0.999, ras_override=0.001)
            for i in range(96)
        ]
        store.write_day("2026-01-01", windows)

        result = self.mod.recompute_day("2026-01-01", dry_run=False, quiet=True)
        assert result["changed"] > 0
        assert result["windows"] == 96

        # Verify the data on disk was updated
        updated = store.read_day("2026-01-01")
        for w in updated:
            assert w["metrics"]["cognitive_load_score"] != 0.999, (
                "Stale metrics should have been overwritten"
            )
            assert w["metrics"]["cognitive_load_score"] <= 1.0

    def test_dry_run_does_not_write(self, tmp_path, monkeypatch):
        """In dry-run mode, the JSONL file should not be modified."""
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        windows = [_make_window(i, cls_override=0.999) for i in range(10)]
        store.write_day("2026-01-02", windows)

        # Get modification time before
        path = tmp_path / "2026-01-02.jsonl"
        mtime_before = path.stat().st_mtime

        self.mod.recompute_day("2026-01-02", dry_run=True, quiet=True)

        # File should not have been modified
        mtime_after = path.stat().st_mtime
        assert mtime_before == mtime_after, "Dry-run should not modify files"

    def test_recompute_day_sets_is_active_window(self, tmp_path, monkeypatch):
        """After recomputation, all windows should have is_active_window set."""
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        # Windows without is_active_window
        windows = [_make_window(i) for i in range(96)]
        # None have is_active_window in metadata
        for w in windows:
            assert "is_active_window" not in w["metadata"]

        store.write_day("2026-01-03", windows)
        self.mod.recompute_day("2026-01-03", dry_run=False, quiet=True)

        updated = store.read_day("2026-01-03")
        for w in updated:
            assert "is_active_window" in w["metadata"], (
                f"Window {w['window_id']} missing is_active_window after recompute"
            )
            assert isinstance(w["metadata"]["is_active_window"], bool)

    def test_recompute_day_corrects_solo_meeting(self, tmp_path, monkeypatch):
        """Solo meeting windows must get SDI=0 and FDI=1.0 after recompute."""
        import engine.store as store
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        windows = [_make_window(i) for i in range(96)]
        # Inject a solo meeting with wrong old metrics
        windows[44] = _make_window(
            44,
            in_meeting=True,
            meeting_attendees=1,
            meeting_duration=60,
            sdi_override=0.35,
            fdi_override=0.60,
        )
        store.write_day("2026-01-04", windows)
        self.mod.recompute_day("2026-01-04", dry_run=False, quiet=True)

        updated = store.read_day("2026-01-04")
        solo_window = updated[44]
        assert solo_window["metrics"]["social_drain_index"] == 0.0
        assert solo_window["metrics"]["focus_depth_index"] == 1.0

    def test_recompute_returns_correct_counts(self, tmp_path, monkeypatch):
        """Result dict must accurately report windows and changed counts."""
        import engine.store as store
        from engine.metrics import compute_metrics
        monkeypatch.setattr(store, "CHUNKS_DIR", tmp_path)
        monkeypatch.setattr(store, "SUMMARY_DIR", tmp_path)

        windows = []
        for i in range(96):
            w = _make_window(i)
            # Pre-compute correct metrics for all but 3 windows
            if i not in (0, 1, 2):
                correct = compute_metrics(w)
                w["metrics"] = correct
                w["metadata"]["is_active_window"] = self.mod._is_active_window(w)
            windows.append(w)

        store.write_day("2026-01-05", windows)
        result = self.mod.recompute_day("2026-01-05", dry_run=False, quiet=True)

        assert result["windows"] == 96
        # At least 3 windows had wrong metrics; is_active_window was None for 3 too
        assert result["changed"] >= 3

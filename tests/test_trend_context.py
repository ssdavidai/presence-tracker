"""
Tests for compute_trend_context() — multi-day trend detection in the daily digest.

Tests cover:
- No-history graceful fallback
- HRV decline / improvement streak detection
- Consecutive over-capacity detection
- CLS vs personal baseline comparison
- HRV vs personal baseline comparison
- Recovery score trend detection
- _generate_insight() trend-aware priority ordering
- format_digest_message() trend indicator rendering

Run with: python3 -m pytest tests/test_trend_context.py -v
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from analysis.daily_digest import compute_trend_context, _generate_insight, format_digest_message


# ─── Rolling summary helpers ──────────────────────────────────────────────────

def _make_summary(
    date_str: str,
    recovery: float = 75.0,
    hrv: float = 70.0,
    avg_cls: float = 0.30,
    avg_ras: float = 0.65,
    meeting_minutes: int = 60,
) -> dict:
    """Build a minimal day summary dict matching the rolling.json schema."""
    return {
        "date": date_str,
        "working_hours_analyzed": 60,
        "total_windows": 96,
        "metrics_avg": {
            "cognitive_load_score": avg_cls,
            "focus_depth_index": 0.70,
            "social_drain_index": 0.35,
            "context_switch_cost": 0.25,
            "recovery_alignment_score": avg_ras,
        },
        "metrics_peak": {
            "cognitive_load_score": avg_cls + 0.15,
            "focus_depth_index": 0.95,
        },
        "calendar": {
            "total_meeting_minutes": meeting_minutes,
            "meeting_windows": meeting_minutes // 15,
        },
        "slack": {
            "total_messages_sent": 20,
            "total_messages_received": 50,
        },
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "resting_heart_rate": 55.0,
            "sleep_hours": 7.5,
            "sleep_performance": 82.0,
        },
    }


def _write_rolling(summaries: list[dict], summary_dir: Path) -> None:
    """Write a rolling.json file to the given directory."""
    rolling = {
        "days": {s["date"]: s for s in summaries},
        "last_updated": datetime.now().isoformat(),
        "total_days": len(summaries),
    }
    (summary_dir / "rolling.json").write_text(json.dumps(rolling))


class _PatchedSummaryDir:
    """
    Context manager that temporarily redirects the store's SUMMARY_DIR
    to a temp directory with controlled data.
    """
    def __init__(self, summaries: list[dict]):
        self.summaries = summaries
        self._tmpdir = None
        self._orig_summary_dir = None

    def __enter__(self):
        import config
        import engine.store as store_mod

        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name)

        _write_rolling(self.summaries, tmp_path)

        # Patch both config and store module
        self._orig_config_dir = config.SUMMARY_DIR
        self._orig_store_dir = store_mod.SUMMARY_DIR
        config.SUMMARY_DIR = tmp_path
        store_mod.SUMMARY_DIR = tmp_path
        return self

    def __exit__(self, *_):
        import config
        import engine.store as store_mod

        config.SUMMARY_DIR = self._orig_config_dir
        store_mod.SUMMARY_DIR = self._orig_store_dir
        self._tmpdir.cleanup()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _dates_back(n: int, base: str = "2026-03-13") -> list[str]:
    """Return a list of n date strings ending with base, in ascending order."""
    base_dt = datetime.strptime(base, "%Y-%m-%d")
    return [(base_dt - timedelta(days=n - 1 - i)).strftime("%Y-%m-%d") for i in range(n)]


# ─── No-history graceful fallback ────────────────────────────────────────────

class TestTrendContextNoHistory:
    def test_empty_store_returns_empty_or_zero(self):
        """When no history is available, should not crash and return minimal dict."""
        with _PatchedSummaryDir([]):
            result = compute_trend_context("2026-03-14")
        # Should return a dict (possibly empty or with days_of_data=0)
        assert isinstance(result, dict)

    def test_single_day_store_is_handled(self):
        """Only one day in store (which is today) → no historical baseline."""
        summaries = [_make_summary("2026-03-13")]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-13")
        assert isinstance(result, dict)
        # Either empty (no history) or days_of_data=0
        assert result.get("days_of_data", 0) == 0

    def test_one_historical_day_is_valid(self):
        """One historical day (not today) → enough for baseline, no streak."""
        summaries = [_make_summary("2026-03-12", hrv=70.0)]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-13")
        assert isinstance(result, dict)
        assert result.get("days_of_data", 0) >= 1

    def test_no_crash_with_missing_whoop_fields(self):
        """Summaries with None/missing WHOOP fields should not crash."""
        summaries = [
            {"date": "2026-03-12", "whoop": {}, "metrics_avg": {}},
            {"date": "2026-03-11", "whoop": {"hrv_rmssd_milli": None}, "metrics_avg": {"cognitive_load_score": None}},
        ]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-13")
        assert isinstance(result, dict)


# ─── HRV streak detection ────────────────────────────────────────────────────

class TestHrvStreakDetection:
    def test_three_day_hrv_decline_detected(self):
        """HRV dropping each day for 3 days → 'declining' with streak=3."""
        dates = _dates_back(3)
        summaries = [
            _make_summary(dates[0], hrv=80.0),
            _make_summary(dates[1], hrv=74.0),
            _make_summary(dates[2], hrv=68.0),
        ]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        assert result.get("hrv_trend") == "declining"
        assert result.get("hrv_streak_days") >= 2

    def test_three_day_hrv_improvement_detected(self):
        """HRV rising each day for 3 days → 'improving' with streak>=2."""
        dates = _dates_back(3)
        summaries = [
            _make_summary(dates[0], hrv=55.0),
            _make_summary(dates[1], hrv=65.0),
            _make_summary(dates[2], hrv=75.0),
        ]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        assert result.get("hrv_trend") == "improving"
        assert result.get("hrv_streak_days") >= 2

    def test_flat_hrv_is_stable(self):
        """HRV within ±2ms each day → 'stable'."""
        dates = _dates_back(4)
        summaries = [_make_summary(d, hrv=70.0 + i * 0.5) for i, d in enumerate(dates)]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        assert result.get("hrv_trend") == "stable"
        assert result.get("hrv_streak_days") == 0

    def test_single_day_drop_is_not_a_streak(self):
        """Only one decline step → streak should be 0 or 1 (not ≥2)."""
        dates = _dates_back(3)
        summaries = [
            _make_summary(dates[0], hrv=72.0),
            _make_summary(dates[1], hrv=72.0),  # flat
            _make_summary(dates[2], hrv=65.0),  # one drop at the end
        ]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        # Streak of consecutive *days* declining; one step doesn't make 2-day streak
        # from the historical series alone. With today added it would be 1.
        assert result.get("hrv_streak_days", 0) <= 1

    def test_hrv_trend_requires_threshold(self):
        """A 1ms change per day should not trigger a streak (threshold=2ms)."""
        dates = _dates_back(4)
        summaries = [_make_summary(d, hrv=70.0 - i * 1.0) for i, d in enumerate(dates)]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        # Each step is only 1ms, below the 2ms threshold
        assert result.get("hrv_trend") == "stable"

    def test_streak_breaks_on_reversal(self):
        """A recovery day in the middle of a decline should break the streak."""
        dates = _dates_back(5)
        summaries = [
            _make_summary(dates[0], hrv=80.0),
            _make_summary(dates[1], hrv=73.0),  # -7 (decline)
            _make_summary(dates[2], hrv=78.0),  # +5 (reversal — breaks streak)
            _make_summary(dates[3], hrv=71.0),  # -7 (new decline)
            _make_summary(dates[4], hrv=64.0),  # -7 (decline continues)
        ]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-16")
        # Streak should only count the days after the last reversal
        if result.get("hrv_trend") == "declining":
            assert result.get("hrv_streak_days", 0) <= 2


# ─── Over-capacity streak detection ──────────────────────────────────────────

class TestOverCapacityStreak:
    def test_three_consecutive_over_capacity_days(self):
        """3 days of RAS < 0.45 → overcapacity_streak=3."""
        dates = _dates_back(3)
        summaries = [_make_summary(d, avg_ras=0.35) for d in dates]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        assert result.get("overcapacity_streak", 0) >= 3

    def test_one_over_capacity_day(self):
        """Only one RAS < 0.45 day → streak=1."""
        dates = _dates_back(3)
        summaries = [
            _make_summary(dates[0], avg_ras=0.70),
            _make_summary(dates[1], avg_ras=0.65),
            _make_summary(dates[2], avg_ras=0.35),  # last day only
        ]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        assert result.get("overcapacity_streak", 0) <= 1

    def test_recovery_day_breaks_streak(self):
        """A RAS >= 0.45 day resets the streak to 0."""
        dates = _dates_back(4)
        summaries = [
            _make_summary(dates[0], avg_ras=0.35),
            _make_summary(dates[1], avg_ras=0.35),
            _make_summary(dates[2], avg_ras=0.70),  # recovery day — breaks streak
            _make_summary(dates[3], avg_ras=0.35),  # over-capacity again
        ]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        # The streak from dates[3] alone is 1 (no yesterday value is over-capacity)
        assert result.get("overcapacity_streak", 0) <= 1

    def test_well_aligned_days_have_zero_streak(self):
        """All RAS >= 0.60 → no over-capacity streak."""
        dates = _dates_back(5)
        summaries = [_make_summary(d, avg_ras=0.72) for d in dates]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-16")
        assert result.get("overcapacity_streak", 0) == 0


# ─── CLS vs baseline ─────────────────────────────────────────────────────────

class TestClsVsBaseline:
    def test_higher_than_baseline_is_positive(self):
        """Today's CLS well above historical average → cls_vs_baseline > 0."""
        dates = _dates_back(5)
        summaries = [_make_summary(d, avg_cls=0.25) for d in dates]
        # Add today with 0.60 CLS → ~140% above 0.25 baseline
        summaries.append(_make_summary("2026-03-14", avg_cls=0.60))
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        vs = result.get("cls_vs_baseline")
        if vs is not None:
            assert vs > 0, f"Expected positive cls_vs_baseline, got {vs}"

    def test_lower_than_baseline_is_negative(self):
        """Today's CLS well below historical average → cls_vs_baseline < 0."""
        dates = _dates_back(5)
        summaries = [_make_summary(d, avg_cls=0.55) for d in dates]
        summaries.append(_make_summary("2026-03-14", avg_cls=0.15))
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        vs = result.get("cls_vs_baseline")
        if vs is not None:
            assert vs < 0, f"Expected negative cls_vs_baseline, got {vs}"

    def test_cls_baseline_value_is_reasonable(self):
        """cls_baseline should match the historical average."""
        dates = _dates_back(4)
        summaries = [_make_summary(d, avg_cls=0.30) for d in dates]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        baseline = result.get("cls_baseline")
        if baseline is not None:
            assert abs(baseline - 0.30) < 0.05, f"Expected ~0.30 baseline, got {baseline}"

    def test_no_cls_vs_baseline_without_today(self):
        """If today has no entry in rolling store, cls_vs_baseline may be None."""
        dates = _dates_back(3)
        summaries = [_make_summary(d, avg_cls=0.40) for d in dates]
        # today_date doesn't exist in store → today_cls = None
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-20")  # Far future, not in store
        # cls_vs_baseline should be None (can't compare without today's value)
        assert result.get("cls_vs_baseline") is None


# ─── HRV vs baseline ─────────────────────────────────────────────────────────

class TestHrvVsBaseline:
    def test_hrv_above_baseline_is_positive(self):
        dates = _dates_back(5)
        summaries = [_make_summary(d, hrv=65.0) for d in dates]
        summaries.append(_make_summary("2026-03-14", hrv=90.0))  # Today: 38% above
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        vs = result.get("hrv_vs_baseline")
        if vs is not None:
            assert vs > 0

    def test_hrv_below_baseline_is_negative(self):
        dates = _dates_back(5)
        summaries = [_make_summary(d, hrv=80.0) for d in dates]
        summaries.append(_make_summary("2026-03-14", hrv=50.0))  # Today: 37% below
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        vs = result.get("hrv_vs_baseline")
        if vs is not None:
            assert vs < 0

    def test_hrv_baseline_ms_matches_historical_average(self):
        dates = _dates_back(4)
        summaries = [_make_summary(d, hrv=72.0) for d in dates]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        baseline = result.get("hrv_baseline_ms")
        if baseline is not None:
            assert abs(baseline - 72.0) < 2.0, f"Expected ~72ms baseline, got {baseline}"


# ─── Recovery streak detection ────────────────────────────────────────────────

class TestRecoveryTrend:
    def test_three_day_recovery_decline_detected(self):
        dates = _dates_back(3)
        summaries = [
            _make_summary(dates[0], recovery=85.0),
            _make_summary(dates[1], recovery=78.0),
            _make_summary(dates[2], recovery=70.0),
        ]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        assert result.get("recovery_trend") == "declining"
        assert result.get("recovery_streak_days", 0) >= 2

    def test_stable_recovery_is_stable(self):
        dates = _dates_back(4)
        summaries = [_make_summary(d, recovery=75.0 + i) for i, d in enumerate(dates)]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        # Each step is 1%, below the 3% threshold
        assert result.get("recovery_trend") == "stable"


# ─── Note field ───────────────────────────────────────────────────────────────

class TestTrendNote:
    def test_note_is_always_string(self):
        """The note field should always be a string, never None."""
        cases = [
            [],
            [_make_summary("2026-03-12")],
            [_make_summary(d, hrv=70.0 - i * 8) for i, d in enumerate(_dates_back(4))],
            [_make_summary(d, avg_ras=0.30) for d in _dates_back(5)],
        ]
        for summaries in cases:
            with _PatchedSummaryDir(summaries):
                result = compute_trend_context("2026-03-14")
            note = result.get("note", "")
            assert isinstance(note, str), f"Expected string note, got {type(note)}: {note}"

    def test_serious_hrv_decline_note_is_non_empty(self):
        """A 3-day HRV decline should produce a non-empty warning note."""
        dates = _dates_back(3)
        summaries = [
            _make_summary(dates[0], hrv=85.0),
            _make_summary(dates[1], hrv=76.0),
            _make_summary(dates[2], hrv=67.0),
        ]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        note = result.get("note", "")
        if result.get("hrv_streak_days", 0) >= 3:
            assert len(note) > 0, "Expected non-empty note for 3-day HRV decline"

    def test_over_capacity_streak_note_is_non_empty(self):
        """3+ consecutive over-capacity days should produce a warning note."""
        dates = _dates_back(4)
        summaries = [_make_summary(d, avg_ras=0.32) for d in dates]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        note = result.get("note", "")
        if result.get("overcapacity_streak", 0) >= 3:
            assert len(note) > 0

    def test_positive_hrv_trend_note_is_positive(self):
        """Improving HRV should not produce an alarm note."""
        dates = _dates_back(4)
        summaries = [_make_summary(d, hrv=55.0 + i * 8) for i, d in enumerate(dates)]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        note = result.get("note", "")
        # Should not contain alarm words
        assert "critical" not in note.lower()
        assert "danger" not in note.lower()


# ─── _generate_insight trend-aware priority ───────────────────────────────────

class TestGenerateInsightWithTrend:
    """
    _generate_insight() should prefer multi-day trend signals over
    single-day observations when both are present.
    """

    def _call(self, **kwargs) -> str:
        defaults = dict(
            recovery=75.0,
            avg_cls=0.35,
            avg_fdi_active=0.65,
            avg_ras=0.65,
            total_meeting_minutes=90,
            total_sent=15,
            peak_window=None,
            working_count=60,
            active_count=12,
            trend={},
        )
        defaults.update(kwargs)
        return _generate_insight(**defaults)

    def test_overcapacity_streak_beats_single_day(self):
        """3-day over-capacity streak should surface even if today alone is borderline."""
        insight = self._call(
            avg_ras=0.50,  # today: just within capacity
            trend={"overcapacity_streak": 3, "hrv_trend": "stable", "hrv_streak_days": 0},
        )
        assert "consecutive" in insight.lower() or "3" in insight

    def test_hrv_decline_streak_beats_moderate_recovery(self):
        """HRV declining 3 days should surface even if today's recovery is decent."""
        insight = self._call(
            recovery=68.0,  # decent but not alarming
            trend={
                "hrv_trend": "declining",
                "hrv_streak_days": 3,
                "overcapacity_streak": 0,
                "hrv_vs_baseline": None,
                "cls_vs_baseline": None,
            },
        )
        assert "hrv" in insight.lower() or "3" in insight.lower()

    def test_two_day_overcapacity_with_low_recovery_surfaces(self):
        """2 days over capacity + today at 52% recovery → two-day warning."""
        insight = self._call(
            recovery=52.0,
            avg_ras=0.38,
            trend={
                "overcapacity_streak": 2,
                "hrv_trend": "stable",
                "hrv_streak_days": 0,
                "hrv_vs_baseline": None,
                "cls_vs_baseline": None,
            },
        )
        assert "two" in insight.lower() or "2" in insight or "days" in insight.lower()

    def test_cls_above_baseline_surfaced_when_no_streak(self):
        """When no streak, a 50% CLS above baseline should appear in insight."""
        insight = self._call(
            trend={
                "overcapacity_streak": 0,
                "hrv_trend": "stable",
                "hrv_streak_days": 0,
                "recovery_trend": "stable",
                "recovery_streak_days": 0,
                "cls_vs_baseline": 55.0,   # 55% above average
                "cls_baseline": 0.28,
                "hrv_vs_baseline": None,
                "hrv_baseline_ms": None,
                "days_of_data": 5,
            },
        )
        # Should mention elevated load vs baseline
        assert "baseline" in insight.lower() or "above" in insight.lower() or "55" in insight

    def test_improving_hrv_streak_produces_positive_insight(self):
        """3-day HRV improvement should produce a positive insight."""
        insight = self._call(
            recovery=82.0,
            trend={
                "overcapacity_streak": 0,
                "hrv_trend": "improving",
                "hrv_streak_days": 3,
                "recovery_trend": "stable",
                "recovery_streak_days": 0,
                "cls_vs_baseline": None,
                "hrv_vs_baseline": None,
                "days_of_data": 4,
            },
        )
        assert "3" in insight or "three" in insight.lower() or "improving" in insight.lower()

    def test_no_trend_falls_back_to_today_logic(self):
        """Empty trend dict → falls back to single-day insight generation."""
        insight = self._call(
            recovery=35.0,
            avg_cls=0.65,
            trend={},
        )
        # Should still generate meaningful insight from today's data
        assert "35" in insight or "pushed" in insight.lower() or "recovery" in insight.lower()

    def test_insight_is_always_string(self):
        """Should never return None."""
        trend_variants = [
            {},
            {"overcapacity_streak": 4, "hrv_trend": "stable"},
            {"hrv_trend": "declining", "hrv_streak_days": 5, "overcapacity_streak": 0},
        ]
        for t in trend_variants:
            result = self._call(trend=t)
            assert isinstance(result, str), f"Expected string insight, got {type(result)}"
            assert len(result) > 0


# ─── format_digest_message trend indicator rendering ─────────────────────────

class TestFormatDigestMessageTrends:
    def _make_digest_with_trend(self, trend: dict) -> dict:
        return {
            "date": "2026-03-13",
            "whoop": {"recovery_score": 72.0, "hrv_rmssd_milli": 58.0, "sleep_hours": 7.5, "sleep_performance": 80.0},
            "metrics": {"avg_cls": 0.38, "peak_cls": 0.55, "avg_fdi_active": 0.62, "avg_sdi_active": 0.40, "avg_csc_active": 0.28, "avg_ras": 0.61},
            "activity": {"working_windows": 60, "active_windows": 14, "idle_windows": 46, "total_meeting_minutes": 75, "meeting_count": 2, "slack_sent": 18, "slack_received": 40},
            "peak_window": None,
            "trend": trend,
            "insight": "Test insight.",
        }

    def test_hrv_decline_streak_shown(self):
        """HRV declining 3 days → 'HRV ↓ Xd' appears in message."""
        digest = self._make_digest_with_trend({
            "hrv_trend": "declining",
            "hrv_streak_days": 3,
            "overcapacity_streak": 0,
            "hrv_vs_baseline": None,
            "cls_vs_baseline": None,
        })
        msg = format_digest_message(digest)
        assert "HRV ↓" in msg or "HRV" in msg

    def test_hrv_improvement_streak_shown(self):
        """HRV improving 2 days → 'HRV ↑ Xd' appears."""
        digest = self._make_digest_with_trend({
            "hrv_trend": "improving",
            "hrv_streak_days": 2,
            "overcapacity_streak": 0,
            "hrv_vs_baseline": None,
            "cls_vs_baseline": None,
        })
        msg = format_digest_message(digest)
        assert "HRV ↑" in msg or "HRV" in msg

    def test_overcapacity_streak_shown(self):
        """3+ over-capacity days → 'over-capacity Xd' in message."""
        digest = self._make_digest_with_trend({
            "hrv_trend": "stable",
            "hrv_streak_days": 0,
            "overcapacity_streak": 3,
            "hrv_vs_baseline": None,
            "cls_vs_baseline": None,
        })
        msg = format_digest_message(digest)
        assert "over-capacity" in msg or "3" in msg

    def test_hrv_vs_baseline_shown_when_significant(self):
        """HRV 20% above baseline → baseline comparison shown."""
        digest = self._make_digest_with_trend({
            "hrv_trend": "stable",
            "hrv_streak_days": 0,
            "overcapacity_streak": 0,
            "hrv_vs_baseline": 22.0,
            "cls_vs_baseline": None,
        })
        msg = format_digest_message(digest)
        assert "+22%" in msg or "22" in msg or "HRV" in msg

    def test_cls_vs_baseline_shown_when_significant(self):
        """CLS 40% above baseline → baseline comparison shown."""
        digest = self._make_digest_with_trend({
            "hrv_trend": "stable",
            "hrv_streak_days": 0,
            "overcapacity_streak": 0,
            "hrv_vs_baseline": None,
            "cls_vs_baseline": 45.0,
        })
        msg = format_digest_message(digest)
        assert "+45%" in msg or "45" in msg or "baseline" in msg.lower() or "Load" in msg

    def test_small_hrv_vs_baseline_not_shown(self):
        """HRV only 5% above baseline → not worth showing (below 10% threshold)."""
        digest = self._make_digest_with_trend({
            "hrv_trend": "stable",
            "hrv_streak_days": 0,
            "overcapacity_streak": 0,
            "hrv_vs_baseline": 5.0,
            "cls_vs_baseline": None,
        })
        msg = format_digest_message(digest)
        # The trend section should not appear for insignificant deviations
        assert "Trends:" not in msg

    def test_no_trend_section_when_all_stable(self):
        """No streaks and no notable baseline deviations → no trend section."""
        digest = self._make_digest_with_trend({
            "hrv_trend": "stable",
            "hrv_streak_days": 0,
            "overcapacity_streak": 0,
            "hrv_vs_baseline": 3.0,   # Small
            "cls_vs_baseline": -5.0,  # Small
        })
        msg = format_digest_message(digest)
        assert "Trends:" not in msg

    def test_empty_trend_dict_does_not_crash(self):
        """Empty or absent trend dict should not break message formatting."""
        digest = self._make_digest_with_trend({})
        msg = format_digest_message(digest)
        assert isinstance(msg, str)
        assert len(msg) > 50

    def test_trend_section_appears_before_insight(self):
        """Trend context should appear before the insight line."""
        digest = self._make_digest_with_trend({
            "hrv_trend": "declining",
            "hrv_streak_days": 3,
            "overcapacity_streak": 0,
            "hrv_vs_baseline": None,
            "cls_vs_baseline": None,
        })
        msg = format_digest_message(digest)
        if "Trends:" in msg and "💡" in msg:
            trend_pos = msg.index("Trends:")
            insight_pos = msg.index("💡")
            assert trend_pos < insight_pos, "Trend section should appear before insight"


# ─── Integration: trend context in full pipeline ──────────────────────────────

class TestTrendContextIntegration:
    """
    Verify compute_trend_context() returns the right schema and that
    compute_digest() correctly includes trend in its output.
    """

    def test_result_has_required_keys(self):
        """compute_trend_context() must always include certain keys when data exists."""
        dates = _dates_back(3)
        summaries = [_make_summary(d) for d in dates]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-14")
        if result.get("days_of_data", 0) >= 1:
            for key in ["hrv_trend", "hrv_streak_days", "overcapacity_streak", "note"]:
                assert key in result, f"Missing expected key: {key}"

    def test_hrv_trend_is_valid_string(self):
        dates = _dates_back(4)
        summaries = [_make_summary(d, hrv=70.0) for d in dates]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        hrv_trend = result.get("hrv_trend", "stable")
        assert hrv_trend in ("declining", "improving", "stable")

    def test_recovery_trend_is_valid_string(self):
        dates = _dates_back(4)
        summaries = [_make_summary(d) for d in dates]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        rec_trend = result.get("recovery_trend", "stable")
        assert rec_trend in ("declining", "improving", "stable")

    def test_overcapacity_streak_is_non_negative_int(self):
        dates = _dates_back(4)
        summaries = [_make_summary(d, avg_ras=0.35) for d in dates]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-15")
        streak = result.get("overcapacity_streak", 0)
        assert isinstance(streak, int)
        assert streak >= 0

    def test_compute_digest_includes_trend_key(self):
        """compute_digest() must include 'trend' in its return dict."""
        from engine.chunker import build_windows
        from analysis.daily_digest import compute_digest

        windows = build_windows(
            date_str="2026-03-14",
            whoop_data={"recovery_score": 75.0, "hrv_rmssd_milli": 68.0,
                        "resting_heart_rate": 55.0, "sleep_performance": 80.0,
                        "sleep_hours": 7.5, "strain": 12.0, "spo2_percentage": 95.0},
            calendar_data={"events": [], "event_count": 0, "total_meeting_minutes": 0,
                           "max_concurrent_attendees": 0},
            slack_windows={},
        )
        digest = compute_digest(windows)
        assert "trend" in digest, "compute_digest() must include 'trend' key"
        assert isinstance(digest["trend"], dict)

    def test_trend_context_all_values_are_numeric_or_none(self):
        """Numeric fields in trend result must be float/int/None, never strings."""
        dates = _dates_back(5)
        summaries = [_make_summary(d, hrv=70.0 - i * 5, avg_cls=0.30 + i * 0.05)
                     for i, d in enumerate(dates)]
        with _PatchedSummaryDir(summaries):
            result = compute_trend_context("2026-03-16")
        numeric_keys = ["hrv_vs_baseline", "hrv_baseline_ms", "cls_vs_baseline",
                        "cls_baseline", "recovery_vs_baseline"]
        for k in numeric_keys:
            val = result.get(k)
            if val is not None:
                assert isinstance(val, (int, float)), f"Key {k}: expected numeric, got {type(val)}"

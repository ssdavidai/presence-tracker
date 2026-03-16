"""
Tests for analysis/sleep_focus_correlator.py — Sleep-to-Focus Correlation

Coverage:
  1. _pearson_r()
     - Returns None for fewer than 3 pairs
     - Returns None when std dev is zero (constant series)
     - Returns +1.0 for perfectly correlated series
     - Returns -1.0 for perfectly anti-correlated series
     - Returns correct r for known 5-element series

  2. _linear_fit()
     - Returns (None, None) for fewer than 3 pairs
     - Returns (None, None) when std dev of x is zero
     - Returns correct slope and intercept for known 5-element series

  3. _compute_sleep_buckets()
     - Empty pairs → all buckets have count=0
     - Pairs correctly bucketed by sleep_hours
     - FDI averages computed correctly per bucket

  4. _compute_recovery_buckets()
     - Pairs correctly bucketed by recovery_score
     - BucketStats.avg_fdi computed correctly

  5. compute_sleep_focus_correlation()
     - Returns is_meaningful=False with < MIN_PAIRS pairs
     - Returns is_meaningful=True with >= MIN_PAIRS pairs
     - insight_lines is non-empty even when not meaningful
     - correlations dict populated for all predictors when meaningful
     - to_dict() serialises without error

  6. format_sleep_insight_line()
     - Returns empty string when not meaningful
     - Returns non-empty string starting with "😴" when meaningful

  7. format_sleep_insight_section()
     - Returns non-empty string with bucket table when meaningful
     - Degrades to empty string when not meaningful and no insight

Run with: python3 -m pytest tests/test_sleep_focus_correlator.py -v
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.sleep_focus_correlator import (
    SleepFocusPair,
    SleepFocusCorrelation,
    BucketStats,
    _pearson_r,
    _linear_fit,
    _paired,
    _compute_sleep_buckets,
    _compute_recovery_buckets,
    _generate_insights,
    compute_sleep_focus_correlation,
    format_sleep_insight_line,
    format_sleep_insight_section,
    MIN_PAIRS,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_pair(
    sleep_date: str,
    sleep_hours: float = 7.5,
    hrv: float = 70.0,
    recovery: float = 75.0,
    sleep_perf: float = 85.0,
    fdi: float = 0.85,
    cls: float = 0.25,
    dps: float = 70.0,
) -> SleepFocusPair:
    """Create a minimal SleepFocusPair for testing."""
    dt = datetime.strptime(sleep_date, "%Y-%m-%d") + timedelta(days=1)
    focus_date = dt.strftime("%Y-%m-%d")
    return SleepFocusPair(
        sleep_date=sleep_date,
        focus_date=focus_date,
        sleep_hours=sleep_hours,
        sleep_performance=sleep_perf,
        hrv_rmssd=hrv,
        recovery_score=recovery,
        next_day_fdi=fdi,
        next_day_cls=cls,
        next_day_dps=dps,
    )


def _make_pairs_from_data(rows: list[tuple]) -> list[SleepFocusPair]:
    """
    Build pairs from (sleep_hours, fdi) tuples, assigning consecutive dates.
    """
    pairs = []
    for i, (sh, fdi) in enumerate(rows):
        date_str = (datetime(2026, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        pairs.append(_make_pair(date_str, sleep_hours=sh, fdi=fdi))
    return pairs


# ─── _pearson_r ───────────────────────────────────────────────────────────────

class TestPearsonR:

    def test_too_few_pairs_returns_none(self):
        assert _pearson_r([1.0, 2.0], [1.0, 2.0]) is None

    def test_perfect_positive_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        r = _pearson_r(xs, ys)
        assert r is not None
        assert abs(r - 1.0) < 1e-9

    def test_perfect_negative_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 8.0, 6.0, 4.0, 2.0]
        r = _pearson_r(xs, ys)
        assert r is not None
        assert abs(r - (-1.0)) < 1e-9

    def test_zero_variance_x_returns_none(self):
        # All x values identical → std dev = 0
        r = _pearson_r([3.0, 3.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0])
        assert r is None

    def test_zero_variance_y_returns_none(self):
        r = _pearson_r([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 5.0, 5.0, 5.0, 5.0])
        assert r is None

    def test_known_correlation(self):
        # Manually verified with the formula
        xs = [6.0, 7.0, 8.0, 6.5, 7.5]
        ys = [0.70, 0.80, 0.90, 0.75, 0.85]
        r = _pearson_r(xs, ys)
        assert r is not None
        assert abs(r - 1.0) < 0.01  # These are linearly related

    def test_result_bounded_between_minus_one_and_one(self):
        import random
        random.seed(42)
        xs = [random.gauss(0, 1) for _ in range(20)]
        ys = [random.gauss(0, 1) for _ in range(20)]
        r = _pearson_r(xs, ys)
        if r is not None:
            assert -1.0 <= r <= 1.0

    def test_unequal_lengths_returns_none(self):
        r = _pearson_r([1.0, 2.0, 3.0], [1.0, 2.0])
        assert r is None


# ─── _linear_fit ──────────────────────────────────────────────────────────────

class TestLinearFit:

    def test_too_few_pairs(self):
        slope, intercept = _linear_fit([1.0, 2.0], [1.0, 2.0])
        assert slope is None and intercept is None

    def test_zero_variance_x(self):
        slope, intercept = _linear_fit([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])
        assert slope is None

    def test_known_slope(self):
        # y = 2*x + 1 → slope=2, intercept=1
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [3.0, 5.0, 7.0, 9.0, 11.0]
        slope, intercept = _linear_fit(xs, ys)
        assert slope is not None
        assert abs(slope - 2.0) < 1e-9
        assert abs(intercept - 1.0) < 1e-9

    def test_horizontal_line(self):
        # All y the same → slope = 0
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 5.0, 5.0, 5.0, 5.0]
        slope, intercept = _linear_fit(xs, ys)
        # slope = 0, intercept = 5
        assert slope is not None
        assert abs(slope) < 1e-9
        assert abs(intercept - 5.0) < 1e-9


# ─── _compute_sleep_buckets ───────────────────────────────────────────────────

class TestSleepBuckets:

    def test_empty_pairs_all_zero(self):
        buckets = _compute_sleep_buckets([])
        assert all(b.count == 0 for b in buckets)
        assert len(buckets) == 4  # <6h, 6-7h, 7-8h, ≥8h

    def test_pairs_bucketed_correctly(self):
        pairs = [
            _make_pair("2026-01-01", sleep_hours=5.5, fdi=0.60),   # <6h
            _make_pair("2026-01-02", sleep_hours=6.5, fdi=0.70),   # 6-7h
            _make_pair("2026-01-03", sleep_hours=7.5, fdi=0.85),   # 7-8h
            _make_pair("2026-01-04", sleep_hours=8.5, fdi=0.92),   # ≥8h
        ]
        buckets = _compute_sleep_buckets(pairs)
        assert buckets[0].count == 1  # <6h
        assert buckets[1].count == 1  # 6-7h
        assert buckets[2].count == 1  # 7-8h
        assert buckets[3].count == 1  # ≥8h

    def test_fdi_averages_correct(self):
        # Two pairs in the 7-8h bucket
        pairs = [
            _make_pair("2026-01-01", sleep_hours=7.0, fdi=0.80),
            _make_pair("2026-01-02", sleep_hours=7.9, fdi=0.90),
        ]
        buckets = _compute_sleep_buckets(pairs)
        seven_eight_bucket = buckets[2]  # "7–8h" is at index 2
        assert seven_eight_bucket.count == 2
        assert abs(seven_eight_bucket.avg_fdi - 0.85) < 1e-9

    def test_none_fdi_excluded_from_average(self):
        pairs = [
            _make_pair("2026-01-01", sleep_hours=7.0, fdi=0.80),
        ]
        # Manually set fdi to None for a second pair
        p2 = _make_pair("2026-01-02", sleep_hours=7.5, fdi=None)
        pairs.append(p2)
        buckets = _compute_sleep_buckets(pairs)
        seven_eight = buckets[2]
        assert seven_eight.count == 2  # Both are in bucket
        assert abs(seven_eight.avg_fdi - 0.80) < 1e-9  # Only one non-None


# ─── _compute_recovery_buckets ────────────────────────────────────────────────

class TestRecoveryBuckets:

    def test_empty_pairs(self):
        buckets = _compute_recovery_buckets([])
        assert all(b.count == 0 for b in buckets)
        assert len(buckets) == 3  # Red, Yellow, Green

    def test_recovery_bucketed_correctly(self):
        pairs = [
            _make_pair("2026-01-01", recovery=40.0, fdi=0.60),  # Red <50
            _make_pair("2026-01-02", recovery=60.0, fdi=0.75),  # Yellow 50-67
            _make_pair("2026-01-03", recovery=80.0, fdi=0.90),  # Green ≥67
        ]
        buckets = _compute_recovery_buckets(pairs)
        assert buckets[0].count == 1  # Red
        assert buckets[1].count == 1  # Yellow
        assert buckets[2].count == 1  # Green

    def test_boundary_67_goes_into_green(self):
        # Recovery=67 should be in Green (≥67)
        p = _make_pair("2026-01-01", recovery=67.0, fdi=0.80)
        buckets = _compute_recovery_buckets([p])
        assert buckets[2].count == 1  # Green


# ─── compute_sleep_focus_correlation ──────────────────────────────────────────

class TestComputeSleepFocusCorrelation:

    def _mock_store(self, pairs_data: list[dict], monkeypatch):
        """
        Patch the store functions so compute_sleep_focus_correlation
        reads from our synthetic data instead of the JSONL store.
        """
        dates = [d["date"] for d in pairs_data]
        all_days = {d["date"]: d["record"] for d in pairs_data}

        def fake_list_available_dates():
            return sorted(dates)

        def fake_read_summary():
            return {"days": all_days}

        import analysis.sleep_focus_correlator as mod
        monkeypatch.setattr(mod, "list_available_dates", fake_list_available_dates)
        monkeypatch.setattr(mod, "read_summary", fake_read_summary)

    def _day_record(
        self,
        date: str,
        sleep_hours: float = 7.5,
        hrv: float = 70.0,
        recovery: float = 75.0,
        sleep_perf: float = 85.0,
        fdi: float = 0.85,
        cls: float = 0.25,
        dps: float = 70.0,
    ) -> dict:
        return {
            "date": date,
            "record": {
                "date": date,
                "whoop": {
                    "sleep_hours": sleep_hours,
                    "hrv_rmssd_milli": hrv,
                    "recovery_score": recovery,
                    "sleep_performance": sleep_perf,
                },
                "metrics_avg": {
                    "focus_depth_index": fdi,
                    "cognitive_load_score": cls,
                },
                "presence_score": {"dps": dps},
            },
        }

    def test_too_few_pairs_not_meaningful(self, monkeypatch):
        # Only 3 days → 2 pairs → below MIN_PAIRS
        days = [
            self._day_record("2026-01-01"),
            self._day_record("2026-01-02"),
            self._day_record("2026-01-03"),
        ]
        self._mock_store(days, monkeypatch)
        corr = compute_sleep_focus_correlation("2026-01-03")
        assert corr.is_meaningful is False
        assert corr.pairs_used < MIN_PAIRS

    def test_enough_pairs_is_meaningful(self, monkeypatch):
        # 7 consecutive days → 6 pairs → above MIN_PAIRS (5)
        days = [
            self._day_record(
                f"2026-01-{i+1:02d}",
                sleep_hours=6.0 + i * 0.3,  # Varying sleep
                fdi=0.70 + i * 0.03,
            )
            for i in range(7)
        ]
        self._mock_store(days, monkeypatch)
        corr = compute_sleep_focus_correlation("2026-01-07")
        assert corr.is_meaningful is True
        assert corr.pairs_used >= MIN_PAIRS

    def test_insight_lines_always_present(self, monkeypatch):
        # Even with no data, insight_lines should be populated
        self._mock_store([], monkeypatch)
        corr = compute_sleep_focus_correlation("2026-01-01")
        assert len(corr.insight_lines) >= 1

    def test_correlations_populated_when_meaningful(self, monkeypatch):
        days = [
            self._day_record(
                f"2026-01-{i+1:02d}",
                sleep_hours=6.0 + i * 0.5,
                fdi=0.65 + i * 0.05,
            )
            for i in range(7)
        ]
        self._mock_store(days, monkeypatch)
        corr = compute_sleep_focus_correlation("2026-01-07")
        if corr.is_meaningful:
            # Should have correlations for all predictor/outcome combos
            assert "sleep_hours__next_day_fdi" in corr.correlations
            assert "hrv_rmssd__next_day_fdi" in corr.correlations

    def test_to_dict_serialises(self, monkeypatch):
        days = [
            self._day_record(f"2026-01-{i+1:02d}")
            for i in range(7)
        ]
        self._mock_store(days, monkeypatch)
        corr = compute_sleep_focus_correlation("2026-01-07")
        d = corr.to_dict()
        # Must be JSON serialisable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        assert "pairs_used" in d

    def test_slope_meaningful_with_correlated_data(self, monkeypatch):
        # Perfect linear: sleep_hours → fdi = sleep_hours * 0.1
        days = []
        for i in range(8):
            sh = 6.0 + i * 0.3
            fdi = sh * 0.1  # Perfect linear relationship
            days.append(self._day_record(f"2026-01-{i+1:02d}", sleep_hours=sh, fdi=fdi))
        self._mock_store(days, monkeypatch)
        corr = compute_sleep_focus_correlation("2026-01-08")
        if corr.is_meaningful and corr.sleep_hours_slope is not None:
            # Should be close to 0.1
            assert abs(corr.sleep_hours_slope - 0.1) < 0.01

    def test_date_range_populated(self, monkeypatch):
        days = [
            self._day_record(f"2026-01-{i+1:02d}")
            for i in range(7)
        ]
        self._mock_store(days, monkeypatch)
        corr = compute_sleep_focus_correlation("2026-01-07")
        if corr.pairs_used > 0:
            assert corr.date_range != ""


# ─── format_sleep_insight_line ────────────────────────────────────────────────

class TestFormatSleepInsightLine:

    def test_not_meaningful_returns_empty(self):
        corr = SleepFocusCorrelation(is_meaningful=False)
        assert format_sleep_insight_line(corr) == ""

    def test_meaningful_starts_with_emoji(self):
        corr = SleepFocusCorrelation(
            is_meaningful=True,
            pairs_used=10,
            insight_lines=["Each extra hour adds +5.0 FDI points (r=+0.40, 10 nights)."],
        )
        result = format_sleep_insight_line(corr)
        assert result.startswith("😴")
        assert "r=+0.40" in result

    def test_meaningful_no_insights_returns_empty(self):
        corr = SleepFocusCorrelation(
            is_meaningful=True,
            pairs_used=10,
            insight_lines=[],
        )
        assert format_sleep_insight_line(corr) == ""


# ─── format_sleep_insight_section ────────────────────────────────────────────

class TestFormatSleepInsightSection:

    def test_not_meaningful_no_insight(self):
        corr = SleepFocusCorrelation(is_meaningful=False, insight_lines=[])
        assert format_sleep_insight_section(corr) == ""

    def test_not_meaningful_with_insight_graceful(self):
        corr = SleepFocusCorrelation(
            is_meaningful=False,
            insight_lines=["Need 3 more days."],
        )
        result = format_sleep_insight_section(corr)
        assert "Need 3 more days" in result

    def test_meaningful_includes_header(self):
        corr = SleepFocusCorrelation(
            is_meaningful=True,
            pairs_used=12,
            date_range="2026-01-01 → 2026-02-01",
            insight_lines=["Each extra hour adds +5.0 FDI points."],
            sleep_buckets=[
                BucketStats("< 6h", 2, 0.65, 0.40, 55.0),
                BucketStats("6–7h", 4, 0.75, 0.30, 65.0),
                BucketStats("7–8h", 5, 0.85, 0.22, 75.0),
                BucketStats("≥ 8h", 1, 0.90, 0.18, 80.0),
            ],
            recovery_buckets=[],
        )
        result = format_sleep_insight_section(corr)
        assert "😴" in result
        assert "12 nights" in result
        assert "Sleep duration buckets" in result

    def test_meaningful_bucket_table_shows_counts(self):
        corr = SleepFocusCorrelation(
            is_meaningful=True,
            pairs_used=8,
            date_range="2026-01-01 → 2026-01-09",
            insight_lines=["Test insight."],
            sleep_buckets=[
                BucketStats("< 6h", 0, None, None, None),
                BucketStats("6–7h", 3, 0.72, 0.32, 62.0),
                BucketStats("7–8h", 4, 0.84, 0.22, 74.0),
                BucketStats("≥ 8h", 1, 0.90, 0.18, 81.0),
            ],
            recovery_buckets=[],
        )
        result = format_sleep_insight_section(corr)
        # Should not show the <6h bucket (count=0)
        assert "< 6h" not in result
        # Should show 6-7h and 7-8h
        assert "6–7h" in result
        assert "7–8h" in result


# ─── _generate_insights ───────────────────────────────────────────────────────

class TestGenerateInsights:

    def test_empty_data_returns_need_more_message(self):
        insights = _generate_insights({}, [], [], None, pairs_used=2)
        assert len(insights) >= 1
        assert any("more day" in s.lower() or "more data" in s.lower() for s in insights)

    def test_meaningful_correlation_produces_insight(self):
        corr_dict = {
            "sleep_hours__next_day_fdi": 0.45,  # meaningful positive
        }
        sleep_buckets = [
            BucketStats("< 6h", 3, 0.65, 0.40, 55.0),
            BucketStats("6–7h", 4, 0.75, 0.30, 65.0),
            BucketStats("7–8h", 3, 0.85, 0.22, 75.0),
            BucketStats("≥ 8h", 2, 0.88, 0.18, 78.0),
        ]
        insights = _generate_insights(corr_dict, sleep_buckets, [], 0.08, pairs_used=12)
        assert len(insights) >= 1
        # First insight should mention slope or correlation
        assert any("hour" in s.lower() or "r=" in s for s in insights)

    def test_no_meaningful_correlations_returns_resilient_message(self):
        # Low r values, no meaningful buckets
        corr_dict = {
            "sleep_hours__next_day_fdi": 0.05,  # not meaningful
        }
        insights = _generate_insights(corr_dict, [], [], None, pairs_used=10)
        assert len(insights) >= 1

    def test_max_three_insights(self):
        corr_dict = {
            "sleep_hours__next_day_fdi": 0.50,
            "hrv_rmssd__next_day_fdi": 0.45,
            "recovery_score__next_day_fdi": 0.40,
        }
        sleep_buckets = [
            BucketStats("< 6h", 3, 0.60, 0.45, 50.0),
            BucketStats("7–8h", 5, 0.88, 0.20, 78.0),
        ]
        insights = _generate_insights(corr_dict, sleep_buckets, [], 0.10, pairs_used=15)
        assert len(insights) <= 3


# ─── Integration smoke test ───────────────────────────────────────────────────

class TestIntegrationSmoke:

    def test_runs_without_error_on_empty_store(self, monkeypatch):
        """Smoke test: should not raise even with zero data."""
        import analysis.sleep_focus_correlator as mod
        monkeypatch.setattr(mod, "list_available_dates", lambda: [])
        monkeypatch.setattr(mod, "read_summary", lambda: {"days": {}})

        corr = compute_sleep_focus_correlation("2026-01-01")
        assert corr is not None
        assert corr.is_meaningful is False
        # Must not raise
        line = format_sleep_insight_line(corr)
        section = format_sleep_insight_section(corr)
        assert isinstance(line, str)
        assert isinstance(section, str)

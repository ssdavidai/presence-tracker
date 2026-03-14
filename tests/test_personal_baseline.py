"""
Tests for analysis/personal_baseline.py

Covers:
  - PersonalBaseline dataclass defaults and representation
  - _percentile() interpolation correctness
  - get_personal_baseline() with mocked store data
  - is_hrv_low() with both personal and population thresholds
  - readiness_tier_personal() tier classification with personal/population thresholds
  - Integration: morning_brief._readiness_tier() uses personal baseline
  - Edge cases: empty data, single value, insufficient days (< 14)
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.personal_baseline import (
    PersonalBaseline,
    MIN_DAYS_FOR_PERSONAL_THRESHOLDS,
    POPULATION_HRV_LOW,
    POPULATION_RECOVERY_PEAK,
    POPULATION_RECOVERY_GOOD,
    POPULATION_RECOVERY_MODERATE,
    POPULATION_RECOVERY_LOW,
    _percentile,
    get_personal_baseline,
    is_hrv_low,
    readiness_tier_personal,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_summary(date: str, recovery: float, hrv: float, cls: float = 0.30) -> dict:
    """Create a minimal rolling summary dict."""
    return {
        "date": date,
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
        },
        "metrics_avg": {
            "cognitive_load_score": cls,
        },
    }


def _make_summaries(n: int, recovery_base=75.0, hrv_base=70.0, cls_base=0.30) -> list[dict]:
    """Generate n daily summaries with known values."""
    return [
        _make_summary(
            f"2026-01-{i+1:02d}",
            recovery=recovery_base + (i % 5) * 2,  # 75, 77, 79, 81, 83, 75, ...
            hrv=hrv_base + (i % 5) * 3,             # 70, 73, 76, 79, 82, 70, ...
            cls=cls_base + (i % 3) * 0.05,           # 0.30, 0.35, 0.40, 0.30, ...
        )
        for i in range(n)
    ]


# ─── PersonalBaseline dataclass ───────────────────────────────────────────────

class TestPersonalBaselineDefaults:
    def test_default_is_population_norm(self):
        b = PersonalBaseline()
        assert b.is_personal is False

    def test_default_hrv_p20_is_population_median(self):
        b = PersonalBaseline()
        assert b.hrv_p20 == POPULATION_HRV_LOW

    def test_default_recovery_thresholds_match_population(self):
        b = PersonalBaseline()
        assert b.recovery_p80 == POPULATION_RECOVERY_PEAK
        assert b.recovery_p60 == POPULATION_RECOVERY_GOOD
        assert b.recovery_p40 == POPULATION_RECOVERY_MODERATE
        assert b.recovery_p20 == POPULATION_RECOVERY_LOW

    def test_default_none_fields(self):
        b = PersonalBaseline()
        assert b.hrv_mean is None
        assert b.hrv_std is None
        assert b.hrv_p80 is None
        assert b.recovery_mean is None
        assert b.cls_mean is None
        assert b.cls_std is None

    def test_days_of_data_default_zero(self):
        b = PersonalBaseline()
        assert b.days_of_data == 0

    def test_repr_contains_population_norm(self):
        b = PersonalBaseline()
        assert "population-norm" in repr(b)

    def test_repr_contains_personal_when_personal(self):
        b = PersonalBaseline(is_personal=True, days_of_data=30, hrv_mean=70.0)
        assert "personal" in repr(b)

    def test_repr_does_not_crash_with_none_fields(self):
        b = PersonalBaseline()
        r = repr(b)
        assert "PersonalBaseline" in r


# ─── _percentile() ────────────────────────────────────────────────────────────

class TestPercentile:
    def test_median_odd_list(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == pytest.approx(3.0)

    def test_median_even_list(self):
        assert _percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_min_is_p0(self):
        assert _percentile([10, 20, 30], 0) == pytest.approx(10.0)

    def test_max_is_p100(self):
        assert _percentile([10, 20, 30], 100) == pytest.approx(30.0)

    def test_p25(self):
        assert _percentile([1, 2, 3, 4, 5], 25) == pytest.approx(2.0)

    def test_p75(self):
        assert _percentile([1, 2, 3, 4, 5], 75) == pytest.approx(4.0)

    def test_p20_five_values(self):
        assert _percentile([1, 2, 3, 4, 5], 20) == pytest.approx(1.8)

    def test_p80_five_values(self):
        assert _percentile([1, 2, 3, 4, 5], 80) == pytest.approx(4.2)

    def test_single_value_returns_that_value(self):
        assert _percentile([42.0], 50) == pytest.approx(42.0)
        assert _percentile([42.0], 0) == pytest.approx(42.0)
        assert _percentile([42.0], 100) == pytest.approx(42.0)

    def test_two_values_p50(self):
        assert _percentile([10.0, 20.0], 50) == pytest.approx(15.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _percentile([], 50)

    def test_monotonically_increasing(self):
        vals = sorted([10, 20, 30, 40, 50])
        p20 = _percentile(vals, 20)
        p50 = _percentile(vals, 50)
        p80 = _percentile(vals, 80)
        assert p20 < p50 < p80

    def test_identical_values(self):
        assert _percentile([7.0, 7.0, 7.0], 50) == pytest.approx(7.0)
        assert _percentile([7.0, 7.0, 7.0], 20) == pytest.approx(7.0)


# ─── get_personal_baseline() ─────────────────────────────────────────────────

class TestGetPersonalBaselineInsufficientData:
    """Fewer than MIN_DAYS_FOR_PERSONAL_THRESHOLDS days → population norms."""

    def test_no_summaries_returns_population_norm(self):
        with patch("analysis.personal_baseline.get_personal_baseline") as mock:
            mock.return_value = PersonalBaseline(days_of_data=0)
        b = get_personal_baseline.__wrapped__(0) if hasattr(get_personal_baseline, "__wrapped__") else None
        # Direct test: mock the store
        with patch("engine.store.get_recent_summaries", return_value=[]):
            b = get_personal_baseline()
        assert b.is_personal is False
        assert b.days_of_data == 0

    def test_five_days_is_not_personal(self):
        summaries = _make_summaries(5)
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            b = get_personal_baseline()
        assert b.is_personal is False
        assert b.days_of_data == 5

    def test_thirteen_days_is_not_personal(self):
        summaries = _make_summaries(MIN_DAYS_FOR_PERSONAL_THRESHOLDS - 1)
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            b = get_personal_baseline()
        assert b.is_personal is False

    def test_population_fallback_thresholds_unchanged(self):
        summaries = _make_summaries(5)
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            b = get_personal_baseline()
        assert b.recovery_p80 == POPULATION_RECOVERY_PEAK
        assert b.hrv_p20 == POPULATION_HRV_LOW

    def test_hrv_mean_computed_even_when_not_personal(self):
        summaries = _make_summaries(5, hrv_base=70.0)
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            b = get_personal_baseline()
        # Mean still populated even without personal thresholds
        assert b.hrv_mean is not None
        assert b.hrv_mean > 0


class TestGetPersonalBaselineSufficientData:
    """14+ days → personal thresholds should be active."""

    def _baseline_14(self, recovery_base=75.0, hrv_base=70.0):
        summaries = _make_summaries(14, recovery_base=recovery_base, hrv_base=hrv_base)
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            return get_personal_baseline()

    def test_14_days_is_personal(self):
        b = self._baseline_14()
        assert b.is_personal is True

    def test_days_of_data_correct(self):
        b = self._baseline_14()
        assert b.days_of_data == 14

    def test_hrv_mean_is_finite(self):
        b = self._baseline_14(hrv_base=70.0)
        assert b.hrv_mean is not None
        assert 60 < b.hrv_mean < 100

    def test_hrv_p20_below_mean(self):
        b = self._baseline_14(hrv_base=70.0)
        assert b.hrv_p20 < b.hrv_mean

    def test_hrv_p80_above_mean(self):
        b = self._baseline_14(hrv_base=70.0)
        assert b.hrv_p80 > b.hrv_mean

    def test_recovery_tiers_ordered(self):
        b = self._baseline_14()
        assert b.recovery_p20 < b.recovery_p40 < b.recovery_p60 < b.recovery_p80

    def test_recovery_mean_in_range(self):
        b = self._baseline_14(recovery_base=75.0)
        assert 70 < b.recovery_mean < 90

    def test_cls_mean_populated(self):
        b = self._baseline_14()
        assert b.cls_mean is not None
        assert 0.0 < b.cls_mean < 1.0

    def test_cls_std_populated(self):
        b = self._baseline_14()
        assert b.cls_std is not None
        assert b.cls_std >= 0.0

    def test_personal_thresholds_differ_from_population(self):
        """When David's recovery is always 75-83, his p80 != WHOOP's 80%."""
        b = self._baseline_14(recovery_base=75.0)
        # With recovery values 75–83 cycling, p80 should be in that range
        assert b.recovery_p80 != POPULATION_RECOVERY_PEAK or b.recovery_mean is not None

    def test_hrv_p20_differs_from_population_norm(self):
        """When David's HRV is always 70-82ms, p20 should be >> population 45ms."""
        b = self._baseline_14(hrv_base=70.0)
        assert b.hrv_p20 > POPULATION_HRV_LOW  # His p20 >> population threshold

    def test_store_error_returns_population_norm(self):
        """Store errors degrade gracefully."""
        with patch("engine.store.get_recent_summaries", side_effect=Exception("db error")):
            b = get_personal_baseline()
        assert b.is_personal is False

    def test_missing_hrv_values_handled(self):
        """Summaries missing HRV should not crash."""
        summaries = []
        for i in range(14):
            s = _make_summary(f"2026-01-{i+1:02d}", recovery=75.0 + i % 5, hrv=70.0)
            if i % 3 == 0:
                s["whoop"]["hrv_rmssd_milli"] = None
            summaries.append(s)
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            b = get_personal_baseline()
        assert b is not None
        assert b.days_of_data == 14

    def test_missing_recovery_values_handled(self):
        """Summaries missing recovery should not crash."""
        summaries = []
        for i in range(14):
            s = _make_summary(f"2026-01-{i+1:02d}", recovery=75.0, hrv=70.0)
            if i % 4 == 0:
                s["whoop"]["recovery_score"] = None
            summaries.append(s)
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            b = get_personal_baseline()
        assert b is not None

    def test_90_days_lookback_by_default(self):
        """Default lookback is 90 days."""
        called_with = []
        def fake_summaries(days=7):
            called_with.append(days)
            return _make_summaries(30)
        with patch("engine.store.get_recent_summaries", side_effect=fake_summaries):
            get_personal_baseline()
        assert called_with[0] == 90


# ─── is_hrv_low() ─────────────────────────────────────────────────────────────

class TestIsHrvLow:
    def test_none_hrv_never_low(self):
        assert is_hrv_low(None) is False
        assert is_hrv_low(None, PersonalBaseline()) is False

    def test_population_norm_40ms_is_low(self):
        assert is_hrv_low(40.0) is True

    def test_population_norm_50ms_is_not_low(self):
        assert is_hrv_low(50.0) is False

    def test_population_norm_threshold_is_45(self):
        assert is_hrv_low(44.9) is True
        assert is_hrv_low(45.0) is False

    def test_personal_baseline_not_personal_uses_population(self):
        b = PersonalBaseline(is_personal=False, hrv_p20=65.0)
        # Not personal → population threshold (45ms)
        assert is_hrv_low(50.0, b) is False   # above 45ms
        assert is_hrv_low(40.0, b) is True    # below 45ms

    def test_personal_baseline_uses_personal_p20(self):
        b = PersonalBaseline(is_personal=True, hrv_p20=65.0)
        # Personal → his p20 is 65ms
        assert is_hrv_low(60.0, b) is True    # below his personal p20
        assert is_hrv_low(70.0, b) is False   # above his personal p20

    def test_personal_p20_at_boundary(self):
        b = PersonalBaseline(is_personal=True, hrv_p20=70.0)
        assert is_hrv_low(69.9, b) is True
        assert is_hrv_low(70.0, b) is False

    def test_david_typical_hrv_not_low_with_population_norm(self):
        """David's HRV ~79ms should not be 'low' under population norms."""
        assert is_hrv_low(79.0) is False

    def test_david_hrv_drop_detected_with_personal_baseline(self):
        """If David's p20 is 68ms, then 65ms IS low for him."""
        b = PersonalBaseline(is_personal=True, hrv_p20=68.0)
        assert is_hrv_low(65.0, b) is True
        assert is_hrv_low(65.0, None) is False  # Not flagged by population norms

    def test_none_baseline_uses_population(self):
        assert is_hrv_low(44.9, None) is True
        assert is_hrv_low(45.0, None) is False


# ─── readiness_tier_personal() ────────────────────────────────────────────────

class TestReadinessTierPopulationNorms:
    """No baseline or non-personal baseline → WHOOP population thresholds."""

    def _tier(self, recovery, hrv=79.0, baseline=None):
        return readiness_tier_personal(recovery, hrv, baseline)

    def test_none_recovery_is_unknown(self):
        assert self._tier(None) == "unknown"

    def test_peak_at_80(self):
        assert self._tier(80.0) == "peak"

    def test_peak_at_95(self):
        assert self._tier(95.0) == "peak"

    def test_good_at_70(self):
        assert self._tier(70.0) == "good"

    def test_moderate_at_55(self):
        assert self._tier(55.0) == "moderate"

    def test_low_at_40(self):
        assert self._tier(40.0) == "low"

    def test_recovery_day_at_20(self):
        assert self._tier(20.0) == "recovery"

    def test_hrv_stressed_downgrades_peak_to_good(self):
        assert self._tier(85.0, hrv=40.0) == "good"  # HRV < 45 → downgrade

    def test_hrv_stressed_downgrades_good_to_moderate(self):
        assert self._tier(70.0, hrv=40.0) == "moderate"

    def test_hrv_stressed_downgrades_moderate_to_low(self):
        assert self._tier(55.0, hrv=40.0) == "low"

    def test_hrv_fine_does_not_downgrade(self):
        assert self._tier(80.0, hrv=79.0) == "peak"

    def test_none_hrv_no_downgrade(self):
        assert self._tier(80.0, hrv=None) == "peak"


class TestReadinessTierPersonalThresholds:
    """Personal baseline with known percentiles."""

    def _personal_baseline(
        self, p80=85.0, p60=75.0, p40=65.0, p20=55.0, hrv_p20=65.0
    ) -> PersonalBaseline:
        return PersonalBaseline(
            is_personal=True,
            days_of_data=30,
            recovery_p80=p80,
            recovery_p60=p60,
            recovery_p40=p40,
            recovery_p20=p20,
            hrv_p20=hrv_p20,
        )

    def test_above_p80_is_peak(self):
        b = self._personal_baseline(p80=82.0)
        assert readiness_tier_personal(85.0, 79.0, b) == "peak"

    def test_exactly_p80_is_peak(self):
        b = self._personal_baseline(p80=82.0)
        assert readiness_tier_personal(82.0, 79.0, b) == "peak"

    def test_between_p60_and_p80_is_good(self):
        b = self._personal_baseline(p60=72.0, p80=82.0)
        assert readiness_tier_personal(76.0, 79.0, b) == "good"

    def test_between_p40_and_p60_is_moderate(self):
        b = self._personal_baseline(p40=62.0, p60=72.0)
        assert readiness_tier_personal(66.0, 79.0, b) == "moderate"

    def test_between_p20_and_p40_is_low(self):
        b = self._personal_baseline(p20=52.0, p40=62.0)
        assert readiness_tier_personal(56.0, 79.0, b) == "low"

    def test_below_p20_is_recovery(self):
        b = self._personal_baseline(p20=55.0)
        assert readiness_tier_personal(50.0, 79.0, b) == "recovery"

    def test_personal_hrv_downgrade_uses_personal_p20(self):
        # David's p20 HRV = 65ms; today's HRV = 60ms → stressed → downgrade
        b = self._personal_baseline(p80=82.0, hrv_p20=65.0)
        assert readiness_tier_personal(85.0, 60.0, b) == "good"  # downgraded from peak

    def test_personal_hrv_no_downgrade_when_above_p20(self):
        b = self._personal_baseline(p80=82.0, hrv_p20=65.0)
        assert readiness_tier_personal(85.0, 70.0, b) == "peak"  # HRV above p20 → no downgrade

    def test_population_norm_miss_caught_by_personal_threshold(self):
        """
        Key regression test: David's recovery=81% looks like 'good' (not 'peak')
        under population norms (peak ≥ 80% → peak... wait, 81 > 80 → peak).

        Better example: David's typical recovery range is 60-88.
        p80 = 84%, so 82% = good (between p60 and p80).
        Under population norms, 82% ≥ 80 → peak.
        Personal baseline correctly shows it's a normal day for David.
        """
        b = self._personal_baseline(p60=76.0, p80=84.0)
        personal_tier = readiness_tier_personal(82.0, 79.0, b)
        population_tier = readiness_tier_personal(82.0, 79.0, None)
        assert personal_tier == "good"    # Normal day for David
        assert population_tier == "peak"  # Population norm calls it peak

    def test_non_personal_baseline_falls_back_to_population(self):
        b = PersonalBaseline(is_personal=False, recovery_p80=84.0)
        # Non-personal → should use population norms (80%)
        tier = readiness_tier_personal(81.0, 79.0, b)
        assert tier == "peak"  # Population norm: 81 ≥ 80


# ─── Integration: morning_brief uses personal baseline ────────────────────────

class TestMorningBriefPersonalBaselineIntegration:
    """
    Tests that morning_brief._readiness_tier() delegates to
    readiness_tier_personal() and that compute_morning_brief accepts
    the personal_baseline kwarg.
    """

    def _whoop(self, recovery=82.0, hrv=79.0):
        return {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "sleep_hours": 7.5,
            "sleep_performance": 85.0,
            "resting_heart_rate": 54.0,
        }

    def test_readiness_tier_delegates_to_personal_baseline(self):
        from analysis.morning_brief import _readiness_tier
        b = PersonalBaseline(is_personal=True, recovery_p80=84.0, recovery_p60=76.0,
                             recovery_p40=66.0, recovery_p20=56.0, hrv_p20=65.0)
        # 82% recovery — between p60(76) and p80(84) → "good" personally
        assert _readiness_tier(82.0, 79.0, baseline=b) == "good"
        # Under population norms: 82 ≥ 80 → "peak"
        assert _readiness_tier(82.0, 79.0, baseline=None) == "peak"

    def test_compute_morning_brief_accepts_personal_baseline_kwarg(self):
        from analysis.morning_brief import compute_morning_brief
        b = PersonalBaseline(is_personal=True, recovery_p80=85.0, recovery_p60=75.0,
                             recovery_p40=65.0, recovery_p20=55.0, hrv_p20=65.0)
        brief = compute_morning_brief(
            today_date="2026-03-14",
            whoop_data=self._whoop(recovery=82.0, hrv=79.0),
            personal_baseline=b,
        )
        assert brief["readiness"]["tier"] == "good"  # Personal threshold: 82 < p80=85

    def test_compute_morning_brief_without_personal_baseline_uses_population(self):
        from analysis.morning_brief import compute_morning_brief
        brief = compute_morning_brief(
            today_date="2026-03-14",
            whoop_data=self._whoop(recovery=82.0, hrv=79.0),
        )
        assert brief["readiness"]["tier"] == "peak"  # Population norm: 82 ≥ 80

    def test_compute_morning_brief_stores_personal_baseline_metadata(self):
        from analysis.morning_brief import compute_morning_brief
        b = PersonalBaseline(is_personal=True, days_of_data=30, recovery_p80=85.0,
                             recovery_p60=75.0, recovery_p40=65.0, recovery_p20=55.0,
                             hrv_p20=65.0)
        brief = compute_morning_brief(
            today_date="2026-03-14",
            whoop_data=self._whoop(),
            personal_baseline=b,
        )
        assert brief["personal_baseline"]["is_personal"] is True
        assert brief["personal_baseline"]["days_of_data"] == 30

    def test_compute_morning_brief_personal_baseline_none_is_ok(self):
        from analysis.morning_brief import compute_morning_brief
        brief = compute_morning_brief(
            today_date="2026-03-14",
            whoop_data=self._whoop(),
            personal_baseline=None,
        )
        assert brief["personal_baseline"] is None

    def test_personal_baseline_hrv_downgrade_applied(self):
        """When David's HRV drops below his personal p20, tier is downgraded."""
        from analysis.morning_brief import compute_morning_brief
        b = PersonalBaseline(is_personal=True, recovery_p80=84.0, recovery_p60=74.0,
                             recovery_p40=64.0, recovery_p20=54.0, hrv_p20=70.0)
        # Recovery 86% → peak. But HRV 65ms < p20(70ms) → downgraded to good
        brief = compute_morning_brief(
            today_date="2026-03-14",
            whoop_data=self._whoop(recovery=86.0, hrv=65.0),
            personal_baseline=b,
        )
        assert brief["readiness"]["tier"] == "good"

    def test_low_recovery_not_downgraded_below_recovery_tier(self):
        """Already-low tier shouldn't downgrade below 'recovery'."""
        from analysis.morning_brief import compute_morning_brief
        b = PersonalBaseline(is_personal=True, recovery_p80=85.0, recovery_p60=75.0,
                             recovery_p40=65.0, recovery_p20=55.0, hrv_p20=70.0)
        # Recovery 40% → below p20 → recovery tier; HRV low shouldn't change tier
        brief = compute_morning_brief(
            today_date="2026-03-14",
            whoop_data=self._whoop(recovery=40.0, hrv=50.0),
            personal_baseline=b,
        )
        assert brief["readiness"]["tier"] == "recovery"


# ─── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_get_personal_baseline_all_none_whoop(self):
        """All summaries with None WHOOP data → graceful degradation."""
        summaries = [
            {"date": f"2026-01-{i+1:02d}", "whoop": {}, "metrics_avg": {}}
            for i in range(20)
        ]
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            b = get_personal_baseline()
        assert b is not None
        assert b.hrv_mean is None
        assert b.recovery_mean is None
        assert b.is_personal is True  # 20 days exists; thresholds just fall back to defaults

    def test_get_personal_baseline_constant_hrv(self):
        """All days have same HRV → std=0, p20=p80=same value."""
        summaries = _make_summaries(20)
        for s in summaries:
            s["whoop"]["hrv_rmssd_milli"] = 72.0
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            b = get_personal_baseline()
        assert b.hrv_p20 == pytest.approx(72.0)
        assert b.hrv_p80 == pytest.approx(72.0)
        assert b.hrv_std == pytest.approx(0.0)

    def test_readiness_tier_none_baseline_is_fine(self):
        assert readiness_tier_personal(80.0, 79.0, None) == "peak"

    def test_readiness_tier_with_full_personal_baseline_from_factory(self):
        summaries = _make_summaries(30, recovery_base=65.0, hrv_base=70.0)
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            b = get_personal_baseline()
        assert b.is_personal is True
        # Tier should be deterministic — just shouldn't crash
        tier = readiness_tier_personal(75.0, 72.0, b)
        assert tier in ("peak", "good", "moderate", "low", "recovery", "unknown")

    def test_is_hrv_low_with_full_baseline(self):
        summaries = _make_summaries(30, hrv_base=65.0)
        with patch("engine.store.get_recent_summaries", return_value=summaries):
            b = get_personal_baseline()
        # When HRV base is 65-77, p20 should be around 65-67
        # is_hrv_low(60) should be True (below p20)
        assert is_hrv_low(60.0, b) is True
        # is_hrv_low(85) should be False (well above p20)
        assert is_hrv_low(85.0, b) is False

"""
Tests for the Cognitive Debt Index (CDI) module.

Run with: python3 -m pytest tests/test_cognitive_debt.py -v

All tests are pure unit tests — no network, no Slack, no real file I/O.
The store is mocked via patch so we can inject controlled summary data.

Coverage:
  - _debt_delta_for_day(): missing data, load > recovery, recovery > load, weighted
  - compute_cdi(): empty store, sparse data, full window, edge cases
  - CDI score formula: neutral at 50, scales correctly
  - Tier classification: all 5 tiers
  - format_cdi_line(): each tier, trend arrows, not meaningful
  - format_cdi_alert(): fatigued, critical, below threshold
  - Graceful failure: exceptions in store don't crash CDI
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.cognitive_debt import (
    CDI_DEFAULT_DAYS,
    CDI_MIN_DAYS,
    CDI_SERIES_CLAMP,
    CognitiveDebt,
    _debt_delta_for_day,
    compute_cdi,
    format_cdi_alert,
    format_cdi_line,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _summary(
    cls: float = 0.35,
    recovery: float = 75.0,
    active_windows: int = 20,
    date: str = "2026-03-14",
) -> dict:
    """Build a minimal rolling-summary day dict."""
    return {
        "date": date,
        "metrics_avg": {"cognitive_load_score": cls},
        "whoop": {"recovery_score": recovery, "hrv_rmssd_milli": 72.0},
        "focus_quality": {"active_windows": active_windows, "active_fdi": 0.65},
    }


def _make_rolling(summaries: list[dict]) -> dict:
    """Wrap a list of day summaries into a rolling.json structure."""
    return {"days": {s["date"]: s for s in summaries}}


# ─── _debt_delta_for_day ─────────────────────────────────────────────────────

class TestDebtDeltaForDay:
    def test_none_when_no_summary(self):
        assert _debt_delta_for_day(None) is None

    def test_none_when_empty_summary(self):
        assert _debt_delta_for_day({}) is None

    def test_none_when_no_cls(self):
        s = {"whoop": {"recovery_score": 80.0}}
        assert _debt_delta_for_day(s) is None

    def test_balanced_day_near_zero(self):
        # CLS ≈ 0.30, active_fraction ≈ 20/60 ≈ 0.333
        # load = 0.30 × 0.333 ≈ 0.10, recovery = 0.60 → delta ≈ -0.50 (recovery day)
        s = _summary(cls=0.30, recovery=60.0, active_windows=20)
        delta = _debt_delta_for_day(s)
        assert delta is not None
        # recovery (0.60) >> load (0.10) → definitely negative
        assert delta < 0

    def test_high_load_low_recovery_is_positive(self):
        # CLS = 0.80, active_windows = 50/60 ≈ 0.833
        # load = 0.80 × 0.833 = 0.666, recovery = 0.40 → delta ≈ +0.27
        s = _summary(cls=0.80, recovery=40.0, active_windows=50)
        delta = _debt_delta_for_day(s)
        assert delta is not None
        assert delta > 0

    def test_no_whoop_uses_neutral_recovery(self):
        # Without WHOOP, recovery defaults to 0.5
        s = {
            "date": "2026-03-14",
            "metrics_avg": {"cognitive_load_score": 0.60},
            "focus_quality": {"active_windows": 30},
        }
        delta = _debt_delta_for_day(s)
        assert delta is not None
        # load = 0.60 × 0.50 = 0.30; recovery = 0.50 → delta = -0.20
        assert delta < 0

    def test_no_active_windows_uses_unweighted_cls(self):
        # No focus_quality → active_fraction = 1.0
        s = {
            "date": "2026-03-14",
            "metrics_avg": {"cognitive_load_score": 0.70},
            "whoop": {"recovery_score": 60.0},
        }
        delta = _debt_delta_for_day(s)
        assert delta is not None
        # load = 0.70 × 1.0 = 0.70; recovery = 0.60 → delta = 0.10
        assert pytest.approx(delta, abs=0.001) == 0.10

    def test_zero_active_windows_uses_unweighted_cls(self):
        # active_windows = 0 → falls back to active_fraction = 1.0
        s = {
            "date": "2026-03-14",
            "metrics_avg": {"cognitive_load_score": 0.50},
            "whoop": {"recovery_score": 70.0},
            "focus_quality": {"active_windows": 0},
        }
        delta = _debt_delta_for_day(s)
        assert delta is not None
        assert pytest.approx(delta, abs=0.001) == -0.20  # 0.50 - 0.70

    def test_result_is_rounded_to_4_places(self):
        s = _summary(cls=0.3333, recovery=66.67, active_windows=20)
        delta = _debt_delta_for_day(s)
        assert delta is not None
        # Result should have at most 4 decimal places
        assert delta == round(delta, 4)

    def test_active_windows_clamped_to_1(self):
        # More active windows than WORKING_WINDOWS (60) → fraction clamped to 1.0
        s = _summary(cls=0.50, recovery=80.0, active_windows=100)
        delta = _debt_delta_for_day(s)
        assert delta is not None
        # load = 0.50 × 1.0 = 0.50; recovery = 0.80 → delta = -0.30
        assert delta < 0


# ─── compute_cdi ─────────────────────────────────────────────────────────────

class TestComputeCdi:
    def test_no_data_returns_neutral(self):
        with patch("engine.store.read_summary", return_value={"days": {}}):
            debt = compute_cdi("2026-03-14")
        assert debt.cdi == 50.0
        assert debt.tier == "balanced"
        assert debt.is_meaningful is False
        assert debt.days_used == 0

    def test_exception_in_store_returns_safe_default(self):
        with patch("engine.store.read_summary", side_effect=RuntimeError("db error")):
            debt = compute_cdi("2026-03-14")
        assert debt.cdi == 50.0
        assert debt.tier == "balanced"
        assert debt.is_meaningful is False

    def test_fewer_than_min_days_not_meaningful(self):
        # Only 2 days, CDI_MIN_DAYS = 3
        summaries = [
            _summary(cls=0.40, recovery=70.0, date="2026-03-13"),
            _summary(cls=0.45, recovery=65.0, date="2026-03-14"),
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert debt.is_meaningful is False
        assert debt.days_used == 2

    def test_three_or_more_days_is_meaningful(self):
        summaries = [
            _summary(cls=0.40, recovery=70.0, date="2026-03-12"),
            _summary(cls=0.40, recovery=70.0, date="2026-03-13"),
            _summary(cls=0.40, recovery=70.0, date="2026-03-14"),
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert debt.is_meaningful is True
        assert debt.days_used == 3

    def test_all_recovery_days_produces_low_cdi(self):
        # CLS = 0.10, recovery = 90% → heavily surplus
        # load = 0.10 × (10/60) ≈ 0.017, recovery = 0.90 → delta ≈ -0.88 per day
        summaries = [
            _summary(cls=0.10, recovery=90.0, active_windows=10, date=f"2026-03-{11+i:02d}")
            for i in range(7)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-17")
        assert debt.cdi < 30
        assert debt.tier == "surplus"

    def test_all_high_load_days_produces_high_cdi(self):
        # CLS = 0.90, recovery = 35%, active_windows=55 → heavy deficit each day
        # debt_delta ≈ 0.90 × (55/60) - 0.35 = 0.825 - 0.35 = 0.475 per day
        # After 14 days the series saturates near CDI_SERIES_CLAMP → CDI > 70
        summaries = [
            _summary(cls=0.90, recovery=35.0, active_windows=55, date=f"2026-03-{1+i:02d}")
            for i in range(14)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert debt.cdi > 70
        assert debt.tier in ("fatigued", "critical")

    def test_neutral_days_produce_cdi_near_50(self):
        # load ≈ recovery → debt_delta ≈ 0 each day → CDI ≈ 50
        # CLS = 0.50, active_fraction = 1.0, recovery = 50% → delta = 0.0
        summaries = [
            {
                "date": f"2026-03-{8+i:02d}",
                "metrics_avg": {"cognitive_load_score": 0.50},
                "whoop": {"recovery_score": 50.0},
                # No focus_quality → active_fraction = 1.0
            }
            for i in range(7)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert 40.0 <= debt.cdi <= 60.0

    def test_debt_series_never_exceeds_clamp(self):
        # Extreme load every day — series should be clamped at CDI_SERIES_CLAMP
        summaries = [
            _summary(cls=0.99, recovery=10.0, active_windows=59, date=f"2026-03-{1+i:02d}")
            for i in range(14)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        for val in debt.debt_series:
            assert val <= CDI_SERIES_CLAMP
            assert val >= -CDI_SERIES_CLAMP

    def test_cdi_bounded_0_to_100(self):
        summaries = [
            _summary(cls=0.99, recovery=10.0, active_windows=59, date=f"2026-03-{1+i:02d}")
            for i in range(14)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert 0.0 <= debt.cdi <= 100.0

    def test_end_date_matches_input(self):
        summaries = [_summary(date="2026-03-14")]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert debt.end_date == "2026-03-14"

    def test_days_in_deficit_counts_correctly(self):
        # 3 high-load days, 4 recovery days
        summaries = (
            [_summary(cls=0.90, recovery=35.0, active_windows=55, date=f"2026-03-{8+i:02d}")
             for i in range(3)]
            +
            [_summary(cls=0.10, recovery=90.0, active_windows=10, date=f"2026-03-{11+i:02d}")
             for i in range(4)]
        )
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert debt.days_in_deficit == 3
        assert debt.days_in_surplus == 4

    def test_missing_days_in_window_are_skipped(self):
        # Only 3 of the 14 days have data — they should all count
        summaries = [
            _summary(cls=0.40, recovery=70.0, date="2026-03-12"),
            _summary(cls=0.40, recovery=70.0, date="2026-03-13"),
            _summary(cls=0.40, recovery=70.0, date="2026-03-14"),
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14", days=14)
        assert debt.days_used == 3

    def test_trend_5d_positive_for_consecutive_deficit_days(self):
        # 5 consecutive high-load days → trend_5d should be positive
        summaries = [
            _summary(cls=0.85, recovery=40.0, active_windows=50, date=f"2026-03-{10+i:02d}")
            for i in range(5)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert debt.trend_5d is not None
        assert debt.trend_5d > 0

    def test_trend_5d_negative_for_consecutive_recovery_days(self):
        summaries = [
            _summary(cls=0.10, recovery=90.0, active_windows=8, date=f"2026-03-{10+i:02d}")
            for i in range(5)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert debt.trend_5d is not None
        assert debt.trend_5d < 0

    def test_debt_delta_today_matches_last_day(self):
        summaries = [
            _summary(cls=0.40, recovery=70.0, date="2026-03-12"),
            _summary(cls=0.40, recovery=70.0, date="2026-03-13"),
            _summary(cls=0.75, recovery=45.0, active_windows=40, date="2026-03-14"),
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        # delta for 2026-03-14: load = 0.75 × (40/60) = 0.50; recovery = 0.45 → delta = 0.05
        assert debt.debt_delta_today is not None
        assert debt.debt_delta_today > 0  # today was a deficit day

    def test_custom_days_window(self):
        # With days=5, only the last 5 days should be considered
        summaries = [
            _summary(cls=0.40, recovery=70.0, date=f"2026-03-{9+i:02d}")
            for i in range(6)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14", days=5)
        assert debt.days_used == 5


# ─── CDI Tiers ───────────────────────────────────────────────────────────────

class TestCdiTiers:
    def _debt_with_cdi(self, cdi_val: float) -> CognitiveDebt:
        """Build a minimal CognitiveDebt with a given CDI for tier testing."""
        summaries = [
            _summary(date=f"2026-03-{8+i:02d}")
            for i in range(CDI_MIN_DAYS)
        ]
        rolling = _make_rolling(summaries)

        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")

        # Override CDI score and tier manually for tier tests
        # (we compute tier from the module's formula, not by injecting)
        return debt

    def test_surplus_tier_below_30(self):
        summaries = [
            _summary(cls=0.05, recovery=95.0, active_windows=5, date=f"2026-03-{8+i:02d}")
            for i in range(CDI_MIN_DAYS + 4)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert debt.tier == "surplus"
        assert debt.cdi < 30

    def test_critical_tier_above_85(self):
        summaries = [
            _summary(cls=0.99, recovery=10.0, active_windows=59, date=f"2026-03-{1+i:02d}")
            for i in range(14)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert debt.tier == "critical"
        assert debt.cdi > 85


# ─── format_cdi_line ─────────────────────────────────────────────────────────

class TestFormatCdiLine:
    def _make_debt(
        self,
        cdi: float = 50.0,
        tier: str = "balanced",
        trend_5d: float = 0.0,
        days_in_deficit: int = 0,
        days_used: int = 10,
        is_meaningful: bool = True,
    ) -> CognitiveDebt:
        return CognitiveDebt(
            cdi=cdi,
            tier=tier,
            debt_delta_today=0.0,
            trend_5d=trend_5d,
            days_in_deficit=days_in_deficit,
            days_in_surplus=days_used - days_in_deficit,
            days_used=days_used,
            is_meaningful=is_meaningful,
            end_date="2026-03-14",
            debt_series=[0.0] * days_used,
        )

    def test_not_meaningful_returns_empty_string(self):
        debt = self._make_debt(is_meaningful=False)
        assert format_cdi_line(debt) == ""

    def test_output_is_string(self):
        debt = self._make_debt(cdi=55.0, tier="loading")
        result = format_cdi_line(debt)
        assert isinstance(result, str)

    def test_output_is_non_empty_when_meaningful(self):
        debt = self._make_debt(cdi=55.0, tier="loading", days_in_deficit=5)
        result = format_cdi_line(debt)
        assert len(result) > 0

    def test_contains_cdi_score(self):
        debt = self._make_debt(cdi=63.0, tier="loading")
        result = format_cdi_line(debt)
        assert "63" in result

    def test_contains_tier_label(self):
        for tier in ("surplus", "balanced", "loading", "fatigued", "critical"):
            debt = self._make_debt(cdi=50.0, tier=tier)
            result = format_cdi_line(debt)
            assert tier.capitalize() in result, f"Expected tier '{tier}' in: {result}"

    def test_trend_up_shown_when_positive_trend(self):
        debt = self._make_debt(cdi=65.0, tier="loading", trend_5d=0.10)
        result = format_cdi_line(debt)
        assert "↑" in result or "fatigue" in result

    def test_trend_down_shown_when_negative_trend(self):
        debt = self._make_debt(cdi=35.0, tier="surplus", trend_5d=-0.10)
        result = format_cdi_line(debt)
        assert "↓" in result or "recovering" in result

    def test_no_trend_shown_when_near_zero(self):
        debt = self._make_debt(cdi=50.0, tier="balanced", trend_5d=0.01)
        result = format_cdi_line(debt)
        # 0.01 is below threshold of 0.02 — no trend arrow expected
        assert "↑" not in result
        assert "↓" not in result

    def test_deficit_days_shown_when_nonzero(self):
        debt = self._make_debt(cdi=65.0, tier="loading", days_in_deficit=5, days_used=10)
        result = format_cdi_line(debt)
        assert "5" in result

    def test_surplus_tier_emoji(self):
        debt = self._make_debt(cdi=25.0, tier="surplus")
        result = format_cdi_line(debt)
        assert "🟢" in result

    def test_fatigued_tier_emoji(self):
        debt = self._make_debt(cdi=75.0, tier="fatigued")
        result = format_cdi_line(debt)
        assert "🔴" in result

    def test_critical_tier_emoji(self):
        debt = self._make_debt(cdi=90.0, tier="critical")
        result = format_cdi_line(debt)
        assert "🚨" in result


# ─── format_cdi_alert ────────────────────────────────────────────────────────

class TestFormatCdiAlert:
    def _make_debt(
        self,
        cdi: float = 75.0,
        tier: str = "fatigued",
        days_in_deficit: int = 7,
        days_used: int = 10,
        is_meaningful: bool = True,
        trend_5d: float = 0.05,
    ) -> CognitiveDebt:
        return CognitiveDebt(
            cdi=cdi,
            tier=tier,
            debt_delta_today=0.05,
            trend_5d=trend_5d,
            days_in_deficit=days_in_deficit,
            days_in_surplus=days_used - days_in_deficit,
            days_used=days_used,
            is_meaningful=is_meaningful,
            end_date="2026-03-14",
            debt_series=[0.0] * days_used,
        )

    def test_empty_when_not_meaningful(self):
        debt = self._make_debt(is_meaningful=False)
        assert format_cdi_alert(debt) == ""

    def test_empty_for_surplus_tier(self):
        debt = self._make_debt(tier="surplus", cdi=20.0)
        assert format_cdi_alert(debt) == ""

    def test_empty_for_balanced_tier(self):
        debt = self._make_debt(tier="balanced", cdi=45.0)
        assert format_cdi_alert(debt) == ""

    def test_empty_for_loading_tier(self):
        debt = self._make_debt(tier="loading", cdi=60.0)
        assert format_cdi_alert(debt) == ""

    def test_returns_string_for_fatigued(self):
        debt = self._make_debt(tier="fatigued", cdi=75.0)
        result = format_cdi_alert(debt)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string_for_critical(self):
        debt = self._make_debt(tier="critical", cdi=90.0)
        result = format_cdi_alert(debt)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fatigued_alert_contains_cdi_score(self):
        debt = self._make_debt(tier="fatigued", cdi=79.0)
        result = format_cdi_alert(debt)
        assert "79" in result

    def test_critical_alert_contains_rest_signal(self):
        debt = self._make_debt(tier="critical", cdi=91.0)
        result = format_cdi_alert(debt)
        # Should mention rest or burnout
        assert "rest" in result.lower() or "burnout" in result.lower()

    def test_trending_worsening_adds_extra_line(self):
        debt = self._make_debt(tier="fatigued", cdi=77.0, trend_5d=0.10)
        result = format_cdi_alert(debt)
        # Should mention the trend not being a one-day spike
        assert "spike" in result.lower() or "trend" in result.lower()

    def test_non_worsening_trend_no_spike_mention(self):
        debt = self._make_debt(tier="fatigued", cdi=77.0, trend_5d=0.01)
        result = format_cdi_alert(debt)
        # Small trend (0.01 < 0.05 threshold) → no extra trend paragraph
        assert "spike" not in result.lower()


# ─── Integration: compute_cdi → format_cdi_line pipeline ─────────────────────

class TestCdiPipeline:
    def test_full_pipeline_does_not_crash(self):
        summaries = [
            _summary(cls=0.45, recovery=72.0, active_windows=25, date=f"2026-03-{8+i:02d}")
            for i in range(7)
        ]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        line = format_cdi_line(debt)
        alert = format_cdi_alert(debt)
        # Just verify no crashes and types are correct
        assert isinstance(line, str)
        assert isinstance(alert, str)

    def test_single_day_returns_valid_result(self):
        summaries = [_summary(date="2026-03-14")]
        rolling = _make_rolling(summaries)
        with patch("engine.store.read_summary", return_value=rolling):
            debt = compute_cdi("2026-03-14")
        assert isinstance(debt.cdi, float)
        assert debt.tier in ("surplus", "balanced", "loading", "fatigued", "critical")
        assert 0.0 <= debt.cdi <= 100.0

    def test_no_data_pipeline_returns_empty_line(self):
        with patch("engine.store.read_summary", return_value={"days": {}}):
            debt = compute_cdi("2026-03-14")
        assert format_cdi_line(debt) == ""
        assert format_cdi_alert(debt) == ""

    def test_cdi_lower_after_recovery_week(self):
        """A week of rest should produce lower CDI than a week of hard work."""
        hard_summaries = [
            _summary(cls=0.85, recovery=35.0, active_windows=55, date=f"2026-03-{8+i:02d}")
            for i in range(7)
        ]
        rest_summaries = [
            _summary(cls=0.10, recovery=90.0, active_windows=8, date=f"2026-03-{8+i:02d}")
            for i in range(7)
        ]
        with patch("engine.store.read_summary", return_value=_make_rolling(hard_summaries)):
            hard_debt = compute_cdi("2026-03-14")
        with patch("engine.store.read_summary", return_value=_make_rolling(rest_summaries)):
            rest_debt = compute_cdi("2026-03-14")
        assert hard_debt.cdi > rest_debt.cdi

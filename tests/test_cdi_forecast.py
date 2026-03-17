"""
Tests for analysis/cdi_forecast.py — CDI Trajectory Forecast

Coverage:
  1. _cdi_tier()
     - Returns correct tier for each CDI band (surplus / balanced / loading / fatigued / critical)
     - Boundary values handled correctly

  2. _mean()
     - Returns None for empty list
     - Returns correct mean for non-empty list

  3. _extract_recovery_signal()
     - Returns fallback 0.60 when no WHOOP data in rolling
     - Computes correct mean from multiple days' WHOOP recovery
     - Clamps to [RECOVERY_SIGNAL_MIN, RECOVERY_SIGNAL_MAX]

  4. _extract_load_signal()
     - Returns fallback 0.15 when no data
     - Computes avg_cls × active_fraction correctly
     - Falls back to DEFAULT_ACTIVE_FRACTION when active_windows unavailable

  5. _project_cdi()
     - Positive delta → CDI rises over time
     - Negative delta → CDI falls over time
     - Running sum clamped to ±CDI_SERIES_CLAMP (never produces CDI > 100 or < 0)
     - Returns correct number of projected days

  6. _detect_days_to_fatigued()
     - Returns None when today_cdi already ≥ 70 (already fatigued)
     - Returns correct day index when CDI crosses 70
     - Returns None when CDI never crosses 70 in projection window

  7. _detect_days_to_recovery()
     - Returns None when today_cdi already ≤ 50 (already recovered)
     - Returns correct day index when CDI drops to ≤ 50
     - Returns None when CDI never recovers in projection window

  8. _detect_trend()
     - 'worsening' when end CDI ≥ start + 5
     - 'improving' when end CDI ≤ start − 5
     - 'stable' when within ±5

  9. _build_headline()
     - fatigued + improving → mentions recovery
     - fatigued + worsening → warning in headline
     - loading + days_to_fatigued → mentions days to fatigued
     - balanced + stable → "stable" in headline

  10. compute_cdi_forecast()
      - Returns is_meaningful=False when insufficient history (mocked store)
      - Returns is_meaningful=True with enough history
      - projected_cdis has exactly FORECAST_DAYS entries
      - projected_tiers match projected_cdis via _cdi_tier()
      - trend_direction is one of 'worsening' | 'stable' | 'improving'
      - days_to_fatigued / days_to_recovery are mutually exclusive

  11. format_cdi_forecast_line()
      - Returns empty string when is_meaningful=False
      - Returns non-empty string when is_meaningful=True
      - Contains sparkline characters

  12. format_cdi_forecast_section()
      - Returns empty string when is_meaningful=False
      - Returns multi-line string when is_meaningful=True
      - Contains "CDI Trajectory Forecast" header
      - Lists all FORECAST_DAYS entries

Run with: python3 -m pytest tests/test_cdi_forecast.py -v
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.cdi_forecast import (
    CDIForecast,
    FORECAST_DAYS,
    CDI_SERIES_CLAMP,
    DEFAULT_ACTIVE_FRACTION,
    RECOVERY_SIGNAL_MIN,
    RECOVERY_SIGNAL_MAX,
    _cdi_tier,
    _mean,
    _extract_recovery_signal,
    _extract_load_signal,
    _project_cdi,
    _detect_days_to_fatigued,
    _detect_days_to_recovery,
    _detect_trend,
    _build_headline,
    _sparkline,
    compute_cdi_forecast,
    format_cdi_forecast_line,
    format_cdi_forecast_section,
)


# ─── Tier tests ───────────────────────────────────────────────────────────────

class TestCdiTier:
    def test_surplus_below_30(self):
        assert _cdi_tier(20.0) == "surplus"
        assert _cdi_tier(0.0) == "surplus"
        assert _cdi_tier(29.9) == "surplus"

    def test_balanced_30_to_50(self):
        assert _cdi_tier(30.0) == "balanced"
        assert _cdi_tier(50.0) == "balanced"
        assert _cdi_tier(45.0) == "balanced"

    def test_loading_50_to_70(self):
        assert _cdi_tier(50.1) == "loading"
        assert _cdi_tier(70.0) == "loading"
        assert _cdi_tier(60.0) == "loading"

    def test_fatigued_70_to_85(self):
        assert _cdi_tier(70.1) == "fatigued"
        assert _cdi_tier(85.0) == "fatigued"

    def test_critical_above_85(self):
        assert _cdi_tier(85.1) == "critical"
        assert _cdi_tier(100.0) == "critical"


# ─── _mean tests ─────────────────────────────────────────────────────────────

class TestMean:
    def test_empty_returns_none(self):
        assert _mean([]) is None

    def test_single_value(self):
        assert _mean([0.6]) == pytest.approx(0.6)

    def test_multiple_values(self):
        assert _mean([0.4, 0.6, 0.8]) == pytest.approx(0.6)


# ─── _extract_recovery_signal tests ──────────────────────────────────────────

class TestExtractRecoverySignal:
    def _make_rolling(self, dates_recoveries: dict) -> dict:
        """Build a minimal rolling summary dict."""
        days = {}
        for d, rec in dates_recoveries.items():
            days[d] = {"whoop": {"recovery_score": rec}} if rec is not None else {}
        return {"days": days}

    def test_no_whoop_data_returns_fallback(self):
        rolling = {"days": {"2026-03-01": {}, "2026-03-02": {}}}
        result = _extract_recovery_signal(rolling, ["2026-03-01", "2026-03-02"])
        assert result == pytest.approx(0.60)

    def test_correct_mean(self):
        rolling = self._make_rolling({"2026-03-01": 80, "2026-03-02": 60})
        result = _extract_recovery_signal(rolling, ["2026-03-01", "2026-03-02"])
        # mean = (0.80 + 0.60) / 2 = 0.70
        assert result == pytest.approx(0.70, rel=1e-3)

    def test_clamped_to_min(self):
        rolling = self._make_rolling({"2026-03-01": 20})  # 0.20 < RECOVERY_SIGNAL_MIN
        result = _extract_recovery_signal(rolling, ["2026-03-01"])
        assert result == pytest.approx(RECOVERY_SIGNAL_MIN)

    def test_clamped_to_max(self):
        rolling = self._make_rolling({"2026-03-01": 99})  # 0.99 > RECOVERY_SIGNAL_MAX
        result = _extract_recovery_signal(rolling, ["2026-03-01"])
        assert result == pytest.approx(RECOVERY_SIGNAL_MAX)


# ─── _extract_load_signal tests ───────────────────────────────────────────────

class TestExtractLoadSignal:
    def _make_rolling(self, days_data: dict) -> dict:
        return {"days": days_data}

    def test_no_data_returns_fallback(self):
        rolling = {"days": {}}
        result = _extract_load_signal(rolling, ["2026-03-01"])
        assert result == pytest.approx(0.15)

    def test_uses_active_fraction_when_available(self):
        rolling = self._make_rolling({
            "2026-03-01": {
                "metrics_avg": {"cognitive_load_score": 0.50},
                "focus_quality": {"active_windows": 30},  # 30/60 = 0.5
            }
        })
        result = _extract_load_signal(rolling, ["2026-03-01"])
        # load = 0.50 * 0.50 = 0.25
        assert result == pytest.approx(0.25, rel=1e-3)

    def test_falls_back_to_default_fraction(self):
        rolling = self._make_rolling({
            "2026-03-01": {
                "metrics_avg": {"cognitive_load_score": 0.40},
                "focus_quality": {},  # no active_windows
            }
        })
        result = _extract_load_signal(rolling, ["2026-03-01"])
        # load = 0.40 * DEFAULT_ACTIVE_FRACTION
        assert result == pytest.approx(0.40 * DEFAULT_ACTIVE_FRACTION, rel=1e-3)

    def test_averages_multiple_days(self):
        rolling = self._make_rolling({
            "2026-03-01": {
                "metrics_avg": {"cognitive_load_score": 0.20},
                "focus_quality": {"active_windows": 60},  # fraction=1.0
            },
            "2026-03-02": {
                "metrics_avg": {"cognitive_load_score": 0.40},
                "focus_quality": {"active_windows": 60},
            },
        })
        result = _extract_load_signal(rolling, ["2026-03-01", "2026-03-02"])
        # day1: 0.20*1.0=0.20, day2: 0.40*1.0=0.40 → mean=0.30
        assert result == pytest.approx(0.30, rel=1e-3)


# ─── _project_cdi tests ───────────────────────────────────────────────────────

class TestProjectCdi:
    def test_positive_delta_raises_cdi(self):
        base = datetime(2026, 3, 14)
        cdis, dates, tiers = _project_cdi(0.0, 1.0, base, forecast_days=3)
        # running sums: 1, 2, 3 → CDIs above 50
        assert all(c > 50.0 for c in cdis)
        assert len(cdis) == 3

    def test_negative_delta_lowers_cdi(self):
        base = datetime(2026, 3, 14)
        cdis, dates, tiers = _project_cdi(0.0, -1.0, base, forecast_days=3)
        assert all(c < 50.0 for c in cdis)

    def test_cdi_never_exceeds_100(self):
        base = datetime(2026, 3, 14)
        # Start at max clamp, positive delta — should stay at 100
        cdis, _, _ = _project_cdi(CDI_SERIES_CLAMP, 10.0, base, forecast_days=5)
        assert all(c <= 100.0 for c in cdis)

    def test_cdi_never_below_0(self):
        base = datetime(2026, 3, 14)
        cdis, _, _ = _project_cdi(-CDI_SERIES_CLAMP, -10.0, base, forecast_days=5)
        assert all(c >= 0.0 for c in cdis)

    def test_correct_number_of_days(self):
        base = datetime(2026, 3, 14)
        cdis, dates, tiers = _project_cdi(0.0, 0.1, base, forecast_days=FORECAST_DAYS)
        assert len(cdis) == FORECAST_DAYS
        assert len(dates) == FORECAST_DAYS
        assert len(tiers) == FORECAST_DAYS

    def test_dates_are_consecutive(self):
        base = datetime(2026, 3, 14)
        _, dates, _ = _project_cdi(0.0, 0.0, base, forecast_days=3)
        expected = ["2026-03-15", "2026-03-16", "2026-03-17"]
        assert dates == expected


# ─── _detect_days_to_fatigued tests ───────────────────────────────────────────

class TestDetectDaysToFatigued:
    def test_already_fatigued_returns_none(self):
        assert _detect_days_to_fatigued(75.0, [76.0, 77.0, 78.0]) is None

    def test_crosses_threshold_day_2(self):
        # today=55, day1=65, day2=72 → crosses 70 on day 2
        result = _detect_days_to_fatigued(55.0, [65.0, 72.0, 74.0, 76.0, 78.0])
        assert result == 2

    def test_crosses_threshold_day_1(self):
        result = _detect_days_to_fatigued(65.0, [71.0, 73.0, 75.0, 77.0, 79.0])
        assert result == 1

    def test_never_crosses_returns_none(self):
        result = _detect_days_to_fatigued(50.0, [52.0, 54.0, 56.0, 58.0, 60.0])
        assert result is None


# ─── _detect_days_to_recovery tests ───────────────────────────────────────────

class TestDetectDaysToRecovery:
    def test_already_recovered_returns_none(self):
        assert _detect_days_to_recovery(45.0, [44.0, 43.0]) is None

    def test_crosses_threshold_day_3(self):
        # today=65, decreasing, crosses 50 on day 3
        result = _detect_days_to_recovery(65.0, [60.0, 55.0, 50.0, 48.0, 46.0])
        assert result == 3

    def test_crosses_threshold_day_1(self):
        result = _detect_days_to_recovery(55.0, [49.0, 47.0, 45.0, 43.0, 41.0])
        assert result == 1

    def test_never_recovers_returns_none(self):
        result = _detect_days_to_recovery(70.0, [68.0, 66.0, 64.0, 62.0, 60.0])
        assert result is None


# ─── _detect_trend tests ─────────────────────────────────────────────────────

class TestDetectTrend:
    def test_worsening_when_end_much_higher(self):
        # today=50, end=60 → delta=10 → worsening
        assert _detect_trend(50.0, [52.0, 54.0, 56.0, 58.0, 60.0]) == "worsening"

    def test_improving_when_end_much_lower(self):
        assert _detect_trend(70.0, [68.0, 66.0, 64.0, 62.0, 60.0]) == "improving"

    def test_stable_within_5(self):
        # today=50, end=54 → delta=4 → stable
        assert _detect_trend(50.0, [50.5, 51.0, 52.0, 53.0, 54.0]) == "stable"

    def test_stable_empty_projections(self):
        assert _detect_trend(50.0, []) == "stable"

    def test_exactly_5_delta_is_worsening(self):
        assert _detect_trend(50.0, [55.0]) == "worsening"

    def test_exactly_minus_5_delta_is_improving(self):
        assert _detect_trend(50.0, [45.0]) == "improving"


# ─── _build_headline tests ────────────────────────────────────────────────────

class TestBuildHeadline:
    def test_fatigued_with_recovery_path(self):
        headline = _build_headline(75.0, "fatigued", "improving", None, 2, [73.0, 71.0, 68.0, 55.0, 48.0])
        assert "balanced" in headline.lower() or "2 day" in headline

    def test_fatigued_worsening_has_warning(self):
        headline = _build_headline(75.0, "fatigued", "worsening", None, None, [77.0, 79.0, 81.0, 83.0, 85.0])
        assert "⚠️" in headline or "rising" in headline.lower() or "compounding" in headline.lower()

    def test_loading_days_to_fatigued(self):
        headline = _build_headline(62.0, "loading", "worsening", 3, None, [64.0, 67.0, 71.0, 73.0, 75.0])
        assert "3 day" in headline or "fatigued" in headline.lower()

    def test_balanced_stable(self):
        headline = _build_headline(45.0, "balanced", "stable", None, None, [45.0, 46.0, 47.0, 46.0, 45.0])
        assert "stable" in headline.lower()

    def test_surplus_worsening(self):
        headline = _build_headline(25.0, "surplus", "worsening", None, None, [30.0, 35.0, 40.0, 45.0, 50.0])
        assert "trending" in headline.lower() or "cdi" in headline.lower()


# ─── _sparkline tests ────────────────────────────────────────────────────────

class TestSparkline:
    def test_length_matches_input(self):
        cdis = [30.0, 50.0, 70.0, 85.0, 95.0]
        result = _sparkline(cdis)
        assert len(result) == 5

    def test_low_cdi_gives_low_bar(self):
        # 0 CDI → first bar character (space)
        result = _sparkline([0.0])
        assert result in " ▁"

    def test_high_cdi_gives_high_bar(self):
        result = _sparkline([100.0])
        assert result == "█"


# ─── compute_cdi_forecast integration tests ──────────────────────────────────

class TestComputeCdiForecast:
    """Integration tests with mocked store and CDI."""

    def _make_debt(self, cdi=55.0, tier="loading", is_meaningful=True, trend_5d=0.05):
        """Build a mock CognitiveDebt object."""
        debt = MagicMock()
        debt.cdi = cdi
        debt.tier = tier
        debt.is_meaningful = is_meaningful
        debt.trend_5d = trend_5d
        debt.debt_series = [cdi / 50.0 - 1.0] * 7  # rough back-calc
        return debt

    def _make_rolling(self, dates: list[str], recovery: float = 75.0, avg_cls: float = 0.35):
        """Build a minimal rolling summary for the given dates."""
        days = {}
        for d in dates:
            days[d] = {
                "whoop": {"recovery_score": recovery},
                "metrics_avg": {"cognitive_load_score": avg_cls},
                "focus_quality": {"active_windows": 30},
            }
        return {"days": days}

    @patch("analysis.cdi_forecast.read_summary")
    @patch("analysis.cdi_forecast.list_available_dates")
    @patch("analysis.cdi_forecast.compute_cdi", create=True)
    def test_not_meaningful_when_insufficient_history(
        self, mock_compute_cdi, mock_dates, mock_summary
    ):
        from analysis.cdi_forecast import compute_cdi_forecast
        # Patch the import inside the function
        mock_compute_cdi.return_value = self._make_debt()
        mock_dates.return_value = ["2026-03-13", "2026-03-14"]  # only 2 days
        mock_summary.return_value = self._make_rolling(["2026-03-13", "2026-03-14"])

        with patch("analysis.cognitive_debt.compute_cdi", mock_compute_cdi):
            result = compute_cdi_forecast("2026-03-15")

        # 2 days < MIN_DAYS_FOR_FORECAST=3 → not meaningful
        assert not result.is_meaningful

    @patch("analysis.cdi_forecast._get_load_forecast_for_date", return_value=None)
    @patch("analysis.cdi_forecast.read_summary")
    @patch("analysis.cdi_forecast.list_available_dates")
    def test_meaningful_with_enough_history(
        self, mock_dates, mock_summary, mock_forecast
    ):
        from analysis.cdi_forecast import compute_cdi_forecast

        dates = [f"2026-03-{i:02d}" for i in range(7, 15)]  # 8 days
        mock_dates.return_value = dates
        mock_summary.return_value = self._make_rolling(dates)

        debt = self._make_debt(cdi=55.0, tier="loading")

        with patch("analysis.cognitive_debt.compute_cdi", return_value=debt):
            result = compute_cdi_forecast("2026-03-15")

        assert result.is_meaningful
        assert len(result.projected_cdis) == FORECAST_DAYS
        assert len(result.projected_dates) == FORECAST_DAYS
        assert len(result.projected_tiers) == FORECAST_DAYS

    @patch("analysis.cdi_forecast._get_load_forecast_for_date", return_value=None)
    @patch("analysis.cdi_forecast.read_summary")
    @patch("analysis.cdi_forecast.list_available_dates")
    def test_projected_tiers_match_cdis(
        self, mock_dates, mock_summary, mock_forecast
    ):
        from analysis.cdi_forecast import compute_cdi_forecast

        dates = [f"2026-03-{i:02d}" for i in range(1, 10)]
        mock_dates.return_value = dates
        mock_summary.return_value = self._make_rolling(dates)

        debt = self._make_debt(cdi=65.0, tier="loading")
        with patch("analysis.cognitive_debt.compute_cdi", return_value=debt):
            result = compute_cdi_forecast("2026-03-10")

        for cdi, tier in zip(result.projected_cdis, result.projected_tiers):
            assert tier == _cdi_tier(cdi)

    @patch("analysis.cdi_forecast._get_load_forecast_for_date", return_value=None)
    @patch("analysis.cdi_forecast.read_summary")
    @patch("analysis.cdi_forecast.list_available_dates")
    def test_trend_direction_is_valid(
        self, mock_dates, mock_summary, mock_forecast
    ):
        from analysis.cdi_forecast import compute_cdi_forecast

        dates = [f"2026-03-{i:02d}" for i in range(1, 10)]
        mock_dates.return_value = dates
        mock_summary.return_value = self._make_rolling(dates)

        debt = self._make_debt()
        with patch("analysis.cognitive_debt.compute_cdi", return_value=debt):
            result = compute_cdi_forecast("2026-03-10")

        assert result.trend_direction in ("worsening", "stable", "improving")

    @patch("analysis.cdi_forecast._get_load_forecast_for_date", return_value=None)
    @patch("analysis.cdi_forecast.read_summary")
    @patch("analysis.cdi_forecast.list_available_dates")
    def test_fatigued_and_recovery_mutually_exclusive_when_fatigued(
        self, mock_dates, mock_summary, mock_forecast
    ):
        """
        When CDI is already fatigued, days_to_fatigued must be None.
        """
        from analysis.cdi_forecast import compute_cdi_forecast

        dates = [f"2026-03-{i:02d}" for i in range(1, 10)]
        mock_dates.return_value = dates
        # High recovery to make CDI improving
        mock_summary.return_value = self._make_rolling(dates, recovery=90.0, avg_cls=0.10)

        debt = self._make_debt(cdi=80.0, tier="fatigued")
        with patch("analysis.cognitive_debt.compute_cdi", return_value=debt):
            result = compute_cdi_forecast("2026-03-10")

        # Already fatigued → days_to_fatigued should be None
        assert result.days_to_fatigued is None

    @patch("analysis.cdi_forecast._get_load_forecast_for_date", return_value=None)
    @patch("analysis.cdi_forecast.read_summary")
    @patch("analysis.cdi_forecast.list_available_dates")
    def test_to_dict_is_json_serialisable(
        self, mock_dates, mock_summary, mock_forecast
    ):
        from analysis.cdi_forecast import compute_cdi_forecast

        dates = [f"2026-03-{i:02d}" for i in range(1, 10)]
        mock_dates.return_value = dates
        mock_summary.return_value = self._make_rolling(dates)

        debt = self._make_debt()
        with patch("analysis.cognitive_debt.compute_cdi", return_value=debt):
            result = compute_cdi_forecast("2026-03-10")

        d = result.to_dict()
        serialised = json.dumps(d)
        assert isinstance(serialised, str)
        parsed = json.loads(serialised)
        assert "today_cdi" in parsed
        assert "projected_cdis" in parsed


# ─── Formatting tests ─────────────────────────────────────────────────────────

class TestFormatting:
    def _make_forecast(self, **kwargs) -> CDIForecast:
        defaults = dict(
            today_cdi=62.0,
            today_tier="loading",
            projected_cdis=[63.0, 65.0, 67.0, 69.0, 71.0],
            projected_dates=[
                "2026-03-16", "2026-03-17", "2026-03-18",
                "2026-03-19", "2026-03-20",
            ],
            projected_tiers=["loading", "loading", "loading", "loading", "fatigued"],
            trend_direction="worsening",
            days_to_fatigued=5,
            days_to_recovery=None,
            headline="At this pace, CDI reaches fatigued in ~5 days.",
            is_meaningful=True,
            recovery_signal_used=0.72,
            load_signal_used=0.14,
            days_of_history=8,
        )
        defaults.update(kwargs)
        return CDIForecast(**defaults)

    def test_line_empty_when_not_meaningful(self):
        f = self._make_forecast(is_meaningful=False)
        assert format_cdi_forecast_line(f) == ""

    def test_line_non_empty_when_meaningful(self):
        f = self._make_forecast()
        line = format_cdi_forecast_line(f)
        assert len(line) > 0

    def test_line_contains_sparkline_chars(self):
        f = self._make_forecast()
        line = format_cdi_forecast_line(f)
        # Sparkline enclosed in brackets
        assert "[" in line and "]" in line

    def test_line_mentions_days_to_fatigued(self):
        f = self._make_forecast(days_to_fatigued=3)
        line = format_cdi_forecast_line(f)
        assert "3d" in line or "fatigued" in line.lower()

    def test_section_empty_when_not_meaningful(self):
        f = self._make_forecast(is_meaningful=False)
        assert format_cdi_forecast_section(f) == ""

    def test_section_contains_header(self):
        f = self._make_forecast()
        section = format_cdi_forecast_section(f)
        assert "CDI Trajectory Forecast" in section

    def test_section_has_all_days(self):
        f = self._make_forecast()
        section = format_cdi_forecast_section(f)
        # Should have entries for +1d through +5d
        for i in range(1, FORECAST_DAYS + 1):
            assert f"+{i}d" in section

    def test_section_marks_tier_crossing(self):
        f = self._make_forecast(
            projected_tiers=["loading", "loading", "fatigued", "fatigued", "fatigued"]
        )
        section = format_cdi_forecast_section(f)
        assert "tier crossed" in section

    def test_section_contains_headline(self):
        forecast = self._make_forecast()
        section = format_cdi_forecast_section(forecast)
        # The headline text should appear somewhere in the section
        assert forecast.headline in section or "5 day" in section

"""
Tests for analysis/burnout_risk.py — Burnout Risk Index

Tests cover:
  - _linear_slope() with exact values
  - _slope_to_component() normalisation
  - _tier_from_bri() tier classification
  - compute_burnout_risk() with mocked store data:
      - not enough data → is_meaningful=False
      - all-flat trends → BRI near 0, healthy tier
      - worsening HRV → hrv component rises
      - worsening CLS + declining FDI → load_creep + focus_erosion components
      - all signals worsening → high BRI
  - format_bri_line() includes key text
  - format_bri_section() includes signal labels
  - format_bri_terminal() renders correctly
  - to_dict() round-trips cleanly
"""

import json
import sys
import tempfile
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.burnout_risk import (
    BurnoutRisk,
    _linear_slope,
    _slope_to_component,
    _clamp,
    _tier_from_bri,
    _extract_daily_series,
    compute_burnout_risk,
    format_bri_line,
    format_bri_section,
    format_bri_terminal,
    BRI_MIN_DAYS,
    BRI_DEFAULT_DAYS,
)


# ─── Helper: build a fake window ──────────────────────────────────────────────

def _make_window(
    hour: int = 9,
    cls: float = 0.30,
    fdi: float = 0.65,
    sdi: float = 0.20,
    hrv: float = 72.0,
    sleep_perf: float = 80.0,
    is_active: bool = True,
    is_working: bool = True,
) -> dict:
    return {
        "metadata": {
            "is_active_window": is_active,
            "is_working_hours": is_working,
            "hour_of_day": hour,
        },
        "whoop": {
            "hrv_rmssd_milli": hrv,
            "sleep_performance": sleep_perf,
        },
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "context_switch_cost": 0.20,
            "recovery_alignment_score": 0.80,
        },
    }


def _make_windows_for_day(cls=0.30, fdi=0.65, sdi=0.20, hrv=72.0, sleep_perf=80.0) -> list:
    """Create a realistic set of windows for a day."""
    windows = []
    for hour in range(8, 20):
        windows.append(_make_window(hour=hour, cls=cls, fdi=fdi, sdi=sdi,
                                    hrv=hrv, sleep_perf=sleep_perf))
    return windows


# ─── Unit tests: _linear_slope ────────────────────────────────────────────────

class TestLinearSlope:
    def test_positive_slope(self):
        xs = [0, 1, 2, 3, 4]
        ys = [1.0, 2.0, 3.0, 4.0, 5.0]
        slope = _linear_slope(xs, ys)
        assert slope is not None
        assert abs(slope - 1.0) < 1e-9

    def test_negative_slope(self):
        xs = [0, 1, 2, 3]
        ys = [10.0, 7.0, 4.0, 1.0]
        slope = _linear_slope(xs, ys)
        assert slope is not None
        assert abs(slope - (-3.0)) < 1e-9

    def test_flat_slope(self):
        xs = [0, 1, 2, 3]
        ys = [5.0, 5.0, 5.0, 5.0]
        slope = _linear_slope(xs, ys)
        assert slope is not None
        assert abs(slope) < 1e-9

    def test_too_few_points_returns_none(self):
        assert _linear_slope([], []) is None
        assert _linear_slope([0], [1.0]) is None

    def test_two_points_exact(self):
        xs = [0, 2]
        ys = [3.0, 7.0]
        slope = _linear_slope(xs, ys)
        assert slope is not None
        assert abs(slope - 2.0) < 1e-9


# ─── Unit tests: _slope_to_component ─────────────────────────────────────────

class TestSlopeToComponent:
    def test_none_slope_returns_zero(self):
        assert _slope_to_component(None, -0.5) == 0.0
        assert _slope_to_component(None, 0.005) == 0.0

    def test_declining_signal_positive_slope_is_zero_risk(self):
        # scale < 0: positive slope (improving HRV) → 0.0 risk
        assert _slope_to_component(+0.3, -0.5) == 0.0

    def test_declining_signal_at_scale_is_max_risk(self):
        # slope == scale (-0.5) → component 1.0
        comp = _slope_to_component(-0.5, -0.5)
        assert abs(comp - 1.0) < 1e-9

    def test_declining_signal_half_scale_is_half_risk(self):
        # slope = -0.25 with scale = -0.5 → component 0.5
        comp = _slope_to_component(-0.25, -0.5)
        assert abs(comp - 0.5) < 1e-9

    def test_declining_signal_beyond_scale_clamps_to_one(self):
        comp = _slope_to_component(-1.0, -0.5)
        assert comp == 1.0

    def test_rising_signal_negative_slope_is_zero_risk(self):
        # scale > 0: negative slope (CLS dropping) → 0.0 risk
        assert _slope_to_component(-0.01, 0.005) == 0.0

    def test_rising_signal_at_scale_is_max_risk(self):
        comp = _slope_to_component(0.005, 0.005)
        assert abs(comp - 1.0) < 1e-9

    def test_rising_signal_half_scale_is_half_risk(self):
        comp = _slope_to_component(0.0025, 0.005)
        assert abs(comp - 0.5) < 1e-9


# ─── Unit tests: _tier_from_bri ───────────────────────────────────────────────

class TestTierFromBri:
    def test_zero_is_healthy(self):
        tier, label = _tier_from_bri(0.0)
        assert tier == "healthy"
        assert "Healthy" in label

    def test_19_is_healthy(self):
        tier, _ = _tier_from_bri(19.9)
        assert tier == "healthy"

    def test_20_is_watch(self):
        tier, label = _tier_from_bri(20.0)
        assert tier == "watch"
        assert "Watch" in label

    def test_39_is_watch(self):
        tier, _ = _tier_from_bri(39.9)
        assert tier == "watch"

    def test_40_is_caution(self):
        tier, label = _tier_from_bri(40.0)
        assert tier == "caution"
        assert "Caution" in label

    def test_60_is_high_risk(self):
        tier, label = _tier_from_bri(60.0)
        assert tier == "high_risk"
        assert "High Risk" in label

    def test_80_is_critical(self):
        tier, label = _tier_from_bri(80.0)
        assert tier == "critical"
        assert "Critical" in label

    def test_100_is_critical(self):
        tier, _ = _tier_from_bri(100.0)
        assert tier == "critical"


# ─── Integration tests using mocked store ─────────────────────────────────────

def _build_mock_store(days_of_data: int, cls_values=None, fdi_values=None,
                      hrv_values=None, sdi_values=None, sleep_values=None):
    """
    Return a mock for engine.store.read_day and list_available_dates.
    Generates `days_of_data` days ending at 2026-03-17.
    """
    end = date(2026, 3, 17)
    dates = []
    for i in range(days_of_data - 1, -1, -1):
        dates.append((end - timedelta(days=i)).strftime("%Y-%m-%d"))

    def mock_list():
        return dates

    def mock_read(date_str):
        if date_str not in dates:
            return []
        idx = dates.index(date_str)
        cls = cls_values[idx] if cls_values else 0.30
        fdi = fdi_values[idx] if fdi_values else 0.65
        hrv = hrv_values[idx] if hrv_values else 72.0
        sdi = sdi_values[idx] if sdi_values else 0.20
        sleep = sleep_values[idx] if sleep_values else 80.0
        return _make_windows_for_day(cls=cls, fdi=fdi, sdi=sdi, hrv=hrv, sleep_perf=sleep)

    return mock_list, mock_read


class TestComputeBurnoutRiskNoData:
    def test_no_data_returns_not_meaningful(self):
        with patch("analysis.burnout_risk.list_available_dates", return_value=[]), \
             patch("analysis.burnout_risk.read_day", return_value=[]):
            result = compute_burnout_risk()
            assert result.is_meaningful is False
            assert result.bri == 0.0

    def test_too_few_days_returns_not_meaningful(self):
        mock_list, mock_read = _build_mock_store(days_of_data=10)
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.is_meaningful is False
            assert result.days_used == 10


class TestComputeBurnoutRiskFlatTrends:
    """All metrics perfectly flat → BRI near 0, healthy tier."""

    def test_flat_data_is_healthy(self):
        n = 28
        mock_list, mock_read = _build_mock_store(
            days_of_data=n,
            cls_values=[0.30] * n,
            fdi_values=[0.65] * n,
            hrv_values=[72.0] * n,
            sdi_values=[0.20] * n,
            sleep_values=[80.0] * n,
        )
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.is_meaningful is True
            assert result.bri < 10.0  # should be near 0 with flat trends
            assert result.tier == "healthy"

    def test_flat_data_slopes_near_zero(self):
        n = 28
        mock_list, mock_read = _build_mock_store(
            days_of_data=n,
            cls_values=[0.25] * n,
            fdi_values=[0.70] * n,
            hrv_values=[75.0] * n,
        )
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.cls_slope is not None
            assert abs(result.cls_slope) < 1e-8
            assert result.hrv_slope is not None
            assert abs(result.hrv_slope) < 1e-8


class TestComputeBurnoutRiskWorsening:
    """Simulated declining trends should raise BRI."""

    def test_declining_hrv_raises_bri(self):
        n = 28
        # HRV declining: 90ms → 55ms over 28 days (slope ≈ -1.3 ms/day)
        hrv_vals = [90.0 - i * 1.3 for i in range(n)]
        mock_list, mock_read = _build_mock_store(
            days_of_data=n,
            hrv_values=hrv_vals,
        )
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.is_meaningful is True
            # HRV declining strongly → component should be near 1.0, BRI elevated
            assert result.components["hrv"] > 0.5
            assert result.bri > 10.0  # meaningfully above baseline

    def test_rising_cls_raises_load_creep(self):
        n = 28
        # CLS rising from 0.20 to 0.60 over 28 days (slope ≈ 0.014 CLS/day)
        cls_vals = [0.20 + i * 0.014 for i in range(n)]
        mock_list, mock_read = _build_mock_store(
            days_of_data=n,
            cls_values=cls_vals,
        )
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.is_meaningful is True
            assert result.components["load_creep"] > 0.5
            assert result.cls_slope is not None
            assert result.cls_slope > 0

    def test_declining_fdi_raises_focus_erosion(self):
        n = 28
        # FDI declining from 0.80 to 0.40 (slope ≈ -0.014 FDI/day)
        fdi_vals = [0.80 - i * 0.014 for i in range(n)]
        mock_list, mock_read = _build_mock_store(
            days_of_data=n,
            fdi_values=fdi_vals,
        )
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.is_meaningful is True
            assert result.components["focus_erosion"] > 0.5

    def test_rising_sdi_raises_social_drain(self):
        n = 28
        # SDI rising from 0.10 to 0.50 (slope ≈ 0.014 SDI/day)
        sdi_vals = [0.10 + i * 0.014 for i in range(n)]
        mock_list, mock_read = _build_mock_store(
            days_of_data=n,
            sdi_values=sdi_vals,
        )
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.is_meaningful is True
            assert result.components["social_drain"] > 0.5

    def test_all_signals_worsening_gives_high_bri(self):
        n = 28
        hrv_vals   = [90.0  - i * 1.3  for i in range(n)]
        cls_vals   = [0.20  + i * 0.014 for i in range(n)]
        fdi_vals   = [0.80  - i * 0.014 for i in range(n)]
        sdi_vals   = [0.10  + i * 0.014 for i in range(n)]
        sleep_vals = [85.0  - i * 1.0  for i in range(n)]

        mock_list, mock_read = _build_mock_store(
            days_of_data=n,
            hrv_values=hrv_vals,
            cls_values=cls_vals,
            fdi_values=fdi_vals,
            sdi_values=sdi_vals,
            sleep_values=sleep_vals,
        )
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.is_meaningful is True
            assert result.bri > 40.0  # should be in caution/high_risk
            assert result.tier in ("caution", "high_risk", "critical")

    def test_improving_trends_give_zero_bri(self):
        n = 28
        # Everything improving: HRV rising, CLS falling, FDI rising, SDI falling
        hrv_vals   = [60.0  + i * 1.0  for i in range(n)]
        cls_vals   = [0.60  - i * 0.01 for i in range(n)]
        fdi_vals   = [0.40  + i * 0.01 for i in range(n)]
        sdi_vals   = [0.50  - i * 0.01 for i in range(n)]
        sleep_vals = [70.0  + i * 0.5  for i in range(n)]

        mock_list, mock_read = _build_mock_store(
            days_of_data=n,
            hrv_values=hrv_vals,
            cls_values=cls_vals,
            fdi_values=fdi_vals,
            sdi_values=sdi_vals,
            sleep_values=sleep_vals,
        )
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.is_meaningful is True
            assert result.bri < 5.0  # all components should be 0
            assert result.tier == "healthy"


class TestComputeBurnoutRiskMinimum:
    """Test exactly at the minimum data boundary."""

    def test_exactly_min_days_is_meaningful(self):
        n = BRI_MIN_DAYS
        mock_list, mock_read = _build_mock_store(days_of_data=n)
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.is_meaningful is True
            assert result.days_used == n

    def test_one_below_min_is_not_meaningful(self):
        n = BRI_MIN_DAYS - 1
        mock_list, mock_read = _build_mock_store(days_of_data=n)
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.is_meaningful is False


# ─── Tests: dominant signal detection ─────────────────────────────────────────

class TestDominantSignal:
    def test_dominant_is_highest_component(self):
        n = 28
        # Only HRV is declining strongly; everything else flat
        hrv_vals = [90.0 - i * 1.3 for i in range(n)]
        mock_list, mock_read = _build_mock_store(
            days_of_data=n,
            hrv_values=hrv_vals,
        )
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.dominant_signal == "hrv"

    def test_dominant_is_load_creep_when_cls_rises(self):
        n = 28
        cls_vals = [0.20 + i * 0.014 for i in range(n)]
        mock_list, mock_read = _build_mock_store(
            days_of_data=n,
            cls_values=cls_vals,
        )
        with patch("analysis.burnout_risk.list_available_dates", mock_list), \
             patch("analysis.burnout_risk.read_day", mock_read):
            result = compute_burnout_risk("2026-03-17")
            assert result.dominant_signal == "load_creep"


# ─── Tests: formatting ────────────────────────────────────────────────────────

class TestFormatBriLine:
    def test_not_meaningful_returns_empty(self):
        bri = BurnoutRisk(
            bri=0.0, tier="healthy", tier_label="🟢 Healthy",
            is_meaningful=False, days_used=5, days_requested=28,
            date_range="no data",
        )
        assert format_bri_line(bri) == ""

    def test_healthy_line_contains_score(self):
        bri = BurnoutRisk(
            bri=12.0, tier="healthy", tier_label="🟢 Healthy",
            is_meaningful=True, days_used=28, days_requested=28,
            date_range="2026-02-18 → 2026-03-17",
            components={"hrv": 0.1, "sleep": 0.05, "load_creep": 0.1,
                        "focus_erosion": 0.05, "social_drain": 0.05},
            component_labels={},
            dominant_signal="hrv",
            trajectory_headline="All good.",
            intervention_advice="Keep going.",
        )
        line = format_bri_line(bri)
        assert "12" in line
        assert "Burnout Risk" in line

    def test_caution_line_includes_dominant_signal(self):
        bri = BurnoutRisk(
            bri=47.0, tier="caution", tier_label="🟠 Caution",
            is_meaningful=True, days_used=28, days_requested=28,
            date_range="2026-02-18 → 2026-03-17",
            components={"hrv": 0.6, "sleep": 0.3, "load_creep": 0.2,
                        "focus_erosion": 0.1, "social_drain": 0.1},
            component_labels={},
            dominant_signal="hrv",
            trajectory_headline="Watch out.",
            intervention_advice="Protect recovery.",
        )
        line = format_bri_line(bri)
        assert "47" in line
        assert "HRV" in line.lower() or "hrv" in line.lower()


class TestFormatBriSection:
    def test_not_meaningful_returns_empty(self):
        bri = BurnoutRisk(
            bri=0.0, tier="healthy", tier_label="🟢 Healthy",
            is_meaningful=False, days_used=5, days_requested=28,
            date_range="no data",
        )
        assert format_bri_section(bri) == ""

    def test_section_contains_all_signals(self):
        bri = BurnoutRisk(
            bri=35.0, tier="watch", tier_label="🟡 Watch",
            is_meaningful=True, days_used=28, days_requested=28,
            date_range="2026-02-18 → 2026-03-17",
            components={"hrv": 0.4, "sleep": 0.2, "load_creep": 0.3,
                        "focus_erosion": 0.1, "social_drain": 0.1},
            component_labels={
                "hrv": "HRV slowly declining",
                "sleep": "Sleep stable",
                "load_creep": "Load mild creep",
                "focus_erosion": "Focus stable",
                "social_drain": "Social load stable",
            },
            dominant_signal="hrv",
            trajectory_headline="Watch for trends.",
            intervention_advice="Protect sleep.",
        )
        section = format_bri_section(bri)
        assert "HRV trend" in section
        assert "Sleep quality" in section
        assert "Load creep" in section
        assert "Focus erosion" in section
        assert "Social drain" in section
        assert "35" in section

    def test_section_includes_intervention_advice(self):
        bri = BurnoutRisk(
            bri=55.0, tier="caution", tier_label="🟠 Caution",
            is_meaningful=True, days_used=28, days_requested=28,
            date_range="2026-02-18 → 2026-03-17",
            components={"hrv": 0.5, "sleep": 0.4, "load_creep": 0.3,
                        "focus_erosion": 0.2, "social_drain": 0.2},
            component_labels={k: "—" for k in
                               ["hrv", "sleep", "load_creep", "focus_erosion", "social_drain"]},
            dominant_signal="hrv",
            trajectory_headline="Multiple signals worsening.",
            intervention_advice="Take a lighter day this week.",
        )
        section = format_bri_section(bri)
        assert "Take a lighter day" in section


class TestFormatBriTerminal:
    def test_not_meaningful_shows_need_more_data(self):
        bri = BurnoutRisk(
            bri=0.0, tier="healthy", tier_label="🟢 Healthy",
            is_meaningful=False, days_used=10, days_requested=28,
            date_range="no data",
        )
        output = format_bri_terminal(bri)
        assert "need" in output.lower() or "more days" in output.lower()

    def test_meaningful_shows_bri_and_tier(self):
        bri = BurnoutRisk(
            bri=42.0, tier="caution", tier_label="🟠 Caution",
            is_meaningful=True, days_used=28, days_requested=28,
            date_range="2026-02-18 → 2026-03-17",
            components={"hrv": 0.3, "sleep": 0.4, "load_creep": 0.5,
                        "focus_erosion": 0.2, "social_drain": 0.3},
            component_labels={k: "moderate" for k in
                               ["hrv", "sleep", "load_creep", "focus_erosion", "social_drain"]},
            dominant_signal="load_creep",
            trend_direction="worsening",
            trajectory_headline="Load creep detected.",
            intervention_advice="Reduce discretionary load.",
        )
        output = format_bri_terminal(bri)
        assert "42" in output
        assert "Caution" in output or "caution" in output


# ─── Tests: to_dict ───────────────────────────────────────────────────────────

class TestToDict:
    def test_round_trip_json(self):
        bri = BurnoutRisk(
            bri=28.5, tier="watch", tier_label="🟡 Watch",
            is_meaningful=True, days_used=21, days_requested=28,
            date_range="2026-02-25 → 2026-03-17",
            components={"hrv": 0.3, "sleep": 0.2, "load_creep": 0.25,
                        "focus_erosion": 0.1, "social_drain": 0.15},
            component_labels={"hrv": "mild", "sleep": "stable", "load_creep": "mild",
                              "focus_erosion": "stable", "social_drain": "stable"},
            dominant_signal="hrv",
            trend_direction="stable",
            trajectory_headline="Mild trends.",
            intervention_advice="Monitor.",
            hrv_slope=-0.2,
            sleep_slope=-0.3,
            cls_slope=0.002,
            fdi_slope=-0.001,
            sdi_slope=0.001,
        )
        d = bri.to_dict()
        # Should be JSON serialisable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["bri"] == 28.5
        assert parsed["tier"] == "watch"
        assert parsed["is_meaningful"] is True
        assert "components" in parsed
        assert "hrv_slope" in parsed

    def test_all_fields_present(self):
        bri = BurnoutRisk(
            bri=0.0, tier="healthy", tier_label="🟢 Healthy",
            is_meaningful=False, days_used=0, days_requested=28,
            date_range="no data",
        )
        d = bri.to_dict()
        expected_keys = [
            "bri", "tier", "tier_label", "is_meaningful", "days_used",
            "days_requested", "date_range", "components", "component_labels",
            "dominant_signal", "trend_direction", "trajectory_headline",
            "intervention_advice", "hrv_slope", "sleep_slope", "cls_slope",
            "fdi_slope", "sdi_slope",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"


# ─── Tests: component clamping ────────────────────────────────────────────────

class TestClamp:
    def test_clamp_below(self):
        assert _clamp(-0.5) == 0.0

    def test_clamp_above(self):
        assert _clamp(1.5) == 1.0

    def test_in_range(self):
        assert _clamp(0.5) == 0.5

    def test_custom_range(self):
        assert _clamp(-5.0, -10.0, 10.0) == -5.0
        assert _clamp(-15.0, -10.0, 10.0) == -10.0
        assert _clamp(15.0, -10.0, 10.0) == 10.0

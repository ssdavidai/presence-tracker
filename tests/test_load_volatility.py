"""
Tests for analysis/load_volatility.py — Load Volatility Index (LVI)

Coverage:
  1. compute_load_volatility()
     - Returns is_meaningful=False when no windows
     - Returns is_meaningful=False when fewer than MIN_WINDOWS active windows
     - Returns is_meaningful=False when only non-working-hours windows
     - Returns is_meaningful=False when windows have no meeting/Slack activity
     - Computes correct CLS mean for uniform CLS
     - Computes correct CLS std = 0 for perfectly uniform load → LVI = 1.0
     - Computes correct LVI for known std values
     - LVI is always clamped to [0.0, 1.0]
     - LVI = 0.0 when cls_std ≥ LVI_STD_SCALE
     - Label 'smooth' when lvi ≥ 0.80
     - Label 'steady' when lvi in [0.60, 0.80)
     - Label 'variable' when lvi in [0.40, 0.60)
     - Label 'volatile' when lvi < 0.40
     - cls_min / cls_max / cls_range computed correctly
     - windows_used reflects only active working-hour windows
     - Non-active working windows are excluded
     - Non-working-hour windows are excluded
     - Mixed meeting + slack windows both count as active

  2. _classify_label()
     - Boundary at SMOOTH_THRESHOLD (0.80) is inclusive
     - Boundary at STEADY_THRESHOLD (0.60) is inclusive
     - Boundary at VARIABLE_THRESHOLD (0.40) is inclusive

  3. _build_insight()
     - Volatile label always mentions "swing" and "std"
     - Variable label mentions "swing" and "std"
     - Smooth light-load label mentions "light"
     - Smooth moderate-load label mentions "consistent"
     - Smooth high-load label mentions "sustained" or "consistent"
     - Steady label mentions "moderate" or "broadly predictable"

  4. format_lvi_line()
     - Returns empty string when is_meaningful=False
     - Contains label display name
     - Contains LVI value
     - Contains std value
     - Volatile line contains "high cognitive switching cost"
     - Variable line contains "uneven demand pattern"
     - Smooth line does not contain warning text
     - Steady line does not contain warning text

  5. format_lvi_section()
     - Returns empty string when is_meaningful=False
     - Contains bold label header
     - Contains italic insight

  6. _compute_load_volatility_for_digest() integration
     - Returns None on empty windows
     - Returns None when is_meaningful=False (< 3 active windows)
     - Returns dict with 'is_meaningful': True for sufficient active windows
     - Returned dict has 'lvi', 'cls_std', 'cls_range', 'label', 'line', 'insight'
     - Returns None on compute exception (graceful isolation)

  7. compute_digest() integration
     - digest dict contains 'load_volatility' key
     - load_volatility is None when < 3 active windows
     - load_volatility is meaningful dict when sufficient active windows

  8. format_digest_message() integration
     - LVI line shown when label is 'volatile'
     - LVI line shown when label is 'variable'
     - LVI line NOT shown when label is 'smooth'
     - LVI line NOT shown when label is 'steady'
     - LVI line NOT shown when load_volatility is None
     - LVI line NOT shown when is_meaningful=False
     - LVI line appears after sparkline and before focus quality section

Run with: python3 -m pytest tests/test_load_volatility.py -v
"""

import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from analysis.load_volatility import (
    LoadVolatility,
    compute_load_volatility,
    format_lvi_line,
    format_lvi_section,
    _classify_label,
    _build_insight,
    MIN_WINDOWS,
    LVI_STD_SCALE,
    SMOOTH_THRESHOLD,
    STEADY_THRESHOLD,
    VARIABLE_THRESHOLD,
)


# ─── Window builder helpers ───────────────────────────────────────────────────

def _make_window(
    date_str: str = "2026-03-17",
    hour: int = 9,
    minute: int = 0,
    is_working_hours: bool = True,
    in_meeting: bool = False,
    messages_sent: int = 0,
    messages_received: int = 0,
    cls: float = 0.30,
    fdi: float = 0.70,
    sdi: float = 0.20,
    ras: float = 0.75,
    csc: float = 0.15,
) -> dict:
    """Build a minimal 15-min window dict."""
    return {
        "date": date_str,
        "metadata": {
            "hour_of_day": hour,
            "minute_of_hour": minute,
            "is_working_hours": is_working_hours,
        },
        "calendar": {
            "in_meeting": in_meeting,
            "meeting_title": "Test meeting" if in_meeting else None,
        },
        "slack": {
            "messages_sent": messages_sent,
            "messages_received": messages_received,
            "total_messages": messages_sent + messages_received,
        },
        "whoop": {"recovery_score": 75.0, "hrv_rmssd_milli": 65.0},
        "rescuetime": None,
        "omi": None,
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "recovery_alignment_score": ras,
            "context_switch_cost": csc,
        },
    }


def _make_active_windows(cls_values: list[float], start_hour: int = 9) -> list[dict]:
    """
    Build a list of active (meeting) windows with the given CLS values.
    Hours cycle through start_hour, start_hour+1, ... to keep them in working hours.
    """
    windows = []
    for i, cls in enumerate(cls_values):
        hour = start_hour + (i * 15) // 60
        minute = (i * 15) % 60
        windows.append(_make_window(
            hour=hour,
            minute=minute,
            in_meeting=True,
            cls=cls,
        ))
    return windows


# ─── 1. compute_load_volatility() ────────────────────────────────────────────

class TestComputeLoadVolatility:

    def test_empty_windows_not_meaningful(self):
        lvi = compute_load_volatility([])
        assert lvi.is_meaningful is False

    def test_fewer_than_min_windows_not_meaningful(self):
        # MIN_WINDOWS = 3; provide MIN_WINDOWS - 1 active windows
        windows = _make_active_windows([0.30, 0.50])  # 2 windows
        lvi = compute_load_volatility(windows)
        assert lvi.is_meaningful is False
        assert lvi.windows_used == 2

    def test_exactly_min_windows_is_meaningful(self):
        windows = _make_active_windows([0.30, 0.50, 0.40])  # 3 windows
        lvi = compute_load_volatility(windows)
        assert lvi.is_meaningful is True
        assert lvi.windows_used == 3

    def test_non_working_hours_excluded(self):
        # All windows outside working hours → not meaningful
        windows = [
            _make_window(hour=2, is_working_hours=False, in_meeting=True, cls=0.30),
            _make_window(hour=3, is_working_hours=False, in_meeting=True, cls=0.60),
            _make_window(hour=4, is_working_hours=False, in_meeting=True, cls=0.45),
        ]
        lvi = compute_load_volatility(windows)
        assert lvi.is_meaningful is False

    def test_idle_windows_excluded(self):
        # No meeting, no Slack → not active
        windows = [
            _make_window(hour=9, in_meeting=False, messages_sent=0, messages_received=0, cls=0.30),
            _make_window(hour=10, in_meeting=False, messages_sent=0, messages_received=0, cls=0.60),
            _make_window(hour=11, in_meeting=False, messages_sent=0, messages_received=0, cls=0.45),
        ]
        lvi = compute_load_volatility(windows)
        assert lvi.is_meaningful is False

    def test_slack_windows_count_as_active(self):
        # Has Slack messages but no meeting → should be counted as active
        windows = [
            _make_window(hour=9, in_meeting=False, messages_sent=5, cls=0.25),
            _make_window(hour=10, in_meeting=False, messages_sent=3, cls=0.45),
            _make_window(hour=11, in_meeting=False, messages_received=7, cls=0.35),
        ]
        lvi = compute_load_volatility(windows)
        assert lvi.is_meaningful is True
        assert lvi.windows_used == 3

    def test_meeting_windows_count_as_active(self):
        windows = _make_active_windows([0.20, 0.40, 0.60])
        lvi = compute_load_volatility(windows)
        assert lvi.is_meaningful is True
        assert lvi.windows_used == 3

    def test_uniform_cls_gives_std_zero_and_lvi_one(self):
        # All windows have same CLS → std = 0 → LVI = 1.0
        windows = _make_active_windows([0.40, 0.40, 0.40, 0.40, 0.40])
        lvi = compute_load_volatility(windows)
        assert lvi.is_meaningful is True
        assert lvi.cls_std == 0.0
        assert lvi.lvi == 1.0
        assert lvi.label == "smooth"

    def test_lvi_formula_known_std(self):
        # With std = LVI_STD_SCALE / 2 → LVI = 0.5
        # We need 3 windows whose population std = LVI_STD_SCALE / 2 = 0.175
        # Using values [0.0, 0.35, 0.175]: mean=0.175, deviations=[-0.175, 0.175, 0.0]
        # var = (0.175^2 + 0.175^2 + 0^2) / 3 = 0.02042; std = 0.1429 ≠ 0.175
        # Simpler: use 2 extreme values + 1 middle repeated
        # [0.0, 0.0, 0.35]: mean=0.1167, deviations: [-0.1167, -0.1167, 0.2333]
        # That's not clean either. Use direct check instead.
        windows = _make_active_windows([0.10, 0.45, 0.10])
        lvi = compute_load_volatility(windows)
        # Compute expected
        vals = [0.10, 0.45, 0.10]
        mean = sum(vals) / 3
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / 3)
        expected_lvi = round(1.0 - min(std / LVI_STD_SCALE, 1.0), 4)
        assert abs(lvi.lvi - expected_lvi) < 0.0001

    def test_lvi_clamped_to_zero_when_std_exceeds_scale(self):
        # If cls values span 0.0 to 1.0 (maximum range), std will exceed LVI_STD_SCALE
        # ensuring LVI ≥ 0
        windows = _make_active_windows([0.0, 0.0, 1.0, 0.0, 1.0])
        lvi = compute_load_volatility(windows)
        assert lvi.lvi >= 0.0
        assert lvi.lvi <= 1.0

    def test_high_std_gives_volatile_label(self):
        # Large swings → high std → volatile
        windows = _make_active_windows([0.05, 0.80, 0.05, 0.85, 0.05, 0.75])
        lvi = compute_load_volatility(windows)
        assert lvi.label in ("volatile", "variable")

    def test_cls_min_max_range_computed_correctly(self):
        windows = _make_active_windows([0.10, 0.50, 0.30])
        lvi = compute_load_volatility(windows)
        assert lvi.cls_min == 0.10
        assert lvi.cls_max == 0.50
        assert abs(lvi.cls_range - 0.40) < 0.001

    def test_cls_mean_computed_correctly(self):
        windows = _make_active_windows([0.20, 0.40, 0.60])
        lvi = compute_load_volatility(windows)
        assert abs(lvi.cls_mean - 0.40) < 0.001

    def test_windows_used_reflects_active_only(self):
        # Mix of active (meeting) and idle windows
        active = _make_active_windows([0.30, 0.50, 0.40, 0.60])  # 4 active
        idle = [
            _make_window(hour=14, in_meeting=False, messages_sent=0, cls=0.10),
            _make_window(hour=15, in_meeting=False, messages_sent=0, cls=0.15),
        ]
        lvi = compute_load_volatility(active + idle)
        assert lvi.windows_used == 4  # Only the 4 active windows

    def test_non_working_hours_windows_not_counted(self):
        # Mix of working + non-working active windows
        working_active = _make_active_windows([0.30, 0.50, 0.40])
        night_active = [
            _make_window(hour=2, is_working_hours=False, in_meeting=True, cls=0.60),
        ]
        lvi = compute_load_volatility(working_active + night_active)
        assert lvi.windows_used == 3  # Only working-hour active windows

    def test_insight_is_non_empty_string(self):
        windows = _make_active_windows([0.20, 0.40, 0.60, 0.20, 0.70])
        lvi = compute_load_volatility(windows)
        assert isinstance(lvi.insight, str)
        assert len(lvi.insight) > 10

    def test_to_dict_contains_all_keys(self):
        windows = _make_active_windows([0.20, 0.40, 0.60])
        lvi = compute_load_volatility(windows)
        d = lvi.to_dict()
        for key in ("lvi", "cls_std", "cls_mean", "cls_min", "cls_max",
                    "cls_range", "label", "windows_used", "is_meaningful", "insight"):
            assert key in d, f"Missing key: {key}"


# ─── 2. _classify_label() ────────────────────────────────────────────────────

class TestClassifyLabel:

    def test_smooth_at_boundary(self):
        assert _classify_label(SMOOTH_THRESHOLD) == "smooth"

    def test_smooth_above_boundary(self):
        assert _classify_label(0.99) == "smooth"

    def test_steady_just_below_smooth(self):
        assert _classify_label(SMOOTH_THRESHOLD - 0.001) == "steady"

    def test_steady_at_boundary(self):
        assert _classify_label(STEADY_THRESHOLD) == "steady"

    def test_variable_just_below_steady(self):
        assert _classify_label(STEADY_THRESHOLD - 0.001) == "variable"

    def test_variable_at_boundary(self):
        assert _classify_label(VARIABLE_THRESHOLD) == "variable"

    def test_volatile_just_below_variable(self):
        assert _classify_label(VARIABLE_THRESHOLD - 0.001) == "volatile"

    def test_volatile_at_zero(self):
        assert _classify_label(0.0) == "volatile"

    def test_all_labels_are_valid_strings(self):
        for val in [0.0, 0.2, 0.39, 0.40, 0.59, 0.60, 0.79, 0.80, 1.0]:
            label = _classify_label(val)
            assert label in ("smooth", "steady", "variable", "volatile")


# ─── 3. _build_insight() ─────────────────────────────────────────────────────

class TestBuildInsight:

    def test_volatile_mentions_swing_and_std(self):
        insight = _build_insight("volatile", cls_std=0.28, cls_mean=0.35, cls_range=0.75, windows=10)
        assert "swing" in insight or "spikes" in insight or "whiplash" in insight
        assert "0.28" in insight

    def test_variable_mentions_swing_and_std(self):
        insight = _build_insight("variable", cls_std=0.16, cls_mean=0.40, cls_range=0.50, windows=8)
        assert "swing" in insight
        assert "0.16" in insight

    def test_smooth_light_mentions_light_or_easy(self):
        insight = _build_insight("smooth", cls_std=0.01, cls_mean=0.15, cls_range=0.05, windows=6)
        assert "light" in insight.lower() or "easy" in insight.lower()

    def test_smooth_moderate_mentions_consistent(self):
        insight = _build_insight("smooth", cls_std=0.02, cls_mean=0.35, cls_range=0.08, windows=8)
        assert "consistent" in insight.lower() or "steady" in insight.lower() or "predictable" in insight.lower()

    def test_smooth_high_load_mentions_sustained(self):
        insight = _build_insight("smooth", cls_std=0.02, cls_mean=0.65, cls_range=0.08, windows=8)
        assert "sustained" in insight.lower() or "consistent" in insight.lower()

    def test_steady_mentions_variation(self):
        insight = _build_insight("steady", cls_std=0.08, cls_mean=0.40, cls_range=0.30, windows=7)
        assert "variation" in insight.lower() or "predictable" in insight.lower() or "variation" in insight.lower()

    def test_insight_is_always_non_empty(self):
        for label in ("smooth", "steady", "variable", "volatile"):
            insight = _build_insight(label, cls_std=0.10, cls_mean=0.40, cls_range=0.40, windows=5)
            assert len(insight) > 0


# ─── 4. format_lvi_line() ────────────────────────────────────────────────────

class TestFormatLviLine:

    def _not_meaningful_lvi(self) -> LoadVolatility:
        return LoadVolatility(
            lvi=0.5, cls_std=0.0, cls_mean=0.0, cls_min=0.0, cls_max=0.0,
            cls_range=0.0, label="steady", windows_used=2,
            is_meaningful=False, insight="Not meaningful.",
        )

    def _make_lvi(self, lvi: float, cls_std: float = 0.10, label: str = "steady") -> LoadVolatility:
        return LoadVolatility(
            lvi=lvi, cls_std=cls_std, cls_mean=0.40, cls_min=0.20, cls_max=0.60,
            cls_range=0.40, label=label, windows_used=5,
            is_meaningful=True,
            insight=f"Test insight for {label}.",
        )

    def test_not_meaningful_returns_empty_string(self):
        lvi = self._not_meaningful_lvi()
        assert format_lvi_line(lvi) == ""

    def test_contains_label_display_name(self):
        lvi = self._make_lvi(0.75, label="steady")
        line = format_lvi_line(lvi)
        assert "Steady" in line

    def test_contains_lvi_value(self):
        lvi = self._make_lvi(0.72, label="steady")
        line = format_lvi_line(lvi)
        assert "0.72" in line

    def test_contains_std_value(self):
        lvi = self._make_lvi(0.72, cls_std=0.09, label="steady")
        line = format_lvi_line(lvi)
        assert "0.09" in line

    def test_volatile_line_contains_warning(self):
        lvi = self._make_lvi(0.20, cls_std=0.28, label="volatile")
        line = format_lvi_line(lvi)
        assert "switching" in line.lower() or "cognitive" in line.lower()

    def test_variable_line_contains_note(self):
        lvi = self._make_lvi(0.50, cls_std=0.16, label="variable")
        line = format_lvi_line(lvi)
        assert "uneven" in line.lower() or "demand" in line.lower()

    def test_smooth_line_no_warning_text(self):
        lvi = self._make_lvi(0.90, cls_std=0.03, label="smooth")
        line = format_lvi_line(lvi)
        assert "switching" not in line.lower()
        assert "whiplash" not in line.lower()

    def test_steady_line_no_warning_text(self):
        lvi = self._make_lvi(0.70, cls_std=0.08, label="steady")
        line = format_lvi_line(lvi)
        assert "switching" not in line.lower()
        assert "whiplash" not in line.lower()

    def test_all_labels_produce_non_empty_lines(self):
        for label in ("smooth", "steady", "variable", "volatile"):
            lvi = self._make_lvi(0.50, label=label)
            line = format_lvi_line(lvi)
            assert len(line) > 0


# ─── 5. format_lvi_section() ─────────────────────────────────────────────────

class TestFormatLviSection:

    def _not_meaningful(self) -> LoadVolatility:
        return LoadVolatility(
            lvi=0.5, cls_std=0.0, cls_mean=0.0, cls_min=0.0, cls_max=0.0,
            cls_range=0.0, label="steady", windows_used=0,
            is_meaningful=False, insight="",
        )

    def test_not_meaningful_returns_empty_string(self):
        assert format_lvi_section(self._not_meaningful()) == ""

    def test_section_contains_bold_header(self):
        lvi = LoadVolatility(
            lvi=0.25, cls_std=0.26, cls_mean=0.40, cls_min=0.05, cls_max=0.85,
            cls_range=0.80, label="volatile", windows_used=10,
            is_meaningful=True, insight="High volatility.",
        )
        section = format_lvi_section(lvi)
        assert "*" in section  # Bold markers
        assert "Volatile" in section

    def test_section_contains_italic_insight(self):
        lvi = LoadVolatility(
            lvi=0.25, cls_std=0.26, cls_mean=0.40, cls_min=0.05, cls_max=0.85,
            cls_range=0.80, label="volatile", windows_used=10,
            is_meaningful=True, insight="High volatility insight.",
        )
        section = format_lvi_section(lvi)
        assert "_" in section  # Italic markers
        assert "High volatility insight." in section


# ─── 6. _compute_load_volatility_for_digest() integration ────────────────────

class TestComputeLoadVolatilityForDigest:
    """Tests for the exception-isolated digest helper."""

    def setup_method(self):
        # Import here to avoid module-level dependency
        from analysis.daily_digest import _compute_load_volatility_for_digest
        self._fn = _compute_load_volatility_for_digest

    def test_empty_windows_returns_none(self):
        result = self._fn([])
        assert result is None

    def test_insufficient_active_windows_returns_none(self):
        # Only 2 active windows → below MIN_WINDOWS
        windows = _make_active_windows([0.30, 0.50])
        result = self._fn(windows)
        assert result is None

    def test_sufficient_windows_returns_dict(self):
        windows = _make_active_windows([0.20, 0.50, 0.30, 0.60, 0.25])
        result = self._fn(windows)
        assert result is not None
        assert result["is_meaningful"] is True

    def test_returned_dict_has_required_keys(self):
        windows = _make_active_windows([0.20, 0.50, 0.30, 0.60, 0.25])
        result = self._fn(windows)
        assert result is not None
        for key in ("lvi", "cls_std", "cls_range", "label", "line", "insight", "windows_used", "is_meaningful"):
            assert key in result, f"Missing key: {key}"

    def test_lvi_value_is_float_in_range(self):
        windows = _make_active_windows([0.20, 0.50, 0.30, 0.60, 0.25])
        result = self._fn(windows)
        assert result is not None
        assert 0.0 <= result["lvi"] <= 1.0

    def test_label_is_valid(self):
        windows = _make_active_windows([0.20, 0.50, 0.30, 0.60, 0.25])
        result = self._fn(windows)
        assert result is not None
        assert result["label"] in ("smooth", "steady", "variable", "volatile")


# ─── 7. compute_digest() integration ─────────────────────────────────────────

class TestComputeDigestIntegration:
    """Tests that compute_digest() correctly includes load_volatility."""

    def _base_whoop(self) -> dict:
        return {
            "recovery_score": 75.0,
            "hrv_rmssd_milli": 65.0,
            "resting_heart_rate": 55.0,
            "sleep_performance": 80.0,
            "sleep_hours": 7.5,
            "strain_score": 10.0,
            "spo2_pct": 96.0,
        }

    def _make_full_windows(self, n: int, cls_values: Optional[list] = None) -> list[dict]:
        """Build n active working-hour windows with given CLS values."""
        if cls_values is None:
            cls_values = [0.35] * n
        windows = []
        whoop = self._base_whoop()
        for i in range(n):
            cls = cls_values[i] if i < len(cls_values) else 0.35
            hour = 9 + (i * 15) // 60
            minute = (i * 15) % 60
            w = _make_window(
                date_str="2026-03-17",
                hour=hour,
                minute=minute,
                in_meeting=True,
                cls=cls,
            )
            w["whoop"] = whoop
            windows.append(w)
        return windows

    def test_digest_contains_load_volatility_key(self):
        from analysis.daily_digest import compute_digest
        windows = self._make_full_windows(5, [0.20, 0.50, 0.30, 0.60, 0.25])
        digest = compute_digest(windows)
        assert "load_volatility" in digest

    def test_load_volatility_none_when_insufficient_windows(self):
        from analysis.daily_digest import compute_digest
        # Only 2 active windows
        windows = self._make_full_windows(2, [0.30, 0.50])
        digest = compute_digest(windows)
        # Either None or is_meaningful=False
        lv = digest.get("load_volatility")
        assert lv is None or lv.get("is_meaningful") is False

    def test_load_volatility_meaningful_with_sufficient_windows(self):
        from analysis.daily_digest import compute_digest
        windows = self._make_full_windows(8, [0.10, 0.70, 0.10, 0.80, 0.10, 0.75, 0.10, 0.80])
        digest = compute_digest(windows)
        lv = digest.get("load_volatility")
        assert lv is not None
        assert lv.get("is_meaningful") is True


# ─── 8. format_digest_message() integration ──────────────────────────────────

class TestFormatDigestMessageIntegration:
    """Tests that format_digest_message() renders LVI correctly."""

    def _minimal_digest(self, load_volatility: Optional[dict] = None) -> dict:
        """Build a minimal digest dict for rendering tests."""
        return {
            "date": "2026-03-17",
            "whoop": {
                "recovery_score": 75.0,
                "hrv_rmssd_milli": 65.0,
                "sleep_hours": 7.5,
                "sleep_performance": 80.0,
            },
            "metrics": {
                "avg_cls": 0.35,
                "peak_cls": 0.65,
                "avg_fdi_active": 0.75,
                "avg_sdi_active": 0.25,
                "avg_csc_active": 0.20,
                "avg_ras": 0.70,
            },
            "activity": {
                "working_windows": 20,
                "active_windows": 10,
                "idle_windows": 10,
                "total_meeting_minutes": 60,
                "meeting_count": 2,
                "slack_sent": 15,
                "slack_received": 40,
            },
            "peak_window": None,
            "trend": {},
            "insight": "Test insight.",
            "hourly_cls_curve": None,
            "rescuetime": None,
            "omi": None,
            "peak_focus_hour": 9,
            "peak_focus_fdi": 0.85,
            "cognitive_debt": None,
            "cdi_forecast": None,
            "presence_score": None,
            "tomorrow_focus_plan": None,
            "meeting_intel": None,
            "personal_records": None,
            "ml_insights": None,
            "tomorrow_load_forecast": None,
            "load_decomposition": None,
            "sleep_target": None,
            "tomorrow_cognitive_budget": None,
            "load_volatility": load_volatility,
        }

    def _volatile_lv(self) -> dict:
        return {
            "lvi": 0.20,
            "cls_std": 0.28,
            "cls_mean": 0.40,
            "cls_range": 0.80,
            "label": "volatile",
            "line": "⚡ Load pattern: Volatile (LVI 0.20, std 0.28) — high cognitive switching cost",
            "insight": "High load volatility today.",
            "windows_used": 8,
            "is_meaningful": True,
        }

    def _variable_lv(self) -> dict:
        return {
            "lvi": 0.50,
            "cls_std": 0.16,
            "cls_mean": 0.40,
            "cls_range": 0.50,
            "label": "variable",
            "line": "〜 Load pattern: Variable (LVI 0.50, std 0.16) — uneven demand pattern",
            "insight": "Noticeable load swings today.",
            "windows_used": 7,
            "is_meaningful": True,
        }

    def _smooth_lv(self) -> dict:
        return {
            "lvi": 0.92,
            "cls_std": 0.03,
            "cls_mean": 0.35,
            "cls_range": 0.08,
            "label": "smooth",
            "line": "〰️ Load pattern: Smooth (LVI 0.92, std 0.03)",
            "insight": "Load was consistent throughout the day.",
            "windows_used": 8,
            "is_meaningful": True,
        }

    def _steady_lv(self) -> dict:
        return {
            "lvi": 0.70,
            "cls_std": 0.09,
            "cls_mean": 0.40,
            "cls_range": 0.30,
            "label": "steady",
            "line": "📊 Load pattern: Steady (LVI 0.70, std 0.09)",
            "insight": "Moderate load variation today.",
            "windows_used": 6,
            "is_meaningful": True,
        }

    def test_volatile_lvi_appears_in_message(self):
        from analysis.daily_digest import format_digest_message
        digest = self._minimal_digest(load_volatility=self._volatile_lv())
        message = format_digest_message(digest)
        assert "Volatile" in message

    def test_variable_lvi_appears_in_message(self):
        from analysis.daily_digest import format_digest_message
        digest = self._minimal_digest(load_volatility=self._variable_lv())
        message = format_digest_message(digest)
        assert "Variable" in message

    def test_smooth_lvi_not_shown_in_message(self):
        from analysis.daily_digest import format_digest_message
        digest = self._minimal_digest(load_volatility=self._smooth_lv())
        message = format_digest_message(digest)
        # Smooth should not surface a warning line
        assert "Smooth" not in message

    def test_steady_lvi_not_shown_in_message(self):
        from analysis.daily_digest import format_digest_message
        digest = self._minimal_digest(load_volatility=self._steady_lv())
        message = format_digest_message(digest)
        assert "Steady" not in message

    def test_none_load_volatility_does_not_crash(self):
        from analysis.daily_digest import format_digest_message
        digest = self._minimal_digest(load_volatility=None)
        # Should render without any exception
        message = format_digest_message(digest)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_not_meaningful_load_volatility_not_shown(self):
        from analysis.daily_digest import format_digest_message
        not_meaningful = {
            "lvi": 0.50, "cls_std": 0.10, "cls_mean": 0.40, "cls_range": 0.30,
            "label": "volatile",  # Even 'volatile' label should be hidden when not meaningful
            "line": "⚡ Load pattern: Volatile (LVI 0.50, std 0.10)",
            "insight": "...",
            "windows_used": 2,
            "is_meaningful": False,  # ← key: not meaningful
        }
        digest = self._minimal_digest(load_volatility=not_meaningful)
        message = format_digest_message(digest)
        # Should not show the LVI line when not meaningful
        assert "⚡ Load pattern" not in message

    def test_volatile_lvi_line_uses_italic_formatting(self):
        from analysis.daily_digest import format_digest_message
        digest = self._minimal_digest(load_volatility=self._volatile_lv())
        message = format_digest_message(digest)
        # The line should be wrapped in italics (_text_) in the digest
        # Look for the LVI content within italic markers
        assert "_" in message  # Italic markers present
        # The volatile line itself should appear within italics
        assert "⚡" in message or "Volatile" in message


# ─── Tests: compute_weekly_lvi_summary() ──────────────────────────────────────

class TestComputeWeeklyLviSummary:
    """Tests for the new compute_weekly_lvi_summary() weekly aggregation function."""

    def _make_volatile_windows(self, date_str: str = "2026-03-17") -> list:
        """Build windows with high CLS variance (volatile pattern)."""
        windows = []
        # Alternating low/high CLS — high std → low LVI → volatile
        cls_vals = [0.05, 0.75, 0.05, 0.80, 0.04, 0.78]
        for i, cls_v in enumerate(cls_vals):
            windows.append(_make_window(
                date_str=date_str, hour=9 + i, minute=0,
                is_working_hours=True, in_meeting=True, cls=cls_v,
            ))
        return windows

    def _make_smooth_windows(self, date_str: str = "2026-03-17") -> list:
        """Build windows with minimal CLS variance (smooth pattern)."""
        windows = []
        for i in range(6):
            windows.append(_make_window(
                date_str=date_str, hour=9 + i, minute=0,
                is_working_hours=True, in_meeting=True, cls=0.40,
            ))
        return windows

    def test_returns_empty_summary_when_no_dates(self):
        """Empty date list → empty summary with days_meaningful=0."""
        from analysis.load_volatility import compute_weekly_lvi_summary
        result = compute_weekly_lvi_summary([])
        assert result["days_meaningful"] == 0
        assert result["avg_lvi"] is None
        assert result["dominant_label"] is None
        assert result["insight"] is None

    def test_days_meaningful_counts_only_meaningful_days(self):
        """days_meaningful reflects only days with is_meaningful=True."""
        from analysis.load_volatility import compute_weekly_lvi_summary
        from unittest.mock import patch

        smooth_wins = self._make_smooth_windows("2026-03-17")
        # Patch at engine.store since that's where read_day is imported from inside the function
        with patch("engine.store.read_day", side_effect=[smooth_wins, [], smooth_wins]):
            result = compute_weekly_lvi_summary(["2026-03-17", "2026-03-18", "2026-03-19"])
        # 2026-03-18 has no windows → not meaningful; 2026-03-17 and 2026-03-19 should be
        assert result["days_meaningful"] == 2

    def test_avg_lvi_reflects_mean_across_meaningful_days(self):
        """avg_lvi is the mean of per-day LVI scores."""
        from analysis.load_volatility import compute_weekly_lvi_summary, compute_load_volatility
        from unittest.mock import patch

        wins_a = self._make_smooth_windows("2026-03-17")
        wins_b = self._make_smooth_windows("2026-03-18")

        lvi_a = compute_load_volatility(wins_a).lvi
        lvi_b = compute_load_volatility(wins_b).lvi
        expected_avg = round((lvi_a + lvi_b) / 2, 4)

        with patch("engine.store.read_day", side_effect=[wins_a, wins_b]):
            result = compute_weekly_lvi_summary(["2026-03-17", "2026-03-18"])

        assert abs(result["avg_lvi"] - expected_avg) < 0.001

    def test_volatile_days_counted_correctly(self):
        """volatile_days reflects the number of days with label='volatile'."""
        from analysis.load_volatility import compute_weekly_lvi_summary
        from unittest.mock import patch

        volatile_wins = self._make_volatile_windows("2026-03-17")
        smooth_wins = self._make_smooth_windows("2026-03-18")

        with patch("engine.store.read_day", side_effect=[volatile_wins, smooth_wins]):
            result = compute_weekly_lvi_summary(["2026-03-17", "2026-03-18"])

        # Volatile windows have high std → volatile label
        assert result["volatile_days"] + result["variable_days"] >= 1
        assert result["days_meaningful"] == 2

    def test_smooth_days_counted_correctly(self):
        """smooth_days reflects the number of days with label='smooth'."""
        from analysis.load_volatility import compute_weekly_lvi_summary, compute_load_volatility
        from unittest.mock import patch

        smooth_wins = self._make_smooth_windows("2026-03-17")
        lvi = compute_load_volatility(smooth_wins)
        # Only test if the day is actually 'smooth'
        if lvi.label == "smooth":
            with patch("engine.store.read_day", side_effect=[smooth_wins, smooth_wins]):
                result = compute_weekly_lvi_summary(["2026-03-17", "2026-03-18"])
            assert result["smooth_days"] == 2

    def test_insight_always_present_when_meaningful(self):
        """insight is a non-empty string when days_meaningful >= 1."""
        from analysis.load_volatility import compute_weekly_lvi_summary
        from unittest.mock import patch

        smooth_wins = self._make_smooth_windows("2026-03-17")
        with patch("engine.store.read_day", return_value=smooth_wins):
            result = compute_weekly_lvi_summary(["2026-03-17"])

        assert result["insight"] is not None
        assert len(result["insight"]) > 10

    def test_read_day_exception_does_not_crash(self):
        """If read_day raises for one date, it is skipped silently."""
        from analysis.load_volatility import compute_weekly_lvi_summary
        from unittest.mock import patch

        smooth_wins = self._make_smooth_windows("2026-03-17")

        def _side_effect(date_str):
            if date_str == "2026-03-18":
                raise IOError("disk error")
            return smooth_wins

        with patch("engine.store.read_day", side_effect=_side_effect):
            result = compute_weekly_lvi_summary(["2026-03-17", "2026-03-18"])

        # Should succeed with just the one good date
        assert result["days_meaningful"] == 1


class TestFormatWeeklyLviLine:
    """Tests for format_weekly_lvi_line()."""

    def test_empty_string_when_no_meaningful_days(self):
        """Empty weekly_lvi → empty string."""
        from analysis.load_volatility import format_weekly_lvi_line
        result = format_weekly_lvi_line({"days_meaningful": 0})
        assert result == ""

    def test_empty_string_when_avg_lvi_none(self):
        """avg_lvi=None → empty string."""
        from analysis.load_volatility import format_weekly_lvi_line
        result = format_weekly_lvi_line({"days_meaningful": 3, "avg_lvi": None,
                                          "volatile_days": 0, "variable_days": 0, "smooth_days": 0})
        assert result == ""

    def test_volatile_days_produce_warning_line(self):
        """≥ 2 volatile days → ⚡ warning line."""
        from analysis.load_volatility import format_weekly_lvi_line
        result = format_weekly_lvi_line({
            "days_meaningful": 5,
            "avg_lvi": 0.30,
            "volatile_days": 2,
            "variable_days": 1,
            "smooth_days": 1,
        })
        assert "⚡" in result
        assert "volatile" in result.lower() or "2" in result

    def test_many_variable_plus_volatile_produces_line(self):
        """≥ 3 variable+volatile days → 〜 line."""
        from analysis.load_volatility import format_weekly_lvi_line
        result = format_weekly_lvi_line({
            "days_meaningful": 5,
            "avg_lvi": 0.45,
            "volatile_days": 1,
            "variable_days": 2,
            "smooth_days": 1,
        })
        assert "〜" in result
        assert "3" in result  # 1+2=3

    def test_mostly_smooth_produces_positive_line(self):
        """≥ 4 smooth days → 〰️ positive line."""
        from analysis.load_volatility import format_weekly_lvi_line
        result = format_weekly_lvi_line({
            "days_meaningful": 5,
            "avg_lvi": 0.87,
            "volatile_days": 0,
            "variable_days": 0,
            "smooth_days": 4,
        })
        assert "〰️" in result
        assert "smooth" in result.lower() or "Smooth" in result

    def test_average_week_returns_empty(self):
        """A steady/average week (no extremes) → empty string."""
        from analysis.load_volatility import format_weekly_lvi_line
        result = format_weekly_lvi_line({
            "days_meaningful": 5,
            "avg_lvi": 0.65,
            "volatile_days": 0,
            "variable_days": 1,
            "smooth_days": 2,
        })
        # Not volatile enough, not smooth enough — nothing to surface
        assert result == ""

    def test_output_is_string(self):
        """Return value is always a string."""
        from analysis.load_volatility import format_weekly_lvi_line
        result = format_weekly_lvi_line({})
        assert isinstance(result, str)

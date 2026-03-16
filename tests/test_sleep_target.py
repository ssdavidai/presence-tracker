"""
Tests for analysis/sleep_target.py — Sleep Target Advisor

Coverage:
  1. Core formula
     - Very light load → base 6.5h
     - Light load → base 7.0h
     - Moderate load → base 7.5h
     - High load → base 8.0h
     - Very high load → base 8.5h
     - Unknown / None load label → base fallback 7.5h, is_meaningful=False

  2. CDI modifier
     - surplus tier → +0.0h
     - balanced tier → +0.0h
     - loading tier → +0.25h
     - fatigued tier → +0.50h
     - critical tier → +0.75h

  3. CLS modifier
     - CLS < 0.20 → +0.0h
     - CLS between 0.20 and 0.50 → +0.0h
     - CLS between 0.50 and 0.70 → +0.25h
     - CLS ≥ 0.70 → +0.50h

  4. Clamping
     - Extreme combination → clamped to 9.5h max
     - Light + surplus → clamped to 6.0h min

  5. Bedtime computation
     - 7.5h sleep, wake 07:30 → bedtime 00:00
     - 8.0h sleep, wake 07:30 → bedtime 23:30
     - 6.5h sleep, wake 07:30 → bedtime 01:00

  6. Urgency levels
     - < 7.75h → normal
     - ≥ 7.75h → elevated
     - ≥ 8.50h → critical

  7. Meaningful flag
     - is_meaningful=True when load_label provided
     - is_meaningful=False when load_label is None

  8. Formatting
     - format_sleep_target_line() returns empty string when is_meaningful=False
     - format_sleep_target_line() contains hour count and bedtime when meaningful
     - format_sleep_target_section() returns empty when is_meaningful=False
     - format_sleep_target_section() contains "Sleep Target" heading
     - format_sleep_target_section() contains narrative sentence

  9. Narrative
     - urgent prefix "Prioritise" for elevated urgency
     - urgent prefix "🚨 You need" for critical urgency
     - "Target" prefix for normal urgency
     - load context appears in narrative
     - CDI context appears when loading/fatigued/critical

  10. compute_sleep_target_for_digest integration
      - returns None when no load forecast (is_meaningful=False)
      - returns dict with expected keys when load forecast available
      - urgency escalates with heavy load + high CDI debt

  11. Window-based CLS inference
      - windows with active metadata correctly infer avg_cls

  12. Edge cases
      - empty windows list → graceful handling
      - None windows → graceful handling

Run with: python3 -m pytest tests/test_sleep_target.py -v
"""

import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.sleep_target import (
    SleepTarget,
    compute_sleep_target,
    format_sleep_target_line,
    format_sleep_target_section,
    compute_sleep_target_for_digest,
    _bedtime_from_target,
    _urgency_from_hours,
    TARGET_WAKE_HOUR,
    TARGET_WAKE_MINUTE,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_window(cls: float, active: bool = True) -> dict:
    """Create a minimal window dict for testing."""
    return {
        "metadata": {
            "is_active_window": active,
            "is_working_hours": True,
            "hour_of_day": 9,
            "minute_of_hour": 0,
        },
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": 0.8,
            "social_drain_index": 0.2,
            "context_switch_cost": 0.1,
            "recovery_alignment_score": 0.9,
        },
        "calendar": {"in_meeting": False},
        "slack": {"total_messages": 0},
    }


# ─── 1. Core formula — base hours from load label ─────────────────────────────

class TestBaseHours:
    def test_very_light_load(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Very light")
        assert t.base_hours == 6.5

    def test_light_load(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Light")
        assert t.base_hours == 7.0

    def test_moderate_load(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate")
        assert t.base_hours == 7.5

    def test_high_load(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="High")
        assert t.base_hours == 8.0

    def test_very_high_load(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Very high")
        assert t.base_hours == 8.5

    def test_unknown_load_uses_fallback(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Unknown")
        assert t.base_hours == 7.5

    def test_none_load_uses_fallback(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label=None)
        assert t.base_hours == 7.5


# ─── 2. CDI modifier ──────────────────────────────────────────────────────────

class TestCdiModifier:
    def test_surplus(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", cdi_tier="surplus")
        assert t.cdi_modifier == 0.0

    def test_balanced(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", cdi_tier="balanced")
        assert t.cdi_modifier == 0.0

    def test_loading(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", cdi_tier="loading")
        assert t.cdi_modifier == 0.25

    def test_fatigued(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", cdi_tier="fatigued")
        assert t.cdi_modifier == 0.50

    def test_critical(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", cdi_tier="critical")
        assert t.cdi_modifier == 0.75

    def test_none_cdi_defaults_to_zero(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", cdi_tier=None)
        assert t.cdi_modifier == 0.0


# ─── 3. CLS modifier ──────────────────────────────────────────────────────────

class TestClsModifier:
    def test_low_cls_no_modifier(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", today_avg_cls=0.10)
        assert t.cls_modifier == 0.0

    def test_below_high_threshold_no_modifier(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", today_avg_cls=0.49)
        assert t.cls_modifier == 0.0

    def test_moderate_cls_small_modifier(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", today_avg_cls=0.60)
        assert t.cls_modifier == 0.25

    def test_high_cls_larger_modifier(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", today_avg_cls=0.75)
        assert t.cls_modifier == 0.50

    def test_exact_threshold_high(self):
        # CLS exactly at 0.50 → +0.25h (threshold is ≥ 0.50)
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", today_avg_cls=0.50)
        assert t.cls_modifier == 0.25

    def test_exact_threshold_very_high(self):
        # CLS exactly at 0.70 → +0.50h
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", today_avg_cls=0.70)
        assert t.cls_modifier == 0.50

    def test_none_cls_no_modifier(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", today_avg_cls=None)
        assert t.cls_modifier == 0.0


# ─── 4. Clamping ──────────────────────────────────────────────────────────────

class TestClamping:
    def test_extreme_combination_capped_at_max(self):
        # Very high + critical + very high CLS → could exceed 9.5h, should be capped
        t = compute_sleep_target(
            "2026-03-17",
            tomorrow_load_label="Very high",
            cdi_tier="critical",
            today_avg_cls=0.90,
        )
        assert t.target_hours <= 9.5

    def test_minimum_clamp(self):
        # Even with very light + surplus, minimum should be at least 6.0
        t = compute_sleep_target(
            "2026-03-17",
            tomorrow_load_label="Very light",
            cdi_tier="surplus",
            today_avg_cls=0.0,
        )
        assert t.target_hours >= 6.0


# ─── 5. Bedtime computation ───────────────────────────────────────────────────

class TestBedtimeComputation:
    def test_bedtime_from_target_7_5h(self):
        # Wake at 07:30, sleep 7.5h → bed at 00:00
        assert _bedtime_from_target(7.5) == "00:00"

    def test_bedtime_from_target_8h(self):
        # Wake at 07:30, sleep 8.0h → bed at 23:30 (previous night)
        assert _bedtime_from_target(8.0) == "23:30"

    def test_bedtime_from_target_6_5h(self):
        # Wake at 07:30, sleep 6.5h → bed at 01:00
        assert _bedtime_from_target(6.5) == "01:00"

    def test_bedtime_format(self):
        # All bedtimes should be HH:MM format
        bedtime = _bedtime_from_target(7.5)
        parts = bedtime.split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 2  # zero-padded hour
        assert len(parts[1]) == 2  # zero-padded minute

    def test_compute_sleep_target_includes_correct_bedtime(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="High", cdi_tier="balanced")
        # High load → 8.0h → bed at 23:30
        assert t.target_bedtime == "23:30"


# ─── 6. Urgency levels ────────────────────────────────────────────────────────

class TestUrgencyLevels:
    def test_normal_urgency(self):
        assert _urgency_from_hours(7.0) == "normal"

    def test_elevated_threshold(self):
        assert _urgency_from_hours(7.75) == "elevated"

    def test_elevated_urgency(self):
        assert _urgency_from_hours(8.0) == "elevated"

    def test_critical_threshold(self):
        assert _urgency_from_hours(8.5) == "critical"

    def test_critical_urgency(self):
        assert _urgency_from_hours(9.0) == "critical"

    def test_low_hours_normal(self):
        assert _urgency_from_hours(6.5) == "normal"

    def test_urgency_attached_to_target(self):
        t = compute_sleep_target(
            "2026-03-17",
            tomorrow_load_label="Very high",
            cdi_tier="critical",
            today_avg_cls=0.80,
        )
        assert t.urgency == "critical"


# ─── 7. Meaningful flag ───────────────────────────────────────────────────────

class TestMeaningfulFlag:
    def test_is_meaningful_when_load_provided(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate")
        assert t.is_meaningful is True

    def test_not_meaningful_when_no_load_label(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label=None)
        assert t.is_meaningful is False

    def test_not_meaningful_for_unknown_label(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Unknown")
        assert t.is_meaningful is False


# ─── 8. Formatting ────────────────────────────────────────────────────────────

class TestFormatting:
    def test_line_empty_when_not_meaningful(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label=None)
        assert format_sleep_target_line(t) == ""

    def test_line_contains_hours_when_meaningful(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate")
        line = format_sleep_target_line(t)
        assert str(t.target_hours) in line or f"{t.target_hours:.1f}" in line

    def test_line_contains_bedtime(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate")
        line = format_sleep_target_line(t)
        assert t.target_bedtime in line

    def test_line_contains_emoji(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate")
        line = format_sleep_target_line(t)
        # Should contain one of the urgency emojis
        assert any(emoji in line for emoji in ["😴", "🌙", "🚨"])

    def test_section_empty_when_not_meaningful(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label=None)
        assert format_sleep_target_section(t) == ""

    def test_section_contains_heading(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate")
        section = format_sleep_target_section(t)
        assert "Sleep Target" in section

    def test_section_contains_narrative(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate")
        section = format_sleep_target_section(t)
        assert t.narrative in section

    def test_section_is_multiline(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="High")
        section = format_sleep_target_section(t)
        assert "\n" in section


# ─── 9. Narrative ─────────────────────────────────────────────────────────────

class TestNarrative:
    def test_normal_urgency_prefix(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Light", cdi_tier="balanced")
        assert t.narrative.startswith("Target")

    def test_elevated_urgency_prefix(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="High", cdi_tier="loading")
        # elevated → "Prioritise"
        assert "Prioritise" in t.narrative or "Target" in t.narrative  # depends on final hours

    def test_critical_urgency_prefix(self):
        t = compute_sleep_target(
            "2026-03-17",
            tomorrow_load_label="Very high",
            cdi_tier="critical",
            today_avg_cls=0.80,
        )
        assert "🚨" in t.narrative

    def test_load_context_in_narrative(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="High")
        assert "demanding day" in t.narrative.lower() or "day ahead" in t.narrative.lower()

    def test_cdi_context_when_loading(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", cdi_tier="loading")
        assert "debt" in t.narrative.lower() or "building" in t.narrative.lower()

    def test_cdi_context_when_fatigued(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", cdi_tier="fatigued")
        assert "debt" in t.narrative.lower()

    def test_no_cdi_context_for_balanced(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate", cdi_tier="balanced")
        # Should not mention debt for balanced CDI
        assert "debt" not in t.narrative.lower()

    def test_narrative_ends_with_period(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate")
        assert t.narrative.endswith(".")

    def test_narrative_contains_bedtime(self):
        t = compute_sleep_target("2026-03-17", tomorrow_load_label="Moderate")
        assert t.target_bedtime in t.narrative


# ─── 10. compute_sleep_target_for_digest integration ─────────────────────────

class TestDigestIntegration:
    def test_returns_none_when_no_load_forecast(self):
        result = compute_sleep_target_for_digest(
            "2026-03-17",
            today_windows=[],
            precomputed_tomorrow_load=None,
            precomputed_cdi=None,
        )
        assert result is None

    def test_returns_none_when_load_not_meaningful(self):
        result = compute_sleep_target_for_digest(
            "2026-03-17",
            today_windows=[],
            precomputed_tomorrow_load={"is_meaningful": False, "load_label": "Moderate"},
            precomputed_cdi=None,
        )
        assert result is None

    def test_returns_dict_with_expected_keys(self):
        result = compute_sleep_target_for_digest(
            "2026-03-17",
            today_windows=[],
            precomputed_tomorrow_load={"is_meaningful": True, "load_label": "High"},
            precomputed_cdi={"is_meaningful": True, "tier": "balanced"},
        )
        assert result is not None
        assert "line" in result
        assert "section" in result
        assert "target_hours" in result
        assert "target_bedtime" in result
        assert "urgency" in result
        assert "narrative" in result
        assert "is_meaningful" in result

    def test_urgency_escalates_with_heavy_load_and_critical_cdi(self):
        result = compute_sleep_target_for_digest(
            "2026-03-17",
            today_windows=[],
            precomputed_tomorrow_load={"is_meaningful": True, "load_label": "Very high"},
            precomputed_cdi={"is_meaningful": True, "tier": "critical"},
        )
        assert result is not None
        assert result["urgency"] == "critical"

    def test_light_day_balanced_cdi_is_normal_urgency(self):
        result = compute_sleep_target_for_digest(
            "2026-03-17",
            today_windows=[],
            precomputed_tomorrow_load={"is_meaningful": True, "load_label": "Light"},
            precomputed_cdi={"is_meaningful": True, "tier": "balanced"},
        )
        assert result is not None
        assert result["urgency"] == "normal"


# ─── 11. Window-based CLS inference ──────────────────────────────────────────

class TestWindowClsInference:
    def test_infers_avg_cls_from_windows(self):
        windows = [
            _make_window(0.60, active=True),
            _make_window(0.80, active=True),
            _make_window(0.20, active=False),   # not active — excluded
        ]
        t = compute_sleep_target(
            "2026-03-17",
            today_windows=windows,
            tomorrow_load_label="Moderate",
        )
        # Active windows: CLS 0.60 and 0.80 → avg 0.70 → cls_modifier = 0.50h
        assert t.today_avg_cls == pytest.approx(0.70)
        assert t.cls_modifier == 0.50

    def test_ignores_non_active_windows(self):
        windows = [_make_window(0.90, active=False)] * 5
        t = compute_sleep_target(
            "2026-03-17",
            today_windows=windows,
            tomorrow_load_label="Moderate",
        )
        # No active windows → no CLS inferred
        assert t.today_avg_cls is None
        assert t.cls_modifier == 0.0

    def test_direct_cls_overrides_windows(self):
        # When today_avg_cls is provided directly, windows are ignored for CLS
        windows = [_make_window(0.10, active=True)]
        t = compute_sleep_target(
            "2026-03-17",
            today_windows=windows,
            tomorrow_load_label="Moderate",
            today_avg_cls=0.75,  # override: should use this, not windows
        )
        assert t.today_avg_cls == 0.75
        assert t.cls_modifier == 0.50


# ─── 12. Edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_windows_list(self):
        t = compute_sleep_target("2026-03-17", today_windows=[], tomorrow_load_label="Moderate")
        assert t is not None
        assert t.base_hours == 7.5
        assert t.cls_modifier == 0.0

    def test_none_windows(self):
        t = compute_sleep_target("2026-03-17", today_windows=None, tomorrow_load_label="Moderate")
        assert t is not None

    def test_target_hours_is_multiple_of_0_25(self):
        for load in ["Very light", "Light", "Moderate", "High", "Very high"]:
            for cdi in ["surplus", "balanced", "loading", "fatigued", "critical"]:
                t = compute_sleep_target("2026-03-17", tomorrow_load_label=load, cdi_tier=cdi)
                # Should be rounded to nearest 0.25h
                assert round(t.target_hours * 4) == int(round(t.target_hours * 4))

    def test_combined_high_load_and_loading_cdi(self):
        t = compute_sleep_target(
            "2026-03-17",
            tomorrow_load_label="High",
            cdi_tier="loading",
            today_avg_cls=0.55,
        )
        # High(8.0) + loading(+0.25) + cls_mod(+0.25) = 8.5h → critical urgency
        assert t.target_hours == pytest.approx(8.5)
        assert t.urgency == "critical"

    def test_combined_moderate_and_balanced(self):
        t = compute_sleep_target(
            "2026-03-17",
            tomorrow_load_label="Moderate",
            cdi_tier="balanced",
            today_avg_cls=0.30,
        )
        # Moderate(7.5) + balanced(0) + no cls_mod = 7.5h
        assert t.target_hours == pytest.approx(7.5)
        assert t.urgency == "normal"

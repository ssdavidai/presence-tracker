"""
Tests for the Daily Cognitive Budget (DCB) module.

Run with: python3 -m pytest tests/test_cognitive_budget.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from analysis.cognitive_budget import (
    CognitiveBudget,
    _budget_tier,
    _clamp,
    _hrv_modifier,
    _lerp,
    _narrative,
    _sleep_modifier,
    _whoop_recovery_tier,
    compute_cognitive_budget,
    format_budget_line,
    format_budget_section,
    DCB_FLOOR,
    DCB_CEILING,
    BASE_HOURS_BY_TIER,
    CDI_TIER_MODIFIERS,
    SLEEP_MODIFIER_MIN,
    SLEEP_MODIFIER_MAX,
    HRV_MODIFIER_MIN,
    HRV_MODIFIER_MAX,
)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _whoop(recovery=85.0, sleep_perf=80.0, hrv=70.0):
    return {
        "recovery_score": recovery,
        "sleep_performance": sleep_perf,
        "hrv_rmssd_milli": hrv,
    }


# ─── _lerp ───────────────────────────────────────────────────────────────────

class TestLerp:
    def test_midpoint(self):
        assert _lerp(0.0, 1.0, 0.5) == pytest.approx(0.5)

    def test_zero_t(self):
        assert _lerp(2.0, 4.0, 0.0) == pytest.approx(2.0)

    def test_one_t(self):
        assert _lerp(2.0, 4.0, 1.0) == pytest.approx(4.0)

    def test_clamps_below_zero(self):
        assert _lerp(0.0, 1.0, -0.5) == pytest.approx(0.0)

    def test_clamps_above_one(self):
        assert _lerp(0.0, 1.0, 1.5) == pytest.approx(1.0)


# ─── _clamp ──────────────────────────────────────────────────────────────────

class TestClamp:
    def test_within_range(self):
        assert _clamp(5.0, 1.0, 9.0) == 5.0

    def test_below_floor(self):
        assert _clamp(-1.0, 1.0, 9.0) == 1.0

    def test_above_ceiling(self):
        assert _clamp(10.0, 1.0, 9.0) == 9.0


# ─── _sleep_modifier ─────────────────────────────────────────────────────────

class TestSleepModifier:
    def test_none_returns_neutral(self):
        assert _sleep_modifier(None) == pytest.approx(1.0)

    def test_below_low_returns_minimum(self):
        assert _sleep_modifier(30.0) == pytest.approx(SLEEP_MODIFIER_MIN)

    def test_above_high_returns_maximum(self):
        assert _sleep_modifier(95.0) == pytest.approx(SLEEP_MODIFIER_MAX)

    def test_at_low_boundary(self):
        assert _sleep_modifier(50.0) == pytest.approx(SLEEP_MODIFIER_MIN)

    def test_at_high_boundary(self):
        assert _sleep_modifier(90.0) == pytest.approx(SLEEP_MODIFIER_MAX)

    def test_midpoint_interpolates(self):
        # Sleep 70% is midway between 50 and 90 → midpoint of modifier range
        result = _sleep_modifier(70.0)
        expected = (SLEEP_MODIFIER_MIN + SLEEP_MODIFIER_MAX) / 2
        assert result == pytest.approx(expected)

    def test_monotone_increasing(self):
        prev = _sleep_modifier(40.0)
        for perf in [55, 60, 70, 80, 90, 100]:
            curr = _sleep_modifier(float(perf))
            assert curr >= prev, f"Expected monotone increase at {perf}%"
            prev = curr


# ─── _hrv_modifier ───────────────────────────────────────────────────────────

class TestHrvModifier:
    def test_none_hrv_returns_neutral(self):
        assert _hrv_modifier(None, 70.0) == pytest.approx(1.0)

    def test_none_baseline_returns_neutral(self):
        assert _hrv_modifier(70.0, None) == pytest.approx(1.0)

    def test_zero_baseline_returns_neutral(self):
        assert _hrv_modifier(70.0, 0.0) == pytest.approx(1.0)

    def test_hrv_equal_to_baseline(self):
        # ratio = 1.0 → should be between min and max (interpolated midpoint)
        result = _hrv_modifier(70.0, 70.0)
        # ratio 1.0 maps to t = (1.0 - 0.8) / (1.2 - 0.8) = 0.5
        expected = _lerp(HRV_MODIFIER_MIN, HRV_MODIFIER_MAX, 0.5)
        assert result == pytest.approx(expected)

    def test_hrv_far_below_baseline(self):
        # ratio = 0.5 < 0.80 → minimum modifier
        result = _hrv_modifier(35.0, 70.0)
        assert result == pytest.approx(HRV_MODIFIER_MIN)

    def test_hrv_far_above_baseline(self):
        # ratio = 2.0 > 1.20 → maximum modifier
        result = _hrv_modifier(140.0, 70.0)
        assert result == pytest.approx(HRV_MODIFIER_MAX)

    def test_hrv_modifier_bounded(self):
        for hrv_ratio in [0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0]:
            result = _hrv_modifier(hrv_ratio * 70.0, 70.0)
            assert HRV_MODIFIER_MIN <= result <= HRV_MODIFIER_MAX


# ─── _whoop_recovery_tier ────────────────────────────────────────────────────

class TestWhoopRecoveryTier:
    def test_none_returns_unknown(self):
        assert _whoop_recovery_tier(None) == "unknown"

    def test_below_34_is_recovery(self):
        assert _whoop_recovery_tier(20.0) == "recovery"
        assert _whoop_recovery_tier(33.9) == "recovery"

    def test_34_to_49_is_low(self):
        assert _whoop_recovery_tier(34.0) == "low"
        assert _whoop_recovery_tier(49.9) == "low"

    def test_50_to_66_is_moderate(self):
        assert _whoop_recovery_tier(50.0) == "moderate"
        assert _whoop_recovery_tier(66.9) == "moderate"

    def test_67_to_79_is_good(self):
        assert _whoop_recovery_tier(67.0) == "good"
        assert _whoop_recovery_tier(79.9) == "good"

    def test_80_and_above_is_peak(self):
        assert _whoop_recovery_tier(80.0) == "peak"
        assert _whoop_recovery_tier(100.0) == "peak"


# ─── _budget_tier ────────────────────────────────────────────────────────────

class TestBudgetTier:
    def test_high_hours_is_peak(self):
        tier, label = _budget_tier(8.5)
        assert tier == "peak"

    def test_good_range(self):
        tier, _ = _budget_tier(6.0)
        assert tier == "good"

    def test_moderate_range(self):
        tier, _ = _budget_tier(4.5)
        assert tier == "moderate"

    def test_low_range(self):
        tier, _ = _budget_tier(2.5)
        assert tier == "low"

    def test_recovery_range(self):
        tier, _ = _budget_tier(1.5)
        assert tier == "recovery"

    def test_boundary_at_5_5(self):
        # 5.5 is the boundary between moderate and good
        tier, _ = _budget_tier(5.5)
        assert tier == "good"

    def test_just_below_5_5(self):
        tier, _ = _budget_tier(5.4)
        assert tier == "moderate"


# ─── _narrative ──────────────────────────────────────────────────────────────

class TestNarrative:
    def test_contains_hours(self):
        text = _narrative(6.0, "good", 85.0, "balanced", 80.0)
        assert "6.0" in text

    def test_contains_recovery_score(self):
        text = _narrative(6.0, "good", 85.0, "balanced", 80.0)
        assert "85%" in text

    def test_none_recovery_fallback(self):
        text = _narrative(5.0, "moderate", None, None, None)
        assert "no WHOOP" in text

    def test_poor_sleep_mention(self):
        # Sleep performance < 65 should add a note
        text = _narrative(4.0, "low", 60.0, "loading", 50.0)
        assert "50%" in text

    def test_recovery_tier_adds_warning(self):
        text = _narrative(1.5, "recovery", 30.0, "critical", 50.0)
        assert "costs tomorrow" in text.lower() or "protect" in text.lower()

    def test_low_tier_adds_front_load_tip(self):
        text = _narrative(3.0, "low", 40.0, "loading", 60.0)
        assert "front-load" in text.lower() or "first available" in text.lower()


# ─── compute_cognitive_budget ────────────────────────────────────────────────

class TestComputeCognitiveBudget:
    def test_peak_recovery_produces_high_budget(self):
        budget = compute_cognitive_budget("2026-03-14", _whoop(recovery=90.0, sleep_perf=85.0))
        assert budget.dcb_hours >= 7.0
        assert budget.tier == "peak"

    def test_low_recovery_produces_low_budget(self):
        budget = compute_cognitive_budget("2026-03-14", _whoop(recovery=30.0, sleep_perf=60.0))
        assert budget.dcb_hours <= 3.5

    def test_critical_cdi_reduces_budget(self):
        base = compute_cognitive_budget(
            "2026-03-14", _whoop(recovery=80.0, sleep_perf=80.0),
            cdi_tier="balanced"
        )
        critical = compute_cognitive_budget(
            "2026-03-14", _whoop(recovery=80.0, sleep_perf=80.0),
            cdi_tier="critical"
        )
        assert critical.dcb_hours < base.dcb_hours

    def test_surplus_cdi_increases_budget(self):
        base = compute_cognitive_budget(
            "2026-03-14", _whoop(recovery=70.0, sleep_perf=75.0),
            cdi_tier="balanced"
        )
        surplus = compute_cognitive_budget(
            "2026-03-14", _whoop(recovery=70.0, sleep_perf=75.0),
            cdi_tier="surplus"
        )
        assert surplus.dcb_hours > base.dcb_hours

    def test_no_whoop_returns_not_meaningful(self):
        budget = compute_cognitive_budget("2026-03-14", None)
        assert not budget.is_meaningful

    def test_no_whoop_still_returns_estimate(self):
        budget = compute_cognitive_budget("2026-03-14", None)
        assert DCB_FLOOR <= budget.dcb_hours <= DCB_CEILING

    def test_dcb_bounded_by_floor_and_ceiling(self):
        # Even extreme inputs should stay within bounds
        extreme = compute_cognitive_budget(
            "2026-03-14",
            _whoop(recovery=100.0, sleep_perf=100.0, hrv=200.0),
            cdi_tier="surplus",
            hrv_baseline=50.0,
        )
        assert extreme.dcb_hours <= DCB_CEILING
        extreme_low = compute_cognitive_budget(
            "2026-03-14",
            _whoop(recovery=1.0, sleep_perf=10.0, hrv=10.0),
            cdi_tier="critical",
            hrv_baseline=100.0,
        )
        assert extreme_low.dcb_hours >= DCB_FLOOR

    def test_dcb_low_lte_point_lte_high(self):
        budget = compute_cognitive_budget("2026-03-14", _whoop(recovery=75.0))
        assert budget.dcb_low <= budget.dcb_hours <= budget.dcb_high

    def test_all_fields_present(self):
        budget = compute_cognitive_budget("2026-03-14", _whoop())
        assert budget.date_str == "2026-03-14"
        assert budget.tier in ("peak", "good", "moderate", "low", "recovery")
        assert budget.label
        assert budget.narrative
        assert budget.guidance

    def test_high_hrv_vs_baseline_increases_budget(self):
        budget_low_hrv = compute_cognitive_budget(
            "2026-03-14", _whoop(recovery=75.0, hrv=50.0), hrv_baseline=70.0
        )
        budget_high_hrv = compute_cognitive_budget(
            "2026-03-14", _whoop(recovery=75.0, hrv=90.0), hrv_baseline=70.0
        )
        assert budget_high_hrv.dcb_hours >= budget_low_hrv.dcb_hours

    def test_sleep_modifier_applied_correctly(self):
        budget_good_sleep = compute_cognitive_budget(
            "2026-03-14", _whoop(recovery=75.0, sleep_perf=90.0)
        )
        budget_poor_sleep = compute_cognitive_budget(
            "2026-03-14", _whoop(recovery=75.0, sleep_perf=40.0)
        )
        assert budget_good_sleep.dcb_hours > budget_poor_sleep.dcb_hours

    def test_modifiers_stored_correctly(self):
        budget = compute_cognitive_budget(
            "2026-03-14",
            _whoop(recovery=80.0, sleep_perf=80.0),
            cdi_tier="loading",
        )
        assert budget.cdi_tier == "loading"
        assert budget.cdi_modifier == pytest.approx(CDI_TIER_MODIFIERS["loading"])

    def test_to_dict_serializable(self):
        budget = compute_cognitive_budget("2026-03-14", _whoop())
        d = budget.to_dict()
        assert isinstance(d, dict)
        assert "dcb_hours" in d
        assert "tier" in d
        assert "narrative" in d

    def test_recovery_score_in_budget(self):
        budget = compute_cognitive_budget("2026-03-14", _whoop(recovery=78.0))
        assert budget.recovery_score == pytest.approx(78.0)


# ─── Formatter tests ──────────────────────────────────────────────────────────

class TestFormatBudgetLine:
    def _make_budget(self, dcb_hours=6.0, tier="good", label="Strong day",
                     recovery=80.0, cdi_tier="balanced", meaningful=True):
        return CognitiveBudget(
            date_str="2026-03-14",
            dcb_hours=dcb_hours,
            dcb_low=max(DCB_FLOOR, dcb_hours - 0.5),
            dcb_high=min(DCB_CEILING, dcb_hours + 0.5),
            tier=tier,
            label=label,
            base_hours=6.5,
            recovery_score=recovery,
            sleep_performance=80.0,
            sleep_modifier=1.0,
            cdi_tier=cdi_tier,
            cdi_modifier=1.0,
            hrv_modifier=1.0,
            narrative="Test narrative.",
            guidance="Test guidance.",
            is_meaningful=meaningful,
        )

    def test_not_meaningful_returns_empty(self):
        budget = self._make_budget(meaningful=False)
        assert format_budget_line(budget) == ""

    def test_contains_hours(self):
        budget = self._make_budget(dcb_hours=6.0)
        line = format_budget_line(budget)
        assert "6.0" in line

    def test_contains_label(self):
        budget = self._make_budget(tier="good", label="Strong day")
        line = format_budget_line(budget)
        assert "Strong day" in line

    def test_contains_recovery_score(self):
        budget = self._make_budget(recovery=80.0)
        line = format_budget_line(budget)
        assert "80%" in line

    def test_contains_cdi_tier(self):
        budget = self._make_budget(cdi_tier="loading")
        line = format_budget_line(budget)
        assert "loading" in line

    def test_range_shown_when_meaningful(self):
        budget = self._make_budget(dcb_hours=6.0)
        line = format_budget_line(budget)
        # Should show the range (5.5–6.5)
        assert "5.5" in line
        assert "6.5" in line

    def test_emoji_present(self):
        budget = self._make_budget()
        line = format_budget_line(budget)
        assert "🧠" in line


class TestFormatBudgetSection:
    def _make_budget(self, dcb_hours=5.0, tier="moderate", label="Steady",
                     recovery=65.0, meaningful=True):
        return CognitiveBudget(
            date_str="2026-03-14",
            dcb_hours=dcb_hours,
            dcb_low=max(DCB_FLOOR, dcb_hours - 0.5),
            dcb_high=min(DCB_CEILING, dcb_hours + 0.5),
            tier=tier,
            label=label,
            base_hours=5.0,
            recovery_score=recovery,
            sleep_performance=75.0,
            sleep_modifier=1.0,
            cdi_tier="balanced",
            cdi_modifier=1.0,
            hrv_modifier=1.0,
            narrative="Test narrative sentence.",
            guidance="Some guidance here.",
            is_meaningful=meaningful,
        )

    def test_not_meaningful_returns_empty(self):
        budget = self._make_budget(meaningful=False)
        assert format_budget_section(budget) == ""

    def test_contains_hours_in_header(self):
        budget = self._make_budget(dcb_hours=5.0)
        section = format_budget_section(budget)
        assert "5.0" in section

    def test_contains_label_in_header(self):
        budget = self._make_budget(label="Steady")
        section = format_budget_section(budget)
        assert "Steady" in section

    def test_contains_narrative(self):
        budget = self._make_budget()
        section = format_budget_section(budget)
        assert "Test narrative sentence." in section

    def test_contains_guidance(self):
        budget = self._make_budget()
        section = format_budget_section(budget)
        assert "Some guidance here." in section

    def test_multi_line_output(self):
        budget = self._make_budget()
        section = format_budget_section(budget)
        assert "\n" in section


# ─── Integration: full pipeline ──────────────────────────────────────────────

class TestIntegration:
    """End-to-end consistency checks: modifiers compose correctly."""

    def test_all_tiers_produce_distinct_budgets(self):
        """Different WHOOP recovery tiers should produce distinct estimates."""
        whoop_by_tier = {
            "recovery": 25.0,
            "low": 40.0,
            "moderate": 55.0,
            "good": 72.0,
            "peak": 85.0,
        }
        budgets = {}
        for tier, recovery in whoop_by_tier.items():
            b = compute_cognitive_budget("2026-03-14", _whoop(recovery=recovery, sleep_perf=75.0))
            budgets[tier] = b.dcb_hours

        # Each tier should produce more hours than the tier below it
        assert budgets["recovery"] < budgets["low"]
        assert budgets["low"] < budgets["moderate"]
        assert budgets["moderate"] < budgets["good"]
        assert budgets["good"] < budgets["peak"]

    def test_all_cdi_tiers_produce_distinct_budgets(self):
        """Different CDI tiers should produce distinct budget estimates."""
        cdi_tiers = ["critical", "fatigued", "loading", "balanced", "surplus"]
        results = []
        for cdi in cdi_tiers:
            b = compute_cognitive_budget(
                "2026-03-14",
                _whoop(recovery=75.0, sleep_perf=75.0),
                cdi_tier=cdi,
            )
            results.append(b.dcb_hours)
        # Each subsequent tier should give >= hours than the previous (more severe) tier
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1], (
                f"CDI tier order violation: {cdi_tiers[i]}={results[i]:.1f} > "
                f"{cdi_tiers[i+1]}={results[i+1]:.1f}"
            )

    def test_realistic_good_day(self):
        """A 'good' day: 80% recovery, 80% sleep, balanced CDI → ~6h"""
        b = compute_cognitive_budget(
            "2026-03-14",
            _whoop(recovery=80.0, sleep_perf=80.0),
            cdi_tier="balanced",
        )
        assert b.tier in ("good", "peak")
        assert b.dcb_hours >= 5.5

    def test_realistic_hard_week_day(self):
        """After a hard week: 55% recovery, 60% sleep, fatigued CDI → ~3h"""
        b = compute_cognitive_budget(
            "2026-03-14",
            _whoop(recovery=55.0, sleep_perf=60.0),
            cdi_tier="fatigued",
        )
        assert b.tier in ("low", "moderate")
        assert b.dcb_hours <= 5.0

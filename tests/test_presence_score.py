"""
Tests for the Daily Presence Score (DPS) module.

Run with: python3 -m pytest tests/test_presence_score.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from analysis.presence_score import (
    compute_presence_score,
    format_presence_score_line,
    format_presence_score_block,
    _dps_tier,
    _load_quality_component,
    _empty_score,
    enrich_summary_with_dps,
    get_historical_dps,
    compute_dps_trend,
    DPS_EXCEPTIONAL,
    DPS_STRONG,
    DPS_GOOD,
    DPS_MODERATE,
    DPS_LOW,
    WEIGHT_FOCUS_QUALITY,
    WEIGHT_RECOVERY_ALIGNMENT,
    WEIGHT_LOAD_QUALITY,
    WEIGHT_SOCIAL_SUSTAIN,
    WEIGHT_ACTIVE_ENGAGEMENT,
    MIN_WORKING_WINDOWS,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_window(
    hour: int = 9,
    is_working: bool = True,
    is_active: bool = True,
    in_meeting: bool = False,
    meeting_attendees: int = 0,
    messages_sent: int = 0,
    messages_received: int = 0,
    cls: float = 0.20,
    fdi: float = 0.80,
    sdi: float = 0.10,
    csc: float = 0.15,
    ras: float = 0.85,
    recovery: float = 80.0,
    hrv: float = 70.0,
) -> dict:
    """Create a minimal valid window dict for testing."""
    return {
        "date": "2026-03-14",
        "window_start": f"2026-03-14T{hour:02d}:00:00+01:00",
        "metadata": {
            "is_working_hours": is_working,
            "is_active_window": is_active,
            "hour_of_day": hour,
            "minute_of_hour": 0,
        },
        "calendar": {
            "in_meeting": in_meeting,
            "meeting_attendees": meeting_attendees,
            "meeting_title": "",
            "meetings_count": 1 if in_meeting else 0,
        },
        "slack": {
            "messages_sent": messages_sent,
            "messages_received": messages_received,
            "total_messages": messages_sent + messages_received,
            "threads_active": 0,
            "channels_active": 0,
        },
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "resting_heart_rate": 55.0,
            "sleep_performance": 85.0,
            "sleep_hours": 7.5,
        },
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "context_switch_cost": csc,
            "recovery_alignment_score": ras,
        },
    }


def _make_day(
    n_windows: int = 10,
    cls: float = 0.25,
    fdi: float = 0.75,
    sdi: float = 0.15,
    csc: float = 0.20,
    ras: float = 0.80,
    n_active: int = 8,
) -> list[dict]:
    """Create a list of N working-hour windows for one day."""
    windows = []
    for i in range(n_windows):
        hour = 9 + (i % 8)
        is_active = i < n_active
        windows.append(_make_window(
            hour=hour,
            is_working=True,
            is_active=is_active,
            messages_received=5 if is_active else 0,
            cls=cls,
            fdi=fdi,
            sdi=sdi,
            csc=csc,
            ras=ras,
        ))
    return windows


# ─── Tier mapping ─────────────────────────────────────────────────────────────

class TestDpsTier:
    def test_exceptional(self):
        assert _dps_tier(DPS_EXCEPTIONAL) == "exceptional"
        assert _dps_tier(100.0) == "exceptional"

    def test_strong(self):
        assert _dps_tier(DPS_STRONG) == "strong"
        assert _dps_tier(DPS_EXCEPTIONAL - 0.1) == "strong"

    def test_good(self):
        assert _dps_tier(DPS_GOOD) == "good"
        assert _dps_tier(DPS_STRONG - 0.1) == "good"

    def test_moderate(self):
        assert _dps_tier(DPS_MODERATE) == "moderate"
        assert _dps_tier(DPS_GOOD - 0.1) == "moderate"

    def test_low(self):
        assert _dps_tier(DPS_LOW) == "low"
        assert _dps_tier(DPS_MODERATE - 0.1) == "low"

    def test_poor(self):
        assert _dps_tier(0.0) == "poor"
        assert _dps_tier(DPS_LOW - 0.1) == "poor"


# ─── Load quality component ───────────────────────────────────────────────────

class TestLoadQualityComponent:
    def test_low_load_high_ras_returns_high(self):
        # Very low load + perfect recovery alignment → near-maximum component
        q = _load_quality_component(avg_cls=0.10, avg_ras=0.95)
        assert q > 0.90, f"Expected > 0.90, got {q}"

    def test_high_load_reduces_quality(self):
        # Load well above sweet spot → significant penalty
        q = _load_quality_component(avg_cls=0.90, avg_ras=0.80)
        assert q < 0.40, f"Expected < 0.40 for very high CLS, got {q}"

    def test_moderate_load_moderate_ras(self):
        # Moderate CLS + moderate RAS → moderate quality
        q = _load_quality_component(avg_cls=0.40, avg_ras=0.60)
        assert 0.30 <= q <= 0.80

    def test_output_range(self):
        for cls in [0.0, 0.25, 0.55, 0.80, 1.0]:
            for ras in [0.0, 0.50, 1.0]:
                q = _load_quality_component(cls, ras)
                assert 0.0 <= q <= 1.0, f"Out of range for cls={cls}, ras={ras}: {q}"

    def test_ras_zero_caps_quality(self):
        # Zero RAS → load quality = 0 regardless of CLS
        q = _load_quality_component(avg_cls=0.10, avg_ras=0.0)
        assert q == 0.0


# ─── Core computation ─────────────────────────────────────────────────────────

class TestComputePresenceScore:
    def test_empty_list_returns_not_meaningful(self):
        score = compute_presence_score([])
        assert not score.is_meaningful

    def test_too_few_windows_not_meaningful(self):
        # Fewer than MIN_WORKING_WINDOWS working-hour windows
        windows = _make_day(n_windows=MIN_WORKING_WINDOWS - 1)
        score = compute_presence_score(windows)
        assert not score.is_meaningful

    def test_minimum_windows_is_meaningful(self):
        windows = _make_day(n_windows=MIN_WORKING_WINDOWS)
        score = compute_presence_score(windows)
        assert score.is_meaningful

    def test_output_range(self):
        # Standard day should produce a score in [0, 100]
        windows = _make_day(n_windows=20)
        score = compute_presence_score(windows)
        assert 0.0 <= score.dps <= 100.0

    def test_excellent_day_scores_high(self):
        # Perfect signals: high FDI, high RAS, low CLS, low SDI, all active
        windows = _make_day(
            n_windows=20,
            cls=0.05,   # very low load
            fdi=0.95,   # near-perfect focus
            sdi=0.02,   # almost no social drain
            csc=0.05,   # minimal fragmentation
            ras=0.95,   # excellent recovery alignment
            n_active=18,
        )
        score = compute_presence_score(windows)
        assert score.is_meaningful
        assert score.dps >= DPS_STRONG, f"Expected ≥ {DPS_STRONG}, got {score.dps}"

    def test_poor_day_scores_low(self):
        # Bad signals: low FDI, low RAS, high CLS, high SDI
        windows = _make_day(
            n_windows=20,
            cls=0.90,   # extreme load
            fdi=0.20,   # fragmented focus
            sdi=0.80,   # high social drain
            csc=0.70,   # high fragmentation
            ras=0.20,   # misaligned with recovery
            n_active=18,
        )
        score = compute_presence_score(windows)
        assert score.is_meaningful
        assert score.dps <= DPS_MODERATE, f"Expected ≤ {DPS_MODERATE}, got {score.dps}"

    def test_idle_day_has_low_active_engagement(self):
        # Day with no active windows → active_engagement = floor
        windows = _make_day(n_windows=20, n_active=0)
        score = compute_presence_score(windows)
        assert score.is_meaningful
        from analysis.presence_score import ACTIVE_FRACTION_FLOOR
        assert score.components["active_engagement"] == pytest.approx(
            ACTIVE_FRACTION_FLOOR, abs=0.01
        )

    def test_components_sum_to_weighted_dps(self):
        # Verify DPS = weighted sum of components × 100
        windows = _make_day(n_windows=20)
        score = compute_presence_score(windows)
        assert score.is_meaningful

        expected = (
            WEIGHT_FOCUS_QUALITY      * score.components["focus_quality"]      +
            WEIGHT_RECOVERY_ALIGNMENT * score.components["recovery_alignment"]  +
            WEIGHT_LOAD_QUALITY       * score.components["load_quality"]        +
            WEIGHT_SOCIAL_SUSTAIN     * score.components["social_sustain"]      +
            WEIGHT_ACTIVE_ENGAGEMENT  * score.components["active_engagement"]
        ) * 100.0

        assert abs(score.dps - round(expected, 1)) < 0.2, (
            f"DPS {score.dps} doesn't match weighted sum {expected:.1f}"
        )

    def test_correct_tier_for_score(self):
        windows = _make_day(n_windows=20)
        score = compute_presence_score(windows)
        assert score.tier == _dps_tier(score.dps)

    def test_date_is_populated(self):
        windows = _make_day(n_windows=20)
        score = compute_presence_score(windows)
        assert score.date == "2026-03-14"

    def test_missing_omi_data_graceful(self):
        # Windows without 'omi' key should not crash
        windows = _make_day(n_windows=20)
        for w in windows:
            if "omi" in w:
                del w["omi"]
        score = compute_presence_score(windows)
        assert score.is_meaningful

    def test_non_working_windows_ignored(self):
        # Only non-working windows → not meaningful
        windows = [
            _make_window(hour=2, is_working=False)
            for _ in range(20)
        ]
        score = compute_presence_score(windows)
        assert not score.is_meaningful

    def test_never_raises(self):
        # Even with garbage input, should not raise
        assert compute_presence_score([{"date": "2026-01-01"}]) is not None
        assert compute_presence_score([{}]) is not None

    def test_high_focus_day_has_high_focus_quality_component(self):
        # FDI=0.95, CSC=0.05 → focus_quality ≈ 0.9025
        windows = _make_day(n_windows=20, fdi=0.95, csc=0.05, n_active=18)
        score = compute_presence_score(windows)
        assert score.components["focus_quality"] > 0.80

    def test_high_sdi_reduces_social_sustain(self):
        # SDI=0.80 → social_sustain=0.20
        windows = _make_day(n_windows=20, sdi=0.80, n_active=18)
        score = compute_presence_score(windows)
        assert score.components["social_sustain"] < 0.30


# ─── Empty score ──────────────────────────────────────────────────────────────

class TestEmptyScore:
    def test_is_not_meaningful(self):
        s = _empty_score("2026-03-14")
        assert not s.is_meaningful

    def test_has_date(self):
        s = _empty_score("2026-03-14")
        assert s.date == "2026-03-14"

    def test_dps_is_neutral(self):
        # Defaults to 50 (middle of range)
        s = _empty_score("2026-03-14")
        assert s.dps == 50.0


# ─── Formatting ───────────────────────────────────────────────────────────────

class TestFormatPresenceScoreLine:
    def test_meaningful_score_returns_non_empty(self):
        windows = _make_day(n_windows=20)
        score = compute_presence_score(windows)
        line = format_presence_score_line(score)
        assert len(line) > 0

    def test_not_meaningful_returns_empty(self):
        score = _empty_score("2026-03-14")
        score.is_meaningful = False
        line = format_presence_score_line(score)
        assert line == ""

    def test_contains_dps_value(self):
        windows = _make_day(n_windows=20)
        score = compute_presence_score(windows)
        line = format_presence_score_line(score)
        assert str(int(score.dps)) in line

    def test_contains_tier_label(self):
        windows = _make_day(n_windows=20)
        score = compute_presence_score(windows)
        line = format_presence_score_line(score)
        assert score.tier.capitalize() in line or score.tier in line.lower()

    def test_exceptional_has_star_emoji(self):
        windows = _make_day(
            n_windows=20, cls=0.05, fdi=0.95, sdi=0.02, csc=0.05, ras=0.95, n_active=18
        )
        score = compute_presence_score(windows)
        if score.tier == "exceptional":
            line = format_presence_score_line(score)
            assert "🌟" in line


class TestFormatPresenceScoreBlock:
    def test_meaningful_score_returns_multiline(self):
        windows = _make_day(n_windows=20)
        score = compute_presence_score(windows)
        block = format_presence_score_block(score)
        assert "\n" in block

    def test_not_meaningful_returns_fallback_string(self):
        score = _empty_score("2026-03-14")
        block = format_presence_score_block(score)
        assert len(block) > 0

    def test_block_contains_all_components(self):
        windows = _make_day(n_windows=20)
        score = compute_presence_score(windows)
        block = format_presence_score_block(score)
        assert "Focus quality" in block
        assert "Recovery alignment" in block
        assert "Load quality" in block
        assert "Social sustain" in block
        assert "Active engagement" in block


# ─── Summary enrichment ───────────────────────────────────────────────────────

class TestEnrichSummaryWithDps:
    def test_injects_presence_score_key(self):
        windows = _make_day(n_windows=20)
        summary = {}
        result = enrich_summary_with_dps(summary, windows)
        assert "presence_score" in result

    def test_presence_score_has_required_keys(self):
        windows = _make_day(n_windows=20)
        summary = {}
        result = enrich_summary_with_dps(summary, windows)
        ps = result.get("presence_score", {})
        assert "dps" in ps
        assert "tier" in ps
        assert "components" in ps

    def test_insufficient_data_omits_key(self):
        # Too few windows → no presence_score key (not meaningful)
        windows = _make_day(n_windows=2)
        summary = {}
        result = enrich_summary_with_dps(summary, windows)
        assert "presence_score" not in result

    def test_existing_keys_preserved(self):
        windows = _make_day(n_windows=20)
        summary = {"date": "2026-03-14", "working_hours": 8}
        result = enrich_summary_with_dps(summary, windows)
        assert result["date"] == "2026-03-14"
        assert result["working_hours"] == 8

    def test_never_crashes(self):
        # Should not raise even with empty windows
        result = enrich_summary_with_dps({}, [])
        assert isinstance(result, dict)


# ─── Digest integration ───────────────────────────────────────────────────────

class TestDigestIntegration:
    def test_digest_includes_presence_score(self):
        """compute_digest() should return a presence_score key."""
        from analysis.daily_digest import compute_digest
        windows = _make_day(n_windows=20)
        digest = compute_digest(windows)
        # presence_score may be None if insufficient data, but key should exist
        assert "presence_score" in digest

    def test_digest_presence_score_has_dps(self):
        """When enough data exists, digest['presence_score']['dps'] should be a float."""
        from analysis.daily_digest import compute_digest
        windows = _make_day(n_windows=20)
        digest = compute_digest(windows)
        ps = digest.get("presence_score")
        if ps is not None:
            assert isinstance(ps["dps"], float)
            assert 0.0 <= ps["dps"] <= 100.0

    def test_format_digest_message_includes_dps_line(self):
        """The Slack DM should have the DPS headline when score is meaningful."""
        from analysis.daily_digest import compute_digest, format_digest_message
        windows = _make_day(n_windows=20)
        digest = compute_digest(windows)
        msg = format_digest_message(digest)
        # Only check if presence_score was computed
        if digest.get("presence_score"):
            assert "DPS" in msg


# ─── Chunker integration ──────────────────────────────────────────────────────

class TestChunkerIntegration:
    def test_summarize_day_includes_presence_score(self):
        """summarize_day() should inject presence_score into the summary."""
        from engine.chunker import summarize_day
        windows = _make_day(n_windows=20)
        summary = summarize_day(windows)
        # DPS enrichment should have run
        assert "presence_score" in summary

    def test_summarize_day_dps_is_valid(self):
        """The presence_score in summary should have dps and tier."""
        from engine.chunker import summarize_day
        windows = _make_day(n_windows=20)
        summary = summarize_day(windows)
        ps = summary.get("presence_score", {})
        if ps:
            assert "dps" in ps
            assert isinstance(ps["dps"], float)
            assert ps["tier"] in ("exceptional", "strong", "good", "moderate", "low", "poor")


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_working_windows_with_no_active(self):
        """Day with working windows but no active ones should still score."""
        windows = _make_day(n_windows=20, n_active=0)
        score = compute_presence_score(windows)
        assert score.is_meaningful
        # With no active windows, FDI/SDI/CSC use fallbacks but score is valid
        assert 0.0 <= score.dps <= 100.0

    def test_single_window_below_minimum(self):
        windows = [_make_window()]
        score = compute_presence_score(windows)
        assert not score.is_meaningful

    def test_components_all_in_0_1_range(self):
        for cls in [0.05, 0.35, 0.70, 0.95]:
            for ras in [0.20, 0.60, 0.90]:
                windows = _make_day(n_windows=20, cls=cls, ras=ras)
                score = compute_presence_score(windows)
                if score.is_meaningful:
                    for key, val in score.components.items():
                        if val is not None:
                            assert 0.0 <= val <= 1.0, (
                                f"Component {key}={val} out of [0,1] for cls={cls}, ras={ras}"
                            )

    def test_dps_always_between_0_and_100(self):
        # Extreme inputs should still produce valid DPS
        for cls_val in [0.0, 0.5, 1.0]:
            for fdi_val in [0.0, 0.5, 1.0]:
                windows = _make_day(n_windows=20, cls=cls_val, fdi=fdi_val)
                score = compute_presence_score(windows)
                if score.is_meaningful:
                    assert 0.0 <= score.dps <= 100.0, (
                        f"DPS {score.dps} out of range for cls={cls_val}, fdi={fdi_val}"
                    )

    def test_windows_with_none_whoop_data(self):
        """Missing WHOOP data should degrade gracefully, not crash."""
        windows = _make_day(n_windows=20)
        for w in windows:
            w["whoop"]["recovery_score"] = None
            w["metrics"]["recovery_alignment_score"] = None
        # Should not raise
        score = compute_presence_score(windows)
        assert score is not None

    def test_weights_sum_to_one(self):
        """Sanity check: defined weights must sum to exactly 1.0."""
        total = (
            WEIGHT_FOCUS_QUALITY +
            WEIGHT_RECOVERY_ALIGNMENT +
            WEIGHT_LOAD_QUALITY +
            WEIGHT_SOCIAL_SUSTAIN +
            WEIGHT_ACTIVE_ENGAGEMENT
        )
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0"

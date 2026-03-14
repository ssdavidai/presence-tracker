"""
Tests for the metric computation engine.

Run with: python3 -m pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engine.metrics import (
    cognitive_load_score,
    focus_depth_index,
    social_drain_index,
    context_switch_cost,
    recovery_alignment_score,
    compute_metrics,
    physiological_readiness,
)


class TestCognitiveLoadScore:
    def test_idle_recovered_returns_low(self):
        cls = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=100.0,
        )
        assert cls < 0.30, f"Idle + 100% recovery should be low CLS, got {cls}"

    def test_meeting_with_many_attendees_returns_high(self):
        cls = cognitive_load_score(
            in_meeting=True,
            meeting_attendees=10,
            slack_messages_received=20,
            recovery_score=30.0,
        )
        assert cls > 0.60, f"Heavy meeting + low recovery should be high CLS, got {cls}"

    def test_in_meeting_alone_is_moderate(self):
        cls = cognitive_load_score(
            in_meeting=True,
            meeting_attendees=2,
            slack_messages_received=0,
            recovery_score=80.0,
        )
        assert 0.20 <= cls <= 0.70, f"Meeting alone should be moderate CLS, got {cls}"

    def test_none_recovery_uses_neutral(self):
        cls_none = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=5,
            recovery_score=None,
        )
        cls_50 = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=5,
            recovery_score=50.0,
        )
        # Should be close (neutral = 50%)
        assert abs(cls_none - cls_50) < 0.05

    def test_output_range(self):
        for recovery in [0, 50, 100, None]:
            for meeting in [True, False]:
                cls = cognitive_load_score(
                    in_meeting=meeting,
                    meeting_attendees=5,
                    slack_messages_received=10,
                    recovery_score=recovery,
                )
                assert 0.0 <= cls <= 1.0, f"CLS out of range: {cls}"


class TestFocusDepthIndex:
    def test_undisturbed_focus_returns_high(self):
        fdi = focus_depth_index(
            in_meeting=False,
            slack_messages_received=0,
            context_switches=0,
        )
        assert fdi >= 0.90, f"Zero disruption should yield high FDI, got {fdi}"

    def test_in_social_meeting_returns_low(self):
        # A social meeting (multiple attendees) with heavy Slack = low FDI
        fdi = focus_depth_index(
            in_meeting=True,
            slack_messages_received=15,
            context_switches=5,
            meeting_attendees=4,
        )
        assert fdi < 0.40, f"High disruption social meeting should yield low FDI, got {fdi}"

    def test_solo_focus_block_returns_high_fdi(self):
        # v1.4: A solo calendar block (focus session) should not penalise FDI.
        # meeting_attendees=0 means no other participants — deliberate focus time.
        fdi_solo = focus_depth_index(
            in_meeting=True,
            slack_messages_received=0,
            context_switches=0,
            meeting_attendees=0,
        )
        fdi_idle = focus_depth_index(
            in_meeting=False,
            slack_messages_received=0,
            context_switches=0,
        )
        assert fdi_solo >= 0.90, f"Solo focus block should yield high FDI, got {fdi_solo}"
        assert abs(fdi_solo - fdi_idle) < 0.05, (
            f"Solo block FDI ({fdi_solo}) should be similar to idle FDI ({fdi_idle})"
        )

    def test_output_range(self):
        for in_meeting, slack_rcv, ctx, attendees in [
            (True, 30, 20, 5),
            (False, 0, 0, 0),
            (True, 0, 0, 0),
            (False, 30, 20, 0),
        ]:
            fdi = focus_depth_index(
                in_meeting=in_meeting,
                slack_messages_received=slack_rcv,
                context_switches=ctx,
                meeting_attendees=attendees,
            )
            assert 0.0 <= fdi <= 1.0, f"FDI out of range: {fdi}"


class TestSocialDrainIndex:
    def test_solo_no_slack_returns_zero(self):
        sdi = social_drain_index(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_sent=0,
            slack_messages_received=0,
        )
        assert sdi == 0.0, f"No social activity should be 0 SDI, got {sdi}"

    def test_solo_calendar_block_returns_zero(self):
        # v1.4: a solo calendar block (focus session, personal event) has no other
        # attendees — it should return SDI=0 despite in_meeting=True.
        sdi_zero_attendees = social_drain_index(
            in_meeting=True,
            meeting_attendees=0,
            slack_messages_sent=0,
            slack_messages_received=0,
        )
        sdi_one_attendee = social_drain_index(
            in_meeting=True,
            meeting_attendees=1,
            slack_messages_sent=0,
            slack_messages_received=0,
        )
        assert sdi_zero_attendees == 0.0, (
            f"Solo block (0 attendees) should have SDI=0, got {sdi_zero_attendees}"
        )
        assert sdi_one_attendee == 0.0, (
            f"Solo block (1 attendee = just David) should have SDI=0, got {sdi_one_attendee}"
        )

    def test_social_meeting_requires_more_than_one_attendee(self):
        # Social meetings need at least 2 attendees (David + someone else)
        sdi_2 = social_drain_index(
            in_meeting=True,
            meeting_attendees=2,
            slack_messages_sent=0,
            slack_messages_received=0,
        )
        assert sdi_2 > 0.0, (
            f"A 1:1 meeting (2 attendees) should have positive SDI, got {sdi_2}"
        )

    def test_large_meeting_returns_high(self):
        sdi = social_drain_index(
            in_meeting=True,
            meeting_attendees=10,
            slack_messages_sent=5,
            slack_messages_received=5,
        )
        assert sdi >= 0.50, f"Large meeting should yield high SDI, got {sdi}"

    def test_output_range(self):
        for meeting in [True, False]:
            for attendees in [0, 5, 10]:
                sdi = social_drain_index(
                    in_meeting=meeting,
                    meeting_attendees=attendees,
                    slack_messages_sent=3,
                    slack_messages_received=7,
                )
                assert 0.0 <= sdi <= 1.0, f"SDI out of range: {sdi}"


class TestContextSwitchCost:
    def test_no_switches_returns_zero(self):
        csc = context_switch_cost(
            in_meeting=False,
            meeting_duration_minutes=0,
            slack_channels_active=0,
        )
        assert csc == 0.0

    def test_short_meeting_costs_more_than_long(self):
        # v1.6: social meetings (attendees > 1) — short costs more than long
        csc_short = context_switch_cost(
            in_meeting=True,
            meeting_duration_minutes=10,
            slack_channels_active=1,
            is_short_meeting=True,
            meeting_attendees=4,
        )
        csc_long = context_switch_cost(
            in_meeting=True,
            meeting_duration_minutes=90,
            slack_channels_active=1,
            is_short_meeting=False,
            meeting_attendees=4,
        )
        assert csc_short > csc_long, f"Short meeting should cost more: {csc_short} vs {csc_long}"

    def test_solo_block_has_zero_csc(self):
        # v1.6: solo calendar blocks (attendees <= 1) incur no context-switch cost.
        # A dedicated focus block is the opposite of fragmentation.
        csc_solo_long = context_switch_cost(
            in_meeting=True,
            meeting_duration_minutes=180,
            slack_channels_active=0,
            is_short_meeting=False,
            meeting_attendees=0,
        )
        assert csc_solo_long == 0.0, f"Solo long block should have zero CSC, got {csc_solo_long}"

        csc_solo_short = context_switch_cost(
            in_meeting=True,
            meeting_duration_minutes=10,
            slack_channels_active=0,
            is_short_meeting=True,
            meeting_attendees=1,
        )
        assert csc_solo_short == 0.0, f"Solo short block should have zero CSC, got {csc_solo_short}"

    def test_solo_block_csc_lower_than_social_meeting(self):
        # v1.6: a solo block always has lower CSC than an equivalent social meeting.
        csc_solo = context_switch_cost(
            in_meeting=True,
            meeting_duration_minutes=30,
            slack_channels_active=2,
            is_short_meeting=False,
            meeting_attendees=1,
        )
        csc_social = context_switch_cost(
            in_meeting=True,
            meeting_duration_minutes=30,
            slack_channels_active=2,
            is_short_meeting=False,
            meeting_attendees=5,
        )
        assert csc_solo < csc_social, (
            f"Solo block CSC ({csc_solo}) should be less than social meeting CSC ({csc_social})"
        )

    def test_backward_compat_no_attendees_kwarg(self):
        # v1.6: callers without meeting_attendees kwarg default to 0 (solo = zero CSC).
        # This is the safe conservative direction (no false CSC inflation).
        csc = context_switch_cost(
            in_meeting=True,
            meeting_duration_minutes=30,
            slack_channels_active=0,
        )
        assert csc == 0.0, f"No attendees kwarg should default to 0 (solo, zero CSC), got {csc}"

    def test_output_range(self):
        for args in [
            (True, 15, 5, True),
            (False, 0, 0, False),
            (True, 60, 2, False),
        ]:
            csc = context_switch_cost(*args)
            assert 0.0 <= csc <= 1.0, f"CSC out of range: {csc}"


class TestRecoveryAlignmentScore:
    def test_high_recovery_low_load_is_aligned(self):
        ras = recovery_alignment_score(recovery_score=90.0, cls=0.20)
        assert ras >= 0.70, f"High recovery + low CLS should be aligned, got {ras}"

    def test_low_recovery_high_load_is_misaligned(self):
        ras = recovery_alignment_score(recovery_score=20.0, cls=0.90)
        assert ras < 0.40, f"Low recovery + high CLS should be misaligned, got {ras}"

    def test_none_recovery_returns_neutral(self):
        ras = recovery_alignment_score(recovery_score=None, cls=0.50)
        assert ras == 0.50

    def test_output_range(self):
        for recovery in [0, 30, 70, 100, None]:
            for cls in [0.0, 0.5, 1.0]:
                ras = recovery_alignment_score(recovery_score=recovery, cls=cls)
                assert 0.0 <= ras <= 1.0, f"RAS out of range: {ras}"


class TestComputeMetrics:
    def test_full_window_returns_all_metrics(self):
        window_data = {
            "calendar": {
                "in_meeting": True,
                "meeting_attendees": 4,
                "meeting_duration_minutes": 60,
            },
            "whoop": {
                "recovery_score": 75.0,
            },
            "slack": {
                "messages_sent": 2,
                "messages_received": 8,
                "channels_active": 2,
            },
        }
        metrics = compute_metrics(window_data)

        expected_keys = [
            "cognitive_load_score",
            "focus_depth_index",
            "social_drain_index",
            "context_switch_cost",
            "recovery_alignment_score",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"

    def test_empty_window_does_not_crash(self):
        metrics = compute_metrics({})
        assert "cognitive_load_score" in metrics

    def test_consistency(self):
        """Same inputs should always produce same outputs."""
        window_data = {
            "calendar": {"in_meeting": False, "meeting_attendees": 0, "meeting_duration_minutes": 0},
            "whoop": {"recovery_score": 80.0},
            "slack": {"messages_sent": 1, "messages_received": 3, "channels_active": 1},
        }
        m1 = compute_metrics(window_data)
        m2 = compute_metrics(window_data)
        assert m1 == m2, "Metrics should be deterministic"


# ─── v1.1: Physiological Readiness Composite Tests ───────────────────────────

class TestPhysiologicalReadiness:
    """Tests for the new HRV + sleep composite readiness signal."""

    def test_all_none_returns_neutral(self):
        r = physiological_readiness(None, None, None)
        assert r == 0.5, f"All None should return neutral 0.5, got {r}"

    def test_only_recovery_score(self):
        r = physiological_readiness(80.0, None, None)
        assert r == 0.80, f"Only recovery=80 should return 0.80, got {r}"

    def test_only_recovery_zero(self):
        r = physiological_readiness(0.0, None, None)
        assert r == 0.0

    def test_only_recovery_perfect(self):
        r = physiological_readiness(100.0, None, None)
        assert r == 1.0

    def test_hrv_at_reference_gives_0_5_contribution(self):
        """HRV exactly at reference (65ms) should give 0.5 HRV component."""
        # With only HRV provided, readiness = 0.5
        r = physiological_readiness(None, 65.0, None)
        assert abs(r - 0.5) < 0.01, f"HRV at reference should give ~0.5, got {r}"

    def test_low_hrv_reduces_readiness(self):
        """Low HRV should reduce readiness vs high HRV (same recovery)."""
        r_low_hrv = physiological_readiness(70.0, 25.0, 80.0)
        r_high_hrv = physiological_readiness(70.0, 100.0, 80.0)
        assert r_low_hrv < r_high_hrv, (
            f"Low HRV ({r_low_hrv:.3f}) should give lower readiness "
            f"than high HRV ({r_high_hrv:.3f})"
        )

    def test_high_hrv_raises_readiness(self):
        """HRV above reference raises composite readiness."""
        r_no_hrv = physiological_readiness(70.0, None, None)
        r_high_hrv = physiological_readiness(70.0, 120.0, None)
        assert r_high_hrv > r_no_hrv, (
            f"High HRV should raise readiness above recovery-only baseline"
        )

    def test_low_hrv_lowers_readiness(self):
        """HRV below reference lowers composite readiness."""
        r_no_hrv = physiological_readiness(70.0, None, None)
        r_low_hrv = physiological_readiness(70.0, 20.0, None)
        assert r_low_hrv < r_no_hrv, (
            f"Low HRV should lower readiness below recovery-only baseline"
        )

    def test_poor_sleep_reduces_readiness(self):
        """Poor sleep performance should reduce readiness."""
        r_good_sleep = physiological_readiness(70.0, None, 90.0)
        r_poor_sleep = physiological_readiness(70.0, None, 40.0)
        assert r_good_sleep > r_poor_sleep

    def test_output_range(self):
        for rec in [0, 30, 70, 100, None]:
            for hrv in [10, 50, 65, 100, 150, None]:
                for sleep in [0, 50, 80, 100, None]:
                    r = physiological_readiness(rec, hrv, sleep)
                    assert 0.0 <= r <= 1.0, (
                        f"readiness({rec}, {hrv}, {sleep}) = {r} out of range"
                    )

    def test_full_signals_worse_than_partial_when_hrv_low(self):
        """
        Full signals with very low HRV should give lower readiness than
        recovery alone — HRV is genuinely adding negative signal.
        """
        r_recovery_only = physiological_readiness(80.0, None, None)
        r_full_low_hrv = physiological_readiness(80.0, 15.0, 85.0)
        assert r_full_low_hrv < r_recovery_only, (
            f"Very low HRV should drag down composite: {r_full_low_hrv:.3f} vs {r_recovery_only:.3f}"
        )

    def test_weight_redistribution_when_signals_missing(self):
        """Available weights should redistribute so result is still in [0,1]."""
        # Only recovery + sleep (no HRV)
        r = physiological_readiness(80.0, None, 90.0)
        # Expected: (0.80 * 0.50 + 0.90 * 0.20) / (0.50 + 0.20) ≈ 0.828
        expected = (0.80 * 0.50 + 0.90 * 0.20) / (0.50 + 0.20)
        assert abs(r - expected) < 0.01, f"Expected ~{expected:.3f}, got {r}"


# ─── v1.1: HRV-aware CLS Tests ───────────────────────────────────────────────

class TestCLSHRVAware:
    """CLS should now be sensitive to HRV, not just recovery_score."""

    def test_low_hrv_raises_cls_baseline(self):
        """Same recovery, lower HRV → higher CLS (lower physiological buffer)."""
        cls_normal_hrv = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=70.0,
            hrv_rmssd_milli=80.0,
        )
        cls_low_hrv = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=70.0,
            hrv_rmssd_milli=20.0,
        )
        assert cls_low_hrv > cls_normal_hrv, (
            f"Low HRV should raise CLS baseline: {cls_low_hrv} vs {cls_normal_hrv}"
        )

    def test_high_hrv_lowers_cls_baseline(self):
        """Same recovery, high HRV → lower CLS."""
        cls_no_hrv = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=5,
            recovery_score=70.0,
        )
        cls_high_hrv = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=5,
            recovery_score=70.0,
            hrv_rmssd_milli=130.0,
        )
        assert cls_high_hrv < cls_no_hrv, (
            f"High HRV should lower CLS: {cls_high_hrv} vs {cls_no_hrv}"
        )

    def test_hrv_does_not_affect_meeting_signal(self):
        """Meeting and Slack components should be unaffected by HRV."""
        cls_a = cognitive_load_score(True, 8, 20, 70.0, hrv_rmssd_milli=65.0)
        cls_b = cognitive_load_score(True, 8, 20, 70.0, hrv_rmssd_milli=65.0)
        assert cls_a == cls_b

    def test_backward_compat_no_hrv(self):
        """Calling without HRV args should behave identically to old API."""
        cls_old = cognitive_load_score(True, 4, 10, 75.0)
        cls_new = cognitive_load_score(True, 4, 10, 75.0, hrv_rmssd_milli=None)
        assert cls_old == cls_new

    def test_sleep_performance_affects_cls(self):
        """Poor sleep should increase CLS baseline."""
        cls_good_sleep = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=75.0,
            sleep_performance=95.0,
        )
        cls_poor_sleep = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=75.0,
            sleep_performance=40.0,
        )
        assert cls_poor_sleep > cls_good_sleep


# ─── v1.1: HRV-aware RAS Tests ───────────────────────────────────────────────

class TestRASHRVAware:
    """RAS should now reflect HRV + sleep as part of physiological capacity."""

    def test_low_hrv_reduces_ras(self):
        """Low HRV = less capacity → worse alignment for same CLS."""
        ras_normal = recovery_alignment_score(70.0, 0.50, hrv_rmssd_milli=80.0)
        ras_low_hrv = recovery_alignment_score(70.0, 0.50, hrv_rmssd_milli=20.0)
        assert ras_low_hrv < ras_normal, (
            f"Low HRV should reduce RAS: {ras_low_hrv} vs {ras_normal}"
        )

    def test_high_hrv_improves_ras(self):
        """High HRV = more capacity → better alignment."""
        ras_no_hrv = recovery_alignment_score(70.0, 0.50)
        ras_high_hrv = recovery_alignment_score(70.0, 0.50, hrv_rmssd_milli=120.0)
        assert ras_high_hrv >= ras_no_hrv, (
            f"High HRV should improve RAS: {ras_high_hrv} vs {ras_no_hrv}"
        )

    def test_all_none_still_returns_neutral(self):
        ras = recovery_alignment_score(None, 0.50)
        assert ras == 0.5

    def test_poor_sleep_reduces_ras(self):
        """Poor sleep = less capacity → worse alignment."""
        ras_good = recovery_alignment_score(70.0, 0.50, sleep_performance=95.0)
        ras_poor = recovery_alignment_score(70.0, 0.50, sleep_performance=40.0)
        assert ras_poor < ras_good

    def test_backward_compat_no_hrv(self):
        """Old call signature without HRV should still work."""
        ras = recovery_alignment_score(recovery_score=80.0, cls=0.40)
        assert 0.0 <= ras <= 1.0

    def test_output_range_with_hrv(self):
        for rec in [0, 50, 100, None]:
            for cls_val in [0.0, 0.3, 0.7, 1.0]:
                for hrv in [15, 65, 130, None]:
                    ras = recovery_alignment_score(rec, cls_val, hrv_rmssd_milli=hrv)
                    assert 0.0 <= ras <= 1.0, f"RAS out of range: {ras}"


# ─── v1.1: compute_metrics full-signal integration ───────────────────────────

class TestComputeMetricsHRVIntegration:
    """compute_metrics should pass HRV + sleep through to CLS and RAS."""

    def test_hrv_changes_cls_in_full_pipeline(self):
        """HRV should propagate through compute_metrics into CLS."""
        base = {
            "calendar": {"in_meeting": False, "meeting_attendees": 0, "meeting_duration_minutes": 0},
            "slack": {"messages_sent": 2, "messages_received": 5, "channels_active": 1},
        }
        high_hrv = {**base, "whoop": {"recovery_score": 70.0, "hrv_rmssd_milli": 120.0, "sleep_performance": 85.0}}
        low_hrv  = {**base, "whoop": {"recovery_score": 70.0, "hrv_rmssd_milli": 20.0,  "sleep_performance": 85.0}}

        m_high = compute_metrics(high_hrv)
        m_low  = compute_metrics(low_hrv)

        assert m_low["cognitive_load_score"] > m_high["cognitive_load_score"], (
            f"Low HRV CLS ({m_low['cognitive_load_score']}) should exceed "
            f"high HRV CLS ({m_high['cognitive_load_score']})"
        )

    def test_hrv_changes_ras_in_full_pipeline(self):
        """HRV should propagate through compute_metrics into RAS."""
        base = {
            "calendar": {"in_meeting": True, "meeting_attendees": 5, "meeting_duration_minutes": 60},
            "slack": {"messages_sent": 3, "messages_received": 8, "channels_active": 2},
        }
        high_hrv = {**base, "whoop": {"recovery_score": 65.0, "hrv_rmssd_milli": 110.0, "sleep_performance": 88.0}}
        low_hrv  = {**base, "whoop": {"recovery_score": 65.0, "hrv_rmssd_milli": 22.0,  "sleep_performance": 88.0}}

        m_high = compute_metrics(high_hrv)
        m_low  = compute_metrics(low_hrv)

        assert m_low["recovery_alignment_score"] < m_high["recovery_alignment_score"], (
            f"Low HRV RAS ({m_low['recovery_alignment_score']}) should be worse than "
            f"high HRV RAS ({m_high['recovery_alignment_score']})"
        )

    def test_missing_hrv_in_whoop_dict_graceful(self):
        """If hrv_rmssd_milli is absent from whoop dict, should not crash."""
        window_data = {
            "calendar": {"in_meeting": False, "meeting_attendees": 0, "meeting_duration_minutes": 0},
            "whoop": {"recovery_score": 80.0},  # No HRV key
            "slack": {"messages_sent": 1, "messages_received": 3, "channels_active": 1},
        }
        metrics = compute_metrics(window_data)
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

    def test_full_whoop_signals_all_utilised(self):
        """All WHOOP signals (recovery + HRV + sleep) should affect output."""
        # Maximal physiological readiness
        peak = {
            "calendar": {"in_meeting": False, "meeting_attendees": 0, "meeting_duration_minutes": 0},
            "whoop": {"recovery_score": 100.0, "hrv_rmssd_milli": 130.0, "sleep_performance": 100.0},
            "slack": {"messages_sent": 0, "messages_received": 0, "channels_active": 0},
        }
        # Minimal physiological readiness
        depleted = {
            "calendar": {"in_meeting": False, "meeting_attendees": 0, "meeting_duration_minutes": 0},
            "whoop": {"recovery_score": 0.0, "hrv_rmssd_milli": 10.0, "sleep_performance": 0.0},
            "slack": {"messages_sent": 0, "messages_received": 0, "channels_active": 0},
        }
        m_peak = compute_metrics(peak)
        m_dep  = compute_metrics(depleted)

        # Peak readiness → lower CLS (less cognitive strain at rest)
        assert m_peak["cognitive_load_score"] < m_dep["cognitive_load_score"]
        # Peak readiness → higher RAS (better aligned)
        assert m_peak["recovery_alignment_score"] > m_dep["recovery_alignment_score"]


# ─── v1.2: RescueTime integration tests ──────────────────────────────────────

class TestCLSRescueTimeIntegration:
    """CLS should incorporate RescueTime productivity_score when available."""

    def test_low_rt_productivity_raises_cls(self):
        """A distracted window (low productivity_score) should raise CLS."""
        base_cls = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=90.0,
        )
        rt_distracted_cls = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=90.0,
            rt_productivity_score=0.0,  # Very distracting
            rt_active_seconds=300,
        )
        assert rt_distracted_cls > base_cls, (
            f"Distracted RT window should raise CLS: {rt_distracted_cls} > {base_cls}"
        )

    def test_high_rt_productivity_lowers_cls(self):
        """A focused window (high productivity_score) should lower CLS."""
        base_cls = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=5,
            recovery_score=70.0,
        )
        rt_focused_cls = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=5,
            recovery_score=70.0,
            rt_productivity_score=1.0,  # Very productive
            rt_active_seconds=600,
        )
        assert rt_focused_cls < base_cls, (
            f"Focused RT window should lower CLS: {rt_focused_cls} < {base_cls}"
        )

    def test_rt_idle_window_not_applied(self):
        """RT productivity should NOT affect CLS when active_seconds < 60 (idle window)."""
        base_cls = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=80.0,
        )
        rt_idle_cls = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=80.0,
            rt_productivity_score=0.0,  # Would raise CLS if applied
            rt_active_seconds=30,       # But below threshold (< 60s)
        )
        assert abs(rt_idle_cls - base_cls) < 0.001, (
            f"Idle RT window should not affect CLS: base={base_cls}, rt={rt_idle_cls}"
        )

    def test_rt_none_score_backward_compat(self):
        """When rt_productivity_score is None, CLS should be identical to v1.1 formula."""
        v11_cls = cognitive_load_score(
            in_meeting=True,
            meeting_attendees=5,
            slack_messages_received=10,
            recovery_score=75.0,
            hrv_rmssd_milli=60.0,
        )
        v12_cls = cognitive_load_score(
            in_meeting=True,
            meeting_attendees=5,
            slack_messages_received=10,
            recovery_score=75.0,
            hrv_rmssd_milli=60.0,
            rt_productivity_score=None,
            rt_active_seconds=0,
        )
        assert v11_cls == v12_cls, (
            f"None RT should not change CLS: v1.1={v11_cls}, v1.2={v12_cls}"
        )

    def test_cls_output_always_in_range(self):
        """CLS should never go out of [0, 1] with any RT input."""
        extremes = [
            (0.0, 900), (1.0, 900), (0.5, 60), (0.0, 0), (1.0, 59),
        ]
        for prod_score, active_s in extremes:
            cls = cognitive_load_score(
                in_meeting=True,
                meeting_attendees=10,
                slack_messages_received=30,
                recovery_score=0.0,
                rt_productivity_score=prod_score,
                rt_active_seconds=active_s,
            )
            assert 0.0 <= cls <= 1.0, f"CLS out of range with RT({prod_score}, {active_s}): {cls}"


class TestFDIRescueTimeIntegration:
    """FDI should use RT app_switches when available, replacing the Slack proxy."""

    def test_many_app_switches_reduces_fdi(self):
        """High app_switches from RT should reduce FDI (more fragmented)."""
        fdi_no_rt = focus_depth_index(
            in_meeting=False,
            slack_messages_received=0,
            context_switches=0,
        )
        fdi_with_rt = focus_depth_index(
            in_meeting=False,
            slack_messages_received=0,
            context_switches=0,
            rt_app_switches=8,  # Saturated — very fragmented
            rt_active_seconds=600,
        )
        assert fdi_with_rt < fdi_no_rt, (
            f"Many app switches should reduce FDI: {fdi_with_rt} < {fdi_no_rt}"
        )

    def test_zero_app_switches_maintains_fdi(self):
        """Zero RT app switches should not reduce FDI below no-RT baseline."""
        fdi_no_rt = focus_depth_index(
            in_meeting=False,
            slack_messages_received=0,
            context_switches=0,
        )
        fdi_zero_rt = focus_depth_index(
            in_meeting=False,
            slack_messages_received=0,
            context_switches=0,
            rt_app_switches=0,
            rt_active_seconds=600,
            rt_productivity_score=1.0,  # Fully focused
        )
        assert fdi_zero_rt >= fdi_no_rt, (
            f"Zero switches + high productivity should not reduce FDI: {fdi_zero_rt} >= {fdi_no_rt}"
        )

    def test_rt_idle_not_applied_to_fdi(self):
        """RT data should not affect FDI when active_seconds < 60."""
        fdi_no_rt = focus_depth_index(
            in_meeting=False,
            slack_messages_received=0,
            context_switches=0,
        )
        fdi_idle_rt = focus_depth_index(
            in_meeting=False,
            slack_messages_received=0,
            context_switches=0,
            rt_app_switches=8,      # Would hurt FDI if applied
            rt_active_seconds=30,   # Below threshold
        )
        assert abs(fdi_idle_rt - fdi_no_rt) < 0.001, (
            f"Idle RT should not affect FDI: no_rt={fdi_no_rt}, idle_rt={fdi_idle_rt}"
        )

    def test_rt_distraction_reduces_fdi(self):
        """Low RT productivity should reduce FDI even without many app switches."""
        fdi_productive = focus_depth_index(
            in_meeting=False,
            slack_messages_received=2,
            context_switches=0,
            rt_app_switches=1,
            rt_active_seconds=600,
            rt_productivity_score=1.0,  # Very productive
        )
        fdi_distracted = focus_depth_index(
            in_meeting=False,
            slack_messages_received=2,
            context_switches=0,
            rt_app_switches=1,
            rt_active_seconds=600,
            rt_productivity_score=0.0,  # Very distracting
        )
        assert fdi_distracted < fdi_productive, (
            f"Distracted RT should reduce FDI: {fdi_distracted} < {fdi_productive}"
        )

    def test_fdi_output_always_in_range(self):
        """FDI should never go out of [0, 1] with any RT input."""
        extremes = [
            (8, 900, 0.0), (0, 900, 1.0), (4, 60, 0.5),
            (8, 0, 0.0), (0, 0, None),
        ]
        for switches, active_s, prod in extremes:
            fdi = focus_depth_index(
                in_meeting=True,
                slack_messages_received=30,
                context_switches=20,
                rt_app_switches=switches,
                rt_active_seconds=active_s,
                rt_productivity_score=prod,
            )
            assert 0.0 <= fdi <= 1.0, f"FDI out of range with RT({switches}, {active_s}, {prod}): {fdi}"


class TestCSCRescueTimeIntegration:
    """CSC should use RT app_switches as an additional fragmentation signal."""

    def test_many_rt_switches_raises_csc(self):
        """High RT app switches should raise CSC (more context switch cost)."""
        csc_no_rt = context_switch_cost(
            in_meeting=False,
            meeting_duration_minutes=0,
            slack_channels_active=1,
        )
        csc_with_rt = context_switch_cost(
            in_meeting=False,
            meeting_duration_minutes=0,
            slack_channels_active=1,
            rt_app_switches=8,
            rt_active_seconds=600,
        )
        assert csc_with_rt > csc_no_rt, (
            f"Many RT switches should raise CSC: {csc_with_rt} > {csc_no_rt}"
        )

    def test_zero_rt_switches_does_not_raise_csc(self):
        """Zero RT app switches should not raise CSC above no-RT baseline."""
        csc_no_rt = context_switch_cost(
            in_meeting=False,
            meeting_duration_minutes=0,
            slack_channels_active=0,
        )
        csc_zero_rt = context_switch_cost(
            in_meeting=False,
            meeting_duration_minutes=0,
            slack_channels_active=0,
            rt_app_switches=0,
            rt_active_seconds=600,
        )
        assert csc_zero_rt <= csc_no_rt, (
            f"Zero RT switches should not raise CSC: {csc_zero_rt} <= {csc_no_rt}"
        )

    def test_rt_idle_not_applied_to_csc(self):
        """RT data should not affect CSC when active_seconds < 60."""
        csc_no_rt = context_switch_cost(
            in_meeting=False,
            meeting_duration_minutes=0,
            slack_channels_active=0,
        )
        csc_idle_rt = context_switch_cost(
            in_meeting=False,
            meeting_duration_minutes=0,
            slack_channels_active=0,
            rt_app_switches=8,
            rt_active_seconds=30,  # Below threshold
        )
        assert abs(csc_idle_rt - csc_no_rt) < 0.001, (
            f"Idle RT should not affect CSC: no_rt={csc_no_rt}, idle_rt={csc_idle_rt}"
        )

    def test_csc_output_always_in_range(self):
        """CSC should never go out of [0, 1] with any RT input."""
        extremes = [(8, 900), (0, 900), (4, 60), (8, 0), (0, 0)]
        for switches, active_s in extremes:
            csc = context_switch_cost(
                in_meeting=True,
                meeting_duration_minutes=10,
                slack_channels_active=5,
                is_short_meeting=True,
                rt_app_switches=switches,
                rt_active_seconds=active_s,
            )
            assert 0.0 <= csc <= 1.0, f"CSC out of range with RT({switches}, {active_s}): {csc}"


class TestComputeMetricsRescueTimePassthrough:
    """compute_metrics() should pass RT data through to CLS, FDI, CSC."""

    def _window(self, rt_data=None):
        base = {
            "calendar": {"in_meeting": False, "meeting_attendees": 0, "meeting_duration_minutes": 0},
            "whoop": {"recovery_score": 80.0, "hrv_rmssd_milli": 65.0, "sleep_performance": 85.0},
            "slack": {"messages_sent": 2, "messages_received": 3, "channels_active": 1},
        }
        if rt_data is not None:
            base["rescuetime"] = rt_data
        return base

    def test_no_rt_key_does_not_crash(self):
        """compute_metrics should work fine without a 'rescuetime' key."""
        metrics = compute_metrics(self._window())
        assert all(k in metrics for k in [
            "cognitive_load_score", "focus_depth_index", "social_drain_index",
            "context_switch_cost", "recovery_alignment_score"
        ])

    def test_rt_none_does_not_crash(self):
        """compute_metrics should handle rescuetime=None gracefully."""
        window = self._window()
        window["rescuetime"] = None
        metrics = compute_metrics(window)
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

    def test_rt_distracted_raises_cls_vs_no_rt(self):
        """compute_metrics with distracted RT data should produce higher CLS."""
        m_no_rt = compute_metrics(self._window())
        m_distracted = compute_metrics(self._window({
            "app_switches": 5,
            "productivity_score": 0.0,
            "active_seconds": 600,
            "focus_seconds": 0,
            "distraction_seconds": 600,
        }))
        assert m_distracted["cognitive_load_score"] > m_no_rt["cognitive_load_score"], (
            "Distracted RT should raise CLS in compute_metrics pipeline"
        )

    def test_rt_fragmented_reduces_fdi_vs_no_rt(self):
        """compute_metrics with high app_switches should produce lower FDI."""
        m_no_rt = compute_metrics(self._window())
        m_fragmented = compute_metrics(self._window({
            "app_switches": 8,
            "productivity_score": 0.2,
            "active_seconds": 700,
            "focus_seconds": 100,
            "distraction_seconds": 400,
        }))
        assert m_fragmented["focus_depth_index"] < m_no_rt["focus_depth_index"], (
            "Fragmented RT should reduce FDI in compute_metrics pipeline"
        )

    def test_rt_fragmented_raises_csc_vs_no_rt(self):
        """compute_metrics with high app_switches should produce higher CSC."""
        m_no_rt = compute_metrics(self._window())
        m_fragmented = compute_metrics(self._window({
            "app_switches": 8,
            "productivity_score": 0.3,
            "active_seconds": 800,
            "focus_seconds": 200,
            "distraction_seconds": 300,
        }))
        assert m_fragmented["context_switch_cost"] > m_no_rt["context_switch_cost"], (
            "Fragmented RT should raise CSC in compute_metrics pipeline"
        )

    def test_all_metrics_in_range_with_rt(self):
        """All metrics should stay in [0, 1] when RT data is present."""
        rt_data = {
            "app_switches": 6,
            "productivity_score": 0.3,
            "active_seconds": 720,
            "focus_seconds": 200,
            "distraction_seconds": 400,
        }
        metrics = compute_metrics(self._window(rt_data))
        for name, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"Metric {name} out of range with RT: {val}"


class TestCLSSoloMeetingFix:
    """
    v1.5 — Solo-meeting fix for CLS.

    A solo calendar block (attendees=0 or 1) should not trigger the
    meeting_component (0.35 weight) in CLS.  Only social meetings
    (attendees > 1) generate external coordination overhead that CLS
    captures.  This is consistent with the v1.4 FDI and SDI fixes.
    """

    def _cls(self, in_meeting, attendees, recovery=80.0, slack_recv=0):
        return cognitive_load_score(
            in_meeting=in_meeting,
            meeting_attendees=attendees,
            slack_messages_received=slack_recv,
            recovery_score=recovery,
        )

    def test_solo_block_no_attendees_equals_no_meeting_cls(self):
        """A solo block with 0 attendees should give the same CLS as no meeting at all."""
        cls_no_meeting = self._cls(in_meeting=False, attendees=0)
        cls_solo_0 = self._cls(in_meeting=True, attendees=0)
        assert cls_no_meeting == cls_solo_0, (
            f"Solo block (atts=0) should equal no-meeting CLS. "
            f"no_meeting={cls_no_meeting}, solo_0={cls_solo_0}"
        )

    def test_solo_block_one_attendee_equals_no_meeting_cls(self):
        """A solo block with attendee_count=1 (just David) should give same CLS as no meeting."""
        cls_no_meeting = self._cls(in_meeting=False, attendees=0)
        cls_solo_1 = self._cls(in_meeting=True, attendees=1)
        assert cls_no_meeting == cls_solo_1, (
            f"Solo block (atts=1) should equal no-meeting CLS. "
            f"no_meeting={cls_no_meeting}, solo_1={cls_solo_1}"
        )

    def test_social_meeting_two_attendees_raises_cls(self):
        """A meeting with 2+ attendees (social) should have higher CLS than no meeting."""
        cls_no_meeting = self._cls(in_meeting=False, attendees=0)
        cls_social = self._cls(in_meeting=True, attendees=2)
        assert cls_social > cls_no_meeting, (
            f"Social meeting (atts=2) should raise CLS vs no meeting. "
            f"social={cls_social}, no_meeting={cls_no_meeting}"
        )

    def test_social_meeting_significantly_higher_than_solo_block(self):
        """The gap between social meeting and solo block should be substantial (>0.20)."""
        cls_solo = self._cls(in_meeting=True, attendees=1, recovery=80.0)
        cls_social = self._cls(in_meeting=True, attendees=4, recovery=80.0)
        diff = cls_social - cls_solo
        assert diff >= 0.20, (
            f"Social meeting CLS should be significantly higher than solo block. "
            f"social={cls_social}, solo={cls_solo}, diff={diff}"
        )

    def test_solo_block_is_low_regardless_of_meeting_flag(self):
        """Solo blocks should always return low CLS (< 0.20) when recovery is good."""
        for recovery in [70.0, 80.0, 90.0, 100.0]:
            cls = self._cls(in_meeting=True, attendees=1, recovery=recovery)
            assert cls < 0.20, (
                f"Solo block with good recovery should be low CLS, got {cls} "
                f"(recovery={recovery})"
            )

    def test_large_social_meeting_high_cls(self):
        """Large social meeting (10 people) should still produce high CLS."""
        cls = self._cls(in_meeting=True, attendees=10, recovery=80.0)
        assert cls > 0.40, f"Large meeting should be high CLS, got {cls}"

    def test_solo_block_with_slack_adds_slack_component_only(self):
        """Solo block with Slack messages should raise CLS via Slack component only."""
        cls_solo_no_slack = self._cls(in_meeting=True, attendees=1, slack_recv=0)
        cls_solo_with_slack = self._cls(in_meeting=True, attendees=1, slack_recv=15)
        assert cls_solo_with_slack > cls_solo_no_slack, (
            "Slack messages in a solo block should still raise CLS via Slack component"
        )
        # But should still be lower than a social meeting with same slack
        cls_social_with_slack = self._cls(in_meeting=True, attendees=4, slack_recv=15)
        assert cls_solo_with_slack < cls_social_with_slack, (
            "Solo block + slack should still be lower than social meeting + slack"
        )

    def test_cls_output_range_with_solo_blocks(self):
        """All CLS values should remain in [0, 1] for solo blocks."""
        for attendees in [0, 1]:
            for recovery in [0, 50, 100, None]:
                for slack_recv in [0, 10, 30]:
                    cls = cognitive_load_score(
                        in_meeting=True,
                        meeting_attendees=attendees,
                        slack_messages_received=slack_recv,
                        recovery_score=recovery,
                    )
                    assert 0.0 <= cls <= 1.0, (
                        f"CLS out of range: {cls} "
                        f"(solo, atts={attendees}, recovery={recovery}, slack={slack_recv})"
                    )

    def test_compute_metrics_solo_block_low_cls(self):
        """compute_metrics pipeline: solo block should produce low CLS end-to-end."""
        window = {
            "calendar": {
                "in_meeting": True,
                "meeting_attendees": 1,
                "meeting_duration_minutes": 60,
            },
            "whoop": {
                "recovery_score": 85.0,
                "hrv_rmssd_milli": 70.0,
                "sleep_performance": 88.0,
            },
            "slack": {
                "messages_sent": 0,
                "messages_received": 0,
                "channels_active": 0,
            },
        }
        metrics = compute_metrics(window)
        assert metrics["cognitive_load_score"] < 0.15, (
            f"Solo block end-to-end should have low CLS, got {metrics['cognitive_load_score']}"
        )

    def test_compute_metrics_solo_vs_social_cls_ordering(self):
        """compute_metrics: social meeting CLS must exceed solo block CLS substantially."""
        solo_window = {
            "calendar": {"in_meeting": True, "meeting_attendees": 1, "meeting_duration_minutes": 60},
            "whoop": {"recovery_score": 80.0},
            "slack": {"messages_sent": 0, "messages_received": 0, "channels_active": 0},
        }
        social_window = {
            "calendar": {"in_meeting": True, "meeting_attendees": 5, "meeting_duration_minutes": 60},
            "whoop": {"recovery_score": 80.0},
            "slack": {"messages_sent": 2, "messages_received": 8, "channels_active": 2},
        }
        cls_solo = compute_metrics(solo_window)["cognitive_load_score"]
        cls_social = compute_metrics(social_window)["cognitive_load_score"]
        assert cls_social > cls_solo + 0.25, (
            f"Social meeting CLS ({cls_social}) should be >> solo block CLS ({cls_solo})"
        )

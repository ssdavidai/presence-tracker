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

    def test_in_meeting_returns_low(self):
        fdi = focus_depth_index(
            in_meeting=True,
            slack_messages_received=15,
            context_switches=5,
        )
        assert fdi < 0.40, f"High disruption should yield low FDI, got {fdi}"

    def test_output_range(self):
        for args in [
            (True, 30, 20),
            (False, 0, 0),
            (True, 0, 0),
            (False, 30, 20),
        ]:
            fdi = focus_depth_index(*args)
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
        csc_short = context_switch_cost(
            in_meeting=True,
            meeting_duration_minutes=10,
            slack_channels_active=1,
            is_short_meeting=True,
        )
        csc_long = context_switch_cost(
            in_meeting=True,
            meeting_duration_minutes=90,
            slack_channels_active=1,
            is_short_meeting=False,
        )
        assert csc_short > csc_long, f"Short meeting should cost more: {csc_short} vs {csc_long}"

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

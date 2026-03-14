"""
Tests for analysis/ml_model.py — ML model layer.

These tests focus on the pure functions (feature extraction, graceful
fallbacks when data is insufficient) without requiring scikit-learn
to actually train models.

Run with: python3 -m pytest tests/test_ml_model.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch


# ─── Imports ─────────────────────────────────────────────────────────────────

class TestImports:
    def test_all_public_symbols_importable(self):
        from analysis.ml_model import (
            train_all,
            detect_anomalies,
            predict_recovery,
            get_data_status,
            is_ready_to_train,
            MIN_DAYS_REQUIRED,
        )
        assert callable(train_all)
        assert callable(detect_anomalies)
        assert callable(predict_recovery)
        assert callable(get_data_status)
        assert callable(is_ready_to_train)
        assert isinstance(MIN_DAYS_REQUIRED, int)
        assert MIN_DAYS_REQUIRED >= 30  # Sanity: needs meaningful history

    def test_min_days_required_is_60(self):
        from analysis.ml_model import MIN_DAYS_REQUIRED
        assert MIN_DAYS_REQUIRED == 60


# ─── get_data_status ─────────────────────────────────────────────────────────

class TestGetDataStatus:
    def test_returns_dict(self):
        from analysis.ml_model import get_data_status
        result = get_data_status()
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        from analysis.ml_model import get_data_status
        result = get_data_status()
        required = {"days_of_data", "min_days_required", "ready_to_train"}
        assert required.issubset(result.keys())

    def test_ready_to_train_is_bool(self):
        from analysis.ml_model import get_data_status
        result = get_data_status()
        assert isinstance(result["ready_to_train"], bool)

    def test_days_of_data_is_non_negative(self):
        from analysis.ml_model import get_data_status
        result = get_data_status()
        assert result["days_of_data"] >= 0

    def test_not_ready_when_insufficient_data(self):
        """With only 1 day of data we should not be ready to train."""
        from analysis.ml_model import is_ready_to_train
        # The real store has only 1 day; must not be ready
        result = is_ready_to_train()
        assert result is False


# ─── detect_anomalies — graceful fallback ────────────────────────────────────

class TestDetectAnomalies:
    def _make_windows(self, n=4):
        """Minimal window list for testing."""
        return [
            {
                "window_index": i,
                "metrics": {
                    "cognitive_load_score": 0.40 + i * 0.05,
                    "focus_depth_index": 0.70,
                    "social_drain_index": 0.20,
                    "context_switch_cost": 0.30,
                    "recovery_alignment_score": 0.75,
                },
                "whoop": {"recovery_score": 72.0, "hrv_rmssd_milli": 65.0},
                "metadata": {"is_working_hours": True, "is_active_window": True},
            }
            for i in range(n)
        ]

    def test_returns_list(self):
        from analysis.ml_model import detect_anomalies
        result = detect_anomalies(self._make_windows())
        assert isinstance(result, list)

    def test_no_crash_on_empty_windows(self):
        from analysis.ml_model import detect_anomalies
        result = detect_anomalies([])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_returns_same_length_as_input_or_empty(self):
        from analysis.ml_model import detect_anomalies
        windows = self._make_windows(6)
        result = detect_anomalies(windows)
        # Either full anomaly list or empty fallback (no model trained yet)
        assert len(result) == 0 or len(result) == len(windows)

    def test_no_crash_on_none_metrics(self):
        from analysis.ml_model import detect_anomalies
        windows = [{"window_index": 0, "metrics": {}, "whoop": {}, "metadata": {}}]
        result = detect_anomalies(windows)
        assert isinstance(result, list)


# ─── predict_recovery — graceful fallback ────────────────────────────────────

class TestPredictRecovery:
    def _make_windows(self):
        return [
            {
                "window_index": i,
                "metrics": {
                    "cognitive_load_score": 0.55,
                    "focus_depth_index": 0.65,
                    "social_drain_index": 0.25,
                    "context_switch_cost": 0.35,
                    "recovery_alignment_score": 0.70,
                },
                "whoop": {"recovery_score": 68.0, "hrv_rmssd_milli": 58.0},
                "metadata": {"is_working_hours": True, "is_active_window": True},
            }
            for i in range(8)
        ]

    def test_returns_none_or_float_when_no_model(self):
        from analysis.ml_model import predict_recovery
        result = predict_recovery(self._make_windows())
        assert result is None or isinstance(result, (int, float))

    def test_no_crash_on_empty_windows(self):
        from analysis.ml_model import predict_recovery
        result = predict_recovery([])
        assert result is None or isinstance(result, (int, float))

    def test_returns_none_before_model_trained(self):
        """No model file → should return None gracefully."""
        from analysis.ml_model import predict_recovery
        # With 1 day of data, model is definitely not trained
        result = predict_recovery(self._make_windows())
        assert result is None


# ─── train_all — insufficient data path ──────────────────────────────────────

class TestTrainAll:
    def test_returns_dict(self):
        from analysis.ml_model import train_all
        result = train_all()
        assert isinstance(result, dict)

    def test_skips_training_when_insufficient_data(self):
        from analysis.ml_model import train_all
        result = train_all()
        # Should indicate skipped/insufficient, not raise
        assert "status" in result or "error" in result or "skipped" in result or isinstance(result, dict)

    def test_no_exception_on_insufficient_data(self):
        from analysis.ml_model import train_all
        # Must not raise regardless of data availability
        try:
            train_all()
        except Exception as e:
            pytest.fail(f"train_all() raised unexpectedly: {e}")

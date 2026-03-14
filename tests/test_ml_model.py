"""
Tests for the ML model layer.

Run with: python3 -m pytest tests/test_ml_model.py -v

All tests are pure unit tests — no trained models required, no network.
Tests cover:
  - Feature extraction from window dicts
  - Data status / readiness checks
  - Personal baseline computation
  - Heuristic anomaly detection
  - Graceful degradation when models are not trained
  - train_all() behaviour (insufficient data path)
  - get_focus_cluster_profiles() with no model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from analysis.ml_model import (
    extract_window_features,
    extract_daily_features,
    get_data_status,
    is_ready_to_train,
    compute_personal_baselines,
    heuristic_anomaly_check,
    detect_anomalies,
    predict_recovery,
    get_focus_cluster_profiles,
    train_all,
    FEATURE_NAMES,
    N_FEATURES,
    MIN_DAYS_REQUIRED,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_window(
    hour: int = 10,
    in_meeting: bool = False,
    meeting_attendees: int = 0,
    meeting_duration: int = 0,
    messages_sent: int = 0,
    messages_received: int = 5,
    channels_active: int = 1,
    recovery: float = 80.0,
    hrv: float = 70.0,
    sleep_perf: float = 85.0,
    sleep_hours: float = 8.0,
    cls: float = 0.20,
    fdi: float = 0.80,
    sdi: float = 0.10,
    csc: float = 0.15,
    ras: float = 0.85,
    is_working: bool = True,
    is_active: bool = True,
    window_index: int = 40,
    date: str = "2026-03-13",
) -> dict:
    """Build a minimal window dict for testing."""
    return {
        "window_id": f"{date}T{hour:02d}:00:00",
        "date": date,
        "window_index": window_index,
        "calendar": {
            "in_meeting": in_meeting,
            "meeting_attendees": meeting_attendees,
            "meeting_duration_minutes": meeting_duration,
            "meetings_count": 1 if in_meeting else 0,
        },
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "sleep_performance": sleep_perf,
            "sleep_hours": sleep_hours,
        },
        "slack": {
            "messages_sent": messages_sent,
            "messages_received": messages_received,
            "channels_active": channels_active,
            "total_messages": messages_sent + messages_received,
        },
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "context_switch_cost": csc,
            "recovery_alignment_score": ras,
        },
        "metadata": {
            "hour_of_day": hour,
            "minute_of_hour": 0,
            "is_working_hours": is_working,
            "is_active_window": is_active,
        },
    }


def _make_day(n_windows: int = 60, **kwargs) -> list[dict]:
    """Build a list of working-hour windows for testing."""
    return [
        _make_window(
            hour=7 + (i // 4),
            window_index=28 + i,
            **kwargs,
        )
        for i in range(n_windows)
    ]


# ─── Feature extraction ───────────────────────────────────────────────────────

class TestExtractWindowFeatures:
    def test_returns_numpy_array(self):
        w = _make_window()
        v = extract_window_features(w)
        assert isinstance(v, np.ndarray)

    def test_correct_length(self):
        w = _make_window()
        v = extract_window_features(w)
        assert len(v) == N_FEATURES

    def test_feature_names_length_matches_n_features(self):
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_returns_none_when_metrics_missing(self):
        w = _make_window()
        del w["metrics"]["cognitive_load_score"]
        v = extract_window_features(w)
        assert v is None

    def test_returns_none_when_all_metrics_missing(self):
        w = _make_window()
        w["metrics"] = {}
        v = extract_window_features(w)
        assert v is None

    def test_cls_is_first_feature(self):
        w = _make_window(cls=0.42)
        v = extract_window_features(w)
        assert v[FEATURE_NAMES.index("cls")] == pytest.approx(0.42)

    def test_fdi_feature_correct(self):
        w = _make_window(fdi=0.77)
        v = extract_window_features(w)
        assert v[FEATURE_NAMES.index("fdi")] == pytest.approx(0.77)

    def test_in_meeting_encoded_as_one(self):
        w = _make_window(in_meeting=True, meeting_attendees=4, meeting_duration=30)
        v = extract_window_features(w)
        assert v[FEATURE_NAMES.index("in_meeting")] == 1.0

    def test_not_in_meeting_encoded_as_zero(self):
        w = _make_window(in_meeting=False)
        v = extract_window_features(w)
        assert v[FEATURE_NAMES.index("in_meeting")] == 0.0

    def test_missing_whoop_uses_defaults(self):
        w = _make_window()
        w["whoop"] = {}  # No WHOOP data
        # Should not return None — uses neutral defaults
        v = extract_window_features(w)
        assert v is not None
        assert v[FEATURE_NAMES.index("recovery_score")] == pytest.approx(50.0)
        assert v[FEATURE_NAMES.index("hrv_rmssd")] == pytest.approx(65.0)

    def test_physiological_load_ratio_positive(self):
        w = _make_window(cls=0.50, recovery=80.0)
        v = extract_window_features(w)
        ratio = v[FEATURE_NAMES.index("physiological_load_ratio")]
        # cls / (recovery/100) = 0.50 / 0.80 = 0.625
        assert ratio == pytest.approx(0.625, abs=0.01)

    def test_slack_intensity_computed_correctly(self):
        w = _make_window(messages_sent=4, messages_received=6)
        v = extract_window_features(w)
        # sent / (total + 1) = 4 / (10 + 1) = 4/11
        assert v[FEATURE_NAMES.index("slack_intensity")] == pytest.approx(4 / 11, abs=0.001)

    def test_hour_feature_correct(self):
        w = _make_window(hour=14)
        v = extract_window_features(w)
        assert v[FEATURE_NAMES.index("hour_of_day")] == pytest.approx(14.0)

    def test_all_values_finite(self):
        w = _make_window()
        v = extract_window_features(w)
        assert np.all(np.isfinite(v))


# ─── Daily feature extraction ─────────────────────────────────────────────────

class TestExtractDailyFeatures:
    def test_returns_numpy_array(self):
        windows = _make_day(20)
        result = extract_daily_features(windows)
        assert isinstance(result, np.ndarray)

    def test_daily_features_are_3x_window_features(self):
        windows = _make_day(20)
        result = extract_daily_features(windows)
        # mean + max + std = 3 × N_FEATURES
        assert len(result) == 3 * N_FEATURES

    def test_returns_none_when_too_few_windows(self):
        windows = _make_day(5)  # Fewer than 10
        result = extract_daily_features(windows)
        assert result is None

    def test_returns_none_when_empty(self):
        result = extract_daily_features([])
        assert result is None

    def test_returns_none_when_no_working_hours(self):
        windows = _make_day(20, is_working=False)
        result = extract_daily_features(windows)
        assert result is None

    def test_daily_features_all_finite(self):
        windows = _make_day(20)
        result = extract_daily_features(windows)
        assert np.all(np.isfinite(result))


# ─── Data status and readiness ────────────────────────────────────────────────

class TestDataStatus:
    def test_get_data_status_returns_dict(self):
        status = get_data_status()
        assert isinstance(status, dict)

    def test_status_has_required_keys(self):
        status = get_data_status()
        required_keys = [
            "days_of_data", "min_days_required", "ready_to_train",
            "days_remaining_until_ready", "models_trained",
        ]
        for key in required_keys:
            assert key in status, f"Missing key: {key}"

    def test_min_days_required_matches_constant(self):
        status = get_data_status()
        assert status["min_days_required"] == MIN_DAYS_REQUIRED

    def test_models_trained_has_all_model_keys(self):
        status = get_data_status()
        models = status["models_trained"]
        assert "anomaly_detector" in models
        assert "recovery_predictor" in models
        assert "focus_clusters" in models
        assert "feature_scaler" in models

    def test_days_remaining_is_non_negative(self):
        status = get_data_status()
        assert status["days_remaining_until_ready"] >= 0

    def test_is_ready_to_train_returns_bool(self):
        result = is_ready_to_train()
        assert isinstance(result, bool)

    def test_min_days_required_is_60(self):
        assert MIN_DAYS_REQUIRED == 60


# ─── Personal baselines ───────────────────────────────────────────────────────

class TestComputePersonalBaselines:
    def test_returns_dict_with_no_dates(self):
        result = compute_personal_baselines([])
        assert isinstance(result, dict)
        assert result["days_analyzed"] == 0

    def test_all_stats_keys_present(self):
        # Test with a non-existent date — should handle gracefully
        result = compute_personal_baselines(["9999-01-01"])
        assert "cls" in result
        assert "fdi" in result
        assert "ras" in result
        assert "hrv_ms" in result
        assert "recovery_pct" in result

    def test_stats_dict_structure(self):
        result = compute_personal_baselines([])
        cls_stats = result["cls"]
        assert "mean" in cls_stats
        assert "std" in cls_stats
        assert "p25" in cls_stats
        assert "p75" in cls_stats

    def test_empty_dates_gives_none_stats(self):
        result = compute_personal_baselines([])
        assert result["cls"]["mean"] is None
        assert result["cls"]["std"] is None


# ─── Heuristic anomaly detection ─────────────────────────────────────────────

class TestHeuristicAnomalyCheck:
    def _baselines_with(self, cls_mean: float, cls_std: float) -> dict:
        return {
            "cls": {
                "mean": cls_mean,
                "std": cls_std,
                "p25": cls_mean - cls_std,
                "p75": cls_mean + cls_std,
            }
        }

    def test_returns_list(self):
        windows = _make_day(10)
        baselines = self._baselines_with(0.20, 0.10)
        result = heuristic_anomaly_check(windows, baselines)
        assert isinstance(result, list)

    def test_no_anomaly_for_normal_cls(self):
        # CLS at mean — well within range
        windows = [_make_window(cls=0.20, is_working=True)]
        baselines = self._baselines_with(0.20, 0.10)
        result = heuristic_anomaly_check(windows, baselines)
        assert len(result) == 0

    def test_anomaly_detected_for_high_cls(self):
        # CLS = 0.80 when mean=0.20, std=0.10 → 6 std deviations above
        windows = [_make_window(cls=0.80, is_working=True)]
        baselines = self._baselines_with(0.20, 0.10)
        result = heuristic_anomaly_check(windows, baselines)
        assert len(result) == 1

    def test_anomaly_window_id_preserved(self):
        windows = [_make_window(cls=0.90, is_working=True, hour=14)]
        baselines = self._baselines_with(0.20, 0.10)
        result = heuristic_anomaly_check(windows, baselines)
        assert len(result) == 1
        assert result[0]["window_id"] == "2026-03-13T14:00:00"

    def test_anomaly_score_is_negative(self):
        # Isolation Forest convention: more anomalous = more negative score
        windows = [_make_window(cls=0.90, is_working=True)]
        baselines = self._baselines_with(0.20, 0.10)
        result = heuristic_anomaly_check(windows, baselines)
        assert result[0]["anomaly_score"] < 0

    def test_no_anomaly_for_non_working_hours(self):
        # Non-working windows should not be flagged
        windows = [_make_window(cls=0.95, is_working=False)]
        baselines = self._baselines_with(0.20, 0.10)
        result = heuristic_anomaly_check(windows, baselines)
        assert len(result) == 0

    def test_sorted_most_anomalous_first(self):
        windows = [
            _make_window(cls=0.60, is_working=True, hour=9),
            _make_window(cls=0.90, is_working=True, hour=10),
            _make_window(cls=0.75, is_working=True, hour=11),
        ]
        baselines = self._baselines_with(0.20, 0.05)
        result = heuristic_anomaly_check(windows, baselines)
        # Most anomalous (lowest score) should come first
        scores = [a["anomaly_score"] for a in result]
        assert scores == sorted(scores)

    def test_empty_windows_returns_empty(self):
        baselines = self._baselines_with(0.20, 0.10)
        result = heuristic_anomaly_check([], baselines)
        assert result == []

    def test_no_baseline_returns_empty(self):
        windows = [_make_window(cls=0.90, is_working=True)]
        result = heuristic_anomaly_check(windows, {})
        assert result == []

    def test_anomaly_features_dict_present(self):
        windows = [_make_window(cls=0.85, is_working=True)]
        baselines = self._baselines_with(0.20, 0.10)
        result = heuristic_anomaly_check(windows, baselines)
        assert len(result) == 1
        features = result[0]["features"]
        assert "cls" in features
        assert "fdi" in features
        assert "ras" in features


# ─── Graceful degradation (no trained model) ─────────────────────────────────

class TestGracefulDegradation:
    """Verify that all inference functions return safe defaults when no model is trained."""

    def test_detect_anomalies_returns_empty_list_no_model(self, tmp_path, monkeypatch):
        # Monkeypatch model paths to non-existent files
        import analysis.ml_model as ml
        monkeypatch.setattr(ml, "_ANOMALY_MODEL_PATH", tmp_path / "no_model.pkl")
        monkeypatch.setattr(ml, "_SCALER_PATH", tmp_path / "no_scaler.pkl")
        windows = _make_day(10)
        result = ml.detect_anomalies(windows)
        assert result == []

    def test_predict_recovery_returns_none_no_model(self, tmp_path, monkeypatch):
        import analysis.ml_model as ml
        monkeypatch.setattr(ml, "_RECOVERY_MODEL_PATH", tmp_path / "no_model.pkl")
        monkeypatch.setattr(ml, "_SCALER_PATH", tmp_path / "no_scaler.pkl")
        windows = _make_day(20)
        result = ml.predict_recovery(windows)
        assert result is None

    def test_get_focus_cluster_profiles_returns_empty_no_model(self, tmp_path, monkeypatch):
        import analysis.ml_model as ml
        monkeypatch.setattr(ml, "_FOCUS_CLUSTER_PATH", tmp_path / "no_model.pkl")
        monkeypatch.setattr(ml, "_SCALER_PATH", tmp_path / "no_scaler.pkl")
        result = ml.get_focus_cluster_profiles([])
        assert result == []


# ─── train_all insufficient data ─────────────────────────────────────────────

class TestTrainAllInsufficientData:
    def test_returns_dict(self):
        # We can't control available dates, but we can test the shape of the response
        result = train_all(force=False)
        assert isinstance(result, dict)

    def test_returns_insufficient_data_when_less_than_min_days(self):
        # If system has fewer than 60 days, should return 'insufficient_data'
        from engine.store import list_available_dates
        dates = list_available_dates()
        if len(dates) < MIN_DAYS_REQUIRED:
            result = train_all(force=False)
            assert result["status"] == "insufficient_data"
            assert result["days_available"] == len(dates)
            assert result["days_required"] == MIN_DAYS_REQUIRED
            assert result["anomaly"] == "skipped"
            assert result["recovery"] == "skipped"
            assert result["clustering"] == "skipped"

    def test_insufficient_data_result_has_required_keys(self):
        from engine.store import list_available_dates
        dates = list_available_dates()
        if len(dates) < MIN_DAYS_REQUIRED:
            result = train_all(force=False)
            assert "status" in result
            assert "days_available" in result
            assert "days_required" in result


# ─── Feature matrix building ──────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_empty_dates_returns_empty_matrix(self):
        from analysis.ml_model import build_feature_matrix
        X, meta = build_feature_matrix([])
        assert X.shape == (0, N_FEATURES)
        assert meta == []

    def test_nonexistent_date_returns_empty(self):
        from analysis.ml_model import build_feature_matrix
        X, meta = build_feature_matrix(["9999-12-31"])
        assert len(X) == 0
        assert len(meta) == 0


# ─── Integration: feature extraction consistency ──────────────────────────────

class TestFeatureExtractionConsistency:
    def test_high_cls_window_has_high_cls_feature(self):
        w = _make_window(cls=0.85, in_meeting=True, meeting_attendees=8, messages_received=20)
        v = extract_window_features(w)
        assert v[FEATURE_NAMES.index("cls")] > 0.70

    def test_high_fdi_window_has_high_fdi_feature(self):
        w = _make_window(fdi=0.95, in_meeting=False, messages_received=1)
        v = extract_window_features(w)
        assert v[FEATURE_NAMES.index("fdi")] > 0.80

    def test_meeting_window_has_nonzero_meeting_features(self):
        w = _make_window(in_meeting=True, meeting_attendees=6, meeting_duration=60)
        v = extract_window_features(w)
        assert v[FEATURE_NAMES.index("in_meeting")] == 1.0
        assert v[FEATURE_NAMES.index("meeting_attendees")] == 6.0
        assert v[FEATURE_NAMES.index("meeting_duration_minutes")] == 60.0

    def test_low_recovery_increases_physiological_load_ratio(self):
        w_low = _make_window(cls=0.50, recovery=40.0)
        w_high = _make_window(cls=0.50, recovery=90.0)
        v_low = extract_window_features(w_low)
        v_high = extract_window_features(w_high)
        ratio_idx = FEATURE_NAMES.index("physiological_load_ratio")
        # Same CLS but lower recovery → higher physiological load ratio
        assert v_low[ratio_idx] > v_high[ratio_idx]

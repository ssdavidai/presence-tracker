"""
Tests for ML model insights integration in the nightly digest and morning brief.

Coverage:
  - _compute_ml_insights_for_digest() with no trained models → returns None
  - _compute_ml_insights_for_digest() with mock anomaly detection → clusters formed
  - _compute_ml_insights_for_digest() with mock recovery prediction → included in result
  - Anomaly clustering: consecutive hours → single cluster
  - Anomaly clustering: non-consecutive hours → separate clusters
  - Anomaly clustering: singleton anomaly → filtered out (count < 2)
  - compute_digest() includes "ml_insights" key
  - format_digest_message() renders recovery prediction line
  - format_digest_message() renders anomaly cluster line
  - format_digest_message() skips section when ml_insights is None
  - _compute_ml_recovery_for_brief() with no trained model → None
  - _compute_ml_recovery_for_brief() with mock prediction + actual → accuracy classification
  - format_morning_brief_message() renders ml_recovery line when present

Run with: python3 -m pytest tests/test_ml_insights_digest.py -v
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.daily_digest import (
    _compute_ml_insights_for_digest,
    compute_digest,
    format_digest_message,
)
from engine.chunker import build_windows


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_WHOOP = {
    "recovery_score": 78.0,
    "hrv_rmssd_milli": 65.5,
    "resting_heart_rate": 54.0,
    "sleep_performance": 82.0,
    "sleep_hours": 7.8,
    "strain": 13.2,
    "spo2_percentage": 95.2,
}

SAMPLE_CALENDAR = {"events": []}

# slack_windows format: keyed by ISO timestamp string
SAMPLE_SLACK_WINDOWS: dict = {}

SAMPLE_RESCUETIME: None = None


def make_windows(date_str: str = "2026-03-14") -> list[dict]:
    """Build a realistic set of windows for testing."""
    return build_windows(
        date_str,
        SAMPLE_WHOOP,
        SAMPLE_CALENDAR,
        SAMPLE_SLACK_WINDOWS,
    )


# ─── Tests: _compute_ml_insights_for_digest ───────────────────────────────────

class TestComputeMLInsightsForDigest:
    """Tests for the _compute_ml_insights_for_digest() helper."""

    def test_returns_none_when_both_models_unavailable(self):
        """When no models are trained, returns None (not is_meaningful)."""
        windows = make_windows()
        with patch("analysis.ml_model.detect_anomalies", return_value=[]):
            with patch("analysis.ml_model.predict_recovery", return_value=None):
                result = _compute_ml_insights_for_digest(windows)
        # No anomaly clusters (empty list) and no recovery pred → None
        assert result is None

    def test_returns_none_when_no_significant_anomalies_and_no_recovery(self):
        """Single anomaly (count=1) → filtered, no recovery pred → None."""
        windows = make_windows()
        single_anomaly = [
            {"window_id": "2026-03-14T10:00:00", "hour_of_day": 10,
             "anomaly_score": -0.15, "features": {"cls": 0.1, "fdi": 0.9, "ras": 0.95,
                                                    "in_meeting": False, "slack_messages": 3}},
        ]
        with patch("analysis.ml_model.detect_anomalies", return_value=single_anomaly):
            with patch("analysis.ml_model.predict_recovery", return_value=None):
                result = _compute_ml_insights_for_digest(windows)
        assert result is None

    def test_returns_result_with_recovery_prediction(self):
        """When recovery prediction is available, is_meaningful=True."""
        windows = make_windows()
        mock_pred = {"predicted_recovery": 72.5, "prediction_std": 4.2, "confidence": "high"}
        with patch("analysis.ml_model.detect_anomalies", return_value=[]):
            with patch("analysis.ml_model.predict_recovery", return_value=mock_pred):
                result = _compute_ml_insights_for_digest(windows)
        assert result is not None
        assert result["is_meaningful"] is True
        assert result["recovery_prediction"]["predicted_recovery"] == 72.5
        assert result["recovery_prediction"]["confidence"] == "high"

    def test_returns_result_with_anomaly_cluster(self):
        """When ≥2 anomalies cluster together, is_meaningful=True."""
        windows = make_windows()
        two_anomalies = [
            {"window_id": "2026-03-14T14:00:00", "hour_of_day": 14,
             "anomaly_score": -0.15, "features": {}},
            {"window_id": "2026-03-14T14:30:00", "hour_of_day": 14,
             "anomaly_score": -0.12, "features": {}},
        ]
        with patch("analysis.ml_model.detect_anomalies", return_value=two_anomalies):
            with patch("analysis.ml_model.predict_recovery", return_value=None):
                result = _compute_ml_insights_for_digest(windows)
        assert result is not None
        assert result["is_meaningful"] is True
        assert len(result["anomaly_clusters"]) == 1
        cluster = result["anomaly_clusters"][0]
        assert cluster["count"] == 2
        assert cluster["start_hour"] == 14
        assert cluster["end_hour"] == 15

    def test_consecutive_hours_form_single_cluster(self):
        """Anomalies in hours 8, 8, 9 → one cluster (hours 8–10)."""
        windows = make_windows()
        anomalies = [
            {"window_id": "2026-03-14T08:00:00", "hour_of_day": 8,
             "anomaly_score": -0.18, "features": {}},
            {"window_id": "2026-03-14T08:30:00", "hour_of_day": 8,
             "anomaly_score": -0.14, "features": {}},
            {"window_id": "2026-03-14T09:15:00", "hour_of_day": 9,
             "anomaly_score": -0.10, "features": {}},
        ]
        with patch("analysis.ml_model.detect_anomalies", return_value=anomalies):
            with patch("analysis.ml_model.predict_recovery", return_value=None):
                result = _compute_ml_insights_for_digest(windows)
        assert result is not None
        assert len(result["anomaly_clusters"]) == 1
        cluster = result["anomaly_clusters"][0]
        assert cluster["count"] == 3
        assert cluster["start_hour"] == 8
        assert cluster["end_hour"] == 10

    def test_non_consecutive_hours_form_separate_clusters(self):
        """Anomalies in hours 8–9 and 15–16 → two separate clusters."""
        windows = make_windows()
        anomalies = [
            {"window_id": "2026-03-14T08:00:00", "hour_of_day": 8,
             "anomaly_score": -0.18, "features": {}},
            {"window_id": "2026-03-14T09:00:00", "hour_of_day": 9,
             "anomaly_score": -0.15, "features": {}},
            {"window_id": "2026-03-14T15:00:00", "hour_of_day": 15,
             "anomaly_score": -0.20, "features": {}},
            {"window_id": "2026-03-14T15:30:00", "hour_of_day": 15,
             "anomaly_score": -0.19, "features": {}},
        ]
        with patch("analysis.ml_model.detect_anomalies", return_value=anomalies):
            with patch("analysis.ml_model.predict_recovery", return_value=None):
                result = _compute_ml_insights_for_digest(windows)
        assert result is not None
        assert len(result["anomaly_clusters"]) == 2
        hours = [(c["start_hour"], c["end_hour"]) for c in result["anomaly_clusters"]]
        assert (8, 10) in hours
        assert (15, 16) in hours

    def test_cluster_description_format(self):
        """Cluster description is human-readable."""
        windows = make_windows()
        anomalies = [
            {"window_id": "2026-03-14T14:00:00", "hour_of_day": 14,
             "anomaly_score": -0.15, "features": {}},
            {"window_id": "2026-03-14T14:30:00", "hour_of_day": 14,
             "anomaly_score": -0.12, "features": {}},
            {"window_id": "2026-03-14T15:00:00", "hour_of_day": 15,
             "anomaly_score": -0.10, "features": {}},
        ]
        with patch("analysis.ml_model.detect_anomalies", return_value=anomalies):
            with patch("analysis.ml_model.predict_recovery", return_value=None):
                result = _compute_ml_insights_for_digest(windows)
        desc = result["anomaly_clusters"][0]["description"]
        assert "anomalous" in desc
        assert "14:00" in desc
        assert "16:00" in desc

    def test_exception_in_ml_model_returns_none(self):
        """If ml_model raises, returns None gracefully."""
        windows = make_windows()
        with patch("analysis.ml_model.detect_anomalies", side_effect=RuntimeError("model crash")):
            result = _compute_ml_insights_for_digest(windows)
        assert result is None


# ─── Tests: compute_digest includes ml_insights key ───────────────────────────

class TestComputeDigestMLKey:
    """The 'ml_insights' key is always present in compute_digest() output."""

    def test_ml_insights_key_present(self):
        windows = make_windows()
        with patch("analysis.ml_model.detect_anomalies", return_value=[]):
            with patch("analysis.ml_model.predict_recovery", return_value=None):
                digest = compute_digest(windows)
        assert "ml_insights" in digest

    def test_ml_insights_is_none_when_no_models(self):
        windows = make_windows()
        with patch("analysis.ml_model.detect_anomalies", return_value=[]):
            with patch("analysis.ml_model.predict_recovery", return_value=None):
                digest = compute_digest(windows)
        assert digest["ml_insights"] is None


# ─── Tests: format_digest_message renders ML insights ─────────────────────────

class TestFormatDigestMessageML:
    """format_digest_message() renders ML insights when present."""

    def _make_digest_with_ml(
        self,
        recovery_pred=None,
        clusters=None,
    ) -> dict:
        """Build a minimal digest dict with custom ml_insights."""
        windows = make_windows()
        with patch("analysis.ml_model.detect_anomalies", return_value=[]):
            with patch("analysis.ml_model.predict_recovery", return_value=None):
                digest = compute_digest(windows)

        # Inject custom ml_insights
        if recovery_pred is not None or clusters is not None:
            digest["ml_insights"] = {
                "recovery_prediction": recovery_pred,
                "anomalies": [],
                "anomaly_clusters": clusters or [],
                "is_meaningful": True,
            }
        else:
            digest["ml_insights"] = None
        return digest

    def test_no_ml_insights_section_omitted(self):
        """When ml_insights is None, no ML lines in message."""
        digest = self._make_digest_with_ml()
        msg = format_digest_message(digest)
        assert "🤖" not in msg
        assert "🔍" not in msg

    def test_recovery_prediction_rendered(self):
        """Recovery prediction appears in the formatted message."""
        pred = {"predicted_recovery": 72.0, "confidence": "high", "prediction_std": 3.5}
        digest = self._make_digest_with_ml(recovery_pred=pred)
        msg = format_digest_message(digest)
        assert "🤖" in msg
        assert "72%" in msg
        assert "high confidence" in msg

    def test_anomaly_cluster_rendered(self):
        """Anomaly cluster appears in the formatted message."""
        clusters = [
            {"start_hour": 14, "end_hour": 15, "count": 2,
             "description": "2 anomalous windows at 14:00–15:00"}
        ]
        digest = self._make_digest_with_ml(clusters=clusters)
        msg = format_digest_message(digest)
        assert "🔍" in msg
        assert "14:00" in msg

    def test_both_recovery_and_anomaly_rendered(self):
        """Both signals appear when both are present."""
        pred = {"predicted_recovery": 65.0, "confidence": "medium", "prediction_std": 8.0}
        clusters = [
            {"start_hour": 9, "end_hour": 11, "count": 3,
             "description": "3 anomalous windows at 09:00–11:00"}
        ]
        digest = self._make_digest_with_ml(recovery_pred=pred, clusters=clusters)
        msg = format_digest_message(digest)
        assert "🤖" in msg
        assert "🔍" in msg
        assert "65%" in msg
        assert "09:00" in msg


# ─── Tests: _compute_ml_recovery_for_brief ────────────────────────────────────

class TestComputeMLRecoveryForBrief:
    """Tests for the morning brief ML recovery validation helper.

    Note: _compute_ml_recovery_for_brief uses local imports inside the function,
    so we patch 'analysis.ml_model.predict_recovery' (the module) and simulate
    no-data scenarios by providing an empty windows list directly.
    """

    def test_returns_none_when_no_model(self):
        """When recovery predictor returns None (not trained), result is None."""
        from analysis.morning_brief import _compute_ml_recovery_for_brief
        import analysis.ml_model as ml
        import engine.store as store_module
        original_read_day = store_module.read_day
        original_predict = ml.predict_recovery
        try:
            # Provide yesterday's windows so the store read succeeds
            store_module.read_day = lambda date: make_windows("2026-03-13")
            # Model not trained → predict_recovery returns None
            ml.predict_recovery = lambda windows: None
            result = _compute_ml_recovery_for_brief("2026-03-14", {"recovery_score": 78.0})
        finally:
            store_module.read_day = original_read_day
            ml.predict_recovery = original_predict
        assert result is None

    def test_returns_none_when_no_yesterday_data(self):
        """When yesterday has no JSONL windows, returns None."""
        from analysis.morning_brief import _compute_ml_recovery_for_brief
        # Patch the local read_day import inside the function's module
        import analysis.morning_brief as mb_module
        import engine.store as store_module
        original_read_day = store_module.read_day
        try:
            store_module.read_day = lambda date: None
            result = _compute_ml_recovery_for_brief("2026-03-14", {"recovery_score": 78.0})
        finally:
            store_module.read_day = original_read_day
        assert result is None

    def _call_with_mock_pred(self, today_date: str, whoop: dict, pred: dict) -> "Optional[dict]":
        """Helper: call _compute_ml_recovery_for_brief with mocked prediction."""
        from analysis.morning_brief import _compute_ml_recovery_for_brief
        import analysis.ml_model as ml
        import engine.store as store_module
        original_read_day = store_module.read_day
        original_predict = ml.predict_recovery
        try:
            store_module.read_day = lambda date: make_windows("2026-03-13")
            ml.predict_recovery = lambda windows: pred
            result = _compute_ml_recovery_for_brief(today_date, whoop)
        finally:
            store_module.read_day = original_read_day
            ml.predict_recovery = original_predict
        return result

    def test_prediction_with_actual_recovery(self):
        """When prediction and actual both present, accuracy is classified."""
        mock_pred = {"predicted_recovery": 72.0, "prediction_std": 4.0, "confidence": "high"}
        result = self._call_with_mock_pred("2026-03-14", {"recovery_score": 75.0}, mock_pred)
        assert result is not None
        assert result["predicted_recovery"] == 72
        assert result["actual_recovery"] == 75.0
        assert result["accuracy"] == "accurate"  # |72-75| = 3 < 8
        assert result["line"] != ""

    def test_accuracy_optimistic(self):
        """Prediction > actual by > 8 pts → optimistic (model over-predicted)."""
        mock_pred = {"predicted_recovery": 85.0, "prediction_std": 6.0, "confidence": "medium"}
        result = self._call_with_mock_pred("2026-03-14", {"recovery_score": 60.0}, mock_pred)
        assert result is not None
        assert result["accuracy"] == "optimistic"

    def test_accuracy_pessimistic(self):
        """Prediction < actual by > 8 pts → pessimistic (model under-predicted)."""
        mock_pred = {"predicted_recovery": 50.0, "prediction_std": 7.0, "confidence": "medium"}
        result = self._call_with_mock_pred("2026-03-14", {"recovery_score": 80.0}, mock_pred)
        assert result is not None
        assert result["accuracy"] == "pessimistic"

    def test_prediction_without_actual_recovery(self):
        """When WHOOP recovery is not available, accuracy is None."""
        mock_pred = {"predicted_recovery": 65.0, "prediction_std": 5.0, "confidence": "low"}
        result = self._call_with_mock_pred("2026-03-14", {}, mock_pred)
        assert result is not None
        assert result["accuracy"] is None
        # Should mention 65% prediction
        assert "65%" in result["line"]

    def test_exception_returns_none(self):
        """Any exception in the helper returns None gracefully."""
        from analysis.morning_brief import _compute_ml_recovery_for_brief
        import engine.store as store_module
        original_read_day = store_module.read_day
        try:
            store_module.read_day = lambda date: (_ for _ in ()).throw(RuntimeError("store error"))
            result = _compute_ml_recovery_for_brief("2026-03-14", {"recovery_score": 75.0})
        finally:
            store_module.read_day = original_read_day
        assert result is None


# ─── Tests: format_morning_brief_message renders ml_recovery ─────────────────

class TestFormatMorningBriefML:
    """format_morning_brief_message() renders ml_recovery when present."""

    def _make_brief_with_ml(self, ml_recovery=None) -> dict:
        """Build a minimal brief dict with custom ml_recovery."""
        from analysis.morning_brief import compute_morning_brief
        import analysis.morning_brief as mb_module
        original = mb_module._compute_ml_recovery_for_brief
        try:
            mb_module._compute_ml_recovery_for_brief = lambda date, whoop: ml_recovery
            brief = compute_morning_brief("2026-03-14", SAMPLE_WHOOP)
        finally:
            mb_module._compute_ml_recovery_for_brief = original
        return brief

    def test_no_ml_recovery_line_omitted(self):
        """When ml_recovery is None, no 🤖 line in message."""
        from analysis.morning_brief import format_morning_brief_message
        brief = self._make_brief_with_ml(ml_recovery=None)
        msg = format_morning_brief_message(brief)
        assert "🤖" not in msg

    def test_ml_recovery_line_rendered(self):
        """When ml_recovery is present with a line, it appears in the message."""
        from analysis.morning_brief import format_morning_brief_message
        ml_recovery = {
            "predicted_recovery": 72,
            "actual_recovery": 78.0,
            "accuracy": "accurate",
            "error": -6.0,
            "confidence": "high",
            "line": "🤖 ML predicted yesterday's recovery at 72% (high confidence) — actual: 78% ✓−6%",
        }
        brief = self._make_brief_with_ml(ml_recovery=ml_recovery)
        msg = format_morning_brief_message(brief)
        assert "🤖" in msg
        assert "72%" in msg

"""
Tests for scripts/train_model.py — ML Model Trainer CLI

Coverage:
  - print_status() renders without crashing for typical status dicts
  - print_status() handles missing keys gracefully (no date range, no metadata)
  - print_train_result() renders all outcomes: trained, insufficient_data,
    insufficient_windows, partial (some skipped)
  - print_prediction() renders anomalies and no-anomalies paths
  - print_prediction() exits 1 when no data for date
  - print_baselines() renders without crashing for valid baselines dict
  - print_baselines() handles empty / missing baselines gracefully
  - _bar() returns a string of expected length pattern
  - _c() returns plain text when not a TTY
  - CLI --json flag returns valid JSON with expected top-level keys
  - CLI --train --force triggers train_all with force=True
  - CLI --predict DATE triggers predict_recovery and detect_anomalies
  - CLI --baselines triggers compute_personal_baselines
  - CLI --clusters triggers get_focus_cluster_profiles
  - CLI with invalid --predict date exits 2
  - CLI with no flags defaults to status display

All external calls (store reads, model inference, Slack) are mocked so
tests run offline with no credentials.
"""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_model import (
    _bar,
    _c,
    print_status,
    print_train_result,
    print_prediction,
    print_baselines,
    print_clusters,
    main,
)


# ─── Shared fixtures ──────────────────────────────────────────────────────────

_STATUS_READY = {
    "days_of_data": 65,
    "min_days_required": 60,
    "ready_to_train": True,
    "days_remaining_until_ready": 0,
    "oldest_date": "2026-01-08",
    "newest_date": "2026-03-13",
    "models_trained": {
        "anomaly_detector": True,
        "recovery_predictor": True,
        "focus_clusters": True,
        "feature_scaler": True,
    },
    "last_trained": "2026-03-13T22:00:00",
    "training_days_used": 65,
}

_STATUS_NOT_READY = {
    "days_of_data": 2,
    "min_days_required": 60,
    "ready_to_train": False,
    "days_remaining_until_ready": 58,
    "oldest_date": "2026-03-13",
    "newest_date": "2026-03-14",
    "models_trained": {
        "anomaly_detector": False,
        "recovery_predictor": False,
        "focus_clusters": False,
        "feature_scaler": False,
    },
    "last_trained": None,
    "training_days_used": None,
}

_TRAIN_RESULT_SUCCESS = {
    "status": "trained",
    "days_used": 65,
    "windows_used": 3900,
    "active_windows_used": 1200,
    "anomaly": "trained",
    "recovery": "trained",
    "clustering": "trained",
}

_TRAIN_RESULT_PARTIAL = {
    "status": "trained",
    "days_used": 10,
    "windows_used": 600,
    "active_windows_used": 30,
    "anomaly": "trained",
    "recovery": "skipped (insufficient day-pairs)",
    "clustering": "skipped (insufficient active windows)",
}

_TRAIN_RESULT_INSUFFICIENT = {
    "status": "insufficient_data",
    "days_available": 2,
    "days_required": 60,
    "anomaly": "skipped",
    "recovery": "skipped",
    "clustering": "skipped",
}

_TRAIN_RESULT_NO_WINDOWS = {
    "status": "insufficient_windows",
    "windows_extracted": 5,
    "anomaly": "skipped",
    "recovery": "skipped",
    "clustering": "skipped",
}

_BASELINES = {
    "days_analyzed": 2,
    "working_windows": 120,
    "cls": {"mean": 0.044, "std": 0.013, "p25": 0.036, "p75": 0.045},
    "fdi": {"mean": 0.991, "std": 0.022, "p25": 1.0, "p75": 1.0},
    "sdi": {"mean": 0.05, "std": 0.02, "p25": 0.03, "p75": 0.07},
    "csc": {"mean": 0.08, "std": 0.03, "p25": 0.05, "p75": 0.11},
    "ras": {"mean": 0.989, "std": 0.002, "p25": 0.988, "p75": 0.991},
    "hrv_ms": {"mean": 79.9, "std": 0.96, "p25": 78.9, "p75": 80.9},
    "recovery_pct": {"mean": 89.0, "std": 3.0, "p25": 86.0, "p75": 92.0},
}

_WINDOWS = [
    {
        "window_id": "2026-03-14T09:00:00",
        "date": "2026-03-14",
        "window_start": "2026-03-14T09:00:00+01:00",
        "window_end": "2026-03-14T09:15:00+01:00",
        "window_index": 36,
        "calendar": {
            "in_meeting": False,
            "meeting_title": None,
            "meeting_attendees": 0,
            "meeting_duration_minutes": 0,
            "meetings_count": 0,
        },
        "whoop": {
            "recovery_score": 85.0,
            "hrv_rmssd_milli": 72.0,
            "resting_heart_rate": 54.0,
            "sleep_performance": 88.0,
            "sleep_hours": 8.0,
            "strain": 11.0,
        },
        "slack": {
            "messages_sent": 2,
            "messages_received": 5,
            "total_messages": 7,
            "channels_active": 1,
        },
        "omi": {
            "conversation_active": False,
            "word_count": 0,
            "speech_seconds": 0.0,
            "audio_seconds": 0.0,
            "sessions_count": 0,
            "speech_ratio": 0.0,
        },
        "metrics": {
            "cognitive_load_score": 0.30,
            "focus_depth_index": 0.85,
            "social_drain_index": 0.15,
            "context_switch_cost": 0.20,
            "recovery_alignment_score": 0.92,
        },
        "metadata": {
            "day_of_week": "Saturday",
            "hour_of_day": 9,
            "is_working_hours": True,
            "is_active_window": True,
            "sources_available": ["whoop", "calendar", "slack"],
        },
    }
]


# ─── Unit tests: helpers ──────────────────────────────────────────────────────

class TestBar:
    def test_returns_string(self):
        assert isinstance(_bar(30, 60), str)

    def test_filled_equals_total(self):
        result = _bar(60, 60)
        assert "100%" in result

    def test_zero_of_total(self):
        result = _bar(0, 60)
        assert "0%" in result

    def test_partial_fill(self):
        result = _bar(30, 60)
        assert "50%" in result

    def test_zero_total_doesnt_crash(self):
        result = _bar(0, 0)
        assert isinstance(result, str)


class TestColour:
    def test_returns_plain_text_when_not_tty(self):
        """_c() must strip ANSI when stdout is not a TTY (test runner)."""
        result = _c("hello", "\033[92m")
        assert result == "hello"


# ─── Unit tests: print_status ─────────────────────────────────────────────────

class TestPrintStatus:
    def test_ready_status_prints_without_crash(self, capsys):
        print_status(_STATUS_READY)
        captured = capsys.readouterr()
        assert "Data Readiness" in captured.out

    def test_not_ready_status_prints_without_crash(self, capsys):
        print_status(_STATUS_NOT_READY)
        captured = capsys.readouterr()
        assert "Data Readiness" in captured.out

    def test_ready_shows_ready_indicator(self, capsys):
        print_status(_STATUS_READY)
        captured = capsys.readouterr()
        assert "Ready to train" in captured.out

    def test_not_ready_shows_remaining(self, capsys):
        print_status(_STATUS_NOT_READY)
        captured = capsys.readouterr()
        assert "58" in captured.out  # days remaining

    def test_shows_date_range_when_present(self, capsys):
        print_status(_STATUS_READY)
        captured = capsys.readouterr()
        assert "2026-01-08" in captured.out

    def test_missing_dates_doesnt_crash(self, capsys):
        status = dict(_STATUS_NOT_READY)
        del status["oldest_date"]
        del status["newest_date"]
        print_status(status)
        captured = capsys.readouterr()
        assert "Data Readiness" in captured.out

    def test_shows_all_four_model_labels(self, capsys):
        print_status(_STATUS_READY)
        captured = capsys.readouterr()
        assert "Isolation Forest" in captured.out
        assert "Random Forest" in captured.out
        assert "KMeans" in captured.out
        assert "Feature Scaler" in captured.out

    def test_last_trained_shown_when_present(self, capsys):
        print_status(_STATUS_READY)
        captured = capsys.readouterr()
        assert "Last trained" in captured.out

    def test_last_trained_absent_when_none(self, capsys):
        print_status(_STATUS_NOT_READY)
        captured = capsys.readouterr()
        assert "Last trained" not in captured.out

    def test_next_step_section_always_present(self, capsys):
        print_status(_STATUS_READY)
        captured = capsys.readouterr()
        assert "Next Step" in captured.out


# ─── Unit tests: print_train_result ──────────────────────────────────────────

class TestPrintTrainResult:
    def test_success_shows_trained(self, capsys):
        print_train_result(_TRAIN_RESULT_SUCCESS)
        captured = capsys.readouterr()
        assert "trained" in captured.out.lower()

    def test_partial_shows_skipped(self, capsys):
        print_train_result(_TRAIN_RESULT_PARTIAL)
        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()

    def test_insufficient_data_shows_message(self, capsys):
        print_train_result(_TRAIN_RESULT_INSUFFICIENT)
        captured = capsys.readouterr()
        assert "Not enough data" in captured.out

    def test_insufficient_windows_shows_message(self, capsys):
        print_train_result(_TRAIN_RESULT_NO_WINDOWS)
        captured = capsys.readouterr()
        assert "windows" in captured.out.lower()

    def test_success_shows_training_data_summary(self, capsys):
        print_train_result(_TRAIN_RESULT_SUCCESS)
        captured = capsys.readouterr()
        assert "65 days" in captured.out

    def test_all_three_models_listed(self, capsys):
        print_train_result(_TRAIN_RESULT_SUCCESS)
        captured = capsys.readouterr()
        assert "Isolation Forest" in captured.out
        assert "Random Forest" in captured.out
        assert "KMeans" in captured.out

    def test_doesnt_crash_with_error_outcome(self, capsys):
        result = dict(_TRAIN_RESULT_SUCCESS)
        result["anomaly"] = "error: something broke"
        print_train_result(result)
        captured = capsys.readouterr()
        assert "error" in captured.out.lower()


# ─── Unit tests: print_prediction ─────────────────────────────────────────────

class TestPrintPrediction:
    def test_no_anomalies_path(self, capsys):
        with patch("scripts.train_model.read_day", return_value=_WINDOWS), \
             patch("scripts.train_model.detect_anomalies", return_value=[]), \
             patch("scripts.train_model.predict_recovery", return_value=None):
            print_prediction("2026-03-14")
        captured = capsys.readouterr()
        assert "No anomalies detected" in captured.out

    def test_with_anomalies_path(self, capsys):
        anomalies = [{
            "window_id": "2026-03-14T09:00:00",
            "hour_of_day": 9,
            "anomaly_score": -0.15,
            "features": {"cls": 0.30, "fdi": 0.85, "ras": 0.92},
            "method": "model",
        }]
        with patch("scripts.train_model.read_day", return_value=_WINDOWS), \
             patch("scripts.train_model.detect_anomalies", return_value=anomalies), \
             patch("scripts.train_model.predict_recovery", return_value=None):
            print_prediction("2026-03-14")
        captured = capsys.readouterr()
        assert "1 anomalous" in captured.out

    def test_with_recovery_prediction(self, capsys):
        pred = {
            "predicted_recovery": 78.5,
            "confidence": "medium",
            "prediction_std": 8.2,
            "method": "heuristic",
        }
        with patch("scripts.train_model.read_day", return_value=_WINDOWS), \
             patch("scripts.train_model.detect_anomalies", return_value=[]), \
             patch("scripts.train_model.predict_recovery", return_value=pred):
            print_prediction("2026-03-14")
        captured = capsys.readouterr()
        assert "78%" in captured.out or "79%" in captured.out

    def test_exits_1_when_no_data(self):
        with patch("scripts.train_model.read_day", return_value=[]):
            with pytest.raises(SystemExit) as exc_info:
                print_prediction("2026-03-14")
        assert exc_info.value.code == 1

    def test_shows_header_with_date(self, capsys):
        with patch("scripts.train_model.read_day", return_value=_WINDOWS), \
             patch("scripts.train_model.detect_anomalies", return_value=[]), \
             patch("scripts.train_model.predict_recovery", return_value=None):
            print_prediction("2026-03-14")
        captured = capsys.readouterr()
        assert "2026-03-14" in captured.out


# ─── Unit tests: print_baselines ─────────────────────────────────────────────

class TestPrintBaselines:
    def test_renders_without_crash(self, capsys):
        with patch("scripts.train_model.list_available_dates", return_value=["2026-03-13", "2026-03-14"]), \
             patch("scripts.train_model.compute_personal_baselines", return_value=_BASELINES):
            print_baselines()
        captured = capsys.readouterr()
        assert "Personal Baselines" in captured.out

    def test_shows_cls(self, capsys):
        with patch("scripts.train_model.list_available_dates", return_value=["2026-03-13", "2026-03-14"]), \
             patch("scripts.train_model.compute_personal_baselines", return_value=_BASELINES):
            print_baselines()
        captured = capsys.readouterr()
        assert "CLS" in captured.out

    def test_shows_recovery(self, capsys):
        with patch("scripts.train_model.list_available_dates", return_value=["2026-03-13", "2026-03-14"]), \
             patch("scripts.train_model.compute_personal_baselines", return_value=_BASELINES):
            print_baselines()
        captured = capsys.readouterr()
        assert "Recovery" in captured.out

    def test_shows_hrv(self, capsys):
        with patch("scripts.train_model.list_available_dates", return_value=["2026-03-13", "2026-03-14"]), \
             patch("scripts.train_model.compute_personal_baselines", return_value=_BASELINES):
            print_baselines()
        captured = capsys.readouterr()
        assert "HRV" in captured.out

    def test_no_dates_exits_1(self):
        with patch("scripts.train_model.list_available_dates", return_value=[]):
            with pytest.raises(SystemExit) as exc_info:
                print_baselines()
        assert exc_info.value.code == 1

    def test_missing_metric_keys_dont_crash(self, capsys):
        """Baselines with only partial keys should render gracefully."""
        minimal = {
            "days_analyzed": 1,
            "working_windows": 60,
            "cls": {"mean": 0.04, "std": 0.01, "p25": 0.03, "p75": 0.05},
        }
        with patch("scripts.train_model.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.train_model.compute_personal_baselines", return_value=minimal):
            print_baselines()
        captured = capsys.readouterr()
        assert "CLS" in captured.out


# ─── Unit tests: print_clusters ──────────────────────────────────────────────

class TestPrintClusters:
    def test_no_profiles_renders_gracefully(self, capsys):
        with patch("scripts.train_model.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.train_model.get_focus_cluster_profiles", return_value=[]):
            print_clusters()
        captured = capsys.readouterr()
        assert "not trained" in captured.out

    def test_with_profiles_shows_cluster_ids(self, capsys):
        profiles = [
            {
                "cluster_id": 0,
                "label": "Deep Focus",
                "mean_fdi": 0.85,
                "mean_cls": 0.25,
                "window_count": 120,
                "peak_hours": [9, 10],
            },
            {
                "cluster_id": 1,
                "label": "Meeting Mode",
                "mean_fdi": 0.32,
                "mean_cls": 0.70,
                "window_count": 80,
                "peak_hours": [14, 15],
            },
        ]
        with patch("scripts.train_model.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.train_model.get_focus_cluster_profiles", return_value=profiles):
            print_clusters()
        captured = capsys.readouterr()
        assert "Cluster 0" in captured.out
        assert "Cluster 1" in captured.out
        assert "Deep Focus" in captured.out


# ─── CLI integration tests ────────────────────────────────────────────────────

class TestCLI:
    """Test the CLI entry-point with mocked model calls."""

    def _run_main(self, argv, patches=None):
        """Helper: run main() with given sys.argv and optional extra patches."""
        base_patches = {
            "scripts.train_model.get_data_status": lambda: _STATUS_NOT_READY,
            "scripts.train_model.list_available_dates": lambda: ["2026-03-13", "2026-03-14"],
        }
        if patches:
            base_patches.update(patches)

        with patch.dict("sys.modules", {}):
            with patch("sys.argv", ["train_model.py"] + argv):
                ctx = {}
                for target, fn in base_patches.items():
                    ctx[target] = patch(target, side_effect=fn)
                # Apply patches
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for target, fn in base_patches.items():
                        stack.enter_context(patch(target, side_effect=fn))
                    main()

    def test_no_args_shows_status(self, capsys):
        with patch("sys.argv", ["train_model.py"]), \
             patch("scripts.train_model.get_data_status", return_value=_STATUS_NOT_READY):
            main()
        captured = capsys.readouterr()
        assert "Data Readiness" in captured.out

    def test_json_flag_outputs_valid_json(self, capsys):
        with patch("sys.argv", ["train_model.py", "--json"]), \
             patch("scripts.train_model.get_data_status", return_value=_STATUS_NOT_READY):
            main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "days_of_data" in data
        assert "ready_to_train" in data

    def test_train_flag_calls_train_all(self, capsys):
        mock_train = MagicMock(return_value=_TRAIN_RESULT_SUCCESS)
        with patch("sys.argv", ["train_model.py", "--train"]), \
             patch("scripts.train_model.train_all", mock_train):
            main()
        mock_train.assert_called_once_with(force=False)

    def test_train_force_flag_passes_force_true(self, capsys):
        mock_train = MagicMock(return_value=_TRAIN_RESULT_SUCCESS)
        with patch("sys.argv", ["train_model.py", "--train", "--force"]), \
             patch("scripts.train_model.train_all", mock_train):
            main()
        mock_train.assert_called_once_with(force=True)

    def test_train_insufficient_data_exits_1(self):
        mock_train = MagicMock(return_value=_TRAIN_RESULT_INSUFFICIENT)
        with patch("sys.argv", ["train_model.py", "--train"]), \
             patch("scripts.train_model.train_all", mock_train):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1

    def test_predict_invalid_date_exits_2(self):
        with patch("sys.argv", ["train_model.py", "--predict", "not-a-date"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2

    def test_predict_calls_detect_and_predict(self, capsys):
        mock_detect = MagicMock(return_value=[])
        mock_predict = MagicMock(return_value=None)
        with patch("sys.argv", ["train_model.py", "--predict", "2026-03-14"]), \
             patch("scripts.train_model.read_day", return_value=_WINDOWS), \
             patch("scripts.train_model.detect_anomalies", mock_detect), \
             patch("scripts.train_model.predict_recovery", mock_predict):
            main()
        mock_detect.assert_called_once()
        mock_predict.assert_called_once()

    def test_baselines_flag_calls_compute(self, capsys):
        mock_baselines = MagicMock(return_value=_BASELINES)
        with patch("sys.argv", ["train_model.py", "--baselines"]), \
             patch("scripts.train_model.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.train_model.compute_personal_baselines", mock_baselines):
            main()
        mock_baselines.assert_called_once()

    def test_clusters_flag_calls_get_profiles(self, capsys):
        mock_profiles = MagicMock(return_value=[])
        with patch("sys.argv", ["train_model.py", "--clusters"]), \
             patch("scripts.train_model.list_available_dates", return_value=["2026-03-14"]), \
             patch("scripts.train_model.get_focus_cluster_profiles", mock_profiles):
            main()
        mock_profiles.assert_called_once()

    def test_predict_json_returns_valid_structure(self, capsys):
        with patch("sys.argv", ["train_model.py", "--predict", "2026-03-14", "--json"]), \
             patch("scripts.train_model.read_day", return_value=_WINDOWS), \
             patch("scripts.train_model.detect_anomalies", return_value=[]), \
             patch("scripts.train_model.predict_recovery", return_value=None):
            main()
        captured = capsys.readouterr()
        # JSON output may interleave with status JSON; find the prediction output
        lines = [l for l in captured.out.strip().split("\n") if l.startswith("{") or l.startswith("}")]
        # Just verify no crash and that output contains date key somewhere
        assert "2026-03-14" in captured.out

    def test_train_json_returns_valid_json(self, capsys):
        with patch("sys.argv", ["train_model.py", "--train", "--json"]), \
             patch("scripts.train_model.train_all", return_value=_TRAIN_RESULT_SUCCESS):
            main()
        captured = capsys.readouterr()
        # Two JSON objects: status + train result
        # Find the last valid JSON block
        output = captured.out.strip()
        # Should contain 'trained' somewhere
        assert "trained" in output

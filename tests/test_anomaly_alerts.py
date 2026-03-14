"""
Tests for the anomaly alerts module (v5).

Run with: python3 -m pytest tests/test_anomaly_alerts.py -v

All tests are pure unit tests — no network, no Slack, no file I/O unless
explicitly mocking the store layer.

Coverage:
  - detect_cls_spike(): not enough history, no spike, spike fires
  - detect_fdi_collapse(): no baseline, collapse fires, no collapse
  - detect_recovery_misalignment_streak(): no streak, short streak, fires
  - check_anomalies(): aggregation, any_triggered flag
  - format_alert_message(): each alert type rendered, combined message
  - Edge cases: missing data, zero values, sparse history
"""

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from analysis.anomaly_alerts import (
    CLS_SPIKE_MIN_BASELINE_DAYS,
    CLS_SPIKE_STD_MULTIPLIER,
    FDI_COLLAPSE_THRESHOLD,
    FDI_MIN_BASELINE,
    RECOVERY_MISALIGN_RAS,
    RECOVERY_MISALIGN_STREAK,
    _mean,
    _std,
    check_anomalies,
    detect_cls_spike,
    detect_fdi_collapse,
    detect_recovery_misalignment_streak,
    format_alert_message,
)


# ─── Math helpers ─────────────────────────────────────────────────────────────

class TestMean:
    def test_empty_list_returns_none(self):
        assert _mean([]) is None

    def test_single_value(self):
        assert _mean([0.5]) == pytest.approx(0.5)

    def test_multiple_values(self):
        assert _mean([0.2, 0.4, 0.6]) == pytest.approx(0.4)

    def test_zeros(self):
        assert _mean([0.0, 0.0]) == pytest.approx(0.0)


class TestStd:
    def test_empty_returns_zero(self):
        assert _std([]) == 0.0

    def test_single_returns_zero(self):
        assert _std([0.5]) == 0.0

    def test_uniform_values_zero_std(self):
        assert _std([0.5, 0.5, 0.5]) == pytest.approx(0.0)

    def test_known_std(self):
        # values [0.2, 0.4, 0.6] → mean=0.4, population std ≈ 0.1633
        result = _std([0.2, 0.4, 0.6])
        assert result == pytest.approx(math.sqrt((0.04 + 0.0 + 0.04) / 3))

    def test_two_values(self):
        # [0.0, 1.0] → mean=0.5, var=0.25, std=0.5
        assert _std([0.0, 1.0]) == pytest.approx(0.5)


# ─── CLS spike detection ──────────────────────────────────────────────────────

class TestDetectClsSpike:
    TODAY = "2026-03-14"
    BASELINE = ["2026-03-13", "2026-03-12", "2026-03-11",
                "2026-03-10", "2026-03-09", "2026-03-08", "2026-03-07"]

    def _patch(self, baseline_cls: list[float], today_cls: Optional[float]):
        """Return patches for _baseline_dates and _get_daily_cls."""
        def mock_get_daily_cls(date_str: str) -> Optional[float]:
            if date_str == self.TODAY:
                return today_cls
            idx = self.BASELINE.index(date_str) if date_str in self.BASELINE else -1
            if idx == -1:
                return None
            return baseline_cls[idx] if idx < len(baseline_cls) else None

        return mock_get_daily_cls

    def test_not_enough_history_returns_none(self):
        """Fewer than CLS_SPIKE_MIN_BASELINE_DAYS baseline dates → no alert."""
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=["2026-03-13"]), \
             patch("analysis.anomaly_alerts._get_daily_cls", return_value=0.95):
            result = detect_cls_spike(self.TODAY)
        assert result is None

    def test_no_today_data_returns_none(self):
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE[:5]), \
             patch("analysis.anomaly_alerts._get_daily_cls", side_effect=lambda d: None if d == self.TODAY else 0.4):
            result = detect_cls_spike(self.TODAY)
        assert result is None

    def test_no_spike_returns_none(self):
        """Today is within 2σ → no alert."""
        # baseline mean ≈ 0.40, std ≈ 0.02, threshold ≈ 0.44
        # today=0.42 is clearly below threshold → no spike
        baseline_cls = [0.40, 0.42, 0.38, 0.41, 0.39, 0.43, 0.37]
        mock_cls = self._patch(baseline_cls, today_cls=0.42)
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE), \
             patch("analysis.anomaly_alerts._get_daily_cls", side_effect=mock_cls):
            result = detect_cls_spike(self.TODAY)
        assert result is None

    def test_spike_fires(self):
        """Today is >2σ above baseline → alert returned."""
        # Baseline near 0.40 with very small std
        baseline_cls = [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40]
        mock_cls = self._patch(baseline_cls, today_cls=0.95)  # huge spike
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE), \
             patch("analysis.anomaly_alerts._get_daily_cls", side_effect=mock_cls):
            result = detect_cls_spike(self.TODAY)
        assert result is not None
        assert result["today_cls"] == pytest.approx(0.95)
        assert result["baseline_mean"] == pytest.approx(0.40)
        assert result["baseline_std"] == pytest.approx(0.0)
        assert result["threshold"] == pytest.approx(0.40)
        assert result["days_used"] == 7

    def test_spike_dict_has_required_keys(self):
        """Spike result has all expected keys."""
        baseline_cls = [0.40] * 7
        mock_cls = self._patch(baseline_cls, today_cls=0.99)
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE), \
             patch("analysis.anomaly_alerts._get_daily_cls", side_effect=mock_cls):
            result = detect_cls_spike(self.TODAY)
        assert result is not None
        for key in ("today_cls", "baseline_mean", "baseline_std", "threshold", "days_used"):
            assert key in result

    def test_exactly_at_threshold_no_alert(self):
        """CLS exactly at mean+2σ is NOT a spike (must be strictly greater)."""
        baseline_cls = [0.30, 0.50, 0.30, 0.50, 0.30, 0.50, 0.40]
        # mean=0.4, std = ... compute expected threshold
        mean = sum(baseline_cls) / len(baseline_cls)
        std = _std(baseline_cls)
        threshold = mean + CLS_SPIKE_STD_MULTIPLIER * std
        mock_cls = self._patch(baseline_cls, today_cls=threshold)
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE), \
             patch("analysis.anomaly_alerts._get_daily_cls", side_effect=mock_cls):
            result = detect_cls_spike(self.TODAY)
        # threshold is not exceeded (today == threshold, not >)
        assert result is None


# ─── FDI collapse detection ───────────────────────────────────────────────────

class TestDetectFdiCollapse:
    TODAY = "2026-03-14"
    BASELINE = ["2026-03-13", "2026-03-12", "2026-03-11"]

    def _mock_fdi(self, baseline_fdi: list[float], today_fdi: Optional[float]):
        dates_map = dict(zip(self.BASELINE, baseline_fdi))
        def side_effect(d):
            if d == self.TODAY:
                return today_fdi
            return dates_map.get(d)
        return side_effect

    def test_no_baseline_returns_none(self):
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=[]), \
             patch("analysis.anomaly_alerts._get_daily_active_fdi", return_value=0.5):
            result = detect_fdi_collapse(self.TODAY)
        assert result is None

    def test_no_today_fdi_returns_none(self):
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE), \
             patch("analysis.anomaly_alerts._get_daily_active_fdi",
                   side_effect=self._mock_fdi([0.7, 0.6, 0.65], None)):
            result = detect_fdi_collapse(self.TODAY)
        assert result is None

    def test_baseline_too_low_skips(self):
        """Baseline below FDI_MIN_BASELINE → no comparison (avoids noise)."""
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE), \
             patch("analysis.anomaly_alerts._get_daily_active_fdi",
                   side_effect=self._mock_fdi([0.01, 0.02, 0.01], 0.0)):
            result = detect_fdi_collapse(self.TODAY)
        assert result is None

    def test_small_drop_no_alert(self):
        """Drop < 30 % → no alert."""
        # baseline avg = 0.70, drop to 0.55 → 21 % drop
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE), \
             patch("analysis.anomaly_alerts._get_daily_active_fdi",
                   side_effect=self._mock_fdi([0.70, 0.70, 0.70], 0.55)):
            result = detect_fdi_collapse(self.TODAY)
        assert result is None

    def test_large_drop_fires(self):
        """Drop > 30 % → alert fires."""
        # baseline avg = 0.70, today = 0.40 → 43 % drop
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE), \
             patch("analysis.anomaly_alerts._get_daily_active_fdi",
                   side_effect=self._mock_fdi([0.70, 0.70, 0.70], 0.40)):
            result = detect_fdi_collapse(self.TODAY)
        assert result is not None
        assert result["today_fdi"] == pytest.approx(0.40)
        assert result["baseline_fdi"] == pytest.approx(0.70)
        assert result["drop_pct"] == pytest.approx((0.70 - 0.40) / 0.70)

    def test_exact_30pct_drop_no_alert(self):
        """Exactly 30 % drop does NOT trigger (must be strictly greater)."""
        baseline = 0.70
        today = baseline * (1 - FDI_COLLAPSE_THRESHOLD)  # exactly 30 % drop
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE), \
             patch("analysis.anomaly_alerts._get_daily_active_fdi",
                   side_effect=self._mock_fdi([baseline, baseline, baseline], today)):
            result = detect_fdi_collapse(self.TODAY)
        assert result is None

    def test_result_has_required_keys(self):
        with patch("analysis.anomaly_alerts._baseline_dates", return_value=self.BASELINE), \
             patch("analysis.anomaly_alerts._get_daily_active_fdi",
                   side_effect=self._mock_fdi([0.80, 0.80, 0.80], 0.30)):
            result = detect_fdi_collapse(self.TODAY)
        assert result is not None
        for key in ("today_fdi", "baseline_fdi", "drop_pct", "days_used"):
            assert key in result


# ─── Recovery misalignment streak detection ───────────────────────────────────

class TestDetectRecoveryStreak:
    TODAY = "2026-03-14"

    def _available(self, dates: list[str]):
        return set(dates)

    def test_no_data_returns_none(self):
        with patch("analysis.anomaly_alerts.list_available_dates", return_value=[]), \
             patch("analysis.anomaly_alerts._get_daily_ras", return_value=0.30):
            result = detect_recovery_misalignment_streak(self.TODAY)
        assert result is None

    def test_today_ras_above_threshold_no_streak(self):
        """If today has good RAS, no streak."""
        dates = [self.TODAY, "2026-03-13", "2026-03-12"]
        with patch("analysis.anomaly_alerts.list_available_dates", return_value=dates), \
             patch("analysis.anomaly_alerts._get_daily_ras", return_value=0.80):
            result = detect_recovery_misalignment_streak(self.TODAY)
        assert result is None

    def test_streak_too_short(self):
        """2 consecutive misaligned days → no alert (need ≥ 3)."""
        dates = [self.TODAY, "2026-03-13", "2026-03-12"]
        ras_map = {
            self.TODAY: 0.30,
            "2026-03-13": 0.35,
            "2026-03-12": 0.80,   # breaks the streak
        }
        with patch("analysis.anomaly_alerts.list_available_dates", return_value=dates), \
             patch("analysis.anomaly_alerts._get_daily_ras", side_effect=lambda d: ras_map.get(d)):
            result = detect_recovery_misalignment_streak(self.TODAY)
        assert result is None

    def test_three_day_streak_fires(self):
        """3 consecutive misaligned days → alert fires."""
        dates = [self.TODAY, "2026-03-13", "2026-03-12", "2026-03-11"]
        ras_map = {
            self.TODAY: 0.30,
            "2026-03-13": 0.35,
            "2026-03-12": 0.40,
            "2026-03-11": 0.80,   # streak stops here
        }
        with patch("analysis.anomaly_alerts.list_available_dates", return_value=dates), \
             patch("analysis.anomaly_alerts._get_daily_ras", side_effect=lambda d: ras_map.get(d)):
            result = detect_recovery_misalignment_streak(self.TODAY)
        assert result is not None
        assert result["streak_days"] == 3
        assert len(result["ras_values"]) == 3
        assert result["avg_ras"] == pytest.approx((0.30 + 0.35 + 0.40) / 3)

    def test_long_streak_counts_all(self):
        """5 consecutive misaligned days → streak_days = 5."""
        base_dt = datetime.strptime(self.TODAY, "%Y-%m-%d")
        dates = [(base_dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6)]
        ras_map = {d: 0.30 for d in dates[:5]}  # 5 misaligned + 1 good
        ras_map[dates[5]] = 0.80
        with patch("analysis.anomaly_alerts.list_available_dates", return_value=dates), \
             patch("analysis.anomaly_alerts._get_daily_ras", side_effect=lambda d: ras_map.get(d)):
            result = detect_recovery_misalignment_streak(self.TODAY)
        assert result is not None
        assert result["streak_days"] == 5

    def test_gap_in_data_breaks_streak(self):
        """A date gap in available data resets the streak."""
        # TODAY and 2 days ago exist; yesterday is missing
        dates = [self.TODAY, "2026-03-12"]   # "2026-03-13" missing
        ras_map = {self.TODAY: 0.30, "2026-03-12": 0.30}
        with patch("analysis.anomaly_alerts.list_available_dates", return_value=dates), \
             patch("analysis.anomaly_alerts._get_daily_ras", side_effect=lambda d: ras_map.get(d)):
            result = detect_recovery_misalignment_streak(self.TODAY)
        assert result is None   # only 1 day before gap

    def test_result_has_required_keys(self):
        dates = [self.TODAY, "2026-03-13", "2026-03-12"]
        ras_map = {d: 0.30 for d in dates}
        with patch("analysis.anomaly_alerts.list_available_dates", return_value=dates), \
             patch("analysis.anomaly_alerts._get_daily_ras", side_effect=lambda d: ras_map.get(d)):
            result = detect_recovery_misalignment_streak(self.TODAY)
        assert result is not None
        for key in ("streak_days", "ras_values", "avg_ras"):
            assert key in result


# ─── Aggregated check_anomalies() ─────────────────────────────────────────────

class TestCheckAnomalies:
    TODAY = "2026-03-14"

    def test_no_alerts_any_triggered_false(self):
        with patch("analysis.anomaly_alerts.detect_cls_spike", return_value=None), \
             patch("analysis.anomaly_alerts.detect_fdi_collapse", return_value=None), \
             patch("analysis.anomaly_alerts.detect_recovery_misalignment_streak", return_value=None):
            result = check_anomalies(self.TODAY)
        assert result["any_triggered"] is False
        assert result["date"] == self.TODAY

    def test_one_alert_any_triggered_true(self):
        cls_alert = {"today_cls": 0.95, "baseline_mean": 0.40, "baseline_std": 0.0,
                     "threshold": 0.40, "days_used": 7}
        with patch("analysis.anomaly_alerts.detect_cls_spike", return_value=cls_alert), \
             patch("analysis.anomaly_alerts.detect_fdi_collapse", return_value=None), \
             patch("analysis.anomaly_alerts.detect_recovery_misalignment_streak", return_value=None):
            result = check_anomalies(self.TODAY)
        assert result["any_triggered"] is True
        assert result["alerts"]["cls_spike"] is not None
        assert result["alerts"]["fdi_collapse"] is None
        assert result["alerts"]["recovery_streak"] is None

    def test_all_alerts_any_triggered_true(self):
        cls_alert = {"today_cls": 0.9, "baseline_mean": 0.4, "baseline_std": 0.05,
                     "threshold": 0.5, "days_used": 5}
        fdi_alert = {"today_fdi": 0.3, "baseline_fdi": 0.7, "drop_pct": 0.57, "days_used": 7}
        streak_alert = {"streak_days": 4, "ras_values": [0.3]*4, "avg_ras": 0.3}
        with patch("analysis.anomaly_alerts.detect_cls_spike", return_value=cls_alert), \
             patch("analysis.anomaly_alerts.detect_fdi_collapse", return_value=fdi_alert), \
             patch("analysis.anomaly_alerts.detect_recovery_misalignment_streak", return_value=streak_alert):
            result = check_anomalies(self.TODAY)
        assert result["any_triggered"] is True
        assert all(v is not None for v in result["alerts"].values())

    def test_result_schema(self):
        with patch("analysis.anomaly_alerts.detect_cls_spike", return_value=None), \
             patch("analysis.anomaly_alerts.detect_fdi_collapse", return_value=None), \
             patch("analysis.anomaly_alerts.detect_recovery_misalignment_streak", return_value=None):
            result = check_anomalies(self.TODAY)
        assert "date" in result
        assert "alerts" in result
        assert "any_triggered" in result
        assert set(result["alerts"].keys()) == {"cls_spike", "fdi_collapse", "recovery_streak"}

    def test_defaults_to_today_date(self):
        """When called with no date, result date == today."""
        with patch("analysis.anomaly_alerts.detect_cls_spike", return_value=None), \
             patch("analysis.anomaly_alerts.detect_fdi_collapse", return_value=None), \
             patch("analysis.anomaly_alerts.detect_recovery_misalignment_streak", return_value=None):
            result = check_anomalies()
        today = datetime.now().strftime("%Y-%m-%d")
        assert result["date"] == today


# ─── format_alert_message() ───────────────────────────────────────────────────

class TestFormatAlertMessage:
    def _result(self, cls_alert=None, fdi_alert=None, streak_alert=None, date="2026-03-14"):
        return {
            "date": date,
            "alerts": {
                "cls_spike": cls_alert,
                "fdi_collapse": fdi_alert,
                "recovery_streak": streak_alert,
            },
            "any_triggered": any(v is not None for v in [cls_alert, fdi_alert, streak_alert]),
        }

    def test_no_alerts_returns_empty_string(self):
        assert format_alert_message(self._result()) == ""

    def test_cls_spike_message_contains_key_info(self):
        cls_alert = {
            "today_cls": 0.92,
            "baseline_mean": 0.41,
            "baseline_std": 0.04,
            "threshold": 0.49,
            "days_used": 7,
        }
        msg = format_alert_message(self._result(cls_alert=cls_alert))
        assert "Cognitive load spike" in msg
        assert "0.92" in msg
        assert "0.41" in msg

    def test_fdi_collapse_message_contains_key_info(self):
        fdi_alert = {
            "today_fdi": 0.35,
            "baseline_fdi": 0.72,
            "drop_pct": 0.514,
            "days_used": 7,
        }
        msg = format_alert_message(self._result(fdi_alert=fdi_alert))
        assert "Focus quality collapsed" in msg
        assert "0.35" in msg
        assert "0.72" in msg

    def test_recovery_streak_message_contains_key_info(self):
        streak_alert = {
            "streak_days": 3,
            "ras_values": [0.35, 0.38, 0.40],
            "avg_ras": 0.376,
        }
        msg = format_alert_message(self._result(streak_alert=streak_alert))
        assert "Recovery misalignment streak" in msg
        assert "3" in msg
        assert "0.38" in msg

    def test_combined_message_contains_all_sections(self):
        cls_alert = {"today_cls": 0.9, "baseline_mean": 0.4, "baseline_std": 0.0,
                     "threshold": 0.4, "days_used": 7}
        fdi_alert = {"today_fdi": 0.3, "baseline_fdi": 0.7, "drop_pct": 0.57, "days_used": 7}
        streak_alert = {"streak_days": 4, "ras_values": [0.3]*4, "avg_ras": 0.3}
        msg = format_alert_message(self._result(cls_alert, fdi_alert, streak_alert))
        assert "Cognitive load spike" in msg
        assert "Focus quality collapsed" in msg
        assert "Recovery misalignment streak" in msg

    def test_message_starts_with_header(self):
        cls_alert = {"today_cls": 0.9, "baseline_mean": 0.4, "baseline_std": 0.0,
                     "threshold": 0.4, "days_used": 7}
        msg = format_alert_message(self._result(cls_alert=cls_alert))
        assert msg.startswith("🧠")

    def test_date_appears_in_header(self):
        cls_alert = {"today_cls": 0.9, "baseline_mean": 0.4, "baseline_std": 0.0,
                     "threshold": 0.4, "days_used": 7}
        msg = format_alert_message(self._result(cls_alert=cls_alert, date="2026-03-15"))
        assert "2026-03-15" in msg

    def test_message_is_string(self):
        msg = format_alert_message(self._result())
        assert isinstance(msg, str)


# ─── Edge cases & integration ─────────────────────────────────────────────────

class TestEdgeCases:
    def test_check_anomalies_with_empty_store(self):
        """Should not raise even with zero historical data."""
        with patch("analysis.anomaly_alerts.list_available_dates", return_value=[]), \
             patch("analysis.anomaly_alerts.read_day", return_value=[]), \
             patch("analysis.anomaly_alerts.read_summary", return_value={"days": {}}):
            result = check_anomalies("2026-03-14")
        assert result["any_triggered"] is False

    def test_format_message_with_empty_alerts_dict(self):
        """format_alert_message() handles malformed input gracefully."""
        result = {"date": "2026-03-14", "alerts": {}, "any_triggered": False}
        assert format_alert_message(result) == ""

    def test_std_large_spread(self):
        """_std works with a realistic spread of CLS values."""
        vals = [0.10, 0.20, 0.30, 0.80, 0.90]
        result = _std(vals)
        assert result > 0
        assert result < 1.0

    def test_cls_spike_uses_correct_multiplier(self):
        """Spike threshold = mean + CLS_SPIKE_STD_MULTIPLIER × std."""
        # std=0.1, mean=0.4 → threshold = 0.4 + 2*0.1 = 0.6
        # today=0.61 should trigger; today=0.59 should not
        baseline_vals = [0.30, 0.50, 0.30, 0.50, 0.30, 0.50, 0.30]  # std > 0
        baseline_dates = ["2026-03-13", "2026-03-12", "2026-03-11",
                          "2026-03-10", "2026-03-09", "2026-03-08", "2026-03-07"]
        mean_b = sum(baseline_vals) / len(baseline_vals)
        std_b = _std(baseline_vals)
        threshold = mean_b + CLS_SPIKE_STD_MULTIPLIER * std_b
        # today slightly above threshold
        today_above = threshold + 0.01
        today_below = threshold - 0.01

        def make_mock(today_val):
            dates_map = dict(zip(baseline_dates, baseline_vals))
            def side_effect(d):
                if d == "2026-03-14":
                    return today_val
                return dates_map.get(d)
            return side_effect

        with patch("analysis.anomaly_alerts._baseline_dates", return_value=baseline_dates), \
             patch("analysis.anomaly_alerts._get_daily_cls", side_effect=make_mock(today_above)):
            result_above = detect_cls_spike("2026-03-14")

        with patch("analysis.anomaly_alerts._baseline_dates", return_value=baseline_dates), \
             patch("analysis.anomaly_alerts._get_daily_cls", side_effect=make_mock(today_below)):
            result_below = detect_cls_spike("2026-03-14")

        assert result_above is not None
        assert result_below is None

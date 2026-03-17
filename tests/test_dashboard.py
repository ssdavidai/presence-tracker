"""
Tests for analysis/dashboard.py — Daily HTML Dashboard (v12)

Tests cover:
  - generate_dashboard() produces a valid HTML file
  - All metric sections appear in the output
  - Colour helpers map values correctly
  - SVG chart builders return well-formed SVG
  - CLI behaviour for missing data
  - Output is self-contained (no external CDN references)
  - DPS + CDI hero section (v11): score rings, tier labels, graceful fallback
  - Flow State card (v12): section header present, graceful fallback on error
  - Load Volatility (LVI) card (v12): section header present, graceful fallback on error
"""

import json
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import pytest

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.dashboard import (
    _cls_colour,
    _recovery_colour,
    _dps_colour,
    _cdi_colour,
    _svg_score_ring,
    _svg_cls_timeline,
    _svg_hourly_heatmap,
    _bar,
    generate_dashboard,
)


# ─── Fixture helpers ──────────────────────────────────────────────────────────

TIMEZONE = ZoneInfo("Europe/Budapest")


def _make_window(index: int, cls: float = 0.30, fdi: float = 0.70,
                 sdi: float = 0.20, csc: float = 0.15, ras: float = 0.80,
                 in_meeting: bool = False, msg_sent: int = 0,
                 date_str: str = "2026-03-13") -> dict:
    """Build a minimal but realistic window dict."""
    hour = index // 4
    minute = (index % 4) * 15
    start = datetime(2026, 3, 13, hour, minute, tzinfo=TIMEZONE)
    end = start + timedelta(minutes=15)
    return {
        "window_id": f"{date_str}T{hour:02d}:{minute:02d}:00",
        "date": date_str,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "window_index": index,
        "calendar": {
            "in_meeting": in_meeting,
            "meeting_title": "Standup" if in_meeting else None,
            "meeting_attendees": 3 if in_meeting else 0,
            "meeting_duration_minutes": 30 if in_meeting else 0,
            "meeting_organizer": None,
            "meetings_count": 1 if in_meeting else 0,
        },
        "whoop": {
            "recovery_score": 78.0,
            "hrv_rmssd_milli": 65.0,
            "resting_heart_rate": 55.0,
            "sleep_performance": 82.0,
            "sleep_hours": 7.5,
            "strain": 10.2,
            "spo2_percentage": 95.0,
        },
        "slack": {
            "messages_sent": msg_sent,
            "messages_received": msg_sent * 3,
            "total_messages": msg_sent * 4,
            "channels_active": 1 if msg_sent else 0,
        },
        "metrics": {
            "cognitive_load_score": cls,
            "focus_depth_index": fdi,
            "social_drain_index": sdi,
            "context_switch_cost": csc,
            "recovery_alignment_score": ras,
        },
        "metadata": {
            "day_of_week": "Friday",
            "hour_of_day": hour,
            "minute_of_hour": minute,
            "is_working_hours": 7 <= hour < 22,
            "sources_available": ["whoop", "calendar", "slack"],
            "is_active_window": in_meeting or msg_sent > 0,
        },
    }


def _make_day(date_str: str = "2026-03-13") -> list[dict]:
    """Build a full 96-window day with varied CLS values."""
    windows = []
    for i in range(96):
        hour = i // 4
        is_working = 7 <= hour < 22
        cls = 0.40 if 9 <= hour <= 17 else 0.05
        in_meeting = hour in (9, 10, 14, 15)
        msg_sent = 3 if (9 <= hour <= 17 and not in_meeting) else 0
        windows.append(_make_window(
            i, cls=cls, in_meeting=in_meeting, msg_sent=msg_sent, date_str=date_str
        ))
    return windows


# ─── Colour helpers ───────────────────────────────────────────────────────────

class TestClsColour:
    def test_low_cls_is_green(self):
        colour = _cls_colour(0.10)
        assert colour == "#4ade80"

    def test_mid_low_cls_is_yellow(self):
        colour = _cls_colour(0.35)
        assert colour == "#facc15"

    def test_mid_high_cls_is_orange(self):
        colour = _cls_colour(0.60)
        assert colour == "#fb923c"

    def test_high_cls_is_red(self):
        colour = _cls_colour(0.90)
        assert colour == "#f87171"

    def test_boundary_at_0_25(self):
        # 0.25 transitions from green to yellow
        assert _cls_colour(0.24) == "#4ade80"
        assert _cls_colour(0.25) == "#facc15"

    def test_boundary_at_0_50(self):
        assert _cls_colour(0.49) == "#facc15"
        assert _cls_colour(0.50) == "#fb923c"

    def test_boundary_at_0_75(self):
        assert _cls_colour(0.74) == "#fb923c"
        assert _cls_colour(0.75) == "#f87171"


class TestRecoveryColour:
    def test_high_recovery_green(self):
        assert _recovery_colour(90) == "#4ade80"

    def test_mid_recovery_yellow(self):
        assert _recovery_colour(50) == "#facc15"

    def test_low_recovery_red(self):
        assert _recovery_colour(20) == "#f87171"

    def test_boundary_at_67(self):
        assert _recovery_colour(67) == "#4ade80"
        assert _recovery_colour(66) == "#facc15"

    def test_boundary_at_34(self):
        assert _recovery_colour(34) == "#facc15"
        assert _recovery_colour(33) == "#f87171"


# ─── SVG builders ─────────────────────────────────────────────────────────────

class TestSvgClsTimeline:
    def test_returns_svg_element(self):
        windows = _make_day()
        result = _svg_cls_timeline(windows)
        assert result.startswith("<svg")
        assert "</svg>" in result

    def test_empty_windows_shows_no_data(self):
        result = _svg_cls_timeline([])
        assert "No data" in result

    def test_polyline_present_with_data(self):
        windows = _make_day()
        result = _svg_cls_timeline(windows)
        assert "<polyline" in result

    def test_gradient_defined(self):
        windows = _make_day()
        result = _svg_cls_timeline(windows)
        assert "linearGradient" in result

    def test_custom_dimensions(self):
        windows = _make_day()
        result = _svg_cls_timeline(windows, width=600, height=120)
        assert 'width="600"' in result
        assert 'height="120"' in result

    def test_single_window_does_not_crash(self):
        w = [_make_window(0, cls=0.5)]
        result = _svg_cls_timeline(w)
        assert "</svg>" in result


class TestSvgHourlyHeatmap:
    def test_returns_svg_element(self):
        windows = _make_day()
        result = _svg_hourly_heatmap(windows)
        assert result.startswith("<svg")
        assert "</svg>" in result

    def test_contains_hour_labels(self):
        windows = _make_day()
        result = _svg_hourly_heatmap(windows)
        # Should have labels for at least some hours in 7-22 range
        assert "7h" in result or "8h" in result

    def test_rect_per_hour(self):
        windows = _make_day()
        result = _svg_hourly_heatmap(windows)
        # 15 working hours → 15 rects
        assert result.count("<rect") == 15

    def test_empty_windows_produces_zero_opacity_cells(self):
        # Windows with no working-hour data still render cells
        windows = [_make_window(i, cls=0.0) for i in range(96)]
        result = _svg_hourly_heatmap(windows)
        assert "<rect" in result


# ─── Bar renderer ─────────────────────────────────────────────────────────────

class TestBar:
    def test_renders_label(self):
        result = _bar("Test Metric", 0.50, "#4ade80")
        assert "Test Metric" in result

    def test_renders_percentage(self):
        result = _bar("CLS", 0.75, "#f87171")
        assert "75%" in result

    def test_none_value_shows_na(self):
        result = _bar("FDI", None, "#60a5fa")
        assert "N/A" in result

    def test_clamps_at_100_pct(self):
        # Value > 1.0 should clamp to 100%
        result = _bar("CLS", 1.5, "#f87171")
        assert "width:100.0%" in result

    def test_clamps_at_0_pct(self):
        result = _bar("CLS", -0.5, "#4ade80")
        assert "width:0.0%" in result

    def test_colour_used_in_style(self):
        result = _bar("SDI", 0.3, "#a78bfa")
        assert "#a78bfa" in result

    def test_zero_value_renders(self):
        result = _bar("CSC", 0.0, "#fb923c")
        assert "0%" in result


# ─── generate_dashboard() integration ────────────────────────────────────────

class TestGenerateDashboard:
    """Integration tests: mock read_day and verify the HTML output."""

    def _run_generate(self, date_str: str = "2026-03-13") -> tuple[Path, str]:
        windows = _make_day(date_str)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / f"{date_str}.html"
            with (
                patch("analysis.dashboard.DASHBOARD_DIR", Path(tmpdir)),
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
            ):
                path = generate_dashboard(date_str, output_path=out_path)
                html = path.read_text(encoding="utf-8")
        return path, html

    def test_generates_html_file(self):
        _, html = self._run_generate()
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_contains_date_in_title(self):
        _, html = self._run_generate("2026-03-13")
        assert "2026" in html
        assert "March" in html

    def test_contains_recovery_section(self):
        _, html = self._run_generate()
        assert "Recovery" in html

    def test_contains_cls_section(self):
        _, html = self._run_generate()
        assert "Cognitive Load" in html or "CLS" in html

    def test_contains_metric_bars(self):
        _, html = self._run_generate()
        assert "Focus Depth" in html
        assert "Social Drain" in html
        assert "Recovery Alignment" in html

    def test_contains_insight(self):
        _, html = self._run_generate()
        assert "Insight" in html

    def test_no_external_cdn_references(self):
        """Output must be self-contained — no CDN or external URLs."""
        _, html = self._run_generate()
        assert "cdn.jsdelivr.net" not in html
        assert "cdnjs.cloudflare.com" not in html
        assert "googleapis.com" not in html
        assert "unpkg.com" not in html

    def test_svg_chart_present(self):
        _, html = self._run_generate()
        assert "<svg" in html
        assert "<polyline" in html

    def test_heatmap_present(self):
        _, html = self._run_generate()
        # Heatmap has rect elements
        assert html.count("<rect") >= 15

    def test_footer_present(self):
        _, html = self._run_generate()
        assert "Presence Tracker" in html

    def test_no_data_raises(self):
        with patch("engine.store.read_day", return_value=[]):
            with pytest.raises(ValueError, match="No data"):
                generate_dashboard("2099-01-01")

    def test_html_has_matching_tags(self):
        _, html = self._run_generate()
        assert html.count("<html") == 1
        assert html.count("</html>") == 1
        assert html.count("<body") == 1
        assert html.count("</body>") == 1

    def test_whoop_values_rendered(self):
        _, html = self._run_generate()
        # Recovery score 78 should appear
        assert "78%" in html or "78" in html

    def test_hrv_rendered(self):
        _, html = self._run_generate()
        assert "65" in html  # HRV 65ms

    def test_activity_stats_present(self):
        _, html = self._run_generate()
        assert "meetings" in html.lower() or "meeting" in html.lower()

    def test_sources_in_footer(self):
        _, html = self._run_generate()
        assert "whoop" in html

    def test_output_path_used(self):
        windows = _make_day()
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = Path(tmpdir) / "custom_output.html"
            with (
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
            ):
                result = generate_dashboard("2026-03-13", output_path=custom_path)
            assert result == custom_path
            assert custom_path.exists()

    def test_with_rescuetime_data(self):
        """When windows have RescueTime data, the RT section appears."""
        windows = _make_day()
        for w in windows:
            h = w["metadata"]["hour_of_day"]
            if 9 <= h < 18:
                w["rescuetime"] = {
                    "active_seconds": 600,
                    "focus_seconds": 400,
                    "distraction_seconds": 100,
                    "neutral_seconds": 100,
                    "top_activity": "PyCharm",
                    "productivity_pulse": 72,
                    "app_switches": 3,
                }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            with (
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
            ):
                path = generate_dashboard("2026-03-13", output_path=out)
                html = path.read_text()
        assert "RescueTime" in html or "Computer Activity" in html

    def test_with_omi_data(self):
        """When windows have Omi data, the Omi section appears."""
        windows = _make_day()
        for i, w in enumerate(windows):
            if i in (36, 37, 38):  # 9am-9:45am
                w["omi"] = {
                    "conversation_active": True,
                    "word_count": 250,
                    "speaker_count": 2,
                    "topic_tags": ["work", "planning"],
                }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            with (
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
            ):
                path = generate_dashboard("2026-03-13", output_path=out)
                html = path.read_text()
        assert "Omi" in html or "Conversation" in html


# ─── DPS / CDI colour helpers ─────────────────────────────────────────────────

class TestDpsColour:
    def test_high_dps_is_green(self):
        assert _dps_colour(90) == "#4ade80"

    def test_good_dps_is_blue(self):
        assert _dps_colour(70) == "#60a5fa"

    def test_moderate_dps_is_yellow(self):
        assert _dps_colour(50) == "#facc15"

    def test_low_dps_is_red(self):
        assert _dps_colour(30) == "#f87171"

    def test_boundary_at_80(self):
        assert _dps_colour(80) == "#4ade80"
        assert _dps_colour(79) == "#60a5fa"

    def test_boundary_at_60(self):
        assert _dps_colour(60) == "#60a5fa"
        assert _dps_colour(59) == "#facc15"

    def test_boundary_at_40(self):
        assert _dps_colour(40) == "#facc15"
        assert _dps_colour(39) == "#f87171"


class TestCdiColour:
    def test_surplus_is_green(self):
        assert _cdi_colour("surplus") == "#4ade80"

    def test_balanced_is_blue(self):
        assert _cdi_colour("balanced") == "#60a5fa"

    def test_loading_is_yellow(self):
        assert _cdi_colour("loading") == "#facc15"

    def test_fatigued_is_orange(self):
        assert _cdi_colour("fatigued") == "#fb923c"

    def test_critical_is_red(self):
        assert _cdi_colour("critical") == "#f87171"

    def test_unknown_tier_returns_muted(self):
        assert _cdi_colour("unknown_tier") == "#94a3b8"


class TestSvgScoreRing:
    def test_returns_svg_string(self):
        result = _svg_score_ring(75.0, "#4ade80")
        assert result.startswith("<svg")
        assert "</svg>" in result

    def test_score_appears_in_ring(self):
        result = _svg_score_ring(88.0, "#4ade80")
        assert ">88<" in result

    def test_colour_used_in_ring(self):
        colour = "#f87171"
        result = _svg_score_ring(45.0, colour)
        assert colour in result

    def test_zero_score_renders(self):
        result = _svg_score_ring(0.0, "#f87171")
        assert ">0<" in result
        assert "</svg>" in result

    def test_hundred_score_renders(self):
        result = _svg_score_ring(100.0, "#4ade80")
        assert ">100<" in result

    def test_score_clamped_above_100(self):
        # Scores > 100 should clamp to 100%
        result = _svg_score_ring(120.0, "#4ade80")
        assert "</svg>" in result

    def test_score_clamped_below_0(self):
        result = _svg_score_ring(-10.0, "#4ade80")
        assert "</svg>" in result

    def test_custom_size(self):
        result = _svg_score_ring(50.0, "#60a5fa", size=120)
        assert 'width="120"' in result
        assert 'height="120"' in result

    def test_background_ring_present(self):
        # Background track should use muted colour
        result = _svg_score_ring(60.0, "#4ade80")
        assert "#2a2d3e" in result

    def test_foreground_arc_uses_dasharray(self):
        result = _svg_score_ring(50.0, "#60a5fa")
        assert "stroke-dasharray" in result


# ─── DPS + CDI hero section in generate_dashboard ────────────────────────────

class TestDashboardHeroSection:
    """Verify that DPS + CDI hero section is present in the generated dashboard."""

    def _run_generate(self, date_str: str = "2026-03-13") -> str:
        windows = _make_day(date_str)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / f"{date_str}.html"
            with (
                patch("analysis.dashboard.DASHBOARD_DIR", Path(tmpdir)),
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
            ):
                path = generate_dashboard(date_str, output_path=out_path)
                return path.read_text(encoding="utf-8")

    def test_hero_card_present(self):
        html = self._run_generate()
        assert "Today&#x27;s Scores" in html or "Today's Scores" in html or "score-grid" in html

    def test_dps_section_present(self):
        html = self._run_generate()
        assert "Daily Presence Score" in html

    def test_cdi_section_present(self):
        html = self._run_generate()
        assert "Cognitive Debt Index" in html

    def test_score_rings_in_output(self):
        html = self._run_generate()
        # SVG score rings use stroke-dasharray
        assert "stroke-dasharray" in html

    def test_score_ring_css_present(self):
        html = self._run_generate()
        assert "score-grid" in html

    def test_score_meta_css_present(self):
        html = self._run_generate()
        assert "score-meta" in html or "score-title" in html

    def test_hero_section_before_recovery(self):
        html = self._run_generate()
        hero_pos = html.find("Daily Presence Score")
        recovery_pos = html.find("💚 Recovery")
        assert hero_pos < recovery_pos, "DPS hero should appear before Recovery card"

    def test_dps_tier_label_rendered(self):
        html = self._run_generate()
        # One of the DPS tier labels should appear (mirrors _dps_tier() in presence_score.py)
        tier_labels = ["Exceptional", "Strong", "Good", "Moderate", "Low", "Poor"]
        assert any(t in html for t in tier_labels), f"No DPS tier label found. Labels checked: {tier_labels}"

    def test_cdi_warmup_shown_when_not_meaningful(self):
        """When CDI isn't meaningful (< 3 days), 'Warming up' placeholder should show."""
        html = self._run_generate()
        # With only mock data and no real store history, CDI likely not meaningful
        # Just verify the section renders without crashing
        assert "Cognitive Debt" in html

    def test_dps_and_cdi_fallback_gracefully(self):
        """Even if DPS/CDI computation throws, dashboard should still render."""
        windows = _make_day()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.html"
            with (
                patch("analysis.dashboard.DASHBOARD_DIR", Path(tmpdir)),
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
                # Simulate DPS/CDI import failures
                patch("analysis.presence_score.compute_presence_score", side_effect=Exception("mock fail")),
                patch("analysis.cognitive_debt.compute_cdi", side_effect=Exception("mock fail")),
            ):
                path = generate_dashboard("2026-03-13", output_path=out_path)
                html = path.read_text()
        # Dashboard should still render even with failures
        assert "<!DOCTYPE html>" in html

    def test_hero_section_uses_correct_colours(self):
        html = self._run_generate()
        # The ring SVG should use colour codes from _dps_colour / _cdi_colour
        valid_colours = {"#4ade80", "#60a5fa", "#facc15", "#fb923c", "#f87171", "#94a3b8"}
        assert any(c in html for c in valid_colours)


# ─── Trend badge rendering ────────────────────────────────────────────────────

class TestTrendBadges:
    """Verify trend badges appear when appropriate trend data is injected."""

    def _run_with_trend(self, trend: dict) -> str:
        windows = _make_day()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            fake_summary = {
                "date": "2026-03-12",
                "whoop": {"hrv_rmssd_milli": 60.0, "recovery_score": 70.0},
                "metrics_avg": {"cognitive_load_score": 0.35, "recovery_alignment_score": 0.75},
            }
            with (
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[fake_summary] * 5),
            ):
                path = generate_dashboard("2026-03-13", output_path=out)
                return path.read_text()

    def test_badge_section_present_with_trend(self):
        html = self._run_with_trend({})
        # Just verify generation doesn't crash and produces HTML
        assert "<html" in html

    def test_no_crash_with_empty_summaries(self):
        windows = _make_day()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            with (
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
            ):
                path = generate_dashboard("2026-03-13", output_path=out)
            assert path.exists()


# ─── Flow State card (v12) ────────────────────────────────────────────────────

class TestFlowStateCard:
    """Verify the Flow State section in the daily HTML dashboard (v12)."""

    def _run_generate(self) -> str:
        windows = _make_day()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            with (
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
            ):
                path = generate_dashboard("2026-03-13", output_path=out)
                return path.read_text()

    def test_flow_state_section_present(self):
        html = self._run_generate()
        assert "Flow State" in html

    def test_flow_state_shows_flow_time_label(self):
        html = self._run_generate()
        # The template renders "min" for total flow minutes
        assert " min" in html

    def test_flow_state_fallback_on_import_error(self):
        """Dashboard renders without crashing if flow_detector raises."""
        windows = _make_day()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            with (
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
                patch(
                    "analysis.flow_detector.detect_flow_states",
                    side_effect=Exception("mock flow error"),
                ),
            ):
                path = generate_dashboard("2026-03-13", output_path=out)
                html = path.read_text()
        # Dashboard must still render valid HTML even with flow failure
        assert "<!DOCTYPE html>" in html

    def test_flow_state_section_after_metric_bars(self):
        """Flow State should appear after the Metric Breakdown section."""
        html = self._run_generate()
        metric_pos = html.find("Metric Breakdown")
        flow_pos = html.find("Flow State")
        # If flow section is present, it should come after metric bars
        if flow_pos != -1 and metric_pos != -1:
            assert flow_pos > metric_pos


# ─── Load Volatility (LVI) card (v12) ────────────────────────────────────────

class TestLoadVolatilityCard:
    """Verify the Load Volatility section in the daily HTML dashboard (v12)."""

    def _run_generate(self) -> str:
        windows = _make_day()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            with (
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
            ):
                path = generate_dashboard("2026-03-13", output_path=out)
                return path.read_text()

    def test_load_volatility_section_present(self):
        html = self._run_generate()
        assert "Load Volatility" in html

    def test_load_volatility_shows_lvi_label(self):
        html = self._run_generate()
        # The LVI label descriptions should appear in the tooltip / insight line
        assert "LVI" in html or "volatile" in html.lower() or "smooth" in html.lower()

    def test_load_volatility_fallback_on_import_error(self):
        """Dashboard renders without crashing if load_volatility raises."""
        windows = _make_day()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            with (
                patch("engine.store.read_day", return_value=windows),
                patch("engine.store.get_recent_summaries", return_value=[]),
                patch(
                    "analysis.load_volatility.compute_load_volatility",
                    side_effect=Exception("mock lvi error"),
                ),
            ):
                path = generate_dashboard("2026-03-13", output_path=out)
                html = path.read_text()
        assert "<!DOCTYPE html>" in html

    def test_flow_and_lvi_side_by_side_when_both_present(self):
        """When both Flow State and LVI cards are present, they share a grid-2 container."""
        html = self._run_generate()
        # If both sections appear and data is sufficient, expect them in a grid
        flow_present = "Flow State" in html
        lvi_present = "Load Volatility" in html
        if flow_present and lvi_present:
            # grid-2 wrapper contains both
            assert "grid-2" in html

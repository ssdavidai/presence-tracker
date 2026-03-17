"""
Presence Tracker — Daily HTML Dashboard (v12)

Generates a self-contained HTML file for a given day:
  data/dashboard/YYYY-MM-DD.html

Charts (pure SVG — no CDN, no external dependencies):
  - CLS timeline: per-window cognitive load across the full day
  - Metric bars: FDI, SDI, CSC, RAS as horizontal progress bars
  - Hourly heatmap: intensity grid over working hours
  - WHOOP recovery panel: recovery %, HRV, sleep

v12 — Flow State + Load Volatility (LVI) cards:
  Two new analytical sections surfaced directly on the dashboard:

  Flow State (from analysis/flow_detector.py):
    Shows today's deep-work sessions detected from the FDI/CLS/CSC signals.
    Displays total flow minutes, best session time block, and the flow label
    (deep_flow / in_zone / brief / none). Lets David see at a glance whether
    he achieved any sustained cognitive immersion today, and when it happened.

  Load Volatility Index / LVI (from analysis/load_volatility.py):
    Shows whether today's cognitive load was smooth and predictable, or
    volatile and spiky. An LVI of 1.0 = perfectly smooth; 0.0 = chaotic
    whiplash between low and high load. Includes a visual LVI bar, the
    label (smooth / steady / variable / volatile), and the key stats
    (CLS std, CLS range). A day with LVI = 0.8 is more sustainable than
    a day with the same *average* CLS but LVI = 0.3.

  These two modules were shipped in v32 and v35 respectively but were not
  previously wired into the HTML dashboard. This release closes that gap,
  giving David a fuller picture in the daily report.

v11 — DPS + CDI hero section:
  The dashboard now leads with the two most important composite scores:
    - DPS (Daily Presence Score, 0-100): "How was my cognitive day overall?"
      Weighted blend of all 5 metrics. The single number that answers whether
      today was a peak day or a difficult one. Shown with tier label and
      colour-coded score ring.
    - CDI (Cognitive Debt Index, 0-100): "How much accumulated fatigue am I carrying?"
      14-day rolling fatigue accumulation. Higher = more debt. Shown with
      tier label (surplus / balanced / loading / fatigued / critical).
  These appear as a hero "Today's Score" card immediately after the header —
  before the recovery panel — because they are the most actionable summary.
  Both scores are computed lazily; if data is insufficient they degrade
  gracefully with a neutral placeholder rather than crashing.

The output is a single HTML file that opens directly in any browser.
It embeds all CSS and JS inline — suitable for archiving or attaching to Slack.

Usage (CLI):
    python3 analysis/dashboard.py [YYYY-MM-DD] [--output PATH]

Usage (programmatic):
    from analysis.dashboard import generate_dashboard
    path = generate_dashboard("2026-03-13")
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR

DASHBOARD_DIR = DATA_DIR / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


# ─── Colour helpers ───────────────────────────────────────────────────────────

def _cls_colour(value: float) -> str:
    """Map CLS 0-1 to a hex colour: green → amber → red."""
    if value < 0.25:
        return "#4ade80"   # green-400
    elif value < 0.50:
        return "#facc15"   # yellow-400
    elif value < 0.75:
        return "#fb923c"   # orange-400
    else:
        return "#f87171"   # red-400


def _recovery_colour(pct: float) -> str:
    """WHOOP-style recovery colour."""
    if pct >= 67:
        return "#4ade80"   # green
    elif pct >= 34:
        return "#facc15"   # yellow
    else:
        return "#f87171"   # red


def _dps_colour(score: float) -> str:
    """Map DPS 0-100 to a hex colour: red → amber → green."""
    if score >= 80:
        return "#4ade80"   # green — exceptional / strong
    elif score >= 60:
        return "#60a5fa"   # blue — moderate
    elif score >= 40:
        return "#facc15"   # yellow — light
    else:
        return "#f87171"   # red — difficult


def _cdi_colour(tier: str) -> str:
    """Map CDI tier to a hex colour."""
    return {
        "surplus": "#4ade80",    # green — fully recovered
        "balanced": "#60a5fa",   # blue — sustainable
        "loading": "#facc15",    # yellow — accumulating
        "fatigued": "#fb923c",   # orange — high debt
        "critical": "#f87171",   # red — burnout risk
    }.get(tier, "#94a3b8")


def _svg_score_ring(score: float, colour: str, size: int = 90) -> str:
    """
    Build a circular SVG progress ring for a 0-100 score.

    Args:
        score: 0–100 value
        colour: hex colour for the ring arc
        size: outer diameter in px (default 90)

    Returns:
        SVG element string (inline, no viewBox attribute — sized by width/height).
    """
    radius = (size - 10) // 2   # ring inside the SVG with 5px padding each side
    cx = cy = size // 2
    circumference = 2 * math.pi * radius
    # Clamp score to [0, 100]
    pct = max(0.0, min(100.0, score)) / 100.0
    filled = circumference * pct
    gap = circumference - filled

    return (
        f'<svg width="{size}" height="{size}" style="display:block;flex-shrink:0">'
        # Background ring
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#2a2d3e" stroke-width="6"/>'
        # Foreground arc — starts at top (rotate -90°)
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none"'
        f' stroke="{colour}" stroke-width="6"'
        f' stroke-dasharray="{filled:.1f} {gap:.1f}"'
        f' stroke-linecap="round"'
        f' transform="rotate(-90 {cx} {cy})"/>'
        # Score label inside ring
        f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central"'
        f' font-size="18" font-weight="700" fill="{colour}">{score:.0f}</text>'
        f'</svg>'
    )


# ─── SVG chart builders ───────────────────────────────────────────────────────

def _svg_cls_timeline(windows: list[dict], width: int = 800, height: int = 160) -> str:
    """
    Build an SVG line + area chart of CLS across all 96 windows.
    Colours the area under the curve by load intensity.
    """
    pad_left = 40
    pad_right = 10
    pad_top = 10
    pad_bottom = 30

    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    if not windows:
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#888">No data</text></svg>'

    # Build (x, y, cls) points for all 96 windows
    n = len(windows)
    points = []
    for i, w in enumerate(windows):
        cls = w["metrics"]["cognitive_load_score"]
        x = pad_left + (i / (n - 1)) * chart_w if n > 1 else pad_left + chart_w / 2
        y = pad_top + (1.0 - cls) * chart_h
        points.append((x, y, cls))

    # Area path (filled, multi-coloured using individual trapezoids)
    # We'll use a single polyline + gradient fill for simplicity
    polyline_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
    # Close area: bottom-right → bottom-left
    area_pts = (
        polyline_pts
        + f" {points[-1][0]:.1f},{pad_top + chart_h:.1f}"
        + f" {points[0][0]:.1f},{pad_top + chart_h:.1f}"
    )

    # Y-axis labels
    y_labels = []
    for val, label in [(0.0, "0%"), (0.25, "25%"), (0.50, "50%"), (0.75, "75%"), (1.0, "100%")]:
        yy = pad_top + (1.0 - val) * chart_h
        y_labels.append(
            f'<line x1="{pad_left - 4}" y1="{yy:.1f}" x2="{pad_left + chart_w}" y2="{yy:.1f}" '
            f'stroke="#333" stroke-width="0.5" stroke-dasharray="2,4"/>'
            f'<text x="{pad_left - 6}" y="{yy + 4:.1f}" text-anchor="end" '
            f'font-size="9" fill="#888">{label}</text>'
        )

    # X-axis hour labels (every 2 hours)
    x_labels = []
    for hour in range(0, 25, 2):
        window_i = hour * 4  # 4 windows/hour
        if window_i >= n:
            window_i = n - 1
        x_pos = pad_left + (window_i / (n - 1)) * chart_w if n > 1 else pad_left
        x_labels.append(
            f'<text x="{x_pos:.1f}" y="{pad_top + chart_h + 18:.1f}" '
            f'text-anchor="middle" font-size="9" fill="#888">{hour:02d}h</text>'
        )

    # Working hours shading (7am–22pm)
    wh_start_i = 7 * 4   # window 28
    wh_end_i = 22 * 4    # window 88
    wh_x1 = pad_left + (wh_start_i / (n - 1)) * chart_w if n > 1 else pad_left
    wh_x2 = pad_left + (wh_end_i / (n - 1)) * chart_w if n > 1 else pad_left
    working_shade = (
        f'<rect x="{wh_x1:.1f}" y="{pad_top}" '
        f'width="{wh_x2 - wh_x1:.1f}" height="{chart_h}" '
        f'fill="#ffffff08" rx="2"/>'
    )

    svg = f"""<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="clsGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f87171" stop-opacity="0.6"/>
      <stop offset="60%" stop-color="#facc15" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#4ade80" stop-opacity="0.05"/>
    </linearGradient>
  </defs>
  {working_shade}
  {"".join(y_labels)}
  {"".join(x_labels)}
  <polygon points="{area_pts}" fill="url(#clsGrad)" stroke="none"/>
  <polyline points="{polyline_pts}" fill="none" stroke="#f87171" stroke-width="1.5" stroke-linejoin="round"/>
</svg>"""
    return svg


def _svg_hourly_heatmap(windows: list[dict], width: int = 800, height: int = 60) -> str:
    """
    Build an SVG heatmap of CLS per hour (7am–22:00), each cell coloured by intensity.
    Shows 15 hours × 4 cells = 60 cells.
    """
    START_HOUR = 7
    END_HOUR = 22
    n_hours = END_HOUR - START_HOUR  # 15

    pad_left = 40
    pad_right = 10
    cell_w = (width - pad_left - pad_right) / n_hours
    cell_h = height - 22  # leave room for labels

    # Group windows by hour
    hourly: dict[int, list[float]] = {}
    for w in windows:
        h = w["metadata"]["hour_of_day"]
        if START_HOUR <= h < END_HOUR:
            hourly.setdefault(h, []).append(w["metrics"]["cognitive_load_score"])

    cells = []
    for i, hour in enumerate(range(START_HOUR, END_HOUR)):
        vals = hourly.get(hour, [])
        avg = sum(vals) / len(vals) if vals else 0.0
        x = pad_left + i * cell_w
        colour = _cls_colour(avg)
        opacity = 0.2 + avg * 0.8  # at least faintly visible
        cells.append(
            f'<rect x="{x:.1f}" y="2" width="{cell_w - 1:.1f}" height="{cell_h:.1f}" '
            f'rx="3" fill="{colour}" opacity="{opacity:.2f}"/>'
            f'<title>{hour:02d}:00 — CLS {avg:.0%}</title>'
        )
        cells.append(
            f'<text x="{x + cell_w / 2:.1f}" y="{cell_h + 16:.1f}" '
            f'text-anchor="middle" font-size="9" fill="#888">{hour}h</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  {"".join(cells)}
</svg>"""


def _bar(label: str, value: Optional[float], colour: str, width: int = 680) -> str:
    """Render a single labelled horizontal progress bar as HTML."""
    if value is None:
        val_str = "N/A"
        pct = 0.0
    else:
        val_str = f"{value:.0%}"
        pct = max(0.0, min(1.0, value)) * 100

    return f"""
<div class="metric-row">
  <span class="metric-label">{label}</span>
  <div class="bar-track">
    <div class="bar-fill" style="width:{pct:.1f}%;background:{colour}"></div>
  </div>
  <span class="metric-value">{val_str}</span>
</div>"""


# ─── Main HTML generator ──────────────────────────────────────────────────────

def generate_dashboard(date_str: str, output_path: Optional[Path] = None) -> Path:
    """
    Generate a self-contained HTML dashboard for the given date.

    Args:
        date_str: "YYYY-MM-DD"
        output_path: Override output path (default: data/dashboard/YYYY-MM-DD.html)

    Returns:
        Path to the generated HTML file.
    """
    from engine.store import read_day
    from analysis.daily_digest import compute_digest, _format_hourly_sparkline

    windows = read_day(date_str)
    if not windows:
        raise ValueError(f"No data for {date_str}")

    digest = compute_digest(windows)

    # ── Unpack data ──
    m = digest.get("metrics", {})
    w_data = digest.get("whoop", {})
    act = digest.get("activity", {})
    trend = digest.get("trend", {})
    insight = digest.get("insight", "")
    rescuetime = digest.get("rescuetime")

    avg_cls = m.get("avg_cls")
    peak_cls = m.get("peak_cls")
    avg_fdi = m.get("avg_fdi_active")
    avg_sdi = m.get("avg_sdi_active")
    avg_csc = m.get("avg_csc_active")
    avg_ras = m.get("avg_ras")

    recovery = w_data.get("recovery_score")
    hrv = w_data.get("hrv_rmssd_milli")
    sleep_h = w_data.get("sleep_hours")
    sleep_perf = w_data.get("sleep_performance")

    meeting_mins = act.get("total_meeting_minutes", 0)
    meeting_count = act.get("meeting_count", 0)
    active_windows = act.get("active_windows", 0)
    working_windows = act.get("working_windows", 0)
    slack_sent = act.get("slack_sent", 0)
    slack_recv = act.get("slack_received", 0)

    # ── DPS — Daily Presence Score ───────────────────────────────────────────
    dps_score: Optional[float] = None
    dps_tier: str = ""
    dps_label: str = ""
    try:
        from analysis.presence_score import compute_presence_score
        ps = compute_presence_score(windows)
        dps_score = ps.dps
        dps_tier = ps.tier
        # Human-readable tier labels (mirrors _dps_tier() in presence_score.py)
        _dps_tier_labels = {
            "exceptional": "Exceptional",
            "strong": "Strong",
            "good": "Good",
            "moderate": "Moderate",
            "low": "Low",
            "poor": "Poor",
        }
        dps_label = _dps_tier_labels.get(dps_tier, dps_tier.title())
    except Exception:
        pass

    # ── CDI — Cognitive Debt Index ────────────────────────────────────────────
    cdi_score: Optional[float] = None
    cdi_tier: str = ""
    cdi_label: str = ""
    cdi_meaningful: bool = False
    try:
        from analysis.cognitive_debt import compute_cdi
        debt = compute_cdi(date_str)
        cdi_score = debt.cdi
        cdi_tier = debt.tier
        cdi_meaningful = debt.is_meaningful
        _cdi_tier_labels = {
            "surplus": "Surplus",
            "balanced": "Balanced",
            "loading": "Loading",
            "fatigued": "Fatigued",
            "critical": "Critical",
        }
        cdi_label = _cdi_tier_labels.get(cdi_tier, cdi_tier.title())
    except Exception:
        pass

    # ── Format date label ──
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_label = dt.strftime("%A, %B %-d, %Y")
    except ValueError:
        date_label = date_str

    # ── Build charts ──
    cls_chart = _svg_cls_timeline(windows)
    heatmap = _svg_hourly_heatmap(windows)

    # ── Recovery colour ──
    rec_colour = _recovery_colour(recovery) if recovery is not None else "#888"
    rec_pct = f"{recovery:.0f}%" if recovery is not None else "N/A"

    # ── Metric bars ──
    metric_bars = ""
    metric_bars += _bar("Cognitive Load (CLS)", avg_cls, _cls_colour(avg_cls) if avg_cls else "#888")
    metric_bars += _bar("Focus Depth (FDI)", avg_fdi, "#60a5fa")
    metric_bars += _bar("Social Drain (SDI)", avg_sdi, "#a78bfa")
    metric_bars += _bar("Context Switch Cost (CSC)", avg_csc, "#fb923c")
    metric_bars += _bar("Recovery Alignment (RAS)", avg_ras, "#4ade80")

    # ── Trend section ──
    trend_html = ""
    if trend and trend.get("days_of_data", 0) >= 1:
        items = []
        hrv_trend = trend.get("hrv_trend", "stable")
        hrv_streak = trend.get("hrv_streak_days", 0)
        overcapacity = trend.get("overcapacity_streak", 0)
        hrv_vs_baseline = trend.get("hrv_vs_baseline")
        cls_vs_baseline = trend.get("cls_vs_baseline")

        if hrv_trend == "declining" and hrv_streak >= 2:
            items.append(f'<span class="badge badge-red">HRV ↓ {hrv_streak} days</span>')
        elif hrv_trend == "improving" and hrv_streak >= 2:
            items.append(f'<span class="badge badge-green">HRV ↑ {hrv_streak} days</span>')

        if overcapacity >= 2:
            items.append(f'<span class="badge badge-red">Over-capacity {overcapacity} days</span>')

        if hrv_vs_baseline is not None and abs(hrv_vs_baseline) >= 10:
            sign = "+" if hrv_vs_baseline > 0 else ""
            colour_cls = "badge-green" if hrv_vs_baseline > 0 else "badge-amber"
            items.append(f'<span class="badge {colour_cls}">HRV {sign}{hrv_vs_baseline:.0f}% vs baseline</span>')

        if cls_vs_baseline is not None and abs(cls_vs_baseline) >= 20:
            sign = "+" if cls_vs_baseline > 0 else ""
            colour_cls = "badge-amber" if cls_vs_baseline > 0 else "badge-green"
            items.append(f'<span class="badge {colour_cls}">Load {sign}{cls_vs_baseline:.0f}% vs baseline</span>')

        if items:
            trend_html = '<div class="trend-badges">' + " ".join(items) + "</div>"

    # ── RescueTime section ──
    rt_html = ""
    if rescuetime:
        rt_active = rescuetime.get("active_minutes", 0)
        rt_focus = rescuetime.get("focus_minutes", 0)
        rt_distraction = rescuetime.get("distraction_minutes", 0)
        rt_pct = rescuetime.get("productive_pct")
        rt_top = rescuetime.get("top_activity", "")

        rt_items = []
        if rt_active > 0:
            rt_items.append(f'<div class="stat-item"><span class="stat-val">{rt_active / 60:.1f}h</span><span class="stat-lbl">on computer</span></div>')
        if rt_focus > 0:
            pct_str = f" ({rt_pct:.0f}%)" if rt_pct is not None else ""
            rt_items.append(f'<div class="stat-item"><span class="stat-val">{rt_focus / 60:.1f}h{pct_str}</span><span class="stat-lbl">focused</span></div>')
        if rt_distraction > 0:
            rt_items.append(f'<div class="stat-item"><span class="stat-val">{rt_distraction / 60:.1f}h</span><span class="stat-lbl">distracted</span></div>')
        if rt_top:
            rt_items.append(f'<div class="stat-item"><span class="stat-val">{rt_top}</span><span class="stat-lbl">top app</span></div>')

        if rt_items:
            rt_html = f"""
<div class="card">
  <h2>💻 Computer Activity (RescueTime)</h2>
  <div class="stat-grid">{"".join(rt_items)}</div>
</div>"""

    # ── Omi section ──
    omi_windows = [w for w in windows if w.get("omi") and w["omi"].get("conversation_active")]
    omi_html = ""
    if omi_windows:
        total_words = sum(w["omi"].get("word_count", 0) for w in omi_windows)
        conv_minutes = len(omi_windows) * 15
        omi_html = f"""
<div class="card">
  <h2>🎙️ Conversation (Omi)</h2>
  <div class="stat-grid">
    <div class="stat-item"><span class="stat-val">{conv_minutes} min</span><span class="stat-lbl">conversation</span></div>
    <div class="stat-item"><span class="stat-val">{total_words:,}</span><span class="stat-lbl">words spoken</span></div>
    <div class="stat-item"><span class="stat-val">{len(omi_windows)}</span><span class="stat-lbl">active windows</span></div>
  </div>
</div>"""

    # ── Peak window ──
    peak_window = digest.get("peak_window")
    peak_html = ""
    if peak_window:
        ph = peak_window["metadata"]["hour_of_day"]
        pm = peak_window["metadata"]["minute_of_hour"]
        pcls = peak_window["metrics"]["cognitive_load_score"]
        ptitle = peak_window["calendar"].get("meeting_title") or ""
        peak_html = f"""
<div class="card">
  <h2>⚡ Peak Load Window</h2>
  <p class="peak-text">
    <strong>{ph:02d}:{pm:02d}</strong> — CLS {pcls:.0%}
    {"· " + ptitle if ptitle else ""}
  </p>
</div>"""

    # ── DPS + CDI hero card ───────────────────────────────────────────────────
    hero_html = ""
    dps_ring_html = ""
    cdi_ring_html = ""

    if dps_score is not None:
        dps_col = _dps_colour(dps_score)
        dps_ring_html = _svg_score_ring(dps_score, dps_col, size=90)
        dps_block = f"""
    <div class="score-item">
      {dps_ring_html}
      <div class="score-meta">
        <div class="score-title">Daily Presence Score</div>
        <div class="score-tier" style="color:{dps_col}">{dps_label}</div>
        <div class="score-desc">Cognitive day quality</div>
      </div>
    </div>"""
    else:
        dps_block = '<div class="score-item score-na"><div class="score-title">DPS</div><div class="score-desc">Not enough data</div></div>'

    if cdi_score is not None and cdi_meaningful:
        cdi_col = _cdi_colour(cdi_tier)
        cdi_ring_html = _svg_score_ring(cdi_score, cdi_col, size=90)
        cdi_block = f"""
    <div class="score-item">
      {cdi_ring_html}
      <div class="score-meta">
        <div class="score-title">Cognitive Debt Index</div>
        <div class="score-tier" style="color:{cdi_col}">{cdi_label}</div>
        <div class="score-desc">14-day fatigue balance</div>
      </div>
    </div>"""
    elif cdi_score is not None:
        # Have a score but not yet meaningful — show greyed out
        cdi_col = "#94a3b8"
        cdi_ring_html = _svg_score_ring(cdi_score, cdi_col, size=90)
        cdi_block = f"""
    <div class="score-item">
      {cdi_ring_html}
      <div class="score-meta">
        <div class="score-title">Cognitive Debt Index</div>
        <div class="score-tier" style="color:{cdi_col}">Warming up</div>
        <div class="score-desc">Need ≥3 days of data</div>
      </div>
    </div>"""
    else:
        cdi_block = '<div class="score-item score-na"><div class="score-title">CDI</div><div class="score-desc">Not enough data</div></div>'

    hero_html = f"""
<div class="card hero-card">
  <h2>🎯 Today's Scores</h2>
  <div class="score-grid">
    {dps_block}
    {cdi_block}
  </div>
</div>"""

    # ── Flow State card ───────────────────────────────────────────────────────
    flow_html = ""
    try:
        from analysis.flow_detector import detect_flow_states, FlowStateResult

        flow_result: FlowStateResult = detect_flow_states(windows)
        if flow_result.is_meaningful:
            # Colour by flow label
            _flow_colours = {
                "deep_flow": "#4ade80",   # green
                "in_zone":   "#60a5fa",   # blue
                "brief":     "#facc15",   # yellow
                "none":      "#94a3b8",   # grey
            }
            flow_colour = _flow_colours.get(flow_result.flow_label, "#94a3b8")
            flow_label_display = {
                "deep_flow": "Deep Flow",
                "in_zone":   "In the Zone",
                "brief":     "Brief Focus",
                "none":      "No Flow",
            }.get(flow_result.flow_label, flow_result.flow_label.replace("_", " ").title())

            # LVI score bar (0–1 → 0–100%)
            flow_pct = flow_result.flow_score * 100
            flow_bar = (
                f'<div class="bar-track" style="margin-top:8px">'
                f'<div class="bar-fill" style="width:{flow_pct:.1f}%;background:{flow_colour}"></div>'
                f'</div>'
            )

            # Best session block
            peak_session_html = ""
            if flow_result.peak_session:
                ps = flow_result.peak_session
                peak_session_html = (
                    f'<div style="margin-top:10px;font-size:0.82rem;color:var(--muted)">'
                    f'Best session: <strong style="color:var(--text)">'
                    f'{ps.start_time}–{ps.end_time}</strong>'
                    f' · {ps.duration_minutes} min'
                    f' · FDI {ps.avg_fdi:.0%}</div>'
                )

            sessions_count = len(flow_result.flow_sessions)
            sessions_label = f"{sessions_count} session{'s' if sessions_count != 1 else ''}" if sessions_count else "no sessions"

            flow_html = f"""
<div class="card">
  <h2>🌊 Flow State</h2>
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
    <div>
      <div style="font-size:1.6rem;font-weight:700;color:{flow_colour}">{flow_result.total_flow_minutes} min</div>
      <div style="font-size:0.78rem;color:var(--muted)">flow time · {sessions_label}</div>
    </div>
    <div style="flex:1">
      <div style="font-size:1.0rem;font-weight:600;color:{flow_colour}">{flow_label_display}</div>
      <div style="font-size:0.78rem;color:var(--muted)">Score: {flow_result.flow_score:.2f}</div>
      {flow_bar}
    </div>
  </div>
  {peak_session_html}
  <div style="margin-top:8px;font-size:0.78rem;color:var(--muted)">{flow_result.insight}</div>
</div>"""
    except Exception:
        pass

    # ── Load Volatility (LVI) card ────────────────────────────────────────────
    lvi_html = ""
    try:
        from analysis.load_volatility import compute_load_volatility, LoadVolatility

        lvi_result: LoadVolatility = compute_load_volatility(windows)
        if lvi_result.is_meaningful:
            _lvi_colours = {
                "smooth":   "#4ade80",   # green
                "steady":   "#60a5fa",   # blue
                "variable": "#facc15",   # yellow
                "volatile": "#f87171",   # red
            }
            lvi_colour = _lvi_colours.get(lvi_result.label, "#94a3b8")
            lvi_label_display = lvi_result.label.title()
            lvi_pct = lvi_result.lvi * 100

            lvi_bar = (
                f'<div class="bar-track" style="margin-top:8px">'
                f'<div class="bar-fill" style="width:{lvi_pct:.1f}%;background:{lvi_colour}"></div>'
                f'</div>'
            )

            lvi_html = f"""
<div class="card">
  <h2>📈 Load Volatility</h2>
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
    <div>
      <div style="font-size:1.6rem;font-weight:700;color:{lvi_colour}">{lvi_result.lvi:.2f}</div>
      <div style="font-size:0.78rem;color:var(--muted)">LVI · 0=volatile 1=smooth</div>
    </div>
    <div style="flex:1">
      <div style="font-size:1.0rem;font-weight:600;color:{lvi_colour}">{lvi_label_display}</div>
      <div style="font-size:0.78rem;color:var(--muted)">
        Std: {lvi_result.cls_std:.3f} · Range: {lvi_result.cls_range:.3f} · Windows: {lvi_result.windows_used}
      </div>
      {lvi_bar}
    </div>
  </div>
  <div style="margin-top:4px;font-size:0.78rem;color:var(--muted)">{lvi_result.insight}</div>
</div>"""
    except Exception:
        pass

    # ── Full HTML template ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Presence Report — {date_label}</title>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3e;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #7c3aed;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px;
    line-height: 1.5;
    padding: 24px;
    max-width: 900px;
    margin: 0 auto;
  }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; }}
  h2 {{ font-size: 0.85rem; font-weight: 600; color: var(--muted); text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 12px; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
  }}
  .card-full {{ grid-column: 1 / -1; }}
  .recovery-circle {{
    display: flex;
    align-items: center;
    gap: 20px;
  }}
  .recovery-badge {{
    width: 72px;
    height: 72px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    font-weight: 700;
    flex-shrink: 0;
    border: 3px solid {rec_colour};
    color: {rec_colour};
  }}
  .recovery-stats {{ display: flex; flex-direction: column; gap: 4px; }}
  .recovery-stat {{ display: flex; gap: 8px; }}
  .recovery-stat .lbl {{ color: var(--muted); min-width: 80px; }}
  .metric-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }}
  .metric-label {{ width: 200px; font-size: 0.82rem; color: var(--muted); flex-shrink: 0; }}
  .bar-track {{
    flex: 1;
    height: 8px;
    background: #2a2d3e;
    border-radius: 4px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
  }}
  .metric-value {{ width: 42px; text-align: right; font-size: 0.82rem; font-weight: 600; }}
  .stat-grid {{
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
  }}
  .stat-item {{ display: flex; flex-direction: column; gap: 2px; }}
  .stat-val {{ font-size: 1.1rem; font-weight: 700; }}
  .stat-lbl {{ font-size: 0.75rem; color: var(--muted); }}
  .insight-box {{
    background: #1e2235;
    border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-top: 8px;
  }}
  .trend-badges {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
  }}
  .badge {{
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge-red {{ background: #7f1d1d; color: #fca5a5; }}
  .badge-green {{ background: #14532d; color: #86efac; }}
  .badge-amber {{ background: #78350f; color: #fcd34d; }}
  .chart-wrapper {{ overflow-x: auto; }}
  svg {{ display: block; }}
  .peak-text {{ font-size: 0.95rem; }}
  footer {{ text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 32px; }}
  /* DPS + CDI hero card */
  .hero-card {{ margin-bottom: 16px; }}
  .score-grid {{
    display: flex;
    gap: 40px;
    flex-wrap: wrap;
    align-items: center;
    padding-top: 4px;
  }}
  .score-item {{
    display: flex;
    align-items: center;
    gap: 16px;
  }}
  .score-item.score-na {{
    color: var(--muted);
    font-size: 0.88rem;
    align-items: center;
    gap: 8px;
    padding: 12px 0;
  }}
  .score-meta {{ display: flex; flex-direction: column; gap: 3px; }}
  .score-title {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
  .score-tier {{ font-size: 1.0rem; font-weight: 700; }}
  .score-desc {{ font-size: 0.75rem; color: var(--muted); }}
</style>
</head>
<body>

<h1>🧠 Presence Report</h1>
<p class="subtitle">{date_label}</p>

{trend_html}

<!-- DPS + CDI hero scores -->
{hero_html}

<!-- Recovery + Activity row -->
<div class="grid-2">
  <div class="card">
    <h2>💚 Recovery</h2>
    <div class="recovery-circle">
      <div class="recovery-badge">{rec_pct}</div>
      <div class="recovery-stats">
        {"<div class='recovery-stat'><span class='lbl'>HRV</span><span>" + f"{hrv:.0f} ms" + "</span></div>" if hrv else ""}
        {"<div class='recovery-stat'><span class='lbl'>Sleep</span><span>" + f"{sleep_h:.1f}h" + "</span></div>" if sleep_h else ""}
        {"<div class='recovery-stat'><span class='lbl'>Sleep perf.</span><span>" + f"{sleep_perf:.0f}%" + "</span></div>" if sleep_perf else ""}
      </div>
    </div>
  </div>

  <div class="card">
    <h2>📊 Activity</h2>
    <div class="stat-grid">
      <div class="stat-item">
        <span class="stat-val">{meeting_count}</span>
        <span class="stat-lbl">meetings ({meeting_mins} min)</span>
      </div>
      <div class="stat-item">
        <span class="stat-val">{slack_sent}</span>
        <span class="stat-lbl">messages sent</span>
      </div>
      <div class="stat-item">
        <span class="stat-val">{active_windows}</span>
        <span class="stat-lbl">active windows</span>
      </div>
      <div class="stat-item">
        <span class="stat-val">{slack_recv}</span>
        <span class="stat-lbl">messages received</span>
      </div>
    </div>
  </div>
</div>

<!-- CLS Timeline -->
<div class="card">
  <h2>⏱ Cognitive Load Timeline (CLS)</h2>
  <div class="chart-wrapper">
    {cls_chart}
  </div>
  <p style="font-size:0.75rem;color:var(--muted);margin-top:6px;">
    Shaded area = working hours (7am–10pm). Area under curve = cumulative load.
  </p>
</div>

<!-- Hourly Heatmap -->
<div class="card">
  <h2>🌡 Hourly Intensity (7am–10pm)</h2>
  <div class="chart-wrapper">
    {heatmap}
  </div>
</div>

<!-- Metric Bars -->
<div class="card">
  <h2>📐 Metric Breakdown</h2>
  {metric_bars}
</div>

{rt_html}

{omi_html}

{peak_html}

<!-- Flow State + Load Volatility side-by-side when both present -->
{f'<div class="grid-2">{flow_html}{lvi_html}</div>' if flow_html and lvi_html else flow_html + lvi_html}

<!-- Insight -->
<div class="card">
  <h2>💡 Insight</h2>
  <div class="insight-box">{insight}</div>
</div>

<footer>
  Presence Tracker · Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} ·
  Data sources: {", ".join(windows[0]["metadata"].get("sources_available", []))}
</footer>

</body>
</html>"""

    # ── Write file ──
    if output_path is None:
        output_path = DASHBOARD_DIR / f"{date_str}.html"

    output_path.write_text(html, encoding="utf-8")
    print(f"[dashboard] Generated: {output_path}")
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate daily HTML presence dashboard")
    parser.add_argument("date", nargs="?", help="Date YYYY-MM-DD (default: latest)")
    parser.add_argument("--output", help="Output path override")
    parser.add_argument("--open", action="store_true", help="Open in browser after generation")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.store import list_available_dates

    if args.date:
        date_str = args.date
    else:
        dates = list_available_dates()
        if not dates:
            print("No data available.", file=sys.stderr)
            sys.exit(1)
        date_str = sorted(dates)[-1]

    out = generate_dashboard(date_str, output_path=Path(args.output) if args.output else None)
    print(f"Dashboard: {out}")

    if args.open:
        import subprocess
        subprocess.run(["open", str(out)])

"""
Presence Tracker — Weekly HTML Presence Dashboard (v14)

Generates a self-contained HTML file covering the 7-day window ending on a given date:
  data/dashboard/week-YYYY-MM-DD.html

Why this exists
---------------
The daily dashboard (analysis/dashboard.py) shows one day in detail.
The weekly Slack summary shows numbers but no visuals.
This dashboard closes the gap: a single page that shows the full week at a glance —
DPS trend, per-day metric bars, WHOOP recovery arc, meeting pattern, and CDI trajectory.

Charts (pure SVG — no CDN, no external dependencies):
  - DPS sparkline: 7-day presence score trend with tier bands
  - Per-day metric grid: CLS / FDI / RAS bars for each day side by side
  - WHOOP recovery arc: 7-day recovery % with HRV overlay
  - Meeting load chart: total meeting minutes per day
  - CDI trajectory: cognitive debt evolution across the week
  - Source coverage heatmap: which sources fired on which day

Usage (CLI):
    python3 analysis/weekly_dashboard.py                  # Last available 7 days
    python3 analysis/weekly_dashboard.py 2026-03-14       # Week ending on this date
    python3 analysis/weekly_dashboard.py --open           # Open in browser after

Usage (programmatic):
    from analysis.weekly_dashboard import generate_weekly_dashboard
    path = generate_weekly_dashboard("2026-03-14")
"""

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR
from engine.store import list_available_dates, read_summary

DASHBOARD_DIR = DATA_DIR / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

# ─── Colour helpers ────────────────────────────────────────────────────────────

def _cls_colour(v: float) -> str:
    if v < 0.25:   return "#4ade80"
    elif v < 0.50: return "#facc15"
    elif v < 0.75: return "#fb923c"
    return "#f87171"


def _fdi_colour(v: float) -> str:
    if v >= 0.75:  return "#4ade80"
    elif v >= 0.50:return "#facc15"
    elif v >= 0.25:return "#fb923c"
    return "#f87171"


def _ras_colour(v: float) -> str:
    if v >= 0.75:  return "#4ade80"
    elif v >= 0.50:return "#facc15"
    return "#f87171"


def _recovery_colour(pct: float) -> str:
    if pct >= 67:  return "#4ade80"
    elif pct >= 34:return "#facc15"
    return "#f87171"


def _dps_colour(score: float) -> str:
    if score >= 80: return "#4ade80"
    elif score >= 60: return "#facc15"
    elif score >= 40: return "#fb923c"
    return "#f87171"


def _dps_tier(score: Optional[float]) -> str:
    if score is None: return "—"
    if score >= 90: return "Exceptional"
    if score >= 80: return "Strong"
    if score >= 70: return "Good"
    if score >= 60: return "Moderate"
    if score >= 40: return "Challenging"
    return "Difficult"


def _fmt_val(v: Optional[float], fmt: str = ".0%") -> str:
    if v is None: return "—"
    return format(v, fmt)


# ─── Week date helpers ─────────────────────────────────────────────────────────

def _week_dates(end_date_str: str) -> list[str]:
    """Return the 7 dates in [end-6 … end], oldest first."""
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    return [(end - timedelta(days=6 - i)).strftime("%Y-%m-%d") for i in range(7)]


# ─── Data loading ──────────────────────────────────────────────────────────────

def _load_week(end_date_str: str) -> list[dict]:
    """
    Load rolling summary data for the 7 days ending on end_date_str.

    Returns a list (oldest→newest) with one entry per day; days without data
    are represented as {"date": YYYY-MM-DD, "missing": True}.
    """
    dates = _week_dates(end_date_str)
    summary = read_summary()
    all_days = summary.get("days", {})
    result = []
    for d in dates:
        if d in all_days:
            row = {"date": d, "missing": False, **all_days[d]}
        else:
            row = {"date": d, "missing": True}
        result.append(row)
    return result


# ─── SVG chart builders ────────────────────────────────────────────────────────

def _svg_dps_sparkline(days: list[dict], width: int = 740, height: int = 160) -> str:
    """
    SVG line chart for DPS across the week.
    Draws tier bands (exceptional≥90, strong≥80, good≥70) as background fills.
    """
    n = len(days)
    pad_l, pad_r, pad_t, pad_b = 48, 12, 16, 32
    cw = width - pad_l - pad_r
    ch = height - pad_t - pad_b

    def _x(i: int) -> float:
        return pad_l + (i / max(n - 1, 1)) * cw

    def _y(score: float) -> float:
        # Score 0–100 maps to height
        return pad_t + ch - (score / 100) * ch

    # Background tier bands
    bands = [
        (90, 100, "#14532d", "Exceptional"),
        (80, 90,  "#166534", "Strong"),
        (70, 80,  "#1e3a1e", "Good"),
        (60, 70,  "#2d2a10", "Moderate"),
        (0,  60,  "#2d1515", ""),
    ]

    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']

    # Draw tier bands
    for lo, hi, colour, label in bands:
        y_hi = _y(hi)
        y_lo = _y(lo)
        rect_h = y_lo - y_hi
        out.append(f'  <rect x="{pad_l}" y="{y_hi:.1f}" width="{cw}" height="{rect_h:.1f}" fill="{colour}" opacity="0.4"/>')
        if label:
            out.append(f'  <text x="{pad_l + 4}" y="{y_hi + 11:.1f}" font-size="9" fill="#6b7280">{label}</text>')

    # Y-axis labels
    for val in [0, 20, 40, 60, 80, 100]:
        y = _y(val)
        out.append(f'  <line x1="{pad_l - 4}" y1="{y:.1f}" x2="{pad_l + cw}" y2="{y:.1f}" stroke="#374151" stroke-width="0.5"/>')
        out.append(f'  <text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="9" fill="#9ca3af" text-anchor="end">{val}</text>')

    # Collect valid points
    pts = []
    for i, day in enumerate(days):
        if day.get("missing"):
            pts.append(None)
            continue
        # Try to get DPS from rolling summary
        dps = (
            day.get("presence_score", {}).get("dps")
            or day.get("dps")
            or None
        )
        pts.append(dps)

    # Draw line segments (skip gaps)
    path_d = ""
    first = True
    for i, score in enumerate(pts):
        if score is None:
            first = True
            continue
        x, y = _x(i), _y(score)
        if first:
            path_d += f"M {x:.1f} {y:.1f} "
            first = False
        else:
            path_d += f"L {x:.1f} {y:.1f} "

    if path_d:
        out.append(f'  <path d="{path_d.strip()}" stroke="#818cf8" stroke-width="2" fill="none" stroke-linejoin="round"/>')

    # Data points + day labels
    for i, (day, score) in enumerate(zip(days, pts)):
        x = _x(i)
        date_label = day["date"][5:]  # MM-DD
        dow = datetime.strptime(day["date"], "%Y-%m-%d").strftime("%a")

        # Day label at bottom
        out.append(f'  <text x="{x:.1f}" y="{height - 4}" font-size="9" fill="#9ca3af" text-anchor="middle">{dow}</text>')
        out.append(f'  <text x="{x:.1f}" y="{height - 14}" font-size="8" fill="#6b7280" text-anchor="middle">{date_label}</text>')

        if score is not None:
            y = _y(score)
            colour = _dps_colour(score)
            out.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}"/>')
            out.append(f'  <text x="{x:.1f}" y="{y - 7:.1f}" font-size="10" fill="{colour}" text-anchor="middle" font-weight="600">{score:.0f}</text>')
        else:
            out.append(f'  <text x="{x:.1f}" y="{pad_t + ch / 2:.1f}" font-size="9" fill="#4b5563" text-anchor="middle">—</text>')

    out.append("</svg>")
    return "\n".join(out)


def _svg_metric_grid(days: list[dict], width: int = 740, height: int = 140) -> str:
    """
    Per-day stacked bar chart: CLS (bad=high), FDI (good=high), RAS (good=high).
    Three mini bars per day, grouped.
    """
    n = len(days)
    pad_l, pad_r, pad_t, pad_b = 40, 12, 16, 40
    cw = width - pad_l - pad_r
    ch = height - pad_t - pad_b

    slot_w = cw / n
    bar_w = max(8.0, slot_w / 5)
    gap = bar_w * 0.4

    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']

    # Y-axis
    for val in [0, 0.25, 0.5, 0.75, 1.0]:
        y = pad_t + ch - val * ch
        out.append(f'  <line x1="{pad_l - 4}" y1="{y:.1f}" x2="{pad_l + cw}" y2="{y:.1f}" stroke="#374151" stroke-width="0.5"/>')
        out.append(f'  <text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="9" fill="#9ca3af" text-anchor="end">{val:.0%}</text>')

    for i, day in enumerate(days):
        slot_cx = pad_l + (i + 0.5) * slot_w
        date_label = day["date"][5:]
        dow = datetime.strptime(day["date"], "%Y-%m-%d").strftime("%a")

        out.append(f'  <text x="{slot_cx:.1f}" y="{height - 4}" font-size="9" fill="#9ca3af" text-anchor="middle">{dow}</text>')
        out.append(f'  <text x="{slot_cx:.1f}" y="{height - 14}" font-size="8" fill="#6b7280" text-anchor="middle">{date_label}</text>')

        if day.get("missing"):
            out.append(f'  <text x="{slot_cx:.1f}" y="{pad_t + ch / 2:.1f}" font-size="9" fill="#4b5563" text-anchor="middle">—</text>')
            continue

        m = day.get("metrics_avg", {}) or {}
        cls = m.get("cognitive_load_score")
        fdi = m.get("focus_depth_index")
        ras = m.get("recovery_alignment_score")

        # Three bars: CLS (centre − gap*2), FDI (centre − gap*0.5), RAS (centre + gap)
        bar_configs = [
            (cls, _cls_colour(cls) if cls is not None else "#374151", slot_cx - bar_w - gap, "CLS"),
            (fdi, _fdi_colour(fdi) if fdi is not None else "#374151", slot_cx, "FDI"),
            (ras, _ras_colour(ras) if ras is not None else "#374151", slot_cx + bar_w + gap, "RAS"),
        ]

        for val, colour, bx, lbl in bar_configs:
            if val is None:
                continue
            bar_h = val * ch
            by = pad_t + ch - bar_h
            out.append(f'  <rect x="{bx - bar_w / 2:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colour}" rx="2"/>')
            out.append(f'  <text x="{bx:.1f}" y="{by - 3:.1f}" font-size="8" fill="{colour}" text-anchor="middle">{val:.0%}</text>')

    # Legend
    legend_items = [
        ("#f87171", "CLS (load)"),
        ("#4ade80", "FDI (focus)"),
        ("#818cf8", "RAS (align)"),
    ]
    lx = pad_l
    for colour, label in legend_items:
        out.append(f'  <rect x="{lx}" y="{pad_t - 12}" width="8" height="8" fill="{colour}" rx="1"/>')
        out.append(f'  <text x="{lx + 10}" y="{pad_t - 5}" font-size="9" fill="#9ca3af">{label}</text>')
        lx += 85

    out.append("</svg>")
    return "\n".join(out)


def _svg_recovery_chart(days: list[dict], width: int = 740, height: int = 120) -> str:
    """
    WHOOP recovery % as filled area + HRV as overlaid line (secondary axis).
    """
    n = len(days)
    pad_l, pad_r, pad_t, pad_b = 48, 12, 16, 32
    cw = width - pad_l - pad_r
    ch = height - pad_t - pad_b

    def _x(i: int) -> float:
        return pad_l + (i / max(n - 1, 1)) * cw

    # Tier bands: green ≥67, yellow 34-66, red <34
    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']

    # Background bands
    bands = [(67, 100, "#14532d"), (34, 67, "#78350f"), (0, 34, "#7f1d1d")]
    for lo, hi, col in bands:
        y_hi = pad_t + ch - (hi / 100) * ch
        y_lo = pad_t + ch - (lo / 100) * ch
        out.append(f'  <rect x="{pad_l}" y="{y_hi:.1f}" width="{cw}" height="{y_lo - y_hi:.1f}" fill="{col}" opacity="0.3"/>')

    # Y-axis ticks
    for val in [0, 25, 50, 67, 75, 100]:
        y = pad_t + ch - (val / 100) * ch
        out.append(f'  <line x1="{pad_l - 4}" y1="{y:.1f}" x2="{pad_l + cw}" y2="{y:.1f}" stroke="#374151" stroke-width="0.5"/>')
        out.append(f'  <text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="9" fill="#9ca3af" text-anchor="end">{val}%</text>')

    # Recovery filled area
    rec_pts = []
    for i, day in enumerate(days):
        if day.get("missing"):
            rec_pts.append(None)
            continue
        rec = (day.get("whoop", {}) or {}).get("recovery_score")
        rec_pts.append(rec)

    # Build area path
    area_d = ""
    line_d = ""
    prev_valid = None
    for i, rec in enumerate(rec_pts):
        if rec is None:
            prev_valid = None
            continue
        x = _x(i)
        y = pad_t + ch - (rec / 100) * ch
        if prev_valid is None:
            area_d += f"M {x:.1f} {pad_t + ch:.1f} L {x:.1f} {y:.1f} "
            line_d += f"M {x:.1f} {y:.1f} "
        else:
            area_d += f"L {x:.1f} {y:.1f} "
            line_d += f"L {x:.1f} {y:.1f} "
        prev_valid = (x, y)

    if prev_valid and area_d:
        area_d += f"L {prev_valid[0]:.1f} {pad_t + ch:.1f} Z"
        out.append(f'  <path d="{area_d}" fill="#3b82f6" opacity="0.25"/>')
        out.append(f'  <path d="{line_d}" stroke="#3b82f6" stroke-width="1.5" fill="none"/>')

    # Data point labels + day labels
    for i, (day, rec) in enumerate(zip(days, rec_pts)):
        x = _x(i)
        dow = datetime.strptime(day["date"], "%Y-%m-%d").strftime("%a")
        date_label = day["date"][5:]
        out.append(f'  <text x="{x:.1f}" y="{height - 4}" font-size="9" fill="#9ca3af" text-anchor="middle">{dow}</text>')
        out.append(f'  <text x="{x:.1f}" y="{height - 14}" font-size="8" fill="#6b7280" text-anchor="middle">{date_label}</text>')

        if rec is not None:
            y = pad_t + ch - (rec / 100) * ch
            col = _recovery_colour(rec)
            out.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{col}"/>')
            out.append(f'  <text x="{x:.1f}" y="{y - 6:.1f}" font-size="9" fill="{col}" text-anchor="middle">{rec:.0f}%</text>')

    out.append("</svg>")
    return "\n".join(out)


def _svg_meeting_bars(days: list[dict], width: int = 740, height: int = 100) -> str:
    """Horizontal grouped bars: meeting minutes per day."""
    n = len(days)
    pad_l, pad_r, pad_t, pad_b = 48, 12, 8, 36
    cw = width - pad_l - pad_r
    ch = height - pad_t - pad_b

    # Find max
    maxm = 0
    for day in days:
        if day.get("missing"):
            continue
        m = (day.get("calendar", {}) or {}).get("total_meeting_minutes", 0) or 0
        if m > maxm:
            maxm = m
    maxm = max(maxm, 60)  # at least 1h scale

    slot_w = cw / n
    bar_w = max(12.0, slot_w * 0.55)

    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']

    # Y-axis ticks at 0, 60, 120, 180, 240
    for val_m in [0, 60, 120, 180, 240]:
        if val_m > maxm * 1.1:
            continue
        y = pad_t + ch - (val_m / maxm) * ch
        out.append(f'  <line x1="{pad_l - 4}" y1="{y:.1f}" x2="{pad_l + cw}" y2="{y:.1f}" stroke="#374151" stroke-width="0.5"/>')
        label = f"{val_m // 60}h" if val_m > 0 else "0"
        out.append(f'  <text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="9" fill="#9ca3af" text-anchor="end">{label}</text>')

    for i, day in enumerate(days):
        slot_cx = pad_l + (i + 0.5) * slot_w
        dow = datetime.strptime(day["date"], "%Y-%m-%d").strftime("%a")
        date_label = day["date"][5:]
        out.append(f'  <text x="{slot_cx:.1f}" y="{height - 4}" font-size="9" fill="#9ca3af" text-anchor="middle">{dow}</text>')
        out.append(f'  <text x="{slot_cx:.1f}" y="{height - 14}" font-size="8" fill="#6b7280" text-anchor="middle">{date_label}</text>')

        if day.get("missing"):
            continue

        mtg_mins = (day.get("calendar", {}) or {}).get("total_meeting_minutes", 0) or 0
        if mtg_mins == 0:
            out.append(f'  <text x="{slot_cx:.1f}" y="{pad_t + ch - 2:.1f}" font-size="8" fill="#4b5563" text-anchor="middle">—</text>')
            continue

        ratio = min(mtg_mins / maxm, 1.0)
        bar_h = ratio * ch
        by = pad_t + ch - bar_h
        # colour by load: < 1h green, 1-2h yellow, 2-3h orange, >3h red
        if mtg_mins < 60:    col = "#4ade80"
        elif mtg_mins < 120: col = "#facc15"
        elif mtg_mins < 180: col = "#fb923c"
        else:                col = "#f87171"
        out.append(f'  <rect x="{slot_cx - bar_w / 2:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{col}" rx="2"/>')
        h_str = f"{mtg_mins // 60}h{mtg_mins % 60:02d}m" if mtg_mins >= 60 else f"{mtg_mins}m"
        out.append(f'  <text x="{slot_cx:.1f}" y="{by - 3:.1f}" font-size="8" fill="{col}" text-anchor="middle">{h_str}</text>')

    out.append("</svg>")
    return "\n".join(out)


def _source_coverage_table(days: list[dict]) -> str:
    """HTML table showing which sources fired on which day."""
    sources = ["whoop", "calendar", "slack", "rescuetime", "omi"]
    emoji = {
        "whoop": "💚", "calendar": "📅", "slack": "💬",
        "rescuetime": "⏱", "omi": "🎙"
    }

    rows = []
    rows.append("<table class='coverage-table'><thead><tr>")
    rows.append("<th>Source</th>")
    for day in days:
        dow = datetime.strptime(day["date"], "%Y-%m-%d").strftime("%a")
        rows.append(f"<th>{dow}<br/><span class='dim'>{day['date'][5:]}</span></th>")
    rows.append("</tr></thead><tbody>")

    for src in sources:
        rows.append(f"<tr><td>{emoji.get(src, '')} {src}</td>")
        for day in days:
            if day.get("missing"):
                rows.append("<td class='cov-miss'>—</td>")
                continue
            avail = day.get("metadata", {}).get("sources_available_set") or []
            # Also check via day-level source_coverage
            src_cov = day.get("source_coverage", {}) or {}
            # Check rolling summary sources_available (stored as a set per day)
            # Fall back to checking any window-level data
            has_src = (
                src in avail
                or src_cov.get(src, 0) > 0
            )
            if has_src:
                rows.append("<td class='cov-yes'>✓</td>")
            else:
                rows.append("<td class='cov-no'>✗</td>")
        rows.append("</tr>")

    rows.append("</tbody></table>")
    return "\n".join(rows)


# ─── Weekly summary stats ──────────────────────────────────────────────────────

def _week_stats(days: list[dict]) -> dict:
    """Compute high-level weekly aggregate stats."""
    valid = [d for d in days if not d.get("missing")]

    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    cls_vals = [d.get("metrics_avg", {}).get("cognitive_load_score") for d in valid]
    fdi_vals = [d.get("metrics_avg", {}).get("focus_depth_index") for d in valid]
    ras_vals = [d.get("metrics_avg", {}).get("recovery_alignment_score") for d in valid]
    rec_vals = [(d.get("whoop", {}) or {}).get("recovery_score") for d in valid]

    dps_vals = [
        d.get("presence_score", {}).get("dps") or d.get("dps")
        for d in valid
    ]
    dps_vals = [v for v in dps_vals if v is not None]

    mtg_total = sum(
        (d.get("calendar", {}) or {}).get("total_meeting_minutes", 0) or 0
        for d in valid
    )

    best_dps_day = None
    worst_dps_day = None
    if dps_vals:
        best_idx = dps_vals.index(max(dps_vals))
        worst_idx = dps_vals.index(min(dps_vals))
        # Map back to valid days
        best_dps_day = (valid[best_idx]["date"], dps_vals[best_idx]) if valid else None
        worst_dps_day = (valid[worst_idx]["date"], dps_vals[worst_idx]) if valid else None

    return {
        "days_with_data": len(valid),
        "avg_cls": _avg(cls_vals),
        "avg_fdi": _avg(fdi_vals),
        "avg_ras": _avg(ras_vals),
        "avg_recovery": _avg(rec_vals),
        "avg_dps": _avg(dps_vals),
        "total_meeting_minutes": mtg_total,
        "best_dps_day": best_dps_day,
        "worst_dps_day": worst_dps_day,
    }


# ─── Main generator ────────────────────────────────────────────────────────────

def generate_weekly_dashboard(
    end_date_str: str,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate the weekly HTML presence dashboard for the 7 days ending on end_date_str.

    Parameters
    ----------
    end_date_str : str
        Last day of the week (YYYY-MM-DD). The dashboard covers [end-6 … end].
    output_path : Path | None
        Override the output path. Defaults to data/dashboard/week-YYYY-MM-DD.html.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    days = _load_week(end_date_str)
    stats = _week_stats(days)

    start_str = days[0]["date"]
    end_str = days[-1]["date"]
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    date_range = f"{start_dt.strftime('%b %-d')} – {end_dt.strftime('%b %-d, %Y')}"

    # Build chart SVGs
    dps_chart = _svg_dps_sparkline(days)
    metric_chart = _svg_metric_grid(days)
    recovery_chart = _svg_recovery_chart(days)
    meeting_chart = _svg_meeting_bars(days)
    coverage_table = _source_coverage_table(days)

    # Stats for the summary hero
    avg_dps = stats["avg_dps"]
    avg_dps_str = f"{avg_dps:.0f}" if avg_dps is not None else "—"
    avg_dps_tier = _dps_tier(avg_dps)
    avg_dps_colour = _dps_colour(avg_dps) if avg_dps is not None else "#6b7280"

    avg_recovery = stats["avg_recovery"]
    avg_rec_str = f"{avg_recovery:.0f}%" if avg_recovery is not None else "—"
    avg_rec_colour = _recovery_colour(avg_recovery) if avg_recovery is not None else "#6b7280"

    avg_fdi = stats["avg_fdi"]
    avg_fdi_str = f"{avg_fdi:.0%}" if avg_fdi is not None else "—"
    avg_fdi_colour = _fdi_colour(avg_fdi) if avg_fdi is not None else "#6b7280"

    avg_cls = stats["avg_cls"]
    avg_cls_str = f"{avg_cls:.0%}" if avg_cls is not None else "—"
    avg_cls_colour = _cls_colour(avg_cls) if avg_cls is not None else "#6b7280"

    mtg_total = stats["total_meeting_minutes"]
    mtg_str = f"{mtg_total // 60}h{mtg_total % 60:02d}m" if mtg_total >= 60 else f"{mtg_total}m"

    days_with_data = stats["days_with_data"]

    best_day_html = ""
    worst_day_html = ""
    if stats["best_dps_day"]:
        d, s = stats["best_dps_day"]
        dow = datetime.strptime(d, "%Y-%m-%d").strftime("%A %b %-d")
        best_day_html = f'<div class="stat-item"><span class="stat-val" style="color:{_dps_colour(s)}">🌟 {s:.0f}</span><span class="stat-lbl">Best day — {dow}</span></div>'
    if stats["worst_dps_day"]:
        d, s = stats["worst_dps_day"]
        dow = datetime.strptime(d, "%Y-%m-%d").strftime("%A %b %-d")
        worst_day_html = f'<div class="stat-item"><span class="stat-val" style="color:{_dps_colour(s)}">📉 {s:.0f}</span><span class="stat-lbl">Lowest day — {dow}</span></div>'

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Weekly Presence — {date_range}</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d2e;
    --surface2: #222538;
    --border: #2d3048;
    --text: #e2e8f0;
    --muted: #6b7280;
    --accent: #818cf8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 24px; max-width: 820px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); margin-bottom: 4px; }}
  .subtitle {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; }}
  h2 {{ font-size: 0.95rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; }}
  .hero-grid {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; margin-bottom: 8px; }}
  .hero-item {{ display: flex; flex-direction: column; gap: 3px; min-width: 80px; }}
  .hero-val {{ font-size: 2rem; font-weight: 800; line-height: 1; }}
  .hero-tier {{ font-size: 0.78rem; font-weight: 600; }}
  .hero-lbl {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 12px; }}
  .stat-item {{ display: flex; flex-direction: column; gap: 3px; }}
  .stat-val {{ font-size: 1.05rem; font-weight: 700; }}
  .stat-lbl {{ font-size: 0.73rem; color: var(--muted); }}
  .chart-wrapper {{ overflow-x: auto; }}
  svg {{ display: block; }}
  footer {{ text-align: center; color: var(--muted); font-size: 0.74rem; margin-top: 32px; }}
  /* Coverage table */
  .coverage-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  .coverage-table th, .coverage-table td {{ padding: 6px 10px; text-align: center; border: 1px solid var(--border); }}
  .coverage-table thead th {{ background: var(--surface2); color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.04em; }}
  .coverage-table tbody td:first-child {{ text-align: left; color: var(--text); }}
  .cov-yes {{ color: #4ade80; font-weight: 700; }}
  .cov-no {{ color: #4b5563; }}
  .cov-miss {{ color: #374151; }}
  .dim {{ font-size: 0.72rem; color: var(--muted); font-weight: 400; }}
  .divider {{ border: none; border-top: 1px solid var(--border); margin: 8px 0; }}
</style>
</head>
<body>

<h1>🧠 Weekly Presence Dashboard</h1>
<p class="subtitle">{date_range} &nbsp;·&nbsp; {days_with_data}/7 days with data &nbsp;·&nbsp; Generated {generated_at}</p>

<!-- Weekly hero summary -->
<div class="card">
  <h2>Week at a Glance</h2>
  <div class="hero-grid">
    <div class="hero-item">
      <span class="hero-val" style="color:{avg_dps_colour}">{avg_dps_str}</span>
      <span class="hero-tier" style="color:{avg_dps_colour}">{avg_dps_tier}</span>
      <span class="hero-lbl">Avg DPS</span>
    </div>
    <div class="hero-item">
      <span class="hero-val" style="color:{avg_rec_colour}">{avg_rec_str}</span>
      <span class="hero-lbl">Avg Recovery</span>
    </div>
    <div class="hero-item">
      <span class="hero-val" style="color:{avg_fdi_colour}">{avg_fdi_str}</span>
      <span class="hero-lbl">Avg FDI</span>
    </div>
    <div class="hero-item">
      <span class="hero-val" style="color:{avg_cls_colour}">{avg_cls_str}</span>
      <span class="hero-lbl">Avg CLS</span>
    </div>
    <div class="hero-item">
      <span class="hero-val" style="color:#9ca3af">{mtg_str}</span>
      <span class="hero-lbl">Total Meetings</span>
    </div>
  </div>
  <hr class="divider"/>
  <div class="stat-grid">
    {best_day_html}
    {worst_day_html}
  </div>
</div>

<!-- DPS Trend -->
<div class="card">
  <h2>📈 Daily Presence Score (DPS) — 7-Day Trend</h2>
  <div class="chart-wrapper">
    {dps_chart}
  </div>
  <p style="font-size:0.72rem;color:var(--muted);margin-top:6px;">Green band ≥90 (exceptional) · Mid-green ≥80 (strong) · Dark ≥70 (good)</p>
</div>

<!-- Metric Grid -->
<div class="card">
  <h2>📐 Metrics Per Day (CLS / FDI / RAS)</h2>
  <div class="chart-wrapper">
    {metric_chart}
  </div>
  <p style="font-size:0.72rem;color:var(--muted);margin-top:6px;">CLS = cognitive load (low is good) · FDI = focus depth (high is good) · RAS = recovery alignment (high is good)</p>
</div>

<!-- Recovery -->
<div class="card">
  <h2>💚 WHOOP Recovery — 7-Day Arc</h2>
  <div class="chart-wrapper">
    {recovery_chart}
  </div>
  <p style="font-size:0.72rem;color:var(--muted);margin-top:6px;">Green ≥67% · Yellow 34–66% · Red &lt;34%</p>
</div>

<!-- Meetings -->
<div class="card">
  <h2>📅 Meeting Load Per Day</h2>
  <div class="chart-wrapper">
    {meeting_chart}
  </div>
  <p style="font-size:0.72rem;color:var(--muted);margin-top:6px;">Total: {mtg_str} across {days_with_data} days. Green &lt;1h · Yellow 1–2h · Orange 2–3h · Red &gt;3h</p>
</div>

<!-- Source coverage -->
<div class="card">
  <h2>🔌 Data Source Coverage</h2>
  {coverage_table}
</div>

<footer>
  Presence Tracker Weekly Dashboard · {date_range} · Generated {generated_at}
</footer>

</body>
</html>"""

    if output_path is None:
        output_path = DASHBOARD_DIR / f"week-{end_date_str}.html"

    output_path.write_text(html, encoding="utf-8")
    print(f"[weekly_dashboard] Generated: {output_path}")
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.

    Usage:
        python3 analysis/weekly_dashboard.py                  # Week ending on latest date
        python3 analysis/weekly_dashboard.py 2026-03-14       # Week ending on this date
        python3 analysis/weekly_dashboard.py --open           # Open in browser after
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate weekly HTML presence dashboard"
    )
    parser.add_argument("date", nargs="?", help="End date YYYY-MM-DD (default: latest)")
    parser.add_argument("--output", help="Output path override")
    parser.add_argument("--open", action="store_true", help="Open in browser after generation")
    args = parser.parse_args()

    if args.date:
        end_date_str = args.date
    else:
        dates = list_available_dates()
        if not dates:
            print("No data available.", file=sys.stderr)
            sys.exit(1)
        end_date_str = sorted(dates)[-1]

    out = generate_weekly_dashboard(
        end_date_str,
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Dashboard: {out}")

    if args.open:
        import subprocess
        subprocess.run(["open", str(out)])


if __name__ == "__main__":
    main()

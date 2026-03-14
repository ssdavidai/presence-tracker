#!/usr/bin/env python3
"""
Presence Tracker — Per-Day Terminal Report

A human-readable breakdown of any single day's presence data, printed
to the terminal.  Complements the HTML dashboard (browser) and CSV export
(spreadsheet) with a fast, ASCII-rich CLI view.

Usage:
    python3 scripts/report.py                   # Today
    python3 scripts/report.py 2026-03-14        # Specific date
    python3 scripts/report.py --json            # Machine-readable JSON
    python3 scripts/report.py --compact         # One-screen summary
    python3 scripts/report.py --windows         # Include per-window table
    python3 scripts/report.py --compare 7       # Compare vs N-day average
    python3 scripts/report.py --trend 14        # Multi-day trend table (last N days)

What it shows (full mode):
  - WHOOP physiological snapshot (recovery, HRV, sleep)
  - 5-metric summary bars (CLS, FDI, SDI, CSC, RAS)
  - Hourly CLS heatmap across the working day
  - Meeting timeline (calendar blocks)
  - Slack activity breakdown
  - Omi conversation summary (if available)
  - RescueTime productivity breakdown (if available)
  - Focus quality analysis (active windows, peak hour)
  - Comparison to recent average (with --compare N)

What --trend shows:
  A compact multi-day table — one row per day — showing DPS, CLS, FDI, RAS,
  WHOOP recovery, and CDI tier.  Lets David see at a glance how the week has
  been tracking, spot inflection points, and compare days without reading
  individual reports.

Design:
  - No external dependencies — pure stdlib + project modules
  - Works offline (reads from JSONL store only)
  - Readable in any terminal with Unicode support
  - Suitable for piping: --json for downstream processing

Examples:
    # Quick daily check
    python3 scripts/report.py --compact

    # Deep dive on a past day
    python3 scripts/report.py 2026-03-13 --windows

    # Machine-readable for Alfred/subagent consumption
    python3 scripts/report.py 2026-03-14 --json

    # See how today compares to the last 14 days
    python3 scripts/report.py --compare 14

    # Multi-day trend table for the last 14 days
    python3 scripts/report.py --trend 14
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.store import read_day, read_summary, list_available_dates


# ─── Colour / formatting helpers ──────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"


def _c(text: str, code: str) -> str:
    """Apply ANSI colour only when writing to a real TTY."""
    if sys.stdout.isatty():
        return f"{code}{text}{RESET}"
    return text


def _bar(value: float, width: int = 12, high_is_bad: bool = False) -> str:
    """Render a 0-1 score as a Unicode block bar with colour."""
    filled = max(0, min(width, round(value * width)))
    bar = "▓" * filled + "░" * (width - filled)
    if high_is_bad:
        colour = GREEN if value < 0.35 else (YELLOW if value < 0.65 else RED)
    else:
        colour = RED if value < 0.35 else (YELLOW if value < 0.65 else GREEN)
    return _c(bar, colour)


def _pct(value: Optional[float], suffix: str = "%") -> str:
    if value is None:
        return "N/A"
    return f"{round(value * 100)}{suffix}"


def _pct_raw(value: Optional[float], suffix: str = "%") -> str:
    """Format a value already in 0-100 range."""
    if value is None:
        return "N/A"
    return f"{round(value)}{suffix}"


def _delta_str(today: Optional[float], baseline: Optional[float],
               high_is_bad: bool = False) -> str:
    """Format a delta between today and baseline with directional colouring."""
    if today is None or baseline is None:
        return ""
    delta = today - baseline
    if abs(delta) < 0.01:
        return _c("→", DIM)
    sign = "+" if delta > 0 else ""
    if high_is_bad:
        colour = GREEN if delta < 0 else RED
    else:
        colour = GREEN if delta > 0 else RED
    return _c(f"{sign}{delta:.2f}", colour)


def _label_cls(cls: float) -> str:
    if cls < 0.20:
        return "light"
    elif cls < 0.40:
        return "moderate"
    elif cls < 0.60:
        return "heavy"
    else:
        return "intense"


def _label_fdi(fdi: float) -> str:
    if fdi < 0.30:
        return "fragmented"
    elif fdi < 0.55:
        return "interrupted"
    elif fdi < 0.75:
        return "solid"
    else:
        return "deep"


def _label_ras(ras: float) -> str:
    if ras < 0.30:
        return "misaligned"
    elif ras < 0.55:
        return "moderate"
    elif ras < 0.75:
        return "aligned"
    else:
        return "excellent"


def _tier_label(recovery: Optional[float]) -> str:
    if recovery is None:
        return "unknown"
    if recovery >= 80:
        return "Peak"
    elif recovery >= 67:
        return "Good"
    elif recovery >= 50:
        return "Moderate"
    elif recovery >= 33:
        return "Low"
    else:
        return "Recovery"


def _tier_colour(recovery: Optional[float]) -> str:
    if recovery is None:
        return DIM
    if recovery >= 67:
        return GREEN
    elif recovery >= 34:
        return YELLOW
    else:
        return RED


# ─── Data extraction ──────────────────────────────────────────────────────────

def _get_windows(date_str: str) -> list[dict]:
    return read_day(date_str)


def _working_windows(windows: list[dict]) -> list[dict]:
    return [w for w in windows if w["metadata"]["is_working_hours"]]


def _active_windows(windows: list[dict]) -> list[dict]:
    return [w for w in _working_windows(windows) if w["metadata"].get("is_active_window", False)]


def _hourly_cls(windows: list[dict]) -> dict[int, float]:
    """Compute mean CLS for each working hour."""
    hour_vals: dict[int, list[float]] = defaultdict(list)
    for w in _working_windows(windows):
        h = w["metadata"]["hour_of_day"]
        hour_vals[h].append(w["metrics"]["cognitive_load_score"])
    return {h: sum(v) / len(v) for h, v in hour_vals.items()}


def _meeting_blocks(windows: list[dict]) -> list[dict]:
    """Extract distinct meeting blocks from windows."""
    blocks = []
    in_block = False
    current: dict = {}
    for w in sorted(windows, key=lambda x: x["window_index"]):
        cal = w["calendar"]
        if cal["in_meeting"]:
            title = cal.get("meeting_title") or "(no title)"
            attendees = cal.get("meeting_attendees", 0)
            if not in_block or current.get("title") != title:
                if in_block:
                    blocks.append(current)
                current = {
                    "title": title,
                    "attendees": attendees,
                    "start": w["window_start"][:16],
                    "windows": 1,
                }
                in_block = True
            else:
                current["windows"] += 1
        else:
            if in_block:
                blocks.append(current)
                in_block = False
    if in_block:
        blocks.append(current)
    return blocks


def _omi_stats(windows: list[dict]) -> Optional[dict]:
    """Extract Omi conversation stats from windows, or None if no data."""
    omi_windows = [w for w in windows if w.get("omi") and w["omi"].get("conversation_active")]
    if not omi_windows:
        return None
    total_words = sum(w["omi"].get("word_count", 0) for w in omi_windows)
    total_speech_secs = sum(w["omi"].get("speech_seconds", 0.0) for w in omi_windows)
    total_sessions = sum(w["omi"].get("sessions_count", 0) for w in omi_windows)
    return {
        "conversation_windows": len(omi_windows),
        "total_sessions": total_sessions,
        "total_words": total_words,
        "total_speech_minutes": round(total_speech_secs / 60.0, 1),
    }


def _rt_stats(windows: list[dict]) -> Optional[dict]:
    """Extract RescueTime stats from windows, or None if no data."""
    rt_windows = [w for w in windows if w.get("rescuetime") and
                  w["rescuetime"].get("active_seconds", 0) > 0]
    if not rt_windows:
        return None
    focus_secs = sum(w["rescuetime"]["focus_seconds"] for w in rt_windows)
    distraction_secs = sum(w["rescuetime"]["distraction_seconds"] for w in rt_windows)
    active_secs = sum(w["rescuetime"]["active_seconds"] for w in rt_windows)
    productive_pct = round(100.0 * focus_secs / active_secs, 1) if active_secs > 0 else None
    return {
        "focus_minutes": round(focus_secs / 60, 1),
        "distraction_minutes": round(distraction_secs / 60, 1),
        "active_minutes": round(active_secs / 60, 1),
        "productive_pct": productive_pct,
    }


def _baseline(days: int) -> Optional[dict]:
    """Compute N-day average metrics from rolling summary (excluding today)."""
    summary = read_summary()
    all_days = sorted(summary.get("days", {}).keys())
    if not all_days:
        return None
    recent = all_days[-days:] if len(all_days) >= days else all_days
    cls_vals = []
    fdi_vals = []
    sdi_vals = []
    csc_vals = []
    ras_vals = []
    for d in recent:
        m = summary["days"][d].get("metrics_avg", {})
        if m.get("cognitive_load_score") is not None:
            cls_vals.append(m["cognitive_load_score"])
        if m.get("focus_depth_index") is not None:
            fdi_vals.append(m["focus_depth_index"])
        if m.get("social_drain_index") is not None:
            sdi_vals.append(m["social_drain_index"])
        if m.get("context_switch_cost") is not None:
            csc_vals.append(m["context_switch_cost"])
        if m.get("recovery_alignment_score") is not None:
            ras_vals.append(m["recovery_alignment_score"])
    def _avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "days": len(recent),
        "cls": _avg(cls_vals),
        "fdi": _avg(fdi_vals),
        "sdi": _avg(sdi_vals),
        "csc": _avg(csc_vals),
        "ras": _avg(ras_vals),
    }


# ─── Report building ──────────────────────────────────────────────────────────

def build_report(date_str: str, compare_days: int = 0) -> dict:
    """
    Build a structured report dict for a given date.

    Returns a dict suitable for JSON output or formatted rendering.
    Returns None if no data exists for the date.
    """
    windows = _get_windows(date_str)
    if not windows:
        return None

    working = _working_windows(windows)
    active = _active_windows(windows)

    first = windows[0]
    whoop = first.get("whoop", {})
    m_avg = {
        "cls": None, "fdi": None, "sdi": None, "csc": None, "ras": None,
    }
    if working:
        def _avg(vals):
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        m_avg = {
            "cls": _avg([w["metrics"]["cognitive_load_score"] for w in working]),
            "fdi": _avg([w["metrics"]["focus_depth_index"] for w in working]),
            "sdi": _avg([w["metrics"]["social_drain_index"] for w in active] if active else []),
            "csc": _avg([w["metrics"]["context_switch_cost"] for w in active] if active else []),
            "ras": _avg([w["metrics"]["recovery_alignment_score"] for w in windows]),
        }

    active_fdi = None
    peak_focus_hour = None
    peak_focus_fdi = None
    if active:
        active_fdi_vals = [w["metrics"]["focus_depth_index"] for w in active]
        active_fdi = round(sum(active_fdi_vals) / len(active_fdi_vals), 4)
        hour_fdi: dict[int, list] = defaultdict(list)
        for w in active:
            hour_fdi[w["metadata"]["hour_of_day"]].append(w["metrics"]["focus_depth_index"])
        for h, vals in hour_fdi.items():
            if len(vals) >= 2:
                h_avg = sum(vals) / len(vals)
                if peak_focus_fdi is None or h_avg > peak_focus_fdi:
                    peak_focus_fdi = round(h_avg, 4)
                    peak_focus_hour = h

    # Slack
    total_sent = sum(w["slack"]["messages_sent"] for w in windows)
    total_recv = sum(w["slack"]["messages_received"] for w in windows)

    # Calendar
    meeting_windows = sum(1 for w in working if w["calendar"]["in_meeting"])
    meeting_minutes = meeting_windows * 15

    return {
        "date": date_str,
        "day_of_week": datetime.strptime(date_str, "%Y-%m-%d").strftime("%A"),
        "whoop": {
            "recovery_score": whoop.get("recovery_score"),
            "hrv_rmssd_milli": whoop.get("hrv_rmssd_milli"),
            "resting_heart_rate": whoop.get("resting_heart_rate"),
            "sleep_hours": whoop.get("sleep_hours"),
            "sleep_performance": whoop.get("sleep_performance"),
            "strain": whoop.get("strain"),
            "spo2_percentage": whoop.get("spo2_percentage"),
        },
        "metrics": m_avg,
        "focus": {
            "active_fdi": active_fdi,
            "active_windows": len(active),
            "total_working_windows": len(working),
            "peak_focus_hour": peak_focus_hour,
            "peak_focus_fdi": peak_focus_fdi,
        },
        "slack": {
            "messages_sent": total_sent,
            "messages_received": total_recv,
            "total_messages": total_sent + total_recv,
        },
        "calendar": {
            "meeting_minutes": meeting_minutes,
            "meeting_windows": meeting_windows,
            "blocks": _meeting_blocks(windows),
        },
        "omi": _omi_stats(windows),
        "rescuetime": _rt_stats(windows),
        "sources_available": list(sorted(set(
            src
            for w in windows
            for src in w["metadata"].get("sources_available", [])
        ))),
        "hourly_cls": _hourly_cls(windows),
        "baseline": _baseline(compare_days) if compare_days > 0 else None,
        # Daily Presence Score (DPS) — single composite 0–100 score
        "presence_score": _compute_dps(windows),
    }


def _compute_dps(windows: list[dict]) -> Optional[dict]:
    """Compute Daily Presence Score; returns None on failure."""
    try:
        from analysis.presence_score import compute_presence_score, format_presence_score_block
        score = compute_presence_score(windows)
        if not score.is_meaningful:
            return None
        return {
            "dps": score.dps,
            "tier": score.tier,
            "components": score.components,
            "block": format_presence_score_block(score),
        }
    except Exception:
        return None


# ─── Terminal rendering ───────────────────────────────────────────────────────

def _heatmap_line(hourly_cls: dict[int, float], start: int = 7, end: int = 22) -> str:
    """
    Render an hourly CLS heatmap as a single Unicode block string.

    Each hour maps to one character:
      ░ = very light (< 0.10)
      ▒ = light (0.10–0.25)
      ▓ = moderate (0.25–0.50)
      █ = heavy (≥ 0.50)
    """
    chars = []
    for h in range(start, end + 1):
        val = hourly_cls.get(h)
        if val is None:
            chars.append("·")
        elif val < 0.10:
            chars.append("░")
        elif val < 0.25:
            chars.append("▒")
        elif val < 0.50:
            chars.append("▓")
        else:
            chars.append("█")
    return "".join(chars)


def print_compact(report: dict) -> None:
    """One-screen summary — metrics + heatmap only."""
    d = report["date"]
    dow = report["day_of_week"]
    m = report["metrics"]
    w = report["whoop"]
    f = report["focus"]

    recovery = w.get("recovery_score")
    tier = _tier_label(recovery)
    tier_colour = _tier_colour(recovery)

    print(_c(f"  {dow}, {d}", BOLD))
    print()

    # WHOOP
    hrv = w.get("hrv_rmssd_milli")
    sleep = w.get("sleep_hours")
    print(f"  Recovery  {_c(f'{tier} ({_pct_raw(recovery)})', tier_colour)}  "
          f"HRV {hrv:.0f}ms  Sleep {sleep:.1f}h" if hrv and sleep else
          f"  Recovery  {_c(f'{tier} ({_pct_raw(recovery)})', tier_colour)}")

    # Metrics
    if m["cls"] is not None:
        print(f"  CLS  {_bar(m['cls'], 10, high_is_bad=True)} {_pct(m['cls'])}  ({_label_cls(m['cls'])})")
    if f["active_fdi"] is not None:
        print(f"  FDI  {_bar(f['active_fdi'], 10)} {_pct(f['active_fdi'])}  ({_label_fdi(f['active_fdi'])})")
    if m["ras"] is not None:
        print(f"  RAS  {_bar(m['ras'], 10)} {_pct(m['ras'])}  ({_label_ras(m['ras'])})")

    # Daily Presence Score
    dps_data = report.get("presence_score")
    if dps_data:
        from analysis.presence_score import _TIER_EMOJI
        emoji = _TIER_EMOJI.get(dps_data["tier"], "⚪")
        print(f"  DPS  {emoji} {dps_data['dps']:.0f}/100  ({dps_data['tier']})")

    # Heatmap
    hmap = _heatmap_line(report["hourly_cls"])
    print(f"\n  7am {hmap} 10pm")
    print()


def print_full(report: dict, show_windows: bool = False) -> None:
    """Full detailed report."""
    d = report["date"]
    dow = report["day_of_week"]

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(_c(f"  Presence Report — {dow}, {d}", BOLD))
    print(_c("  " + "─" * 50, DIM))
    print()

    # ── WHOOP ─────────────────────────────────────────────────────────────────
    w = report["whoop"]
    recovery = w.get("recovery_score")
    hrv = w.get("hrv_rmssd_milli")
    rhr = w.get("resting_heart_rate")
    sleep = w.get("sleep_hours")
    sleep_perf = w.get("sleep_performance")
    strain = w.get("strain")
    spo2 = w.get("spo2_percentage")
    tier = _tier_label(recovery)
    tier_colour = _tier_colour(recovery)

    print(_c("  ① WHOOP", BOLD))
    rec_bar = _bar(recovery / 100 if recovery else 0, 12,
                   high_is_bad=False)
    print(f"  Recovery   {rec_bar} {_c(f'{_pct_raw(recovery)}  {tier}', tier_colour)}")
    if hrv is not None:
        hrv_colour = GREEN if hrv >= 60 else (YELLOW if hrv >= 40 else RED)
        print(f"  HRV        {_c(f'{hrv:.1f} ms', hrv_colour)}")
    if rhr is not None:
        print(f"  RHR        {rhr:.0f} bpm")
    if sleep is not None:
        sleep_colour = GREEN if sleep >= 7.5 else (YELLOW if sleep >= 6 else RED)
        print(f"  Sleep      {_c(f'{sleep:.1f}h', sleep_colour)}", end="")
        if sleep_perf is not None:
            print(f"  ({sleep_perf:.0f}% performance)", end="")
        print()
    if strain is not None:
        print(f"  Strain     {strain:.1f}")
    if spo2 is not None:
        print(f"  SpO₂       {spo2:.1f}%")
    print()

    # ── Cognitive Metrics ─────────────────────────────────────────────────────
    m = report["metrics"]
    f = report["focus"]
    baseline = report.get("baseline")

    print(_c("  ② Cognitive Metrics", BOLD))

    def _metric_line(label: str, value: Optional[float], high_is_bad: bool,
                     base_val: Optional[float] = None, note: str = "") -> None:
        if value is None:
            print(f"  {label:<6} N/A")
            return
        bar = _bar(value, 12, high_is_bad=high_is_bad)
        pct_str = f"{round(value * 100):>3}%"
        delta = ""
        if base_val is not None:
            delta = "  " + _delta_str(value, base_val, high_is_bad=high_is_bad)
        note_str = f"  {_c(note, DIM)}" if note else ""
        print(f"  {label:<6} {bar}  {pct_str}{delta}{note_str}")

    _metric_line("CLS", m["cls"], high_is_bad=True,
                 base_val=baseline["cls"] if baseline else None,
                 note=_label_cls(m["cls"]) if m["cls"] is not None else "")
    # Show active FDI as the meaningful signal
    fdi_to_show = f["active_fdi"] if f["active_fdi"] is not None else m["fdi"]
    fdi_note = _label_fdi(fdi_to_show) if fdi_to_show is not None else ""
    if f["active_fdi"] is not None and m["fdi"] is not None and abs(f["active_fdi"] - m["fdi"]) > 0.05:
        fdi_note += " (active windows)"
    _metric_line("FDI", fdi_to_show, high_is_bad=False,
                 base_val=baseline["fdi"] if baseline else None,
                 note=fdi_note)
    _metric_line("SDI", m["sdi"], high_is_bad=True,
                 base_val=baseline["sdi"] if baseline else None)
    _metric_line("CSC", m["csc"], high_is_bad=True,
                 base_val=baseline["csc"] if baseline else None)
    _metric_line("RAS", m["ras"], high_is_bad=False,
                 base_val=baseline["ras"] if baseline else None,
                 note=_label_ras(m["ras"]) if m["ras"] is not None else "")

    if baseline:
        print(_c(f"\n  (Δ vs {baseline['days']}-day average)", DIM))

    print()

    # ── Daily Presence Score ──────────────────────────────────────────────────
    # Composite 0–100 score: the cognitive equivalent of WHOOP's day strain.
    dps_data = report.get("presence_score")
    if dps_data and dps_data.get("block"):
        print(_c("  ② DPS — Daily Presence Score", BOLD))
        for line in dps_data["block"].split("\n"):
            print(f"  {line}")
        print()

    # ── Hourly Heatmap ────────────────────────────────────────────────────────
    print(_c("  ③ Cognitive Load — Hourly", BOLD))
    hmap = _heatmap_line(report["hourly_cls"])
    print(f"  7am  {hmap}  10pm")
    print(_c("        ░=light  ▒=mild  ▓=moderate  █=heavy", DIM))
    print()

    # ── Focus Quality ─────────────────────────────────────────────────────────
    print(_c("  ④ Focus Quality", BOLD))
    print(f"  Active windows   {f['active_windows']} of {f['total_working_windows']} working-hour windows")
    if f["active_fdi"] is not None:
        print(f"  Active FDI       {_pct(f['active_fdi'])}  {_bar(f['active_fdi'], 10)}")
    if f["peak_focus_hour"] is not None and f["peak_focus_fdi"] is not None:
        t_start = f"{f['peak_focus_hour']:02d}:00"
        t_end = f"{f['peak_focus_hour']+1:02d}:00"
        print(f"  Peak focus       {t_start}–{t_end}  ({_pct(f['peak_focus_fdi'])} FDI)")
    print()

    # ── Calendar ─────────────────────────────────────────────────────────────
    cal = report["calendar"]
    if cal["meeting_minutes"] > 0:
        print(_c("  ⑤ Calendar", BOLD))
        print(f"  Meetings         {cal['meeting_minutes']} min  ({cal['meeting_windows']} windows)")
        for b in cal["blocks"]:
            dur = b["windows"] * 15
            ppl = f"  {b['attendees']} attendees" if b["attendees"] > 1 else ""
            print(f"  {b['start']}  {b['title'][:35]}  ({dur} min){ppl}")
        print()
    else:
        print(_c("  ⑤ Calendar", BOLD))
        print(_c("  No meetings today", DIM))
        print()

    # ── Slack ─────────────────────────────────────────────────────────────────
    s = report["slack"]
    print(_c("  ⑥ Slack", BOLD))
    print(f"  Sent      {s['messages_sent']}")
    print(f"  Received  {s['messages_received']}")
    print(f"  Total     {s['total_messages']}")
    print()

    # ── Omi ──────────────────────────────────────────────────────────────────
    omi = report.get("omi")
    if omi:
        print(_c("  ⑦ Omi Conversations", BOLD))
        print(f"  Sessions   {omi['total_sessions']}")
        print(f"  Windows    {omi['conversation_windows']}")
        print(f"  Words      {omi['total_words']:,}")
        print(f"  Speaking   {omi['total_speech_minutes']:.0f} min")
        print()

    # ── RescueTime ────────────────────────────────────────────────────────────
    rt = report.get("rescuetime")
    if rt:
        section_num = "⑧" if omi else "⑦"
        print(_c(f"  {section_num} RescueTime", BOLD))
        print(f"  Computer   {rt['active_minutes']:.0f} min active")
        print(f"  Focus      {rt['focus_minutes']:.0f} min")
        print(f"  Distract   {rt['distraction_minutes']:.0f} min")
        if rt.get("productive_pct") is not None:
            pct_val = rt["productive_pct"] / 100
            print(f"  Productiv  {_bar(pct_val, 10)} {rt['productive_pct']:.0f}%")
        print()

    # ── Sources ───────────────────────────────────────────────────────────────
    sources = report.get("sources_available", [])
    print(_c("  Sources: ", DIM) + _c(", ".join(sources) if sources else "none", DIM))
    print()

    # ── Per-window table ─────────────────────────────────────────────────────
    if show_windows:
        _print_window_table(_get_windows(report["date"]))


def _print_window_table(windows: list[dict]) -> None:
    """Print a per-window detail table for active working-hour windows."""
    active = [w for w in windows
              if w["metadata"]["is_working_hours"]
              and w["metadata"].get("is_active_window", False)]

    if not active:
        print(_c("  No active windows to display.", DIM))
        return

    print(_c("  ── Per-Window Detail (active working hours) ──────────", BOLD))
    print(_c("  Time    CLS  FDI  SDI  Mtg  Slack  Flags", DIM))
    print(_c("  " + "─" * 52, DIM))

    for w in sorted(active, key=lambda x: x["window_index"]):
        t = w["window_start"][11:16]
        m = w["metrics"]
        cal = w["calendar"]
        sl = w["slack"]
        cls_s = f"{m['cognitive_load_score']:.2f}"
        fdi_s = f"{m['focus_depth_index']:.2f}"
        sdi_s = f"{m['social_drain_index']:.2f}"
        mtg = "M" if cal["in_meeting"] else " "
        slack_s = f"{sl.get('total_messages', 0):>3}"
        flags = []
        if cal["in_meeting"] and cal.get("meeting_attendees", 0) > 1:
            flags.append("social")
        if w.get("omi") and w["omi"].get("conversation_active"):
            flags.append("omi")
        if w.get("rescuetime") and w["rescuetime"].get("active_seconds", 0) > 0:
            flags.append("rt")
        flag_str = " ".join(flags)
        print(f"  {t}  {cls_s}  {fdi_s}  {sdi_s}  {mtg}    {slack_s}  {flag_str}")

    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

def _dps_emoji(dps: Optional[float]) -> str:
    """Return a compact emoji + score string for the trend table."""
    if dps is None:
        return "  — "
    if dps >= 85:
        return f"🌟{dps:4.0f}"
    if dps >= 75:
        return f"✅{dps:4.0f}"
    if dps >= 60:
        return f"🟡{dps:4.0f}"
    if dps >= 45:
        return f"🟠{dps:4.0f}"
    return f"🔴{dps:4.0f}"


def _cdi_tier_short(date_str: str) -> str:
    """Return a short CDI tier label for the trend table, or '—' if unavailable."""
    try:
        from analysis.cognitive_debt import compute_cdi
        debt = compute_cdi(date_str)
        if debt is None:
            return "—"
        tier = debt.tier
        tier_map = {
            "surplus":  "surplus ",
            "balanced": "balanced",
            "loading":  "loading ",
            "fatigued": "fatigued",
            "critical": "critical",
        }
        return tier_map.get(tier, tier[:8])
    except Exception:
        return "—"


def _sparkbar(value: Optional[float], width: int = 6, high_is_bad: bool = False) -> str:
    """
    Compact progress bar for use in the trend table.

    Uses half-width to keep the table narrow.  Filled chars represent
    the metric value on a 0–1 scale.
    """
    if value is None:
        return "·" * width
    filled = round(value * width)
    filled = max(0, min(width, filled))
    bar = "▓" * filled + "░" * (width - filled)
    return bar


def build_trend_rows(days: int) -> list[dict]:
    """
    Build a list of summary dicts for the last `days` available days.

    Each dict contains the key values needed for the trend table.
    Only days with actual data are included (sparse history is fine —
    the table shows dates in ascending order with gaps filled by
    whatever data exists).
    """
    available = list_available_dates()
    if not available:
        return []

    # Take the last `days` available dates
    selected = available[-days:] if len(available) >= days else available

    rows = []
    for date_str in selected:
        report = build_report(date_str)
        if report is None:
            continue

        m = report["metrics"]
        w = report["whoop"]
        f = report["focus"]
        dps_data = report.get("presence_score")
        dps = dps_data["dps"] if dps_data else None

        rows.append({
            "date": date_str,
            "dow": datetime.strptime(date_str, "%Y-%m-%d").strftime("%a"),
            "dps": dps,
            "cls": m.get("cls"),
            "fdi": f.get("active_fdi") or m.get("fdi"),
            "ras": m.get("ras"),
            "recovery": w.get("recovery_score"),
            "hrv": w.get("hrv_rmssd_milli"),
            "sleep": w.get("sleep_hours"),
            "meeting_mins": report["calendar"].get("meeting_minutes", 0),
            "slack_sent": report["slack"].get("messages_sent", 0),
        })

    return rows


def print_trend(days: int = 14) -> None:
    """
    Print a compact multi-day trend table.

    Columns: Date  Day  DPS  CLS▓  FDI▓  RAS▓  Recov  HRV  Sleep  Mtg  CDI-tier

    DPS is colour-coded; metrics have compact bar representations.
    One row per day, oldest first.  Empty cells shown as '—'.

    CDI is computed on-the-fly for each day (it's a lookback metric, so
    the value for a given date reflects the accumulated debt *up to that date*).
    """
    available = list_available_dates()
    if not available:
        print("  No data available.", file=sys.stderr)
        return

    rows = build_trend_rows(days)
    if not rows:
        print("  No data for the requested range.", file=sys.stderr)
        return

    # Compute CDI for each row date (can be slow for many days — warn if > 14)
    for row in rows:
        row["cdi_tier"] = _cdi_tier_short(row["date"])

    # Header
    date_range = f"{rows[0]['date']} → {rows[-1]['date']}"
    print()
    print(_c(f"  Presence Tracker — {days}-Day Trend  ", BOLD) + _c(f"({date_range})", DIM))
    print()

    # Column widths:  date(10)  dow(3)  dps(6)  cls(8)  fdi(8)  ras(8)  recov(6)  hrv(5)  sleep(5)  mtg(5)  cdi(9)
    hdr = (
        f"  {'Date':<10}  {'Day':<3}  "
        f"{'DPS':>6}  {'CLS':>8}  {'FDI':>8}  {'RAS':>8}  "
        f"{'Recov':>5}  {'HRV':>5}  {'Sleep':>5}  {'Mtg':>4}  "
        f"{'CDI':<9}"
    )
    print(_c(hdr, DIM))
    print(_c("  " + "─" * 85, DIM))

    for row in rows:
        date_s = row["date"]
        dow_s  = row["dow"]

        # DPS — emoji + number, colour-coded
        dps_val = row["dps"]
        dps_str = _dps_emoji(dps_val)

        # Metric bars + values
        def _col(val: Optional[float], high_is_bad: bool = False) -> str:
            if val is None:
                return f"{'—':>8}"
            bar = _sparkbar(val, 5, high_is_bad=high_is_bad)
            return f"{bar} {val*100:3.0f}%"

        cls_col  = _col(row["cls"],  high_is_bad=True)
        fdi_col  = _col(row["fdi"])
        ras_col  = _col(row["ras"])

        # WHOOP
        rec  = row["recovery"]
        hrv  = row["hrv"]
        slp  = row["sleep"]
        rec_s  = f"{rec:3.0f}%" if rec  is not None else "  — "
        hrv_s  = f"{hrv:3.0f}"  if hrv  is not None else "  —"
        slp_s  = f"{slp:3.1f}h" if slp  is not None else "  —"

        # Meeting load
        mtg = row.get("meeting_mins", 0) or 0
        mtg_s = f"{mtg:3.0f}m" if mtg > 0 else "   —"

        # CDI
        cdi_s = row.get("cdi_tier", "—")

        # Colour the row by DPS tier
        if dps_val is not None:
            if dps_val >= 85:
                row_colour = GREEN
            elif dps_val >= 60:
                row_colour = ""  # plain
            elif dps_val >= 45:
                row_colour = YELLOW
            else:
                row_colour = RED
        else:
            row_colour = ""

        line = (
            f"  {date_s:<10}  {dow_s:<3}  "
            f"{dps_str:>6}  {cls_col}  {fdi_col}  {ras_col}  "
            f"{rec_s:>5}  {hrv_s:>5}  {slp_s:>5}  {mtg_s:>4}  "
            f"{cdi_s:<9}"
        )
        if row_colour:
            print(_c(line, row_colour))
        else:
            print(line)

    # Summary footer: averages across shown days
    print(_c("  " + "─" * 85, DIM))
    cls_vals  = [r["cls"]      for r in rows if r["cls"]      is not None]
    fdi_vals  = [r["fdi"]      for r in rows if r["fdi"]      is not None]
    ras_vals  = [r["ras"]      for r in rows if r["ras"]      is not None]
    dps_vals  = [r["dps"]      for r in rows if r["dps"]      is not None]
    rec_vals  = [r["recovery"] for r in rows if r["recovery"] is not None]

    def _mean(vals: list) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    avg_dps = _mean(dps_vals)
    avg_cls = _mean(cls_vals)
    avg_fdi = _mean(fdi_vals)
    avg_ras = _mean(ras_vals)
    avg_rec = _mean(rec_vals)

    avg_dps_s = f"{avg_dps:.0f}" if avg_dps is not None else "—"
    avg_cls_s = f"{avg_cls*100:.0f}%" if avg_cls is not None else "—"
    avg_fdi_s = f"{avg_fdi*100:.0f}%" if avg_fdi is not None else "—"
    avg_ras_s = f"{avg_ras*100:.0f}%" if avg_ras is not None else "—"
    avg_rec_s = f"{avg_rec:.0f}%" if avg_rec is not None else "—"

    summary = (
        f"  {'Avg':>10}  {'':3}  "
        f"{avg_dps_s:>6}  {'':>8}  {'':>8}  {'':>8}  "
        f"{'CLS '+avg_cls_s:>5}  {'FDI '+avg_fdi_s:>8}  {'RAS '+avg_ras_s:>9}  "
        f"{'Rec '+avg_rec_s:>6}"
    )
    # Simpler summary line
    print(_c(
        f"  {'Averages':>10}   DPS {avg_dps_s:>3} | CLS {avg_cls_s:>4} | "
        f"FDI {avg_fdi_s:>4} | RAS {avg_ras_s:>4} | Recovery {avg_rec_s:>4}",
        DIM
    ))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Presence Tracker — Per-Day Terminal Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "date",
        nargs="?",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date to report on (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON instead of formatted text",
    )
    parser.add_argument(
        "--compact", "-c", action="store_true",
        help="One-screen compact summary (metrics + heatmap only)",
    )
    parser.add_argument(
        "--windows", "-w", action="store_true",
        help="Include per-window detail table (active working-hour windows)",
    )
    parser.add_argument(
        "--compare", "-n", type=int, default=0, metavar="DAYS",
        help="Compare today's metrics vs N-day rolling average",
    )
    parser.add_argument(
        "--trend", "-t", type=int, default=0, metavar="DAYS",
        help="Show compact multi-day trend table for the last N available days "
             "(default: all available if -t given without a value)",
    )
    args = parser.parse_args()

    # ── Trend mode — no date argument needed ─────────────────────────────────
    if args.trend:
        print_trend(days=args.trend)
        return

    # Validate date
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date: {args.date}. Use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    # Check data exists
    available = list_available_dates()
    if args.date not in available:
        print(f"No data for {args.date}.", file=sys.stderr)
        if available:
            print(f"Available: {available[0]} → {available[-1]}", file=sys.stderr)
        sys.exit(1)

    report = build_report(args.date, compare_days=args.compare)
    if report is None:
        print(f"Failed to build report for {args.date}.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    elif args.compact:
        print_compact(report)
    else:
        print_full(report, show_windows=args.windows)


if __name__ == "__main__":
    main()

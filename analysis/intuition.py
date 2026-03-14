"""
Presence Tracker — Alfred Intuition Layer

LLM-powered weekly pattern analysis.
Runs on Sunday evenings, synthesizes the week's data,
and delivers a Slack report to David.

Uses the OpenClaw gateway to spawn a subagent.

v1.3 — Rich pattern analytics:
  The analysis prompt now includes pre-computed cross-metric statistics
  (hourly CLS/FDI patterns, HRV-vs-load correlation, day-of-week profile,
  meeting-load impact, best focus hours) so the LLM can write a report
  grounded in actual patterns rather than just daily aggregate numbers.
  Previously windows_sample was collected but never given to the LLM —
  this is now fixed.
"""

import json
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GATEWAY_URL, GATEWAY_TOKEN, SLACK_DM_CHANNEL
from engine.store import get_recent_summaries, read_range, list_available_dates


def _gateway_invoke(tool: str, args: dict, timeout: int = 120) -> dict:
    """Call an OpenClaw tool via the gateway."""
    headers = {
        "Authorization": f"Bearer {GATEWAY_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({"tool": tool, "args": args}).encode()
    req = urllib.request.Request(
        f"{GATEWAY_URL}/tools/invoke",
        data=payload,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ─── Pattern analytics ────────────────────────────────────────────────────────

def compute_hourly_patterns(all_windows: list[dict]) -> dict:
    """
    Aggregate working-hours windows by hour of day and compute mean CLS and FDI.

    Returns a dict mapping hour → {avg_cls, avg_fdi, window_count} for
    hours that have at least one active window.
    """
    from collections import defaultdict

    hour_cls: dict[int, list[float]] = defaultdict(list)
    hour_fdi: dict[int, list[float]] = defaultdict(list)

    for w in all_windows:
        if not w["metadata"]["is_working_hours"]:
            continue
        # Only count active windows (meeting or Slack) so idle hours don't dilute
        if not w["calendar"]["in_meeting"] and w["slack"]["total_messages"] == 0:
            continue
        h = w["metadata"]["hour_of_day"]
        hour_cls[h].append(w["metrics"]["cognitive_load_score"])
        hour_fdi[h].append(w["metrics"]["focus_depth_index"])

    result = {}
    for h in sorted(set(list(hour_cls.keys()) + list(hour_fdi.keys()))):
        cls_vals = hour_cls.get(h, [])
        fdi_vals = hour_fdi.get(h, [])
        result[h] = {
            "avg_cls": round(sum(cls_vals) / len(cls_vals), 3) if cls_vals else None,
            "avg_fdi": round(sum(fdi_vals) / len(fdi_vals), 3) if fdi_vals else None,
            "active_windows": len(cls_vals),
        }
    return result


def compute_hrv_cls_correlation(all_windows: list[dict]) -> dict:
    """
    Bin days by HRV quartile and compute mean CLS per quartile.

    Returns a dict with:
    - quartile_cls: {low, medium, high} → avg_cls
    - correlation_direction: 'inverse' | 'flat' | 'positive'
    - days_with_low_hrv: count
    - days_with_high_hrv: count
    - note: human-readable description
    """
    # Collect (hrv, avg_cls_for_day) pairs
    # Use only days where both signals are present
    from engine.store import read_day

    daily_pairs: list[tuple[float, float]] = []
    seen_dates: set[str] = set()

    for w in all_windows:
        date = w["date"]
        if date in seen_dates:
            continue
        seen_dates.add(date)

        hrv = w["whoop"].get("hrv_rmssd_milli")
        if hrv is None:
            continue

        # Get all working-hour CLS for this day from the all_windows list
        day_cls = [
            x["metrics"]["cognitive_load_score"]
            for x in all_windows
            if x["date"] == date and x["metadata"]["is_working_hours"]
        ]
        if not day_cls:
            continue
        avg_cls = sum(day_cls) / len(day_cls)
        daily_pairs.append((hrv, avg_cls))

    if len(daily_pairs) < 2:
        return {
            "insufficient_data": True,
            "note": f"Only {len(daily_pairs)} day(s) with HRV data — need at least 2 for correlation",
        }

    # Split into low/medium/high HRV thirds
    hrv_vals = sorted(p[0] for p in daily_pairs)
    n = len(hrv_vals)
    low_threshold = hrv_vals[n // 3]
    high_threshold = hrv_vals[(2 * n) // 3]

    low_cls = [cls for hrv, cls in daily_pairs if hrv <= low_threshold]
    mid_cls = [cls for hrv, cls in daily_pairs if low_threshold < hrv <= high_threshold]
    high_cls = [cls for hrv, cls in daily_pairs if hrv > high_threshold]

    def _mean(vals: list) -> Optional[float]:
        return round(sum(vals) / len(vals), 3) if vals else None

    low_avg = _mean(low_cls)
    mid_avg = _mean(mid_cls)
    high_avg = _mean(high_cls)

    # Determine direction
    direction = "flat"
    if low_avg is not None and high_avg is not None:
        diff = high_avg - low_avg
        if diff < -0.05:
            direction = "inverse"  # High HRV → lower CLS (good)
        elif diff > 0.05:
            direction = "positive"  # High HRV → higher CLS (unexpected)

    days_low_hrv = sum(1 for hrv, _ in daily_pairs if hrv < 60)
    days_high_hrv = sum(1 for hrv, _ in daily_pairs if hrv >= 75)

    note = ""
    if direction == "inverse":
        note = "HRV and cognitive load are inversely correlated — lower HRV days tend to push you harder."
    elif direction == "positive":
        note = "Unusually, higher HRV days showed higher load — may indicate you schedule more on high-readiness days."
    else:
        note = "No strong HRV-load correlation this week."

    return {
        "insufficient_data": False,
        "quartile_cls": {
            "low_hrv": low_avg,
            "mid_hrv": mid_avg,
            "high_hrv": high_avg,
        },
        "correlation_direction": direction,
        "days_with_low_hrv": days_low_hrv,
        "days_with_high_hrv": days_high_hrv,
        "note": note,
    }


def compute_day_of_week_profile(all_windows: list[dict]) -> dict:
    """
    For each day in the window set, compute a profile: total meeting minutes,
    avg CLS, peak CLS, avg FDI (active only).

    Returns a list of day dicts sorted chronologically.
    """
    by_date: dict[str, list[dict]] = defaultdict(list)
    for w in all_windows:
        by_date[w["date"]].append(w)

    profiles = []
    for date_str in sorted(by_date):
        day_windows = by_date[date_str]
        working = [w for w in day_windows if w["metadata"]["is_working_hours"]]
        active = [w for w in working if w["calendar"]["in_meeting"] or w["slack"]["total_messages"] > 0]

        def _avg(vals: list) -> Optional[float]:
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        cls_vals = [w["metrics"]["cognitive_load_score"] for w in working]
        fdi_active = [w["metrics"]["focus_depth_index"] for w in active]
        ras_vals = [w["metrics"]["recovery_alignment_score"] for w in day_windows]

        meeting_windows = [w for w in working if w["calendar"]["in_meeting"]]
        meeting_minutes = len(meeting_windows) * 15

        whoop = day_windows[0]["whoop"] if day_windows else {}
        dow = day_windows[0]["metadata"]["day_of_week"] if day_windows else "?"

        profiles.append({
            "date": date_str,
            "day_of_week": dow,
            "recovery_score": whoop.get("recovery_score"),
            "hrv_rmssd_milli": whoop.get("hrv_rmssd_milli"),
            "avg_cls": _avg(cls_vals),
            "peak_cls": max(cls_vals) if cls_vals else None,
            "avg_fdi_active": _avg(fdi_active),
            "avg_ras": _avg(ras_vals),
            "total_meeting_minutes": meeting_minutes,
            "active_windows": len(active),
            "slack_sent": sum(w["slack"]["messages_sent"] for w in day_windows),
            "slack_received": sum(w["slack"]["messages_received"] for w in day_windows),
        })
    return profiles


def compute_focus_window_analysis(all_windows: list[dict]) -> dict:
    """
    Identify the best and worst hours for deep focus across the week.

    Returns:
    - best_focus_hours: list of (hour, avg_fdi) sorted descending
    - worst_focus_hours: list of (hour, avg_fdi) sorted ascending
    - peak_load_hours: list of (hour, avg_cls) sorted descending
    - recommended_deep_work_hours: hours where FDI > 0.7 consistently
    """
    hourly = compute_hourly_patterns(all_windows)
    if not hourly:
        return {"insufficient_data": True}

    fdi_by_hour = [
        (h, v["avg_fdi"])
        for h, v in hourly.items()
        if v["avg_fdi"] is not None and v["active_windows"] >= 1
    ]
    cls_by_hour = [
        (h, v["avg_cls"])
        for h, v in hourly.items()
        if v["avg_cls"] is not None and v["active_windows"] >= 1
    ]

    fdi_by_hour.sort(key=lambda x: x[1], reverse=True)
    cls_by_hour.sort(key=lambda x: x[1], reverse=True)

    recommended = [
        f"{h:02d}:00–{h+1:02d}:00"
        for h, fdi in fdi_by_hour
        if fdi >= 0.65
    ]

    return {
        "best_focus_hours": [
            {"hour": f"{h:02d}:00", "avg_fdi": fdi}
            for h, fdi in fdi_by_hour[:4]
        ],
        "worst_focus_hours": [
            {"hour": f"{h:02d}:00", "avg_fdi": fdi}
            for h, fdi in reversed(fdi_by_hour[-3:])
        ],
        "peak_load_hours": [
            {"hour": f"{h:02d}:00", "avg_cls": cls}
            for h, cls in cls_by_hour[:4]
        ],
        "recommended_deep_work_hours": recommended[:5],
    }


def compute_meeting_impact(all_windows: list[dict]) -> dict:
    """
    Quantify how meetings affect cognitive load and focus quality.

    Returns:
    - avg_cls_in_meeting: mean CLS when calendar shows a meeting
    - avg_cls_not_in_meeting: mean CLS in non-meeting working windows
    - avg_fdi_in_meeting: mean FDI during meetings
    - avg_fdi_not_in_meeting: mean FDI in focused (non-meeting) working windows
    - meeting_cls_premium: how much higher CLS is during meetings
    - total_meeting_minutes_week: total calendar time in meetings
    - meeting_fragmentation_score: % of working windows that are in short (<30min) meetings
    """
    working = [w for w in all_windows if w["metadata"]["is_working_hours"]]
    meeting_windows = [w for w in working if w["calendar"]["in_meeting"]]
    non_meeting_working = [
        w for w in working
        if not w["calendar"]["in_meeting"] and w["slack"]["total_messages"] > 0
    ]

    def _avg_metric(windows_list: list, metric: str) -> Optional[float]:
        vals = [w["metrics"][metric] for w in windows_list]
        return round(sum(vals) / len(vals), 3) if vals else None

    avg_cls_meeting = _avg_metric(meeting_windows, "cognitive_load_score")
    avg_cls_focused = _avg_metric(non_meeting_working, "cognitive_load_score")
    avg_fdi_meeting = _avg_metric(meeting_windows, "focus_depth_index")
    avg_fdi_focused = _avg_metric(non_meeting_working, "focus_depth_index")

    cls_premium = None
    if avg_cls_meeting is not None and avg_cls_focused is not None:
        cls_premium = round(avg_cls_meeting - avg_cls_focused, 3)

    # Short meeting fragmentation
    short_meeting_windows = [
        w for w in meeting_windows
        if w["calendar"]["meeting_duration_minutes"] < 30
    ]
    fragmentation_score = (
        len(short_meeting_windows) / len(working)
        if working else 0.0
    )

    return {
        "meeting_window_count": len(meeting_windows),
        "total_meeting_minutes_week": len(meeting_windows) * 15,
        "avg_cls_in_meeting": avg_cls_meeting,
        "avg_cls_focused_work": avg_cls_focused,
        "avg_fdi_in_meeting": avg_fdi_meeting,
        "avg_fdi_focused_work": avg_fdi_focused,
        "meeting_cls_premium": cls_premium,
        "short_meeting_fragmentation_pct": round(fragmentation_score * 100, 1),
    }


def compute_weekly_analytics(all_windows: list[dict]) -> dict:
    """
    Compute all pre-analysis statistics from the full window set.

    This is the single entry point for building the rich data packet
    that gets included in the weekly report prompt.
    """
    if not all_windows:
        return {}

    return {
        "day_profiles": compute_day_of_week_profile(all_windows),
        "hourly_patterns": compute_hourly_patterns(all_windows),
        "focus_analysis": compute_focus_window_analysis(all_windows),
        "meeting_impact": compute_meeting_impact(all_windows),
        "hrv_cls_correlation": compute_hrv_cls_correlation(all_windows),
    }


# ─── Prompt construction ──────────────────────────────────────────────────────

def _build_analysis_prompt(summaries: list[dict], windows_sample: list[dict]) -> str:
    """
    Build the prompt for the weekly analysis subagent.

    v1.3: Pre-computes cross-metric pattern analytics from the full window
    set and includes them directly in the prompt. The LLM no longer needs
    to re-derive patterns from raw numbers — it receives pre-digested
    correlations, hourly profiles, and focus quality stats it can directly
    reference in the report.

    windows_sample: the active working-hours windows from the week
    (previously collected but never included in the prompt — now fixed).
    """
    # ── Pre-compute analytics ─────────────────────────────────────────────
    analytics = compute_weekly_analytics(windows_sample) if windows_sample else {}

    # ── Top-level weekly summary numbers ─────────────────────────────────
    avg_recovery = None
    recovery_vals = [
        s["whoop"].get("recovery_score")
        for s in summaries
        if s.get("whoop", {}).get("recovery_score") is not None
    ]
    if recovery_vals:
        avg_recovery = round(sum(recovery_vals) / len(recovery_vals), 1)

    avg_cls = None
    cls_vals = [
        s["metrics_avg"].get("cognitive_load_score")
        for s in summaries
        if s.get("metrics_avg", {}).get("cognitive_load_score") is not None
    ]
    if cls_vals:
        avg_cls = round(sum(cls_vals) / len(cls_vals), 3)

    # Peak CLS day
    peak_day = None
    peak_cls = 0.0
    for s in summaries:
        cls = s.get("metrics_avg", {}).get("cognitive_load_score") or 0
        if cls > peak_cls:
            peak_cls = cls
            peak_day = s.get("date")

    # Best focus day
    best_focus_day = None
    best_fdi = 0.0
    for s in summaries:
        fdi = s.get("metrics_avg", {}).get("focus_depth_index") or 0
        if fdi > best_fdi:
            best_fdi = fdi
            best_focus_day = s.get("date")

    # ── Format focus analysis into readable text ──────────────────────────
    focus_analysis = analytics.get("focus_analysis", {})
    focus_section = ""
    if focus_analysis and not focus_analysis.get("insufficient_data"):
        best_hours = focus_analysis.get("best_focus_hours", [])
        worst_hours = focus_analysis.get("worst_focus_hours", [])
        peak_load = focus_analysis.get("peak_load_hours", [])
        deep_work_slots = focus_analysis.get("recommended_deep_work_hours", [])

        if best_hours:
            hours_str = ", ".join(f"{h['hour']} (FDI {h['avg_fdi']:.0%})" for h in best_hours[:3])
            focus_section += f"Best focus hours this week: {hours_str}\n"
        if peak_load:
            hours_str = ", ".join(f"{h['hour']} (CLS {h['avg_cls']:.0%})" for h in peak_load[:3])
            focus_section += f"Highest-load hours: {hours_str}\n"
        if deep_work_slots:
            focus_section += f"Recommended deep-work slots: {', '.join(deep_work_slots)}\n"

    # ── Format meeting impact ─────────────────────────────────────────────
    meeting_impact = analytics.get("meeting_impact", {})
    meeting_section = ""
    if meeting_impact:
        total_min = meeting_impact.get("total_meeting_minutes_week", 0)
        cls_premium = meeting_impact.get("meeting_cls_premium")
        fdi_meeting = meeting_impact.get("avg_fdi_in_meeting")
        fdi_focused = meeting_impact.get("avg_fdi_focused_work")
        frag_pct = meeting_impact.get("short_meeting_fragmentation_pct", 0)

        if total_min:
            meeting_section += f"Total meeting time this week: {total_min} min\n"
        if cls_premium is not None:
            meeting_section += f"Meeting CLS premium: +{cls_premium:.0%} vs focused work\n"
        if fdi_meeting is not None and fdi_focused is not None:
            meeting_section += (
                f"FDI in meetings: {fdi_meeting:.0%} vs focused work: {fdi_focused:.0%}\n"
            )
        if frag_pct:
            meeting_section += f"Short meeting fragmentation: {frag_pct:.1f}% of working windows\n"

    # ── Format HRV correlation ────────────────────────────────────────────
    hrv_section = ""
    hrv_corr = analytics.get("hrv_cls_correlation", {})
    if hrv_corr and not hrv_corr.get("insufficient_data"):
        quartiles = hrv_corr.get("quartile_cls", {})
        direction = hrv_corr.get("correlation_direction", "flat")
        hrv_section += f"HRV–CLS correlation: {direction}\n"
        if quartiles.get("low_hrv") is not None:
            hrv_section += f"  Low HRV days → avg CLS: {quartiles['low_hrv']:.0%}\n"
        if quartiles.get("high_hrv") is not None:
            hrv_section += f"  High HRV days → avg CLS: {quartiles['high_hrv']:.0%}\n"
        hrv_section += f"  {hrv_corr.get('note', '')}\n"
    elif hrv_corr.get("insufficient_data"):
        hrv_section = hrv_corr.get("note", "")

    # ── Build the day-by-day table ────────────────────────────────────────
    day_profiles = analytics.get("day_profiles", [])
    day_table = ""
    if day_profiles:
        day_table = "Day-by-day breakdown:\n"
        for d in day_profiles:
            recovery = d.get("recovery_score")
            hrv = d.get("hrv_rmssd_milli")
            avg_cls_d = d.get("avg_cls")
            peak_cls_d = d.get("peak_cls")
            fdi = d.get("avg_fdi_active")
            ras = d.get("avg_ras")
            meet_min = d.get("total_meeting_minutes", 0)
            dow = d.get("day_of_week", d["date"])
            day_table += (
                f"  {dow} ({d['date']}): "
                f"Recovery={recovery}% HRV={hrv}ms | "
                f"CLS avg={avg_cls_d:.0%} peak={peak_cls_d:.0%} | "
                f"FDI={fdi:.0%} RAS={ras:.0%} | "
                f"Meetings={meet_min}min\n"
                if all(v is not None for v in [recovery, hrv, avg_cls_d, peak_cls_d, fdi, ras])
                else f"  {dow} ({d['date']}): "
                     f"Recovery={recovery}% | CLS avg={avg_cls_d} | Meetings={meet_min}min\n"
            )

    prompt = f"""You are Alfred, David Szabo-Stuban's AI butler. Analyze the past week of Presence Tracker data and write David's weekly cognitive load report.

SYSTEM CONTEXT:
Presence Tracker monitors David's cognitive load, focus depth, and mental strain by correlating:
- WHOOP physiological data (recovery %, HRV, sleep)
- Google Calendar (meetings, duration, attendees)
- Slack activity (messages sent/received, channels)

METRICS (all 0.0–1.0):
- CLS (Cognitive Load Score): how mentally demanding — higher = more demanding
- FDI (Focus Depth Index): depth of focus — higher = deeper, less interrupted
- RAS (Recovery Alignment): load vs physiological readiness — higher = better aligned
- SDI (Social Drain Index): social energy spent — higher = more drained
- CSC (Context Switch Cost): fragmentation penalty — higher = more fragmented

WEEKLY OVERVIEW:
- Days analyzed: {len(summaries)}
- Average recovery: {avg_recovery or 'unavailable'}%
- Average cognitive load: {f"{avg_cls:.0%}" if avg_cls is not None else 'unavailable'}
- Peak load day: {peak_day or 'unknown'} (CLS: {peak_cls:.0%})
- Best focus day: {best_focus_day or 'unknown'} (FDI: {best_fdi:.0%})

{day_table}

PATTERN ANALYTICS (pre-computed from {len(windows_sample)} active windows):
{focus_section if focus_section else '(insufficient active windows for hourly pattern analysis)'}

MEETING IMPACT:
{meeting_section if meeting_section else '(no meeting data)'}

HRV–LOAD CORRELATION:
{hrv_section if hrv_section else '(insufficient data)'}

DAILY SUMMARIES (full data):
{json.dumps(summaries, indent=2, default=str)}

REPORT TASK:
Write David's weekly Presence Report. Be specific — use the actual numbers above. Reference exact days, hours, and metric values where they're meaningful.

REQUIRED FORMAT:

**Weekly Presence Report — {datetime.now().strftime('%B %d, %Y')}**

**Health Baseline**
2–3 sentences on the week's recovery and HRV trend. Mention the best and worst days. Note whether physiological readiness was sufficient for the workload delivered.

**Cognitive Load Pattern**
Which days had the highest load? When was focus deepest vs most fragmented? Mention specific hours if the hourly data shows a clear pattern.

**Meeting Load**
How much of the week was in meetings? What was the impact on load and focus quality? Flag if meetings are dominating.

**Recovery Alignment**
Were the high-load days well-timed relative to physiological readiness? Any days where David pushed too hard for his recovery state?

**Key Insight**
One specific, non-obvious correlation or pattern from the data this week. Be precise and data-grounded.

**Recommendation**
One concrete, actionable thing to change or protect next week. Specific to this week's patterns — not generic advice.

TONE: Alfred's voice — reserved, precise, no filler. Direct but warm. Short paragraphs, no bullet lists. No emojis. Write as if speaking to someone who respects their own data and wants honest analysis.

After writing the report, send it to David's Slack DM: {SLACK_DM_CHANNEL}
Use: message tool, action=send, channel=slack, target={SLACK_DM_CHANNEL}
"""
    return prompt


def run_weekly_analysis() -> bool:
    """
    Run the weekly Alfred Intuition analysis.
    Returns True if successful.
    """
    print("[intuition] Starting weekly analysis")

    # Get last 7 days of summaries
    summaries = get_recent_summaries(days=7)
    if not summaries:
        print("[intuition] No data available for analysis", file=sys.stderr)
        return False

    # Get ALL active working-hours windows from the past 7 days
    # (previously only last 2 days were fetched, and they were never included in the prompt)
    dates = list_available_dates()
    recent_dates = sorted(dates)[-7:]
    windows_sample = []

    if recent_dates:
        from engine.store import read_day
        for d in recent_dates:
            day_windows = read_day(d)
            # Include ALL active working-hours windows for pattern analysis
            active = [
                w for w in day_windows
                if w["metadata"]["is_working_hours"]
                and (w["calendar"]["in_meeting"] or w["slack"]["total_messages"] > 0)
            ]
            windows_sample.extend(active)

    print(
        f"[intuition] Building report from {len(summaries)} day summaries, "
        f"{len(windows_sample)} active windows"
    )

    prompt = _build_analysis_prompt(summaries, windows_sample)

    print(f"[intuition] Spawning analysis agent")
    try:
        result = _gateway_invoke(
            "sessions_spawn",
            {
                "task": prompt,
                "cleanup": "delete",
                "runTimeoutSeconds": 180,
            },
        )
        if result.get("ok"):
            print("[intuition] Analysis complete")
            return True
        else:
            print(f"[intuition] Spawn failed: {result.get('error')}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[intuition] Error: {e}", file=sys.stderr)
        return False


def run_daily_alert(day_summary: dict) -> None:
    """
    Check if today's summary warrants a proactive alert to David.
    Fires if recovery < 50 and CLS > 0.70.
    """
    recovery = day_summary.get("whoop", {}).get("recovery_score")
    avg_cls = day_summary.get("metrics_avg", {}).get("cognitive_load_score", 0)

    if recovery is not None and recovery < 50 and avg_cls > 0.70:
        date = day_summary.get("date", "today")
        msg = (
            f"Presence alert — {date}: "
            f"Your recovery is at {recovery}% but your cognitive load averaged {avg_cls:.0%} today. "
            f"You're running above capacity. Consider protecting tomorrow morning."
        )
        try:
            _gateway_invoke("message", {
                "action": "send",
                "channel": "slack",
                "target": SLACK_DM_CHANNEL,
                "message": msg,
            })
        except Exception as e:
            print(f"[intuition] Alert send failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    success = run_weekly_analysis()
    sys.exit(0 if success else 1)

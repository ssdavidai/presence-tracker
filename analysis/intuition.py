"""
Presence Tracker — Alfred Intuition Layer

LLM-powered weekly pattern analysis.
Runs on Sunday evenings, synthesizes the week's data,
and delivers a Slack report to David.

Uses the OpenClaw gateway to spawn a subagent.
"""

import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

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


def _build_analysis_prompt(summaries: list[dict], windows_sample: list[dict]) -> str:
    """Build the prompt for the weekly analysis subagent."""

    # Compute some quick stats for the prompt
    avg_recovery = None
    recovery_vals = [s["whoop"].get("recovery_score") for s in summaries if s.get("whoop", {}).get("recovery_score") is not None]
    if recovery_vals:
        avg_recovery = round(sum(recovery_vals) / len(recovery_vals), 1)

    avg_cls = None
    cls_vals = [s["metrics_avg"].get("cognitive_load_score") for s in summaries if s.get("metrics_avg", {}).get("cognitive_load_score") is not None]
    if cls_vals:
        avg_cls = round(sum(cls_vals) / len(cls_vals), 3)

    # Find peak CLS day
    peak_day = None
    peak_cls = 0
    for s in summaries:
        cls = s.get("metrics_avg", {}).get("cognitive_load_score", 0)
        if cls and cls > peak_cls:
            peak_cls = cls
            peak_day = s.get("date")

    # Find best focus day
    best_focus_day = None
    best_fdi = 0
    for s in summaries:
        fdi = s.get("metrics_avg", {}).get("focus_depth_index", 0)
        if fdi and fdi > best_fdi:
            best_fdi = fdi
            best_focus_day = s.get("date")

    prompt = f"""You are Alfred, David Szabo-Stuban's AI butler. You are analyzing the past week of presence data for David.

CONTEXT:
This is the Presence Tracker system — it monitors David's cognitive load, focus depth, and mental strain by correlating WHOOP health data, calendar events, and Slack activity.

WEEKLY SUMMARY DATA:
{json.dumps(summaries, indent=2, default=str)}

KEY METRICS THIS WEEK:
- Average recovery score: {avg_recovery or 'unavailable'}%
- Average cognitive load score: {avg_cls or 'unavailable'} (0=low, 1=high)
- Peak load day: {peak_day or 'unknown'} (CLS: {round(peak_cls, 3)})
- Best focus day: {best_focus_day or 'unknown'} (FDI: {round(best_fdi, 3)})

ANALYSIS TASK:
Write David's weekly Presence Report. Be concise, specific, and actionable.

Structure your report as follows:

**Weekly Presence Report — {datetime.now().strftime('%B %d, %Y')}**

**Health Baseline**
2-3 sentences on WHOOP recovery and HRV trend. Note if recovery was good enough to support the workload.

**Cognitive Load Pattern**
Which days/windows were highest load? Any concerning patterns? When was focus deepest?

**Recovery Alignment**
Was David working within his physiological capacity, or pushing beyond it?

**Key Insight**
One specific, non-obvious insight from the data.

**Recommendation**
One concrete thing to change or protect for next week.

TONE: Reserved, precise, no fluff. Alfred's voice — warm but direct. No bullet lists, just short paragraphs. No emojis.

After writing the report, send it to David's Slack DM channel: {SLACK_DM_CHANNEL}
Use the message tool with action=send, channel=slack, target={SLACK_DM_CHANNEL}.
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

    # Get a sample of high-activity windows for context
    dates = list_available_dates()
    recent_dates = sorted(dates)[-7:]
    windows_sample = []
    if recent_dates:
        from engine.store import read_day
        for d in recent_dates[-2:]:  # Last 2 days in detail
            day_windows = read_day(d)
            # Only working hours with some activity
            active = [
                w for w in day_windows
                if w["metadata"]["is_working_hours"]
                and (w["calendar"]["in_meeting"] or w["slack"]["total_messages"] > 0)
            ]
            windows_sample.extend(active[:20])  # Cap at 20 per day

    prompt = _build_analysis_prompt(summaries, windows_sample)

    print(f"[intuition] Spawning analysis agent with {len(summaries)} day summaries")
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

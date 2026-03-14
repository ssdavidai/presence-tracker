"""
Presence Tracker — Daily Digest

Sends David a personal Slack DM at the end of each day with a
concise cognitive load summary: how he spent his mental energy,
whether he was within physiological capacity, and one insight.

This is the primary human-facing output of the Presence Tracker —
the difference between data sitting in JSONL files and David actually
knowing how his day went cognitively.
"""

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GATEWAY_URL, GATEWAY_TOKEN, SLACK_DM_CHANNEL


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _score_bar(value: float, width: int = 10) -> str:
    """Convert a 0-1 score to a visual bar: ▓▓▓▓▓░░░░░"""
    filled = round(value * width)
    return "▓" * filled + "░" * (width - filled)


def _fmt_score(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.0%}"


def _cls_label(cls: float) -> str:
    """Human label for a CLS value."""
    if cls < 0.20:
        return "light"
    elif cls < 0.40:
        return "moderate"
    elif cls < 0.60:
        return "heavy"
    elif cls < 0.80:
        return "intense"
    else:
        return "maximal"


def _ras_label(ras: float) -> str:
    """Human label for a RAS value."""
    if ras >= 0.80:
        return "well within capacity"
    elif ras >= 0.60:
        return "within capacity"
    elif ras >= 0.40:
        return "slightly stretched"
    elif ras >= 0.20:
        return "over capacity"
    else:
        return "significantly over capacity"


def _fdi_label(fdi: float) -> str:
    """Human label for a FDI value (active hours only)."""
    if fdi >= 0.80:
        return "deep"
    elif fdi >= 0.60:
        return "reasonable"
    elif fdi >= 0.40:
        return "fragmented"
    else:
        return "highly fragmented"


# ─── Digest computation ───────────────────────────────────────────────────────

def compute_digest(windows: list[dict]) -> dict:
    """
    Compute the digest data from a day's windows.

    Returns a structured dict with all the numbers needed for the DM.
    Works only on working-hours windows, and only on active windows
    (meeting or Slack activity) for focus quality.
    """
    if not windows:
        return {}

    date_str = windows[0]["date"]
    whoop = windows[0]["whoop"]  # Same for all windows (daily data)

    # Working hours: 7am-10pm
    working = [w for w in windows if w["metadata"]["is_working_hours"]]

    # Active windows: in a meeting or had Slack messages
    active = [w for w in working if w["calendar"]["in_meeting"] or w["slack"]["total_messages"] > 0]

    # Idle working windows: no meeting, no Slack — pure quiet time
    idle = [w for w in working if not w["calendar"]["in_meeting"] and w["slack"]["total_messages"] == 0]

    def _avg(vals: list) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _peak(vals: list) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    # CLS: computed over all working hours (includes idle, which is legit—low CLS is good)
    cls_vals_working = [w["metrics"]["cognitive_load_score"] for w in working]
    avg_cls = _avg(cls_vals_working)
    peak_cls = _peak(cls_vals_working)

    # FDI: only meaningful over active windows — idle windows trivially = 1.0
    fdi_vals_active = [w["metrics"]["focus_depth_index"] for w in active]
    avg_fdi_active = _avg(fdi_vals_active)

    # SDI: active windows only
    sdi_vals_active = [w["metrics"]["social_drain_index"] for w in active]
    avg_sdi_active = _avg(sdi_vals_active)

    # CSC: active windows only
    csc_vals_active = [w["metrics"]["context_switch_cost"] for w in active]
    avg_csc_active = _avg(csc_vals_active)

    # RAS: all windows (recovery alignment is meaningful throughout the day)
    ras_vals = [w["metrics"]["recovery_alignment_score"] for w in windows]
    avg_ras = _avg(ras_vals)

    # Peak load window: when was CLS highest?
    peak_window = None
    if working:
        peak_window = max(working, key=lambda w: w["metrics"]["cognitive_load_score"])

    # Meeting stats
    meeting_windows = [w for w in working if w["calendar"]["in_meeting"]]
    total_meeting_minutes = len(meeting_windows) * 15
    meeting_count = len(set(
        w["calendar"]["meeting_title"]
        for w in meeting_windows
        if w["calendar"]["meeting_title"]
    ))

    # Slack stats
    total_sent = sum(w["slack"]["messages_sent"] for w in windows)
    total_received = sum(w["slack"]["messages_received"] for w in windows)

    # Recovery alignment insight
    recovery = whoop.get("recovery_score")
    hrv = whoop.get("hrv_rmssd_milli")
    sleep_h = whoop.get("sleep_hours")

    # Generate one key insight
    insight = _generate_insight(
        recovery=recovery,
        avg_cls=avg_cls,
        avg_fdi_active=avg_fdi_active,
        avg_ras=avg_ras,
        total_meeting_minutes=total_meeting_minutes,
        total_sent=total_sent,
        peak_window=peak_window,
        working_count=len(working),
        active_count=len(active),
    )

    return {
        "date": date_str,
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "sleep_hours": sleep_h,
            "sleep_performance": whoop.get("sleep_performance"),
        },
        "metrics": {
            "avg_cls": avg_cls,
            "peak_cls": peak_cls,
            "avg_fdi_active": avg_fdi_active,  # FDI over active windows only
            "avg_sdi_active": avg_sdi_active,
            "avg_csc_active": avg_csc_active,
            "avg_ras": avg_ras,
        },
        "activity": {
            "working_windows": len(working),
            "active_windows": len(active),
            "idle_windows": len(idle),
            "total_meeting_minutes": total_meeting_minutes,
            "meeting_count": meeting_count,
            "slack_sent": total_sent,
            "slack_received": total_received,
        },
        "peak_window": peak_window,
        "insight": insight,
    }


def _generate_insight(
    recovery: Optional[float],
    avg_cls: Optional[float],
    avg_fdi_active: Optional[float],
    avg_ras: Optional[float],
    total_meeting_minutes: int,
    total_sent: int,
    peak_window: Optional[dict],
    working_count: int,
    active_count: int,
) -> str:
    """
    Generate one data-driven insight for today.
    Priority: alignment issues > focus quality > meeting load > quiet day.
    """
    if recovery is not None and avg_cls is not None:
        if recovery < 50 and avg_cls > 0.50:
            return (
                f"You pushed hard ({avg_cls:.0%} avg load) on a {recovery:.0f}% recovery day. "
                f"Consider a lighter schedule tomorrow."
            )
        if recovery < 50 and avg_cls <= 0.30:
            return (
                f"Good self-management: recovery was low ({recovery:.0f}%) and you kept load light. "
                f"HRV should bounce back tomorrow."
            )
        if recovery >= 80 and avg_cls is not None and avg_cls < 0.20 and active_count < 5:
            return (
                f"High recovery ({recovery:.0f}%) but very light cognitive load today. "
                f"You have capacity to take on more if needed."
            )

    if avg_fdi_active is not None and avg_fdi_active < 0.50 and active_count >= 4:
        return (
            f"Focus was fragmented during active work (FDI {avg_fdi_active:.0%}). "
            f"Try protecting at least one uninterrupted 90-minute block tomorrow."
        )

    if total_meeting_minutes >= 240:
        hours = total_meeting_minutes // 60
        return (
            f"{hours}+ hours in meetings today. "
            f"Heavy meeting load reduces recovery and deep work — consider blocking tomorrow morning."
        )

    if active_count == 0:
        return "No significant cognitive activity detected today — rest day or data gap."

    if avg_cls is not None and avg_cls < 0.15:
        return "Light cognitive day. Good for recovery — HRV should hold or improve."

    return "Load within normal range. No anomalies detected."


# ─── Slack message builder ────────────────────────────────────────────────────

def format_digest_message(digest: dict) -> str:
    """
    Format the digest data into a Slack DM message.

    Designed to be readable in Slack without markdown rendering issues.
    """
    if not digest:
        return "Presence Tracker: no data available for today."

    date_str = digest.get("date", "today")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_label = dt.strftime("%A, %B %-d")
    except ValueError:
        date_label = date_str

    m = digest.get("metrics", {})
    w = digest.get("whoop", {})
    act = digest.get("activity", {})

    avg_cls = m.get("avg_cls")
    peak_cls = m.get("peak_cls")
    avg_fdi = m.get("avg_fdi_active")
    avg_ras = m.get("avg_ras")
    recovery = w.get("recovery_score")
    hrv = w.get("hrv_rmssd_milli")
    sleep_h = w.get("sleep_hours")

    meeting_mins = act.get("total_meeting_minutes", 0)
    meeting_count = act.get("meeting_count", 0)
    active_windows = act.get("active_windows", 0)
    slack_sent = act.get("slack_sent", 0)

    peak_window = digest.get("peak_window")
    insight = digest.get("insight", "")

    lines = [
        f"*Presence Report — {date_label}*",
        "",
    ]

    # ── Health baseline ──
    if recovery is not None:
        lines.append(
            f"*Recovery* {_score_bar(recovery / 100)} {recovery:.0f}%"
            + (f"  ·  HRV {hrv:.0f}ms" if hrv else "")
            + (f"  ·  Sleep {sleep_h:.1f}h" if sleep_h else "")
        )
    else:
        lines.append("*Recovery* — unavailable")

    lines.append("")

    # ── Cognitive load ──
    if avg_cls is not None:
        cls_bar = _score_bar(avg_cls)
        label = _cls_label(avg_cls)
        lines.append(f"*Cognitive Load* {cls_bar} {avg_cls:.0%} avg ({label})")
        if peak_cls and peak_cls > avg_cls + 0.10:
            peak_info = ""
            if peak_window:
                h = peak_window["metadata"]["hour_of_day"]
                mi = peak_window["metadata"]["minute_of_hour"]
                peak_info = f" at {h:02d}:{mi:02d}"
            lines.append(f"  Peak: {peak_cls:.0%}{peak_info}")
    else:
        lines.append("*Cognitive Load* — no data")

    # ── Focus quality (active windows only) ──
    if avg_fdi is not None and active_windows > 0:
        fdi_bar = _score_bar(avg_fdi)
        fdi_label = _fdi_label(avg_fdi)
        lines.append(f"*Focus Quality* {fdi_bar} {avg_fdi:.0%} ({fdi_label}, active windows)")
    elif active_windows == 0:
        lines.append("*Focus Quality* — no active work detected")

    # ── Recovery alignment ──
    if avg_ras is not None:
        ras_bar = _score_bar(avg_ras)
        ras_label = _ras_label(avg_ras)
        lines.append(f"*Alignment* {ras_bar} {avg_ras:.0%} ({ras_label})")

    lines.append("")

    # ── Activity summary ──
    activity_parts = []
    if meeting_count > 0:
        activity_parts.append(f"{meeting_count} meeting{'s' if meeting_count != 1 else ''} ({meeting_mins} min)")
    if slack_sent > 0:
        activity_parts.append(f"{slack_sent} messages sent")
    if active_windows > 0:
        activity_parts.append(f"{active_windows} active windows")

    if activity_parts:
        lines.append("_" + "  ·  ".join(activity_parts) + "_")
    else:
        lines.append("_No significant activity detected_")

    # ── Insight ──
    if insight:
        lines.append("")
        lines.append(f"💡 {insight}")

    return "\n".join(lines)


# ─── Send digest ──────────────────────────────────────────────────────────────

def _send_slack_dm(message: str, target: str = SLACK_DM_CHANNEL) -> bool:
    """Send a message to David's Slack DM via the gateway."""
    try:
        headers = {
            "Authorization": f"Bearer {GATEWAY_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "tool": "message",
            "args": {
                "action": "send",
                "channel": "slack",
                "target": target,
                "message": message,
            }
        }).encode()
        req = urllib.request.Request(
            f"{GATEWAY_URL}/tools/invoke",
            data=payload,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"[digest] Failed to send DM: {e}", file=sys.stderr)
        return False


def send_daily_digest(windows: list[dict]) -> bool:
    """
    Compute and send the daily digest DM to David.

    Args:
        windows: List of 96 window dicts for the day.

    Returns:
        True if the DM was sent successfully.
    """
    if not windows:
        print("[digest] No windows to digest", file=sys.stderr)
        return False

    digest = compute_digest(windows)
    message = format_digest_message(digest)

    print(f"[digest] Sending daily DM for {digest.get('date')}")
    return _send_slack_dm(message)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send today's Presence Digest to David")
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD, default: today's chunk)")
    parser.add_argument("--dry-run", action="store_true", help="Print the message without sending")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.store import read_day, list_available_dates

    if args.date:
        date_str = args.date
    else:
        dates = list_available_dates()
        if not dates:
            print("No data available.", file=sys.stderr)
            sys.exit(1)
        date_str = sorted(dates)[-1]

    windows = read_day(date_str)
    if not windows:
        print(f"No data for {date_str}", file=sys.stderr)
        sys.exit(1)

    digest = compute_digest(windows)
    message = format_digest_message(digest)

    print("=" * 60)
    print(message)
    print("=" * 60)

    if not args.dry_run:
        ok = _send_slack_dm(message)
        print(f"\n{'✓ Sent' if ok else '✗ Failed to send'} to David's DM")
    else:
        print("\n[dry-run] Not sent.")

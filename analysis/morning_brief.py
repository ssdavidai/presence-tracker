"""
Presence Tracker — Morning Readiness Brief

Sends David a morning Slack DM at 07:00 Budapest time with:
- Today's WHOOP readiness (recovery score, HRV, sleep quality)
- A capacity label and day-planning recommendation
- Yesterday's cognitive load as context
- A scheduling suggestion based on physiological state

This is the forward-looking complement to the end-of-day digest.
The digest tells you how the day went; the morning brief tells you
what kind of day to plan.

Architecture:
    1. Collect WHOOP data for today (WHOOP posts last night's data by ~6am)
    2. Load yesterday's daily summary from the store (if available)
    3. Compute readiness tier and recommendation
    4. Send DM to David

Design principle: actionable specificity.
Generic "rest today" advice is useless. The brief gives David one
concrete scheduling action based on the actual numbers.
"""

import json
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GATEWAY_URL, GATEWAY_TOKEN, SLACK_DM_CHANNEL


# ─── Readiness tiers ──────────────────────────────────────────────────────────

def _readiness_tier(recovery: Optional[float], hrv: Optional[float]) -> str:
    """
    Classify today's physiological readiness into a tier.

    Returns one of: 'peak', 'good', 'moderate', 'low', 'recovery'

    Tier logic uses WHOOP recovery as the primary signal and HRV as a
    secondary confirmation.  This mirrors the WHOOP UX but adds nuance:
    moderate recovery with low HRV is treated as 'low' because the
    autonomic nervous system is signalling stress even if WHOOP's composite
    is in the moderate range.
    """
    if recovery is None:
        return "unknown"

    # HRV modifier: flag autonomic stress even when recovery looks moderate
    hrv_stressed = hrv is not None and hrv < 45  # well below typical population median

    if recovery >= 80:
        return "peak" if not hrv_stressed else "good"
    elif recovery >= 67:
        return "good" if not hrv_stressed else "moderate"
    elif recovery >= 50:
        return "moderate" if not hrv_stressed else "low"
    elif recovery >= 33:
        return "low"
    else:
        return "recovery"


def _tier_label(tier: str) -> str:
    """Human-readable label for the readiness tier."""
    return {
        "peak": "Peak",
        "good": "Good",
        "moderate": "Moderate",
        "low": "Low",
        "recovery": "Recovery Day",
        "unknown": "Unknown",
    }.get(tier, tier.title())


def _tier_recommendation(
    tier: str,
    recovery: Optional[float],
    hrv: Optional[float],
    yesterday_cls: Optional[float],
    yesterday_meeting_mins: Optional[int],
) -> str:
    """
    Generate a specific, actionable scheduling recommendation.

    Combines today's physiological state with yesterday's cognitive
    load to produce a concrete suggestion — not just "rest" or "go hard"
    but specifically what kind of work to front-load or protect.

    Parameters
    ----------
    tier : readiness tier string
    recovery : WHOOP recovery score (0–100)
    hrv : HRV RMSSD in milliseconds
    yesterday_cls : average cognitive load score from yesterday (0–1)
    yesterday_meeting_mins : total meeting time yesterday in minutes
    """
    # Context from yesterday
    heavy_yesterday = yesterday_cls is not None and yesterday_cls > 0.45
    meeting_heavy = yesterday_meeting_mins is not None and yesterday_meeting_mins >= 240

    if tier == "peak":
        if heavy_yesterday:
            return (
                "You're fully recovered from yesterday's demanding session. "
                "Good window for creative or strategic work that requires full cognitive bandwidth."
            )
        return (
            "High physiological readiness. Front-load complex, creative, or high-stakes work "
            "into this morning while capacity is at its peak."
        )

    elif tier == "good":
        return (
            "Solid readiness — capable of sustained demanding work. "
            "Normal scheduling is fine; protect at least one deep-work block of 90+ minutes."
        )

    elif tier == "moderate":
        if meeting_heavy:
            return (
                "Moderate readiness after a meeting-heavy day. "
                "Avoid stacking another heavy meeting block today — protect afternoon for recovery."
            )
        if heavy_yesterday:
            return (
                "Moderate readiness following a high-load day. "
                "Lighter cognitive work preferred; defer decisions requiring deep analysis to tomorrow."
            )
        return (
            "Moderate readiness. Manageable day, but avoid stacking meetings. "
            "One focused deep-work session is realistic; more than two demanding blocks will drain reserves."
        )

    elif tier == "low":
        hrv_note = (
            f" (HRV at {hrv:.0f}ms signals autonomic pressure)" if hrv else ""
        )
        return (
            f"Low readiness{hrv_note}. Keep today's schedule light: "
            "routine tasks, async communication, no major strategic decisions. "
            "Prioritise sleep hygiene tonight."
        )

    elif tier == "recovery":
        rec_note = f"Recovery at {recovery:.0f}%" if recovery else "Very low recovery"
        return (
            f"{rec_note} — this is a genuine recovery day. "
            "Cancel or reschedule anything cognitively demanding. "
            "Short walks, admin tasks, and protecting sleep tonight are the priority."
        )

    else:  # unknown
        return "WHOOP data unavailable — check your device charge and sync."


# ─── Score bar helper (shared style with daily digest) ───────────────────────

def _score_bar(value: float, width: int = 10) -> str:
    """Convert a 0–1 score to a visual progress bar."""
    filled = round(value * width)
    return "▓" * filled + "░" * (width - filled)


def _hrv_context(hrv: Optional[float], hrv_baseline: Optional[float]) -> str:
    """Format HRV with a relative context note if baseline is available."""
    if hrv is None:
        return "N/A"
    if hrv_baseline is None:
        return f"{hrv:.0f}ms"
    diff_pct = (hrv - hrv_baseline) / hrv_baseline * 100
    if abs(diff_pct) < 8:
        return f"{hrv:.0f}ms (baseline)"
    elif diff_pct > 0:
        return f"{hrv:.0f}ms (+{diff_pct:.0f}% vs baseline)"
    else:
        return f"{hrv:.0f}ms ({diff_pct:.0f}% vs baseline)"


# ─── Brief computation ────────────────────────────────────────────────────────

def compute_morning_brief(
    today_date: str,
    whoop_data: dict,
    yesterday_summary: Optional[dict] = None,
    hrv_baseline: Optional[float] = None,
) -> dict:
    """
    Compute the morning brief data structure.

    Parameters
    ----------
    today_date : "YYYY-MM-DD"
    whoop_data : raw output from collectors.whoop.collect(today_date)
    yesterday_summary : daily summary dict from store (optional)
    hrv_baseline : 7-day average HRV in ms (optional, for relative context)

    Returns
    -------
    dict with all fields needed to format the morning brief message.
    """
    recovery = whoop_data.get("recovery_score")
    hrv = whoop_data.get("hrv_rmssd_milli")
    sleep_hours = whoop_data.get("sleep_hours")
    sleep_performance = whoop_data.get("sleep_performance")
    rhr = whoop_data.get("resting_heart_rate")

    # Yesterday's context
    yesterday_cls = None
    yesterday_meeting_mins = None
    yesterday_date = None
    if yesterday_summary:
        yesterday_date = yesterday_summary.get("date")
        yesterday_cls = yesterday_summary.get("metrics_avg", {}).get("cognitive_load_score")
        yesterday_meeting_mins = yesterday_summary.get("calendar", {}).get("total_meeting_minutes")

    tier = _readiness_tier(recovery, hrv)
    recommendation = _tier_recommendation(
        tier, recovery, hrv, yesterday_cls, yesterday_meeting_mins
    )

    return {
        "date": today_date,
        "whoop": {
            "recovery_score": recovery,
            "hrv_rmssd_milli": hrv,
            "sleep_hours": sleep_hours,
            "sleep_performance": sleep_performance,
            "resting_heart_rate": rhr,
        },
        "readiness": {
            "tier": tier,
            "label": _tier_label(tier),
            "recommendation": recommendation,
        },
        "yesterday": {
            "date": yesterday_date,
            "avg_cls": yesterday_cls,
            "meeting_minutes": yesterday_meeting_mins,
        },
        "hrv_baseline": hrv_baseline,
    }


# ─── Message formatter ────────────────────────────────────────────────────────

def format_morning_brief_message(brief: dict) -> str:
    """
    Format the morning brief into a Slack DM.

    Designed to be scannable in under 30 seconds at 7am.
    Key info is front-loaded; recommendation is prominent.
    """
    if not brief:
        return "Presence Tracker: no morning data available."

    date_str = brief.get("date", "today")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_label = dt.strftime("%A, %B %-d")
    except ValueError:
        date_label = date_str

    w = brief.get("whoop", {})
    r = brief.get("readiness", {})
    yesterday = brief.get("yesterday", {})
    hrv_baseline = brief.get("hrv_baseline")

    recovery = w.get("recovery_score")
    hrv = w.get("hrv_rmssd_milli")
    sleep_h = w.get("sleep_hours")
    sleep_perf = w.get("sleep_performance")
    rhr = w.get("resting_heart_rate")

    tier = r.get("tier", "unknown")
    tier_label = r.get("label", "Unknown")
    recommendation = r.get("recommendation", "")

    # Tier emoji
    tier_emoji = {
        "peak": "🟢",
        "good": "🔵",
        "moderate": "🟡",
        "low": "🟠",
        "recovery": "🔴",
        "unknown": "⚪",
    }.get(tier, "⚪")

    lines = [
        f"*Morning Readiness — {date_label}*",
        "",
    ]

    # ── Readiness headline ──
    if recovery is not None:
        rec_bar = _score_bar(recovery / 100)
        lines.append(f"{tier_emoji} *{tier_label}*  {rec_bar} Recovery {recovery:.0f}%")
    else:
        lines.append(f"{tier_emoji} *{tier_label}*  (no WHOOP data)")

    # ── WHOOP signals ──
    detail_parts = []
    if hrv is not None:
        detail_parts.append(_hrv_context(hrv, hrv_baseline))
    if rhr is not None:
        detail_parts.append(f"RHR {rhr:.0f}bpm")
    if sleep_h is not None:
        sleep_str = f"Sleep {sleep_h:.1f}h"
        if sleep_perf is not None:
            sleep_str += f" ({sleep_perf:.0f}%)"
        detail_parts.append(sleep_str)
    if detail_parts:
        lines.append("_" + "  ·  ".join(detail_parts) + "_")

    lines.append("")

    # ── Recommendation ──
    lines.append(f"*Today:* {recommendation}")

    # ── Yesterday context ──
    if yesterday.get("avg_cls") is not None:
        y_cls = yesterday["avg_cls"]
        y_mins = yesterday.get("meeting_minutes", 0) or 0
        y_date = yesterday.get("date", "yesterday")
        try:
            y_dt = datetime.strptime(y_date, "%Y-%m-%d")
            y_label = y_dt.strftime("%A")
        except ValueError:
            y_label = "Yesterday"

        cls_parts = [f"CLS {y_cls:.0%}"]
        if y_mins:
            cls_parts.append(f"{y_mins}min meetings")
        lines.append("")
        lines.append(f"_Yesterday ({y_label}): {' · '.join(cls_parts)}_")

    return "\n".join(lines)


# ─── Gateway helpers ──────────────────────────────────────────────────────────

def _gateway_invoke(tool: str, args: dict, timeout: int = 30) -> dict:
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


def _send_slack_dm(message: str) -> bool:
    """Send the morning brief to David's Slack DM."""
    try:
        result = _gateway_invoke("message", {
            "action": "send",
            "channel": "slack",
            "target": SLACK_DM_CHANNEL,
            "message": message,
        })
        return result.get("ok", False)
    except Exception as e:
        print(f"[morning] Failed to send DM: {e}", file=sys.stderr)
        return False


# ─── Main entry point ─────────────────────────────────────────────────────────

def send_morning_brief(date_str: Optional[str] = None) -> bool:
    """
    Collect today's WHOOP data and send the morning readiness brief.

    Args:
        date_str: Date to run for (YYYY-MM-DD). Defaults to today.

    Returns:
        True if the DM was sent successfully.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"[morning] Running morning brief for {date_str}")

    # ── Collect WHOOP data ────────────────────────────────────────────────
    try:
        from collectors.whoop import collect as whoop_collect
        whoop_data = whoop_collect(date_str)
        print(
            f"[morning] WHOOP: recovery={whoop_data.get('recovery_score')}% "
            f"HRV={whoop_data.get('hrv_rmssd_milli')}ms"
        )
    except Exception as e:
        print(f"[morning] WHOOP collection failed: {e}", file=sys.stderr)
        whoop_data = {}

    # ── Load yesterday's summary ──────────────────────────────────────────
    yesterday_summary = None
    yesterday_date = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    try:
        from engine.store import read_summary
        rolling = read_summary()
        yesterday_summary = rolling.get("days", {}).get(yesterday_date)
        if yesterday_summary:
            print(
                f"[morning] Yesterday ({yesterday_date}): "
                f"CLS={yesterday_summary.get('metrics_avg', {}).get('cognitive_load_score')}"
            )
        else:
            print(f"[morning] No summary for yesterday ({yesterday_date})")
    except Exception as e:
        print(f"[morning] Could not load yesterday's summary: {e}", file=sys.stderr)

    # ── Compute 7-day HRV baseline for context ────────────────────────────
    hrv_baseline = None
    try:
        from engine.store import get_recent_summaries
        recent = get_recent_summaries(days=7)
        hrv_vals = [
            s.get("whoop", {}).get("hrv_rmssd_milli")
            for s in recent
            if s.get("whoop", {}).get("hrv_rmssd_milli") is not None
        ]
        if len(hrv_vals) >= 3:
            hrv_baseline = sum(hrv_vals) / len(hrv_vals)
            print(f"[morning] 7-day HRV baseline: {hrv_baseline:.1f}ms")
    except Exception as e:
        print(f"[morning] Could not compute HRV baseline: {e}", file=sys.stderr)

    # ── Compute and send ──────────────────────────────────────────────────
    brief = compute_morning_brief(
        today_date=date_str,
        whoop_data=whoop_data,
        yesterday_summary=yesterday_summary,
        hrv_baseline=hrv_baseline,
    )
    message = format_morning_brief_message(brief)

    print(f"[morning] Sending brief to David...")
    ok = _send_slack_dm(message)
    print(f"[morning] {'Sent' if ok else 'Failed'}")
    return ok


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send morning readiness brief to David")
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD, default: today)")
    parser.add_argument("--dry-run", action="store_true", help="Print without sending")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    # Collect WHOOP
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from collectors.whoop import collect as whoop_collect
        whoop_data = whoop_collect(date_str)
    except Exception as e:
        print(f"WHOOP failed: {e}", file=sys.stderr)
        whoop_data = {}

    # Yesterday summary
    yesterday_date = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")
    yesterday_summary = None
    try:
        from engine.store import read_summary
        yesterday_summary = read_summary().get("days", {}).get(yesterday_date)
    except Exception:
        pass

    # HRV baseline
    hrv_baseline = None
    try:
        from engine.store import get_recent_summaries
        recent = get_recent_summaries(days=7)
        vals = [s["whoop"]["hrv_rmssd_milli"] for s in recent
                if s.get("whoop", {}).get("hrv_rmssd_milli")]
        if len(vals) >= 3:
            hrv_baseline = sum(vals) / len(vals)
    except Exception:
        pass

    brief = compute_morning_brief(date_str, whoop_data, yesterday_summary, hrv_baseline)
    message = format_morning_brief_message(brief)

    print("=" * 60)
    print(message)
    print("=" * 60)

    if not args.dry_run:
        ok = _send_slack_dm(message)
        print(f"\n{'✓ Sent' if ok else '✗ Failed'} to David's DM")
    else:
        print("\n[dry-run] Not sent.")

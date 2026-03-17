"""
Presence Tracker — Evening Wind-Down (v30)

Answers: *"How did today go, and how should I close out the day?"*

The morning brief (07:00) starts the day with capacity + plan.
The midday check-in (13:00) pulses morning load vs plan.
The nightly digest (23:45) reviews the full day in retrospect.

There's a gap between the midday check-in and the nightly digest.
The Evening Wind-Down closes it — fired at 18:00 Budapest (end of workday).

It answers three questions David needs at 6pm:
  1. How did today go?         — day type classification + DPS projection
  2. Was it balanced?          — afternoon vs morning load delta
  3. What should I do now?     — concrete wind-down recommendation

## Day type classification

Based on the full day's data (00:00–18:00), each day is classified:

| Day Type   | Condition                                                          |
|------------|--------------------------------------------------------------------|
| PRODUCTIVE | FDI ≥ 0.65, CLS moderate, meetings < 3h, low CSC                  |
| DEEP       | FDI ≥ 0.75, CLS moderate-high, very few meetings, long focus runs |
| REACTIVE   | SDI ≥ 0.55 OR Slack msgs sent ≥ 30 OR meetings > 3h, FDI < 0.55  |
| FRAGMENTED | CSC ≥ 0.35 OR many short meetings, FDI < 0.55                     |
| RECOVERY   | CLS < 0.20, FDI high (little active load, mostly reading/resting) |
| MIXED      | None of the above cleanly apply                                    |

## Load arc

The module computes the AM/PM load split:
  morning = windows 08:00–13:00 (active)
  afternoon = windows 13:00–18:00 (active)

  load_arc = "front-loaded" | "back-loaded" | "even" | "spiky"
  arc_delta = afternoon_cls - morning_cls
    > 0.15 → back-loaded (rising pressure)
    < -0.15 → front-loaded (morning heaviest)
    else → even

## Wind-down recommendation logic

| Day Type   | Recovery   | Load Arc      | Recommendation                              |
|------------|------------|---------------|---------------------------------------------|
| REACTIVE   | any        | any           | Social recovery: no screens, quiet time     |
| FRAGMENTED | any        | any           | Defrag: one clear task to finish cleanly    |
| PRODUCTIVE | < 60%      | any           | Protect sleep; good output already          |
| PRODUCTIVE | ≥ 60%      | front-loaded  | Light planning for tomorrow is fine         |
| DEEP       | ≥ 60%      | any           | Strong day; protect the quality into sleep  |
| RECOVERY   | any        | any           | Rest day confirmed; early sleep             |
| MIXED      | any        | back-loaded   | Step away now; don't push into evening      |
| MIXED      | any        | other         | Wrap up; protect the evening                |

## Output

    EveningWindDown dataclass:
      - date_str: str
      - day_type: str              — PRODUCTIVE | DEEP | REACTIVE | FRAGMENTED | RECOVERY | MIXED
      - day_type_label: str        — human-readable label
      - day_type_emoji: str        — emoji for the type
      - load_arc: str              — "front-loaded" | "back-loaded" | "even" | "spiky"
      - morning_cls: float | None  — avg CLS 08:00–13:00 active windows
      - afternoon_cls: float | None — avg CLS 13:00–18:00 active windows
      - full_day_cls: float | None — avg CLS all active windows
      - full_day_fdi: float | None — avg FDI all active windows
      - full_day_sdi: float | None — avg SDI all active windows
      - full_day_csc: float | None — avg CSC all active windows
      - total_meeting_minutes: int
      - slack_sent: int
      - projected_dps: float | None — projected DPS based on partial day
      - wind_down_recommendation: str — one concrete sentence
      - wind_down_detail: str     — two-sentence context
      - active_windows_count: int
      - is_meaningful: bool       — False when < MIN_ACTIVE_WINDOWS for the day

## API

    from analysis.evening_winddown import compute_evening_winddown, format_winddown_message

    winddown = compute_evening_winddown(date_str)
    message = format_winddown_message(winddown)   # Slack-ready string

## CLI

    python3 analysis/evening_winddown.py                # Today
    python3 analysis/evening_winddown.py 2026-03-14     # Specific date
    python3 analysis/evening_winddown.py --json         # JSON output
    python3 analysis/evening_winddown.py --dry-run      # Print without sending

"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Constants ────────────────────────────────────────────────────────────────

# Windows where hour_of_day < MORNING_START are pre-work (ignored for day type)
MORNING_START_HOUR = 8
# Windows where hour_of_day < MIDDAY_HOUR are "morning"
MIDDAY_HOUR = 13
# Windows where hour_of_day < EVENING_HOUR are "afternoon"
EVENING_HOUR = 18

# Minimum active windows to consider the check-in meaningful
MIN_ACTIVE_WINDOWS = 3

# Day type thresholds
DT_FDI_HIGH      = 0.75   # FDI threshold for DEEP / PRODUCTIVE
DT_FDI_MED       = 0.55   # FDI threshold below which we lean REACTIVE/FRAGMENTED
DT_CLS_LOW       = 0.20   # CLS below this = RECOVERY
DT_SDI_HIGH      = 0.55   # SDI above this → REACTIVE
DT_CSC_HIGH      = 0.35   # CSC above this → FRAGMENTED
DT_SLACK_HIGH    = 30     # Slack messages sent above this → REACTIVE
DT_MTG_HIGH_MIN  = 180    # Meeting minutes above this → REACTIVE

# Load arc thresholds
ARC_DELTA_BACK   =  0.15  # afternoon_cls - morning_cls > this → "back-loaded"
ARC_DELTA_FRONT  = -0.15  # afternoon_cls - morning_cls < this → "front-loaded"

# Day types
DAY_TYPES = {
    "PRODUCTIVE": {
        "label": "Productive",
        "emoji": "✅",
        "description": "Good balance of focus and manageable load",
    },
    "DEEP": {
        "label": "Deep Work",
        "emoji": "🔵",
        "description": "High-quality, uninterrupted cognitive work",
    },
    "REACTIVE": {
        "label": "Reactive",
        "emoji": "🔴",
        "description": "High social/meeting load dominated the day",
    },
    "FRAGMENTED": {
        "label": "Fragmented",
        "emoji": "🟡",
        "description": "Frequent context-switching disrupted focus",
    },
    "RECOVERY": {
        "label": "Recovery",
        "emoji": "🟢",
        "description": "Light-load day — body and mind getting a rest",
    },
    "MIXED": {
        "label": "Mixed",
        "emoji": "⚪",
        "description": "Varied intensity without a dominant pattern",
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _avg(vals: list) -> Optional[float]:
    """Safe mean of a list of floats."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _classify_day_type(
    fdi: Optional[float],
    cls: Optional[float],
    sdi: Optional[float],
    csc: Optional[float],
    meeting_minutes: int,
    slack_sent: int,
) -> str:
    """
    Classify the day type from full-day averaged metrics.

    Priority order: RECOVERY → DEEP → REACTIVE → FRAGMENTED → PRODUCTIVE → MIXED
    """
    if cls is None:
        return "MIXED"

    # RECOVERY: very low load, not much happening
    if cls < DT_CLS_LOW:
        return "RECOVERY"

    # DEEP: high FDI, not dominated by meetings/social
    if (fdi is not None and fdi >= DT_FDI_HIGH
            and meeting_minutes < DT_MTG_HIGH_MIN
            and slack_sent < DT_SLACK_HIGH):
        return "DEEP"

    # REACTIVE: social/meeting dominated
    reactive_signals = 0
    if sdi is not None and sdi >= DT_SDI_HIGH:
        reactive_signals += 1
    if slack_sent >= DT_SLACK_HIGH:
        reactive_signals += 1
    if meeting_minutes >= DT_MTG_HIGH_MIN:
        reactive_signals += 1
    if fdi is not None and fdi < DT_FDI_MED:
        reactive_signals += 1

    if reactive_signals >= 2:
        return "REACTIVE"

    # FRAGMENTED: context-switching disrupted focus
    if (csc is not None and csc >= DT_CSC_HIGH
            and fdi is not None and fdi < DT_FDI_MED):
        return "FRAGMENTED"

    # PRODUCTIVE: good FDI, manageable load
    if fdi is not None and fdi >= (DT_FDI_HIGH - 0.10):  # ≥ 0.65
        return "PRODUCTIVE"

    return "MIXED"


def _compute_load_arc(
    morning_cls: Optional[float],
    afternoon_cls: Optional[float],
) -> str:
    """Compute load arc from morning vs afternoon CLS."""
    if morning_cls is None and afternoon_cls is None:
        return "even"
    if morning_cls is None:
        return "back-loaded"
    if afternoon_cls is None:
        return "front-loaded"

    delta = afternoon_cls - morning_cls
    if delta > ARC_DELTA_BACK:
        return "back-loaded"
    elif delta < ARC_DELTA_FRONT:
        return "front-loaded"
    return "even"


def _build_wind_down_recommendation(
    day_type: str,
    load_arc: str,
    recovery_score: Optional[float],
    full_day_cls: Optional[float],
) -> tuple[str, str]:
    """
    Return (recommendation_line, detail_text) for the wind-down recommendation.

    recommendation_line: short, action-oriented (e.g. "Step away from screens now.")
    detail_text: 1–2 sentence context
    """
    high_recovery = recovery_score is not None and recovery_score >= 60.0
    back_loaded = load_arc == "back-loaded"
    high_cls = full_day_cls is not None and full_day_cls >= 0.50

    if day_type == "REACTIVE":
        rec = "Social recovery mode: put the phone down, no more pings."
        detail = (
            "Your nervous system absorbed a lot of social and meeting load today. "
            "The best thing you can do now is create silence — no Slack, no notifications, "
            "even a short walk without podcasts."
        )

    elif day_type == "FRAGMENTED":
        rec = "Finish one small, clean task before you close everything."
        detail = (
            "Fragmented days often end without closure, which creates lingering "
            "cognitive overhead overnight. Pick the smallest open loop and close it "
            "so your mind can actually stop."
        )

    elif day_type == "DEEP":
        if high_recovery:
            rec = "Strong focus day — protect the evening to let it consolidate."
        else:
            rec = "Deep work took a physiological toll — prioritise rest tonight."
        detail = (
            "Deep focus work consumes significant cognitive resources even when it "
            "doesn't feel tiring. The gains you made today consolidate during sleep, "
            "so guard this evening carefully."
        )

    elif day_type == "PRODUCTIVE":
        if back_loaded:
            rec = "Step away now — afternoon load is rising, cut it here."
            detail = (
                "Your load has been building through the afternoon. If you push into "
                "the evening, you'll pay it back in tomorrow's recovery. A clear cutoff "
                "now is the productive choice."
            )
        elif high_recovery:
            rec = "Good day — a brief tomorrow-planning session is fine, then close."
            detail = (
                "You built a good output day without overextending. A 10-minute planning "
                "pass for tomorrow works here — then genuinely close the laptop."
            )
        else:
            rec = "Good output today — protect sleep to extend the streak."
            detail = (
                "Solid day on limited recovery. The way to make tomorrow equally "
                "productive is to not squeeze more out of today's tank."
            )

    elif day_type == "RECOVERY":
        rec = "Recovery day confirmed — aim for an early sleep tonight."
        detail = (
            "Low-load days are valuable; they're when physiological restoration "
            "happens. Lean into it — early wind-down, minimal screens, earlier bedtime."
        )

    else:  # MIXED
        if back_loaded:
            rec = "The afternoon pushed higher — step away, don't extend it further."
            detail = (
                "Back-loaded days carry momentum that's hard to break. The best "
                "pattern-interrupt is a hard stop now."
            )
        elif high_cls:
            rec = "Demanding day overall — protect the evening as genuine downtime."
            detail = (
                "Mixed days can feel inconclusive, which tempts you to squeeze in "
                "more. But the load was real — give your system a real break."
            )
        else:
            rec = "Wrap up and close — a clean finish beats a scattered one."
            detail = (
                "Closing the day with intention (not just running out of energy) "
                "improves tomorrow's start quality."
            )

    return rec, detail


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class EveningWindDown:
    """Evening wind-down signal for a given date."""
    date_str: str

    # Day classification
    day_type: str                          # PRODUCTIVE | DEEP | REACTIVE | FRAGMENTED | RECOVERY | MIXED
    day_type_label: str                    # human-readable
    day_type_emoji: str                    # emoji

    # Load arc
    load_arc: str                          # "front-loaded" | "back-loaded" | "even"

    # Metrics (full day up to EVENING_HOUR)
    morning_cls: Optional[float]           # 08:00–13:00 active CLS
    afternoon_cls: Optional[float]         # 13:00–18:00 active CLS
    full_day_cls: Optional[float]          # avg CLS all active windows
    full_day_fdi: Optional[float]          # avg FDI all active windows
    full_day_sdi: Optional[float]          # avg SDI all active windows
    full_day_csc: Optional[float]          # avg CSC all active windows

    # Activity
    total_meeting_minutes: int
    slack_sent: int
    active_windows_count: int

    # WHOOP
    recovery_score: Optional[float]

    # Projection
    projected_dps: Optional[float]         # DPS estimate from partial-day data

    # Recommendation
    wind_down_recommendation: str          # one-line action
    wind_down_detail: str                  # 2-sentence context

    is_meaningful: bool

    def to_dict(self) -> dict:
        return {
            "date_str": self.date_str,
            "day_type": self.day_type,
            "day_type_label": self.day_type_label,
            "day_type_emoji": self.day_type_emoji,
            "load_arc": self.load_arc,
            "morning_cls": round(self.morning_cls, 3) if self.morning_cls is not None else None,
            "afternoon_cls": round(self.afternoon_cls, 3) if self.afternoon_cls is not None else None,
            "full_day_cls": round(self.full_day_cls, 3) if self.full_day_cls is not None else None,
            "full_day_fdi": round(self.full_day_fdi, 3) if self.full_day_fdi is not None else None,
            "full_day_sdi": round(self.full_day_sdi, 3) if self.full_day_sdi is not None else None,
            "full_day_csc": round(self.full_day_csc, 3) if self.full_day_csc is not None else None,
            "total_meeting_minutes": self.total_meeting_minutes,
            "slack_sent": self.slack_sent,
            "active_windows_count": self.active_windows_count,
            "recovery_score": self.recovery_score,
            "projected_dps": round(self.projected_dps, 1) if self.projected_dps is not None else None,
            "wind_down_recommendation": self.wind_down_recommendation,
            "wind_down_detail": self.wind_down_detail,
            "is_meaningful": self.is_meaningful,
        }


# ─── Core computation ─────────────────────────────────────────────────────────

def compute_evening_winddown(
    date_str: str,
    windows: Optional[list[dict]] = None,
) -> "EveningWindDown":
    """
    Compute the evening wind-down signal for a given date.

    Parameters
    ----------
    date_str : str
        Date (YYYY-MM-DD).
    windows : list[dict] | None
        Pre-loaded day windows. If None, loaded from store.

    Returns
    -------
    EveningWindDown
        Always returns a valid object. is_meaningful=False when data is sparse.
    """
    _EMPTY = EveningWindDown(
        date_str=date_str,
        day_type="MIXED",
        day_type_label=DAY_TYPES["MIXED"]["label"],
        day_type_emoji=DAY_TYPES["MIXED"]["emoji"],
        load_arc="even",
        morning_cls=None,
        afternoon_cls=None,
        full_day_cls=None,
        full_day_fdi=None,
        full_day_sdi=None,
        full_day_csc=None,
        total_meeting_minutes=0,
        slack_sent=0,
        active_windows_count=0,
        recovery_score=None,
        projected_dps=None,
        wind_down_recommendation="No data available — check in later.",
        wind_down_detail="",
        is_meaningful=False,
    )

    # ── Load windows ──────────────────────────────────────────────────────
    if windows is None:
        try:
            from engine.store import read_day
            windows = read_day(date_str)
        except Exception:
            return _EMPTY

    if not windows:
        return _EMPTY

    # ── Filter to workday windows (up to EVENING_HOUR) ────────────────────
    workday_windows = [
        w for w in windows
        if MORNING_START_HOUR <= w["metadata"]["hour_of_day"] < EVENING_HOUR
    ]

    # ── Identify active windows ───────────────────────────────────────────
    def _is_active(w: dict) -> bool:
        return (
            w["metadata"].get("is_active_window", False)
            or w["calendar"]["in_meeting"]
            or w["slack"]["total_messages"] > 0
            or w.get("rescuetime", {}).get("active_seconds", 0) > 0
        )

    active_windows = [w for w in workday_windows if _is_active(w)]

    if len(active_windows) < MIN_ACTIVE_WINDOWS:
        return _EMPTY

    # ── Morning vs afternoon split ────────────────────────────────────────
    morning_active = [w for w in active_windows if w["metadata"]["hour_of_day"] < MIDDAY_HOUR]
    afternoon_active = [w for w in active_windows if w["metadata"]["hour_of_day"] >= MIDDAY_HOUR]

    # ── Metric extraction helpers ─────────────────────────────────────────
    def _cls_vals(ws: list) -> list:
        return [w["metrics"]["cognitive_load_score"] for w in ws
                if w["metrics"].get("cognitive_load_score") is not None]

    def _fdi_vals(ws: list) -> list:
        return [w["metrics"]["focus_depth_index"] for w in ws
                if w["metrics"].get("focus_depth_index") is not None]

    def _sdi_vals(ws: list) -> list:
        return [w["metrics"]["social_drain_index"] for w in ws
                if w["metrics"].get("social_drain_index") is not None]

    def _csc_vals(ws: list) -> list:
        return [w["metrics"]["context_switch_cost"] for w in ws
                if w["metrics"].get("context_switch_cost") is not None]

    # ── Compute metrics ───────────────────────────────────────────────────
    morning_cls   = _avg(_cls_vals(morning_active))
    afternoon_cls = _avg(_cls_vals(afternoon_active))
    full_day_cls  = _avg(_cls_vals(active_windows))
    full_day_fdi  = _avg(_fdi_vals(active_windows))
    full_day_sdi  = _avg(_sdi_vals(active_windows))
    full_day_csc  = _avg(_csc_vals(active_windows))

    # ── Meeting minutes (entire workday) ──────────────────────────────────
    total_meeting_minutes = sum(
        15 for w in workday_windows if w["calendar"]["in_meeting"]
    )

    # ── Slack messages sent ───────────────────────────────────────────────
    slack_sent = sum(w["slack"].get("messages_sent", 0) for w in workday_windows)

    # ── WHOOP recovery (from first window that has it) ────────────────────
    recovery_score: Optional[float] = None
    for w in windows:
        rs = w.get("whoop", {}).get("recovery_score")
        if rs is not None:
            recovery_score = rs
            break

    # ── Day type classification ───────────────────────────────────────────
    day_type = _classify_day_type(
        fdi=full_day_fdi,
        cls=full_day_cls,
        sdi=full_day_sdi,
        csc=full_day_csc,
        meeting_minutes=total_meeting_minutes,
        slack_sent=slack_sent,
    )

    # ── Load arc ──────────────────────────────────────────────────────────
    load_arc = _compute_load_arc(morning_cls, afternoon_cls)

    # ── Projected DPS (simple linear projection from active windows) ──────
    projected_dps: Optional[float] = None
    try:
        from analysis.presence_score import compute_presence_score
        score = compute_presence_score(active_windows)
        if score.is_meaningful:
            projected_dps = score.dps
    except Exception:
        pass

    # ── Wind-down recommendation ──────────────────────────────────────────
    rec, detail = _build_wind_down_recommendation(
        day_type=day_type,
        load_arc=load_arc,
        recovery_score=recovery_score,
        full_day_cls=full_day_cls,
    )

    dt_info = DAY_TYPES[day_type]
    return EveningWindDown(
        date_str=date_str,
        day_type=day_type,
        day_type_label=dt_info["label"],
        day_type_emoji=dt_info["emoji"],
        load_arc=load_arc,
        morning_cls=morning_cls,
        afternoon_cls=afternoon_cls,
        full_day_cls=full_day_cls,
        full_day_fdi=full_day_fdi,
        full_day_sdi=full_day_sdi,
        full_day_csc=full_day_csc,
        total_meeting_minutes=total_meeting_minutes,
        slack_sent=slack_sent,
        active_windows_count=len(active_windows),
        recovery_score=recovery_score,
        projected_dps=projected_dps,
        wind_down_recommendation=rec,
        wind_down_detail=detail,
        is_meaningful=True,
    )


# ─── Formatting ───────────────────────────────────────────────────────────────

def _fmt_minutes(m: int) -> str:
    """Format minutes as '1h30m' or '45min'."""
    if m >= 60:
        h = m // 60
        r = m % 60
        return f"{h}h{r:02d}min" if r else f"{h}h"
    return f"{m}min"


def _fmt_cls(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.0f}%"


def _fmt_fdi(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.0f}%"


def _cls_label(cls: Optional[float]) -> str:
    if cls is None:
        return "unknown"
    if cls < 0.15:
        return "very light"
    if cls < 0.30:
        return "light"
    if cls < 0.50:
        return "moderate"
    if cls < 0.70:
        return "heavy"
    return "intense"


def _arc_description(arc: str) -> str:
    return {
        "front-loaded": "peaked in the morning",
        "back-loaded":  "rose through the afternoon",
        "even":         "consistent throughout",
        "spiky":        "uneven with spikes",
    }.get(arc, arc)


def format_winddown_message(winddown: "EveningWindDown") -> str:
    """
    Format the evening wind-down into a Slack DM.

    Designed to be read in under 20 seconds at 6pm.
    Compact, action-forward, no fluff.
    """
    if not winddown or not winddown.is_meaningful:
        return "🌆 *Evening Check-In* — not enough data yet for today."

    try:
        dt = datetime.strptime(winddown.date_str, "%Y-%m-%d")
        date_label = dt.strftime("%A, %B %-d")
    except ValueError:
        date_label = winddown.date_str

    lines = [
        f"*Evening Wind-Down — {date_label}*",
        "",
    ]

    # ── Day type headline ──
    emoji = winddown.day_type_emoji
    label = winddown.day_type_label
    arc_desc = _arc_description(winddown.load_arc)

    # Build the headline line
    dps_str = f" · DPS {winddown.projected_dps:.0f}" if winddown.projected_dps is not None else ""
    lines.append(f"{emoji} *{label} Day*{dps_str}  _{arc_desc}_")

    # ── Metrics row ──
    metric_parts = []
    if winddown.full_day_cls is not None:
        metric_parts.append(f"Load {_fmt_cls(winddown.full_day_cls)} ({_cls_label(winddown.full_day_cls)})")
    if winddown.full_day_fdi is not None:
        metric_parts.append(f"Focus {_fmt_fdi(winddown.full_day_fdi)}")
    if winddown.total_meeting_minutes > 0:
        metric_parts.append(f"Meetings {_fmt_minutes(winddown.total_meeting_minutes)}")
    if metric_parts:
        lines.append("_" + "  ·  ".join(metric_parts) + "_")

    # ── AM/PM split (if both exist and arc is interesting) ──
    if (winddown.morning_cls is not None
            and winddown.afternoon_cls is not None
            and winddown.load_arc != "even"):
        am_str = _fmt_cls(winddown.morning_cls)
        pm_str = _fmt_cls(winddown.afternoon_cls)
        arrow = "↗" if winddown.load_arc == "back-loaded" else "↘"
        lines.append(f"_AM {am_str} {arrow} PM {pm_str}_")

    lines.append("")

    # ── Recommendation ──
    lines.append(f"*Now:* {winddown.wind_down_recommendation}")

    # ── Detail ──
    if winddown.wind_down_detail:
        lines.append(f"_{winddown.wind_down_detail}_")

    return "\n".join(lines)


# ─── Slack delivery ───────────────────────────────────────────────────────────

def _send_slack_dm(message: str) -> bool:
    """Send a Slack DM via the Alfred gateway."""
    try:
        import urllib.request

        from config import GATEWAY_URL, GATEWAY_TOKEN, SLACK_DM_CHANNEL

        payload = json.dumps({
            "tool": "message",
            "arguments": {
                "action": "send",
                "target": SLACK_DM_CHANNEL,
                "message": message,
            },
        }).encode()

        req = urllib.request.Request(
            f"{GATEWAY_URL}/invoke",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GATEWAY_TOKEN}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[evening_winddown] Slack delivery failed: {e}", file=sys.stderr)
        return False


def send_evening_winddown(date_str: Optional[str] = None) -> bool:
    """
    Compute and send the evening wind-down for the given date.

    Returns True when the message was sent, False otherwise.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    winddown = compute_evening_winddown(date_str)
    if not winddown.is_meaningful:
        print(f"[evening_winddown] Skipping {date_str} — not enough data (< {MIN_ACTIVE_WINDOWS} active windows).")
        return False

    message = format_winddown_message(winddown)
    success = _send_slack_dm(message)
    if success:
        print(f"[evening_winddown] Sent evening wind-down for {date_str}")
    else:
        print(f"[evening_winddown] Failed to send for {date_str}")
        # Print the message so it's not lost
        print(message)
    return success


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evening Wind-Down signal")
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of formatted text.")
    parser.add_argument("--dry-run", action="store_true", help="Print message without sending to Slack.")
    parser.add_argument("--send", action="store_true", help="Send to Slack (default: dry-run).")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    winddown = compute_evening_winddown(date_str)

    if args.json:
        print(json.dumps(winddown.to_dict(), indent=2))
        return

    if not winddown.is_meaningful:
        print(f"Evening Wind-Down: not enough data for {date_str}.")
        return

    message = format_winddown_message(winddown)
    print(message)

    if args.send and not args.dry_run:
        success = _send_slack_dm(message)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

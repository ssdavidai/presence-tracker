"""
Presence Tracker — Midday Cognitive Check-In (v28)

Answers: *"How is my morning going versus plan?"*

The morning brief (07:00) gives David a forward-looking readiness signal.
The nightly digest (23:45) reviews the full day in retrospect.

But there's a 16-hour gap with no signal. The Midday Check-In closes it.

It fires at 13:00 Budapest, ingests the partial-day data (00:00–13:00),
and sends a brief Slack pulse with:

  1. Morning load so far     — actual CLS for the last 4–5 hours of work
  2. Load vs plan            — actual vs predicted (from morning brief forecast)
  3. Focus quality           — how deep was the morning focus?
  4. Afternoon recommendation — one concrete nudge based on current state
  5. Remaining budget        — how much quality cognitive time is left today

## Design

The check-in is intentionally brief — it must not overwhelm David mid-day.
Target: 4–6 lines in Slack. Compare with the morning brief (15–20 lines).

Key constraints:
  - Uses only data already ingested (no new API calls from this module)
  - Reads partial-day JSONL up to the current hour
  - Degrades gracefully when morning data is sparse (e.g. Monday mornings)
  - Never sends if there are fewer than 3 active windows in the morning

## Midday metrics

    morning_windows = windows where hour_of_day < MIDDAY_HOUR (13)
    active_windows  = morning_windows where is_active_window=True

    morning_cls     = mean CLS across active morning windows
    morning_fdi     = mean FDI across active morning windows
    morning_sdi     = mean SDI across active morning windows (social drain)
    meeting_minutes = total minutes in meetings before 13:00

    pace_ratio      = morning_cls / daily_baseline_cls
                      > 1.25 → "Running hot"
                      0.75–1.25 → "On track"
                      < 0.75 → "Light morning"

    remaining_budget = compute_cognitive_budget() × (1 - morning_fraction)
                       where morning_fraction = active_morning_hours / DCB_hours

## Output

    MidDayCheckIn dataclass:
      - date_str: str
      - morning_cls: float | None
      - morning_fdi: float | None
      - morning_sdi: float | None
      - meeting_minutes: int
      - active_windows: int
      - pace_label: str          — "Running hot" | "On track" | "Light morning"
      - pace_ratio: float | None
      - afternoon_nudge: str     — one actionable sentence
      - remaining_budget_hours: float | None
      - is_meaningful: bool

## API

    from analysis.midday_checkin import compute_midday_checkin, format_checkin_message

    checkin = compute_midday_checkin(date_str)
    message = format_checkin_message(checkin)   # Slack-ready string

## CLI

    python3 analysis/midday_checkin.py                # Today
    python3 analysis/midday_checkin.py 2026-03-14     # Specific date
    python3 analysis/midday_checkin.py --json         # JSON output
    python3 analysis/midday_checkin.py --dry-run      # Print without sending

"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Constants ────────────────────────────────────────────────────────────────

# Windows with hour_of_day < MIDDAY_HOUR are considered "morning"
MIDDAY_HOUR = 13

# Minimum active windows in the morning to consider the check-in meaningful
MIN_ACTIVE_WINDOWS = 3

# Pace thresholds — morning_cls relative to 7-day baseline cls
PACE_HOT_RATIO    = 1.25   # > this → "Running hot"
PACE_LIGHT_RATIO  = 0.75   # < this → "Light morning"

# CLS labels (used in pace narrative)
CLS_LABELS = [
    (0.00, 0.15, "very light"),
    (0.15, 0.30, "light"),
    (0.30, 0.50, "moderate"),
    (0.50, 0.70, "heavy"),
    (0.70, 1.00, "intense"),
]

# FDI quality labels (for active hours)
FDI_LABELS = [
    (0.80, 1.00, "deep"),
    (0.60, 0.80, "solid"),
    (0.40, 0.60, "fragmented"),
    (0.00, 0.40, "scattered"),
]

# Afternoon time blocks for recommendations
AFTERNOON_BLOCKS = [
    (13, 15, "early afternoon"),
    (15, 17, "mid-afternoon"),
    (17, 19, "late afternoon"),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _cls_label(cls: Optional[float]) -> str:
    if cls is None:
        return "unknown"
    for lo, hi, label in CLS_LABELS:
        if lo <= cls < hi:
            return label
    return "intense"


def _fdi_label(fdi: Optional[float]) -> str:
    if fdi is None:
        return "unknown"
    for lo, hi, label in FDI_LABELS:
        if lo <= fdi < hi:
            return label
    return "scattered"


def _fmt_minutes(m: int) -> str:
    """Format minutes as '1h30m' or '45min'."""
    if m >= 60:
        h = m // 60
        r = m % 60
        return f"{h}h{r:02d}min" if r else f"{h}h"
    return f"{m}min"


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class MidDayCheckIn:
    """Midday cognitive check-in for a given date."""
    date_str: str
    morning_cls: Optional[float]         # mean CLS for active morning windows
    morning_fdi: Optional[float]         # mean FDI for active morning windows
    morning_sdi: Optional[float]         # mean SDI for active morning windows
    meeting_minutes: int                 # total meeting minutes before MIDDAY_HOUR
    active_windows: int                  # number of active morning windows
    pace_label: str                      # "Running hot" | "On track" | "Light morning"
    pace_ratio: Optional[float]          # morning_cls / baseline_cls
    afternoon_nudge: str                 # one concrete recommendation
    remaining_budget_hours: Optional[float]  # estimated quality hours left today
    is_meaningful: bool                  # False when < MIN_ACTIVE_WINDOWS

    def to_dict(self) -> dict:
        return {
            "date_str": self.date_str,
            "morning_cls": round(self.morning_cls, 3) if self.morning_cls is not None else None,
            "morning_fdi": round(self.morning_fdi, 3) if self.morning_fdi is not None else None,
            "morning_sdi": round(self.morning_sdi, 3) if self.morning_sdi is not None else None,
            "meeting_minutes": self.meeting_minutes,
            "active_windows": self.active_windows,
            "pace_label": self.pace_label,
            "pace_ratio": round(self.pace_ratio, 2) if self.pace_ratio is not None else None,
            "afternoon_nudge": self.afternoon_nudge,
            "remaining_budget_hours": (
                round(self.remaining_budget_hours, 1)
                if self.remaining_budget_hours is not None
                else None
            ),
            "is_meaningful": self.is_meaningful,
        }


# ─── Core computation ─────────────────────────────────────────────────────────

def compute_midday_checkin(
    date_str: str,
    windows: Optional[list[dict]] = None,
    baseline_cls: Optional[float] = None,
    dcb_hours: Optional[float] = None,
) -> "MidDayCheckIn":
    """
    Compute the midday cognitive check-in for a given date.

    Parameters
    ----------
    date_str : str
        Date (YYYY-MM-DD).
    windows : list[dict] | None
        Pre-loaded day windows. If None, loaded from store.
    baseline_cls : float | None
        Personal 7-day baseline CLS for pace comparison.
        If None, loaded from store or falls back to population norm (0.25).
    dcb_hours : float | None
        Today's Daily Cognitive Budget (quality hours). Used to estimate
        remaining budget after the morning.  If None, computed from store.

    Returns
    -------
    MidDayCheckIn
        Always returns a valid object. is_meaningful=False when not enough
        morning data exists.
    """
    # ── Load windows ──────────────────────────────────────────────────────
    if windows is None:
        try:
            from engine.store import read_day
            windows = read_day(date_str)
        except Exception:
            windows = []

    if not windows:
        return MidDayCheckIn(
            date_str=date_str,
            morning_cls=None,
            morning_fdi=None,
            morning_sdi=None,
            meeting_minutes=0,
            active_windows=0,
            pace_label="Light morning",
            pace_ratio=None,
            afternoon_nudge="No data available for today yet.",
            remaining_budget_hours=None,
            is_meaningful=False,
        )

    # ── Filter morning windows ────────────────────────────────────────────
    morning_windows = [
        w for w in windows
        if w["metadata"]["hour_of_day"] < MIDDAY_HOUR
    ]
    active_morning = [
        w for w in morning_windows
        if w["metadata"].get("is_active_window", False)
        or w["calendar"]["in_meeting"]
        or w["slack"]["total_messages"] > 0
        or w.get("rescuetime", {}).get("active_seconds", 0) > 0
        or w.get("omi", {}) != {}
    ]

    # ── Check meaningfulness ──────────────────────────────────────────────
    if len(active_morning) < MIN_ACTIVE_WINDOWS:
        return MidDayCheckIn(
            date_str=date_str,
            morning_cls=None,
            morning_fdi=None,
            morning_sdi=None,
            meeting_minutes=0,
            active_windows=len(active_morning),
            pace_label="Light morning",
            pace_ratio=None,
            afternoon_nudge="Morning is still quiet — no check-in data yet.",
            remaining_budget_hours=None,
            is_meaningful=False,
        )

    # ── Morning metrics ───────────────────────────────────────────────────
    cls_vals = [
        w["metrics"]["cognitive_load_score"]
        for w in active_morning
        if w["metrics"].get("cognitive_load_score") is not None
    ]
    fdi_vals = [
        w["metrics"]["focus_depth_index"]
        for w in active_morning
        if w["metrics"].get("focus_depth_index") is not None
    ]
    sdi_vals = [
        w["metrics"]["social_drain_index"]
        for w in active_morning
        if w["metrics"].get("social_drain_index") is not None
    ]

    morning_cls = sum(cls_vals) / len(cls_vals) if cls_vals else None
    morning_fdi = sum(fdi_vals) / len(fdi_vals) if fdi_vals else None
    morning_sdi = sum(sdi_vals) / len(sdi_vals) if sdi_vals else None

    # ── Meeting minutes before midday ─────────────────────────────────────
    meeting_minutes = sum(
        15 for w in morning_windows if w["calendar"]["in_meeting"]
    )

    # ── Pace ratio ────────────────────────────────────────────────────────
    # Compare morning CLS to personal baseline or population norm
    if baseline_cls is None:
        baseline_cls = _load_baseline_cls(date_str)

    pace_ratio: Optional[float] = None
    if morning_cls is not None and baseline_cls is not None and baseline_cls > 0:
        pace_ratio = morning_cls / baseline_cls

    # ── Pace label ────────────────────────────────────────────────────────
    if pace_ratio is None:
        pace_label = "On track"
    elif pace_ratio > PACE_HOT_RATIO:
        pace_label = "Running hot"
    elif pace_ratio < PACE_LIGHT_RATIO:
        pace_label = "Light morning"
    else:
        pace_label = "On track"

    # ── Remaining budget ──────────────────────────────────────────────────
    remaining_budget_hours: Optional[float] = None
    if dcb_hours is None:
        dcb_hours = _load_dcb_hours(date_str, windows)

    if dcb_hours is not None and morning_cls is not None:
        # Estimate consumed budget: active morning hours × load intensity
        # Each active window = 15min = 0.25h; intensity = cls / 0.5 (midpoint)
        morning_hours = len(active_morning) * 0.25
        intensity = min(morning_cls / 0.5, 1.5)  # clamped: 1.5× max
        consumed = morning_hours * intensity
        remaining_budget_hours = max(0.0, round(dcb_hours - consumed, 1))

    # ── Afternoon nudge ───────────────────────────────────────────────────
    afternoon_nudge = _build_afternoon_nudge(
        morning_cls=morning_cls,
        morning_fdi=morning_fdi,
        morning_sdi=morning_sdi,
        pace_label=pace_label,
        meeting_minutes=meeting_minutes,
        remaining_budget_hours=remaining_budget_hours,
    )

    return MidDayCheckIn(
        date_str=date_str,
        morning_cls=round(morning_cls, 3) if morning_cls is not None else None,
        morning_fdi=round(morning_fdi, 3) if morning_fdi is not None else None,
        morning_sdi=round(morning_sdi, 3) if morning_sdi is not None else None,
        meeting_minutes=meeting_minutes,
        active_windows=len(active_morning),
        pace_label=pace_label,
        pace_ratio=round(pace_ratio, 2) if pace_ratio is not None else None,
        afternoon_nudge=afternoon_nudge,
        remaining_budget_hours=remaining_budget_hours,
        is_meaningful=True,
    )


# ─── Loaders ──────────────────────────────────────────────────────────────────

def _load_baseline_cls(date_str: str) -> Optional[float]:
    """
    Load the 7-day average CLS as a personal baseline for pace comparison.
    Falls back to a population-norm default (0.25) when insufficient history.
    """
    try:
        from engine.store import get_recent_summaries
        summaries = get_recent_summaries(days=7)
        cls_vals = [
            s["metrics_avg"]["cognitive_load_score"]
            for s in summaries
            if s.get("date") != date_str
            and s.get("metrics_avg", {}).get("cognitive_load_score") is not None
        ]
        if cls_vals:
            return sum(cls_vals) / len(cls_vals)
    except Exception:
        pass
    # Population-norm fallback: moderate-ish daytime load
    return 0.25


def _load_dcb_hours(date_str: str, windows: list[dict]) -> Optional[float]:
    """
    Load today's Daily Cognitive Budget (quality hours).
    Uses compute_cognitive_budget() with the first window's WHOOP data.
    """
    try:
        from analysis.cognitive_budget import compute_cognitive_budget
        from analysis.cognitive_debt import compute_cdi
        from analysis.personal_baseline import get_personal_baseline

        whoop_data = windows[0].get("whoop") if windows else None
        cdi_tier = None
        hrv_baseline = None

        try:
            debt = compute_cdi(date_str)
            if debt.is_meaningful:
                cdi_tier = debt.tier
        except Exception:
            pass

        try:
            baseline = get_personal_baseline()
            hrv_baseline = baseline.hrv_mean
        except Exception:
            pass

        budget = compute_cognitive_budget(
            date_str=date_str,
            whoop_data=whoop_data,
            cdi_tier=cdi_tier,
            hrv_baseline=hrv_baseline,
        )
        if budget.is_meaningful:
            return budget.dcb_hours
    except Exception:
        pass
    return None


# ─── Afternoon nudge builder ──────────────────────────────────────────────────

def _build_afternoon_nudge(
    morning_cls: Optional[float],
    morning_fdi: Optional[float],
    morning_sdi: Optional[float],
    pace_label: str,
    meeting_minutes: int,
    remaining_budget_hours: Optional[float],
) -> str:
    """
    Build one concrete afternoon recommendation from the morning signals.

    Logic:
      - "Running hot" + high SDI → protect the afternoon from social load
      - "Running hot" + low FDI → schedule a focus block before more meetings
      - "On track" + budget > 2h → ideal window for a 90-min deep-work block
      - "Light morning" → good conditions for deep work now
      - Low FDI overall → fragmented morning, avoid stacking more shallow tasks
    """
    # Running hot — warn and suggest recovery
    if pace_label == "Running hot":
        if morning_sdi is not None and morning_sdi > 0.40:
            return (
                "High social load this morning — protect early afternoon from more meetings. "
                "Async-only for 60–90min if possible."
            )
        if morning_fdi is not None and morning_fdi < 0.50:
            return (
                "Morning was fragmented and intense. "
                "Use 13:00–14:30 for one focused task before the afternoon starts."
            )
        return (
            "Running hot — pace yourself this afternoon. "
            "Avoid stacking decisions; delegate or defer low-value items."
        )

    # Light morning — encourage deep work
    if pace_label == "Light morning":
        if remaining_budget_hours is not None and remaining_budget_hours >= 2.0:
            return (
                f"Light morning with ~{remaining_budget_hours:.1f}h budget left — "
                "good conditions for a 90-min focused block now."
            )
        return (
            "Light morning — ideal time to tackle your hardest open problem "
            "before the afternoon calendar fills up."
        )

    # On track — give a budget-aware suggestion
    if remaining_budget_hours is not None:
        if remaining_budget_hours >= 3.0:
            return (
                f"On track. ~{remaining_budget_hours:.1f}h of quality time remaining — "
                "protect a 90-min block this afternoon for deep work."
            )
        if remaining_budget_hours >= 1.5:
            return (
                f"On track. ~{remaining_budget_hours:.1f}h left — "
                "one focused session before end of day."
            )
        return (
            f"Budget nearly spent (~{remaining_budget_hours:.1f}h remaining). "
            "Switch to async/admin this afternoon; preserve energy."
        )

    # Fallback — meeting-load-aware
    if meeting_minutes >= 120:
        return (
            "Heavy meeting morning. Use any free afternoon block for focused work "
            "and avoid reactive tasks."
        )
    return "Morning steady — protect at least one 60-min block this afternoon for focused work."


# ─── Formatter ────────────────────────────────────────────────────────────────

def format_checkin_message(checkin: "MidDayCheckIn") -> str:
    """
    Format the midday check-in as a brief Slack message.

    Target: 4–6 lines. Concise, actionable.

    Example:
        ☀️ *Midday pulse — Monday 13:00*

        Morning load:   light (CLS 0.08) — deep focus quality
        Meetings:       45min
        Pace:           ✅ On track
        Budget left:    ~4.5h quality hours remaining

        → Protect a 90-min block this afternoon for deep work.
    """
    if not checkin.is_meaningful:
        return (
            f"☀️ *Midday pulse — {checkin.date_str}*\n"
            f"_No active windows recorded this morning yet._"
        )

    try:
        dt = datetime.strptime(checkin.date_str, "%Y-%m-%d")
        day_label = dt.strftime("%A")
    except ValueError:
        day_label = checkin.date_str

    lines = [f"☀️ *Midday pulse — {day_label}*", ""]

    # Morning load
    cls_str = f"CLS {checkin.morning_cls:.2f}" if checkin.morning_cls is not None else "N/A"
    fdi_str = f"{_fdi_label(checkin.morning_fdi)} focus quality"
    lines.append(f"Morning load:   {_cls_label(checkin.morning_cls)} ({cls_str}) — {fdi_str}")

    # Meetings this morning
    if checkin.meeting_minutes > 0:
        lines.append(f"Meetings:       {_fmt_minutes(checkin.meeting_minutes)}")

    # Pace indicator
    pace_emoji = {
        "Running hot":  "🔥",
        "On track":     "✅",
        "Light morning": "🟢",
    }.get(checkin.pace_label, "⚪")
    lines.append(f"Pace:           {pace_emoji} {checkin.pace_label}")

    # Remaining budget
    if checkin.remaining_budget_hours is not None:
        lines.append(
            f"Budget left:    ~{checkin.remaining_budget_hours:.1f}h quality hours remaining"
        )

    # Afternoon nudge
    lines.append("")
    lines.append(f"→ {checkin.afternoon_nudge}")

    return "\n".join(lines)


# ─── Send helper ──────────────────────────────────────────────────────────────

def send_midday_checkin(date_str: str) -> bool:
    """
    Compute and send the midday check-in DM to David.

    Returns True if the DM was sent successfully.
    """
    import urllib.request

    try:
        from config import GATEWAY_URL, GATEWAY_TOKEN, SLACK_DM_CHANNEL
    except ImportError:
        import os
        GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:9999")
        GATEWAY_TOKEN = os.environ.get("GATEWAY_TOKEN", "")
        SLACK_DM_CHANNEL = "U08UGBQL5J5"

    checkin = compute_midday_checkin(date_str)

    if not checkin.is_meaningful:
        print(
            f"[midday] Check-in not meaningful for {date_str} "
            f"({checkin.active_windows} active windows < {MIN_ACTIVE_WINDOWS}). "
            "Skipping.",
            file=sys.stderr,
        )
        return False

    message = format_checkin_message(checkin)

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
                "target": SLACK_DM_CHANNEL,
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
        print(f"[midday] Failed to send DM: {e}", file=sys.stderr)
        return False


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.

    Usage:
        python3 analysis/midday_checkin.py                  # Today
        python3 analysis/midday_checkin.py 2026-03-14       # Specific date
        python3 analysis/midday_checkin.py --json           # JSON output
        python3 analysis/midday_checkin.py --dry-run        # Print without sending
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Midday cognitive check-in — morning pulse + afternoon recommendation"
    )
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD), default = today")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print without sending")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    checkin = compute_midday_checkin(date_str)

    if args.json:
        print(json.dumps(checkin.to_dict(), indent=2))
        return

    if args.dry_run or True:  # always print in CLI mode
        print()
        print(f"Midday Check-In — {date_str}")
        print("=" * 50)
        if not checkin.is_meaningful:
            print(f"  Not meaningful — only {checkin.active_windows} active morning windows.")
        else:
            print(f"  Morning CLS:    {checkin.morning_cls:.3f}  ({_cls_label(checkin.morning_cls)})")
            print(f"  Morning FDI:    {checkin.morning_fdi:.3f}  ({_fdi_label(checkin.morning_fdi)})")
            if checkin.morning_sdi is not None:
                print(f"  Morning SDI:    {checkin.morning_sdi:.3f}")
            if checkin.meeting_minutes:
                print(f"  Meetings:       {_fmt_minutes(checkin.meeting_minutes)}")
            print(f"  Pace:           {checkin.pace_label}", end="")
            if checkin.pace_ratio is not None:
                print(f"  (×{checkin.pace_ratio:.2f} baseline)")
            else:
                print()
            if checkin.remaining_budget_hours is not None:
                print(f"  Budget left:    ~{checkin.remaining_budget_hours:.1f}h")
            print()
            print(f"  → {checkin.afternoon_nudge}")
        print()

    if not args.dry_run and checkin.is_meaningful:
        ok = send_midday_checkin(date_str)
        print(f"\n{'✓ Sent' if ok else '✗ Failed to send'} to David's DM")


if __name__ == "__main__":
    main()

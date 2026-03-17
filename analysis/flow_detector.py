"""
Presence Tracker — Flow State Detector (v35)

Answers: *"Did David actually hit flow today — and when?"*

WHOOP tracks physical strain and recovery.
CLS tracks how much cognitive pressure was on.
FDI tracks how deep the focus was.

But none of these together say: "You were in flow from 9:15–11:30."

The Flow State Detector identifies contiguous windows where the
physiological and behavioural signals align with the known markers
of cognitive flow — deep focus, sustainable load, minimal fragmentation.

## What is flow (operationally)?

Csikszentmihalyi's flow state has four measurable proxies in this system:

  1. **High focus depth** — FDI ≥ FDI_MIN (default 0.65)
     The person is not jumping between tasks; concentration is sustained.

  2. **Moderate cognitive load** — CLS in [CLS_MIN, CLS_MAX] (default 0.12–0.62)
     Not so low that it's idle (drifting attention), not so high that it's
     overwhelm. The "challenge-skill balance" zone.

  3. **Low fragmentation** — CSC ≤ CSC_MAX (default 0.35)
     Context switching is minimal — no app flipping, no rapid Slack ping-pong,
     no back-to-back meeting transitions.

  4. **Duration** — ≥ MIN_SESSION_WINDOWS consecutive qualifying windows
     (default 2 × 15-min windows = 30 minutes minimum).
     Flow requires time to establish; a single qualifying window is noise.

A **flow session** is a maximal run of consecutive qualifying windows
(each passing all four criteria above) that meets the minimum duration.

## Why not just use FDI?

High FDI alone can include:
  - Low-CLS idle reading (not really flow — just undisturbed)
  - Post-meeting exhausted quiet time (depleted, not focused)

High CLS alone is obviously not flow — it's overwhelm.

The combination of high FDI + moderate CLS + low CSC in a sustained block
is the true flow fingerprint: engaged enough to be challenged, focused enough
to sustain it, undisrupted enough to maintain momentum.

## Formula

For each working-hour window w:
    is_flow_candidate = (
        w.fdi >= FDI_MIN                      # deep focus
        and CLS_MIN <= w.cls <= CLS_MAX        # moderate load (challenge zone)
        and w.csc <= CSC_MAX                   # low fragmentation
        and w.is_working_hours                 # within work day
        and not w.in_social_meeting            # social meetings break flow
    )

Sessions: maximal runs of consecutive is_flow_candidate windows of length ≥ MIN_SESSION_WINDOWS.

    flow_sessions = [session for session in runs if len(session) >= MIN_SESSION_WINDOWS]
    total_flow_minutes = sum(len(s) × 15 for s in flow_sessions)
    peak_session = longest session by duration
    flow_score = min(total_flow_minutes / TARGET_FLOW_MINUTES, 1.0)   # 0–1

TARGET_FLOW_MINUTES = 120  (2 hours of flow is a "full" day)

## Interpretation

    flow_score ≥ 0.75   → Deep Flow     — exceptional focus day (≥ 90 min quality flow)
    0.50–0.75           → In the Zone   — solid focused work block (60–90 min)
    0.25–0.50           → Brief Flow    — some flow, but fragmented or short (30–60 min)
    < 0.25              → No Flow       — conditions didn't align for sustained focus

## Output

    FlowStateResult dataclass:
      - date_str: str
      - flow_sessions: list[FlowSession]   — each detected flow session
      - total_flow_minutes: int            — total minutes across all sessions
      - peak_session: FlowSession | None   — longest single flow session
      - flow_score: float                  — 0.0 (none) → 1.0 (full day in flow)
      - flow_label: str                    — 'deep_flow' | 'in_zone' | 'brief' | 'none'
      - windows_analysed: int              — total working-hour windows examined
      - candidate_windows: int             — windows that individually qualified
      - is_meaningful: bool                — False when < 3 working-hour windows
      - insight: str                       — one-line human explanation
      - peak_hour: int | None              — start hour of the longest flow session

    FlowSession dataclass:
      - start_time: str          — "HH:MM"
      - end_time: str            — "HH:MM"
      - duration_minutes: int
      - avg_fdi: float
      - avg_cls: float
      - avg_csc: float
      - window_count: int

## API

    from analysis.flow_detector import detect_flow_states, format_flow_line
    from analysis.flow_detector import format_flow_section

    result = detect_flow_states(windows)
    line    = format_flow_line(result)        # Compact one-liner for digest
    section = format_flow_section(result)     # Full Slack section

## Integration

    # In nightly digest (after FDI/LVI block):
    flow = detect_flow_states(windows)
    if flow.is_meaningful and flow.flow_score > 0:
        lines.append(format_flow_line(flow))

    # In weekly summary:
    for date in week_dates:
        windows = read_day(date)
        flow = detect_flow_states(windows)
        # Track weekly flow totals, best flow day

## CLI

    python3 analysis/flow_detector.py                 # Today
    python3 analysis/flow_detector.py 2026-03-14      # Specific date
    python3 analysis/flow_detector.py --json           # JSON output
    python3 analysis/flow_detector.py --week           # Last 7 days

## Design principles

  - Pure functions — fully testable with mock window data
  - No ML, no external dependencies — deterministic rule-based detection
  - Graceful degradation: insufficient data → is_meaningful = False
  - Social meetings are flow-breaking by definition; they are excluded
  - The CLS floor (0.12) filters idle/sleep windows where FDI appears high
    but no actual cognitive work is occurring
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Constants ────────────────────────────────────────────────────────────────

# Minimum FDI for a window to qualify as flow
FLOW_FDI_MIN: float = 0.65

# CLS range for the flow "challenge zone"
# Below this = idle/drifting; above this = overwhelm
FLOW_CLS_MIN: float = 0.12
FLOW_CLS_MAX: float = 0.62

# Maximum CSC for flow (above this = too fragmented)
FLOW_CSC_MAX: float = 0.35

# Minimum consecutive qualifying windows to count as a session
# 2 windows = 30 minutes minimum
MIN_SESSION_WINDOWS: int = 2

# "Full" flow target: 120 minutes = flow_score 1.0
TARGET_FLOW_MINUTES: int = 120

# Minimum working-hour windows to consider analysis meaningful
MIN_WORKING_WINDOWS: int = 3

# Flow label thresholds (flow_score → label)
DEEP_FLOW_THRESHOLD: float   = 0.75   # ≥ 90 min
IN_ZONE_THRESHOLD: float     = 0.50   # ≥ 60 min
BRIEF_FLOW_THRESHOLD: float  = 0.25   # ≥ 30 min


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class FlowSession:
    """A single continuous flow state session."""
    start_time: str          # "HH:MM" — window_start of first qualifying window
    end_time: str            # "HH:MM" — window_end of last qualifying window
    duration_minutes: int    # Total duration (15 × window_count)
    avg_fdi: float           # Mean FDI across this session
    avg_cls: float           # Mean CLS across this session
    avg_csc: float           # Mean CSC across this session
    window_count: int        # Number of qualifying windows
    start_hour: int          # Hour of day when session started

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": self.duration_minutes,
            "avg_fdi": self.avg_fdi,
            "avg_cls": self.avg_cls,
            "avg_csc": self.avg_csc,
            "window_count": self.window_count,
            "start_hour": self.start_hour,
        }


@dataclass
class FlowStateResult:
    """Daily flow state analysis result."""
    date_str: str

    # Detected sessions
    flow_sessions: list = field(default_factory=list)   # list[FlowSession]
    total_flow_minutes: int = 0
    peak_session: Optional[object] = None               # FlowSession | None

    # Aggregate score
    flow_score: float = 0.0            # 0.0 → 1.0
    flow_label: str = "none"           # deep_flow | in_zone | brief | none

    # Metadata
    windows_analysed: int = 0
    candidate_windows: int = 0
    is_meaningful: bool = False

    # Human output
    insight: str = ""
    peak_hour: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "date_str": self.date_str,
            "flow_sessions": [s.to_dict() for s in self.flow_sessions],
            "total_flow_minutes": self.total_flow_minutes,
            "peak_session": self.peak_session.to_dict() if self.peak_session else None,
            "flow_score": self.flow_score,
            "flow_label": self.flow_label,
            "windows_analysed": self.windows_analysed,
            "candidate_windows": self.candidate_windows,
            "is_meaningful": self.is_meaningful,
            "insight": self.insight,
            "peak_hour": self.peak_hour,
        }


# ─── Core detection ───────────────────────────────────────────────────────────

def _is_flow_window(window: dict) -> bool:
    """
    Determine if a single 15-minute window meets flow state criteria.

    Requires:
      - Working hours (is_working_hours = True)
      - FDI ≥ FLOW_FDI_MIN
      - FLOW_CLS_MIN ≤ CLS ≤ FLOW_CLS_MAX
      - CSC ≤ FLOW_CSC_MAX
      - Not a social meeting (attendees > 1 breaks flow)
    """
    meta = window.get("metadata", {}) or {}
    if not meta.get("is_working_hours", False):
        return False

    metrics = window.get("metrics", {}) or {}
    fdi = metrics.get("focus_depth_index")
    cls = metrics.get("cognitive_load_score")
    csc = metrics.get("context_switch_cost")

    # All three metrics must be present and valid
    if fdi is None or cls is None or csc is None:
        return False

    if fdi < FLOW_FDI_MIN:
        return False
    if cls < FLOW_CLS_MIN or cls > FLOW_CLS_MAX:
        return False
    if csc > FLOW_CSC_MAX:
        return False

    # Social meetings break flow
    cal = window.get("calendar", {}) or {}
    if cal.get("in_meeting", False) and cal.get("meeting_attendees", 0) > 1:
        return False

    return True


def _extract_time(window: dict, key: str) -> str:
    """Extract HH:MM from an ISO datetime string in the window."""
    ts = window.get(key, "")
    if not ts:
        return "??"
    try:
        # Handle both naive and tz-aware ISO strings
        dt_str = ts[:19]  # "YYYY-MM-DDTHH:MM:SS"
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%H:%M")
    except (ValueError, IndexError):
        return "??"


def _extract_hour(window: dict) -> int:
    """Extract the hour_of_day from a window's metadata."""
    meta = window.get("metadata", {}) or {}
    h = meta.get("hour_of_day")
    if h is not None:
        return int(h)
    # Fallback: parse from window_start
    ts = window.get("window_start", "")
    if ts:
        try:
            return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").hour
        except (ValueError, IndexError):
            pass
    return 0


def _classify_label(flow_score: float) -> str:
    """Map flow_score to a human label."""
    if flow_score >= DEEP_FLOW_THRESHOLD:
        return "deep_flow"
    if flow_score >= IN_ZONE_THRESHOLD:
        return "in_zone"
    if flow_score >= BRIEF_FLOW_THRESHOLD:
        return "brief"
    return "none"


def _build_insight(
    label: str,
    total_minutes: int,
    peak_session: Optional[FlowSession],
    flow_score: float,
) -> str:
    """Generate a one-line human-readable insight."""
    if label == "none":
        return (
            "No sustained flow sessions detected today — "
            "load, focus, or fragmentation conditions didn't align."
        )

    total_h = total_minutes // 60
    total_m = total_minutes % 60
    time_str = (
        f"{total_h}h {total_m}m" if total_h > 0 else f"{total_m}m"
    )

    if peak_session:
        peak_str = f"{peak_session.start_time}–{peak_session.end_time}"
        peak_dur = peak_session.duration_minutes
        peak_h   = peak_dur // 60
        peak_m_  = peak_dur % 60
        peak_dur_str = (
            f"{peak_h}h {peak_m_}m" if peak_h > 0 else f"{peak_m_}m"
        )
    else:
        peak_str = "unknown"
        peak_dur_str = "?"

    if label == "deep_flow":
        return (
            f"Exceptional flow day — {time_str} in sustained flow "
            f"(peak {peak_str}, {peak_dur_str}). "
            f"This is your cognitive peak performance zone."
        )
    elif label == "in_zone":
        return (
            f"Solid focus block — {time_str} of flow "
            f"(best run {peak_str}, {peak_dur_str}). "
            f"Quality focused work."
        )
    elif label == "brief":
        return (
            f"Brief flow detected — {time_str} in flow zone "
            f"({peak_str}). "
            f"Conditions were partially right but fragmented."
        )
    return f"{time_str} in flow state."


def detect_flow_states(windows: list[dict]) -> FlowStateResult:
    """
    Detect flow state sessions from a day's 15-minute windows.

    Scans working-hour windows for consecutive qualifying sequences
    (high FDI, moderate CLS, low CSC, no social meetings) that are
    at least MIN_SESSION_WINDOWS long (≥ 30 minutes).

    Args:
        windows: List of 15-min window dicts for a single day.

    Returns:
        FlowStateResult with all detected sessions and aggregate stats.
    """
    # Sort windows by window_index or window_start for correct ordering
    sorted_windows = sorted(
        windows,
        key=lambda w: w.get("window_index", w.get("window_start", ""))
    )

    # Filter to working-hours windows for analysis
    working_windows = [
        w for w in sorted_windows
        if w.get("metadata", {}).get("is_working_hours", False)
    ]

    date_str = ""
    if sorted_windows:
        date_str = sorted_windows[0].get("date", "")

    not_meaningful = FlowStateResult(
        date_str=date_str,
        windows_analysed=len(working_windows),
        is_meaningful=False,
        flow_label="none",
        insight="Insufficient working-hour windows for flow analysis.",
    )

    if len(working_windows) < MIN_WORKING_WINDOWS:
        return not_meaningful

    # Tag each working window as flow candidate
    is_candidate = [_is_flow_window(w) for w in working_windows]
    candidate_count = sum(is_candidate)

    # Find maximal runs of consecutive True values
    sessions: list[FlowSession] = []
    i = 0
    n = len(working_windows)

    while i < n:
        if not is_candidate[i]:
            i += 1
            continue

        # Start of a run — find its end
        j = i
        while j < n and is_candidate[j]:
            j += 1

        run_length = j - i
        if run_length >= MIN_SESSION_WINDOWS:
            # Build the FlowSession from this run
            run_windows = working_windows[i:j]

            # Aggregate metrics across the session
            fdi_vals = [
                w["metrics"]["focus_depth_index"]
                for w in run_windows
                if w.get("metrics", {}).get("focus_depth_index") is not None
            ]
            cls_vals = [
                w["metrics"]["cognitive_load_score"]
                for w in run_windows
                if w.get("metrics", {}).get("cognitive_load_score") is not None
            ]
            csc_vals = [
                w["metrics"]["context_switch_cost"]
                for w in run_windows
                if w.get("metrics", {}).get("context_switch_cost") is not None
            ]

            avg_fdi = sum(fdi_vals) / len(fdi_vals) if fdi_vals else 0.0
            avg_cls = sum(cls_vals) / len(cls_vals) if cls_vals else 0.0
            avg_csc = sum(csc_vals) / len(csc_vals) if csc_vals else 0.0

            start_time = _extract_time(run_windows[0], "window_start")
            end_time   = _extract_time(run_windows[-1], "window_end")
            start_hour = _extract_hour(run_windows[0])

            sessions.append(FlowSession(
                start_time=start_time,
                end_time=end_time,
                duration_minutes=run_length * 15,
                avg_fdi=round(avg_fdi, 4),
                avg_cls=round(avg_cls, 4),
                avg_csc=round(avg_csc, 4),
                window_count=run_length,
                start_hour=start_hour,
            ))

        i = j

    # Aggregate
    total_flow_minutes = sum(s.duration_minutes for s in sessions)
    peak_session = max(sessions, key=lambda s: s.duration_minutes) if sessions else None
    flow_score = round(min(total_flow_minutes / TARGET_FLOW_MINUTES, 1.0), 4)
    flow_label = _classify_label(flow_score)
    peak_hour = peak_session.start_hour if peak_session else None
    insight = _build_insight(flow_label, total_flow_minutes, peak_session, flow_score)

    return FlowStateResult(
        date_str=date_str,
        flow_sessions=sessions,
        total_flow_minutes=total_flow_minutes,
        peak_session=peak_session,
        flow_score=flow_score,
        flow_label=flow_label,
        windows_analysed=len(working_windows),
        candidate_windows=candidate_count,
        is_meaningful=True,
        insight=insight,
        peak_hour=peak_hour,
    )


# ─── Formatting ───────────────────────────────────────────────────────────────

# Flow label → emoji
_FLOW_EMOJI = {
    "deep_flow": "🌊",
    "in_zone":   "🎯",
    "brief":     "✨",
    "none":      "🌑",
}

# Flow label → display name
_FLOW_DISPLAY = {
    "deep_flow": "Deep Flow",
    "in_zone":   "In the Zone",
    "brief":     "Brief Flow",
    "none":      "No Flow",
}


def format_flow_line(result: FlowStateResult) -> str:
    """
    Format a compact Slack-ready flow line.

    Examples:
        🌊 Flow: Deep Flow — 2h 15m · peak 09:15–11:30 (2h 15m)
        🎯 Flow: In the Zone — 1h 15m · best block 10:00–11:15
        ✨ Flow: Brief — 30m · 09:45–10:15
        🌑 Flow: None — conditions didn't align
    """
    if not result.is_meaningful:
        return ""

    emoji   = _FLOW_EMOJI.get(result.flow_label, "🧠")
    display = _FLOW_DISPLAY.get(result.flow_label, result.flow_label)

    if result.flow_label == "none" or result.total_flow_minutes == 0:
        return f"{emoji} Flow: {display}"

    total_h  = result.total_flow_minutes // 60
    total_m  = result.total_flow_minutes % 60
    time_str = f"{total_h}h {total_m}m" if total_h > 0 else f"{total_m}m"

    line = f"{emoji} Flow: {display} — {time_str}"

    if result.peak_session:
        ps = result.peak_session
        line += f" · peak {ps.start_time}–{ps.end_time}"

    return line


def format_flow_section(result: FlowStateResult) -> str:
    """
    Format a multi-line Slack section with headline, sessions list, and insight.

    Example:
        🌊 *Flow State: Deep Flow* (score 0.94)
        09:15–11:30  2h 15m  · FDI 82% · CLS 38%
        14:00–14:45  45m     · FDI 79% · CLS 31%
        _Exceptional flow day — 3h of sustained flow ..._
    """
    if not result.is_meaningful:
        return ""

    emoji   = _FLOW_EMOJI.get(result.flow_label, "🧠")
    display = _FLOW_DISPLAY.get(result.flow_label, result.flow_label)

    header = f"{emoji} *Flow State: {display}* (score {result.flow_score:.2f})"
    lines = [header]

    if result.flow_sessions:
        for s in sorted(result.flow_sessions, key=lambda x: x.start_hour):
            dur_h = s.duration_minutes // 60
            dur_m = s.duration_minutes % 60
            dur_str = f"{dur_h}h {dur_m:02d}m" if dur_h > 0 else f"{dur_m}m"
            lines.append(
                f"  {s.start_time}–{s.end_time}  {dur_str:<8} "
                f"FDI {s.avg_fdi:.0%} · CLS {s.avg_cls:.0%}"
            )

    lines.append(f"_{result.insight}_")
    return "\n".join(lines)


# ─── Weekly summary helper ────────────────────────────────────────────────────

def compute_weekly_flow_summary(days: list[dict]) -> dict:
    """
    Aggregate flow stats across a week of days.

    Args:
        days: List of dicts, each with 'date' and 'windows' keys,
              OR list of FlowStateResult.to_dict() dicts.

    Returns:
        dict with:
          - total_flow_minutes: int
          - avg_flow_minutes_per_day: float
          - best_flow_day: str (YYYY-MM-DD) or None
          - best_flow_minutes: int
          - flow_days: int  (days with any flow)
          - avg_flow_score: float
    """
    from engine.store import read_day

    results = []
    for entry in days:
        if "flow_score" in entry:
            # Already a FlowStateResult dict
            results.append(entry)
        else:
            date_str = entry.get("date", "")
            windows = entry.get("windows") or (read_day(date_str) if date_str else [])
            if windows:
                r = detect_flow_states(windows)
                results.append(r.to_dict())

    valid = [r for r in results if r.get("is_meaningful", False)]
    if not valid:
        return {
            "total_flow_minutes": 0,
            "avg_flow_minutes_per_day": 0.0,
            "best_flow_day": None,
            "best_flow_minutes": 0,
            "flow_days": 0,
            "avg_flow_score": 0.0,
        }

    total_minutes = sum(r.get("total_flow_minutes", 0) for r in valid)
    flow_days = sum(1 for r in valid if r.get("total_flow_minutes", 0) > 0)
    avg_score = sum(r.get("flow_score", 0.0) for r in valid) / len(valid)

    best = max(valid, key=lambda r: r.get("total_flow_minutes", 0))
    best_day = best.get("date_str") if best.get("total_flow_minutes", 0) > 0 else None

    return {
        "total_flow_minutes": total_minutes,
        "avg_flow_minutes_per_day": round(total_minutes / len(valid), 1),
        "best_flow_day": best_day,
        "best_flow_minutes": best.get("total_flow_minutes", 0) if best_day else 0,
        "flow_days": flow_days,
        "avg_flow_score": round(avg_score, 4),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _run_cli() -> None:
    import argparse
    from engine.store import read_day, list_available_dates
    from datetime import datetime, timedelta

    parser = argparse.ArgumentParser(description="Detect flow state sessions for a date.")
    parser.add_argument("date", nargs="?", default=None, help="Date (YYYY-MM-DD), default: today")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--week", action="store_true", help="Show last 7 available days")
    args = parser.parse_args()

    if args.week:
        dates = sorted(list_available_dates())[-7:]
        for d in dates:
            windows = read_day(d)
            result = detect_flow_states(windows)
            if result.is_meaningful:
                line = format_flow_line(result)
                print(f"{d}: {line or 'No flow'}")
            else:
                print(f"{d}: insufficient data")
        return

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    windows = read_day(date_str)

    if not windows:
        print(f"No data found for {date_str}")
        sys.exit(1)

    result = detect_flow_states(windows)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"\nFlow State Analysis — {date_str}")
    print("─" * 48)

    if not result.is_meaningful:
        print("Not meaningful (insufficient working-hour windows)")
        return

    print(f"  Flow Score:   {result.flow_score:.4f}  ({result.flow_label})")
    print(f"  Total Flow:   {result.total_flow_minutes} min")
    print(f"  Sessions:     {len(result.flow_sessions)}")
    print(f"  Candidates:   {result.candidate_windows}/{result.windows_analysed} windows qualified")
    print()

    if result.flow_sessions:
        print("  Sessions:")
        for s in sorted(result.flow_sessions, key=lambda x: x.start_hour):
            dur_h = s.duration_minutes // 60
            dur_m = s.duration_minutes % 60
            dur_str = f"{dur_h}h {dur_m:02d}m" if dur_h > 0 else f"  {dur_m}m"
            print(f"    {s.start_time}–{s.end_time}  {dur_str}  FDI {s.avg_fdi:.0%}  CLS {s.avg_cls:.0%}  CSC {s.avg_csc:.0%}")

    print()
    print(f"  {result.insight}")
    print()
    print(f"  Slack:  {format_flow_line(result)}")


if __name__ == "__main__":
    _run_cli()

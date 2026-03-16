"""
Presence Tracker — Meeting Intelligence Module (v13)

Answers: *"How much did today's meetings actually cost David cognitively?"*

Existing modules track meeting *quantity* (total_meeting_minutes, event_count).
This module analyses meeting *quality* — the relationship between meeting patterns
and cognitive state:

  1. **Focus Fragmentation Score (FFS)** — 0.0 to 1.0
     How severely do meetings break the day into fragments too short for deep work?
     A day with 3× 45-min meetings scattered across the morning scores higher than
     one 3-hour afternoon block, even at the same total meeting time.

  2. **Cognitive Meeting Cost (CMC)** — 0.0 to 1.0
     Actual CLS elevation during meeting windows vs. the baseline CLS in non-meeting
     windows. High CMC = meetings drove up cognitive load significantly.
     Low CMC = meetings were cognitively light (passive attendance, no real demand).

  3. **Social Drain Rate (SDR)** — 0.0 to 1.0
     Mean SDI across meeting windows, normalised. Captures how much social energy
     was consumed — relevant for introverts where meeting drain outlasts the meeting.

  4. **Meeting Recovery Fit (MRF)** — 'aligned' | 'overloaded' | 'underutilised' | 'unknown'
     Were today's meetings appropriate given WHOOP recovery score?
     - aligned: meeting load proportional to recovery readiness
     - overloaded: heavy meeting load on a low-recovery day (risk signal)
     - underutilised: light meeting load on a peak-recovery day (opportunity signal)
     - unknown: insufficient data

  5. **Peak Focus Threat** — list[str]
     Which historically strong focus hours (from the FDI profile) were blocked by
     meetings today? e.g. ["9:00", "10:00"] if David's best hours were taken.

## Composite: Meeting Intelligence Score (MIS) — 0 to 100

A single number summarising how well today's meetings were structured relative
to cognitive capacity. Higher = better meeting management.

  MIS = 100 × (1 − FFS×0.40 − CMC×0.30 − SDR×0.30)

Clipped to [0, 100].

## API

    from analysis.meeting_intel import compute_meeting_intel, format_meeting_intel_section

    intel = compute_meeting_intel(windows, whoop_data, date_str)
    section = format_meeting_intel_section(intel)

## Usage (CLI)

    python3 analysis/meeting_intel.py 2026-03-13
    python3 analysis/meeting_intel.py 2026-03-13 --json
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import read_day

# ─── Constants ────────────────────────────────────────────────────────────────

# Minimum free gap (minutes) between meetings to count as a focus opportunity
MIN_FOCUS_GAP_MINUTES = 45

# Historical lookback for peak-hour FDI profile
FDI_PROFILE_LOOKBACK_DAYS = 30

# Recovery thresholds for MRF classification
RECOVERY_LOW = 50     # Below this = low recovery
RECOVERY_PEAK = 75    # Above this = peak recovery

# Meeting load thresholds (total meeting minutes in a day)
MEETING_LOAD_LIGHT = 60    # < 1 hour
MEETING_LOAD_HEAVY = 180   # ≥ 3 hours

# FDI threshold for a "focus hour" (historically productive)
PEAK_FDI_THRESHOLD = 0.70


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class MeetingIntel:
    """Meeting intelligence for a single day."""
    date_str: str

    # Core metrics (None when no meetings)
    focus_fragmentation_score: Optional[float] = None   # 0.0–1.0; higher = more fragmented
    cognitive_meeting_cost: Optional[float] = None       # 0.0–1.0; CLS delta in meetings
    social_drain_rate: Optional[float] = None            # 0.0–1.0; mean SDI during meetings
    meeting_recovery_fit: str = "unknown"                # aligned|overloaded|underutilised|unknown

    # Composite
    meeting_intelligence_score: Optional[int] = None    # 0–100; higher = better

    # Supporting data
    total_meeting_minutes: int = 0
    meeting_count: int = 0                              # distinct calendar events
    meeting_windows: int = 0                            # 15-min windows in meetings
    free_gap_minutes: int = 0                           # longest contiguous free gap
    peak_focus_threats: list = field(default_factory=list)   # hours blocked by meetings

    is_meaningful: bool = False     # False when no meetings or insufficient data

    # Narrative
    headline: str = ""
    advisory: str = ""

    def to_dict(self) -> dict:
        return {
            "date_str": self.date_str,
            "focus_fragmentation_score": self.focus_fragmentation_score,
            "cognitive_meeting_cost": self.cognitive_meeting_cost,
            "social_drain_rate": self.social_drain_rate,
            "meeting_recovery_fit": self.meeting_recovery_fit,
            "meeting_intelligence_score": self.meeting_intelligence_score,
            "total_meeting_minutes": self.total_meeting_minutes,
            "meeting_count": self.meeting_count,
            "meeting_windows": self.meeting_windows,
            "free_gap_minutes": self.free_gap_minutes,
            "peak_focus_threats": self.peak_focus_threats,
            "is_meaningful": self.is_meaningful,
            "headline": self.headline,
            "advisory": self.advisory,
        }


# ─── Helper: window classification ───────────────────────────────────────────

def _get_meeting_windows(windows: list[dict]) -> list[dict]:
    """Return windows where David was in a meeting."""
    return [w for w in windows if w.get("calendar", {}).get("in_meeting", False)]


def _get_active_non_meeting_windows(windows: list[dict]) -> list[dict]:
    """Return active (working hours) windows where David was NOT in a meeting."""
    return [
        w for w in windows
        if w.get("metadata", {}).get("is_active_window", False)
        and not w.get("calendar", {}).get("in_meeting", False)
    ]


# ─── Metric 1: Focus Fragmentation Score ─────────────────────────────────────

def compute_focus_fragmentation(windows: list[dict]) -> float:
    """
    Compute how badly meetings fragment the working day.

    Uses 15-minute window resolution. For each contiguous block of free
    (non-meeting) working-hour time, check if it's long enough for deep work
    (>= MIN_FOCUS_GAP_MINUTES). Count minutes trapped in gaps too short for
    focus as "fragmented" time.

    FFS = fragmented_minutes / total_working_minutes
    where fragmented = free working-hour slots shorter than MIN_FOCUS_GAP_MINUTES.

    Returns 0.0 when no meetings (no fragmentation), 1.0 when every free slot
    is too short to do deep work.
    """
    working_windows = [
        w for w in windows
        if w.get("metadata", {}).get("is_working_hours", False)
    ]

    if not working_windows:
        return 0.0

    # Classify each working window as meeting (M) or free (F)
    flags = ["M" if w.get("calendar", {}).get("in_meeting", False) else "F"
             for w in working_windows]

    total_working = len(working_windows)
    if total_working == 0:
        return 0.0

    # Count meeting windows to check if there are any
    meeting_count = flags.count("M")
    if meeting_count == 0:
        return 0.0

    # Find contiguous free runs and classify them
    fragmented_slots = 0
    i = 0
    while i < len(flags):
        if flags[i] == "F":
            run_start = i
            while i < len(flags) and flags[i] == "F":
                i += 1
            run_len = i - run_start
            run_minutes = run_len * 15
            if run_minutes < MIN_FOCUS_GAP_MINUTES:
                fragmented_slots += run_len
        else:
            i += 1

    free_slots = flags.count("F")
    if free_slots == 0:
        return 1.0  # All working time is in meetings

    return min(1.0, fragmented_slots / free_slots)


# ─── Metric 2: Cognitive Meeting Cost ────────────────────────────────────────

def compute_cognitive_meeting_cost(
    meeting_wins: list[dict],
    non_meeting_wins: list[dict],
) -> Optional[float]:
    """
    Compute the CLS elevation attributable to meetings.

    CMC = (mean CLS in meeting windows) − (mean CLS in non-meeting windows)
    normalised to [0, 1] (negative values clipped to 0).

    Returns None if either group is empty.
    """
    if not meeting_wins or not non_meeting_wins:
        return None

    def _mean_cls(wins: list[dict]) -> Optional[float]:
        cls_vals = [
            w.get("metrics", {}).get("cognitive_load_score")
            for w in wins
            if w.get("metrics", {}).get("cognitive_load_score") is not None
        ]
        return sum(cls_vals) / len(cls_vals) if cls_vals else None

    m_cls = _mean_cls(meeting_wins)
    nm_cls = _mean_cls(non_meeting_wins)

    if m_cls is None or nm_cls is None:
        return None

    # Delta, clipped to [0, 1]
    delta = m_cls - nm_cls
    return max(0.0, min(1.0, delta))


# ─── Metric 3: Social Drain Rate ─────────────────────────────────────────────

def compute_social_drain_rate(meeting_wins: list[dict]) -> Optional[float]:
    """
    Mean SDI across meeting windows, clipped to [0, 1].

    Returns None when no meeting windows.
    """
    if not meeting_wins:
        return None

    sdi_vals = [
        w.get("metrics", {}).get("social_drain_index", 0.0)
        for w in meeting_wins
    ]

    if not sdi_vals:
        return None

    return min(1.0, sum(sdi_vals) / len(sdi_vals))


# ─── Metric 4: Meeting Recovery Fit ──────────────────────────────────────────

def compute_meeting_recovery_fit(
    total_meeting_minutes: int,
    recovery_score: Optional[float],
) -> str:
    """
    Classify whether today's meeting load was appropriate for David's recovery.

    Overloaded: heavy meeting day (≥ MEETING_LOAD_HEAVY) + low recovery (< RECOVERY_LOW)
    Underutilised: light meeting day (< MEETING_LOAD_LIGHT) + peak recovery (≥ RECOVERY_PEAK)
    Aligned: everything else with valid data
    Unknown: missing recovery data
    """
    if recovery_score is None:
        return "unknown"

    is_heavy = total_meeting_minutes >= MEETING_LOAD_HEAVY
    is_light = total_meeting_minutes < MEETING_LOAD_LIGHT
    is_low_recovery = recovery_score < RECOVERY_LOW
    is_peak_recovery = recovery_score >= RECOVERY_PEAK

    if is_heavy and is_low_recovery:
        return "overloaded"
    if is_light and is_peak_recovery:
        return "underutilised"
    return "aligned"


# ─── Metric 5: Peak Focus Threats ────────────────────────────────────────────

def compute_peak_focus_threats(
    windows: list[dict],
    date_str: str,
    lookback_days: int = FDI_PROFILE_LOOKBACK_DAYS,
) -> list[str]:
    """
    Find historically high-FDI hours that were blocked by meetings today.

    Returns a list of hour strings like ["9:00", "10:00"] for each hour
    where David's historical FDI is above PEAK_FDI_THRESHOLD and he was
    in a meeting today during that hour.
    """
    # Build historical hourly FDI profile
    try:
        from engine.store import list_available_dates, read_day as _read_day

        available = sorted(
            [d for d in list_available_dates() if d < date_str],
            reverse=True,
        )[:lookback_days]

        hourly_fdi: dict[int, list[float]] = {}
        for d in available:
            try:
                day_windows = _read_day(d)
            except Exception:
                continue
            for w in day_windows:
                meta = w.get("metadata", {}) or {}
                if not meta.get("is_active_window"):
                    continue
                hour = meta.get("hour_of_day")
                fdi = (w.get("metrics") or {}).get("focus_depth_index")
                if hour is not None and fdi is not None:
                    hourly_fdi.setdefault(hour, []).append(fdi)

        peak_hours = set(
            h for h, vals in hourly_fdi.items()
            if len(vals) >= 2 and (sum(vals) / len(vals)) >= PEAK_FDI_THRESHOLD
        )
    except Exception:
        peak_hours = set()

    if not peak_hours:
        return []

    # Find hours blocked by meetings today
    meeting_hours = set(
        w.get("metadata", {}).get("hour_of_day")
        for w in windows
        if w.get("calendar", {}).get("in_meeting", False)
        and w.get("metadata", {}).get("hour_of_day") is not None
    )

    threatened = peak_hours & meeting_hours
    return sorted(f"{h}:00" for h in threatened)


# ─── Longest free gap ─────────────────────────────────────────────────────────

def compute_longest_free_gap(windows: list[dict]) -> int:
    """
    Return the longest contiguous free (non-meeting) working-hour gap in minutes.
    """
    working_windows = [
        w for w in windows
        if w.get("metadata", {}).get("is_working_hours", False)
    ]

    if not working_windows:
        return 0

    max_gap = 0
    current_gap = 0

    for w in working_windows:
        if w.get("calendar", {}).get("in_meeting", False):
            max_gap = max(max_gap, current_gap)
            current_gap = 0
        else:
            current_gap += 15

    max_gap = max(max_gap, current_gap)
    return max_gap


# ─── Composite score ──────────────────────────────────────────────────────────

def compute_mis(
    ffs: Optional[float],
    cmc: Optional[float],
    sdr: Optional[float],
) -> Optional[int]:
    """
    Meeting Intelligence Score — single composite 0–100.

    MIS = 100 × (1 − FFS×0.40 − CMC×0.30 − SDR×0.30)

    Higher = better meeting management. Returns None when all inputs are None.
    """
    if ffs is None and cmc is None and sdr is None:
        return None

    ffs_val = ffs if ffs is not None else 0.0
    cmc_val = cmc if cmc is not None else 0.0
    sdr_val = sdr if sdr is not None else 0.0

    raw = 1.0 - ffs_val * 0.40 - cmc_val * 0.30 - sdr_val * 0.30
    return max(0, min(100, int(round(raw * 100))))


# ─── Narrative ────────────────────────────────────────────────────────────────

def _build_narrative(
    intel: "MeetingIntel",
) -> tuple[str, str]:
    """Build headline and advisory strings."""

    total_mins = intel.total_meeting_minutes
    mis = intel.meeting_intelligence_score
    fit = intel.meeting_recovery_fit
    ffs = intel.focus_fragmentation_score or 0.0
    cmc = intel.cognitive_meeting_cost or 0.0
    threats = intel.peak_focus_threats
    longest_gap = intel.free_gap_minutes

    # Hours/minutes formatting
    def _fmt_mins(m: int) -> str:
        if m >= 60:
            h = m // 60
            r = m % 60
            return f"{h}h{r:02d}m" if r else f"{h}h"
        return f"{m}m"

    # Headline
    if mis is None:
        headline = "No meeting data available."
    elif mis >= 80:
        headline = f"Meeting-efficient day — {_fmt_mins(total_mins)} of meetings, well-structured."
    elif mis >= 60:
        headline = f"Acceptable meeting load — {_fmt_mins(total_mins)} with moderate fragmentation."
    elif mis >= 40:
        headline = f"Meeting pressure day — {_fmt_mins(total_mins)} fragmented focus windows."
    else:
        headline = f"High meeting cost — {_fmt_mins(total_mins)} disrupted cognitive flow significantly."

    # Advisory
    advisories = []

    if fit == "overloaded":
        advisories.append("Heavy meeting load on a low-recovery day — reschedule non-essential calls next time.")
    elif fit == "underutilised" and mis is not None and mis >= 70:
        advisories.append("Peak recovery day with light meeting load — ideal for deep work.")

    if threats:
        hour_list = ", ".join(threats[:3])
        advisories.append(f"Meetings blocked peak focus hour{'s' if len(threats) > 1 else ''}: {hour_list}.")

    if ffs > 0.6 and longest_gap < MIN_FOCUS_GAP_MINUTES:
        advisories.append("No focus gap long enough for deep work — cluster meetings in future.")
    elif longest_gap >= 120:
        advisories.append(f"Longest free block: {_fmt_mins(longest_gap)} — good for deep work.")

    if cmc > 0.3:
        advisories.append("Meetings elevated cognitive load significantly; schedule recovery buffer.")

    advisory = " ".join(advisories) if advisories else "Meeting structure was appropriate for the day."

    return headline, advisory


# ─── Main entry point ─────────────────────────────────────────────────────────

def compute_meeting_intel(
    windows: list[dict],
    whoop_data: Optional[dict] = None,
    date_str: Optional[str] = None,
) -> MeetingIntel:
    """
    Compute meeting intelligence for a given day's windows.

    Parameters
    ----------
    windows : list[dict]
        All 96 windows for the day (from read_day or build_windows).
    whoop_data : dict | None
        WHOOP data dict with at least 'recovery_score'. Used for MRF.
    date_str : str | None
        Date string (YYYY-MM-DD). Used for peak focus threat lookback.

    Returns
    -------
    MeetingIntel dataclass.  is_meaningful=False when no meetings exist.
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    recovery = (whoop_data or {}).get("recovery_score")

    meeting_wins = _get_meeting_windows(windows)
    non_meeting_wins = _get_active_non_meeting_windows(windows)

    # Count distinct meetings by (title, duration) pair — the duration field is
    # the total meeting duration stamped on every window of that meeting.
    # We sum distinct (title, duration) pairs rather than summing across windows.
    # Fallback: if no title information, estimate from window count × 15 min.
    seen_meetings: dict[str, int] = {}  # title → declared_duration
    for w in meeting_wins:
        cal = w.get("calendar", {})
        title = cal.get("meeting_title") or "unknown"
        dur = cal.get("meeting_duration_minutes", 0) or 0
        if title not in seen_meetings:
            seen_meetings[title] = dur

    meeting_count = len(seen_meetings)
    # Use declared durations where available; otherwise estimate from window count
    if all(d > 0 for d in seen_meetings.values()):
        total_meeting_minutes = sum(seen_meetings.values())
    else:
        # Fall back to window count × 15 min (more reliable than zero-duration records)
        total_meeting_minutes = len(meeting_wins) * 15

    if not meeting_wins:
        return MeetingIntel(
            date_str=date_str,
            is_meaningful=False,
            meeting_recovery_fit="aligned" if recovery is not None else "unknown",
            headline="No meetings today.",
            advisory="Full day available for focused work.",
        )

    # Compute metrics
    ffs = compute_focus_fragmentation(windows)
    cmc = compute_cognitive_meeting_cost(meeting_wins, non_meeting_wins)
    sdr = compute_social_drain_rate(meeting_wins)
    mrf = compute_meeting_recovery_fit(total_meeting_minutes, recovery)
    mis = compute_mis(ffs, cmc, sdr)
    threats = compute_peak_focus_threats(windows, date_str)
    longest_gap = compute_longest_free_gap(windows)

    intel = MeetingIntel(
        date_str=date_str,
        focus_fragmentation_score=round(ffs, 3),
        cognitive_meeting_cost=round(cmc, 3) if cmc is not None else None,
        social_drain_rate=round(sdr, 3) if sdr is not None else None,
        meeting_recovery_fit=mrf,
        meeting_intelligence_score=mis,
        total_meeting_minutes=total_meeting_minutes,
        meeting_count=meeting_count,
        meeting_windows=len(meeting_wins),
        free_gap_minutes=longest_gap,
        peak_focus_threats=threats,
        is_meaningful=True,
    )

    intel.headline, intel.advisory = _build_narrative(intel)
    return intel


# ─── Slack formatter ──────────────────────────────────────────────────────────

def format_meeting_intel_section(intel: MeetingIntel) -> str:
    """
    Format the meeting intelligence as a Slack DM section.

    Example output:
        *📅 Meeting Intelligence:*
        MIS 72/100 — Acceptable meeting load — 2h30m with moderate fragmentation.
        • FFS: 42%  (fragmentation)
        • CMC: 18%  (cognitive cost)
        • SDR: 24%  (social drain)
        • Recovery fit: aligned
        _Longest free gap: 90m. Meetings blocked peak hour: 9:00._

    Returns empty string when is_meaningful=False.
    """
    if not intel.is_meaningful:
        return ""

    lines = ["*📅 Meeting Intelligence:*"]

    mis = intel.meeting_intelligence_score
    mis_str = f"MIS {mis}/100 — " if mis is not None else ""
    lines.append(f"{mis_str}{intel.headline}")

    if intel.focus_fragmentation_score is not None:
        lines.append(f"• FFS: {intel.focus_fragmentation_score:.0%}  (fragmentation)")
    if intel.cognitive_meeting_cost is not None:
        lines.append(f"• CMC: {intel.cognitive_meeting_cost:.0%}  (cognitive cost)")
    if intel.social_drain_rate is not None:
        lines.append(f"• SDR: {intel.social_drain_rate:.0%}  (social drain)")

    fit_emoji = {
        "aligned": "✅",
        "overloaded": "⚠️",
        "underutilised": "💡",
        "unknown": "❓",
    }
    lines.append(f"• Recovery fit: {fit_emoji.get(intel.meeting_recovery_fit, '')} {intel.meeting_recovery_fit}")

    # Advisory
    if intel.advisory:
        detail_parts = []
        if intel.free_gap_minutes > 0:
            def _fmt(m: int) -> str:
                return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"
            detail_parts.append(f"Longest free gap: {_fmt(intel.free_gap_minutes)}.")
        if intel.peak_focus_threats:
            detail_parts.append(f"Peak hours blocked: {', '.join(intel.peak_focus_threats[:3])}.")
        if detail_parts:
            lines.append(f"_{' '.join(detail_parts)}_")
        lines.append(f"_{intel.advisory}_")

    return "\n".join(lines)


# ─── Terminal formatter ───────────────────────────────────────────────────────

def format_meeting_intel_terminal(intel: MeetingIntel) -> str:
    """
    Format meeting intelligence for terminal output (scripts/report.py integration).
    """
    if not intel.is_meaningful:
        return "  No meetings today."

    def _bar(val: Optional[float], width: int = 10) -> str:
        if val is None:
            return "—"
        filled = round(val * width)
        return "▓" * filled + "░" * (width - filled)

    lines = []
    mis = intel.meeting_intelligence_score
    lines.append(f"  MIS {mis:3d}/100  {_bar(mis / 100 if mis else 0)}" if mis is not None else "  MIS: —")
    lines.append(f"  FFS  {intel.focus_fragmentation_score:.0%}  {_bar(intel.focus_fragmentation_score)}  (fragmentation)" if intel.focus_fragmentation_score is not None else "  FFS: —")
    lines.append(f"  CMC  {intel.cognitive_meeting_cost:.0%}  {_bar(intel.cognitive_meeting_cost)}  (cognitive cost)" if intel.cognitive_meeting_cost is not None else "  CMC: —")
    lines.append(f"  SDR  {intel.social_drain_rate:.0%}  {_bar(intel.social_drain_rate)}  (social drain)" if intel.social_drain_rate is not None else "  SDR: —")
    lines.append(f"  Fit: {intel.meeting_recovery_fit}")

    if intel.peak_focus_threats:
        lines.append(f"  Peak hours blocked: {', '.join(intel.peak_focus_threats)}")

    if intel.advisory and intel.advisory != "Meeting structure was appropriate for the day.":
        lines.append(f"  → {intel.advisory}")

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.

    Usage:
        python3 analysis/meeting_intel.py                  # Today
        python3 analysis/meeting_intel.py 2026-03-13       # Specific date
        python3 analysis/meeting_intel.py --json           # JSON output
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Meeting Intelligence — cognitive cost of today's meetings"
    )
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD), default = today")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    try:
        windows = read_day(date_str)
    except FileNotFoundError:
        print(f"No data for {date_str}. Run scripts/run_daily.py {date_str} first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading data: {e}")
        sys.exit(1)

    # Try to get WHOOP data from windows
    whoop_data = {}
    for w in windows:
        wd = w.get("whoop") or {}
        if wd.get("recovery_score") is not None:
            whoop_data = wd
            break

    intel = compute_meeting_intel(windows, whoop_data, date_str)

    if args.json:
        print(json.dumps(intel.to_dict(), indent=2))
        return

    print()
    print(f"Meeting Intelligence — {date_str}")
    print("=" * 50)

    if not intel.is_meaningful:
        print("  No meetings today.")
    else:
        print(format_meeting_intel_terminal(intel))

    print()


if __name__ == "__main__":
    main()

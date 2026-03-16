"""
Presence Tracker — Cognitive Load Decomposer (v24)

Answers: *"What actually drove my cognitive load today — and how much?"*

The CLS metric is a composite of multiple signal components:
  - Meeting load     (social meetings × attendees)
  - Slack volume     (incoming message pressure)
  - Recovery deficit (low HRV / WHOOP → higher baseline load)
  - Distraction      (RescueTime low-productivity windows)
  - Conversation     (Omi spoken conversations)

The existing digest surfaces the *result* (CLS = 0.72) but not the *cause*.
The Decomposer breaks CLS into its constituent source attributions so David
can answer: "Was today hard because of meetings or because I was distracted?"

## How attribution works

For each 15-min window, the metrics engine computes CLS from components:

  base_cls = 0.35 × meeting_component
           + 0.20 × calendar_pressure
           + 0.25 × slack_component
           + 0.20 × recovery_inverse

  if RT active:
      base_cls = 0.75 × base_cls + 0.25 × rt_demand

  if Omi active:
      cls = 0.90 × base_cls + 0.10 × omi_component

The Decomposer reconstructs each component per window and aggregates
across the day. It then expresses each source as a % of total load,
surfacing which factor was the biggest driver.

## Attribution categories

  meetings      — meeting_component + calendar_pressure (social meeting overhead)
  slack         — slack_component (incoming message pressure)
  physiology    — recovery_inverse (HRV/sleep/recovery deficit as load baseline)
  rescuetime    — rt_demand (distraction/low-productivity screen time)
  omi           — omi_component (spoken conversation cognitive cost)

## Output

  LoadDecomposition dataclass:
    - date_str: str
    - total_cls_mean: float           — avg CLS across active windows
    - total_cls_sum: float            — sum of CLS across all windows
    - source_shares: dict[str, float] — fraction of load from each source (sums to 1.0)
    - source_cls: dict[str, float]    — absolute avg CLS attributed to each source
    - dominant_source: str            — source with highest share
    - insight_lines: list[str]        — 1-3 actionable sentences
    - windows_analysed: int
    - active_windows: int
    - is_meaningful: bool             — False when < 2 active windows

## API

    from analysis.load_decomposer import compute_load_decomposition, format_decomposition_section

    decomp = compute_load_decomposition(date_str)
    section = format_decomposition_section(decomp)   # Slack markdown section
    line = format_decomposition_line(decomp)          # compact one-liner

## Integration

  In nightly digest — after daily ingestion:
    decomp = compute_load_decomposition(date_str)
    if decomp.is_meaningful and decomp.total_cls_mean > 0.15:
        lines.append(format_decomposition_section(decomp))

  In weekly summary — for each day of the past week:
    decomp = compute_load_decomposition(date_str)
    # Roll up source shares across the week

## CLI

    python3 analysis/load_decomposer.py             # Today
    python3 analysis/load_decomposer.py 2026-03-14  # Specific date
    python3 analysis/load_decomposer.py --json      # JSON output
    python3 analysis/load_decomposer.py --week      # 7-day decomposition
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import list_available_dates, read_day


# ─── Constants ────────────────────────────────────────────────────────────────

# Minimum active windows before decomposition is considered meaningful
MIN_ACTIVE_WINDOWS = 2

# Minimum CLS for a window to be included in attribution
MIN_WINDOW_CLS = 0.01

# RescueTime min active seconds to count as behaviorally active
RT_MIN_ACTIVE_SECONDS = 60

# Normalisation caps (matching metrics.py constants)
_MAX_SLACK_MSGS = 30
_MAX_MEETING_ATTENDEES = 10
_MAX_RT_APP_SWITCHES = 8
_HRV_REFERENCE_MS = 65.0
_HRV_SATURATION_MS = 130.0
_OMI_WORD_SATURATION = 500


# ─── Helpers (mirroring metrics.py internals) ────────────────────────────────

def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _norm(val, max_val: float, min_val: float = 0.0) -> float:
    if val is None:
        return 0.0
    if max_val == min_val:
        return 0.0
    return _clamp((val - min_val) / (max_val - min_val))


def _physiological_readiness(
    recovery_score: Optional[float],
    hrv_rmssd_milli: Optional[float] = None,
    sleep_performance: Optional[float] = None,
) -> float:
    """Mirror of metrics.physiological_readiness for component attribution."""
    w_r, w_h, w_s = 0.50, 0.30, 0.20

    r = (recovery_score / 100.0) if recovery_score is not None else 0.5
    r = _clamp(r)

    if hrv_rmssd_milli is not None:
        hrv_ratio = _clamp(
            (hrv_rmssd_milli - _HRV_REFERENCE_MS) / (_HRV_SATURATION_MS - _HRV_REFERENCE_MS),
            lo=-1.0, hi=1.0,
        )
        h = 0.5 + 0.5 * hrv_ratio
    else:
        h = r  # fallback: same as recovery

    s = (sleep_performance / 100.0) if sleep_performance is not None else 0.5
    s = _clamp(s)

    return _clamp(w_r * r + w_h * h + w_s * s)


def _decompose_window(window: dict) -> dict:
    """
    Decompose a single window into CLS source components.

    Returns a dict with keys:
      meetings, slack, physiology, rescuetime, omi
    where each value is the raw (unnormalised) contribution before
    the final CLS formula is applied.  Components sum approximately
    to the window's CLS (within blending rounding).

    Note: the CLS formula applies weights and blending, so component
    magnitudes are pre-blend; the final `cls_computed` is used to
    normalise shares so they sum to 1.0.
    """
    cal = window.get("calendar", {})
    whoop = window.get("whoop", {})
    slack = window.get("slack", {})
    rt = window.get("rescuetime", {})
    omi = window.get("omi", {})

    in_meeting = cal.get("in_meeting", False)
    meeting_attendees = cal.get("meeting_attendees", 0)
    is_social = in_meeting and meeting_attendees > 1

    # ── Meeting component ──────────────────────────────────────────────────
    meeting_component = 1.0 if is_social else 0.0
    calendar_pressure = _norm(meeting_attendees if is_social else 0, max_val=_MAX_MEETING_ATTENDEES)
    meetings_raw = 0.35 * meeting_component + 0.20 * calendar_pressure

    # ── Slack component ───────────────────────────────────────────────────
    slack_msgs = slack.get("messages_received", 0) or 0
    slack_raw = 0.25 * _norm(slack_msgs, max_val=_MAX_SLACK_MSGS)

    # ── Physiology component ──────────────────────────────────────────────
    readiness = _physiological_readiness(
        whoop.get("recovery_score"),
        whoop.get("hrv_rmssd_milli"),
        whoop.get("sleep_performance"),
    )
    recovery_inverse = 1.0 - readiness
    physiology_raw = 0.20 * recovery_inverse

    # ── Base CLS (before RT and Omi blending) ────────────────────────────
    base_cls = meetings_raw + slack_raw + physiology_raw

    # ── RescueTime distraction component ──────────────────────────────────
    rt_active = rt.get("active_seconds", 0) or 0
    rt_prod = rt.get("productivity_score")
    rt_raw = 0.0
    rt_demand_component = 0.0
    if rt_prod is not None and rt_active >= RT_MIN_ACTIVE_SECONDS:
        rt_demand = 1.0 - rt_prod
        rt_demand_component = rt_demand
        # RT blends at 25%; the other sources take 75%
        rt_raw_contribution = 0.25 * rt_demand
        # The remaining 75% of RT-blended base_cls comes from the 4 base components
        # We express RT's share in the blended space
        rt_raw = rt_raw_contribution  # RT absolute contribution in blended space
        # Rescale the base components to 75% of their original values
        scale = 0.75
        meetings_raw *= scale
        slack_raw *= scale
        physiology_raw *= scale
        base_cls = meetings_raw + slack_raw + physiology_raw + rt_raw

    # ── Omi component ─────────────────────────────────────────────────────
    omi_active = omi.get("conversation_active", False)
    omi_raw = 0.0
    if omi_active:
        word_count = omi.get("word_count", 0) or 0
        word_density = _norm(word_count, max_val=_OMI_WORD_SATURATION)
        cognitive_density = omi.get("cognitive_density", 0.0) or 0.0
        cls_weight = omi.get("cls_weight", 1.0) or 1.0

        if cognitive_density > 0:
            omi_component = cognitive_density * 0.7 + word_density * 0.3
            omi_component_weighted = _clamp(omi_component * cls_weight)
        else:
            omi_component_weighted = 0.5 + 0.5 * word_density

        # Omi blends at 10%; base_cls takes 90%
        omi_raw = 0.10 * omi_component_weighted
        # Rescale base to 90%
        scale_omi = 0.90
        meetings_raw *= scale_omi
        slack_raw *= scale_omi
        physiology_raw *= scale_omi
        rt_raw *= scale_omi
        # omi_raw is 10% of the final blend — already absolute

    cls_computed = meetings_raw + slack_raw + physiology_raw + rt_raw + omi_raw
    cls_computed = _clamp(cls_computed)

    return {
        "meetings": meetings_raw,
        "slack": slack_raw,
        "physiology": physiology_raw,
        "rescuetime": rt_raw,
        "omi": omi_raw,
        "cls_computed": cls_computed,
        # Store the stored CLS for comparison
        "cls_stored": window.get("metrics", {}).get("cognitive_load_score", None),
    }


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class LoadDecomposition:
    """
    Attribution of cognitive load to its source components for one day.
    """
    date_str: str

    # Overall load summary
    total_cls_mean: float = 0.0            # avg CLS across active windows
    active_cls_mean: float = 0.0           # avg CLS only for windows with CLS > MIN_WINDOW_CLS
    windows_analysed: int = 0
    active_windows: int = 0

    # Attribution: fraction of load (0.0–1.0) summing to 1.0
    source_shares: dict = field(default_factory=dict)

    # Attribution: absolute avg CLS attributed to each source
    source_cls: dict = field(default_factory=dict)

    # Dominant source name
    dominant_source: str = "unknown"

    # Human-readable insight lines
    insight_lines: list = field(default_factory=list)

    # Quality flag
    is_meaningful: bool = False

    def to_dict(self) -> dict:
        return {
            "date_str": self.date_str,
            "total_cls_mean": round(self.total_cls_mean, 4),
            "active_cls_mean": round(self.active_cls_mean, 4),
            "windows_analysed": self.windows_analysed,
            "active_windows": self.active_windows,
            "source_shares": {k: round(v, 4) for k, v in self.source_shares.items()},
            "source_cls": {k: round(v, 4) for k, v in self.source_cls.items()},
            "dominant_source": self.dominant_source,
            "insight_lines": self.insight_lines,
            "is_meaningful": self.is_meaningful,
        }


# ─── Source labels and emojis ─────────────────────────────────────────────────

_SOURCE_LABELS = {
    "meetings":    "Meetings",
    "slack":       "Slack",
    "physiology":  "Recovery deficit",
    "rescuetime":  "Screen distraction",
    "omi":         "Conversations",
}

_SOURCE_EMOJI = {
    "meetings":    "📅",
    "slack":       "💬",
    "physiology":  "💤",
    "rescuetime":  "💻",
    "omi":         "🎙",
}


# ─── Core computation ─────────────────────────────────────────────────────────

def compute_load_decomposition(
    date_str: Optional[str] = None,
) -> LoadDecomposition:
    """
    Decompose a single day's cognitive load into source attributions.

    Parameters
    ----------
    date_str : str, optional
        Date string 'YYYY-MM-DD'. Defaults to today if not provided.

    Returns
    -------
    LoadDecomposition
        If no data exists for the date or fewer than MIN_ACTIVE_WINDOWS
        active windows, returns a LoadDecomposition with is_meaningful=False.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    windows = read_day(date_str)
    if not windows:
        return LoadDecomposition(date_str=date_str, is_meaningful=False)

    # Accumulate component totals across windows
    sources = ["meetings", "slack", "physiology", "rescuetime", "omi"]
    totals = {s: 0.0 for s in sources}
    cls_total = 0.0
    active_cls_total = 0.0
    active_count = 0
    n = len(windows)

    for w in windows:
        comp = _decompose_window(w)
        cls_val = comp["cls_computed"]
        cls_total += cls_val
        for s in sources:
            totals[s] += comp[s]
        if cls_val > MIN_WINDOW_CLS:
            active_cls_total += cls_val
            active_count += 1

    if active_count < MIN_ACTIVE_WINDOWS:
        return LoadDecomposition(
            date_str=date_str,
            windows_analysed=n,
            active_windows=active_count,
            total_cls_mean=cls_total / n if n else 0.0,
            is_meaningful=False,
        )

    # Mean components
    mean_components = {s: totals[s] / n for s in sources}
    total_attributed = sum(mean_components.values())

    # Compute shares
    if total_attributed > 0:
        shares = {s: mean_components[s] / total_attributed for s in sources}
    else:
        shares = {s: 0.0 for s in sources}

    # Dominant source
    dominant = max(shares, key=lambda k: shares[k])

    # Build insight lines
    insight_lines = _build_insights(
        date_str=date_str,
        shares=shares,
        source_cls=mean_components,
        total_cls_mean=cls_total / n,
        active_cls_mean=active_cls_total / active_count if active_count else 0.0,
        dominant=dominant,
    )

    return LoadDecomposition(
        date_str=date_str,
        total_cls_mean=round(cls_total / n, 4),
        active_cls_mean=round(active_cls_total / active_count, 4) if active_count else 0.0,
        windows_analysed=n,
        active_windows=active_count,
        source_shares=shares,
        source_cls=mean_components,
        dominant_source=dominant,
        insight_lines=insight_lines,
        is_meaningful=True,
    )


def compute_week_decomposition(
    end_date_str: Optional[str] = None,
    days: int = 7,
) -> dict:
    """
    Aggregate load decomposition across multiple days.

    Returns a dict with:
      - daily: list[LoadDecomposition]
      - weekly_shares: dict[str, float]   — avg share per source across days
      - weekly_cls: float                 — mean daily avg CLS
      - dominant_source: str
      - days_meaningful: int
    """
    if end_date_str is None:
        end_date_str = datetime.now().strftime("%Y-%m-%d")

    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    daily = []
    for i in range(days - 1, -1, -1):
        d = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        daily.append(compute_load_decomposition(d))

    meaningful = [d for d in daily if d.is_meaningful]
    if not meaningful:
        return {
            "daily": daily,
            "weekly_shares": {},
            "weekly_cls": 0.0,
            "dominant_source": "unknown",
            "days_meaningful": 0,
        }

    sources = ["meetings", "slack", "physiology", "rescuetime", "omi"]
    agg_shares = {s: sum(d.source_shares.get(s, 0.0) for d in meaningful) / len(meaningful)
                  for s in sources}
    weekly_cls = sum(d.total_cls_mean for d in meaningful) / len(meaningful)
    dominant = max(agg_shares, key=lambda k: agg_shares[k])

    return {
        "daily": daily,
        "weekly_shares": agg_shares,
        "weekly_cls": weekly_cls,
        "dominant_source": dominant,
        "days_meaningful": len(meaningful),
    }


# ─── Insight generation ───────────────────────────────────────────────────────

def _build_insights(
    date_str: str,
    shares: dict,
    source_cls: dict,
    total_cls_mean: float,
    active_cls_mean: float,
    dominant: str,
) -> list:
    """Generate 1–3 actionable insight sentences from decomposition data."""
    lines = []

    # Low total load — nothing meaningful to decompose
    if total_cls_mean < 0.10:
        lines.append("Load was minimal today — most windows were cognitively quiet.")
        return lines

    # Dominant source insight
    dom_pct = round(shares.get(dominant, 0) * 100)
    dom_label = _SOURCE_LABELS.get(dominant, dominant)

    if dominant == "meetings" and dom_pct >= 35:
        lines.append(
            f"Meetings drove {dom_pct}% of today's cognitive load — "
            "consider batching them to protect focus blocks."
        )
    elif dominant == "slack" and dom_pct >= 30:
        lines.append(
            f"Slack interruptions were the biggest cognitive drain ({dom_pct}%) — "
            "try batched notification windows tomorrow."
        )
    elif dominant == "physiology" and dom_pct >= 35:
        lines.append(
            f"Recovery deficit contributed {dom_pct}% of load — today's effort came "
            "at higher physiological cost. Prioritise sleep tonight."
        )
    elif dominant == "rescuetime" and dom_pct >= 30:
        lines.append(
            f"Screen distraction drove {dom_pct}% of load — low-productivity app "
            "usage was the biggest cost. Try a distraction blocker tomorrow."
        )
    elif dominant == "omi" and dom_pct >= 25:
        lines.append(
            f"Spoken conversations drove {dom_pct}% of load — high talk-time is "
            "draining even when enjoyable. Schedule quiet recovery time."
        )
    else:
        lines.append(
            f"{dom_label} was the largest load contributor today ({dom_pct}%)."
        )

    # Secondary insight: second-largest source if meaningful
    sorted_sources = sorted(shares.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_sources) >= 2:
        second, second_share = sorted_sources[1]
        second_pct = round(second_share * 100)
        if second_pct >= 20 and second != dominant:
            second_label = _SOURCE_LABELS.get(second, second)
            lines.append(
                f"{second_label} added another {second_pct}% — "
                + _source_action_hint(second)
            )

    # Physiology caveat if recovery deficit is high but not dominant
    physio_share = shares.get("physiology", 0)
    if physio_share >= 0.25 and dominant != "physiology":
        lines.append(
            f"Recovery deficit amplified all load today ({round(physio_share * 100)}% of CLS) — "
            "better recovery tomorrow will lower the baseline."
        )

    return lines[:3]  # cap at 3 lines


def _source_action_hint(source: str) -> str:
    hints = {
        "meetings":   "blocking a longer uninterrupted block could reduce this.",
        "slack":      "a focused Slack-off window after lunch could help.",
        "physiology": "prioritising sleep and recovery will lower this over time.",
        "rescuetime": "reducing tab-switching or social media would help.",
        "omi":        "lighter conversation load tomorrow may ease this.",
    }
    return hints.get(source, "monitor this trend.")


# ─── Formatting ──────────────────────────────────────────────────────────────

def format_decomposition_line(decomp: LoadDecomposition) -> str:
    """
    Compact one-liner for Slack embedding.

    Example:
        "🔍 Load breakdown: Meetings 42% · Slack 28% · Recovery deficit 18% · other 12%"
    """
    if not decomp.is_meaningful:
        return ""

    sources = ["meetings", "slack", "physiology", "rescuetime", "omi"]
    parts = []
    other_pct = 0
    for s in sources:
        pct = round(decomp.source_shares.get(s, 0) * 100)
        if pct >= 10:
            emoji = _SOURCE_EMOJI.get(s, "")
            label = _SOURCE_LABELS.get(s, s)
            parts.append(f"{emoji} {label} {pct}%")
        else:
            other_pct += pct

    line = " · ".join(parts)
    if other_pct > 0:
        line += f" · other {other_pct}%"

    return f"🔍 *Load breakdown:* {line}"


def format_decomposition_section(decomp: LoadDecomposition) -> str:
    """
    Full Slack markdown section with source breakdown bar chart and insights.

    Example:
        🔍 *What drove today's load*

        📅 Meetings          ████████░░  42%
        💬 Slack             ██████░░░░  28%
        💤 Recovery deficit  ████░░░░░░  18%
        💻 Screen distract   ██░░░░░░░░  10%
        🎙 Conversations     ░░░░░░░░░░   2%

        _Meetings were the main driver — consider batching them tomorrow._
    """
    if not decomp.is_meaningful:
        return ""

    sources = ["meetings", "slack", "physiology", "rescuetime", "omi"]
    lines = ["🔍 *What drove today's load*", ""]

    # Bar chart
    bar_total = 10
    for s in sources:
        pct = round(decomp.source_shares.get(s, 0) * 100)
        bars = round(pct / 10)
        bar_str = "█" * bars + "░" * (bar_total - bars)
        label = _SOURCE_LABELS.get(s, s).ljust(18)
        emoji = _SOURCE_EMOJI.get(s, "")
        lines.append(f"{emoji} `{label}` {bar_str}  {pct:3d}%")

    # Insights
    if decomp.insight_lines:
        lines.append("")
        for insight in decomp.insight_lines:
            lines.append(f"_{insight}_")

    return "\n".join(lines)


def format_decomposition_terminal(decomp: LoadDecomposition) -> str:
    """Terminal ANSI-formatted decomposition report."""
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    if not decomp.is_meaningful:
        return f"\n  {BOLD}Load Decomposition{RESET}\n  {DIM}Not enough data.{RESET}\n"

    sources = ["meetings", "slack", "physiology", "rescuetime", "omi"]

    lines = [
        "",
        f"{BOLD}Cognitive Load Decomposition — {decomp.date_str}{RESET}",
        "=" * 60,
        "",
        f"  Avg CLS: {BOLD}{decomp.total_cls_mean:.3f}{RESET}   "
        f"Active windows: {decomp.active_windows}/{decomp.windows_analysed}",
        "",
    ]

    bar_total = 20
    for s in sources:
        pct = round(decomp.source_shares.get(s, 0) * 100)
        bars = round(pct * bar_total / 100)
        bar_filled = "█" * bars
        bar_empty = "░" * (bar_total - bars)

        # Colour by dominance
        if s == decomp.dominant_source:
            colour = YELLOW
        elif pct < 5:
            colour = DIM
        else:
            colour = RESET

        label = _SOURCE_LABELS.get(s, s).ljust(20)
        emoji = _SOURCE_EMOJI.get(s, " ")
        lines.append(
            f"  {emoji} {colour}{label}{RESET}  {colour}{bar_filled}{bar_empty}{RESET}  {pct:3d}%"
        )

    # Insights
    if decomp.insight_lines:
        lines.append("")
        for insight in decomp.insight_lines:
            lines.append(f"  {DIM}→ {insight}{RESET}")

    lines.append("")
    return "\n".join(lines)


def format_week_decomposition_section(week: dict) -> str:
    """
    Slack markdown section for a weekly decomposition summary.

    Example:
        🔍 *Weekly Load Breakdown (7-day avg)*

        📅 Meetings          ████████░░  40%
        💬 Slack             █████░░░░░  25%
        ...

        _Meetings were your #1 cognitive cost this week (avg 40% of CLS)._
    """
    if not week.get("days_meaningful", 0):
        return ""

    sources = ["meetings", "slack", "physiology", "rescuetime", "omi"]
    weekly_shares = week.get("weekly_shares", {})
    dom = week.get("dominant_source", "unknown")
    dom_pct = round(weekly_shares.get(dom, 0) * 100)
    dom_label = _SOURCE_LABELS.get(dom, dom)

    lines = [
        f"🔍 *Weekly Load Breakdown ({week['days_meaningful']}-day avg)*",
        "",
    ]

    bar_total = 10
    for s in sources:
        pct = round(weekly_shares.get(s, 0) * 100)
        bars = round(pct / 10)
        bar_str = "█" * bars + "░" * (bar_total - bars)
        label = _SOURCE_LABELS.get(s, s).ljust(18)
        emoji = _SOURCE_EMOJI.get(s, "")
        lines.append(f"{emoji} `{label}` {bar_str}  {pct:3d}%")

    lines.append("")
    lines.append(
        f"_{dom_label} was your #1 cognitive cost this week ({dom_pct}% of avg CLS)._"
    )

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cognitive Load Decomposer — what drove your CLS today?"
    )
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--week", action="store_true", help="7-day aggregation")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    if args.week:
        print(f"[load_decomposer] Computing 7-day decomposition ending {date_str}...", flush=True)
        week = compute_week_decomposition(date_str, days=7)
        if args.json:
            out = {
                "daily": [d.to_dict() for d in week["daily"]],
                "weekly_shares": {k: round(v, 4) for k, v in week["weekly_shares"].items()},
                "weekly_cls": round(week["weekly_cls"], 4),
                "dominant_source": week["dominant_source"],
                "days_meaningful": week["days_meaningful"],
            }
            print(json.dumps(out, indent=2))
        else:
            meaningful = [d for d in week["daily"] if d.is_meaningful]
            for d in meaningful:
                print(format_decomposition_terminal(d))
            print(format_week_decomposition_section(week))
        return

    print(f"[load_decomposer] Decomposing CLS for {date_str}...", flush=True)
    decomp = compute_load_decomposition(date_str)

    if args.json:
        print(json.dumps(decomp.to_dict(), indent=2))
        return

    print(format_decomposition_terminal(decomp))


if __name__ == "__main__":
    main()

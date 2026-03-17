"""
Presence Tracker — Load Volatility Index (LVI)

Answers: *"Was today's cognitive load chaotic and spiky, or smooth and predictable?"*

The Cognitive Load Score (CLS) tells David *how much* mental effort he expended.
But two days can have identical average CLS (say, 0.40) yet feel completely
different:

  Day A: steady 0.40 all day — predictable, manageable, sustainable
  Day B: 0.05 → 0.80 → 0.05 → 0.75 → 0.05 — whiplash, draining, mentally costly

The Load Volatility Index (LVI) captures this distinction.  It measures the
*consistency* of cognitive load throughout the day by computing the standard
deviation of CLS across active working-hour windows.

## Why volatility matters

High CLS variance — rapid switches between intense cognitive demand and idle
periods — is physiologically costly for reasons independent of the average load:

  1. Task-switching overhead: each transition from deep focus to high-pressure
     activity (meeting, urgent Slack) incurs a cognitive reset cost.
  2. Anticipatory stress: unpredictable load spikes keep the nervous system
     on alert even during apparently quiet windows.
  3. Incomplete recovery: brief low-CLS gaps between spikes don't allow the
     parasympathetic nervous system to fully disengage.

The WHOOP research equivalent is heart rate variability *within* a workout —
erratic heart rate during effort is harder on the body than steady-state effort
at the same average.

## Formula

Computed from active working-hour windows (meeting OR Slack activity present):

    cls_values = [w["metrics"]["cognitive_load_score"] for w in active_windows]
    cls_std    = statistics.stdev(cls_values)   # population std when ≥ 2 values

    # Normalise: 0.35 std is the "volatile" reference point
    # (0.35 means typical swing between near-zero and 0.70, i.e. light↔heavy)
    lvi = round(1.0 - min(cls_std / LVI_STD_SCALE, 1.0), 4)
    lvi = max(0.0, min(1.0, lvi))

    LVI_STD_SCALE = 0.35   (configurable constant)

## Interpretation

    LVI ≥ 0.80  → Smooth      load was consistent all day
    0.60–0.80   → Steady      mild variation, broadly predictable
    0.40–0.60   → Variable    noticeable swings between low and high load
    < 0.40      → Volatile    high-frequency spikes, cognitive whiplash risk

## Output

    LoadVolatility dataclass:
      - lvi: float                  — 0.0 (volatile) to 1.0 (perfectly smooth)
      - cls_std: float              — raw standard deviation of CLS values
      - cls_mean: float             — mean CLS (for context)
      - cls_min: float              — minimum CLS in active windows
      - cls_max: float              — maximum CLS in active windows
      - label: str                  — 'smooth' | 'steady' | 'variable' | 'volatile'
      - windows_used: int           — number of active windows analysed
      - is_meaningful: bool         — False when < MIN_WINDOWS active windows
      - insight: str                — one-line human explanation

## API

    from analysis.load_volatility import compute_load_volatility, format_lvi_line

    lvi = compute_load_volatility(windows)
    if lvi.is_meaningful:
        line = format_lvi_line(lvi)      # Slack-ready one-liner

## Integration

    In nightly digest (after Cognitive Load section):
        lvi = compute_load_volatility(windows)
        if lvi.is_meaningful and lvi.label in ("variable", "volatile"):
            # Only surface when there's something noteworthy to flag
            lines.append(format_lvi_line(lvi))

## Design principles

  - Pure computation — fully testable with mock windows
  - Graceful degradation: < 3 active windows → is_meaningful = False
  - No external dependencies — uses only Python stdlib (statistics module)
  - Low threshold for surfacing: only shown when volatility is notable (< 'steady')
  - Complement, not replacement, to avg CLS — the two signals together tell
    the full story of a day's cognitive load pattern

## CLI

    python3 analysis/load_volatility.py             # Today
    python3 analysis/load_volatility.py 2026-03-14  # Specific date
    python3 analysis/load_volatility.py --json      # Machine-readable JSON

"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Constants ────────────────────────────────────────────────────────────────

# Minimum active windows required for a meaningful LVI calculation
MIN_WINDOWS: int = 3

# std = 0.35 maps to LVI = 0.0 (maximally volatile reference point)
# A std of 0.35 represents swings between near-zero and ~0.70 CLS (light↔heavy)
LVI_STD_SCALE: float = 0.35

# LVI tier boundaries
SMOOTH_THRESHOLD: float   = 0.80
STEADY_THRESHOLD: float   = 0.60
VARIABLE_THRESHOLD: float = 0.40


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class LoadVolatility:
    """Per-day cognitive load volatility metrics."""

    # Core LVI signal
    lvi: float                  # 0.0 (volatile) → 1.0 (smooth)
    cls_std: float              # Raw standard deviation of CLS values
    cls_mean: float             # Mean CLS (for context)
    cls_min: float              # Min CLS across active windows
    cls_max: float              # Max CLS across active windows
    cls_range: float            # cls_max - cls_min (peak swing)

    # Classification
    label: str                  # 'smooth' | 'steady' | 'variable' | 'volatile'

    # Metadata
    windows_used: int           # Number of active windows analysed
    is_meaningful: bool         # False when < MIN_WINDOWS active windows

    # Human-readable explanation
    insight: str                # One-line explanation

    def to_dict(self) -> dict:
        return {
            "lvi": self.lvi,
            "cls_std": self.cls_std,
            "cls_mean": self.cls_mean,
            "cls_min": self.cls_min,
            "cls_max": self.cls_max,
            "cls_range": self.cls_range,
            "label": self.label,
            "windows_used": self.windows_used,
            "is_meaningful": self.is_meaningful,
            "insight": self.insight,
        }


# ─── Computation ──────────────────────────────────────────────────────────────

def _classify_label(lvi: float) -> str:
    """Map LVI score to a human label."""
    if lvi >= SMOOTH_THRESHOLD:
        return "smooth"
    if lvi >= STEADY_THRESHOLD:
        return "steady"
    if lvi >= VARIABLE_THRESHOLD:
        return "variable"
    return "volatile"


def _build_insight(label: str, cls_std: float, cls_mean: float, cls_range: float, windows: int) -> str:
    """
    Generate a one-line human-readable insight about load volatility.

    Explains *what* the volatility means for the day, not just the raw number.
    """
    swing = f"{cls_range:.0%} swing"

    if label == "smooth":
        if cls_mean < 0.20:
            return f"Load was smooth and light all day — an easy, consistent {windows}-window day."
        elif cls_mean < 0.50:
            return (
                f"Load was consistent throughout the day ({swing}) — steady, predictable effort."
            )
        else:
            return (
                f"Load was sustained but consistent ({swing}) — high demand held steady, not erratic."
            )

    elif label == "steady":
        return (
            f"Moderate load variation today ({swing}) — some peaks and troughs "
            f"but broadly predictable across {windows} active windows."
        )

    elif label == "variable":
        return (
            f"Noticeable load swings today ({swing}, std {cls_std:.2f}) — "
            f"the rhythm was uneven; high-demand bursts interrupted quieter periods."
        )

    else:  # volatile
        return (
            f"High load volatility today ({swing}, std {cls_std:.2f}) — "
            f"frequent spikes between light and heavy demand. "
            f"Cognitive whiplash risk even if the average looks moderate."
        )


def compute_load_volatility(windows: list[dict]) -> LoadVolatility:
    """
    Compute the Load Volatility Index from a day's windows.

    Uses active working-hour windows (meeting OR Slack activity present)
    to measure how consistently cognitive load was maintained vs how
    erratically it spiked and dropped throughout the day.

    Args:
        windows: List of 15-min window dicts (full day or working-hour subset)

    Returns:
        LoadVolatility dataclass.  is_meaningful=False when insufficient data.
    """
    # Filter: working hours (7am–10pm)
    working = [w for w in windows if w.get("metadata", {}).get("is_working_hours", False)]

    # Active windows: in meeting OR slack activity
    active = [
        w for w in working
        if w.get("calendar", {}).get("in_meeting", False)
        or w.get("slack", {}).get("total_messages", 0) > 0
    ]

    not_meaningful = LoadVolatility(
        lvi=0.5,
        cls_std=0.0,
        cls_mean=0.0,
        cls_min=0.0,
        cls_max=0.0,
        cls_range=0.0,
        label="steady",
        windows_used=len(active),
        is_meaningful=False,
        insight="Insufficient active windows for volatility analysis.",
    )

    if len(active) < MIN_WINDOWS:
        return not_meaningful

    cls_vals = [w["metrics"]["cognitive_load_score"] for w in active]

    cls_mean = sum(cls_vals) / len(cls_vals)
    cls_min  = min(cls_vals)
    cls_max  = max(cls_vals)
    cls_range = cls_max - cls_min

    # Population standard deviation (we're describing the whole day, not sampling)
    n = len(cls_vals)
    variance = sum((v - cls_mean) ** 2 for v in cls_vals) / n
    cls_std  = math.sqrt(variance)

    # Normalise to LVI: 0 = maximally volatile, 1 = perfectly smooth
    lvi = round(1.0 - min(cls_std / LVI_STD_SCALE, 1.0), 4)
    lvi = max(0.0, min(1.0, lvi))

    label   = _classify_label(lvi)
    insight = _build_insight(label, cls_std, cls_mean, cls_range, n)

    return LoadVolatility(
        lvi=lvi,
        cls_std=round(cls_std, 4),
        cls_mean=round(cls_mean, 4),
        cls_min=round(cls_min, 4),
        cls_max=round(cls_max, 4),
        cls_range=round(cls_range, 4),
        label=label,
        windows_used=n,
        is_meaningful=True,
        insight=insight,
    )


# ─── Formatting ───────────────────────────────────────────────────────────────

# Label to emoji mapping for Slack display
_LABEL_EMOJI = {
    "smooth":   "〰️",
    "steady":   "📊",
    "variable": "〜",
    "volatile": "⚡",
}

_LABEL_DISPLAY = {
    "smooth":   "Smooth",
    "steady":   "Steady",
    "variable": "Variable",
    "volatile": "Volatile",
}


def format_lvi_line(lvi: "LoadVolatility") -> str:
    """
    Format a compact Slack-ready LVI one-liner.

    Example outputs:
        〰️ Load pattern: Smooth (LVI 0.91, std 0.03)
        📊 Load pattern: Steady (LVI 0.72, std 0.08)
        〜 Load pattern: Variable (LVI 0.54, std 0.16)  — peaks at 09:00, 15:00
        ⚡ Load pattern: Volatile (LVI 0.28, std 0.26)  — cognitive whiplash risk
    """
    if not lvi.is_meaningful:
        return ""

    emoji   = _LABEL_EMOJI.get(lvi.label, "📊")
    display = _LABEL_DISPLAY.get(lvi.label, lvi.label.capitalize())

    line = f"{emoji} Load pattern: {display} (LVI {lvi.lvi:.2f}, std {lvi.cls_std:.2f})"

    # Add a warning note for volatile days
    if lvi.label == "volatile":
        line += " — high cognitive switching cost"
    elif lvi.label == "variable":
        line += " — uneven demand pattern"

    return line


def format_lvi_section(lvi: "LoadVolatility") -> str:
    """
    Format a two-line Slack section with headline and insight.

    Example:
        ⚡ *Load Pattern: Volatile* (LVI 0.28)
        _High load volatility today (62% swing, std 0.26) — ..._
    """
    if not lvi.is_meaningful:
        return ""

    emoji   = _LABEL_EMOJI.get(lvi.label, "📊")
    display = _LABEL_DISPLAY.get(lvi.label, lvi.label.capitalize())

    header = f"{emoji} *Load Pattern: {display}* (LVI {lvi.lvi:.2f})"
    detail = f"_{lvi.insight}_"

    return "\n".join([header, detail])


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _run_cli() -> None:
    import argparse
    from engine.store import read_range
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Compute Load Volatility Index for a date.")
    parser.add_argument("date", nargs="?", default=None, help="Date (YYYY-MM-DD), default: today")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    windows = read_range(date_str, date_str)

    if not windows:
        print(f"No data found for {date_str}")
        sys.exit(1)

    lvi = compute_load_volatility(windows)

    if args.json:
        print(json.dumps(lvi.to_dict(), indent=2))
        return

    print(f"\nLoad Volatility Index — {date_str}")
    print("─" * 40)
    if not lvi.is_meaningful:
        print("Not meaningful (insufficient active windows)")
        return

    print(f"  LVI:         {lvi.lvi:.4f}  ({lvi.label})")
    print(f"  CLS mean:    {lvi.cls_mean:.4f}")
    print(f"  CLS std:     {lvi.cls_std:.4f}")
    print(f"  CLS range:   {lvi.cls_min:.4f} → {lvi.cls_max:.4f}  (swing: {lvi.cls_range:.0%})")
    print(f"  Windows:     {lvi.windows_used} active")
    print(f"\n  {lvi.insight}")
    print(f"\n  Slack line:  {format_lvi_line(lvi)}")


if __name__ == "__main__":
    _run_cli()

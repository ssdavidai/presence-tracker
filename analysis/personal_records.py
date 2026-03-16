"""
Presence Tracker — Personal Records (v15)

Answers: *"Is today special? Did David just set a personal best — or worst?"*

Tracks all-time records and positive streaks across the full JSONL history.
Designed to surface achievements and milestones in the nightly digest and
morning brief — the cognitive equivalent of Strava's "new record" and
WHOOP's monthly stats.

## What it tracks

### All-time records (per metric)

| Metric | Best means | Worst means |
|--------|-----------|-------------|
| CLS  | Lowest avg CLS day (lightest load) | Highest avg CLS day (most intense) |
| FDI  | Highest active FDI day (deepest focus) | Lowest active FDI day (most fragmented) |
| SDI  | Lowest SDI day (most restorative) | Highest SDI day (most draining) |
| RAS  | Highest avg RAS day (best alignment) | Lowest avg RAS day (most misaligned) |
| DPS  | Highest DPS day (best overall day) | Lowest DPS day (hardest overall day) |
| Recovery | Highest WHOOP recovery day | Lowest WHOOP recovery day |
| HRV | Highest HRV day | Lowest HRV day |

### Positive streaks (consecutive days meeting threshold)

| Streak | Threshold | What it means |
|--------|-----------|--------------|
| Low load | avg CLS < 0.25 | Consecutive light/sustainable days |
| Deep focus | active FDI ≥ 0.70 | Consecutive high-focus days |
| Recovery aligned | avg RAS ≥ 0.70 | Consecutive well-aligned days |
| Green recovery | WHOOP recovery ≥ 67 | Consecutive high-recovery days |

### Lifetime stats

- Total days tracked
- Total meeting minutes (all time)
- Total Slack messages sent (all time)
- Best week (highest avg DPS over any 7-day window)
- Days with all 5 sources present

## API

    compute_personal_records(as_of_date_str=None) → PersonalRecords
    check_today_records(date_str, records) → TodayRecordSummary
    format_records_line(today_summary) → str   (Slack-ready one-liner or "")
    format_records_section(today_summary) → str (longer optional section)

## Integration

    # In nightly digest — after daily data is ingested
    records = compute_personal_records(date_str)
    today = check_today_records(date_str, records)
    if today.has_records:
        # Append to digest message
        lines.append(format_records_line(today))

    # In morning brief — after reading yesterday's data
    # (shows "Yesterday you set a new FDI record")
    records = compute_personal_records(yesterday_str)
    today = check_today_records(yesterday_str, records)
    if today.has_records:
        lines.append(format_records_line(today))

## CLI

    python3 analysis/personal_records.py            # Full records report
    python3 analysis/personal_records.py 2026-03-14 # As of a specific date
    python3 analysis/personal_records.py --json     # Machine-readable JSON

"""

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.store import list_available_dates, read_day, read_summary


# ─── Constants ────────────────────────────────────────────────────────────────

# Streak thresholds
STREAK_LOW_LOAD_CLS = 0.25          # CLS below this = "light load day"
STREAK_DEEP_FOCUS_FDI = 0.70        # active FDI above this = "deep focus day"
STREAK_RECOVERY_ALIGNED_RAS = 0.70  # avg RAS above this = "well-aligned day"
STREAK_GREEN_RECOVERY = 67.0        # WHOOP recovery above this = "green zone"

# Minimum active windows in a day to be considered a "real" tracked day
# (avoids counting 2am insomnia data as a complete day)
MIN_WORKING_WINDOWS = 4

# Minimum data days before records are considered meaningful
MIN_DAYS_FOR_RECORDS = 2


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DayRecord:
    """A single record: the best or worst value for a metric."""
    date_str: str
    value: float          # The metric value on that day
    metric: str           # e.g. "avg_cls", "active_fdi", "dps", "recovery_score"
    label: str            # Human-readable label, e.g. "lowest CLS", "highest FDI"

    def to_dict(self) -> dict:
        return {
            "date": self.date_str,
            "value": round(self.value, 4),
            "metric": self.metric,
            "label": self.label,
        }


@dataclass
class StreakRecord:
    """Current streak and all-time longest streak for a threshold."""
    streak_name: str          # e.g. "low_load", "deep_focus"
    threshold_description: str # e.g. "CLS < 0.25"
    current_streak: int       # Days running (ending today / as-of date)
    current_streak_start: Optional[str]  # Date streak started
    longest_streak: int       # All-time longest streak
    longest_streak_start: Optional[str]  # All-time best streak start date
    longest_streak_end: Optional[str]    # All-time best streak end date

    def to_dict(self) -> dict:
        return {
            "streak_name": self.streak_name,
            "threshold_description": self.threshold_description,
            "current_streak": self.current_streak,
            "current_streak_start": self.current_streak_start,
            "longest_streak": self.longest_streak,
            "longest_streak_start": self.longest_streak_start,
            "longest_streak_end": self.longest_streak_end,
        }


@dataclass
class LifetimeStats:
    """Aggregate stats across all tracked days."""
    total_days: int = 0
    total_meeting_minutes: int = 0
    total_slack_sent: int = 0
    total_active_windows: int = 0
    days_all_sources: int = 0          # Days with all 5 sources present
    best_week_avg_dps: Optional[float] = None
    best_week_end_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "total_days": self.total_days,
            "total_meeting_minutes": self.total_meeting_minutes,
            "total_slack_sent": self.total_slack_sent,
            "total_active_windows": self.total_active_windows,
            "days_all_sources": self.days_all_sources,
            "best_week_avg_dps": round(self.best_week_avg_dps, 1) if self.best_week_avg_dps is not None else None,
            "best_week_end_date": self.best_week_end_date,
        }


@dataclass
class PersonalRecords:
    """Complete personal records as of a given date."""
    as_of_date: str
    total_days_analyzed: int

    # All-time bests
    best_fdi_day: Optional[DayRecord] = None       # Highest active FDI
    worst_fdi_day: Optional[DayRecord] = None      # Lowest active FDI
    best_cls_day: Optional[DayRecord] = None       # Lowest CLS (lightest)
    worst_cls_day: Optional[DayRecord] = None      # Highest CLS (most intense)
    best_ras_day: Optional[DayRecord] = None       # Highest RAS
    worst_ras_day: Optional[DayRecord] = None      # Lowest RAS
    best_dps_day: Optional[DayRecord] = None       # Highest DPS
    worst_dps_day: Optional[DayRecord] = None      # Lowest DPS
    best_recovery_day: Optional[DayRecord] = None  # Highest WHOOP recovery
    worst_recovery_day: Optional[DayRecord] = None # Lowest WHOOP recovery
    best_hrv_day: Optional[DayRecord] = None       # Highest HRV
    worst_hrv_day: Optional[DayRecord] = None      # Lowest HRV

    # Streaks
    low_load_streak: Optional[StreakRecord] = None
    deep_focus_streak: Optional[StreakRecord] = None
    recovery_aligned_streak: Optional[StreakRecord] = None
    green_recovery_streak: Optional[StreakRecord] = None

    # Lifetime stats
    lifetime: Optional[LifetimeStats] = None

    def is_meaningful(self) -> bool:
        return self.total_days_analyzed >= MIN_DAYS_FOR_RECORDS

    def to_dict(self) -> dict:
        def _dr(r: Optional[DayRecord]) -> Optional[dict]:
            return r.to_dict() if r else None

        def _sr(r: Optional[StreakRecord]) -> Optional[dict]:
            return r.to_dict() if r else None

        return {
            "as_of_date": self.as_of_date,
            "total_days_analyzed": self.total_days_analyzed,
            "is_meaningful": self.is_meaningful(),
            "bests": {
                "fdi": _dr(self.best_fdi_day),
                "cls": _dr(self.best_cls_day),
                "ras": _dr(self.best_ras_day),
                "dps": _dr(self.best_dps_day),
                "recovery": _dr(self.best_recovery_day),
                "hrv": _dr(self.best_hrv_day),
            },
            "worsts": {
                "fdi": _dr(self.worst_fdi_day),
                "cls": _dr(self.worst_cls_day),
                "ras": _dr(self.worst_ras_day),
                "dps": _dr(self.worst_dps_day),
                "recovery": _dr(self.worst_recovery_day),
                "hrv": _dr(self.worst_hrv_day),
            },
            "streaks": {
                "low_load": _sr(self.low_load_streak),
                "deep_focus": _sr(self.deep_focus_streak),
                "recovery_aligned": _sr(self.recovery_aligned_streak),
                "green_recovery": _sr(self.green_recovery_streak),
            },
            "lifetime": self.lifetime.to_dict() if self.lifetime else None,
        }


@dataclass
class TodayRecordSummary:
    """What records (if any) today's data set or extended."""
    date_str: str
    has_records: bool = False

    # New all-time bests set today
    new_best_fdi: bool = False       # FDI personal best
    new_best_cls: bool = False       # CLS personal best (lowest)
    new_best_ras: bool = False       # RAS personal best
    new_best_dps: bool = False       # DPS personal best
    new_best_recovery: bool = False  # Recovery personal best
    new_best_hrv: bool = False       # HRV personal best

    # Streak milestones hit today
    low_load_streak_days: int = 0      # Current low-load streak (if ≥ 2)
    deep_focus_streak_days: int = 0    # Current deep-focus streak (if ≥ 2)
    recovery_aligned_streak_days: int = 0  # Current RAS streak (if ≥ 2)
    green_recovery_streak_days: int = 0    # Current green-recovery streak (if ≥ 2)

    # Is a streak an all-time record today?
    new_streak_records: list[str] = field(default_factory=list)  # e.g. ["deep_focus"]

    def all_new_bests(self) -> list[str]:
        """Return list of metric names where a new best was set today."""
        result = []
        if self.new_best_fdi:
            result.append("FDI")
        if self.new_best_cls:
            result.append("CLS")
        if self.new_best_ras:
            result.append("RAS")
        if self.new_best_dps:
            result.append("DPS")
        if self.new_best_recovery:
            result.append("Recovery")
        if self.new_best_hrv:
            result.append("HRV")
        return result

    def active_streaks(self) -> list[tuple[str, int]]:
        """Return (name, days) for all active streaks ≥ 2."""
        result = []
        if self.low_load_streak_days >= 2:
            result.append(("Low load", self.low_load_streak_days))
        if self.deep_focus_streak_days >= 2:
            result.append(("Deep focus", self.deep_focus_streak_days))
        if self.recovery_aligned_streak_days >= 2:
            result.append(("Recovery aligned", self.recovery_aligned_streak_days))
        if self.green_recovery_streak_days >= 2:
            result.append(("Green recovery", self.green_recovery_streak_days))
        return result

    def to_dict(self) -> dict:
        return {
            "date": self.date_str,
            "has_records": self.has_records,
            "new_bests": self.all_new_bests(),
            "active_streaks": [
                {"name": n, "days": d} for n, d in self.active_streaks()
            ],
            "new_streak_records": self.new_streak_records,
        }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _mean(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _extract_day_metrics(date_str: str, summary: dict) -> Optional[dict]:
    """
    Extract per-day metric summary from the rolling.json summary dict.
    Returns None if the day has insufficient working windows.
    """
    day_data = summary.get("days", {}).get(date_str)
    if not day_data:
        return None

    working_count = day_data.get("working_hours_analyzed", 0)
    if working_count < MIN_WORKING_WINDOWS:
        return None

    metrics = day_data.get("metrics_avg", {})
    focus_quality = day_data.get("focus_quality", {})
    whoop = day_data.get("whoop", {})
    presence = day_data.get("presence_score", {})

    avg_cls = metrics.get("cognitive_load_score")
    active_fdi = focus_quality.get("active_fdi") or metrics.get("focus_depth_index")
    avg_ras = metrics.get("recovery_alignment_score")
    recovery = whoop.get("recovery_score")
    hrv = whoop.get("hrv_rmssd_milli")
    dps = presence.get("dps")

    return {
        "date": date_str,
        "avg_cls": avg_cls,
        "active_fdi": active_fdi,
        "avg_ras": avg_ras,
        "recovery_score": recovery,
        "hrv": hrv,
        "dps": dps,
        "working_count": working_count,
    }


def _compute_streaks(
    dates: list[str],
    day_metrics: dict,   # date_str → metric dict
    metric_fn,           # callable: day_metrics[d] → bool (True = streak day)
    streak_name: str,
    threshold_description: str,
    as_of_date: str,
) -> StreakRecord:
    """
    Compute current and all-time longest streak for a given boolean condition.

    dates: sorted list of all dates (oldest → newest)
    as_of_date: the reference date (compute current streak ending here)

    A calendar gap (missing date in the sequence) breaks the streak just like
    a non-matching day would, because we cannot assume missing data was a
    qualifying day.
    """
    # Build boolean series (oldest → newest), inserting None for calendar gaps
    # between consecutive tracked dates
    bools: list[tuple[str, object]] = []
    cutoff_dates = [d for d in dates if d <= as_of_date]

    for idx, d in enumerate(cutoff_dates):
        # Check for a gap between this date and the previous one
        if idx > 0:
            prev_d = cutoff_dates[idx - 1]
            prev_dt = datetime.strptime(prev_d, "%Y-%m-%d")
            curr_dt = datetime.strptime(d, "%Y-%m-%d")
            if (curr_dt - prev_dt).days > 1:
                # Calendar gap — insert a sentinel None to break the streak
                bools.append(("__gap__", None))

        m = day_metrics.get(d)
        if m is None:
            bools.append((d, None))  # date exists but insufficient data
        else:
            bools.append((d, metric_fn(m)))

    if not bools:
        return StreakRecord(
            streak_name=streak_name,
            threshold_description=threshold_description,
            current_streak=0,
            current_streak_start=None,
            longest_streak=0,
            longest_streak_start=None,
            longest_streak_end=None,
        )

    # Find current streak (ending with as_of_date, backwards)
    current_streak = 0
    current_start = None
    for d, val in reversed(bools):
        if val is None:
            # Data gap breaks the streak
            break
        if val:
            current_streak += 1
            current_start = d
        else:
            break

    # Find all-time longest streak
    best_len = 0
    best_start = None
    best_end = None
    run = 0
    run_start = None
    for d, val in bools:
        if val is True:
            run += 1
            if run == 1:
                run_start = d
            if run > best_len:
                best_len = run
                best_start = run_start
                best_end = d
        else:
            run = 0
            run_start = None

    return StreakRecord(
        streak_name=streak_name,
        threshold_description=threshold_description,
        current_streak=current_streak,
        current_streak_start=current_start,
        longest_streak=best_len,
        longest_streak_start=best_start,
        longest_streak_end=best_end,
    )


def _compute_lifetime_stats(dates: list[str], summary: dict) -> LifetimeStats:
    """Compute lifetime aggregate stats from the rolling summary."""
    all_days = summary.get("days", {})
    total_days = 0
    total_meeting_min = 0
    total_slack_sent = 0
    total_active_windows = 0
    days_all_sources = 0

    for d in dates:
        day = all_days.get(d)
        if not day:
            continue
        total_days += 1
        total_meeting_min += day.get("calendar", {}).get("total_meeting_minutes", 0)
        total_slack_sent += day.get("slack", {}).get("total_sent", 0) or day.get("slack", {}).get("messages_sent", 0)
        total_active_windows += day.get("focus_quality", {}).get("active_windows", 0)
        sources = day.get("sources_available", [])
        if len(sources) >= 5 or set(sources) >= {"whoop", "calendar", "slack", "rescuetime", "omi"}:
            days_all_sources += 1

    # Best 7-day rolling DPS window
    best_week_dps = None
    best_week_end = None
    valid_dates = [d for d in dates if all_days.get(d)]
    for i in range(len(valid_dates)):
        if i + 7 > len(valid_dates):
            break
        window = valid_dates[i:i + 7]
        dps_vals = [
            all_days[d].get("presence_score", {}).get("dps")
            for d in window
        ]
        avg = _mean(dps_vals)
        if avg is not None and (best_week_dps is None or avg > best_week_dps):
            best_week_dps = avg
            best_week_end = window[-1]

    return LifetimeStats(
        total_days=total_days,
        total_meeting_minutes=total_meeting_min,
        total_slack_sent=total_slack_sent,
        total_active_windows=total_active_windows,
        days_all_sources=days_all_sources,
        best_week_avg_dps=best_week_dps,
        best_week_end_date=best_week_end,
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_personal_records(as_of_date_str: Optional[str] = None) -> PersonalRecords:
    """
    Compute all-time personal records up to and including as_of_date_str.

    If as_of_date_str is None, uses the most recent available date.
    """
    all_dates = list_available_dates()  # sorted oldest → newest
    if not all_dates:
        today = as_of_date_str or datetime.now().strftime("%Y-%m-%d")
        return PersonalRecords(as_of_date=today, total_days_analyzed=0)

    if as_of_date_str is None:
        as_of_date_str = all_dates[-1]

    # Only consider dates up to as_of_date_str
    dates = [d for d in all_dates if d <= as_of_date_str]

    summary = read_summary()

    # Extract per-day metrics
    day_metrics = {}
    for d in dates:
        m = _extract_day_metrics(d, summary)
        if m:
            day_metrics[d] = m

    # If no summary data, try reading from raw JSONL
    if not day_metrics:
        for d in dates:
            windows = read_day(d)
            if not windows:
                continue
            working = [w for w in windows if w.get("metadata", {}).get("is_working_hours")]
            if len(working) < MIN_WORKING_WINDOWS:
                continue
            active = [w for w in working if w.get("metadata", {}).get("is_active_window")]
            metrics = [w.get("metrics", {}) for w in working]
            active_metrics = [w.get("metrics", {}) for w in active]
            whoop = windows[0].get("whoop", {}) if windows else {}
            presence = next((w.get("presence_score", {}) for w in windows if w.get("presence_score")), {})

            day_metrics[d] = {
                "date": d,
                "avg_cls": _mean([m.get("cognitive_load_score") for m in metrics]),
                "active_fdi": _mean([m.get("focus_depth_index") for m in active_metrics]),
                "avg_ras": _mean([m.get("recovery_alignment_score") for m in metrics]),
                "recovery_score": whoop.get("recovery_score"),
                "hrv": whoop.get("hrv_rmssd_milli"),
                "dps": presence.get("dps"),
                "working_count": len(working),
            }

    records = PersonalRecords(
        as_of_date=as_of_date_str,
        total_days_analyzed=len(day_metrics),
    )

    if not day_metrics:
        return records

    # ── All-time bests / worsts ────────────────────────────────────────────

    # FDI: best = highest active FDI; worst = lowest
    fdi_days = [(d, m["active_fdi"]) for d, m in day_metrics.items() if m["active_fdi"] is not None]
    if fdi_days:
        best_d, best_v = max(fdi_days, key=lambda x: x[1])
        worst_d, worst_v = min(fdi_days, key=lambda x: x[1])
        records.best_fdi_day = DayRecord(best_d, best_v, "active_fdi", "highest focus depth (FDI)")
        records.worst_fdi_day = DayRecord(worst_d, worst_v, "active_fdi", "lowest focus depth (FDI)")

    # CLS: best = lowest (lightest); worst = highest (most intense)
    cls_days = [(d, m["avg_cls"]) for d, m in day_metrics.items() if m["avg_cls"] is not None]
    if cls_days:
        best_d, best_v = min(cls_days, key=lambda x: x[1])
        worst_d, worst_v = max(cls_days, key=lambda x: x[1])
        records.best_cls_day = DayRecord(best_d, best_v, "avg_cls", "lowest cognitive load (CLS)")
        records.worst_cls_day = DayRecord(worst_d, worst_v, "avg_cls", "highest cognitive load (CLS)")

    # RAS: best = highest; worst = lowest
    ras_days = [(d, m["avg_ras"]) for d, m in day_metrics.items() if m["avg_ras"] is not None]
    if ras_days:
        best_d, best_v = max(ras_days, key=lambda x: x[1])
        worst_d, worst_v = min(ras_days, key=lambda x: x[1])
        records.best_ras_day = DayRecord(best_d, best_v, "avg_ras", "best recovery alignment (RAS)")
        records.worst_ras_day = DayRecord(worst_d, worst_v, "avg_ras", "worst recovery alignment (RAS)")

    # DPS: best = highest; worst = lowest
    dps_days = [(d, m["dps"]) for d, m in day_metrics.items() if m["dps"] is not None]
    if dps_days:
        best_d, best_v = max(dps_days, key=lambda x: x[1])
        worst_d, worst_v = min(dps_days, key=lambda x: x[1])
        records.best_dps_day = DayRecord(best_d, best_v, "dps", "highest Daily Presence Score (DPS)")
        records.worst_dps_day = DayRecord(worst_d, worst_v, "dps", "lowest Daily Presence Score (DPS)")

    # Recovery: best = highest; worst = lowest
    rec_days = [(d, m["recovery_score"]) for d, m in day_metrics.items() if m["recovery_score"] is not None]
    if rec_days:
        best_d, best_v = max(rec_days, key=lambda x: x[1])
        worst_d, worst_v = min(rec_days, key=lambda x: x[1])
        records.best_recovery_day = DayRecord(best_d, best_v, "recovery_score", "highest WHOOP recovery")
        records.worst_recovery_day = DayRecord(worst_d, worst_v, "recovery_score", "lowest WHOOP recovery")

    # HRV: best = highest; worst = lowest
    hrv_days = [(d, m["hrv"]) for d, m in day_metrics.items() if m["hrv"] is not None]
    if hrv_days:
        best_d, best_v = max(hrv_days, key=lambda x: x[1])
        worst_d, worst_v = min(hrv_days, key=lambda x: x[1])
        records.best_hrv_day = DayRecord(best_d, best_v, "hrv", "highest HRV")
        records.worst_hrv_day = DayRecord(worst_d, worst_v, "hrv", "lowest HRV")

    # ── Streaks ────────────────────────────────────────────────────────────

    records.low_load_streak = _compute_streaks(
        dates, day_metrics,
        lambda m: m["avg_cls"] is not None and m["avg_cls"] < STREAK_LOW_LOAD_CLS,
        "low_load", f"CLS < {STREAK_LOW_LOAD_CLS}",
        as_of_date_str,
    )

    records.deep_focus_streak = _compute_streaks(
        dates, day_metrics,
        lambda m: m["active_fdi"] is not None and m["active_fdi"] >= STREAK_DEEP_FOCUS_FDI,
        "deep_focus", f"active FDI ≥ {STREAK_DEEP_FOCUS_FDI}",
        as_of_date_str,
    )

    records.recovery_aligned_streak = _compute_streaks(
        dates, day_metrics,
        lambda m: m["avg_ras"] is not None and m["avg_ras"] >= STREAK_RECOVERY_ALIGNED_RAS,
        "recovery_aligned", f"avg RAS ≥ {STREAK_RECOVERY_ALIGNED_RAS}",
        as_of_date_str,
    )

    records.green_recovery_streak = _compute_streaks(
        dates, day_metrics,
        lambda m: m["recovery_score"] is not None and m["recovery_score"] >= STREAK_GREEN_RECOVERY,
        "green_recovery", f"WHOOP recovery ≥ {STREAK_GREEN_RECOVERY:.0f}%",
        as_of_date_str,
    )

    # ── Lifetime stats ─────────────────────────────────────────────────────
    records.lifetime = _compute_lifetime_stats(dates, summary)

    return records


def check_today_records(date_str: str, records: PersonalRecords) -> TodayRecordSummary:
    """
    Check whether today_date_str set any new personal records.

    Compares today's metrics against the all-time records in `records`.
    `records` must have been computed *including* today (as_of_date == date_str).
    """
    summary_obj = TodayRecordSummary(date_str=date_str)

    if not records.is_meaningful():
        return summary_obj

    # New all-time bests: today is the record-holder
    if records.best_fdi_day and records.best_fdi_day.date_str == date_str:
        summary_obj.new_best_fdi = True

    if records.best_cls_day and records.best_cls_day.date_str == date_str:
        summary_obj.new_best_cls = True

    if records.best_ras_day and records.best_ras_day.date_str == date_str:
        summary_obj.new_best_ras = True

    if records.best_dps_day and records.best_dps_day.date_str == date_str:
        summary_obj.new_best_dps = True

    if records.best_recovery_day and records.best_recovery_day.date_str == date_str:
        summary_obj.new_best_recovery = True

    if records.best_hrv_day and records.best_hrv_day.date_str == date_str:
        summary_obj.new_best_hrv = True

    # Active streaks (only surface ≥ 2 days to avoid false positives)
    if records.low_load_streak:
        summary_obj.low_load_streak_days = records.low_load_streak.current_streak
        # New streak record?
        if (records.low_load_streak.current_streak >= 2
                and records.low_load_streak.current_streak == records.low_load_streak.longest_streak
                and records.low_load_streak.longest_streak_end == date_str):
            summary_obj.new_streak_records.append("low_load")

    if records.deep_focus_streak:
        summary_obj.deep_focus_streak_days = records.deep_focus_streak.current_streak
        if (records.deep_focus_streak.current_streak >= 2
                and records.deep_focus_streak.current_streak == records.deep_focus_streak.longest_streak
                and records.deep_focus_streak.longest_streak_end == date_str):
            summary_obj.new_streak_records.append("deep_focus")

    if records.recovery_aligned_streak:
        summary_obj.recovery_aligned_streak_days = records.recovery_aligned_streak.current_streak
        if (records.recovery_aligned_streak.current_streak >= 2
                and records.recovery_aligned_streak.current_streak == records.recovery_aligned_streak.longest_streak
                and records.recovery_aligned_streak.longest_streak_end == date_str):
            summary_obj.new_streak_records.append("recovery_aligned")

    if records.green_recovery_streak:
        summary_obj.green_recovery_streak_days = records.green_recovery_streak.current_streak
        if (records.green_recovery_streak.current_streak >= 2
                and records.green_recovery_streak.current_streak == records.green_recovery_streak.longest_streak
                and records.green_recovery_streak.longest_streak_end == date_str):
            summary_obj.new_streak_records.append("green_recovery")

    summary_obj.has_records = bool(
        summary_obj.all_new_bests()
        or [s for s in summary_obj.active_streaks() if s[1] >= 2]
    )

    return summary_obj


def format_records_line(today: TodayRecordSummary) -> str:
    """
    Format a compact one-line Slack summary of today's records.

    Returns empty string if nothing notable happened.
    """
    if not today.has_records:
        return ""

    parts = []

    new_bests = today.all_new_bests()
    if new_bests:
        bests_str = " · ".join(new_bests)
        parts.append(f"🏆 *New personal best:* {bests_str}")

    active_streaks = today.active_streaks()
    for name, days in active_streaks:
        is_record = (
            (name == "Low load" and "low_load" in today.new_streak_records) or
            (name == "Deep focus" and "deep_focus" in today.new_streak_records) or
            (name == "Recovery aligned" and "recovery_aligned" in today.new_streak_records) or
            (name == "Green recovery" and "green_recovery" in today.new_streak_records)
        )
        if is_record:
            parts.append(f"🔥 *{name} streak:* {days} days (new record!)")
        elif days >= 3:
            parts.append(f"🔥 *{name} streak:* {days} days running")
        elif days >= 2:
            parts.append(f"✅ *{name} streak:* {days} days")

    return "\n".join(parts)


def format_records_section(records: PersonalRecords) -> str:
    """
    Format a full personal records section for terminal output or reports.

    Shows all-time bests, active streaks, and lifetime stats.
    """
    if not records.is_meaningful():
        return f"Personal records: not enough data yet ({records.total_days_analyzed} days tracked; need ≥ {MIN_DAYS_FOR_RECORDS})"

    lines = [
        "━━━ Personal Records ━━━",
        f"Based on {records.total_days_analyzed} tracked days",
        "",
    ]

    # All-time bests
    def _fmt_date(d: Optional[str]) -> str:
        if not d:
            return "—"
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            return dt.strftime("%-d %b")
        except ValueError:
            return d

    def _record_line(label: str, rec: Optional[DayRecord], fmt: str = ".2f") -> str:
        if not rec:
            return f"  {label}: —"
        return f"  {label}: {rec.value:{fmt}}  ({_fmt_date(rec.date_str)})"

    lines.append("All-time bests")
    lines.append(_record_line("Best FDI  (deepest focus)", records.best_fdi_day))
    lines.append(_record_line("Best CLS  (lightest load)", records.best_cls_day))
    lines.append(_record_line("Best RAS  (alignment)",     records.best_ras_day))
    if records.best_dps_day:
        lines.append(_record_line("Best DPS  (overall day)",   records.best_dps_day, ".0f"))
    if records.best_recovery_day:
        lines.append(_record_line("Best recovery          ", records.best_recovery_day, ".0f"))
    if records.best_hrv_day:
        lines.append(_record_line("Best HRV (ms)          ", records.best_hrv_day, ".0f"))

    lines.append("")
    lines.append("Active streaks")

    def _streak_line(name: str, streak: Optional[StreakRecord]) -> str:
        if not streak:
            return f"  {name}: —"
        cur = streak.current_streak
        best = streak.longest_streak
        cur_str = f"{cur}d current" if cur > 0 else "none"
        best_str = f"{best}d best" if best > 0 else "—"
        return f"  {name}: {cur_str}  ·  {best_str}"

    lines.append(_streak_line("Low load          ", records.low_load_streak))
    lines.append(_streak_line("Deep focus        ", records.deep_focus_streak))
    lines.append(_streak_line("Recovery aligned  ", records.recovery_aligned_streak))
    lines.append(_streak_line("Green recovery    ", records.green_recovery_streak))

    if records.lifetime:
        lt = records.lifetime
        lines.append("")
        lines.append("Lifetime")
        lines.append(f"  Total days tracked:       {lt.total_days}")
        lines.append(f"  Total meeting minutes:    {lt.total_meeting_minutes:,}")
        lines.append(f"  Total Slack msgs sent:    {lt.total_slack_sent:,}")
        if lt.best_week_avg_dps is not None:
            lines.append(f"  Best week avg DPS:        {lt.best_week_avg_dps:.1f}  (w/e {_fmt_date(lt.best_week_end_date)})")
        lines.append(f"  Days with all 5 sources:  {lt.days_all_sources}")

    return "\n".join(lines)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Presence Tracker — Personal Records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="As-of date (YYYY-MM-DD). Defaults to latest available date.",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument(
        "--today",
        metavar="DATE",
        help="Also check what records this specific date set.",
    )
    args = parser.parse_args()

    records = compute_personal_records(args.date)

    if args.json:
        print(json.dumps(records.to_dict(), indent=2, default=str))
        return

    print(format_records_section(records))

    if args.today:
        records_for_today = compute_personal_records(args.today)
        today_summary = check_today_records(args.today, records_for_today)
        line = format_records_line(today_summary)
        if line:
            print("\nToday's highlights:")
            print(line)
        else:
            print("\nNo new personal records today.")


if __name__ == "__main__":
    main()

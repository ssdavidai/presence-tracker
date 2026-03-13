# Presence Tracker

> What WHOOP does for physical strain, this does for cognitive load.

Presence Tracker ingests signals from WHOOP, Google Calendar, and Slack, slices every day into 96 × 15-minute observation windows, and computes five derived metrics that together describe your mental state throughout the day.

An AI-powered weekly report (Alfred Intuition) synthesises the patterns and delivers them to Slack. An ML model trains on the accumulated history once enough data is available.

---

## Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **CLS** — Cognitive Load Score | How mentally demanding was this window? | 0 (idle) → 1 (maximum) |
| **FDI** — Focus Depth Index | How deep and uninterrupted was focus? | 0 (fragmented) → 1 (deep) |
| **SDI** — Social Drain Index | How much social energy was spent? | 0 (isolated) → 1 (maxed) |
| **CSC** — Context Switch Cost | How much fragmentation penalty? | 0 (none) → 1 (max) |
| **RAS** — Recovery Alignment | Is workload appropriate for physiology? | 0 (misaligned) → 1 (aligned) |

---

## Data Sources (v1)

| Source | Signals |
|--------|---------|
| **WHOOP** | Recovery %, HRV, RHR, sleep hours, sleep performance, strain, SpO2 |
| **Google Calendar** | Event presence, duration, attendee count, organizer |
| **Slack** | Messages sent/received per 15-min window, channels active |

**Planned (v2):** RescueTime (app usage, focus/distraction time), Omi meeting transcripts.

---

## Architecture

```
Data Sources → Collectors → Chunker → JSONL Store → Analysis
                                           ↓
                              Alfred Intuition (weekly report)
                              ML Model Layer (after 60 days)
```

Each day produces `data/chunks/YYYY-MM-DD.jsonl` — 96 lines, one per 15-minute window. Everything else reads from these files.

---

## Setup

### Prerequisites

- Python 3.11+
- WHOOP developer app configured (uses existing `~/.clawdbot/whoop-tokens.json`)
- `gog` CLI authenticated with Google Calendar (`david@szabostuban.com`)
- OpenClaw running locally (for Slack API access and gateway)
- Temporal server running (for scheduled workflows)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run for today

```bash
python3 scripts/run_daily.py
```

### Run for a specific date

```bash
python3 scripts/run_daily.py 2026-03-13
```

### Backfill historical data

```bash
python3 scripts/backfill.py --days 30
python3 scripts/backfill.py --start 2026-01-01 --end 2026-03-01
```

### Run weekly analysis (manual trigger)

```bash
python3 scripts/run_analysis.py
```

---

## Scheduled Automation (Temporal)

Two Temporal workflows run automatically:

| Workflow | Schedule | Action |
|----------|----------|--------|
| `DailyIngestionWorkflow` | 23:45 Budapest daily | Ingest all sources, write JSONL |
| `WeeklyAnalysisWorkflow` | Sunday 21:00 Budapest | Deliver Presence Report to Slack |

### Start the worker

```bash
python3 temporal/worker.py
```

### Register schedules

```bash
python3 temporal/schedules.py
```

---

## Output Format

Each line in `data/chunks/YYYY-MM-DD.jsonl`:

```json
{
  "window_id": "2026-03-13T09:00:00",
  "date": "2026-03-13",
  "window_start": "2026-03-13T09:00:00+01:00",
  "window_end": "2026-03-13T09:15:00+01:00",
  "window_index": 36,
  "calendar": {
    "in_meeting": true,
    "meeting_title": "Product Sync",
    "meeting_attendees": 4,
    "meeting_duration_minutes": 60,
    "meetings_count": 1
  },
  "whoop": {
    "recovery_score": 85.0,
    "hrv_rmssd_milli": 72.4,
    "resting_heart_rate": 55.0,
    "sleep_performance": 89.0,
    "sleep_hours": 8.2,
    "strain": 12.1
  },
  "slack": {
    "messages_sent": 3,
    "messages_received": 12,
    "total_messages": 15,
    "channels_active": 2
  },
  "metrics": {
    "cognitive_load_score": 0.72,
    "focus_depth_index": 0.31,
    "social_drain_index": 0.58,
    "context_switch_cost": 0.45,
    "recovery_alignment_score": 0.88
  },
  "metadata": {
    "day_of_week": "Friday",
    "hour_of_day": 9,
    "is_working_hours": true,
    "sources_available": ["whoop", "calendar", "slack"]
  }
}
```

---

## Tests

```bash
# Unit tests (no credentials needed)
python3 -m pytest tests/test_metrics.py tests/test_chunker.py tests/test_store.py -v

# Integration tests (requires live APIs)
python3 -m pytest tests/test_collectors.py -v

# All tests
python3 -m pytest tests/ -v
```

---

## Directory Structure

```
presence-tracker/
├── SPEC.md                    # Full system specification
├── README.md                  # This file
├── requirements.txt
├── config.py                  # All configuration
├── collectors/
│   ├── whoop.py               # WHOOP API collector
│   ├── gcal.py                # Google Calendar collector
│   └── slack.py               # Slack collector
├── engine/
│   ├── chunker.py             # 15-min window builder + daily summary
│   ├── metrics.py             # Derived metric computation
│   └── store.py               # JSONL read/write + rolling stats
├── analysis/
│   └── intuition.py           # LLM-powered weekly pattern analysis
├── temporal/
│   ├── worker.py              # Temporal worker
│   ├── workflows.py           # DailyIngestion + WeeklyAnalysis
│   ├── activities.py          # Temporal activities
│   └── schedules.py           # Schedule registration
├── scripts/
│   ├── run_daily.py           # Manual daily ingestion
│   ├── run_analysis.py        # Manual weekly analysis trigger
│   └── backfill.py            # Historical data backfill
├── data/
│   ├── chunks/                # YYYY-MM-DD.jsonl files
│   └── summary/               # Rolling stats
└── tests/
    ├── test_metrics.py        # 21 unit tests
    ├── test_chunker.py        # 12 unit tests
    ├── test_store.py          # 7 unit tests
    └── test_collectors.py     # 14 integration tests
```

---

## Roadmap

- **v2:** RescueTime integration, Omi transcript analysis
- **v3:** ML model (Isolation Forest + Random Forest) after 60 days of data
- **v4:** Daily presence dashboard, recovery prediction
- **v5:** Multi-source anomaly alerts (real-time)

See [SPEC.md](SPEC.md) for the full architecture, metric formulas, and ML model design.

---

*Built by Alfred for David Szabo-Stuban — March 2026*

# Presence Tracker

> What WHOOP does for physical strain, this does for cognitive load.

Presence Tracker ingests signals from WHOOP, Google Calendar, Slack, RescueTime, and Omi, slices every day into 96 × 15-minute observation windows, and computes five derived metrics that together describe your mental state throughout the day.

An AI-powered weekly report (Alfred Intuition) synthesises the patterns and delivers them to Slack. A morning readiness brief fires at 07:00 Budapest time. An HTML dashboard is generated after each daily ingestion. An ML model trains on the accumulated history once 60 days of data are available.

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

## Data Sources

| Source | Signals | Status |
|--------|---------|--------|
| **WHOOP** | Recovery %, HRV, RHR, sleep hours, sleep performance, strain, SpO₂ | ✅ Live |
| **Google Calendar** | Event presence, duration, attendee count, organizer | ✅ Live |
| **Slack** | Messages sent/received per 15-min window, channels active | ✅ Live |
| **Omi** | Conversation active, word count, speech seconds, speech ratio | ✅ Live (v2) |
| **RescueTime** | App categories, focus/distraction time, context switches | ✅ Wired (requires API key) |

---

## Architecture

```
Data Sources → Collectors → Chunker → JSONL Store → Analysis
   WHOOP           ↓           ↓           ↓
   Calendar    per-window   96 × 15min  rolling.json
   Slack       signals      windows
   RescueTime
   Omi
                                           ↓
                              ┌────────────┴────────────┐
                              │                         │
                    Morning Brief (07:00)     Daily Digest (23:45)
                    Readiness tier            CLS/FDI/RAS summary
                    Day planning tip          Hourly sparkline
                                              Anomaly alerts
                                              HTML dashboard
                                              Alfred Intuition (weekly)
                                              ML Model (after 60 days)
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

### Check system status

```bash
python3 scripts/status.py               # Full health report
python3 scripts/status.py --brief       # One-line summary
python3 scripts/status.py --json        # Machine-readable JSON
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

### Morning brief (manual trigger)

```bash
python3 scripts/run_morning.py          # Send brief for today
python3 scripts/run_morning.py --dry-run  # Preview without sending
```

### Weekly analysis (manual trigger)

```bash
python3 scripts/run_analysis.py
```

### Recompute metrics on historical data

```bash
python3 scripts/recompute_metrics.py         # All days
python3 scripts/recompute_metrics.py --dry-run  # Preview changes
```

---

## Scheduled Automation (Temporal)

| Workflow | Schedule (Budapest) | Action |
|----------|---------------------|--------|
| `MorningBriefWorkflow` | 07:00 daily | WHOOP readiness + day planning brief |
| `DailyIngestionWorkflow` | 23:45 daily | Ingest all sources, write JSONL, dashboard, anomaly check |
| `WeeklyAnalysisWorkflow` | Sunday 21:00 | Alfred Intuition pattern report |
| `MonthlyMLRetrainWorkflow` | 1st of month 02:00 | Retrain Isolation Forest + Random Forest + KMeans |

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
  "omi": {
    "conversation_active": false,
    "word_count": 0,
    "speech_seconds": 0.0,
    "audio_seconds": 0.0,
    "sessions_count": 0,
    "speech_ratio": 0.0
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
    "is_active_window": true,
    "sources_available": ["whoop", "calendar", "slack"]
  }
}
```

---

## Daily Outputs

After each ingestion, the pipeline produces:

1. **JSONL chunk** — `data/chunks/YYYY-MM-DD.jsonl` (96 windows)
2. **Rolling summary** — `data/summary/rolling.json` (cumulative stats)
3. **HTML dashboard** — `data/dashboard/YYYY-MM-DD.html` (SVG charts, heatmap)
4. **Slack digest DM** — CLS/FDI/RAS summary + hourly sparkline
5. **Anomaly alerts** — Slack DM if CLS spike, FDI collapse, or RAS streak detected

Morning at 07:00:
- **Morning brief** — WHOOP readiness tier + scheduling recommendation

Sunday at 21:00:
- **Alfred Intuition report** — LLM pattern analysis across the week

---

## Anomaly Alerts (v5)

After each daily ingestion, three checks run against the 7-day baseline:

| Alert | Trigger |
|-------|---------|
| **CLS Spike** | Today's avg CLS > 7-day mean + 2× std dev |
| **FDI Collapse** | Today's active FDI drops >30% below 7-day baseline |
| **Recovery Streak** | RAS < 0.45 for 3+ consecutive days |

Alerts are delivered as Slack DMs to David.

---

## ML Model Layer (v3)

Activates after 60 days of data. Three models:

| Model | Purpose |
|-------|---------|
| **Isolation Forest** | Anomaly detection on daily CLS/FDI patterns |
| **Random Forest** | Predict tomorrow's WHOOP recovery from today's load |
| **KMeans** | Focus cluster labelling (when does David achieve deep focus?) |

```bash
python3 analysis/ml_model.py --status    # Check readiness
python3 analysis/ml_model.py --train     # Train all models
python3 analysis/ml_model.py --predict 2026-03-14  # Run inference
```

---

## Tests

```bash
# Unit tests (no credentials needed) — 765 tests
python3 -m pytest tests/ -v --ignore=tests/test_collectors.py

# Integration tests (requires live APIs)
python3 -m pytest tests/test_collectors.py -v

# All tests
python3 -m pytest tests/ -v
```

---

## Directory Structure

```
presence-tracker/
├── SPEC.md                       # Full system specification
├── README.md                     # This file
├── requirements.txt
├── config.py                     # All configuration
├── collectors/
│   ├── whoop.py                  # WHOOP API collector
│   ├── gcal.py                   # Google Calendar collector
│   ├── slack.py                  # Slack collector
│   ├── rescuetime.py             # RescueTime collector (v1.2)
│   └── omi.py                    # Omi transcript collector (v2.0)
├── engine/
│   ├── chunker.py                # 15-min window builder + daily summary
│   ├── metrics.py                # Derived metric computation
│   └── store.py                  # JSONL read/write + rolling stats
├── analysis/
│   ├── intuition.py              # LLM-powered weekly pattern analysis
│   ├── daily_digest.py           # End-of-day Slack DM
│   ├── morning_brief.py          # Morning readiness brief (07:00)
│   ├── dashboard.py              # HTML dashboard generator (v4)
│   ├── anomaly_alerts.py         # Multi-source anomaly detection (v5)
│   └── ml_model.py               # scikit-learn model layer (v3)
├── temporal/
│   ├── worker.py                 # Temporal worker
│   ├── workflows.py              # All Temporal workflows
│   ├── activities.py             # Temporal activities
│   └── schedules.py              # Schedule registration
├── scripts/
│   ├── status.py                 # System health status CLI (v5.2)
│   ├── run_daily.py              # Manual daily ingestion
│   ├── run_morning.py            # Manual morning brief
│   ├── run_analysis.py           # Manual weekly analysis trigger
│   ├── generate_dashboard.py     # Manual dashboard generation
│   ├── recompute_metrics.py      # Retroactive metric recomputation
│   └── backfill.py               # Historical data backfill
├── data/
│   ├── chunks/                   # YYYY-MM-DD.jsonl files
│   ├── dashboard/                # YYYY-MM-DD.html dashboards
│   ├── models/                   # Trained ML models (joblib)
│   └── summary/                  # Rolling stats (rolling.json)
└── tests/
    ├── test_metrics.py           # Metric formula unit tests
    ├── test_chunker.py           # Window builder tests
    ├── test_store.py             # JSONL store tests
    ├── test_collectors.py        # Integration tests (live APIs)
    ├── test_omi_collector.py     # Omi collector unit tests (v2)
    ├── test_daily_digest.py      # Daily digest tests
    ├── test_morning_brief.py     # Morning brief tests
    ├── test_dashboard.py         # Dashboard generator tests (v4)
    ├── test_anomaly_alerts.py    # Anomaly alert tests (v5)
    ├── test_ml_model.py          # ML model layer tests (v3)
    ├── test_status.py            # System status CLI tests (v5.2)
    ├── test_trend_context.py     # Multi-day trend tests
    ├── test_intuition.py         # Intuition layer tests
    └── test_recompute_metrics.py # Metric recomputation tests
```

---

## Changelog

| Version | What shipped |
|---------|-------------|
| **v1.0** | WHOOP + Calendar + Slack collectors, 5 metrics, JSONL store, Temporal workflows |
| **v1.1** | HRV and sleep_performance feeding into CLS and RAS |
| **v1.2** | RescueTime collector wired into FDI/CSC/CLS |
| **v1.3** | `is_active_window` flag, accurate active-window FDI |
| **v1.4** | Solo calendar blocks no longer inflate SDI/FDI/CLS/CSC |
| **v1.5** | RescueTime aggregation in daily summary; morning brief multi-day trends |
| **v2.0** | Omi transcript collector — 4th behavioral signal (conversation, word count, speech time) |
| **v3.0** | ML model layer — Isolation Forest, Random Forest, KMeans (graceful fallback before 60 days) |
| **v4.0** | Daily HTML dashboard — SVG CLS timeline, hourly heatmap, metric bars, recovery panel |
| **v5.0** | Multi-source anomaly alerts — CLS spike, FDI collapse, RAS streak; wired into DailyIngestionWorkflow |
| **v5.1** | Test coverage expansion — store layer (7→32 tests), Omi assertion fixes |
| **v5.2** | `scripts/status.py` — system health CLI with 7-section report, `--brief`, `--json` modes |
| **v7.2** | `MonthlyMLRetrainWorkflow` — automated monthly ML retraining; worker + schedules updated to register all workflows/activities |

---

*Built by Alfred for David Szabo-Stuban — March 2026*

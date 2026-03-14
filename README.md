# Presence Tracker

> What WHOOP does for physical strain, this does for cognitive load.

Presence Tracker ingests signals from WHOOP, Google Calendar, Slack, RescueTime, and Omi, slices every day into 96 × 15-minute observation windows, and computes five derived metrics that together describe your mental state throughout the day.

An AI-powered weekly report (Alfred Intuition) synthesises patterns and delivers them to Slack. An ML model trains on accumulated history for anomaly detection, recovery prediction, and focus clustering. A daily HTML dashboard visualises the full picture.

**Current version:** v9.2.0 — built autonomously in a single day (2026-03-14)

---

## The Five Metrics

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **CLS** — Cognitive Load Score | How mentally demanding was this window? Driven by meetings, Slack volume, RescueTime app switches, and calendar density | 0 (idle) → 1 (max load) |
| **FDI** — Focus Depth Index | How deep and uninterrupted was your focus? Penalised by context switching, rewarded by long uninterrupted active periods | 0 (fragmented) → 1 (deep flow) |
| **SDI** — Social Drain Index | How much social energy was spent? Meetings with multiple attendees, high Slack send volume | 0 (isolated) → 1 (maxed out) |
| **CSC** — Context Switch Cost | Fragmentation penalty. App switches from RescueTime + multi-channel Slack + back-to-back meetings | 0 (none) → 1 (max fragmentation) |
| **RAS** — Recovery Alignment Score | Is your workload appropriate for your physiology today? Compares CLS against WHOOP recovery using a smooth tanh curve | 0 (misaligned) → 1 (perfectly aligned) |

### Composite Scores

| Score | Formula | What it answers |
|-------|---------|----------------|
| **DPS** — Daily Presence Score | Weighted blend of all 5 metrics, 0–100 | "How was my cognitive day overall?" — the mental equivalent of WHOOP strain |
| **CDI** — Cognitive Debt Index | Rolling multi-day load vs recovery delta | "Am I accumulating fatigue I haven't paid back?" |

---

## Data Sources

| Source | Signals | Status |
|--------|---------|--------|
| **WHOOP** | Recovery %, HRV (rMSSD), RHR, sleep hours, sleep performance, strain, SpO2 | ✅ Live |
| **Google Calendar** | In-meeting, duration, attendee count, organizer | ✅ Live |
| **Slack** | Messages sent/received per 15-min window, channels active | ✅ Live |
| **RescueTime** | Focus seconds, distraction seconds, app switches, productivity score, top activity | ✅ Live |
| **Omi** | Ambient conversation transcripts — word count, language, topic tags, conversation duration | ✅ Live |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                          │
│   WHOOP · Calendar · Slack · RescueTime · Omi transcripts   │
└─────────────────────────────┬────────────────────────────────┘
                              │  one collector per source
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                     COLLECTORS (collectors/)                  │
│   whoop.py · gcal.py · slack.py · rescuetime.py · omi.py    │
│   Each returns normalised dict for its source's signals      │
└─────────────────────────────┬────────────────────────────────┘
                              │  raw signals per source
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                     CHUNKER (engine/chunker.py)              │
│   Splits day into 96 × 15-min windows (00:00 → 23:45)       │
│   Attaches all source signals to the correct window          │
│   Tags: is_working_hours, is_active_window, sources_available│
└─────────────────────────────┬────────────────────────────────┘
                              │  windows with raw signals
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                  METRIC ENGINE (engine/metrics.py)           │
│   Computes CLS · FDI · SDI · CSC · RAS per window           │
│   Uses RescueTime app_switches, WHOOP HRV, Slack volume,     │
│   calendar density — all source signals combined             │
└─────────────────────────────┬────────────────────────────────┘
                              │  96-window JSONL
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              DATA STORE  data/chunks/YYYY-MM-DD.jsonl        │
│   One file per day · One JSON line per 15-min window         │
│   Immutable once written — recompute_metrics.py for updates  │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────┼───────────────────────────────────┐
       ▼       ▼                                   ▼
┌──────────┐ ┌────────────────────────────┐ ┌─────────────────┐
│ ANALYSIS │ │  SCHEDULED DELIVERY        │ │  ML MODELS      │
│          │ │                            │ │                 │
│ Daily    │ │ 07:00 — Morning brief      │ │ Isolation Forest│
│ digest   │ │ 23:45 — Daily ingestion    │ │ anomaly detect  │
│          │ │ Sunday — Weekly intuition  │ │                 │
│ Weekly   │ │ 1st/month — ML retrain     │ │ Random Forest   │
│ summary  │ │                            │ │ recovery predict│
│          │ │ All delivered to Slack DM  │ │                 │
│ Anomaly  │ └────────────────────────────┘ │ KMeans focus    │
│ alerts   │                                │ clustering      │
│          │                                │                 │
│ Dashboard│                                │ Trained monthly │
│ (HTML)   │                                │ on all history  │
└──────────┘                                └─────────────────┘
```

---

## What You Receive, and When

### 07:00 Budapest — Morning Readiness Brief (Slack DM)

```
🧠 Morning Brief — Saturday 14 Mar

Recovery: 86% (Green) · HRV: 79ms · Sleep: 6.7h (82%)
Readiness: HIGH — push today, your body can take it

Yesterday: CLS 0.07 | FDI 0.94 | DPS 81 | CDI: 92 (no debt)

📈 7-day trends: HRV +3ms · Recovery stable · Load light
```

Tiers: **HIGH** (≥75% recovery) → push. **MODERATE** (50–74%) → steady. **LOW** (<50%) → protect.

### 23:45 Budapest — Nightly Digest (Slack DM)

After ingesting the full day's data:

```
🧠 Presence Report — Saturday 14 Mar

Recovery: 86% · HRV: 79ms · Sleep: 6.7h
─────────────────────────────────────────
CLS   0.07  ████░░░░░░ light
FDI   0.94  █████████░ deep focus
SDI   0.01  ░░░░░░░░░░ isolated
RAS   0.98  █████████░ aligned
DPS   81    ████████░░ strong day

Load curve: ▁▁▄▃▂▁▂▂▁▁▁▁▁  (7am–10pm)
Peak: 04:45 (0.245 — 31 Slack msgs)

CDI: 92 · 7-day trend: ↑ recovering
```

### Sunday 21:00 — Weekly Report (Slack DM, two parts)

**Part 1 — Deterministic summary:** Week-over-week metric deltas. No LLM. Reliable numbers.

**Part 2 — Alfred Intuition:** LLM-powered pattern interpretation. Identifies correlations (e.g. "HRV >75ms → FDI >0.8 on 4/5 days"), flags anomalies, surfaces insights.

### On-demand anomaly alerts

Fires during nightly ingestion if any threshold is breached:
- CLS spike >2 standard deviations above 7-day baseline
- FDI collapse >30% vs 7-day average
- Recovery misalignment 3+ consecutive days

---

## Directory Structure

```
presence-tracker/
├── README.md                       # This file
├── SPEC.md                         # Full mathematical specification
├── config.py                       # All configuration (API keys, thresholds, paths)
├── requirements.txt
│
├── collectors/                     # One file per data source
│   ├── whoop.py                    # WHOOP API — recovery, HRV, sleep, strain
│   ├── gcal.py                     # Google Calendar — meetings, attendees
│   ├── slack.py                    # Slack API — message volume per window
│   ├── rescuetime.py               # RescueTime — focus/distraction/app switches
│   ├── omi.py                      # Omi transcripts — ambient conversation signals
│   └── omi_topics.py               # Topic tagging for Omi conversations
│
├── engine/                         # Core data processing
│   ├── chunker.py                  # Slices day → 96 × 15-min windows
│   ├── metrics.py                  # Computes CLS/FDI/SDI/CSC/RAS per window
│   └── store.py                    # JSONL read/write + rolling stats
│
├── analysis/                       # Intelligence layer
│   ├── daily_digest.py             # Nightly Slack DM with metrics + sparkline
│   ├── morning_brief.py            # 7am WHOOP readiness brief
│   ├── intuition.py                # LLM weekly pattern analysis
│   ├── anomaly_alerts.py           # Multi-source threshold alerts
│   ├── presence_score.py           # DPS composite score (0–100)
│   ├── cognitive_debt.py           # CDI multi-day fatigue accumulation
│   ├── personal_baseline.py        # Personalized readiness tier thresholds
│   ├── ml_model.py                 # Isolation Forest + RF + KMeans models
│   ├── dashboard.py                # HTML dashboard generator
│   └── focus_planner.py           # Focus window recommendations
│
├── temporal/                       # Scheduled automation (Temporal workflows)
│   ├── workflows.py                # 4 workflows (see below)
│   ├── activities.py               # Temporal activity wrappers
│   ├── schedules.py                # Schedule registration
│   └── worker.py                   # Worker process
│
├── scripts/                        # CLI tools — run manually or in cron
│   ├── run_daily.py                # Manual daily ingestion trigger
│   ├── run_morning.py              # Manual morning brief trigger
│   ├── run_analysis.py             # Manual weekly analysis trigger
│   ├── backfill.py                 # Historical data backfill
│   ├── recompute_metrics.py        # Recompute metrics on existing JSONL
│   ├── report.py                   # Terminal presence report (--date, --trend, --week)
│   ├── weekly_summary.py           # Deterministic week-over-week summary
│   ├── generate_dashboard.py       # Regenerate HTML dashboard
│   ├── train_model.py              # ML model trainer CLI
│   ├── export.py                   # Export data as CSV/JSON
│   └── status.py                   # 7-section system health check
│
├── data/
│   ├── chunks/                     # YYYY-MM-DD.jsonl — one per day
│   ├── summary/                    # Rolling stats
│   ├── dashboard/                  # YYYY-MM-DD.html dashboards
│   └── models/                     # Trained ML model files (.pkl)
│
└── tests/                          # 26 test files, ~470 passing tests
    ├── test_metrics.py
    ├── test_chunker.py
    ├── test_store.py
    ├── test_collectors.py
    ├── test_ml_model.py
    └── ... (21 more)
```

---

## Temporal Workflows (Scheduled Automation)

Four workflows run automatically via the Alfred Temporal worker:

| Workflow | Schedule | What it does |
|----------|----------|-------------|
| `DailyIngestionWorkflow` | 23:45 Budapest, daily | Ingests all sources → JSONL, generates dashboard, runs anomaly alerts, sends nightly digest |
| `MorningBriefWorkflow` | 07:00 Budapest, daily | Reads yesterday's WHOOP data, sends readiness brief with CDI and 7-day trends |
| `WeeklyAnalysisWorkflow` | Sunday 21:00 Budapest | Deterministic weekly summary + LLM Intuition report, both sent to Slack |
| `MonthlyMLRetrainWorkflow` | 1st of month, 02:00 Budapest | Retrains Isolation Forest, Random Forest, and KMeans on all accumulated data |

The presence tracker uses its **own** Temporal worker (`temporal/worker.py`) — separate from Alfred's main worker. It registers via the Alfred worker in `~/clawd/temporal-workflows/worker.py`.

---

## CLI Tools

```bash
# System health — 7-section status report
python3 scripts/status.py

# Terminal presence report for a specific day
python3 scripts/report.py --date 2026-03-14

# Multi-day trend table (last 7 days)
python3 scripts/report.py --trend 7

# Weekly terminal summary
python3 scripts/report.py --week

# Run daily ingestion manually
python3 scripts/run_daily.py
python3 scripts/run_daily.py 2026-03-13   # specific date

# Backfill historical data
python3 scripts/backfill.py --days 30
python3 scripts/backfill.py --start 2026-01-01 --end 2026-03-01

# Recompute metrics on existing JSONL (after formula changes)
python3 scripts/recompute_metrics.py --all
python3 scripts/recompute_metrics.py --date 2026-03-14

# Train ML models manually
python3 scripts/train_model.py
python3 scripts/train_model.py --force    # train even with <60 days data

# Export data
python3 scripts/export.py --format csv --out ~/presence-export.csv
python3 scripts/export.py --format json --days 30

# Regenerate HTML dashboard
python3 scripts/generate_dashboard.py --date 2026-03-14
```

---

## ML Model Layer

Three models, trained on all accumulated JSONL data, retrained monthly:

| Model | Algorithm | What it does |
|-------|-----------|-------------|
| **Anomaly detector** | Isolation Forest | Flags days where the combination of metrics is unusual vs your personal baseline |
| **Recovery predictor** | Random Forest | Predicts tomorrow's WHOOP recovery from today's CLS/FDI/SDI/RAS pattern |
| **Focus clusters** | KMeans (k=4) | Labels each day: Deep Work / Meeting Heavy / Recovery / Fragmented |

Models stored in `data/models/`. Require ≥60 days of data to be meaningful; retrain runs anyway from day 1 as a warm-up baseline.

---

## Data Format

Each line in `data/chunks/YYYY-MM-DD.jsonl` represents one 15-minute window:

```json
{
  "window_id": "2026-03-14T09:00:00",
  "date": "2026-03-14",
  "window_start": "2026-03-14T09:00:00+01:00",
  "window_end":   "2026-03-14T09:15:00+01:00",
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
    "strain": 12.1,
    "spo2_percentage": 97.2
  },
  "slack": {
    "messages_sent": 3,
    "messages_received": 12,
    "total_messages": 15,
    "channels_active": 2
  },
  "rescuetime": {
    "focus_seconds": 540,
    "distraction_seconds": 120,
    "neutral_seconds": 240,
    "active_seconds": 900,
    "app_switches": 4,
    "productivity_score": 72,
    "top_activity": "VS Code"
  },
  "omi": {
    "conversation_active": true,
    "word_count": 187,
    "duration_seconds": 420,
    "language": "en",
    "topic_tags": ["technical", "planning"]
  },
  "metrics": {
    "cognitive_load_score": 0.72,
    "focus_depth_index": 0.31,
    "social_drain_index": 0.58,
    "context_switch_cost": 0.45,
    "recovery_alignment_score": 0.88
  },
  "metadata": {
    "day_of_week": "Saturday",
    "hour_of_day": 9,
    "minute_of_hour": 0,
    "is_working_hours": true,
    "is_active_window": true,
    "sources_available": ["whoop", "calendar", "slack", "rescuetime", "omi"]
  }
}
```

---

## Setup

### Prerequisites

- Python 3.11+
- WHOOP developer app credentials → `~/.clawdbot/whoop-tokens.json`
- `gog` CLI authenticated: `gog auth david@szabostuban.com`
- RescueTime API key → `config.py` / env var `RESCUETIME_API_KEY`
- Omi transcripts at `~/omi/` (auto-populated by Omi wearable pipeline)
- OpenClaw running locally (Slack API access + gateway)
- Temporal server running (`cd ~/services/temporal && docker compose up -d`)

### Install

```bash
pip install -r requirements.txt
```

### First run

```bash
# Ingest today
python3 scripts/run_daily.py

# Check system health
python3 scripts/status.py

# Backfill last 30 days
python3 scripts/backfill.py --days 30

# Re-register Temporal schedules
python3 temporal/schedules.py
```

---

## Tests

```bash
# All tests (fast — no credentials needed except collectors)
python3 -m pytest tests/ -v

# Unit tests only (no live API calls)
python3 -m pytest tests/ -v --ignore=tests/test_collectors.py

# Single module
python3 -m pytest tests/test_metrics.py -v
```

~470 tests across 26 files. All passing as of v9.2.0.

---

## Version History

| Version | Date | What shipped |
|---------|------|-------------|
| v1.0 | 2026-03-13 | Initial release — WHOOP + Calendar + Slack, 5 metrics, Temporal workflows |
| v1.1–v1.16 | 2026-03-14 morning | Daily digest, RAS formula fix, rich weekly analytics, RescueTime collector, metric quality fixes, ML model layer, morning brief, CLS sparkline, historical recompute |
| v2.0 | 2026-03-14 | Omi transcript integration — ambient conversation signals in all metrics |
| v4.0 | 2026-03-14 | Daily HTML presence dashboard |
| v5.0 | 2026-03-14 | Multi-source anomaly alerts |
| v6.0 | 2026-03-14 | Personal baseline — personalized readiness tier thresholds |
| v7.0 | 2026-03-14 | Calendar-aware morning brief, Cognitive Debt Index (CDI) |
| v7.2 | 2026-03-14 | MonthlyMLRetrainWorkflow — automated monthly retraining |
| v7.3 | 2026-03-14 | Daily Presence Score (DPS) — single composite 0–100 score |
| v9.1–9.2 | 2026-03-14 | Multi-day trend table, weekly terminal summary |

---

*Built by Alfred for David Szabo-Stuban — March 2026*
*v1.0 to v9.2 shipped autonomously in 13 hours via self-improvement loop*

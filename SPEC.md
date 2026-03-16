# Presence Tracker — System Specification

**Version:** 1.0  
**Author:** Alfred (AI Butler)  
**Status:** Active  
**Last updated:** 2026-03-13

---

## Overview

Presence Tracker is a personal cognitive load and mental strain analytics system. It ingests signals from multiple data sources, chunks the day into 15-minute observation windows, infers composite metrics, and builds toward an ML model trained on real behavioral + physiological data.

The goal: understand mental and cognitive state with the same depth WHOOP brings to physical strain.

---

## Problem Statement

WHOOP answers: *"How physically recovered are you?"*  
Presence Tracker answers: *"How cognitively loaded were you, and why?"*

No single wearable captures the full picture. Mental strain is a composite of:
- Physiological signals (WHOOP: HRV, recovery, sleep, RHR)
- Behavioral signals (RescueTime: app usage, focus depth, context switching)
- Social signals (Slack: message volume, response latency, meeting load)
- Calendar intent vs. actual execution
- Recorded conversations (Omi/meeting transcripts)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  DATA SOURCES                        │
│  WHOOP • Calendar • RescueTime • Slack • Omi        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              INGESTION LAYER (Python)                │
│  One collector per source → normalized JSON          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              CHUNKING ENGINE                         │
│  Splits day into 96 × 15-min windows                 │
│  Attaches all signals to each window                 │
│  Outputs: daily JSONL file                           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              METRIC INFERENCE                        │
│  Computes derived metrics per window:                │
│  • Cognitive Load Score (CLS)                        │
│  • Focus Depth Index (FDI)                           │
│  • Social Drain Index (SDI)                          │
│  • Context Switch Cost (CSC)                         │
│  • Recovery Alignment Score (RAS)                    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         STORAGE: data/chunks/YYYY-MM-DD.jsonl       │
│         Each line = one 15-min window record         │
└──────────────────────┬──────────────────────────────┘
                       │
             ┌─────────┴────────┐
             ▼                  ▼
┌────────────────────┐  ┌──────────────────────────┐
│  ALFRED INTUITION  │  │    ML MODEL LAYER         │
│  (LLM-powered)     │  │    (scikit-learn)         │
│  Pattern detection │  │    Trained after 60 days  │
│  Narrative reports │  │    Anomaly detection       │
│  Slack delivery    │  │    Prediction              │
└────────────────────┘  └──────────────────────────┘
```

---

## Data Sources

### Currently Implemented (v1)

| Source | Data | Status |
|--------|------|--------|
| WHOOP | Recovery, HRV, RHR, sleep performance, sleep stages, strain | ✅ Live |
| Google Calendar | Events, duration, attendee count, titles | ✅ Live |
| Slack | Messages sent, messages received, channels active | ✅ Live |

### Planned (v2+)

| Source | Data | Notes |
|--------|------|-------|
| RescueTime | App categories, focus time, distraction time | Needs API key |
| Omi | Meeting transcripts, sentiment, topic density | Via existing Omi integration |
| Meeting recordings | Transcript analysis, speaker time | Future |

---

## The 15-Minute Window Schema

Each JSONL line represents one 15-minute observation window:

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
    "meeting_organizer": "david@szabostuban.com",
    "meetings_count": 1
  },

  "whoop": {
    "recovery_score": 85.0,
    "hrv_rmssd_milli": 72.4,
    "resting_heart_rate": 55.0,
    "sleep_performance": 89.0,
    "sleep_hours": 8.2,
    "strain": null,
    "spo2_percentage": 95.1
  },

  "slack": {
    "messages_sent": 3,
    "messages_received": 12,
    "threads_active": 2,
    "channels_active": 1
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

## Derived Metrics

### Cognitive Load Score (CLS) — 0.0 to 1.0
Higher = more cognitively demanding.

```
CLS = weighted_avg(
  meeting_density:     0.35,  # % of window in meetings
  slack_volume:        0.25,  # normalized messages per 15min
  calendar_pressure:   0.20,  # meetings × attendees
  recovery_inverse:    0.20,  # 1 - (recovery_score / 100)
)
```

### Focus Depth Index (FDI) — 0.0 to 1.0
Higher = deeper focus, fewer interruptions.

```
FDI = 1 - weighted_avg(
  in_meeting:          0.40,  # binary: 0 or 1
  slack_interruptions: 0.40,  # messages received / max_expected
  context_switches:    0.20,  # app switches / max_expected (v2: RescueTime)
)
```

### Social Drain Index (SDI) — 0.0 to 1.0
Higher = more social energy expenditure.

```
SDI = weighted_avg(
  meeting_attendees:   0.50,  # normalized by typical max (10)
  meetings_in_window:  0.30,  # binary / count
  slack_sent_ratio:    0.20,  # messages_sent / (sent + received)
)
```

### Context Switch Cost (CSC) — 0.0 to 1.0
Higher = more costly context fragmentation.

```
CSC = weighted_avg(
  meetings_per_hour:   0.50,  # switches in meetings
  slack_channel_switches: 0.30,  # cross-channel activity
  calendar_fragmentation: 0.20,  # short meetings < 30min
)
```

### Recovery Alignment Score (RAS) — 0.0 to 1.0
Higher = activity matches physiological readiness.

```
RAS = recovery_score / 100 * (1 - CLS)
# High recovery + low CLS = aligned (good)
# Low recovery + high CLS = misaligned (concerning)
```

---

## ML Model Layer

**Timeline:** Activate after 60 days of data (≥ 60 × 96 windows with full signals).

**Features (per window):**
- All raw signal values
- All derived metrics
- Temporal features (hour, day-of-week, window_index)
- Rolling 7-day averages for WHOOP metrics

**Initial models:**
1. **Anomaly detector** (Isolation Forest) — flags unusual cognitive load patterns
2. **Recovery predictor** (Random Forest) — predicts next-day WHOOP recovery from today's CLS pattern
3. **Focus optimizer** (clustering) — identifies your best focus windows by day-of-week

**Retraining:** Monthly via Temporal schedule.

---

## Alfred Intuition Layer

An LLM (Claude Sonnet) runs weekly pattern analysis over the JSONL data and delivers a Slack report.

**Report includes:**
- "Your highest CLS days this week: Wednesday (0.81), Monday (0.74)"
- "Your deepest focus windows: Tuesday 9–11am, Thursday 7–9am"
- "When your HRV drops below 60ms: your afternoon CLS spikes 40%"
- "Recovery alignment warning: 3 days of >0.7 CLS on <60% recovery"
- Actionable recommendation for the coming week

**Delivery:** Weekly (Sunday evening) to David's Slack DM.

---

## Temporal Workflow

**Schedule:** Daily at 23:45 Budapest time (end of day).

**Steps:**
1. Ingest WHOOP data for today
2. Ingest Calendar events for today
3. Ingest Slack activity for today
4. Build 96 window chunks with all signals
5. Compute derived metrics for each window
6. Write `data/chunks/YYYY-MM-DD.jsonl`
7. Update rolling summary stats
8. Notify #alfred-logs with daily digest

**Weekly analysis:** Sunday 21:00 Budapest time — Alfred Intuition report.

---

## Directory Structure

```
presence-tracker/
├── SPEC.md                    # This document
├── README.md                  # Setup and usage guide
├── requirements.txt           # Python dependencies
├── config.py                  # Configuration (paths, weights, thresholds)
├── collectors/
│   ├── __init__.py
│   ├── whoop.py               # WHOOP API collector
│   ├── calendar.py            # Google Calendar collector
│   ├── slack.py               # Slack collector
│   └── rescuetime.py          # RescueTime (v2)
├── engine/
│   ├── __init__.py
│   ├── chunker.py             # 15-min window builder
│   ├── metrics.py             # Derived metric computation
│   └── store.py               # JSONL read/write
├── analysis/
│   ├── __init__.py
│   ├── intuition.py           # LLM-powered pattern analysis
│   ├── ml_model.py            # scikit-learn model training/inference
│   ├── load_decomposer.py     # CLS source attribution (meetings/Slack/physiology/RT/Omi)
│   └── ...                    # (many more: anomaly_alerts, daily_digest, morning_brief, ...)
├── temporal/
│   ├── __init__.py
│   ├── worker.py              # Temporal worker
│   ├── workflows.py           # Presence tracking workflows
│   ├── activities.py          # Temporal activities
│   └── schedules.py           # Schedule registration
├── scripts/
│   ├── run_daily.py           # Manual trigger: run today's ingestion
│   ├── run_analysis.py        # Manual trigger: run intuition analysis
│   └── backfill.py            # Backfill historical data
├── data/
│   ├── chunks/                # YYYY-MM-DD.jsonl files
│   ├── models/                # Trained ML models
│   └── summary/               # Rolling stats
└── tests/
    ├── test_chunker.py
    ├── test_metrics.py
    └── test_collectors.py
```

---

## Exit Criteria (v1 Complete)

- [x] All collectors implemented and tested (WHOOP, Calendar, Slack)
- [x] Chunking engine produces valid 96-window JSONL for any date
- [x] All 5 derived metrics computed correctly
- [x] Daily Temporal workflow runs and writes output file
- [x] Weekly Intuition analysis runs and delivers Slack report
- [x] End-to-end test: run today → verify JSONL output → verify metrics
- [x] Documentation complete (README + SPEC)
- [x] GitHub repo: ssdavidai/presence-tracker

---

## Future Roadmap

**v2:** RescueTime integration, Omi transcript analysis  
**v3:** ML model activation (after 60 days of data)  
**v4:** Real-time dashboard (web UI), daily recovery prediction  
**v5:** Multi-person (family) tracking, comparative analysis

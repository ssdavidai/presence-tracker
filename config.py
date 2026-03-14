"""
Presence Tracker — Configuration

All tuneable parameters, paths, and credentials live here.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
MODELS_DIR = DATA_DIR / "models"
SUMMARY_DIR = DATA_DIR / "summary"

# Ensure directories exist
for _dir in [CHUNKS_DIR, MODELS_DIR, SUMMARY_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── RescueTime ──────────────────────────────────────────────────────────────

# API key loaded from environment or ~/.clawdbot/clawdbot.json
# Set RESCUETIME_API_KEY env var, or add "rescuetime_api_key" to clawdbot.json
RESCUETIME_API_KEY = os.environ.get("RESCUETIME_API_KEY", "")

# RescueTime API base URL (key-based access)
RESCUETIME_API_BASE = "https://www.rescuetime.com/anapi/data"
RESCUETIME_DAILY_SUMMARY_URL = "https://www.rescuetime.com/anapi/daily_summary_feed"

# Maximum app switches per 15-min window before saturation (for CSC normalization)
MAX_APP_SWITCHES_PER_WINDOW = 8

# Minimum active seconds in a window to count as "computer-active"
# (prevents noisy single-second events from inflating metrics)
RT_MIN_ACTIVE_SECONDS = 10


# ─── WHOOP ────────────────────────────────────────────────────────────────────

# WHOOP tokens are managed by the existing skill at:
WHOOP_TOKENS_PATH = Path.home() / ".clawdbot" / "whoop-tokens.json"

# WHOOP API base
WHOOP_API_BASE = "https://api.prod.whoop.com/developer/v1"

# ─── Google Calendar ──────────────────────────────────────────────────────────

# Calendar ID for David's main calendar
CALENDAR_ID = "david@szabostuban.com"

# gog CLI path
GOG_CLI = "gog"

# ─── Slack ────────────────────────────────────────────────────────────────────

SLACK_USER_ID = "U08UGBQL5J5"       # David
SLACK_DM_CHANNEL = "D08U7C4G6TE"    # David's DM
SLACK_LOGS_CHANNEL = "C0ACVH414JC"  # #alfred-logs
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")  # Set in environment

# ─── OpenClaw Gateway ─────────────────────────────────────────────────────────

GATEWAY_URL = "http://localhost:18789"
def _read_gateway_token() -> str:
    """
    Read gateway token from OpenClaw config.

    OpenClaw uses two auth modes:
    - Bearer token: gateway.auth.token (for external callers)
    - Password: gateway.auth.password (for local callers)

    The existing temporal-workflows use the password as the bearer token.
    """
    import json
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    try:
        config = json.loads(config_path.read_text())
        # Use the password field — this is what the existing workflows use
        password = config.get("gateway", {}).get("auth", {}).get("password")
        if password:
            return password
    except Exception:
        pass
    return "OmurpJoKRVzBjNtSBx-kh3El9vnfeVLE"

GATEWAY_TOKEN = _read_gateway_token()

# ─── Temporal ─────────────────────────────────────────────────────────────────

TEMPORAL_ADDRESS = "127.0.0.1:7233"
TASK_QUEUE = "presence-tracker"

# ─── Timezone ─────────────────────────────────────────────────────────────────

TIMEZONE = "Europe/Budapest"

# ─── Working Hours ────────────────────────────────────────────────────────────

WORKING_HOURS_START = 7   # 7am
WORKING_HOURS_END = 22    # 10pm

# ─── Metric Weights ───────────────────────────────────────────────────────────

# Cognitive Load Score weights
CLS_WEIGHTS = {
    "meeting_density": 0.35,
    "slack_volume": 0.25,
    "calendar_pressure": 0.20,
    "recovery_inverse": 0.20,
}

# Focus Depth Index weights
FDI_WEIGHTS = {
    "in_meeting": 0.40,
    "slack_interruptions": 0.40,
    "context_switches": 0.20,
}

# Social Drain Index weights
SDI_WEIGHTS = {
    "meeting_attendees": 0.50,
    "meetings_in_window": 0.30,
    "slack_sent_ratio": 0.20,
}

# Context Switch Cost weights
CSC_WEIGHTS = {
    "meetings_per_hour": 0.50,
    "slack_channel_switches": 0.30,
    "calendar_fragmentation": 0.20,
}

# ─── Normalization Maxima ─────────────────────────────────────────────────────

# Used to normalize raw values to [0, 1]
MAX_SLACK_MESSAGES_PER_WINDOW = 30     # messages per 15min
MAX_MEETING_ATTENDEES = 10
MAX_MEETINGS_PER_WINDOW = 2
MAX_SLACK_CHANNELS_PER_WINDOW = 5

# ─── ML Model ────────────────────────────────────────────────────────────────

# Minimum days of data required before training
ML_MIN_DAYS = 60

# Model file paths
ISOLATION_FOREST_PATH = MODELS_DIR / "isolation_forest.pkl"
RECOVERY_PREDICTOR_PATH = MODELS_DIR / "recovery_predictor.pkl"

# ─── Intuition Analysis ───────────────────────────────────────────────────────

# Days of history to include in weekly intuition report
INTUITION_LOOKBACK_DAYS = 7

# Recovery alert threshold
RECOVERY_ALERT_THRESHOLD = 50.0  # below this = flag

# CLS high-load threshold
CLS_HIGH_LOAD_THRESHOLD = 0.70

# ─── Schedules ───────────────────────────────────────────────────────────────

# Daily ingestion: 23:45 Budapest time
DAILY_CRON = "45 23 * * *"

# Weekly analysis: Sunday 21:00 Budapest time
WEEKLY_CRON = "0 21 * * 0"

# ML retraining: 1st of month 02:00
MONTHLY_RETRAIN_CRON = "0 2 1 * *"

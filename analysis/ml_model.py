"""
Presence Tracker — ML Model Layer

Trains and runs three scikit-learn models on the accumulated JSONL history:

  1. Isolation Forest (anomaly detector)
     Detects windows where the cognitive load pattern is unusual relative
     to David's personal baseline.  Flags days where the combination of
     signals (HRV, CLS, meeting load, Slack volume) is anomalous.

  2. Random Forest Regressor (recovery predictor)
     Predicts tomorrow's WHOOP recovery score from today's cognitive load
     pattern.  After 60 days of data this becomes a meaningful personalised
     model; before that it returns a baseline estimate.

  3. KMeans Clustering (focus optimizer)
     Groups working-hour windows by behavioral signature to identify when
     David reliably achieves deep focus.  Output: the top N window types
     that correspond to high FDI.

Activation: The SPEC requires ≥ 60 days of data before training.  Functions
return graceful fallbacks (None / empty dicts) when insufficient data exists.
All models are persisted to data/models/ using joblib.

Design principles:
  - Feature extraction is deterministic and testable (pure functions)
  - Training and inference are separated — models can be retrained without
    changing the inference path
  - All functions degrade gracefully when data is insufficient
  - No model is required for the pipeline to run — ML is an enhancement layer

Usage:
    # Train all models (run monthly via Temporal)
    from analysis.ml_model import train_all, MIN_DAYS_REQUIRED
    result = train_all()  # returns dict with status per model

    # Run inference on today's windows
    from analysis.ml_model import detect_anomalies, predict_recovery
    anomalies = detect_anomalies(today_windows)
    recovery_pred = predict_recovery(today_windows)

    # Check readiness
    from analysis.ml_model import is_ready_to_train, get_data_status
    status = get_data_status()

    # CLI: python3 analysis/ml_model.py [--train] [--status] [--predict DATE]
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS_DIR, ML_MIN_DAYS
from engine.store import list_available_dates, read_day

# Suppress convergence / n_init warnings from sklearn in non-training contexts
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ─── Constants ────────────────────────────────────────────────────────────────

MIN_DAYS_REQUIRED = ML_MIN_DAYS   # 60 days before training
MIN_WINDOWS_FOR_CLUSTER = 50      # minimum active windows for clustering

# Model paths (set in config.py)
_ANOMALY_MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
_RECOVERY_MODEL_PATH = MODELS_DIR / "recovery_predictor.pkl"
_FOCUS_CLUSTER_PATH = MODELS_DIR / "focus_clusters.pkl"
_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
_MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Feature names — must stay in sync with extract_window_features()
FEATURE_NAMES = [
    # Derived metrics (the 5 core signals)
    "cls",
    "fdi",
    "sdi",
    "csc",
    "ras",
    # Raw physiological signals
    "recovery_score",
    "hrv_rmssd",
    "sleep_performance",
    "sleep_hours",
    # Calendar signals
    "in_meeting",
    "meeting_attendees",
    "meeting_duration_minutes",
    "meetings_count",
    # Slack signals
    "messages_sent",
    "messages_received",
    "channels_active",
    # Temporal features
    "hour_of_day",
    "window_index",
    "is_working_hours",
    # Derived behavioral signals
    "slack_intensity",          # messages_sent / (sent + received + 1)
    "meeting_density_hour",     # fraction of hour in meetings (0–1)
    "physiological_load_ratio", # cls / max(recovery_score/100, 0.01)
]

N_FEATURES = len(FEATURE_NAMES)


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_window_features(window: dict) -> Optional[np.ndarray]:
    """
    Extract a fixed-length feature vector from one 15-minute window.

    Returns None if critical fields are missing (e.g. no WHOOP data).
    All values are raw — call with a StandardScaler before training/inference.

    Feature vector layout: see FEATURE_NAMES (N_FEATURES elements).
    """
    metrics = window.get("metrics", {})
    whoop = window.get("whoop", {})
    cal = window.get("calendar", {})
    slack = window.get("slack", {})
    meta = window.get("metadata", {})

    # Require at minimum the 5 derived metrics
    cls = metrics.get("cognitive_load_score")
    fdi = metrics.get("focus_depth_index")
    sdi = metrics.get("social_drain_index")
    csc = metrics.get("context_switch_cost")
    ras = metrics.get("recovery_alignment_score")

    if any(v is None for v in [cls, fdi, sdi, csc, ras]):
        return None

    # Physiological signals — use neutral defaults when absent
    recovery = whoop.get("recovery_score") or 50.0
    hrv = whoop.get("hrv_rmssd_milli") or 65.0
    sleep_perf = whoop.get("sleep_performance") or 75.0
    sleep_h = whoop.get("sleep_hours") or 7.0

    # Calendar signals
    in_meeting = 1.0 if cal.get("in_meeting", False) else 0.0
    attendees = float(cal.get("meeting_attendees", 0))
    duration = float(cal.get("meeting_duration_minutes", 0))
    meetings_count = float(cal.get("meetings_count", 0))

    # Slack signals
    sent = float(slack.get("messages_sent", 0))
    received = float(slack.get("messages_received", 0))
    channels = float(slack.get("channels_active", 0))

    # Temporal features
    hour = float(meta.get("hour_of_day", 12))
    window_idx = float(window.get("window_index", 48))
    is_working = 1.0 if meta.get("is_working_hours", False) else 0.0

    # Derived behavioral signals
    total_msg = sent + received
    slack_intensity = sent / (total_msg + 1.0)  # +1 to avoid division by zero
    meeting_density = duration / 60.0            # hours (meeting duration / 1 hour)
    phys_load_ratio = cls / max(recovery / 100.0, 0.01)

    features = np.array([
        cls, fdi, sdi, csc, ras,
        recovery, hrv, sleep_perf, sleep_h,
        in_meeting, attendees, duration, meetings_count,
        sent, received, channels,
        hour, window_idx, is_working,
        slack_intensity, meeting_density, phys_load_ratio,
    ], dtype=np.float64)

    assert len(features) == N_FEATURES, f"Feature count mismatch: {len(features)} vs {N_FEATURES}"
    return features


def extract_daily_features(windows: list[dict]) -> Optional[np.ndarray]:
    """
    Extract a single daily feature vector by aggregating a day's windows.

    Used for the recovery predictor (which predicts next-day recovery from
    the current day's aggregate pattern — not per-window).

    Returns None if fewer than 10 valid windows were extracted.
    """
    working = [w for w in windows if w["metadata"].get("is_working_hours", False)]
    if not working:
        return None

    vectors = []
    for w in working:
        v = extract_window_features(w)
        if v is not None:
            vectors.append(v)

    if len(vectors) < 10:
        return None

    arr = np.vstack(vectors)

    # Daily features: mean, max, and std of each window feature
    daily = np.concatenate([
        arr.mean(axis=0),   # 22 mean features
        arr.max(axis=0),    # 22 max features
        arr.std(axis=0),    # 22 std features
    ])

    return daily


def build_feature_matrix(
    dates: list[str],
    working_hours_only: bool = True,
    active_only: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """
    Build a 2D feature matrix from all windows across the given dates.

    Returns:
        X: np.ndarray of shape (n_windows, N_FEATURES) — raw features
        meta: list of dicts with {date, window_id, window_index, hour_of_day}
              for tracing anomaly predictions back to real windows.

    Windows for which feature extraction fails are silently skipped.
    """
    all_vectors = []
    all_meta = []

    for date_str in dates:
        day_windows = read_day(date_str)
        for w in day_windows:
            if working_hours_only and not w["metadata"].get("is_working_hours", False):
                continue
            if active_only and not w["metadata"].get("is_active_window", False):
                continue
            v = extract_window_features(w)
            if v is not None:
                all_vectors.append(v)
                all_meta.append({
                    "date": w["date"],
                    "window_id": w["window_id"],
                    "window_index": w.get("window_index", -1),
                    "hour_of_day": w["metadata"].get("hour_of_day", -1),
                })

    if not all_vectors:
        return np.empty((0, N_FEATURES)), []

    return np.vstack(all_vectors), all_meta


# ─── Data readiness ───────────────────────────────────────────────────────────

def get_data_status() -> dict:
    """
    Return a summary of how much data is available and whether ML is ready.

    Useful for health checks and status reporting.
    """
    dates = list_available_dates()
    n_days = len(dates)
    oldest = dates[0] if dates else None
    newest = dates[-1] if dates else None

    ready = n_days >= MIN_DAYS_REQUIRED
    days_remaining = max(0, MIN_DAYS_REQUIRED - n_days)

    # Check which models are trained
    models_present = {
        "anomaly_detector": _ANOMALY_MODEL_PATH.exists(),
        "recovery_predictor": _RECOVERY_MODEL_PATH.exists(),
        "focus_clusters": _FOCUS_CLUSTER_PATH.exists(),
        "feature_scaler": _SCALER_PATH.exists(),
    }

    metadata = {}
    if _MODEL_METADATA_PATH.exists():
        try:
            metadata = json.loads(_MODEL_METADATA_PATH.read_text())
        except Exception:
            pass

    return {
        "days_of_data": n_days,
        "min_days_required": MIN_DAYS_REQUIRED,
        "ready_to_train": ready,
        "days_remaining_until_ready": days_remaining,
        "oldest_date": oldest,
        "newest_date": newest,
        "models_trained": models_present,
        "last_trained": metadata.get("trained_at"),
        "training_days_used": metadata.get("training_days"),
    }


def is_ready_to_train() -> bool:
    """Return True if enough data exists to train meaningful models."""
    return len(list_available_dates()) >= MIN_DAYS_REQUIRED


# ─── Model training ───────────────────────────────────────────────────────────

def _save_metadata(info: dict) -> None:
    """Persist training metadata for status reporting."""
    info["trained_at"] = datetime.now().isoformat()
    _MODEL_METADATA_PATH.write_text(json.dumps(info, indent=2))


def train_anomaly_detector(X: np.ndarray, scaler) -> object:
    """
    Train an Isolation Forest on the window feature matrix.

    Isolation Forest is unsupervised — it learns David's normal cognitive
    load patterns and flags windows that deviate significantly.

    contamination=0.05 means we expect ~5% of windows to be anomalous
    (unusual cognitive load, extreme meeting days, recovery crashes).
    """
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    X_scaled = scaler.transform(X)
    model.fit(X_scaled)
    return model


def train_recovery_predictor(dates: list[str]) -> Optional[object]:
    """
    Train a Random Forest regressor to predict next-day WHOOP recovery.

    Features: today's aggregate daily feature vector (mean + max + std of
    all working-hour window features).
    Target: tomorrow's recovery_score (0–100).

    Returns None if insufficient paired (today, tomorrow) days are available.
    Requires at least 30 day-pairs for a meaningful model.
    """
    from sklearn.ensemble import RandomForestRegressor

    pairs_X = []
    pairs_y = []

    for i, date_str in enumerate(dates[:-1]):
        tomorrow_str = dates[i + 1]

        # Only use consecutive days (no gaps)
        today_dt = datetime.strptime(date_str, "%Y-%m-%d")
        tomorrow_dt = datetime.strptime(tomorrow_str, "%Y-%m-%d")
        if (tomorrow_dt - today_dt).days != 1:
            continue

        # Extract today's daily features
        today_windows = read_day(date_str)
        today_feat = extract_daily_features(today_windows)
        if today_feat is None:
            continue

        # Get tomorrow's recovery score
        tomorrow_windows = read_day(tomorrow_str)
        if not tomorrow_windows:
            continue
        tomorrow_recovery = tomorrow_windows[0].get("whoop", {}).get("recovery_score")
        if tomorrow_recovery is None:
            continue

        pairs_X.append(today_feat)
        pairs_y.append(tomorrow_recovery)

    if len(pairs_X) < 30:
        print(
            f"[ml_model] Recovery predictor: only {len(pairs_X)} day-pairs available, "
            f"need 30. Skipping.",
            file=sys.stderr,
        )
        return None

    X = np.vstack(pairs_X)
    y = np.array(pairs_y)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    print(f"[ml_model] Recovery predictor trained on {len(pairs_X)} day-pairs")
    return model


def train_focus_clusters(X: np.ndarray, scaler, n_clusters: int = 6) -> Optional[object]:
    """
    Cluster active working-hour windows to find David's focus archetypes.

    Each cluster represents a distinct behavioral state (deep focus, heavy
    meeting load, Slack-heavy communication, recovery mode, etc.).

    After training, clusters with the highest mean FDI are labeled as
    "focus clusters" — the ideal conditions for deep work.

    n_clusters=6 gives: focus, light-social, heavy-meeting, high-load,
                         slack-heavy, recovery/idle.
    """
    from sklearn.cluster import KMeans

    if len(X) < MIN_WINDOWS_FOR_CLUSTER:
        print(
            f"[ml_model] Focus clustering: only {len(X)} active windows, "
            f"need {MIN_WINDOWS_FOR_CLUSTER}. Skipping.",
            file=sys.stderr,
        )
        return None

    X_scaled = scaler.transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        max_iter=500,
        random_state=42,
    )
    model.fit(X_scaled)
    print(f"[ml_model] Focus clusters trained: {n_clusters} clusters from {len(X)} windows")
    return model


def _build_scaler(X: np.ndarray):
    """Fit a StandardScaler on the training data."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def train_all(force: bool = False) -> dict:
    """
    Train all three models if sufficient data is available.

    Args:
        force: If True, train even if we have fewer than MIN_DAYS_REQUIRED days.
               Use for testing or when you want a preliminary model.

    Returns:
        dict with keys: 'anomaly', 'recovery', 'clustering', 'status', 'days_used'
        Each model key maps to 'trained' | 'skipped' | 'error'.
    """
    import joblib

    dates = sorted(list_available_dates())
    n_days = len(dates)

    if not force and n_days < MIN_DAYS_REQUIRED:
        print(
            f"[ml_model] Insufficient data: {n_days}/{MIN_DAYS_REQUIRED} days. "
            f"Use force=True to train anyway.",
            file=sys.stderr,
        )
        return {
            "status": "insufficient_data",
            "days_available": n_days,
            "days_required": MIN_DAYS_REQUIRED,
            "anomaly": "skipped",
            "recovery": "skipped",
            "clustering": "skipped",
        }

    print(f"[ml_model] Training models on {n_days} days of data...")

    # ── Build window-level feature matrix ────────────────────────────────
    print("[ml_model] Extracting features from all windows...")
    X_all, meta_all = build_feature_matrix(dates, working_hours_only=True, active_only=False)
    X_active, meta_active = build_feature_matrix(dates, working_hours_only=True, active_only=True)

    # Need at least 20 windows for a meaningful model (Isolation Forest minimum).
    # In practice, 60 days × 60 working-hour windows = 3600 windows gives robust results.
    # The force flag bypasses the day count check above but we still need some windows.
    if len(X_all) < 20:
        return {
            "status": "insufficient_windows",
            "windows_extracted": len(X_all),
            "anomaly": "skipped",
            "recovery": "skipped",
            "clustering": "skipped",
        }

    print(f"[ml_model] Extracted {len(X_all)} windows ({len(X_active)} active)")

    # ── Fit shared scaler on all working-hour windows ─────────────────────
    scaler = _build_scaler(X_all)
    joblib.dump(scaler, _SCALER_PATH)
    print(f"[ml_model] Scaler saved to {_SCALER_PATH}")

    result = {
        "status": "trained",
        "days_used": n_days,
        "windows_used": len(X_all),
        "active_windows_used": len(X_active),
    }

    # ── Anomaly detector (Isolation Forest) ──────────────────────────────
    try:
        anomaly_model = train_anomaly_detector(X_all, scaler)
        joblib.dump(anomaly_model, _ANOMALY_MODEL_PATH)
        print(f"[ml_model] Anomaly detector saved to {_ANOMALY_MODEL_PATH}")
        result["anomaly"] = "trained"
    except Exception as e:
        print(f"[ml_model] Anomaly detector training failed: {e}", file=sys.stderr)
        result["anomaly"] = f"error: {e}"

    # ── Recovery predictor (Random Forest) ───────────────────────────────
    try:
        recovery_model = train_recovery_predictor(dates)
        if recovery_model is not None:
            joblib.dump(recovery_model, _RECOVERY_MODEL_PATH)
            print(f"[ml_model] Recovery predictor saved to {_RECOVERY_MODEL_PATH}")
            result["recovery"] = "trained"
        else:
            result["recovery"] = "skipped (insufficient day-pairs)"
    except Exception as e:
        print(f"[ml_model] Recovery predictor training failed: {e}", file=sys.stderr)
        result["recovery"] = f"error: {e}"

    # ── Focus clusters (KMeans) ───────────────────────────────────────────
    try:
        cluster_model = train_focus_clusters(X_active, scaler)
        if cluster_model is not None:
            joblib.dump(cluster_model, _FOCUS_CLUSTER_PATH)
            print(f"[ml_model] Focus clusters saved to {_FOCUS_CLUSTER_PATH}")
            result["clustering"] = "trained"
        else:
            result["clustering"] = "skipped (insufficient active windows)"
    except Exception as e:
        print(f"[ml_model] Focus clustering failed: {e}", file=sys.stderr)
        result["clustering"] = f"error: {e}"

    # ── Save training metadata ────────────────────────────────────────────
    _save_metadata({
        "training_days": n_days,
        "training_dates": {"start": dates[0], "end": dates[-1]},
        "windows_used": len(X_all),
        "active_windows_used": len(X_active),
        "models": {
            k: result[k] for k in ["anomaly", "recovery", "clustering"]
        },
    })

    return result


# ─── Inference ────────────────────────────────────────────────────────────────

def detect_anomalies(windows: list[dict]) -> list[dict]:
    """
    Run the Isolation Forest anomaly detector on a day's windows.

    Returns a list of anomalous windows, each with:
    - window_id: the window identifier
    - hour_of_day: hour (for human-readable location)
    - anomaly_score: Isolation Forest score (lower = more anomalous)
    - features: dict of the key feature values

    Returns [] if the model is not trained yet or no windows are anomalous.
    """
    import joblib

    if not _ANOMALY_MODEL_PATH.exists() or not _SCALER_PATH.exists():
        return []

    try:
        model = joblib.load(_ANOMALY_MODEL_PATH)
        scaler = joblib.load(_SCALER_PATH)
    except Exception as e:
        print(f"[ml_model] Failed to load anomaly model: {e}", file=sys.stderr)
        return []

    anomalies = []
    for w in windows:
        if not w["metadata"].get("is_working_hours", False):
            continue
        v = extract_window_features(w)
        if v is None:
            continue

        v_scaled = scaler.transform(v.reshape(1, -1))
        # predict returns -1 for anomaly, 1 for normal
        pred = model.predict(v_scaled)[0]
        score = model.decision_function(v_scaled)[0]

        if pred == -1:  # anomalous
            anomalies.append({
                "window_id": w["window_id"],
                "hour_of_day": w["metadata"].get("hour_of_day", -1),
                "anomaly_score": round(float(score), 4),
                "features": {
                    "cls": w["metrics"]["cognitive_load_score"],
                    "fdi": w["metrics"]["focus_depth_index"],
                    "ras": w["metrics"]["recovery_alignment_score"],
                    "in_meeting": w["calendar"]["in_meeting"],
                    "slack_messages": w["slack"]["total_messages"],
                },
            })

    # Sort by most anomalous first
    anomalies.sort(key=lambda x: x["anomaly_score"])
    return anomalies


def predict_recovery(windows: list[dict]) -> Optional[dict]:
    """
    Predict tomorrow's WHOOP recovery score from today's windows.

    Returns:
        dict with 'predicted_recovery' (0–100), 'confidence' ('low'/'medium'/'high'),
        and 'basis' (description of the prediction basis).

    Returns None if the model is not trained or features can't be extracted.
    """
    import joblib

    if not _RECOVERY_MODEL_PATH.exists() or not _SCALER_PATH.exists():
        return None

    try:
        model = joblib.load(_RECOVERY_MODEL_PATH)
    except Exception as e:
        print(f"[ml_model] Failed to load recovery model: {e}", file=sys.stderr)
        return None

    daily_feat = extract_daily_features(windows)
    if daily_feat is None:
        return None

    try:
        predicted = float(model.predict(daily_feat.reshape(1, -1))[0])
        predicted = max(0.0, min(100.0, predicted))

        # Estimate confidence from the number of training samples
        # More trees agreeing = higher confidence (approximated by std of tree predictions)
        tree_preds = np.array([
            est.predict(daily_feat.reshape(1, -1))[0]
            for est in model.estimators_
        ])
        std = float(np.std(tree_preds))

        confidence = "high" if std < 5 else ("medium" if std < 12 else "low")

        return {
            "predicted_recovery": round(predicted, 1),
            "prediction_std": round(std, 1),
            "confidence": confidence,
            "basis": "ml_model",
        }
    except Exception as e:
        print(f"[ml_model] Recovery prediction failed: {e}", file=sys.stderr)
        return None


def get_focus_cluster_label(
    window: dict,
    cluster_model=None,
    scaler=None,
) -> Optional[dict]:
    """
    Classify a window into a focus cluster and return the cluster label.

    Returns None if the model is not trained or the window features can't
    be extracted.

    This is used to identify whether the current window is a "focus window"
    (high FDI cluster) or a disruptive state (high CLS, meeting load).
    """
    import joblib

    if cluster_model is None:
        if not _FOCUS_CLUSTER_PATH.exists():
            return None
        try:
            cluster_model = joblib.load(_FOCUS_CLUSTER_PATH)
        except Exception:
            return None

    if scaler is None:
        if not _SCALER_PATH.exists():
            return None
        try:
            scaler = joblib.load(_SCALER_PATH)
        except Exception:
            return None

    v = extract_window_features(window)
    if v is None:
        return None

    v_scaled = scaler.transform(v.reshape(1, -1))
    cluster_id = int(cluster_model.predict(v_scaled)[0])

    return {
        "cluster_id": cluster_id,
        "features": {
            "cls": window["metrics"]["cognitive_load_score"],
            "fdi": window["metrics"]["focus_depth_index"],
        },
    }


def get_focus_cluster_profiles(dates: list[str]) -> list[dict]:
    """
    After training, compute the profile of each focus cluster.

    Returns a list of cluster profiles sorted by mean FDI (best focus first):
    [
        {
            "cluster_id": 0,
            "mean_fdi": 0.85,
            "mean_cls": 0.12,
            "window_count": 142,
            "label": "deep focus",
            "peak_hours": [9, 10, 7],
        },
        ...
    ]

    Returns [] if the model is not trained.
    """
    import joblib
    from collections import defaultdict

    if not _FOCUS_CLUSTER_PATH.exists() or not _SCALER_PATH.exists():
        return []

    try:
        cluster_model = joblib.load(_FOCUS_CLUSTER_PATH)
        scaler = joblib.load(_SCALER_PATH)
    except Exception:
        return []

    X_active, meta_active = build_feature_matrix(dates, working_hours_only=True, active_only=True)
    if len(X_active) == 0:
        return []

    X_scaled = scaler.transform(X_active)
    labels = cluster_model.predict(X_scaled)

    # fdi index in FEATURE_NAMES
    fdi_idx = FEATURE_NAMES.index("fdi")
    cls_idx = FEATURE_NAMES.index("cls")
    hour_idx = FEATURE_NAMES.index("hour_of_day")

    cluster_fdi: dict[int, list[float]] = defaultdict(list)
    cluster_cls: dict[int, list[float]] = defaultdict(list)
    cluster_hours: dict[int, list[int]] = defaultdict(list)

    for i, label in enumerate(labels):
        cluster_fdi[int(label)].append(float(X_active[i, fdi_idx]))
        cluster_cls[int(label)].append(float(X_active[i, cls_idx]))
        cluster_hours[int(label)].append(int(X_active[i, hour_idx]))

    profiles = []
    for cid in sorted(cluster_fdi.keys()):
        fdi_vals = cluster_fdi[cid]
        cls_vals = cluster_cls[cid]
        hours = cluster_hours[cid]
        mean_fdi = sum(fdi_vals) / len(fdi_vals)
        mean_cls = sum(cls_vals) / len(cls_vals)

        # Top 3 hours for this cluster (by frequency)
        from collections import Counter
        top_hours = [h for h, _ in Counter(hours).most_common(3)]

        # Label the cluster based on its FDI/CLS profile
        if mean_fdi >= 0.75 and mean_cls < 0.25:
            label = "deep focus"
        elif mean_fdi >= 0.60 and mean_cls < 0.40:
            label = "focused work"
        elif mean_cls >= 0.55:
            label = "high cognitive load"
        elif mean_fdi < 0.45:
            label = "fragmented / meeting-heavy"
        else:
            label = "moderate engagement"

        profiles.append({
            "cluster_id": cid,
            "mean_fdi": round(mean_fdi, 3),
            "mean_cls": round(mean_cls, 3),
            "window_count": len(fdi_vals),
            "label": label,
            "peak_hours": top_hours,
        })

    # Sort by mean FDI descending (best focus clusters first)
    profiles.sort(key=lambda p: p["mean_fdi"], reverse=True)
    return profiles


# ─── Baseline statistics (used before ML is ready) ────────────────────────────

def compute_personal_baselines(dates: list[str]) -> dict:
    """
    Compute David's personal baseline statistics from available history.

    These are used for anomaly heuristics before the ML model is trained,
    and for explaining predictions after training.

    Returns:
        dict with keys: cls_baseline, fdi_baseline, ras_baseline,
                         hrv_baseline_ms, recovery_baseline,
                         std values for each, and data coverage info.
    """
    all_cls, all_fdi, all_ras, all_hrv, all_recovery = [], [], [], [], []

    for date_str in dates:
        windows = read_day(date_str)
        working = [w for w in windows if w["metadata"].get("is_working_hours", False)]

        for w in working:
            m = w.get("metrics", {})
            cls = m.get("cognitive_load_score")
            fdi = m.get("focus_depth_index")
            ras = m.get("recovery_alignment_score")
            hrv = w.get("whoop", {}).get("hrv_rmssd_milli")
            rec = w.get("whoop", {}).get("recovery_score")

            if cls is not None:
                all_cls.append(cls)
            if fdi is not None:
                all_fdi.append(fdi)
            if ras is not None:
                all_ras.append(ras)
            if hrv is not None:
                all_hrv.append(hrv)
            if rec is not None:
                all_recovery.append(rec)

    def _stats(vals: list) -> dict:
        if not vals:
            return {"mean": None, "std": None, "p25": None, "p75": None}
        arr = np.array(vals)
        return {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
        }

    return {
        "days_analyzed": len(dates),
        "working_windows": len(all_cls),
        "cls": _stats(all_cls),
        "fdi": _stats(all_fdi),
        "ras": _stats(all_ras),
        "hrv_ms": _stats(all_hrv),
        "recovery_pct": _stats(all_recovery),
    }


def heuristic_anomaly_check(windows: list[dict], baselines: dict) -> list[dict]:
    """
    Rule-based anomaly detection for use before the ML model is ready.

    Flags windows where a metric is more than 2 std deviations from baseline.
    Returns a list of anomaly dicts in the same format as detect_anomalies().

    This is the fallback path — once the Isolation Forest is trained it
    provides superior pattern-based detection.
    """
    anomalies = []

    cls_baseline = baselines.get("cls", {})
    cls_mean = cls_baseline.get("mean")
    cls_std = cls_baseline.get("std")

    if cls_mean is None or cls_std is None:
        return []

    threshold = cls_mean + 2.0 * max(cls_std, 0.05)

    for w in windows:
        if not w["metadata"].get("is_working_hours", False):
            continue
        cls = w["metrics"].get("cognitive_load_score", 0)
        if cls > threshold:
            anomalies.append({
                "window_id": w["window_id"],
                "hour_of_day": w["metadata"].get("hour_of_day", -1),
                "anomaly_score": round(-(cls - cls_mean) / max(cls_std, 0.05), 4),
                "features": {
                    "cls": cls,
                    "fdi": w["metrics"].get("focus_depth_index"),
                    "ras": w["metrics"].get("recovery_alignment_score"),
                    "in_meeting": w["calendar"].get("in_meeting", False),
                    "slack_messages": w["slack"].get("total_messages", 0),
                },
                "method": "heuristic",
            })

    anomalies.sort(key=lambda x: x["anomaly_score"])
    return anomalies


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Presence Tracker — ML Model Layer")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--force", action="store_true", help="Train even if < 60 days")
    parser.add_argument("--status", action="store_true", help="Show data and model status")
    parser.add_argument("--baselines", action="store_true", help="Compute personal baselines")
    parser.add_argument("--predict", metavar="DATE", help="Run anomaly detection on DATE")
    parser.add_argument("--clusters", action="store_true", help="Show focus cluster profiles")
    args = parser.parse_args()

    if args.status or not any([args.train, args.baselines, args.predict, args.clusters]):
        status = get_data_status()
        print(json.dumps(status, indent=2))

    if args.baselines:
        dates = sorted(list_available_dates())
        baselines = compute_personal_baselines(dates)
        print(json.dumps(baselines, indent=2))

    if args.train:
        result = train_all(force=args.force)
        print(json.dumps(result, indent=2))

    if args.predict:
        from engine.store import read_day
        windows = read_day(args.predict)
        if not windows:
            print(f"No data for {args.predict}")
            sys.exit(1)

        anomalies = detect_anomalies(windows)
        recovery_pred = predict_recovery(windows)

        print(f"\n=== Anomaly Detection — {args.predict} ===")
        if anomalies:
            print(f"Found {len(anomalies)} anomalous window(s):")
            for a in anomalies:
                print(f"  {a['window_id']} (hour {a['hour_of_day']}): "
                      f"score={a['anomaly_score']:.3f}, CLS={a['features']['cls']:.2f}")
        else:
            print("No anomalies detected (or model not trained yet).")

        print(f"\n=== Recovery Prediction for tomorrow ===")
        if recovery_pred:
            print(f"Predicted: {recovery_pred['predicted_recovery']:.0f}% "
                  f"(confidence: {recovery_pred['confidence']}, "
                  f"±{recovery_pred['prediction_std']:.1f}%)")
        else:
            print("Recovery model not trained yet.")

    if args.clusters:
        dates = sorted(list_available_dates())
        profiles = get_focus_cluster_profiles(dates)
        if profiles:
            print("\n=== Focus Cluster Profiles ===")
            for p in profiles:
                hours_str = ", ".join(f"{h:02d}:00" for h in p["peak_hours"])
                print(f"  Cluster {p['cluster_id']} — {p['label']}")
                print(f"    FDI: {p['mean_fdi']:.0%}  CLS: {p['mean_cls']:.0%}  "
                      f"Windows: {p['window_count']}  Peak hours: {hours_str}")
        else:
            print("Focus cluster model not trained yet.")

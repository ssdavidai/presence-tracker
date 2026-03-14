#!/usr/bin/env python3
"""
Presence Tracker — ML Model Trainer

Train, inspect, and run inference on the three scikit-learn models that
power the ML layer of the Presence Tracker.

Usage:
    python3 scripts/train_model.py                  # Status check (default)
    python3 scripts/train_model.py --train          # Train all models
    python3 scripts/train_model.py --train --force  # Train even if < 60 days
    python3 scripts/train_model.py --predict DATE   # Run inference on DATE
    python3 scripts/train_model.py --clusters       # Show focus cluster profiles
    python3 scripts/train_model.py --baselines      # Show personal baselines
    python3 scripts/train_model.py --json           # Machine-readable output

Models
------
  Isolation Forest — anomaly detection on working-hour windows
  Random Forest    — predicts tomorrow's WHOOP recovery from today's load
  KMeans           — focus cluster labelling (when does David achieve deep focus?)

The models activate automatically after 60 days of data.  Before that,
``--train`` requires ``--force`` and the models fall back to heuristic rules.

Exit codes
----------
  0 — success
  1 — training failed or insufficient data (without --force)
  2 — invalid arguments

Examples
--------
    # Check how close to ML readiness
    python3 scripts/train_model.py

    # Train after 60+ days of data have accumulated
    python3 scripts/train_model.py --train

    # Force train for testing with limited data
    python3 scripts/train_model.py --train --force

    # Run anomaly detection + recovery prediction on a specific day
    python3 scripts/train_model.py --predict 2026-03-14

    # Discover personal focus clusters
    python3 scripts/train_model.py --clusters
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.ml_model import (
    MIN_DAYS_REQUIRED,
    detect_anomalies,
    get_data_status,
    get_focus_cluster_profiles,
    compute_personal_baselines,
    predict_recovery,
    train_all,
)
from engine.store import list_available_dates, read_day


# ─── ANSI helpers ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"


def _c(text: str, code: str) -> str:
    """Apply colour only to real TTY output."""
    if sys.stdout.isatty():
        return f"{code}{text}{RESET}"
    return text


def _bar(filled: int, total: int, width: int = 20) -> str:
    """Render a progress bar."""
    n = round(filled / total * width) if total > 0 else 0
    n = max(0, min(n, width))
    pct = round(filled / total * 100) if total > 0 else 0
    bar = "▓" * n + "░" * (width - n)
    colour = GREEN if pct >= 100 else (YELLOW if pct >= 50 else RED)
    return _c(bar, colour) + f" {filled}/{total} ({pct}%)"


# ─── Status display ───────────────────────────────────────────────────────────

def print_status(status: dict) -> None:
    """Print a human-readable ML readiness status."""
    days = status["days_of_data"]
    required = status["min_days_required"]
    ready = status["ready_to_train"]
    models = status["models_trained"]

    print()
    print(_c("  ML Model Layer — Status", BOLD))
    print(_c("  " + "─" * 44, DIM))
    print()

    # Data readiness
    print(_c("  ① Data Readiness", BOLD))
    print(f"  Days collected   {_bar(days, required)}")
    if status.get("oldest_date") and status.get("newest_date"):
        print(f"  Date range       {status['oldest_date']} → {status['newest_date']}")
    if not ready:
        remaining = status["days_remaining_until_ready"]
        print(f"  {_c(f'  ✖ Need {remaining} more days before standard training', YELLOW)}")
        print(f"  {_c('  Use --train --force to train with current data anyway', DIM)}")
    else:
        print(f"  {_c('  ✔ Ready to train', GREEN)}")
    print()

    # Model status
    print(_c("  ② Trained Models", BOLD))
    model_labels = {
        "anomaly_detector": "Isolation Forest  (anomaly detection)",
        "recovery_predictor": "Random Forest     (recovery prediction)",
        "focus_clusters": "KMeans            (focus clustering)",
        "feature_scaler": "Feature Scaler    (preprocessing)",
    }
    for key, label in model_labels.items():
        present = models.get(key, False)
        icon = _c("✔", GREEN) if present else _c("✖", DIM)
        print(f"  {icon}  {label}")

    if status.get("last_trained"):
        print()
        print(f"  Last trained:    {status['last_trained']}")
    if status.get("training_days_used") is not None:
        print(f"  Training days:   {status['training_days_used']}")
    print()

    # Recommendation
    print(_c("  ③ Next Step", BOLD))
    all_trained = all(models.values())
    if all_trained and ready:
        print(f"  {_c('Models are current.  Re-train monthly or after significant data accumulation.', DIM)}")
        print(f"  {_c('  python3 scripts/train_model.py --train', CYAN)}")
    elif ready:
        print(f"  {_c('You have enough data.  Run --train to build the models:', DIM)}")
        print(f"  {_c('  python3 scripts/train_model.py --train', CYAN)}")
    else:
        print(f"  {_c('Keep collecting data.  Standard training unlocks at 60 days.', DIM)}")
        print(f"  {_c('  python3 scripts/train_model.py --train --force  # train early', CYAN)}")
    print()


# ─── Training display ─────────────────────────────────────────────────────────

def print_train_result(result: dict) -> None:
    """Print a human-readable training outcome."""
    status = result.get("status", "unknown")

    print()
    print(_c("  ML Model Training — Results", BOLD))
    print(_c("  " + "─" * 44, DIM))
    print()

    if status == "insufficient_data":
        days = result["days_available"]
        needed = result["days_required"]
        print(_c(f"  ✖ Not enough data: {days}/{needed} days", YELLOW))
        print()
        print(f"  {_c('Use --force to train with the available data:', DIM)}")
        print(f"  {_c('  python3 scripts/train_model.py --train --force', CYAN)}")
        print()
        return

    if status == "insufficient_windows":
        windows = result.get("windows_extracted", 0)
        print(_c(f"  ✖ Not enough windows extracted: {windows} (need ≥ 20)", YELLOW))
        print()
        return

    # Successful training
    days = result.get("days_used", "?")
    windows = result.get("windows_used", "?")
    active = result.get("active_windows_used", "?")

    print(f"  Training data    {days} days  ({windows} windows, {active} active)")
    print()

    model_rows = [
        ("anomaly",    "Isolation Forest  (anomaly detection)"),
        ("recovery",   "Random Forest     (recovery prediction)"),
        ("clustering", "KMeans            (focus clustering)"),
    ]
    for key, label in model_rows:
        outcome = result.get(key, "unknown")
        if outcome == "trained":
            icon = _c("✔", GREEN)
            outcome_str = _c("trained", GREEN)
        elif outcome and outcome.startswith("skipped"):
            icon = _c("–", YELLOW)
            outcome_str = _c(outcome, YELLOW)
        else:
            icon = _c("✖", RED)
            outcome_str = _c(outcome, RED)
        print(f"  {icon}  {label:<40}  {outcome_str}")

    print()
    all_ok = all(result.get(k) == "trained" for k in ["anomaly", "recovery", "clustering"])
    if all_ok:
        print(_c("  All models trained successfully.", GREEN))
        print(f"  {_c('Run inference with:', DIM)}")
        print(f"  {_c('  python3 scripts/train_model.py --predict DATE', CYAN)}")
    else:
        print(_c("  Some models failed or were skipped.  See above.", YELLOW))
    print()


# ─── Prediction display ───────────────────────────────────────────────────────

def print_prediction(date_str: str) -> None:
    """Load a day's data and run anomaly detection + recovery prediction."""
    windows = read_day(date_str)
    if not windows:
        print(f"No data for {date_str}.", file=sys.stderr)
        sys.exit(1)

    print()
    print(_c(f"  ML Inference — {date_str}", BOLD))
    print(_c("  " + "─" * 44, DIM))
    print()

    # Anomaly detection
    print(_c("  ① Anomaly Detection (Isolation Forest)", BOLD))
    anomalies = detect_anomalies(windows)
    if anomalies:
        print(f"  {_c(f'Found {len(anomalies)} anomalous window(s):', YELLOW)}")
        for a in sorted(anomalies, key=lambda x: x["anomaly_score"]):
            hour = a["hour_of_day"]
            score = a["anomaly_score"]
            cls = a["features"]["cls"]
            method = a.get("method", "model")
            label = _c(f"({method})", DIM)
            print(f"    {hour:02d}:00  score={score:.3f}  CLS={cls:.2f}  {label}")
    else:
        print(_c("  No anomalies detected.", GREEN))
    print()

    # Recovery prediction
    print(_c("  ② Recovery Prediction (tomorrow)", BOLD))
    recovery_pred = predict_recovery(windows)
    if recovery_pred:
        pred = recovery_pred["predicted_recovery"]
        conf = recovery_pred["confidence"]
        std = recovery_pred["prediction_std"]
        method = recovery_pred.get("method", "model")
        if pred >= 67:
            colour = GREEN
        elif pred >= 33:
            colour = YELLOW
        else:
            colour = RED
        print(f"  Predicted recovery  {_c(f'{pred:.0f}%', colour)}")
        print(f"  Confidence          {conf}")
        print(f"  Uncertainty         ±{std:.1f}%")
        print(f"  Method              {_c(method, DIM)}")
    else:
        print(_c("  Recovery prediction unavailable (model not trained or insufficient features).", DIM))
    print()


# ─── Cluster display ──────────────────────────────────────────────────────────

def print_clusters() -> None:
    """Show focus cluster profiles."""
    dates = sorted(list_available_dates())
    profiles = get_focus_cluster_profiles(dates)

    print()
    print(_c("  Focus Cluster Profiles (KMeans)", BOLD))
    print(_c("  " + "─" * 44, DIM))
    print()

    if not profiles:
        print(_c("  Focus cluster model not trained yet.", DIM))
        print(f"  {_c('Run: python3 scripts/train_model.py --train --force', CYAN)}")
        print()
        return

    for p in profiles:
        cid = p["cluster_id"]
        label = p["label"]
        fdi = p["mean_fdi"]
        cls = p["mean_cls"]
        count = p["window_count"]
        peak_hours = p.get("peak_hours", [])
        hours_str = ", ".join(f"{h:02d}:00" for h in peak_hours) if peak_hours else "N/A"

        fdi_colour = GREEN if fdi >= 0.7 else (YELLOW if fdi >= 0.4 else RED)
        print(f"  Cluster {cid}  {_c(label, BOLD)}")
        print(f"    FDI: {_c(f'{fdi:.0%}', fdi_colour)}  CLS: {cls:.0%}  "
              f"Windows: {count}  Peak hours: {hours_str}")
        print()


# ─── Baseline display ─────────────────────────────────────────────────────────

def print_baselines() -> None:
    """Compute and print personal baselines."""
    dates = sorted(list_available_dates())
    if not dates:
        print("No data available.", file=sys.stderr)
        sys.exit(1)

    baselines = compute_personal_baselines(dates)
    n_days = baselines.get("days_analyzed", len(dates))
    n_windows = baselines.get("working_windows", "?")

    print()
    print(_c(f"  Personal Baselines ({n_days} days, {n_windows} working-hour windows)", BOLD))
    print(_c("  " + "─" * 54, DIM))
    print()

    # Each metric key maps to {mean, std, p25, p75}
    metric_rows = [
        ("cls", "CLS", True),
        ("fdi", "FDI", False),
        ("sdi", "SDI", True),
        ("csc", "CSC", True),
        ("ras", "RAS", False),
    ]
    for key, label, high_is_bad in metric_rows:
        bucket = baselines.get(key)
        if not isinstance(bucket, dict):
            continue
        mean = bucket.get("mean")
        std = bucket.get("std")
        if mean is None:
            print(f"  {label:<6} N/A")
            continue
        if high_is_bad:
            colour = GREEN if mean < 0.35 else (YELLOW if mean < 0.65 else RED)
        else:
            colour = RED if mean < 0.35 else (YELLOW if mean < 0.65 else GREEN)
        std_str = f"  ±{std:.3f}" if std is not None else ""
        print(f"  {label:<6} {_c(f'{mean:.1%}', colour)}{std_str}  (personal mean)")

    # WHOOP baselines (stored as separate keys in the baselines dict)
    print()
    recovery_bucket = baselines.get("recovery_pct")
    hrv_bucket = baselines.get("hrv_ms")
    if isinstance(recovery_bucket, dict) and recovery_bucket.get("mean") is not None:
        rec = recovery_bucket["mean"]
        rec_std = recovery_bucket.get("std")
        rec_colour = GREEN if rec >= 67 else (YELLOW if rec >= 33 else RED)
        std_str = f"  ±{rec_std:.1f}%" if rec_std is not None else ""
        print(f"  Recovery  {_c(f'{rec:.0f}%', rec_colour)}{std_str}")
    if isinstance(hrv_bucket, dict) and hrv_bucket.get("mean") is not None:
        hrv = hrv_bucket["mean"]
        hrv_std = hrv_bucket.get("std")
        hrv_colour = GREEN if hrv >= 60 else (YELLOW if hrv >= 40 else RED)
        std_str = f"  ±{hrv_std:.1f} ms" if hrv_std is not None else ""
        print(f"  HRV       {_c(f'{hrv:.1f} ms', hrv_colour)}{std_str}")
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Presence Tracker — ML Model Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train all three models",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Train even if fewer than 60 days of data are available",
    )
    parser.add_argument(
        "--predict", metavar="DATE",
        help="Run anomaly detection + recovery prediction on DATE (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--clusters", action="store_true",
        help="Display focus cluster profiles",
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Display personal metric baselines",
    )
    parser.add_argument(
        "--json", dest="output_json", action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    args = parser.parse_args()

    # Validate --predict DATE
    if args.predict:
        try:
            datetime.strptime(args.predict, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date: {args.predict}. Use YYYY-MM-DD.", file=sys.stderr)
            sys.exit(2)

    # Default mode: status
    no_action = not any([args.train, args.predict, args.clusters, args.baselines])

    if no_action or args.output_json:
        status = get_data_status()
        if args.output_json:
            print(json.dumps(status, indent=2, default=str))
            if not any([args.train, args.predict, args.clusters, args.baselines]):
                return
        else:
            print_status(status)

    if args.train:
        result = train_all(force=args.force)
        if args.output_json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print_train_result(result)

        # Exit 1 when training was blocked by insufficient data
        status_val = result.get("status", "")
        if status_val in ("insufficient_data", "insufficient_windows"):
            sys.exit(1)

    if args.predict:
        if args.output_json:
            windows = read_day(args.predict)
            if not windows:
                print(json.dumps({"error": f"No data for {args.predict}"}))
                sys.exit(1)
            anomalies = detect_anomalies(windows)
            recovery_pred = predict_recovery(windows)
            print(json.dumps({
                "date": args.predict,
                "anomalies": anomalies,
                "recovery_prediction": recovery_pred,
            }, indent=2, default=str))
        else:
            print_prediction(args.predict)

    if args.clusters:
        if args.output_json:
            dates = sorted(list_available_dates())
            profiles = get_focus_cluster_profiles(dates)
            print(json.dumps(profiles, indent=2, default=str))
        else:
            print_clusters()

    if args.baselines:
        if args.output_json:
            dates = sorted(list_available_dates())
            baselines = compute_personal_baselines(dates)
            print(json.dumps(baselines, indent=2, default=str))
        else:
            print_baselines()


if __name__ == "__main__":
    main()

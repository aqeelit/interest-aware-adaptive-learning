"""Shared utilities: paths, reproducibility logging, seeding."""
import os
import json
import csv
from datetime import datetime, timezone

DATA_RAW = "/workspace/data/raw"
DATA_PROCESSED = "/workspace/data/processed"
RESULTS_TABLES = "/workspace/results/tables"
RESULTS_FIGURES = "/workspace/results/figures"
RESULTS_STATS = "/workspace/results/statistical_tests"

OULAD_RAW_DIR = os.path.join(DATA_RAW, "oulad")
ASSISTMENTS_RAW_DIR = os.path.join(DATA_RAW, "assistments")

EXPERIMENT_LOG_PATH = os.path.join(RESULTS_TABLES, "experiment_log.csv")

RANDOM_SEED = 42


def ensure_dirs():
    for d in [DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGURES, RESULTS_STATS]:
        os.makedirs(d, exist_ok=True)


def log_experiment(experiment_id: str, dataset_version: str, model: str,
                    parameters: dict, metrics: dict, notes: str = ""):
    """Append one reproducibility record, per CLAUDE.md's mandatory logging fields:
    experiment_id, dataset_version, random_seed, model, parameters, metrics, timestamp.
    Never overwrites prior rows — every run is preserved for audit.
    """
    ensure_dirs()
    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "random_seed": RANDOM_SEED,
        "model": model,
        "parameters": json.dumps(parameters, ensure_ascii=False),
        "metrics": json.dumps(metrics, ensure_ascii=False),
        "notes": notes,
    }
    file_exists = os.path.isfile(EXPERIMENT_LOG_PATH)
    with open(EXPERIMENT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[log_experiment] logged {experiment_id} -> {EXPERIMENT_LOG_PATH}")

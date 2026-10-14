"""
Runs the ml_ round (2026-10-14) stages in order and stops at the first failing stage.
Each stage writes its outputs when it finishes, then appends to experiment_log.csv.
With env ML_SMOKE=1 (see ml_common.py) everything runs on a tiny subset into /tmp/ml_smoke, no logging.
Pre-registration: results/statistical_tests/prereg_ml_2026-10-14.md
"""
import subprocess
import sys

STAGES = [("Step 1 within/between", "src/ml_1_within_between.py"),
          ("Step 2 leave-one-module-out", "src/ml_2_lomo.py"),
          ("Step 3 per-module replication", "src/ml_3_module_replication.py")]


def main():
    for label, script in STAGES:
        print(f"\n=== {label} ===", flush=True)
        r = subprocess.run([sys.executable, script])
        if r.returncode != 0:
            print(f"\nStage '{label}' FAILED (exit code {r.returncode}).", flush=True)
            sys.exit(r.returncode)
    print("\nAll ml_ stages completed.", flush=True)


if __name__ == "__main__":
    main()

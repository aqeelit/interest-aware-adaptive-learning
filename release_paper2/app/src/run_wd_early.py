"""
Runs the 2026-10-13 WD/EARLY round in order. The sanity check runs first and a
failure there (or anywhere) stops the run. Pre-registration:
results/statistical_tests/prereg_wd_early_2026-10-13.md
"""
import subprocess
import sys

STAGES = [
    ("Step 1 sanity: rebuilt full sample vs ci1", "src/wd_0_sanity_fullsample.py"),
    ("Step 2 WD stratified 5-fold", "src/wd_1_withdrawn_excluded.py"),
    ("Step 3 EARLY stratified 5-fold", "src/early_1_cutoff_stratcv.py"),
    ("Step 4 LOPO for WD and EARLY", "src/wd_early_2_lopo.py"),
]


def main():
    for label, script in STAGES:
        print(f"\n=== {label} ===", flush=True)
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\nStage '{label}' FAILED (exit code {result.returncode}).", flush=True)
            sys.exit(result.returncode)
    print("\nAll WD/EARLY stages completed.", flush=True)


if __name__ == "__main__":
    main()

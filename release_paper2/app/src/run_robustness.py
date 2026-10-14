"""
Runs the 2026-10-13 robustness round (scripts robustness_1..5) in order. Stops
at the first failing stage, mirroring run_pipeline.py.
"""
import subprocess
import sys

STAGES = [
    ("R1 second model family (GBM) + LR KEY-aligned", "src/robustness_1_gbm_second_family.py"),
    ("R2 course-level validation (leave-one-presentation-out)", "src/robustness_2_course_level_lopo.py"),
    ("R3 bootstrap CIs for D - E", "src/robustness_3_bootstrap_ci.py"),
    ("R4 dimensionality controls", "src/robustness_4_dimensionality_control.py"),
    ("R5 gender FNR gaps", "src/robustness_5_fairness_gender.py"),
]


def main():
    for label, script in STAGES:
        print(f"\n=== {label} ===", flush=True)
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\nStage '{label}' FAILED (exit code {result.returncode}).")
            sys.exit(result.returncode)
    print("\nAll robustness stages completed.")


if __name__ == "__main__":
    main()

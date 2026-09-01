"""
Runs the full Phase 9-11 pipeline in order. Each stage is independent and
traceable — if a stage fails (e.g., data not yet placed), it exits with a clear
message rather than silently continuing, per CLAUDE.md's rule against claiming
completion without producing the artifact.
"""
import subprocess
import sys

STAGES = [
    ("OULAD data cleaning", "src/data_cleaning_oulad.py"),
    ("OULAD feature engineering", "src/feature_engineering_oulad.py"),
    ("OULAD train & evaluate (Study 1)", "src/train_evaluate_oulad.py"),
    ("OULAD statistical comparison", "src/statistics_tests_oulad.py"),
    ("ASSISTments Rasch/1PL validation (Study 2)", "src/irt_rasch_assistments.py"),
]


def main():
    for label, script in STAGES:
        print(f"\n=== {label} ===")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\nStage '{label}' failed (exit code {result.returncode}). "
                  f"Fix the issue above before continuing to the next stage.")
            sys.exit(result.returncode)
    print("\nAll stages completed. Check results/statistical_tests/ and "
          "results/tables/experiment_log.csv for outputs before writing any "
          "numbers into the manuscript.")


if __name__ == "__main__":
    main()

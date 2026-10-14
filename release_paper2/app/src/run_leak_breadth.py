"""
Runs the 2026-10-13 leak-audit / breadth round in order and stops at the first
failing stage (leak_1_audit.py exits non-zero if the sanity rebuild does not match).
breadth_1_control.py itself enforces the per-family stop rule from leak_audit_verdict.csv.
Pre-registration: results/statistical_tests/prereg_leak_breadth_2026-10-13.md
"""
import subprocess
import sys

STAGES = [
    ("Step 1a-1d leak audit", "src/leak_1_audit.py"),
    ("Step 1e-1f diagnostics", "src/leak_2_diagnostics.py"),
    ("Step 2 breadth control", "src/breadth_1_control.py"),
]


def main():
    for label, script in STAGES:
        print(f"\n=== {label} ===", flush=True)
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\nStage '{label}' FAILED (exit code {result.returncode}).", flush=True)
            sys.exit(result.returncode)
    print("\nAll leak/breadth stages completed.", flush=True)


if __name__ == "__main__":
    main()

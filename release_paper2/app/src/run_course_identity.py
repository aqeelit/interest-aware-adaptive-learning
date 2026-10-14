"""
Runs the 2026-10-13 course-identity follow-up (ci_1..ci_5) in order. Stops at
the first failing stage, mirroring run_pipeline.py.
"""
import subprocess
import sys

STAGES = [
    ("CI-5 key-aligned re-derivation of manuscript pairwise statistics", "src/ci_5_keyaligned_rederivation.py"),
    ("CI-4 per-course D vs E under LOPO", "src/ci_4_per_course_lopo.py"),
    ("CI-1 course-identity controls (stratified 5-fold)", "src/ci_1_course_identity_stratcv.py"),
    ("CI-2 LOPO with course-centered interest", "src/ci_2_lopo_centered.py"),
    ("CI-3 student-disjoint validation", "src/ci_3_student_disjoint.py"),
]


def main():
    for label, script in STAGES:
        print(f"\n=== {label} ===", flush=True)
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\nStage '{label}' FAILED (exit code {result.returncode}).", flush=True)
            sys.exit(result.returncode)
    print("\nAll course-identity stages completed.", flush=True)


if __name__ == "__main__":
    main()

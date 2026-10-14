"""
Phase 9 — Data cleaning for OULAD.

Expects raw CSVs (as distributed by the Open University) at:
    data/raw/oulad/studentInfo.csv
    data/raw/oulad/studentVle.csv
    data/raw/oulad/vle.csv
    data/raw/oulad/studentAssessment.csv
    data/raw/oulad/assessments.csv
    data/raw/oulad/courses.csv
    data/raw/oulad/studentRegistration.csv

Produces:
    data/processed/oulad_clean.parquet   (merged, deduplicated, typed)

Design note: this script only cleans and merges — no feature engineering happens
here (that is feature_engineering.py's job), so each step stays independently
auditable per CLAUDE.md's traceability rule.
"""
import os
import sys
import pandas as pd
from utils import OULAD_RAW_DIR, DATA_PROCESSED, ensure_dirs

REQUIRED_FILES = [
    "studentInfo.csv",
    "studentVle.csv",
    "vle.csv",
    "studentAssessment.csv",
    "assessments.csv",
    "courses.csv",
    "studentRegistration.csv",
]


def check_files_present():
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(OULAD_RAW_DIR, f))]
    if missing:
        print("ERROR: missing required OULAD files in", OULAD_RAW_DIR)
        for m in missing:
            print("  -", m)
        print("Download OULAD from https://analyse.kmi.open.ac.uk/open_dataset "
              "and place the extracted CSVs in the path above.")
        sys.exit(1)


def load_raw():
    student_info = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentInfo.csv"))
    student_vle = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentVle.csv"))
    vle = pd.read_csv(os.path.join(OULAD_RAW_DIR, "vle.csv"))
    student_assessment = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentAssessment.csv"))
    assessments = pd.read_csv(os.path.join(OULAD_RAW_DIR, "assessments.csv"))
    courses = pd.read_csv(os.path.join(OULAD_RAW_DIR, "courses.csv"))
    registration = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentRegistration.csv"))
    return student_info, student_vle, vle, student_assessment, assessments, courses, registration


def clean_student_info(student_info: pd.DataFrame) -> pd.DataFrame:
    df = student_info.copy()
    df = df.drop_duplicates(subset=["id_student", "code_module", "code_presentation"])
    # Drop rows with no final_result (should not happen, but guard against it)
    df = df.dropna(subset=["final_result"])
    return df


def clean_student_vle(student_vle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    df = student_vle.merge(
        vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    )
    df = df.dropna(subset=["activity_type"])
    df = df[df["sum_click"] >= 0]
    return df


def clean_student_assessment(student_assessment: pd.DataFrame, assessments: pd.DataFrame) -> pd.DataFrame:
    df = student_assessment.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "assessment_type", "weight"]],
        on="id_assessment",
        how="left",
    )
    df = df.dropna(subset=["score"])
    df = df[(df["score"] >= 0) & (df["score"] <= 100)]
    return df


def main():
    ensure_dirs()
    check_files_present()
    student_info, student_vle, vle, student_assessment, assessments, courses, registration = load_raw()

    student_info_clean = clean_student_info(student_info)
    student_vle_clean = clean_student_vle(student_vle, vle)
    student_assessment_clean = clean_student_assessment(student_assessment, assessments)

    out_path_info = os.path.join(DATA_PROCESSED, "oulad_student_info_clean.parquet")
    out_path_vle = os.path.join(DATA_PROCESSED, "oulad_student_vle_clean.parquet")
    out_path_assess = os.path.join(DATA_PROCESSED, "oulad_student_assessment_clean.parquet")

    student_info_clean.to_parquet(out_path_info, index=False)
    student_vle_clean.to_parquet(out_path_vle, index=False)
    student_assessment_clean.to_parquet(out_path_assess, index=False)

    print(f"Wrote {len(student_info_clean)} rows -> {out_path_info}")
    print(f"Wrote {len(student_vle_clean)} rows -> {out_path_vle}")
    print(f"Wrote {len(student_assessment_clean)} rows -> {out_path_assess}")


if __name__ == "__main__":
    main()

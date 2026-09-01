"""
Phase 18 — Construct validity check (2026-09-02): is the "implicit interest"
signal actually capturing interest, or could it partly reflect deprivation/access
constraints instead?

This script computes, and ONLY computes, four statistical tests relating the
interest signal to two demographic variables available in raw OULAD data
(imd_band, disability). It does not interpret the results — no verdict is printed
or written, only the raw statistic/p-value/sample-size for each test, per explicit
instruction. Interpretation is left to the owner.

Inputs (read-only, not modified):
    data/raw/oulad/studentInfo.csv           (imd_band, disability columns)
    data/processed/oulad_features_D_interest.parquet
        (resource_diversity_entropy, click_share_* columns)

Output:
    results/statistical_tests/construct_validity_check.json

Tests:
    1. Spearman correlation: resource_diversity_entropy vs imd_band (ordinal,
       coded 1=most deprived [0-10%] .. 10=least deprived [90-100%]; rows with
       missing imd_band are dropped for this test only, count reported).
    2. Mann-Whitney U: resource_diversity_entropy, disability=Y vs disability=N.
    3. Spearman correlation: click_share_forumng vs imd_band (same coding).
    4. Mann-Whitney U: click_share_forumng, disability=Y vs disability=N.
"""
import os
import json
import pandas as pd
from scipy import stats as scipy_stats

from utils import OULAD_RAW_DIR, DATA_PROCESSED, RESULTS_STATS, ensure_dirs

KEY = ["id_student", "code_module", "code_presentation"]

# Ordinal coding of OULAD's imd_band, most-deprived (1) to least-deprived (10).
# Handles the raw data's "10-20" entries (missing a trailing '%', unlike every
# other band) by stripping '%' before matching.
IMD_BAND_ORDER = [
    "0-10", "10-20", "20-30", "30-40", "40-50",
    "50-60", "60-70", "70-80", "80-90", "90-100",
]


def normalize_imd_band(value):
    if pd.isna(value):
        return None
    stripped = str(value).strip().rstrip("%")
    if stripped in IMD_BAND_ORDER:
        return IMD_BAND_ORDER.index(stripped) + 1  # 1 = most deprived
    return None


def load_and_merge():
    student_info = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentInfo.csv"))
    interest = pd.read_parquet(
        os.path.join(DATA_PROCESSED, "oulad_features_D_interest.parquet")
    )

    needed_cols = {"click_share_forumng", "resource_diversity_entropy"}
    missing = needed_cols - set(interest.columns)
    if missing:
        raise ValueError(
            f"oulad_features_D_interest.parquet is missing expected column(s) "
            f"{missing}. Found columns: {interest.columns.tolist()}"
        )

    merged = student_info.merge(interest, on=KEY, how="inner")
    return student_info, interest, merged


def spearman_test(df: pd.DataFrame, value_col: str, imd_col: str = "imd_band_ordinal") -> dict:
    sub = df.dropna(subset=[value_col, imd_col])
    rho, p = scipy_stats.spearmanr(sub[value_col], sub[imd_col])
    return {
        "test": "Spearman rank correlation",
        "variable": value_col,
        "against": "imd_band (ordinal, 1=most deprived .. 10=least deprived)",
        "n": int(len(sub)),
        "rho": float(rho),
        "p_value": float(p),
    }


def mannwhitney_test(df: pd.DataFrame, value_col: str, group_col: str = "disability") -> dict:
    sub = df.dropna(subset=[value_col, group_col])
    group_y = sub.loc[sub[group_col] == "Y", value_col]
    group_n = sub.loc[sub[group_col] == "N", value_col]
    statistic, p = scipy_stats.mannwhitneyu(group_y, group_n, alternative="two-sided")
    return {
        "test": "Mann-Whitney U",
        "variable": value_col,
        "against": "disability (Y vs N)",
        "n_disability_Y": int(len(group_y)),
        "n_disability_N": int(len(group_n)),
        "median_disability_Y": float(group_y.median()),
        "median_disability_N": float(group_n.median()),
        "statistic": float(statistic),
        "p_value": float(p),
    }


def main():
    ensure_dirs()
    student_info, interest, merged = load_and_merge()

    merged["imd_band_ordinal"] = merged["imd_band"].apply(normalize_imd_band)
    n_missing_imd = merged["imd_band_ordinal"].isna().sum()

    results = {
        "merge_info": {
            "n_studentInfo_rows": int(len(student_info)),
            "n_oulad_features_D_interest_rows": int(len(interest)),
            "n_merged_rows": int(len(merged)),
            "n_merged_rows_missing_imd_band": int(n_missing_imd),
        },
        "entropy_vs_imd_band": spearman_test(merged, "resource_diversity_entropy"),
        "entropy_vs_disability": mannwhitney_test(merged, "resource_diversity_entropy"),
        "click_share_forumng_vs_imd_band": spearman_test(merged, "click_share_forumng"),
        "click_share_forumng_vs_disability": mannwhitney_test(merged, "click_share_forumng"),
    }

    out_path = os.path.join(RESULTS_STATS, "construct_validity_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""
Phase 9-11 — Study 2: Rasch (1PL) IRT validation on ASSISTments (approved 2026-08-30).

Expects a raw ASSISTments skill-builder CSV at:
    data/raw/assistments/skill_builder_data.csv
(2009 or 2017 skill-builder release — must contain at minimum: user_id, problem_id
or skill_id equivalent, and a binary 'correct' column.)

This study is intentionally kept separate from Study 1 (OULAD) — see
experiments/configs/experimental_design.md: its purpose is to validate the IRT
reasoning at true item-level granularity, not to feed data into the OULAD models.

Output:
    results/statistical_tests/assistments_person_ability.csv
    results/statistical_tests/assistments_item_difficulty.csv
    results/statistical_tests/assistments_rasch_fit.json
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from girth import rasch_mml, ability_eap, tag_missing_data, standard_errors_bootstrap, INVALID_RESPONSE
from utils import ASSISTMENTS_RAW_DIR, RESULTS_STATS, ensure_dirs, log_experiment

RAW_FILE = "skill_builder_data.csv"


RAW_COLUMN_MAP = {
    # The 2009 skill-builder release (as distributed) uses studentId/problemId
    # instead of the user_id/problem_id names used elsewhere in this module.
    # Renamed here (not in the raw file) so the raw CSV stays untouched and
    # this mapping stays auditable in version control.
    "studentId": "user_id",
    "problemId": "problem_id",
}


def load_raw() -> pd.DataFrame:
    path = os.path.join(ASSISTMENTS_RAW_DIR, RAW_FILE)
    if not os.path.isfile(path):
        print(f"ERROR: {path} not found.")
        print("Download an ASSISTments skill-builder release (2009 or 2017) and "
              f"place it at {path}.")
        sys.exit(1)
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df = df.rename(columns={k: v for k, v in RAW_COLUMN_MAP.items() if k in df.columns})
    return df


def build_response_matrix(df: pd.DataFrame, min_responses_per_user=5, min_responses_per_item=20) -> pd.DataFrame:
    """Builds a student x item binary correctness matrix, filtering out students
    and items with too few observations for stable Rasch estimation."""
    needed_cols = {"user_id", "problem_id", "correct"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns {needed_cols}, missing {missing}. "
                          f"Check the ASSISTments file version/schema.")

    df = df.dropna(subset=["user_id", "problem_id", "correct"])
    df["correct"] = df["correct"].astype(int)
    # keep first response per user-item pair (avoid double-attempts skewing the matrix)
    df = df.drop_duplicates(subset=["user_id", "problem_id"], keep="first")

    # Filtering users and items by independent one-shot counts, then applying
    # both filters together, can leave some items/users below the stated
    # threshold: removing a user drops that user's items' counts, and removing
    # an item drops that item's users' counts. Iterate the filter to a fixed
    # point so every retained user/item actually meets its threshold in the
    # final matrix (verified on the real ASSISTments file: a one-shot filter
    # left 167 items and 1 user below threshold, down to 3 responses).
    while True:
        user_counts = df.groupby("user_id").size()
        valid_users = user_counts[user_counts >= min_responses_per_user].index
        item_counts = df.groupby("problem_id").size()
        valid_items = item_counts[item_counts >= min_responses_per_item].index

        if len(valid_users) == df["user_id"].nunique() and len(valid_items) == df["problem_id"].nunique():
            break
        df = df[df["user_id"].isin(valid_users) & df["problem_id"].isin(valid_items)]

    matrix = df.pivot(index="problem_id", columns="user_id", values="correct")
    return matrix


def main():
    ensure_dirs()
    raw = load_raw()
    matrix = build_response_matrix(raw)

    print(f"Response matrix shape (items x persons): {matrix.shape}")

    # girth 0.8.0's API: items as rows, persons as columns, missing responses
    # tagged with the library's INVALID_RESPONSE sentinel (not NaN) via
    # tag_missing_data, and the array cast to int (rasch_mml indexes responses
    # as array indices internally, which requires an integer dtype).
    data_array = tag_missing_data(matrix.to_numpy(dtype=float), [0, 1]).astype(int)

    # Rasch / 1PL: single difficulty parameter per item, fixed discrimination.
    # With missing data present, girth 0.8.0's rasch_mml returns a dict with
    # 'Difficulty' and 'Discrimination' — there is no 'Ability' key (unlike
    # what this script originally assumed); ability must be estimated
    # separately from the fitted item parameters via ability_eap.
    estimates = rasch_mml(data_array)
    difficulty = estimates["Difficulty"]
    discrimination = np.full_like(difficulty, estimates["Discrimination"])
    ability = ability_eap(data_array, difficulty, discrimination)

    # Standard errors / fit diagnostics. girth 0.8.0 has no top-level
    # `standard_errors` function (the name this script originally imported
    # does not exist in this version) — the available equivalent is
    # standard_errors_bootstrap. Its default of 2000 bootstrap iterations is
    # impractically slow for this matrix size, so it is reduced here to 100
    # iterations; this reduction is intentional and disclosed in
    # manuscript/ai_disclosure_log.md and in the fit summary itself.
    N_BOOTSTRAP = 100
    try:
        se = standard_errors_bootstrap(
            data_array, rasch_mml, bootstrap_iterations=N_BOOTSTRAP,
            n_processors=1, solution=estimates, seed=42,
        )
        se_summary = {
            "difficulty_se_mean": float(np.nanmean(se["Standard Errors"]["Difficulty"])),
        }
    except Exception as e:
        se_summary = {"error": str(e)}

    item_df = pd.DataFrame({
        "problem_id": matrix.index,
        "difficulty": difficulty,
    })
    person_df = pd.DataFrame({
        "user_id": matrix.columns,
        "ability": ability,
    })

    item_out = os.path.join(RESULTS_STATS, "assistments_item_difficulty.csv")
    person_out = os.path.join(RESULTS_STATS, "assistments_person_ability.csv")
    item_df.to_csv(item_out, index=False)
    person_df.to_csv(person_out, index=False)

    fit_summary = {
        "n_items": int(matrix.shape[0]),
        "n_persons": int(matrix.shape[1]),
        "difficulty_mean": float(np.nanmean(difficulty)),
        "difficulty_sd": float(np.nanstd(difficulty)),
        "ability_mean": float(np.nanmean(ability)),
        "ability_sd": float(np.nanstd(ability)),
        "bootstrap_iterations": N_BOOTSTRAP,
        **se_summary,
    }
    fit_out = os.path.join(RESULTS_STATS, "assistments_rasch_fit.json")
    with open(fit_out, "w", encoding="utf-8") as f:
        json.dump(fit_summary, f, ensure_ascii=False, indent=2)

    log_experiment(
        experiment_id="EXP-ASSISTMENTS-RASCH-1",
        dataset_version="assistments-skill-builder-v1",
        model="Rasch-1PL-MML",
        parameters={"min_responses_per_user": 5, "min_responses_per_item": 20},
        metrics=fit_summary,
        notes="Study 2 IRT validation sub-study, kept separate from OULAD Study 1",
    )

    print(f"Wrote item difficulties -> {item_out}")
    print(f"Wrote person abilities -> {person_out}")
    print(f"Wrote fit summary -> {fit_out}")


if __name__ == "__main__":
    main()

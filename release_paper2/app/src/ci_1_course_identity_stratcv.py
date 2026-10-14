"""
CI-1 (2026-10-13) — course-identity controls, stratified 5-fold (seed 42), LR + GBM.

Row set: D/E rows (identical), N reported in outputs. Models:
  course_only        one-hot(code_module+code_presentation) only
  E_course           E + course one-hot
  D_course           D + course one-hot
  E_course_ctr       E + course one-hot + 21 interest features centered within course
  E_course_ctrshuf   as E_course_ctr, centered block row-permuted within course (rng 42)
Comparisons (a -> b, OR = b_only/a_only, bootstrap = b - a), BH across all 6
(3 comparisons x 2 families):
  E_course -> D_course ; E_course -> E_course_ctr ; E_course_ctrshuf -> E_course_ctr

Outputs: ci1_stratcv_fold_metrics.csv, ci1_stratcv_metrics_summary.csv,
         ci1_stratcv_comparisons_bh.csv, ci1_stratcv_oof_predictions.csv
"""
import numpy as np
import pandas as pd

from utils import ensure_dirs
from robustness_common import stratified_folds, run_cv, summarize, add_bh, log_summary
from ci_common import load_base, center_within_course, shuffle_within_course, compare, write_csv, CENTERING_NOTE

SCHEME = "stratified_5fold"
COMPARISONS = [("E_course", "D_course"), ("E_course", "E_course_ctr"), ("E_course_ctrshuf", "E_course_ctr")]


def main():
    ensure_dirs()
    base = load_base()
    y, course = base["y"], base["course"]
    centered = center_within_course(base["interest"], course)
    ctr_shuf, n_unchanged = shuffle_within_course(centered, course)
    print(f"Row info: {base['rowinfo']}; course one-hot columns: {len(base['onehot_cols'])}")
    print(f"Centered rows unchanged after within-course shuffle: {n_unchanged} / {len(y)}")

    variants = {
        "course_only": base["onehot"],
        "E_course": np.hstack([base["X_E"], base["onehot"]]),
        "D_course": np.hstack([base["X_D"], base["onehot"]]),
        "E_course_ctr": np.hstack([base["X_E"], base["onehot"], centered]),
        "E_course_ctrshuf": np.hstack([base["X_E"], base["onehot"], ctr_shuf]),
    }
    folds = stratified_folds(y)
    preds_l, folds_l = [], []
    for name, X in variants.items():
        for fam in ["LR", "GBM"]:
            p, f = run_cv(base["frame"], X, y, folds, fam, name, SCHEME)
            f["n_features"] = X.shape[1]
            preds_l.append(p)
            folds_l.append(f)
    preds = pd.concat(preds_l, ignore_index=True)
    fold_df = pd.concat(folds_l, ignore_index=True)
    summary = summarize(preds, fold_df)
    summary["n_features"] = summary["model"].map({k: v.shape[1] for k, v in variants.items()})
    for k, v in base["rowinfo"].items():
        summary[k] = v
    summary["centered_shuffle_rows_unchanged"] = n_unchanged

    comps = add_bh(pd.DataFrame([compare(preds, fam, SCHEME, a, b)
                                 for fam in ["LR", "GBM"] for a, b in COMPARISONS]))

    hdr = [CENTERING_NOTE, "Stratified 5-fold, seed 42; N and dropped-row counts in columns."]
    write_csv(fold_df, "ci1_stratcv_fold_metrics.csv", hdr)
    write_csv(summary, "ci1_stratcv_metrics_summary.csv", hdr)
    write_csv(comps, "ci1_stratcv_comparisons_bh.csv", hdr + ["OR = b_only_correct / a_only_correct; bootstrap differences are b - a; BH across all rows of this file."])
    write_csv(preds, "ci1_stratcv_oof_predictions.csv")
    log_summary(summary, "CI1", {"round": "course-identity controls"})

    print(summary.to_string(index=False))
    print(comps.to_string(index=False))


if __name__ == "__main__":
    main()

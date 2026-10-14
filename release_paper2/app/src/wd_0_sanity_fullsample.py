"""
WD/EARLY Step 1 (2026-10-13) — sanity check of the rebuilt feature path.

Rebuilds the full-sample setting (Withdrawn included, full-course features) with
wdearly_common.build_sample('FULL') and re-runs E_course and D_course (LR, GBM,
stratified 5-fold, seed 42). Predictions must equal ci1_stratcv_oof_predictions.csv.
Exits with code 1 (stopping run_wd_early.py) if they do not.

Output: wd_sanity_fullsample_check.csv
"""
import os
import sys
import numpy as np
import pandas as pd

from utils import ensure_dirs, RESULTS_STATS
from train_evaluate_oulad import KEY, load_target
from robustness_common import stratified_folds
from wdearly_common import build_sample, design, fit_all, write_csv

SCHEME = "stratified_5fold"


def main():
    ensure_dirs()
    s = build_sample("FULL")
    print(s["rowinfo"], flush=True)
    tgt = load_target()
    same_target = (len(tgt) == s["rowinfo"]["n_studentinfo_rows"])
    variants, _ = design(s, s["B"] + s["C"], "E", "D")
    variants = {k: variants[k] for k in ["E_course", "D_course"]}
    preds, _, _ = fit_all(s, variants, stratified_folds(s["frame"]["target"].to_numpy()), SCHEME, "FULL")

    old = pd.read_csv(os.path.join(RESULTS_STATS, "ci1_stratcv_oof_predictions.csv"), comment="#")
    rows, ok = [], True
    for fam in ["LR", "GBM"]:
        for mod in ["E_course", "D_course"]:
            a = preds[(preds["family"] == fam) & (preds["model"] == mod)]
            b = old[(old["family"] == fam) & (old["model"] == mod)]
            m = a.merge(b, on=KEY, suffixes=("_new", "_ci1"))
            same_order = len(a) == len(b) and (a[KEY].to_numpy() == b[KEY].to_numpy()).all()
            n_pred_same = int((m["y_pred_new"] == m["y_pred_ci1"]).sum())
            n_prob_exact = int((m["y_prob_new"] == m["y_prob_ci1"]).sum())
            maxdiff = float((m["y_prob_new"] - m["y_prob_ci1"]).abs().max())
            n_fold_same = int((m["fold_new"] == m["fold_ci1"]).sum())
            match = (len(a) == len(b) == len(m) and same_order and n_pred_same == len(m)
                     and n_fold_same == len(m) and maxdiff < 1e-9)
            ok &= match
            rows.append({"family": fam, "model": mod, "n_new": len(a), "n_ci1": len(b), "n_matched": len(m),
                         "row_order_identical": bool(same_order), "y_pred_identical": n_pred_same,
                         "fold_identical": n_fold_same, "y_prob_bitwise_identical": n_prob_exact,
                         "max_abs_prob_diff": maxdiff, "match": bool(match)})
    df = pd.DataFrame(rows)
    df["n_studentinfo_rows"] = s["rowinfo"]["n_studentinfo_rows"]
    df["n_rows_used"] = s["rowinfo"]["n_rows_used"]
    df["n_dropped_by_feature_inner_join"] = s["rowinfo"]["n_dropped_by_feature_inner_join"]
    df["target_row_count_equals_oulad_target_parquet"] = same_target
    write_csv(df, "wd_sanity_fullsample_check.csv",
              ["Match = same rows/order, same folds, identical y_pred, max |prob diff| < 1e-9 vs ci1_stratcv_oof_predictions.csv."])
    print(df.to_string(index=False), flush=True)
    if not ok:
        print("SANITY CHECK FAILED: rebuilt full-sample predictions do not match ci1 outputs.", flush=True)
        sys.exit(1)
    print("SANITY CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()

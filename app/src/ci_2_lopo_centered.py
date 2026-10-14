"""
CI-2 (2026-10-13) — leave-one-presentation-out (22 groups), LR + GBM, course-centered.

Course one-hot is unusable for a held-out course, so interest features are
centered within each presentation using only that presentation's own students
(see CENTERING_NOTE: unsupervised, uses held-out-course FEATURES, never labels).
Models (D/E row set):
  E, D                       (re-run; checked against robustness_oof_predictions_lopo.csv)
  E_shufint                  E + raw interest block row-permuted within course (rng 42,
                             identical procedure to robustness_4)
  E_ctr                      E + course-centered interest
  E_ctrshuf                  E + course-centered interest, row-permuted within course (rng 42)
Comparisons (a -> b), BH across all 8 (4 x 2 families):
  E -> D ; E -> E_ctr ; E_shufint -> D ; E_ctrshuf -> E_ctr

Outputs: ci2_lopo_per_course_metrics.csv, ci2_lopo_metrics_summary.csv,
         ci2_lopo_comparisons_bh.csv, ci2_lopo_oof_predictions.csv,
         ci2_lopo_rerun_check.csv
"""
import os
import numpy as np
import pandas as pd

from utils import ensure_dirs, RESULTS_STATS
from train_evaluate_oulad import KEY
from robustness_common import course_folds, run_cv, summarize, add_bh, log_summary
from ci_common import load_base, center_within_course, shuffle_within_course, compare, write_csv, CENTERING_NOTE

SCHEME = "leave_one_presentation_out"
COMPARISONS = [("E", "D"), ("E", "E_ctr"), ("E_shufint", "D"), ("E_ctrshuf", "E_ctr")]


def main():
    ensure_dirs()
    base = load_base()
    y, course = base["y"], base["course"]
    centered = center_within_course(base["interest"], course)
    raw_shuf, n_unch_raw = shuffle_within_course(base["interest"], course)
    ctr_shuf, n_unch_ctr = shuffle_within_course(centered, course)
    print(f"Row info: {base['rowinfo']}; unchanged after shuffle raw={n_unch_raw} centered={n_unch_ctr}")

    variants = {
        "E": base["X_E"], "D": base["X_D"],
        "E_shufint": np.hstack([base["X_E"], raw_shuf]),
        "E_ctr": np.hstack([base["X_E"], centered]),
        "E_ctrshuf": np.hstack([base["X_E"], ctr_shuf]),
    }
    folds = course_folds(base["frame"])
    preds_l, folds_l = [], []
    for name, X in variants.items():
        for fam in ["LR", "GBM"]:
            p, f = run_cv(base["frame"], X, y, folds, fam, name, SCHEME)
            preds_l.append(p)
            folds_l.append(f)
    preds = pd.concat(preds_l, ignore_index=True)
    fold_df = pd.concat(folds_l, ignore_index=True)
    summary = summarize(preds, fold_df)
    summary["n_features"] = summary["model"].map({k: v.shape[1] for k, v in variants.items()})
    for k, v in base["rowinfo"].items():
        summary[k] = v
    summary["n_groups"] = len(folds)

    # Re-run check of E and D against the previous robustness round's LOPO predictions.
    prev = pd.read_csv(os.path.join(RESULTS_STATS, "robustness_oof_predictions_lopo.csv"))
    checks = []
    for fam in ["LR", "GBM"]:
        for mod in ["E", "D"]:
            a = preds[(preds["family"] == fam) & (preds["model"] == mod)]
            b = prev[(prev["family"] == fam) & (prev["model"] == mod)]
            m = a.merge(b, on=KEY, suffixes=("_new", "_prev"))
            checks.append({"family": fam, "model": mod, "n_new": len(a), "n_prev": len(b), "n_matched": len(m),
                           "y_pred_identical": int((m["y_pred_new"] == m["y_pred_prev"]).sum()),
                           "max_abs_prob_diff": float((m["y_prob_new"] - m["y_prob_prev"]).abs().max())})
    check_df = pd.DataFrame(checks)

    comps = add_bh(pd.DataFrame([compare(preds, fam, SCHEME, a, b)
                                 for fam in ["LR", "GBM"] for a, b in COMPARISONS]))
    hdr = [CENTERING_NOTE, "LeaveOneGroupOut over code_module+code_presentation (22 groups)."]
    write_csv(fold_df, "ci2_lopo_per_course_metrics.csv", hdr)
    write_csv(summary, "ci2_lopo_metrics_summary.csv", hdr)
    write_csv(comps, "ci2_lopo_comparisons_bh.csv", hdr + ["OR = b_only_correct / a_only_correct; bootstrap differences are b - a; BH across all rows of this file."])
    write_csv(check_df, "ci2_lopo_rerun_check.csv")
    write_csv(preds, "ci2_lopo_oof_predictions.csv", hdr)
    log_summary(summary, "CI2", {"round": "course-identity LOPO centered"})

    print(check_df.to_string(index=False))
    print(summary.to_string(index=False))
    print(comps.to_string(index=False))


if __name__ == "__main__":
    main()

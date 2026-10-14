"""
Leak audit, Step 1e-1f (2026-10-13).

e. Single-feature diagnostics, EARLY t=28/56, stratified 5-fold (seed 42), LR and GBM, pooled
   out-of-fold Accuracy/F1/AUC: each of mean_score, weighted_mean_score, n_assessments, score_std,
   total_clicks alone, and with course one-hot as the only other feature; plus course one-hot alone
   as a reference row.
f. Risk-set students with no scored submission by t and with no clicks by t: counts and pass rate.
"""
import numpy as np
import pandas as pd

import wdearly_common as W
from utils import ensure_dirs
from robustness_common import stratified_folds, run_cv, summarize
from leakbreadth_common import build_early, course_info, write_csv, CUTOFFS, FAMILIES

FEATURES = ["mean_score", "weighted_mean_score", "n_assessments", "score_std", "total_clicks"]


def main():
    ensure_dirs()
    data = W.load_clean()
    summ_l, grp = [], []
    for t in CUTOFFS:
        s, _ = build_early(t, data)
        f = s["frame"]
        y = f["target"].to_numpy()
        _, oh = course_info(f)
        folds = stratified_folds(y)
        variants = {"course_onehot_only": oh}
        for c in FEATURES:
            x = f[[c]].to_numpy(dtype=float)
            variants[f"{c}_alone"] = x
            variants[f"{c}_plus_course_onehot"] = np.hstack([x, oh])
        preds_l, folds_l = [], []
        for name, X in variants.items():
            for fam in FAMILIES:
                p, fd = run_cv(f, X, y, folds, fam, name, "stratified_5fold")
                preds_l.append(p)
                folds_l.append(fd)
        sm = summarize(pd.concat(preds_l, ignore_index=True), pd.concat(folds_l, ignore_index=True))
        sm["feature"] = sm["model"].str.replace("_alone", "").str.replace("_plus_course_onehot", "")
        sm["n_missing_feature"] = [int(f[c].isna().sum()) if c in f else 0 for c in sm["feature"]]
        sm["setting"] = f"EARLY_t{t}"
        sm["cutoff_days"] = t
        summ_l.append(sm)

        nosub = f["n_assessments"] == 0
        noclk = f["total_clicks"] == 0
        for label, mask in [("no_scored_submission_by_t", nosub), ("no_clicks_by_t", noclk),
                            ("no_submission_and_no_clicks", nosub & noclk)]:
            grp.append({"setting": f"EARLY_t{t}", "cutoff_days": t, "group": label, "n_risk_set": len(f),
                        "n_in_group": int(mask.sum()), "pass_rate_in_group": float(f.loc[mask, "target"].mean()),
                        "n_pass_in_group": int(f.loc[mask, "target"].sum())})
    sdf, gdf = pd.concat(summ_l, ignore_index=True), pd.DataFrame(grp)
    write_csv(sdf, "leak_1e_single_feature_metrics.csv",
              ["Pooled out-of-fold metrics; stratified 5-fold seed 42; N = EARLY risk set (no rows dropped)."])
    write_csv(gdf, "leak_1f_nosubmission_noclick_groups.csv")
    print(sdf[["setting", "family", "model", "n", "accuracy_pooled", "auc_pooled", "n_missing_feature"]]
          .to_string(index=False), flush=True)
    print(gdf.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

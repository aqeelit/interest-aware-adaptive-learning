"""
Paper 2, Step 1 (2026-10-14; EXPLORATORY / POST HOC): course-normalised base.
Settings EARLY_t28/t56 x {VLEonly, withAssess} and FULL. Arms: base_course, base_course_z (base z-scored
within presentation, features only, no labels), base_course_z_full21 (+ 21 interest columns centered within
presentation). Comparisons base_course -> base_course_z, base_course_z -> base_course_z_full21; LR and GBM,
stratified 5-fold seed 42; key-aligned McNemar, OR (Wald CI), student-level bootstrap (1000, seed 42);
BH within each setting (4 tests). Sanity: base_course reproduces earlier predictions of the same construction.
"""
import os
import numpy as np
import pandas as pd

import wdearly_common as Wd
from utils import ensure_dirs, RESULTS_STATS
from robustness_common import stratified_folds, run_cv, summarize, add_bh
from ml_common import boot_compare, center_and_mean, onehot, repro_check
import p2_common as P

SETTINGS = ["EARLY_t28_VLEonly", "EARLY_t28_withAssess", "EARLY_t56_VLEonly", "EARLY_t56_withAssess", "FULL"]
SCHEME = "stratified_5fold"
COMPS = [("base_course", "base_course_z"), ("base_course_z", "base_course_z_full21")]


def main():
    ensure_dirs()
    data = Wd.load_clean()
    summ_l, comp_l, pred_l, fold_l, repro = [], [], [], [], []
    for setting in SETTINGS:
        s = P.load(setting, data)
        y, course = s["y"], s["course"]
        oh = onehot(course)
        ctr, _ = center_and_mean(s["I"], course)
        Xz = P.zscore_within(s["Xb"], course)
        arms = {"base_course": np.hstack([s["Xb"], oh]), "base_course_z": np.hstack([Xz, oh]),
                "base_course_z_full21": np.hstack([Xz, oh, ctr])}
        folds = stratified_folds(y)
        pl, fl = [], []
        for name, X in arms.items():
            for fam in ["LR", "GBM"]:
                p, fd = run_cv(s["frame"], X, y, folds, fam, name, SCHEME)
                pl.append(p)
                fl.append(fd)
        preds, fdf = pd.concat(pl, ignore_index=True), pd.concat(fl, ignore_index=True)
        sm = summarize(preds, fdf)
        sm["n_features"] = sm["model"].map({k: v.shape[1] for k, v in arms.items()})
        sm["base_columns"] = ";".join(s["base_cols"])
        for k, v in s["rowinfo"].items():
            sm[k] = v
        rows = []
        for fam in ["LR", "GBM"]:
            for a, b in COMPS:
                r = boot_compare(preds, fam, SCHEME, a, b)
                r["setting"] = setting
                rows.append(r)
        c = add_bh(pd.DataFrame(rows))
        for k in ["n_rows_used", "n_dropped_rows"]:
            c[k] = s["rowinfo"][k]
        if not P.SMOKE:
            if setting == "FULL":
                old = pd.read_csv(os.path.join(RESULTS_STATS, "ci1_stratcv_oof_predictions.csv"), comment="#")
                om = "E_course"
            else:
                t, fam_lab = setting.split("_")[1], setting.split("_")[2]
                old = pd.read_csv(os.path.join(RESULTS_STATS, "breadth_stratcv_oof_predictions.csv"), comment="#")
                old = old[(old["setting"] == f"EARLY_{t}") & (old["feature_family"] == fam_lab)]
                om = "base_course"
            for fam in ["LR", "GBM"]:
                repro.append(repro_check(preds, old, "base_course", om, fam, {"setting": setting}))
        for d in (preds, fdf, sm):
            d.insert(0, "setting", setting)
        summ_l.append(sm)
        comp_l.append(c)
        pred_l.append(preds)
        fold_l.append(fdf)
        print(f"done {setting}", flush=True)

    summ = pd.concat(summ_l, ignore_index=True)
    comps = pd.concat(comp_l, ignore_index=True)
    first = ["setting", "family", "model_a", "model_b", "n_common", "n_students", "n_dropped_rows", "acc_a", "acc_b",
             "delta_acc", "delta_acc_ci_low", "delta_acc_ci_high", "delta_acc_boot_se", "auc_a", "auc_b",
             "delta_auc", "delta_auc_ci_low", "delta_auc_ci_high", "a_only_correct", "b_only_correct", "p_value",
             "p_value_bh", "bh_family_size", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high"]
    comps = comps[first + [c for c in comps.columns if c not in first]]
    hdr = [P.NORM_NOTE, "Stratified 5-fold seed 42. Missing values stay missing after z-scoring (LR median "
           "imputation, GBM native). SD 0 within a presentation -> z = 0.",
           "Bootstrap: student-level (id_student), 1000 resamples, seed 42; differences b - a. "
           "OR = b_only/a_only (Wald CI). BH within each setting (4 tests)."]
    P.write_csv(summ, "p2_1_zbase_metrics_summary.csv", hdr[:2])
    P.write_csv(comps, "p2_1_zbase_comparisons_bh.csv", hdr)
    P.write_csv(pd.concat(fold_l, ignore_index=True), "p2_1_zbase_fold_metrics.csv", hdr[:2])
    P.write_csv(pd.concat(pred_l, ignore_index=True), "p2_1_zbase_oof_predictions.csv", hdr[:2])
    if repro:
        P.write_csv(pd.DataFrame(repro), "p2_1_zbase_reproduction_check.csv",
                    ["base_course vs earlier predictions (EARLY: breadth_stratcv base_course; FULL: ci1 E_course)."])
    for setting in SETTINGS:
        P.log(summ[summ["setting"] == setting], f"ZBASE-{setting}", {"setting": setting, "step": "p2_1"})
    pd.set_option("display.width", 250)
    print(comps[first[:22]].to_string(index=False))
    if repro:
        print(pd.DataFrame(repro).to_string(index=False))


if __name__ == "__main__":
    main()

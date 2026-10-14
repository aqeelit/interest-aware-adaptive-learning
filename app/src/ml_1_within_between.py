"""
ml_ round Step 1 (2026-10-14; EXPLORATORY / POST HOC): within/between decomposition.
FULL (base = E), EARLY_t28 / EARLY_t56 (VLE-only, base = C_early).
a. ICC(1) across 22 presentations and 7 modules: outcome + 21 interest columns.
b. Arms base, base_course, W (centered), M (course means), WB; LR and GBM; stratified 5-fold.
   Comparisons W vs base, M vs base, WB vs M; BH within each setting x model family (3 tests).
"""
import os
import numpy as np
import pandas as pd

import wdearly_common as Wd
from utils import ensure_dirs, RESULTS_STATS
from robustness_common import stratified_folds, run_cv, summarize, add_bh
import ml_common as C

SCHEME = "stratified_5fold"
COMPS = [("base", "W"), ("base", "M"), ("M", "WB")]


def main():
    ensure_dirs()
    data = Wd.load_clean()
    icc_rows, summ_l, comp_l, fold_l, pred_l, repro = [], [], [], [], [], []
    for setting in C.SETTINGS:
        s = C.load_setting(setting, data)
        y, course, module = s["y"], s["course"], s["module"]
        ri = s["rowinfo"]
        for grouping, g in [("presentation", course), ("module", module)]:
            for var, x in [("outcome_pass", y)] + [(c, s["I"][:, k]) for k, c in enumerate(s["icols"])]:
                icc_rows.append({"setting": setting, "grouping": grouping, "variable": var, **C.icc1(x, g)})
        ctr, mean = C.center_and_mean(s["I"], course)
        oh = C.onehot(course)
        Xb = s["Xb"]
        arms = {"base": Xb, "base_course": np.hstack([Xb, oh]), "W": np.hstack([Xb, ctr]),
                "M": np.hstack([Xb, mean]), "WB": np.hstack([Xb, ctr, mean])}
        folds = stratified_folds(y)
        pl, fl = [], []
        for name, X in arms.items():
            for fam in C.FAMILIES:
                p, fd = run_cv(s["frame"], X, y, folds, fam, name, SCHEME)
                pl.append(p)
                fl.append(fd)
        preds, fdf = pd.concat(pl, ignore_index=True), pd.concat(fl, ignore_index=True)
        sm = summarize(preds, fdf)
        sm["n_features"] = sm["model"].map({k: v.shape[1] for k, v in arms.items()})
        for d in (preds, fdf, sm):
            d.insert(0, "setting", setting)
        for k, v in ri.items():
            sm[k] = v
        for fam in C.FAMILIES:
            rows = []
            for a, b in COMPS:
                r = C.boot_compare(preds.drop(columns=["setting"]), fam, SCHEME, a, b)
                r["setting"] = setting
                rows.append(r)
            comp_l.append(add_bh(pd.DataFrame(rows)))
        summ_l.append(sm)
        fold_l.append(fdf)
        pred_l.append(preds)
        if not C.SMOKE:
            pp = preds.drop(columns=["setting"])
            if setting == "FULL":
                old = pd.read_csv(os.path.join(RESULTS_STATS, "robustness_oof_predictions_stratcv.csv"))
                old1 = pd.read_csv(os.path.join(RESULTS_STATS, "ci1_stratcv_oof_predictions.csv"), comment="#")
                checks = [("base", "E", old), ("base_course", "E_course", old1)]
            else:
                t = setting.split("_t")[1]
                old = pd.read_csv(os.path.join(RESULTS_STATS, "early_stratcv_oof_predictions.csv"), comment="#")
                old = old[old["setting"] == f"EARLY_t{t}_VLEonly"]
                checks = [("base_course", f"C_t{t}_course", old)]
            for nm, om, od in checks:
                for fam in C.FAMILIES:
                    repro.append(C.repro_check(pp, od, nm, om, fam, {"setting": setting}))
        print(f"done {setting}", flush=True)

    icc = pd.DataFrame(icc_rows)
    summ = pd.concat(summ_l, ignore_index=True)
    comps = pd.concat(comp_l, ignore_index=True)
    first = ["setting", "family", "model_a", "model_b", "n_common", "n_students", "acc_a", "acc_b", "delta_acc",
             "delta_acc_ci_low", "delta_acc_ci_high", "delta_acc_boot_se", "auc_a", "auc_b", "delta_auc",
             "delta_auc_ci_low", "delta_auc_ci_high", "a_only_correct", "b_only_correct", "p_value", "p_value_bh",
             "bh_family_size", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high"]
    comps = comps[first + [c for c in comps.columns if c not in first]]
    hdr = [C.CENTER_NOTE, "Stratified 5-fold seed 42. N and dropped rows in columns n_rows_used / n_dropped_rows.",
           "Arms: base (FULL = E; EARLY = C_early), base_course (+ presentation one-hot), W (+ centered 21), "
           "M (+ course-mean 21), WB (+ both)."]
    C.write_csv(icc, "ml_1a_icc.csv", ["Raw (uncentered) values; no centering used in this file.",
                                         "ICC(1) one-way ANOVA estimator; icc1 empty = undefined (denominator 0, "
                                         "constant column). n_per_group = group:N."])
    C.write_csv(summ, "ml_1b_metrics_summary.csv", hdr)
    C.write_csv(comps, "ml_1b_comparisons_bh.csv",
                hdr + [C.BOOT_NOTE, "OR = b_only_correct / a_only_correct. BH within each setting x model family (3)."])
    C.write_csv(pd.concat(fold_l, ignore_index=True), "ml_1b_fold_metrics.csv", hdr)
    C.write_csv(pd.concat(pred_l, ignore_index=True), "ml_1b_oof_predictions.csv", hdr)
    if repro:
        C.write_csv(pd.DataFrame(repro), "ml_1b_reproduction_check.csv",
                    ["Arms rebuilt identically to earlier predictions (robustness E, ci1 E_course, early C_t*_course)."])
    for setting in C.SETTINGS:
        C.log(summ[summ["setting"] == setting], f"STEP1-{setting}", {"setting": setting, "step": "1b"})
    pd.set_option("display.width", 250)
    print(comps[first[:17] + ["p_value_bh"]].to_string(index=False))
    if repro:
        print(pd.DataFrame(repro).to_string(index=False))


if __name__ == "__main__":
    main()

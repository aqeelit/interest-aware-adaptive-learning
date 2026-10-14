"""
ml_ round Step 2 (2026-10-14; EXPLORATORY / POST HOC): leave-one-module-out (7 folds).
For each held-out module, training excludes every row of a student (id_student) who also appears in
the held-out module. No course one-hot. base vs base + course-centered 21 interest columns.
FULL, EARLY_t28, EARLY_t56; LR and GBM. BH within each setting x model family (1 test).
"""
import numpy as np
import pandas as pd

import wdearly_common as Wd
from utils import ensure_dirs
from robustness_common import run_cv, summarize, add_bh
import ml_common as C

SCHEME = "leave_one_module_out"


def module_folds(frame):
    mod = frame["code_module"].to_numpy()
    ids = frame["id_student"].to_numpy()
    folds, info = [], []
    for m in sorted(set(mod)):
        te = np.flatnonzero(mod == m)
        other = mod != m
        overlap = other & np.isin(ids, ids[te])
        tr = np.flatnonzero(other & ~overlap)
        folds.append((tr, te, m))
        info.append({"held_out_module": m, "n_test_rows": len(te), "n_train_rows_other_modules": int(other.sum()),
                     "n_train_rows_removed_student_overlap": int(overlap.sum()),
                     "n_students_removed": int(len(set(ids[overlap]))), "n_train_rows_used": len(tr)})
    return folds, pd.DataFrame(info)


def main():
    ensure_dirs()
    data = Wd.load_clean()
    summ_l, comp_l, fold_l, pred_l, info_l, mod_l = [], [], [], [], [], []
    for setting in C.SETTINGS:
        s = C.load_setting(setting, data)
        y = s["y"]
        ctr, _ = C.center_and_mean(s["I"], s["course"])
        arms = {"base": s["Xb"], "base_ctr": np.hstack([s["Xb"], ctr])}
        folds, info = module_folds(s["frame"])
        info.insert(0, "setting", setting)
        pl, fl = [], []
        for name, X in arms.items():
            for fam in C.FAMILIES:
                p, fd = run_cv(s["frame"], X, y, folds, fam, name, SCHEME)
                pl.append(p)
                fl.append(fd)
        preds, fdf = pd.concat(pl, ignore_index=True), pd.concat(fl, ignore_index=True)
        sm = summarize(preds, fdf)
        for k, v in s["rowinfo"].items():
            sm[k] = v
        for fam in C.FAMILIES:
            r = C.boot_compare(preds, fam, SCHEME, "base", "base_ctr")
            r["setting"] = setting
            comp_l.append(add_bh(pd.DataFrame([r])))
            p = preds[preds["family"] == fam].copy()
            p["correct"] = (p["y_pred"] == p["y_true"]).astype(int)
            w = p.pivot_table(index=["id_student", "code_module", "code_presentation", "y_true"], columns="model",
                              values="correct").reset_index()
            for m, g in w.groupby("code_module"):
                mod_l.append({"setting": setting, "family": fam, "held_out_module": m, "n": len(g),
                              "pass_rate": g["y_true"].mean(), "acc_base": g["base"].mean(),
                              "acc_base_ctr": g["base_ctr"].mean(),
                              "diff_base_ctr_minus_base": g["base_ctr"].mean() - g["base"].mean()})
        for d in (preds, fdf, sm):
            d.insert(0, "setting", setting)
        summ_l.append(sm)
        fold_l.append(fdf)
        pred_l.append(preds)
        info_l.append(info)
        print(f"done {setting}", flush=True)

    summ = pd.concat(summ_l, ignore_index=True)
    comps = pd.concat(comp_l, ignore_index=True)
    first = ["setting", "family", "model_a", "model_b", "n_common", "n_students", "acc_a", "acc_b", "delta_acc",
             "delta_acc_ci_low", "delta_acc_ci_high", "delta_acc_boot_se", "auc_a", "auc_b", "delta_auc",
             "delta_auc_ci_low", "delta_auc_ci_high", "a_only_correct", "b_only_correct", "p_value", "p_value_bh",
             "bh_family_size", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high"]
    comps = comps[first + [c for c in comps.columns if c not in first]]
    hdr = [C.CENTER_NOTE, "Leave-one-module-out (7 folds); training rows of students who also appear in the "
           "held-out module are removed (see ml_2_lomo_fold_rows_removed.csv). No course one-hot."]
    C.write_csv(pd.concat(info_l, ignore_index=True), "ml_2_lomo_fold_rows_removed.csv", hdr[1:])
    C.write_csv(summ, "ml_2_lomo_metrics_summary.csv", hdr)
    C.write_csv(comps, "ml_2_lomo_comparisons_bh.csv",
                hdr + [C.BOOT_NOTE, "OR = b_only_correct / a_only_correct. BH within each setting x model family (1)."])
    C.write_csv(pd.DataFrame(mod_l), "ml_2_lomo_per_module.csv", hdr)
    C.write_csv(pd.concat(fold_l, ignore_index=True), "ml_2_lomo_fold_metrics.csv", hdr)
    C.write_csv(pd.concat(pred_l, ignore_index=True), "ml_2_lomo_oof_predictions.csv", hdr)
    for setting in C.SETTINGS:
        C.log(summ[summ["setting"] == setting], f"STEP2-LOMO-{setting}", {"setting": setting, "step": "2"})
    pd.set_option("display.width", 250)
    print(pd.concat(info_l).to_string(index=False))
    print(comps[first[:21]].to_string(index=False))
    print(pd.DataFrame(mod_l).to_string(index=False))


if __name__ == "__main__":
    main()

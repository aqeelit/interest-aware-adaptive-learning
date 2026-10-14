"""
ml_ round Step 3 (2026-10-14; EXPLORATORY / POST HOC): per-module replication.
Inside each of the 7 modules (all its presentations; presentation one-hot), stratified 5-fold seed 42 on
the module's rows: base_course -> base_course + course-centered 21 interest columns. EARLY_t28, EARLY_t56;
LR, GBM. BH across all 7 x 2 x 2 = 28 tests. Random-effects pooling of the 7 module delta-Accuracy values
per (cutoff x model family) with bootstrap SEs: DerSimonian-Laird and Hartung-Knapp (t, df = k - 1).
Replication labels follow prereg_ml_2026-10-14.md (rules copied verbatim there).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

import wdearly_common as Wd
from utils import ensure_dirs
from robustness_common import stratified_folds, run_cv, summarize, add_bh
import ml_common as C

SCHEME = "stratified_5fold_within_module"
SETTINGS = ["EARLY_t28", "EARLY_t56"]


def pool(y, se):
    y, se = np.asarray(y, float), np.asarray(se, float)
    k = len(y)
    out = {"k": k, "any_se_zero": bool((se == 0).any())}
    if out["any_se_zero"]:
        return {**out, "note": "pooling undefined: a module has bootstrap SE 0"}
    w = 1 / se ** 2
    mu_f = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - mu_f) ** 2))
    dfq = k - 1
    tau2 = max(0.0, (Q - dfq) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (se ** 2 + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    se_dl = float(np.sqrt(1 / np.sum(ws)))
    z = norm.ppf(0.975)
    q_hk = float(np.sum(ws * (y - mu) ** 2) / dfq)
    se_hk = float(np.sqrt(q_hk / np.sum(ws)))
    tc = float(tdist.ppf(0.975, dfq))
    return {**out, "pooled_dl": mu, "se_dl": se_dl, "ci_low_dl": mu - z * se_dl, "ci_high_dl": mu + z * se_dl,
            "Q": Q, "Q_df": dfq, "tau2": tau2, "I2_percent": float(max(0.0, (Q - dfq) / Q) * 100) if Q > 0 else 0.0,
            "se_hk": se_hk, "hk_t_crit_df": dfq, "hk_t_crit": tc, "ci_low_hk": mu - tc * se_hk,
            "ci_high_hk": mu + tc * se_hk, "note": ""}


def main():
    ensure_dirs()
    data = Wd.load_clean()
    rows, summ_l, pred_l, fold_l, nrows = [], [], [], [], []
    for setting in SETTINGS:
        s = C.load_setting(setting, data)
        ctr_all, _ = C.center_and_mean(s["I"], s["course"])
        for m in sorted(set(s["module"])):
            idx = np.flatnonzero(s["module"] == m)
            f = s["frame"].iloc[idx].reset_index(drop=True)
            y = s["y"][idx]
            oh = C.onehot(s["course"][idx])
            Xbc = np.hstack([s["Xb"][idx], oh])
            arms = {"base_course": Xbc, "base_course_ctr": np.hstack([Xbc, ctr_all[idx]])}
            folds = stratified_folds(y)
            pl, fl = [], []
            for name, X in arms.items():
                for fam in C.FAMILIES:
                    p, fd = run_cv(f, X, y, folds, fam, name, SCHEME)
                    pl.append(p)
                    fl.append(fd)
            preds, fdf = pd.concat(pl, ignore_index=True), pd.concat(fl, ignore_index=True)
            sm = summarize(preds, fdf)
            for d in (preds, fdf, sm):
                d.insert(0, "module", m)
                d.insert(0, "setting", setting)
            nrows.append({"setting": setting, "module": m, "n_rows": len(f), "n_presentations": oh.shape[1],
                          "pass_rate": float(y.mean()), "n_dropped_rows": 0})
            for fam in C.FAMILIES:
                r = C.boot_compare(preds.drop(columns=["setting", "module"]), fam, SCHEME, "base_course",
                                   "base_course_ctr")
                r.update({"setting": setting, "module": m})
                rows.append(r)
            summ_l.append(sm)
            pred_l.append(preds)
            fold_l.append(fdf)
            print(f"done {setting} {m}", flush=True)

    comps = add_bh(pd.DataFrame(rows))
    comps["ci_dir"] = (comps["delta_acc_ci_low"] > 0).astype(int) - (comps["delta_acc_ci_high"] < 0).astype(int)
    crit = {}
    for (st, m), g in comps.groupby(["setting", "module"]):
        g = g.set_index("family")
        crit[(st, m)] = bool(g.loc["LR", "ci_dir"] != 0 and g.loc["LR", "ci_dir"] == g.loc["GBM", "ci_dir"])
    comps["module_meets_criterion"] = [crit[(a, b)] for a, b in zip(comps["setting"], comps["module"])]

    pooled = []
    for (st, fam), g in comps.groupby(["setting", "family"], sort=False):
        g = g.sort_values("module")
        pooled.append({"setting": st, "family": fam, **pool(g["delta_acc"], g["delta_acc_boot_se"])})
    pooled = pd.DataFrame(pooled)
    comps = comps.merge(pooled[["setting", "family", "pooled_dl"]], on=["setting", "family"], how="left")
    comps["ci_contains_0"] = (comps["delta_acc_ci_low"] <= 0) & (comps["delta_acc_ci_high"] >= 0)
    comps["ci_contains_pooled_dl"] = ((comps["delta_acc_ci_low"] <= comps["pooled_dl"])
                                      & (comps["delta_acc_ci_high"] >= comps["pooled_dl"]))
    comps["label_inconclusive"] = comps["ci_contains_0"] & comps["ci_contains_pooled_dl"]

    lab = []
    for st, g in comps.groupby("setting", sort=False):
        n_meet = int(g.drop_duplicates("module")["module_meets_criterion"].sum())
        p = pooled[pooled["setting"] == st].set_index("family")
        i2 = {f: p.loc[f, "I2_percent"] if "I2_percent" in p.columns else np.nan for f in C.FAMILIES}
        if any(pd.isna(v) for v in i2.values()):
            label = "undefined (pooling undefined)"
        elif n_meet >= 5 and all(v < 50 for v in i2.values()):
            label = "consistent"
        elif n_meet < 5 and all(v >= 50 for v in i2.values()):
            label = "course-specific"
        else:
            label = "mixed"
        inc = g[g["label_inconclusive"]]
        lab.append({"setting": st, "n_modules": g["module"].nunique(), "n_modules_meeting_criterion": n_meet,
                    "modules_meeting_criterion": ";".join(sorted(g.loc[g["module_meets_criterion"], "module"].unique())),
                    "I2_percent_LR": i2["LR"], "I2_percent_GBM": i2["GBM"], "replication_label": label,
                    "inconclusive_LR": ";".join(inc.loc[inc["family"] == "LR", "module"]),
                    "inconclusive_GBM": ";".join(inc.loc[inc["family"] == "GBM", "module"])})
    lab = pd.DataFrame(lab)

    first = ["setting", "module", "family", "model_a", "model_b", "n_common", "n_students", "acc_a", "acc_b",
             "delta_acc", "delta_acc_ci_low", "delta_acc_ci_high", "delta_acc_boot_se", "delta_auc",
             "delta_auc_ci_low", "delta_auc_ci_high", "a_only_correct", "b_only_correct", "p_value", "p_value_bh",
             "bh_family_size", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high", "module_meets_criterion",
             "ci_contains_0", "ci_contains_pooled_dl", "label_inconclusive"]
    comps = comps[first + [c for c in comps.columns if c not in first]]
    hdr = [C.CENTER_NOTE, "Within each module: stratified 5-fold seed 42 on that module's rows; presentation one-hot.",
           C.BOOT_NOTE + " Bootstrap SE = SD (ddof 1) of the bootstrap delta-Accuracy values.",
           "BH across all 28 tests (7 modules x 2 cutoffs x 2 model families)."]
    summ = pd.concat(summ_l, ignore_index=True)
    C.write_csv(pd.DataFrame(nrows), "ml_3_module_n.csv", ["Rows per module = EARLY risk-set rows of that module."])
    C.write_csv(summ, "ml_3_module_metrics_summary.csv", hdr[:2])
    C.write_csv(comps, "ml_3_module_comparisons_bh.csv", hdr +
                ["module_meets_criterion: CI of delta-Accuracy excludes 0 in the same direction for LR and GBM.",
                 "label_inconclusive: CI contains 0 and the DL pooled estimate of that model family and cutoff."])
    C.write_csv(pooled, "ml_3_random_effects.csv",
                ["Random effects over the 7 module delta-Accuracy values with bootstrap SEs. DL: normal 95% CI.",
                 "Hartung-Knapp: t distribution, df = k - 1 = 6. I2 = max(0, (Q - df) / Q) x 100."])
    C.write_csv(lab, "ml_3_replication_labels.csv",
                ["Rule (prereg): >= 5 of 7 modules meet the criterion AND I2 < 50% for both model families = "
                 "consistent; < 5 modules AND I2 >= 50% (implemented: for both model families) = course-specific; "
                 "otherwise mixed."])
    C.write_csv(pd.concat(pred_l, ignore_index=True), "ml_3_module_oof_predictions.csv", hdr[:2])
    C.write_csv(pd.concat(fold_l, ignore_index=True), "ml_3_module_fold_metrics.csv", hdr[:2])
    for st in SETTINGS:
        C.log(summ[summ["setting"] == st].assign(model=lambda d: d["module"] + "-" + d["model"]),
              f"STEP3-MODULE-{st}", {"setting": st, "step": "3"})
    pd.set_option("display.width", 250)
    print(comps[first[:13] + ["p_value_bh", "module_meets_criterion", "label_inconclusive"]].to_string(index=False))
    print(pooled.to_string(index=False))
    print(lab.to_string(index=False))


if __name__ == "__main__":
    main()

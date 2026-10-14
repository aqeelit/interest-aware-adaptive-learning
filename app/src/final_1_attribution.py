"""
Final round, Step 1 (2026-10-14; EXPLORATORY / POST HOC): resource-type attribution of the
early individual-level signal. EARLY t28 and t56, VLE-only family, stratified 5-fold seed 42.

Units = the 21 interest columns (20 click_share_<type> + resource_diversity_entropy; see the
ASSUMPTION in prereg_final_2026-10-14.md). All unit columns are course-centered.
1a add-one-in   base_course -> base_course + unit_k
1b leave-one-out full21_ctr -> full21_ctr without unit_k
   BH across the 21 units within each (analysis x cutoff x model family).
1c LR standardized coefficient and permutation importance in AUC (LR and GBM) in full21_ctr,
   per held-out fold; mean and SD across the 5 folds.
1d Spearman between add-one-in dAUC, leave-one-out dAUC and permutation importance.
1e Descriptives per unit.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

import wdearly_common as W
from utils import ensure_dirs, RANDOM_SEED
from robustness_common import stratified_folds, add_bh, log_summary, FAMILIES as EST
from ci_common import center_within_course, compare
from leakbreadth_common import build_early, course_info, fit, CUTOFFS, FAMILIES
from final_common import write_csv, flag_units

SCHEME = "stratified_5fold"


def unit_kind(c):
    return "entropy" if c == "resource_diversity_entropy" else "activity_type"


def unit_name(c):
    return c.replace("click_share_", "")


def main():
    ensure_dirs()
    data = W.load_clean()
    comps_all, summ_all, stab_all, desc_all, rho_all, repro = [], [], [], [], [], []
    old = pd.read_csv("/workspace/results/statistical_tests/breadth_stratcv_oof_predictions.csv", comment="#")
    for t in CUTOFFS:
        s, _ = build_early(t, data)
        f = s["frame"]
        y = f["target"].to_numpy()
        course, oh = course_info(f)
        units = s["I"]
        assert len(units) == 21
        Xb = np.hstack([f[s["C"]].to_numpy(float), oh])
        I = center_within_course(f[units].to_numpy(float), course)
        setting = f"EARLY_t{t}"
        labels = {"setting": setting, "cutoff_days": t, "feature_family": "VLEonly", "scheme_label": SCHEME}
        arms = {"base_course": Xb, "full21_ctr": np.hstack([Xb, I])}
        for k, u in enumerate(units):
            arms[f"add_{unit_name(u)}"] = np.hstack([Xb, I[:, [k]]])
            arms[f"drop_{unit_name(u)}"] = np.hstack([Xb, np.delete(I, k, axis=1)])
        folds = stratified_folds(y)
        p, fd, sm = fit(f, arms, folds, SCHEME, labels)
        sm["n_rows_used"] = len(f)
        sm["n_excluded_unregistered_by_t"] = s["rowinfo"]["n_excluded_unregistered_by_t"]
        sm["n_studentinfo_rows"] = s["rowinfo"]["n_studentinfo_rows"]
        sm["n_dropped_rows"] = 0
        summ_all.append(sm)
        log_summary(sm[sm["model"].isin(["base_course", "full21_ctr"])], f"FINAL-ATTR-t{t}-VLEonly",
                    {"setting": setting, "feature_family": "VLEonly", "n_units": 21})

        # reproduction of the breadth-round arms (same construction)
        for m in ["base_course", "full21_ctr"]:
            for fam in FAMILIES:
                a = p[(p["family"] == fam) & (p["model"] == m)]
                b = old[(old["setting"] == setting) & (old["feature_family"] == "VLEonly")
                        & (old["family"] == fam) & (old["model"] == m)]
                mm = a.merge(b, on=["id_student", "code_module", "code_presentation"], suffixes=("_n", "_o"))
                repro.append({"setting": setting, "feature_family": "VLEonly", "family": fam, "model": m,
                              "n_new": len(a), "n_breadth_round": len(b), "n_matched": len(mm),
                              "y_pred_identical": int((mm["y_pred_n"] == mm["y_pred_o"]).sum()),
                              "max_abs_prob_diff": float((mm["y_prob_n"] - mm["y_prob_o"]).abs().max())})

        for analysis, prefix, a_of in [("add_one_in", "add_", lambda u: "base_course"),
                                       ("leave_one_out", "drop_", lambda u: "full21_ctr")]:
            for fam in FAMILIES:
                rows = []
                for u in units:
                    a, b = (a_of(u), f"{prefix}{unit_name(u)}") if analysis == "add_one_in" else \
                        ("full21_ctr", f"{prefix}{unit_name(u)}")
                    r = compare(p, fam, SCHEME, a, b)
                    r.update({"analysis": analysis, "unit": unit_name(u), "unit_column": u,
                              "unit_kind": unit_kind(u), "setting": setting, "cutoff_days": t,
                              "feature_family": "VLEonly"})
                    rows.append(r)
                comps_all.append(add_bh(pd.DataFrame(rows)))

        # 1c stability in full21_ctr
        Xf = arms["full21_ctr"]
        nb = Xb.shape[1]
        for fam in FAMILIES:
            coefs, imps = [], []
            for tr, te, lab in folds:
                est = EST[fam]()
                est.fit(Xf[tr], y[tr])
                if fam == "LR":
                    cf = est.named_steps["clf"].coef_[0]
                    assert len(cf) == Xf.shape[1]
                    coefs.append(cf[nb:])
                pi = permutation_importance(est, Xf[te], y[te], scoring="roc_auc", n_repeats=5,
                                            random_state=RANDOM_SEED)
                imps.append(pi.importances_mean[nb:])
            imps = np.array(imps)
            for k, u in enumerate(units):
                row = {"setting": setting, "cutoff_days": t, "feature_family": "VLEonly", "scheme": SCHEME,
                       "family": fam, "model": "full21_ctr", "unit": unit_name(u), "unit_kind": unit_kind(u),
                       "n_rows": len(f), "n_folds": len(folds),
                       "perm_importance_auc_mean": float(imps[:, k].mean()),
                       "perm_importance_auc_sd_across_folds": float(imps[:, k].std(ddof=1))}
                if fam == "LR":
                    cc = np.array(coefs)[:, k]
                    row.update({"lr_std_coef_mean": float(cc.mean()), "lr_std_coef_sd_across_folds": float(cc.std(ddof=1))})
                stab_all.append(row)

        # 1e descriptives (uncentered values)
        raw = f[units]
        for u in units:
            v = raw[u]
            used = v > 0
            per_course = used.groupby(course).any()
            desc_all.append({"setting": setting, "cutoff_days": t, "unit": unit_name(u), "unit_kind": unit_kind(u),
                             "n_risk_set": len(f), "n_presentations_total": int(len(per_course)),
                             "n_presentations_used": int(per_course.sum()),
                             "n_students_any_click": int(used.sum()),
                             "pct_students_any_click": float(100 * used.mean()),
                             "n_pass_distinction": int((y == 1).sum()), "n_fail_withdrawn": int((y == 0).sum()),
                             "mean_value_pass_distinction": float(v[y == 1].mean()),
                             "mean_value_fail_withdrawn": float(v[y == 0].mean())})
        print(f"done t={t}", flush=True)

    comps = flag_units(pd.concat(comps_all, ignore_index=True), ("analysis", "setting", "unit"))
    stab = pd.DataFrame(stab_all)

    # 1d Spearman
    for t in CUTOFFS:
        setting = f"EARLY_t{t}"
        for fam in FAMILIES:
            c = comps[(comps["setting"] == setting) & (comps["family"] == fam)]
            add = c[c["analysis"] == "add_one_in"].set_index("unit")["boot_auc_diff_b_minus_a"]
            loo = c[c["analysis"] == "leave_one_out"].set_index("unit")["boot_auc_diff_b_minus_a"]
            imp = stab[(stab["setting"] == setting) & (stab["family"] == fam)].set_index("unit")["perm_importance_auc_mean"]
            vals = {"add_one_in_dAUC": add, "leave_one_out_dAUC": loo, "perm_importance_auc": imp}
            for x, z in [("add_one_in_dAUC", "leave_one_out_dAUC"), ("add_one_in_dAUC", "perm_importance_auc"),
                         ("leave_one_out_dAUC", "perm_importance_auc")]:
                a, b = vals[x].sort_index(), vals[z].sort_index()
                assert (a.index == b.index).all()
                rho, pv = spearmanr(a.to_numpy(), b.to_numpy())
                rho_all.append({"setting": setting, "cutoff_days": t, "feature_family": "VLEonly", "family": fam,
                                "measure_x": x, "measure_y": z, "n_units": len(a), "spearman_rho": rho,
                                "spearman_p_unadjusted_descriptive": pv})

    note = ["Units: 20 click_share_<type> + resource_diversity_entropy (unit_kind=entropy); all course-centered.",
            "Stratified 5-fold seed 42; course-centering over the risk set (features only, no labels).",
            "N = EARLY risk set; no rows dropped. OR = b_only/a_only; bootstrap differences b - a (1000, seed 42).",
            "BH across the 21 units within each (analysis x cutoff x model family). flagged per prereg rule."]
    keep = ["analysis", "setting", "cutoff_days", "feature_family", "scheme", "unit", "unit_kind", "family",
            "model_a", "model_b", "n_common", "boot_n_students", "boot_acc_a", "boot_acc_b",
            "boot_acc_diff_b_minus_a", "boot_acc_diff_ci_low", "boot_acc_diff_ci_high", "boot_auc_a", "boot_auc_b",
            "boot_auc_diff_b_minus_a", "boot_auc_diff_ci_low", "boot_auc_diff_ci_high", "a_only_correct",
            "b_only_correct", "statistic", "p_value", "p_value_bh", "bh_family_size", "odds_ratio",
            "odds_ratio_ci_low", "odds_ratio_ci_high", "flagged"]
    keep = [k for k in keep if k in comps.columns] + [c for c in comps.columns if c not in keep]
    write_csv(comps[keep], "final_1ab_attribution_comparisons_bh.csv", note)
    write_csv(pd.concat(summ_all, ignore_index=True), "final_1_attribution_metrics_summary.csv", note[:3])
    write_csv(stab, "final_1c_stability.csv",
              ["Model full21_ctr. LR coef = coefficient after StandardScaler (standardized). Permutation importance:",
               "sklearn permutation_importance, scoring roc_auc, n_repeats 5, random_state 42, on each held-out fold;",
               "mean over repeats per fold, then mean / SD (ddof 1) across 5 folds."])
    write_csv(pd.DataFrame(rho_all), "final_1d_rank_agreement.csv",
              ["Spearman over 21 units; leave_one_out_dAUC is without - with; p-values unadjusted, descriptive."])
    write_csv(pd.DataFrame(desc_all), "final_1e_unit_descriptives.csv",
              ["Uncentered values over the risk set; students with no clicks contribute 0.",
               "Entropy row: 'used' = entropy > 0; means are of entropy, not of a share."])
    write_csv(pd.DataFrame(repro), "final_1_reproduction_check.csv",
              ["base_course and full21_ctr compared with breadth_stratcv_oof_predictions.csv (VLEonly)."])

    pd.set_option("display.width", 250)
    print(comps[["analysis", "setting", "unit", "family", "boot_acc_diff_b_minus_a", "boot_acc_diff_ci_low",
                 "boot_acc_diff_ci_high", "boot_auc_diff_b_minus_a", "p_value_bh", "flagged"]].to_string(index=False))
    print(pd.DataFrame(rho_all).to_string(index=False))
    print(pd.DataFrame(repro).to_string(index=False))


if __name__ == "__main__":
    main()

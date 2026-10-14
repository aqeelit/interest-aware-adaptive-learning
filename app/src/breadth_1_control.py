"""
Breadth control, Step 2 (2026-10-13). Runs ONLY for feature families whose leak audit verdict
(leak_audit_verdict.csv) is PASS — pre-registered stop rule.

Stratified 5-fold (seed 42) per cutoff x feature family, arms (course one-hot + base +):
  base_course, breadth2, breadth2_ctr, breadth2_ctrshuf, full21_ctr, full21_ctrshuf,
  breadth3_ctr (SECONDARY)
LOPO (22 presentations, no one-hot): base, base_breadth2_ctr, base_full21_ctr.
Comparisons, statistics, BH families and criterion as pre-registered.
full21 arms are also checked against the earlier EARLY predictions (same construction).

Outputs: breadth_stratcv_*, breadth_lopo_*, breadth_full21_reproduction_check.csv
"""
import os
import sys
import numpy as np
import pandas as pd

import wdearly_common as W
from utils import ensure_dirs, RESULTS_STATS
from train_evaluate_oulad import KEY
from robustness_common import stratified_folds, course_folds, add_bh, log_summary
from ci_common import center_within_course, shuffle_within_course, compare
from leakbreadth_common import (build_early, course_info, fit, criterion, write_csv, CUTOFFS,
                                BREADTH2, BREADTH3, STRAT_NOTE, LOPO_NOTE)

STRAT_COMPS = [("C1", "base_course", "breadth2_ctr", "standard"),
               ("C2", "breadth2_ctr", "full21_ctr", "PRIMARY"),
               ("C3", "breadth2_ctrshuf", "breadth2_ctr", "standard"),
               ("C4", "full21_ctrshuf", "full21_ctr", "standard"),
               ("C5", "base_course", "full21_ctr", "standard"),
               ("C6", "breadth2_ctr", "breadth3_ctr", "SECONDARY")]
LOPO_COMPS = [("L1", "base", "base_breadth2_ctr", "standard"),
              ("L2", "base_breadth2_ctr", "base_full21_ctr", "PRIMARY_LOPO"),
              ("L3", "base", "base_full21_ctr", "standard")]
OLD_NAMES = {"VLEonly": ("C_t{t}", "CI_t{t}"), "withAssess": ("E_t{t}", "D_t{t}")}


def run_comps(preds, scheme, comps, labels):
    rows = []
    for fam in ["LR", "GBM"]:
        for cid, a, b, role in comps:
            r = compare(preds, fam, scheme, a, b)
            r.update({"comparison_id": cid, "comparison_role": role, **labels})
            rows.append(r)
    return add_bh(pd.DataFrame(rows))


def main():
    ensure_dirs()
    verdict = pd.read_csv(os.path.join(RESULTS_STATS, "leak_audit_verdict.csv"), comment="#")
    passing = set(verdict.loc[verdict["audit"] == "PASS", "feature_family"])
    print("audit verdicts:", dict(zip(verdict["feature_family"], verdict["audit"])), flush=True)
    if not passing:
        print("No feature family passed the leak audit; Step 2 not run.", flush=True)
        sys.exit(0)
    old_s = pd.read_csv(os.path.join(RESULTS_STATS, "early_stratcv_oof_predictions.csv"), comment="#")
    old_l = pd.read_csv(os.path.join(RESULTS_STATS, "early_lopo_oof_predictions.csv"), comment="#")
    data = W.load_clean()
    out = {k: [] for k in ["sp", "sf", "ss", "sc", "lp", "lf", "ls", "lc", "pc", "pcs", "repro"]}

    for t in CUTOFFS:
        s, _ = build_early(t, data)
        f = s["frame"]
        course, oh = course_info(f)
        b2 = center_within_course(f[BREADTH2].to_numpy(float), course)
        b2s, _ = shuffle_within_course(b2, course)
        b3 = center_within_course(f[BREADTH3].to_numpy(float), course)
        I = center_within_course(f[s["I"]].to_numpy(float), course)
        Is, _ = shuffle_within_course(I, course)
        for fam_label, base_cols in [("VLEonly", s["C"]), ("withAssess", s["B"] + s["C"])]:
            if fam_label not in passing:
                print(f"SKIPPED {fam_label} t={t}: leak audit FAIL", flush=True)
                continue
            labels = {"setting": f"EARLY_t{t}", "cutoff_days": t, "feature_family": fam_label}
            Xb = f[base_cols].to_numpy(float)
            arms = {"base_course": np.hstack([Xb, oh]),
                    "breadth2": np.hstack([Xb, oh, f[BREADTH2].to_numpy(float)]),
                    "breadth2_ctr": np.hstack([Xb, oh, b2]),
                    "breadth2_ctrshuf": np.hstack([Xb, oh, b2s]),
                    "full21_ctr": np.hstack([Xb, oh, I]),
                    "full21_ctrshuf": np.hstack([Xb, oh, Is]),
                    "breadth3_ctr": np.hstack([Xb, oh, b3])}
            p, fd, sm = fit(f, arms, stratified_folds(f["target"].to_numpy()), "stratified_5fold",
                            {**labels, "scheme_label": "stratified_5fold"})
            sm["arm_role"] = np.where(sm["model"] == "breadth3_ctr", "SECONDARY", "standard")
            for k in ["n_rows_used", "n_excluded_unregistered_by_t", "n_studentinfo_rows"]:
                sm[k] = s["rowinfo"][k]
            c = run_comps(p, "stratified_5fold", STRAT_COMPS, {**labels, "scheme": "stratified_5fold"})
            out["sp"].append(p); out["sf"].append(fd); out["ss"].append(sm); out["sc"].append(c)
            log_summary(sm, f"BREADTH-RERUN-t{t}-{fam_label}", {"setting": f"EARLY_t{t}", "feature_family": fam_label})

            lopo = {"base": Xb, "base_breadth2_ctr": np.hstack([Xb, b2]), "base_full21_ctr": np.hstack([Xb, I])}
            lp, lfd, lsm = fit(f, lopo, course_folds(f), "leave_one_presentation_out",
                               {**labels, "scheme_label": "leave_one_presentation_out"})
            for k in ["n_rows_used", "n_excluded_unregistered_by_t", "n_studentinfo_rows"]:
                lsm[k] = s["rowinfo"][k]
            lc = run_comps(lp, "leave_one_presentation_out", LOPO_COMPS,
                           {**labels, "scheme": "leave_one_presentation_out"})
            out["lp"].append(lp); out["lf"].append(lfd); out["ls"].append(lsm); out["lc"].append(lc)
            log_summary(lsm, f"BREADTH-RERUN-LOPO-t{t}-{fam_label}", {"setting": f"EARLY_t{t}", "feature_family": fam_label})
            for base, plus in [("base", "base_breadth2_ctr"), ("base_breadth2_ctr", "base_full21_ctr")]:
                pc, pcs = W.per_course(lp, f"EARLY_t{t}", base, plus)
                for d in (pc, pcs):
                    d["feature_family"] = fam_label
                    d["comparison"] = f"{base}->{plus}"
                out["pc"].append(pc); out["pcs"].append(pcs)

            ob, _ = OLD_NAMES[fam_label]
            ob = ob.format(t=t)
            setting_old = f"EARLY_t{t}_{fam_label}"
            for new_name, old_name, newp, oldp in [("full21_ctr", f"{ob}_course_ctr", p, old_s),
                                                   ("full21_ctrshuf", f"{ob}_course_ctrshuf", p, old_s),
                                                   ("base_course", f"{ob}_course", p, old_s),
                                                   ("base_full21_ctr", f"{ob}_ctr", lp, old_l),
                                                   ("base", ob, lp, old_l)]:
                for fam in ["LR", "GBM"]:
                    a = newp[(newp["family"] == fam) & (newp["model"] == new_name)]
                    b = oldp[(oldp["setting"] == setting_old) & (oldp["family"] == fam) & (oldp["model"] == old_name)]
                    m = a.merge(b, on=KEY, suffixes=("_n", "_o"))
                    out["repro"].append({**labels, "family": fam, "new_model": new_name, "earlier_model": old_name,
                                         "n_new": len(a), "n_earlier": len(b), "n_matched": len(m),
                                         "y_pred_identical": int((m["y_pred_n"] == m["y_pred_o"]).sum()),
                                         "max_abs_prob_diff": float((m["y_prob_n"] - m["y_prob_o"]).abs().max())})
            print(f"done t={t} {fam_label}", flush=True)

    if not out["sc"]:
        print("Nothing to write.", flush=True)
        return
    # 'repro' holds dict rows; every other key holds DataFrames.
    cat = {k: (pd.DataFrame(v) if k == "repro" else pd.concat(v, ignore_index=True)) for k, v in out.items()}
    sc = criterion(cat["sc"])
    lc = criterion(cat["lc"])
    hdr_s = [STRAT_NOTE, "Arms include course one-hot + base features. breadth3_ctr / C6 are SECONDARY.",
             "OR = b_only_correct / a_only_correct; bootstrap differences b - a; BH within each cutoff x feature family (12 rows)."]
    hdr_l = [LOPO_NOTE, "OR = b_only_correct / a_only_correct; bootstrap differences b - a; "
             "BH within each cutoff x feature family (6 rows)."]
    write_csv(cat["ss"], "breadth_stratcv_metrics_summary.csv", hdr_s)
    write_csv(cat["sf"], "breadth_stratcv_fold_metrics.csv", hdr_s)
    write_csv(sc, "breadth_stratcv_comparisons_bh.csv", hdr_s)
    write_csv(cat["sp"], "breadth_stratcv_oof_predictions.csv", hdr_s)
    write_csv(cat["ls"], "breadth_lopo_metrics_summary.csv", hdr_l)
    write_csv(cat["lf"], "breadth_lopo_per_course_fold_metrics.csv", hdr_l)
    write_csv(lc, "breadth_lopo_comparisons_bh.csv", hdr_l)
    write_csv(cat["lp"], "breadth_lopo_oof_predictions.csv", hdr_l)
    write_csv(cat["pc"], "breadth_lopo_per_course.csv", hdr_l)
    write_csv(cat["pcs"], "breadth_lopo_per_course_summary.csv", hdr_l + ["Spearman over spearman_n presentations."])
    write_csv(cat["repro"], "breadth_full21_reproduction_check.csv",
              ["Arms built identically to earlier EARLY models, compared with early_*_oof_predictions.csv."])

    keep = ["setting", "feature_family", "scheme", "comparison_id", "comparison_role", "model_a", "model_b",
            "family", "n_common", "boot_n_students", "boot_acc_diff_b_minus_a", "boot_acc_diff_ci_low",
            "boot_acc_diff_ci_high", "boot_auc_diff_b_minus_a", "boot_auc_diff_ci_low", "boot_auc_diff_ci_high",
            "a_only_correct", "b_only_correct", "p_value", "p_value_bh", "odds_ratio", "odds_ratio_ci_low",
            "odds_ratio_ci_high", "criterion_met"]
    summary = pd.concat([sc[keep], lc[keep]], ignore_index=True).rename(columns={
        "family": "model_family", "n_common": "N", "boot_acc_diff_b_minus_a": "delta_acc",
        "boot_acc_diff_ci_low": "delta_acc_ci_low", "boot_acc_diff_ci_high": "delta_acc_ci_high",
        "boot_auc_diff_b_minus_a": "delta_auc", "boot_auc_diff_ci_low": "delta_auc_ci_low",
        "boot_auc_diff_ci_high": "delta_auc_ci_high", "p_value_bh": "bh_p", "odds_ratio": "OR",
        "odds_ratio_ci_low": "OR_ci_low", "odds_ratio_ci_high": "OR_ci_high"})
    summary.insert(4, "comparison", summary["model_a"] + "->" + summary["model_b"])
    write_csv(summary, "leak_breadth_SUMMARY.csv",
              ["One row per (setting, feature_family, scheme, comparison, model_family). criterion_met per "
               "prereg_leak_breadth_2026-10-13.md (same value on the LR and GBM rows of a comparison)."])
    print(summary.to_string(index=False), flush=True)
    print(cat["pcs"].to_string(index=False), flush=True)
    print(cat["repro"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

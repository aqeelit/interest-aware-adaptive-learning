"""
Final round, Step 2 (2026-10-14; EXPLORATORY / POST HOC): no-click sensitivity.
EARLY t28/t56 x {VLEonly, withAssess} x {LR, GBM}, stratified 5-fold seed 42.
Comparisons: breadth2_ctr -> full21_ctr, base_course -> full21_ctr (all arms include course one-hot).
  variant exclude_noclick : drop risk-set students with total_clicks = 0 by t; centering recomputed
                            within the reduced sample; folds re-drawn on the reduced sample.
  variant noclick_indicator: everyone kept; binary no-click indicator added to the base of every arm.
BH within each setting = cutoff x feature family x variant (4 rows).
"""
import numpy as np
import pandas as pd

import wdearly_common as W
from utils import ensure_dirs
from robustness_common import stratified_folds, add_bh, log_summary
from ci_common import center_within_course, compare
from leakbreadth_common import build_early, course_info, fit, criterion, CUTOFFS, FAMILIES, BREADTH2
from final_common import write_csv

SCHEME = "stratified_5fold"
COMPS = [("breadth2_ctr", "full21_ctr"), ("base_course", "full21_ctr")]


def main():
    ensure_dirs()
    data = W.load_clean()
    comps, summs, groups = [], [], []
    for t in CUTOFFS:
        s, _ = build_early(t, data)
        full = s["frame"]
        noclk = (full["total_clicks"] == 0).to_numpy()
        groups.append({"setting": f"EARLY_t{t}", "cutoff_days": t, "n_risk_set": len(full),
                       "n_noclick_excluded": int(noclk.sum()),
                       "pass_rate_noclick_group": float(full.loc[noclk, "target"].mean()),
                       "n_pass_noclick_group": int(full.loc[noclk, "target"].sum()),
                       "n_after_exclusion": int((~noclk).sum()),
                       "pass_rate_after_exclusion": float(full.loc[~noclk, "target"].mean())})
        for variant in ["exclude_noclick", "noclick_indicator"]:
            f = full[~noclk].reset_index(drop=True) if variant == "exclude_noclick" else full
            course, oh = course_info(f)
            b2 = center_within_course(f[BREADTH2].to_numpy(float), course)
            I = center_within_course(f[s["I"]].to_numpy(float), course)
            for fam_label, base_cols in [("VLEonly", s["C"]), ("withAssess", s["B"] + s["C"])]:
                setting = f"EARLY_t{t}_{fam_label}_{variant}"
                labels = {"setting": setting, "cutoff_days": t, "feature_family": fam_label, "variant": variant,
                          "scheme_label": SCHEME}
                Xb = f[base_cols].to_numpy(float)
                if variant == "noclick_indicator":
                    Xb = np.hstack([Xb, (f["total_clicks"] == 0).to_numpy(float)[:, None]])
                arms = {"base_course": np.hstack([Xb, oh]), "breadth2_ctr": np.hstack([Xb, oh, b2]),
                        "full21_ctr": np.hstack([Xb, oh, I])}
                p, fd, sm = fit(f, arms, stratified_folds(f["target"].to_numpy()), SCHEME, labels)
                sm["n_rows_used"] = len(f)
                sm["n_risk_set"] = len(full)
                sm["n_dropped_rows"] = len(full) - len(f)
                summs.append(sm)
                log_summary(sm, f"FINAL-NOCLICK-t{t}-{fam_label}-{variant}",
                            {"setting": setting, "feature_family": fam_label, "variant": variant})
                rows = []
                for fam in FAMILIES:
                    for a, b in COMPS:
                        r = compare(p, fam, SCHEME, a, b)
                        r.update({k: v for k, v in labels.items() if k != "scheme_label"})
                        r["n_dropped_rows"] = len(full) - len(f)
                        rows.append(r)
                comps.append(add_bh(pd.DataFrame(rows)))
                print(f"done {setting}", flush=True)
    c = criterion(pd.concat(comps, ignore_index=True))
    note = ["Arms include course one-hot. BH within each setting (cutoff x feature family x variant; 4 rows).",
            "OR = b_only/a_only; bootstrap differences b - a (student-level, 1000, seed 42).",
            "criterion_met = rule of prereg_leak_breadth_2026-10-13.md, reported descriptively."]
    write_csv(c, "final_2_noclick_comparisons_bh.csv", note)
    write_csv(pd.concat(summs, ignore_index=True), "final_2_noclick_metrics_summary.csv", note[:1])
    write_csv(pd.DataFrame(groups), "final_2_noclick_excluded_group.csv")
    pd.set_option("display.width", 250)
    print(c[["setting", "family", "model_a", "model_b", "n_common", "boot_acc_diff_b_minus_a", "boot_acc_diff_ci_low",
             "boot_acc_diff_ci_high", "boot_auc_diff_b_minus_a", "p_value_bh", "criterion_met"]].to_string(index=False))
    print(pd.DataFrame(groups).to_string(index=False))


if __name__ == "__main__":
    main()

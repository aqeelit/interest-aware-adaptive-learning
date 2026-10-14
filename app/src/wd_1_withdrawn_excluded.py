"""
WD/EARLY Step 2 (2026-10-13) — setting WD: Withdrawn excluded, stratified 5-fold.

Sample: final_result in {Pass, Distinction, Fail}; target Pass/Distinction = 1, Fail = 0.
Full-course features exactly as before (main and Bsens variants). Models per variant:
base, base+I, base_course, (base+I)_course, base_course_ctr, base_course_ctrshuf.
Pre-registered comparisons (prereg_wd_early_2026-10-13.md, P2.1-P2.6); BH across all 12 rows.

Outputs: wd_sample_description.csv, wd_course_passrates.csv, wd_stratcv_fold_metrics.csv,
wd_stratcv_metrics_summary.csv, wd_stratcv_comparisons_bh.csv, wd_stratcv_criterion_summary.csv,
wd_stratcv_oof_predictions.csv
"""
import pandas as pd

from utils import ensure_dirs
from robustness_common import stratified_folds, add_bh, log_summary
from wdearly_common import (load_clean, build_sample, course_table, design, fit_all, run_comparisons,
                            criterion, write_csv, STRAT_NOTE)

SCHEME = "stratified_5fold"


def main():
    ensure_dirs()
    data = load_clean()
    descs, courses, preds_l, folds_l, summ_l, comp_l = [], [], [], [], [], []
    for sens, base, plus, label in [(False, "E", "D", "WD"), (True, "Esens", "Dsens", "WD_sens")]:
        s = build_sample("WD", sens=sens, data=data)
        print(label, s["rowinfo"], flush=True)
        descs.append(s["rowinfo"])
        courses.append(course_table(s, label))
        variants, n_unch = design(s, s["B"] + s["C"], base, plus)
        preds, fold_df, summ = fit_all(s, variants, stratified_folds(s["frame"]["target"].to_numpy()), SCHEME, label)
        summ["ctrshuf_rows_unchanged"] = n_unch
        comps = [(f"{base}_course", f"{plus}_course"), (f"{base}_course", f"{base}_course_ctr"),
                 (f"{base}_course_ctrshuf", f"{base}_course_ctr")]
        comp_l.append(run_comparisons(preds, SCHEME, comps, label))
        preds_l.append(preds)
        folds_l.append(fold_df)
        summ_l.append(summ)
        log_summary(summ, f"WD-{label}", {"setting": label})

    comps_bh = add_bh(pd.concat(comp_l, ignore_index=True))
    detailed, crit = criterion(comps_bh)
    hdr = [STRAT_NOTE, "Withdrawn excluded; target Pass/Distinction=1, Fail=0. Stratified 5-fold, seed 42."]
    write_csv(pd.DataFrame(descs), "wd_sample_description.csv", hdr)
    write_csv(pd.concat(courses, ignore_index=True), "wd_course_passrates.csv", hdr)
    write_csv(pd.concat(folds_l, ignore_index=True), "wd_stratcv_fold_metrics.csv", hdr)
    write_csv(pd.concat(summ_l, ignore_index=True), "wd_stratcv_metrics_summary.csv", hdr)
    write_csv(detailed, "wd_stratcv_comparisons_bh.csv", hdr + [
        "OR = b_only_correct / a_only_correct; bootstrap differences b - a; BH across all 12 rows of this file."])
    write_csv(crit, "wd_stratcv_criterion_summary.csv",
              ["Pre-registered criterion (prereg_wd_early_2026-10-13.md)."])
    write_csv(pd.concat(preds_l, ignore_index=True), "wd_stratcv_oof_predictions.csv", hdr)
    print(crit.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

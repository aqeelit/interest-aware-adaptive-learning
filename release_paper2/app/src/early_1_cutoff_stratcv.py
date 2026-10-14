"""
WD/EARLY Step 3 (2026-10-13) — setting EARLY, cutoff-first truncation, stratified 5-fold.

Cutoffs t = 28, 56 days. studentVle rows with date <= t and studentAssessment rows
with date_submitted <= t are kept BEFORE any aggregation / share / entropy step.
Risk set: drop date_unregistration <= t. Target Pass/Distinction = 1, Fail/Withdrawn = 0.
Feature families: VLE-only (base C_t) and with assessments (base E_t = B_t + C_t).
Comparisons per cutoff x feature family; BH within each (6 rows).

Outputs: early_sample_description.csv, early_course_passrates.csv, early_stratcv_fold_metrics.csv,
early_stratcv_metrics_summary.csv, early_stratcv_comparisons_bh.csv,
early_stratcv_criterion_summary.csv, early_stratcv_oof_predictions.csv
"""
import pandas as pd

from utils import ensure_dirs
from robustness_common import stratified_folds, add_bh, log_summary
from wdearly_common import (load_clean, build_sample, course_table, design, fit_all, run_comparisons,
                            criterion, write_csv, STRAT_NOTE, ASSESS_NOTE, EARLY_FILL_NOTE)

SCHEME = "stratified_5fold"
CUTOFFS = [28, 56]


def main():
    ensure_dirs()
    data = load_clean()
    descs, courses, preds_l, folds_l, summ_l, comp_l = [], [], [], [], [], []
    for t in CUTOFFS:
        s = build_sample("EARLY", cutoff=t, data=data)
        print(t, s["rowinfo"], flush=True)
        descs.append(s["rowinfo"])
        courses.append(course_table(s, f"EARLY_t{t}"))
        folds = stratified_folds(s["frame"]["target"].to_numpy())
        for fam_label, base_cols, base, plus in [("VLEonly", s["C"], f"C_t{t}", f"CI_t{t}"),
                                                 ("withAssess", s["B"] + s["C"], f"E_t{t}", f"D_t{t}")]:
            label = f"EARLY_t{t}_{fam_label}"
            variants, n_unch = design(s, base_cols, base, plus)
            preds, fold_df, summ = fit_all(s, variants, folds, SCHEME, label)
            summ["ctrshuf_rows_unchanged"] = n_unch
            comps = [(f"{base}_course", f"{plus}_course"), (f"{base}_course", f"{base}_course_ctr"),
                     (f"{base}_course_ctrshuf", f"{base}_course_ctr")]
            comp_l.append(add_bh(run_comparisons(preds, SCHEME, comps, label)))
            preds_l.append(preds)
            folds_l.append(fold_df)
            summ_l.append(summ)
            log_summary(summ, label, {"setting": label, "cutoff_days": t})

    detailed, crit = criterion(pd.concat(comp_l, ignore_index=True))
    hdr = [STRAT_NOTE, ASSESS_NOTE + " (applies to withAssess settings)", EARLY_FILL_NOTE,
           "Truncation before aggregation: studentVle date <= t (negative days included), studentAssessment "
           "date_submitted <= t. Risk set excludes date_unregistration <= t. Target Pass/Distinction=1, Fail/Withdrawn=0."]
    write_csv(pd.DataFrame(descs), "early_sample_description.csv", hdr)
    write_csv(pd.concat(courses, ignore_index=True), "early_course_passrates.csv", hdr)
    write_csv(pd.concat(folds_l, ignore_index=True), "early_stratcv_fold_metrics.csv", hdr)
    write_csv(pd.concat(summ_l, ignore_index=True), "early_stratcv_metrics_summary.csv", hdr)
    write_csv(detailed, "early_stratcv_comparisons_bh.csv", hdr + [
        "OR = b_only_correct / a_only_correct; bootstrap differences b - a; BH within each setting (bh_family_size)."])
    write_csv(crit, "early_stratcv_criterion_summary.csv",
              ["Pre-registered criterion (prereg_wd_early_2026-10-13.md)."])
    write_csv(pd.concat(preds_l, ignore_index=True), "early_stratcv_oof_predictions.csv", hdr)
    print(crit.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

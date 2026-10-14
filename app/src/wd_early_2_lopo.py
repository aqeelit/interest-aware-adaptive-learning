"""
WD/EARLY Step 4 (2026-10-13) — leave-one-presentation-out for WD and EARLY.

Settings: WD (base E, full-course features); EARLY t=28 / t=56 x {VLEonly (base C_t),
withAssess (base E_t)}. Models: base, base_ctr, base_ctrshuf (no course one-hot).
Centering per presentation with that presentation's own students only, features only,
no labels. Comparisons base -> base_ctr and base_ctrshuf -> base_ctr; BH within each
setting (4 rows). Per-course breakdown base vs base_ctr with Spearman (n = 22).

Outputs: wd_lopo_* and early_lopo_* (fold metrics, summary, comparisons, criterion,
per-course, per-course summary, oof predictions).
"""
import pandas as pd

from utils import ensure_dirs
from robustness_common import course_folds, add_bh, log_summary
from wdearly_common import (load_clean, build_sample, lopo_design, fit_all, run_comparisons, criterion,
                            per_course, write_csv, LOPO_NOTE, ASSESS_NOTE, EARLY_FILL_NOTE)

SCHEME = "leave_one_presentation_out"


def run_setting(s, base_cols, base, label):
    variants, n_unch = lopo_design(s, base_cols, base)
    folds = course_folds(s["frame"])
    print(label, "groups:", len(folds), s["rowinfo"], flush=True)
    preds, fold_df, summ = fit_all(s, variants, folds, SCHEME, label)
    summ["ctrshuf_rows_unchanged"] = n_unch
    summ["n_groups"] = len(folds)
    comps = add_bh(run_comparisons(preds, SCHEME, [(base, f"{base}_ctr"), (f"{base}_ctrshuf", f"{base}_ctr")], label))
    pc, pcs = per_course(preds, label, base, f"{base}_ctr")
    log_summary(summ, f"LOPO-{label}", {"setting": label})
    return preds, fold_df, summ, comps, pc, pcs


def emit(prefix, results, hdr):
    preds, folds, summ, comps, pc, pcs = (pd.concat(x, ignore_index=True) for x in zip(*results))
    detailed, crit = criterion(comps)
    write_csv(folds, f"{prefix}_lopo_per_course_fold_metrics.csv", hdr)
    write_csv(summ, f"{prefix}_lopo_metrics_summary.csv", hdr)
    write_csv(detailed, f"{prefix}_lopo_comparisons_bh.csv", hdr + [
        "OR = b_only_correct / a_only_correct; bootstrap differences b - a; BH within each setting (bh_family_size)."])
    write_csv(crit, f"{prefix}_lopo_criterion_summary.csv", ["Pre-registered criterion (prereg_wd_early_2026-10-13.md)."])
    write_csv(pc, f"{prefix}_lopo_per_course_base_vs_ctr.csv", hdr)
    write_csv(pcs, f"{prefix}_lopo_per_course_summary.csv", hdr + ["Spearman over spearman_n course presentations."])
    write_csv(preds, f"{prefix}_lopo_oof_predictions.csv", hdr)
    print(crit.to_string(index=False), flush=True)
    print(pcs.to_string(index=False), flush=True)


def main():
    ensure_dirs()
    data = load_clean()
    s = build_sample("WD", data=data)
    emit("wd", [run_setting(s, s["B"] + s["C"], "E", "WD")],
         [LOPO_NOTE, "Withdrawn excluded; target Pass/Distinction=1, Fail=0."])

    res = []
    for t in [28, 56]:
        s = build_sample("EARLY", cutoff=t, data=data)
        res.append(run_setting(s, s["C"], f"C_t{t}", f"EARLY_t{t}_VLEonly"))
        res.append(run_setting(s, s["B"] + s["C"], f"E_t{t}", f"EARLY_t{t}_withAssess"))
    emit("early", res, [LOPO_NOTE, ASSESS_NOTE + " (applies to withAssess settings)", EARLY_FILL_NOTE,
                        "Truncation before aggregation at t; risk set excludes date_unregistration <= t; "
                        "target Pass/Distinction=1, Fail/Withdrawn=0."])


if __name__ == "__main__":
    main()

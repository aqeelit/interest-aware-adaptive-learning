"""
Robustness 2 (2026-10-13) — course-level validation.

Leave-one-presentation-out (sklearn LeaveOneGroupOut), groups =
code_module + code_presentation (22 groups in the matched D/E data). Every
held-out group is a course presentation never seen in training. Run for LR and
HistGradientBoostingClassifier on B, Bsens, C, D, E, Dsens, Esens.

Note: groups are course presentations, as specified; the same id_student can be
enrolled in more than one presentation, so a student may appear in both train
and test across groups. The count of such students is printed and saved.

Outputs (all new files):
  robustness_oof_predictions_lopo.csv
  robustness_lopo_per_course_metrics.csv
  robustness_lopo_metrics_summary.csv
  robustness_lopo_pairwise_mcnemar_bh.csv   (BH within each family, 21 pairs)
  robustness_lopo_student_overlap.csv
"""
import pandas as pd

from utils import ensure_dirs
from robustness_common import (MODELS, load_model_frame, course_folds, run_cv, summarize,
                               mcnemar_pair, add_bh, all_pairs, log_summary, out, guard_new)


def main():
    ensure_dirs()
    all_preds, all_folds, overlap = [], [], []
    for model in MODELS:
        merged, cols = load_model_frame(model)
        X = merged[cols].to_numpy(dtype=float)
        y = merged["target"].to_numpy()
        folds = course_folds(merged)
        n_groups_per_student = merged.groupby("id_student")[["code_module", "code_presentation"]].apply(
            lambda g: len(g.drop_duplicates()))
        overlap.append({"model": model, "n_rows": int(len(merged)), "n_groups": len(folds),
                        "n_students": int(n_groups_per_student.size),
                        "n_students_in_more_than_one_group": int((n_groups_per_student > 1).sum())})
        print(f"[{model}] n={len(y)} groups={len(folds)}")
        for family in ["LR", "GBM"]:
            preds, fold_df = run_cv(merged, X, y, folds, family, model, "leave_one_presentation_out")
            all_preds.append(preds)
            all_folds.append(fold_df)

    preds = pd.concat(all_preds, ignore_index=True)
    fold_df = pd.concat(all_folds, ignore_index=True)
    summary = summarize(preds, fold_df)
    preds.to_csv(guard_new(out("robustness_oof_predictions_lopo.csv")), index=False)
    fold_df.to_csv(guard_new(out("robustness_lopo_per_course_metrics.csv")), index=False)
    summary.to_csv(guard_new(out("robustness_lopo_metrics_summary.csv")), index=False)
    ov = pd.DataFrame(overlap)
    ov.to_csv(guard_new(out("robustness_lopo_student_overlap.csv")), index=False)

    pws = []
    for family in ["LR", "GBM"]:
        pws.append(add_bh(pd.DataFrame([mcnemar_pair(preds, family, "leave_one_presentation_out", a, b)
                                        for a, b in all_pairs(MODELS)])))
    pw = pd.concat(pws, ignore_index=True)
    pw.to_csv(guard_new(out("robustness_lopo_pairwise_mcnemar_bh.csv")), index=False)

    log_summary(summary, "R2", {"cv": "LeaveOneGroupOut(code_module+code_presentation)"})

    print("\n=== Student overlap across groups ===")
    print(ov.to_string(index=False))
    print("\n=== LOPO metrics summary ===")
    print(summary.to_string(index=False))
    print("\n=== LOPO pairwise McNemar (KEY-aligned, BH within family) ===")
    print(pw.to_string(index=False))
    de = fold_df[fold_df["model"].isin(["D", "E"])].pivot_table(
        index=["family", "fold"], columns="model", values=["accuracy", "auc"])
    print("\n=== Per-course D vs E ===")
    print(de.to_string())


if __name__ == "__main__":
    main()

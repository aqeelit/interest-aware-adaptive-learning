"""
CI-3 (2026-10-13) — student-disjoint validation, models E and D, LR + GBM.

 a. GroupKFold(n_splits=5), groups = id_student. sklearn 1.5.1 GroupKFold has no
    shuffle / random_state, so the split is deterministic and seed-independent.
 b. LOPO student-disjoint: for each held-out presentation, every training row whose
    id_student also appears in the held-out presentation is removed from training.
Comparison E -> D in each scheme x family; BH across all 4.

Outputs: ci3_studentdisjoint_fold_metrics.csv (incl. rows removed per fold),
         ci3_studentdisjoint_metrics_summary.csv, ci3_studentdisjoint_comparisons_bh.csv,
         ci3_studentdisjoint_oof_predictions.csv
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from utils import ensure_dirs
from robustness_common import course_folds, run_cv, summarize, add_bh, log_summary
from ci_common import load_base, compare, write_csv

N_SPLITS = 5


def main():
    ensure_dirs()
    base = load_base()
    y, frame = base["y"], base["frame"]
    ids = frame["id_student"].to_numpy()
    print(f"Row info: {base['rowinfo']}")

    gkf = [(tr, te, f"fold{k}") for k, (tr, te) in
           enumerate(GroupKFold(n_splits=N_SPLITS).split(np.zeros(len(y)), y, groups=ids))]
    fold_info = []
    for tr, te, lab in gkf:
        overlap = len(set(ids[tr]) & set(ids[te]))
        assert overlap == 0
        fold_info.append({"scheme": "groupkfold_student", "fold": lab, "n_train_before_removal": len(tr),
                          "n_train_rows_removed": 0, "n_train": len(tr), "n_test": len(te),
                          "n_test_students": len(set(ids[te])), "train_test_student_overlap_after": overlap})

    lopo_sd = []
    for tr, te, lab in course_folds(frame):
        keep = ~np.isin(ids[tr], ids[te])
        tr2 = tr[keep]
        overlap = len(set(ids[tr2]) & set(ids[te]))
        assert overlap == 0
        lopo_sd.append((tr2, te, lab))
        fold_info.append({"scheme": "lopo_student_disjoint", "fold": lab, "n_train_before_removal": len(tr),
                          "n_train_rows_removed": int((~keep).sum()), "n_train": len(tr2), "n_test": len(te),
                          "n_test_students": len(set(ids[te])), "train_test_student_overlap_after": overlap})
    fold_info = pd.DataFrame(fold_info)

    schemes = {"groupkfold_student": gkf, "lopo_student_disjoint": lopo_sd}
    preds_l, folds_l = [], []
    for sch, folds in schemes.items():
        for name, X in [("E", base["X_E"]), ("D", base["X_D"])]:
            for fam in ["LR", "GBM"]:
                p, f = run_cv(frame, X, y, folds, fam, name, sch)
                preds_l.append(p)
                folds_l.append(f)
    preds = pd.concat(preds_l, ignore_index=True)
    fold_df = pd.concat(folds_l, ignore_index=True).merge(
        fold_info[["scheme", "fold", "n_train_before_removal", "n_train_rows_removed", "n_test_students"]],
        on=["scheme", "fold"], how="left")
    summary = summarize(preds, fold_df)
    for k, v in base["rowinfo"].items():
        summary[k] = v

    comps = add_bh(pd.DataFrame([compare(preds, fam, sch, "E", "D")
                                 for sch in schemes for fam in ["LR", "GBM"]]))
    write_csv(fold_df, "ci3_studentdisjoint_fold_metrics.csv",
              ["n_train_rows_removed = training rows dropped because their id_student is in the held-out set."])
    write_csv(summary, "ci3_studentdisjoint_metrics_summary.csv")
    write_csv(comps, "ci3_studentdisjoint_comparisons_bh.csv",
              ["OR = b_only_correct / a_only_correct; bootstrap differences are b - a; BH across all rows of this file."])
    write_csv(preds, "ci3_studentdisjoint_oof_predictions.csv")
    log_summary(summary, "CI3", {"round": "student-disjoint validation"})

    print(fold_info.to_string(index=False))
    print("total LOPO training rows removed (sum over folds):",
          int(fold_info.loc[fold_info["scheme"] == "lopo_student_disjoint", "n_train_rows_removed"].sum()))
    print(summary.to_string(index=False))
    print(comps.to_string(index=False))


if __name__ == "__main__":
    main()

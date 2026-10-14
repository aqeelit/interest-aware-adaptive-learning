"""
Robustness 1 (2026-10-13) — second model family.

Trains LR (original protocol, re-run for KEY-aligned predictions) and
HistGradientBoostingClassifier on B, Bsens, C, D, E, Dsens, Esens using the
SAME 5 stratified folds (seed 42) per feature set. Reports per-fold and pooled
Accuracy / macro-F1 / AUC, plus all pairwise McNemar tests (KEY-aligned) with
matched-pairs odds ratio + Wald 95% CI and Benjamini-Hochberg correction across
all 21 pairwise comparisons within each model family.

Also checks that the re-run LR predictions reproduce
results/statistical_tests/oulad_fold_predictions.csv exactly.

Outputs (all new files):
  robustness_oof_predictions_stratcv.csv
  robustness_stratcv_fold_metrics.csv
  robustness_stratcv_metrics_summary.csv
  robustness_gbm_pairwise_mcnemar_bh.csv
  robustness_lr_pairwise_mcnemar_bh_keyaligned.csv
  robustness_lr_reproduction_check.csv
"""
import os
import pandas as pd

from utils import RESULTS_STATS, ensure_dirs
from robustness_common import (MODELS, load_model_frame, stratified_folds, run_cv, summarize,
                               mcnemar_pair, add_bh, all_pairs, log_summary, out, guard_new)


def main():
    ensure_dirs()
    all_preds, all_folds, repro = [], [], []
    original = pd.read_csv(os.path.join(RESULTS_STATS, "oulad_fold_predictions.csv"))

    for model in MODELS:
        merged, cols = load_model_frame(model)
        X = merged[cols].to_numpy(dtype=float)
        y = merged["target"].to_numpy()
        folds = stratified_folds(y)
        print(f"[{model}] n={len(y)} n_features={len(cols)}")
        for family in ["LR", "GBM"]:
            preds, fold_df = run_cv(merged, X, y, folds, family, model, "stratified_5fold")
            all_preds.append(preds)
            all_folds.append(fold_df)

        # Reproduction check: original file stores positional row_index into this same merged frame.
        lr = all_preds[-2].reset_index(drop=True)
        orig = original[original["model"] == model].sort_values("row_index")
        same_rows = len(orig) == len(lr) and (orig["row_index"].to_numpy() == lr.index.to_numpy()).all()
        repro.append({
            "model": model, "n_original": int(len(orig)), "n_rerun": int(len(lr)),
            "row_indices_identical": bool(same_rows),
            "y_pred_identical_count": int((orig["y_pred"].to_numpy() == lr["y_pred"].to_numpy()).sum()) if same_rows else None,
            "max_abs_prob_diff": float((orig["y_prob"].to_numpy() - lr["y_prob"].to_numpy()).__abs__().max()) if same_rows else None,
            "fold_identical": bool((orig["fold"].to_numpy() == lr["fold"].str.replace("fold", "").astype(int).to_numpy()).all()) if same_rows else None,
        })

    preds = pd.concat(all_preds, ignore_index=True)
    fold_df = pd.concat(all_folds, ignore_index=True)
    summary = summarize(preds, fold_df)

    preds.to_csv(guard_new(out("robustness_oof_predictions_stratcv.csv")), index=False)
    fold_df.to_csv(guard_new(out("robustness_stratcv_fold_metrics.csv")), index=False)
    summary.to_csv(guard_new(out("robustness_stratcv_metrics_summary.csv")), index=False)
    repro_df = pd.DataFrame(repro)
    repro_df.to_csv(guard_new(out("robustness_lr_reproduction_check.csv")), index=False)

    for family, fname in [("GBM", "robustness_gbm_pairwise_mcnemar_bh.csv"),
                          ("LR", "robustness_lr_pairwise_mcnemar_bh_keyaligned.csv")]:
        pw = add_bh(pd.DataFrame([mcnemar_pair(preds, family, "stratified_5fold", a, b)
                                  for a, b in all_pairs(MODELS)]))
        pw.to_csv(guard_new(out(fname)), index=False)
        print(f"\n=== {family} pairwise McNemar (KEY-aligned, BH over {len(pw)}) ===")
        print(pw.to_string(index=False))

    log_summary(summary, "R1", {"gbm": "HistGradientBoostingClassifier defaults, random_state=42"})

    print("\n=== LR reproduction check vs oulad_fold_predictions.csv ===")
    print(repro_df.to_string(index=False))
    print("\n=== Metrics summary (stratified 5-fold) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

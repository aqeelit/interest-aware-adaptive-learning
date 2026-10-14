"""
Robustness 4 (2026-10-13) — dimensionality controls for the D-vs-E gain.

D adds 21 interest columns (20 click_share_* + resource_diversity_entropy) to E's
7 columns. Two controls with the same column count as D:
  E_noise21     = E + 21 i.i.d. standard-normal noise columns (default_rng(42))
  E_shufint     = E + the REAL 21 interest columns, with the interest block
                  permuted as whole rows across students WITHIN the same
                  code_module+code_presentation (default_rng(42)); the block's
                  joint/marginal within-course distribution is preserved, its
                  link to each student is broken.
Both run for LR and GBM on the same stratified 5 folds as D/E (D and E share
rows and order, so folds are identical). Comparisons: D vs each control and E vs
each control — McNemar (KEY-aligned) + odds ratio CI, BH across all 8
comparisons; plus student-level bootstrap (1000) of D - control.

Each control is a single seeded realization.

Outputs (all new files):
  robustness_dimctrl_oof_predictions.csv
  robustness_dimctrl_fold_metrics.csv
  robustness_dimctrl_metrics_summary.csv   (D and E rows copied from robustness_1 for side-by-side)
  robustness_dimctrl_mcnemar_bh.csv
  robustness_dimctrl_bootstrap.csv
"""
import numpy as np
import pandas as pd

from utils import RANDOM_SEED, ensure_dirs
from robustness_common import (load_model_frame, stratified_folds, run_cv, summarize, mcnemar_pair,
                               add_bh, align_two, paired_bootstrap_diff, log_summary, out, guard_new,
                               COURSE_COLS, N_BOOT)
from train_evaluate_oulad import KEY


def main():
    ensure_dirs()
    d_merged, d_cols = load_model_frame("D")
    e_merged, e_cols = load_model_frame("E")
    assert (d_merged[KEY].to_numpy() == e_merged[KEY].to_numpy()).all(), "D and E rows differ."
    interest_cols = [c for c in d_cols if c not in e_cols]
    print(f"E columns ({len(e_cols)}): {e_cols}")
    print(f"Interest columns in D not in E ({len(interest_cols)}): {interest_cols}")
    assert len(interest_cols) == 21, f"Expected 21 interest columns, found {len(interest_cols)}"

    y = e_merged["target"].to_numpy()
    X_e = e_merged[e_cols].to_numpy(dtype=float)
    folds = stratified_folds(y)

    rng_noise = np.random.default_rng(RANDOM_SEED)
    noise = rng_noise.standard_normal((len(y), len(interest_cols)))

    rng_shuf = np.random.default_rng(RANDOM_SEED)
    interest = d_merged[interest_cols].to_numpy(dtype=float)
    shuffled = interest.copy()
    groups = (d_merged["code_module"] + "_" + d_merged["code_presentation"]).to_numpy()
    for g in sorted(set(groups)):
        idx = np.flatnonzero(groups == g)
        shuffled[idx] = interest[rng_shuf.permutation(idx)]
    n_unchanged = int((shuffled == interest).all(axis=1).sum())
    print(f"Rows whose interest block is identical after within-course shuffle: {n_unchanged} / {len(y)}")

    variants = {"E_noise21": np.hstack([X_e, noise]), "E_shufint": np.hstack([X_e, shuffled])}
    all_preds, all_folds = [], []
    for name, X in variants.items():
        for family in ["LR", "GBM"]:
            p, f = run_cv(e_merged, X, y, folds, family, name, "stratified_5fold")
            all_preds.append(p)
            all_folds.append(f)
    ctrl_preds = pd.concat(all_preds, ignore_index=True)
    ctrl_folds = pd.concat(all_folds, ignore_index=True)

    base_preds = pd.read_csv(out("robustness_oof_predictions_stratcv.csv"))
    base_preds = base_preds[base_preds["model"].isin(["D", "E"])]
    base_folds = pd.read_csv(out("robustness_stratcv_fold_metrics.csv"))
    base_folds = base_folds[base_folds["model"].isin(["D", "E"])]
    preds = pd.concat([base_preds, ctrl_preds], ignore_index=True)
    fold_df = pd.concat([base_folds, ctrl_folds], ignore_index=True)
    summary = summarize(preds, fold_df)

    ctrl_preds.to_csv(guard_new(out("robustness_dimctrl_oof_predictions.csv")), index=False)
    ctrl_folds.to_csv(guard_new(out("robustness_dimctrl_fold_metrics.csv")), index=False)
    summary.to_csv(guard_new(out("robustness_dimctrl_metrics_summary.csv")), index=False)

    comps = [(c, "D") for c in variants] + [("E", c) for c in variants]  # (a, b)
    mc = add_bh(pd.DataFrame([mcnemar_pair(preds, fam, "stratified_5fold", a, b)
                              for fam in ["LR", "GBM"] for a, b in comps]))
    mc.to_csv(guard_new(out("robustness_dimctrl_mcnemar_bh.csv")), index=False)

    boot = []
    for fam in ["LR", "GBM"]:
        for a, b in comps:
            m = align_two(preds, fam, "stratified_5fold", a, b)
            res = paired_bootstrap_diff(m["id_student"].to_numpy(), m["y_true_a"].to_numpy(),
                                        m["y_pred_a"].to_numpy(), m["y_prob_a"].to_numpy(),
                                        m["y_pred_b"].to_numpy(), m["y_prob_b"].to_numpy(), n_boot=N_BOOT)
            boot.append({"family": fam, "model_a": a, "model_b": b, **res})
    boot_df = pd.DataFrame(boot)
    boot_df.to_csv(guard_new(out("robustness_dimctrl_bootstrap.csv")), index=False)

    log_summary(summary[summary["model"].isin(variants)], "R4",
                {"control": "E + 21 noise cols / E + within-course-shuffled interest cols"})

    print("\n=== Metrics summary ===")
    print(summary.to_string(index=False))
    print("\n=== McNemar (BH across 8) ===")
    print(mc.to_string(index=False))
    print("\n=== Bootstrap b - a ===")
    print(boot_df.to_string(index=False))


if __name__ == "__main__":
    main()

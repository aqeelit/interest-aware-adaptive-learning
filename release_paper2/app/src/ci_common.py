"""
Course-identity follow-up (2026-10-13) — shared helpers for ci_1..ci_5.

Builds on robustness_common.py (imported, not modified). All model/fold/metric/
McNemar/bootstrap code is reused from there so this round uses exactly the same
protocol as the robustness round:
  LR = train_evaluate_oulad.make_pipeline (median impute + scale + LR)
  GBM = HistGradientBoostingClassifier(random_state=42), library defaults
  McNemar on KEY-aligned predictions, exact if discordant < 25; matched-pairs OR
  with Wald 95% CI (statistics_tests_oulad.odds_ratio_ci); BH via statsmodels;
  student-level (id_student) cluster bootstrap, 1000 resamples, default_rng(42).

Every output file written by this round must not already exist (guard_absent),
so no earlier file can be overwritten.
"""
import os
import numpy as np
import pandas as pd

from utils import RANDOM_SEED, RESULTS_STATS
from train_evaluate_oulad import KEY, load_target
from robustness_common import (load_model_frame, mcnemar_pair, paired_bootstrap_diff, align_two)

CENTERING_NOTE = (
    "Interest features are centered within each code_module+code_presentation by "
    "subtracting that presentation's mean computed over ALL of its students in the "
    "D/E row set. This is unsupervised (features only, no labels) but uses the "
    "features of test-fold / held-out-course students when computing the mean."
)


def guard_absent(path: str) -> str:
    assert not os.path.exists(path), f"Refusing to overwrite existing file {path}"
    return path


def out(name: str) -> str:
    return guard_absent(os.path.join(RESULTS_STATS, name))


def write_csv(df: pd.DataFrame, name: str, header_lines=()):
    """CSV with optional leading '#'-comment header lines (read back with comment='#')."""
    path = out(name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)
    print(f"wrote {path} ({len(df)} rows)")


def load_base():
    """D/E row set (identical rows and order) + column groups + course labels."""
    d, d_cols = load_model_frame("D")
    e, e_cols = load_model_frame("E")
    assert (d[KEY].to_numpy() == e[KEY].to_numpy()).all(), "D and E rows differ."
    interest_cols = [c for c in d_cols if c not in e_cols]
    assert len(interest_cols) == 21, f"Expected 21 interest columns, found {len(interest_cols)}"
    course = (d["code_module"] + "_" + d["code_presentation"]).to_numpy()
    onehot = pd.get_dummies(pd.Series(course), dtype=float)
    onehot = onehot[sorted(onehot.columns)]
    n_target = len(load_target())
    rowinfo = {"n_target_rows": n_target, "n_rows_used": len(d),
               "n_rows_dropped_by_feature_inner_join": n_target - len(d)}
    return {"frame": d, "y": d["target"].to_numpy(), "course": course,
            "X_E": d[e_cols].to_numpy(dtype=float), "X_D": d[d_cols].to_numpy(dtype=float),
            "interest": d[interest_cols].to_numpy(dtype=float), "interest_cols": interest_cols,
            "onehot": onehot.to_numpy(), "onehot_cols": list(onehot.columns), "rowinfo": rowinfo}


def center_within_course(arr: np.ndarray, course: np.ndarray) -> np.ndarray:
    outarr = arr.copy()
    for g in sorted(set(course)):
        idx = np.flatnonzero(course == g)
        outarr[idx] = arr[idx] - arr[idx].mean(axis=0)
    return outarr


def shuffle_within_course(arr: np.ndarray, course: np.ndarray, seed: int = RANDOM_SEED):
    """Same procedure as robustness_4_dimensionality_control.py: one default_rng(seed),
    groups in sorted order, whole-row permutation inside each group."""
    rng = np.random.default_rng(seed)
    outarr = arr.copy()
    for g in sorted(set(course)):
        idx = np.flatnonzero(course == g)
        outarr[idx] = arr[rng.permutation(idx)]
    n_unchanged = int((outarr == arr).all(axis=1).sum())
    return outarr, n_unchanged


def compare(preds: pd.DataFrame, family: str, scheme: str, a: str, b: str) -> dict:
    """McNemar + OR (b_only/a_only) + student-level bootstrap of (b - a)."""
    row = mcnemar_pair(preds, family, scheme, a, b)
    m = align_two(preds, family, scheme, a, b)
    boot = paired_bootstrap_diff(m["id_student"].to_numpy(), m["y_true_a"].to_numpy(),
                                 m["y_pred_a"].to_numpy(), m["y_prob_a"].to_numpy(),
                                 m["y_pred_b"].to_numpy(), m["y_prob_b"].to_numpy())
    row.update({f"boot_{k}": v for k, v in boot.items()})
    return row

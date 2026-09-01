"""
Phase 18 — Subgroup fairness breakdown for Model D (2026-09-02).

Reuses the EXACT training protocol already used for Model D in
train_evaluate_oulad.py (same feature set D_fused, same KEY, same make_pipeline(),
same StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED=42)) by
importing those pieces directly rather than reimplementing them, so results are
guaranteed identical to the model already reported in results/. This script adds
no new features and does not retrain any other model (A, B, C, E, *_sens).

For each of 5 held-out test folds, this script pools out-of-fold predictions
together with each test row's demographic subgroup membership (disability,
gender, imd_band bucketed into 3 groups), then computes, per subgroup value:
accuracy, macro-F1, AUC-ROC, and a "false_negative_rate" defined EXACTLY as:
    among students who ACTUALLY failed/withdrew (true target = 0),
    what fraction did the model predict would pass (predicted target = 1)?
This is the harm-relevant error named in the task: a student who needed a
supportive intervention but was predicted "will pass" and therefore not flagged.

No fairness gap or composite index is computed — only raw per-subgroup numbers,
per explicit instruction. Interpretation is left to the owner.

Output: results/statistical_tests/oulad_subgroup_fairness.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import OULAD_RAW_DIR, RESULTS_STATS, RANDOM_SEED, ensure_dirs
from train_evaluate_oulad import KEY, N_FOLDS, load_target, load_feature_set, make_pipeline

# imd_band ordinal coding (1 = most deprived .. 10 = least deprived), bucketed
# into 3 groups. Bands 1-3 = most deprived, 4-7 = middle, 8-10 = least deprived
# (a 3/4/3-band split of the 10 official OULAD deprivation deciles).
IMD_BAND_ORDER = [
    "0-10", "10-20", "20-30", "30-40", "40-50",
    "50-60", "60-70", "70-80", "80-90", "90-100",
]


def normalize_imd_band(value):
    if pd.isna(value):
        return None
    stripped = str(value).strip().rstrip("%")
    if stripped in IMD_BAND_ORDER:
        return IMD_BAND_ORDER.index(stripped) + 1
    return None


def imd_bucket(ordinal):
    if ordinal is None:
        return "missing"
    if ordinal <= 3:
        return "most_deprived"
    if ordinal <= 7:
        return "middle"
    return "least_deprived"


def build_subgroup_labels(merged_keys: pd.DataFrame) -> pd.DataFrame:
    """merged_keys: the KEY columns of the exact `merged` dataframe used for
    training, in that dataframe's row order. Returns subgroup labels aligned to
    that same order via a left merge (preserves row count/order exactly)."""
    student_info = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentInfo.csv"))
    labels = merged_keys.merge(
        student_info[KEY + ["disability", "gender", "imd_band"]], on=KEY, how="left"
    )
    labels["imd_band_group"] = labels["imd_band"].apply(
        lambda v: imd_bucket(normalize_imd_band(v))
    )
    return labels


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Rate of predicted=1 among actually-failed (true=0) students. Returns
    (rate, n_actual_fail); rate is NaN if no actually-failed students in group."""
    actual_fail_mask = (y_true == 0)
    n_actual_fail = int(actual_fail_mask.sum())
    if n_actual_fail == 0:
        return float("nan"), 0
    missed = int((y_pred[actual_fail_mask] == 1).sum())
    return missed / n_actual_fail, n_actual_fail


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    fnr, n_actual_fail = false_negative_rate(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "auc": float(auc),
        "false_negative_rate": fnr,
        "n_actual_fail_in_group": n_actual_fail,
    }


def main():
    ensure_dirs()
    target_df = load_target()
    feats = load_feature_set("D")
    if feats is None:
        raise FileNotFoundError(
            "oulad_features_D_fused.parquet not found — run "
            "feature_engineering_oulad.py first."
        )

    merged = target_df.merge(feats, on=KEY, how="inner")
    feature_cols = [c for c in merged.columns if c not in KEY + ["final_result", "target"]]
    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged["target"].to_numpy()

    subgroup_labels = build_subgroup_labels(merged[KEY])
    assert len(subgroup_labels) == len(merged), (
        f"Subgroup label merge changed row count: {len(subgroup_labels)} vs {len(merged)} "
        f"— studentInfo.csv may contain duplicate KEY rows."
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_true = np.full(len(y), -1, dtype=int)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_prob = np.full(len(y), np.nan, dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        pipe = make_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        probs = pipe.predict_proba(X[test_idx])[:, 1]
        oof_true[test_idx] = y[test_idx]
        oof_pred[test_idx] = preds
        oof_prob[test_idx] = probs

    assert (oof_true != -1).all(), "Not every row received an out-of-fold prediction."

    rows = []
    overall = compute_group_metrics(oof_true, oof_pred, oof_prob)
    overall.update({"subgroup_dimension": "ALL", "subgroup_value": "ALL"})
    rows.append(overall)

    for dim, col in [("disability", "disability"), ("gender", "gender"), ("imd_band", "imd_band_group")]:
        for value in sorted(subgroup_labels[col].dropna().unique().tolist()):
            mask = (subgroup_labels[col] == value).to_numpy()
            metrics = compute_group_metrics(oof_true[mask], oof_pred[mask], oof_prob[mask])
            metrics.update({"subgroup_dimension": dim, "subgroup_value": value})
            rows.append(metrics)

    out_df = pd.DataFrame(rows)[
        ["subgroup_dimension", "subgroup_value", "n", "accuracy", "f1_macro",
         "auc", "false_negative_rate", "n_actual_fail_in_group"]
    ]
    out_path = os.path.join(RESULTS_STATS, "oulad_subgroup_fairness.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

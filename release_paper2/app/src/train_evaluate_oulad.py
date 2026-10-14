"""
Phase 9-11 — Train & evaluate feature sets A/B/C/D on OULAD (Study 1).

Uses the SAME model family (logistic regression) across all feature sets, per
experimental_design.md, so differences in performance are attributable to the
FEATURES, not model complexity.

Saves per-fold predictions (not just aggregate metrics) to
results/statistical_tests/oulad_fold_predictions.csv so statistics_tests.py can
run a proper paired comparison — never just compare point-estimate accuracies.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import DATA_PROCESSED, RESULTS_STATS, RESULTS_TABLES, RANDOM_SEED, ensure_dirs, log_experiment

KEY = ["id_student", "code_module", "code_presentation"]
N_FOLDS = 5


def load_target():
    return pd.read_parquet(os.path.join(DATA_PROCESSED, "oulad_target.parquet"))


def load_feature_set(name: str) -> pd.DataFrame | None:
    path_map = {
        "B": "oulad_features_B_performance.parquet",
        "Bsens": "oulad_features_Bsens_performance.parquet",
        "C": "oulad_features_C_behavior.parquet",
        "D": "oulad_features_D_fused.parquet",
        "E": "oulad_features_E_fused.parquet",
        "Dsens": "oulad_features_Dsens_fused.parquet",
        "Esens": "oulad_features_Esens_fused.parquet",
    }
    if name not in path_map:
        return None
    path = os.path.join(DATA_PROCESSED, path_map[name])
    if not os.path.isfile(path):
        return None
    return pd.read_parquet(path)


def make_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])


def evaluate_model(X: np.ndarray, y: np.ndarray, model_name: str) -> list[dict]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_records = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        pipe = make_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        probs = pipe.predict_proba(X[test_idx])[:, 1]

        acc = accuracy_score(y[test_idx], preds)
        f1 = f1_score(y[test_idx], preds, average="macro")
        try:
            auc = roc_auc_score(y[test_idx], probs)
        except ValueError:
            auc = float("nan")  # single-class fold edge case

        for i, idx in enumerate(test_idx):
            fold_records.append({
                "model": model_name,
                "fold": fold_idx,
                "row_index": int(idx),
                "y_true": int(y[idx]),
                "y_pred": int(preds[i]),
                "y_prob": float(probs[i]),
            })

        log_experiment(
            experiment_id=f"EXP-OULAD-{model_name}-fold{fold_idx}",
            dataset_version="oulad-v1",
            model=f"LogisticRegression-{model_name}",
            parameters={"n_folds": N_FOLDS, "max_iter": 1000},
            metrics={"accuracy": acc, "f1_macro": f1, "auc": auc},
            notes=f"Feature set {model_name}, fold {fold_idx}",
        )
    return fold_records


def main():
    ensure_dirs()
    target_df = load_target()
    all_fold_records = []

    # Model A: random/majority baseline — no real features, just class prior.
    y_full = target_df["target"].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_full)), y_full)):
        majority_class = int(pd.Series(y_full[train_idx]).mode()[0])
        preds = np.full(len(test_idx), majority_class)
        prob = np.full(len(test_idx), float(np.mean(y_full[train_idx])))
        acc = accuracy_score(y_full[test_idx], preds)
        f1 = f1_score(y_full[test_idx], preds, average="macro")
        try:
            auc = roc_auc_score(y_full[test_idx], prob)
        except ValueError:
            auc = float("nan")
        for i, idx in enumerate(test_idx):
            all_fold_records.append({
                "model": "A_majority_baseline", "fold": fold_idx, "row_index": int(idx),
                "y_true": int(y_full[idx]), "y_pred": int(preds[i]), "y_prob": float(prob[i]),
            })
        log_experiment(
            experiment_id=f"EXP-OULAD-A-fold{fold_idx}",
            dataset_version="oulad-v1", model="MajorityBaseline",
            parameters={"n_folds": N_FOLDS}, metrics={"accuracy": acc, "f1_macro": f1, "auc": auc},
            notes=f"Majority-class baseline, fold {fold_idx}",
        )

    # Models B, Bsens, C, D, E, Dsens, Esens: merge features to target, keep matched rows.
    for model_name in ["B", "Bsens", "C", "D", "E", "Dsens", "Esens"]:
        feats = load_feature_set(model_name)
        if feats is None:
            print(f"WARNING: features for model {model_name} not found — skipping. "
                  f"Run feature_engineering_oulad.py first.")
            continue
        merged = target_df.merge(feats, on=KEY, how="inner")
        feature_cols = [c for c in merged.columns if c not in KEY + ["final_result", "target"]]
        X = merged[feature_cols].to_numpy(dtype=float)
        y = merged["target"].to_numpy()
        records = evaluate_model(X, y, model_name)
        all_fold_records.extend(records)

    out_path = os.path.join(RESULTS_STATS, "oulad_fold_predictions.csv")
    pd.DataFrame(all_fold_records).to_csv(out_path, index=False)
    print(f"Wrote per-fold predictions -> {out_path}")
    print("Run statistics_tests.py next for paired significance comparisons "
          "before drawing any conclusion about which model performs best.")


if __name__ == "__main__":
    main()

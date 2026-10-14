"""
Robustness round (post-JEDM desk rejection, 2026-10-13) — shared helpers.

New module only; the original Study 1 scripts are imported, never modified, so
the LR protocol here is byte-for-byte the one already reported:
  - same KEY, same feature parquet files, same target
  - same LogisticRegression pipeline (train_evaluate_oulad.make_pipeline)
  - same StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
The second model family is sklearn's HistGradientBoostingClassifier with library
defaults and random_state=42 (no tuning — any tuning would need owner approval).

All pairwise comparisons here align predictions on KEY (id_student, code_module,
code_presentation), NOT on positional row_index. This matters: feature sets B, C,
Bsens and the *sens fusions have different row sets than D/E, so positional
indices do not refer to the same student across those models.
"""
import os
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from utils import RESULTS_STATS, RANDOM_SEED, log_experiment
from train_evaluate_oulad import KEY, N_FOLDS, load_target, load_feature_set, make_pipeline
from statistics_tests_oulad import odds_ratio_ci

MODELS = ["B", "Bsens", "C", "D", "E", "Dsens", "Esens"]
COURSE_COLS = ["code_module", "code_presentation"]
N_BOOT = 1000


def make_gbm():
    return HistGradientBoostingClassifier(random_state=RANDOM_SEED)


FAMILIES = {"LR": make_pipeline, "GBM": make_gbm}


def load_model_frame(name: str):
    """Same merge as train_evaluate_oulad.main(): target inner-joined to features."""
    feats = load_feature_set(name)
    if feats is None:
        raise FileNotFoundError(f"Feature set {name} not found — run feature_engineering_oulad.py first.")
    merged = load_target().merge(feats, on=KEY, how="inner")
    feature_cols = [c for c in merged.columns if c not in KEY + ["final_result", "target"]]
    return merged, feature_cols


def stratified_folds(y: np.ndarray):
    # StratifiedKFold.split depends only on y and n, so this reproduces the folds of
    # train_evaluate_oulad.evaluate_model (which passes X) exactly.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return [(tr, te, f"fold{k}") for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y))]


def course_folds(merged: pd.DataFrame):
    """Leave-one-presentation-out: one held-out group per code_module+code_presentation."""
    groups = (merged["code_module"] + "_" + merged["code_presentation"]).to_numpy()
    logo = LeaveOneGroupOut()
    return [(tr, te, str(groups[te][0])) for tr, te in logo.split(np.zeros(len(merged)), groups=groups)]


def safe_auc(y_true, y_prob, sample_weight=None):
    try:
        return float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight))
    except ValueError:
        return float("nan")


def metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "auc": safe_auc(y_true, y_prob),
    }


def run_cv(merged, X, y, folds, family, model_name, scheme):
    """Fits one estimator per fold; returns (out-of-fold predictions keyed by KEY,
    per-fold metric rows)."""
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_prob = np.full(len(y), np.nan)
    oof_fold = np.empty(len(y), dtype=object)
    fold_rows = []
    for tr, te, label in folds:
        est = FAMILIES[family]()
        est.fit(X[tr], y[tr])
        pred = est.predict(X[te])
        prob = est.predict_proba(X[te])[:, 1]
        oof_pred[te], oof_prob[te], oof_fold[te] = pred, prob, label
        row = {"family": family, "model": model_name, "scheme": scheme, "fold": label,
               "n_train": int(len(tr)), "n_test": int(len(te))}
        row.update(metrics(y[te], pred, prob))
        fold_rows.append(row)
    assert (oof_pred != -1).all(), "Not every row received an out-of-fold prediction."
    preds = merged[KEY].copy()
    preds["family"], preds["model"], preds["scheme"] = family, model_name, scheme
    preds["fold"], preds["y_true"], preds["y_pred"], preds["y_prob"] = oof_fold, y, oof_pred, oof_prob
    return preds, pd.DataFrame(fold_rows)


def summarize(preds: pd.DataFrame, fold_df: pd.DataFrame) -> pd.DataFrame:
    """Per (family, model, scheme): mean/SD of per-fold metrics + pooled out-of-fold metrics."""
    rows = []
    for (fam, mod, sch), g in preds.groupby(["family", "model", "scheme"], sort=False):
        f = fold_df[(fold_df["family"] == fam) & (fold_df["model"] == mod) & (fold_df["scheme"] == sch)]
        row = {"family": fam, "model": mod, "scheme": sch, "n": int(len(g)), "n_folds": int(len(f))}
        for m in ["accuracy", "f1_macro", "auc"]:
            row[f"{m}_fold_mean"] = float(f[m].mean())
            row[f"{m}_fold_sd"] = float(f[m].std(ddof=1))
        pooled = metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), g["y_prob"].to_numpy())
        row.update({f"{k}_pooled": v for k, v in pooled.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_pair(preds: pd.DataFrame, family: str, scheme: str, model_a: str, model_b: str) -> dict:
    """McNemar on correct/incorrect, aligned on KEY. Same exact-test rule and the
    same odds-ratio/Wald-CI function as statistics_tests_oulad.py
    (odds_ratio = b_only_correct / a_only_correct)."""
    sel = (preds["family"] == family) & (preds["scheme"] == scheme)
    a = preds[sel & (preds["model"] == model_a)][KEY + ["y_true", "y_pred"]]
    b = preds[sel & (preds["model"] == model_b)][KEY + ["y_true", "y_pred"]]
    m = a.merge(b, on=KEY, how="inner", suffixes=("_a", "_b"))
    assert (m["y_true_a"] == m["y_true_b"]).all(), "Target mismatch after KEY alignment."
    ac = m["y_pred_a"] == m["y_true_a"]
    bc = m["y_pred_b"] == m["y_true_a"]
    both, a_only, b_only, neither = int((ac & bc).sum()), int((ac & ~bc).sum()), int((~ac & bc).sum()), int((~ac & ~bc).sum())
    res = mcnemar([[both, a_only], [b_only, neither]], exact=(a_only + b_only) < 25)
    row = {"family": family, "scheme": scheme, "model_a": model_a, "model_b": model_b,
           "n_common": int(len(m)), "acc_a_on_common": float(ac.mean()), "acc_b_on_common": float(bc.mean()),
           "a_only_correct": a_only, "b_only_correct": b_only,
           "statistic": float(res.statistic), "p_value": float(res.pvalue)}
    row.update(odds_ratio_ci(a_only, b_only))
    row["odds_ratio_definition"] = "b_only_correct / a_only_correct"
    return row


def add_bh(df: pd.DataFrame, pcol: str = "p_value") -> pd.DataFrame:
    reject, p_bh, _, _ = multipletests(df[pcol].to_numpy(dtype=float), alpha=0.05, method="fdr_bh")
    df = df.copy()
    df["p_value_bh"] = p_bh
    df["significant_after_bh_0.05"] = reject
    df["bh_family_size"] = len(df)
    return df


def all_pairs(models):
    return list(itertools.combinations(models, 2))


def cluster_weights(rng, inv: np.ndarray, n_clusters: int) -> np.ndarray:
    """One student-level bootstrap resample, expressed as per-row integer weights
    (equivalent to duplicating every row of each drawn student)."""
    draw = rng.integers(0, n_clusters, n_clusters)
    return np.bincount(draw, minlength=n_clusters)[inv].astype(float)


def paired_bootstrap_diff(ids, y, pred_a, prob_a, pred_b, prob_b, n_boot=N_BOOT, seed=RANDOM_SEED) -> dict:
    """Student-level (id_student) cluster bootstrap of (b - a) for pooled accuracy
    and pooled AUC. Both models are evaluated on the SAME resample each time."""
    _, inv = np.unique(ids, return_inverse=True)
    n_clusters = inv.max() + 1
    corr_a, corr_b = (pred_a == y).astype(float), (pred_b == y).astype(float)
    rng = np.random.default_rng(seed)
    d_acc, d_auc = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        w = cluster_weights(rng, inv, n_clusters)
        d_acc[i] = np.dot(w, corr_b) / w.sum() - np.dot(w, corr_a) / w.sum()
        d_auc[i] = safe_auc(y, prob_b, w) - safe_auc(y, prob_a, w)
    return {
        "n_rows": int(len(y)), "n_students": int(n_clusters), "n_boot": n_boot, "seed": seed,
        "acc_a": float(corr_a.mean()), "acc_b": float(corr_b.mean()),
        "acc_diff_b_minus_a": float(corr_b.mean() - corr_a.mean()),
        "acc_diff_ci_low": float(np.percentile(d_acc, 2.5)), "acc_diff_ci_high": float(np.percentile(d_acc, 97.5)),
        "acc_diff_boot_frac_le_0": float(np.mean(d_acc <= 0)),
        "auc_a": safe_auc(y, prob_a), "auc_b": safe_auc(y, prob_b),
        "auc_diff_b_minus_a": safe_auc(y, prob_b) - safe_auc(y, prob_a),
        "auc_diff_ci_low": float(np.nanpercentile(d_auc, 2.5)), "auc_diff_ci_high": float(np.nanpercentile(d_auc, 97.5)),
        "auc_diff_boot_frac_le_0": float(np.mean(d_auc <= 0)),
        "auc_boot_nan_count": int(np.isnan(d_auc).sum()),
    }


def align_two(preds, family, scheme, model_a, model_b):
    sel = (preds["family"] == family) & (preds["scheme"] == scheme)
    a = preds[sel & (preds["model"] == model_a)][KEY + ["y_true", "y_pred", "y_prob"]]
    b = preds[sel & (preds["model"] == model_b)][KEY + ["y_true", "y_pred", "y_prob"]]
    m = a.merge(b, on=KEY, how="inner", suffixes=("_a", "_b"))
    assert (m["y_true_a"] == m["y_true_b"]).all(), "Target mismatch after KEY alignment."
    return m


def log_summary(summary: pd.DataFrame, tag: str, extra_params: dict):
    for _, r in summary.iterrows():
        log_experiment(
            experiment_id=f"EXP-OULAD-ROBUST-{tag}-{r['family']}-{r['model']}-{r['scheme']}",
            dataset_version="oulad-v1",
            model=f"{'LogisticRegression' if r['family'] == 'LR' else 'HistGradientBoostingClassifier'}-{r['model']}",
            parameters={"scheme": r["scheme"], "n_folds": int(r["n_folds"]), **extra_params},
            metrics={k: float(r[k]) for k in ["accuracy_pooled", "f1_macro_pooled", "auc_pooled",
                                               "accuracy_fold_mean", "f1_macro_fold_mean", "auc_fold_mean"]},
            notes=f"Robustness round 2026-10-13 ({tag}); pooled = out-of-fold predictions pooled across folds",
        )


def out(name: str) -> str:
    return os.path.join(RESULTS_STATS, name)


def guard_new(path: str):
    """Never overwrite pre-existing results files from earlier phases."""
    protected = {"oulad_fold_predictions.csv", "oulad_pairwise_comparison.csv",
                 "oulad_pairwise_comparison_corrected.csv", "oulad_subgroup_fairness.csv"}
    assert os.path.basename(path) not in protected, f"Refusing to overwrite {path}"
    return path

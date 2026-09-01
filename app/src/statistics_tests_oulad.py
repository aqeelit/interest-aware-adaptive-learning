"""
Phase 11 — Statistical comparison across feature sets A/B/C/D (Study 1).

Per CLAUDE.md and experimental_design.md: never conclude one model "wins" from
point-estimate metrics alone. This script runs a paired comparison (McNemar's test
on classification agreement/disagreement) between every pair of models, using the
per-fold predictions saved by train_evaluate_oulad.py.

Output: results/statistical_tests/oulad_pairwise_comparison.csv (unchanged, original)

Phase 18 addendum (2026-09-02) — two additional analyses on the SAME 28 pairwise
comparisons, written to a SEPARATE file (oulad_pairwise_comparison_corrected.csv)
so the original file is never overwritten:
  1. Benjamini-Hochberg FDR correction across all 28 comparisons at once
     (statsmodels.stats.multitest.multipletests, method='fdr_bh').
  2. Matched-pairs (McNemar) odds ratio = b_only_correct / a_only_correct, with a
     Wald confidence interval on the log odds ratio — the standard closed-form CI
     for matched-pairs designs (Fleiss, Levin & Paik, "Statistical Methods for
     Rates and Proportions", 3rd ed., ch. 10), using scipy.stats.norm for the
     critical value rather than a hardcoded 1.96. A Haldane-Anscombe continuity
     correction (+0.5 to both discordant cells) is applied only when either
     discordant count is zero, to avoid division by zero / log(0); such rows are
     flagged in a boolean column.
"""
import os
import itertools
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from utils import RESULTS_STATS, ensure_dirs


def load_predictions() -> pd.DataFrame:
    path = os.path.join(RESULTS_STATS, "oulad_fold_predictions.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found — run train_evaluate_oulad.py first."
        )
    return pd.read_csv(path)


def pairwise_mcnemar(df: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Aligns predictions on row_index (same held-out sample across models where
    both were evaluated) and runs McNemar's test on correct/incorrect agreement.
    """
    a = df[df["model"] == model_a][["row_index", "y_true", "y_pred"]].rename(
        columns={"y_pred": "pred_a"})
    b = df[df["model"] == model_b][["row_index", "y_true", "y_pred"]].rename(
        columns={"y_pred": "pred_b"})
    merged = a.merge(b, on=["row_index", "y_true"], how="inner")
    if len(merged) == 0:
        return {"model_a": model_a, "model_b": model_b, "n_common": 0,
                "statistic": None, "p_value": None, "note": "no overlapping evaluated rows"}

    a_correct = (merged["pred_a"] == merged["y_true"])
    b_correct = (merged["pred_b"] == merged["y_true"])

    both_correct = int((a_correct & b_correct).sum())
    a_only_correct = int((a_correct & ~b_correct).sum())
    b_only_correct = int((~a_correct & b_correct).sum())
    both_wrong = int((~a_correct & ~b_correct).sum())

    table = [[both_correct, a_only_correct], [b_only_correct, both_wrong]]
    result = mcnemar(table, exact=(a_only_correct + b_only_correct) < 25)

    return {
        "model_a": model_a, "model_b": model_b, "n_common": len(merged),
        "a_only_correct": a_only_correct, "b_only_correct": b_only_correct,
        "statistic": float(result.statistic), "p_value": float(result.pvalue),
        "significant_at_0.05": bool(result.pvalue < 0.05),
    }


def odds_ratio_ci(a_only_correct: int, b_only_correct: int, alpha: float = 0.05) -> dict:
    """Matched-pairs (McNemar) odds ratio = b_only_correct / a_only_correct, with a
    Wald 95% CI on the log odds ratio. See module docstring for the method
    reference. Returns None-filled fields only if both discordant counts are zero
    (no information to estimate an odds ratio at all).
    """
    a, b = float(a_only_correct), float(b_only_correct)
    if a == 0 and b == 0:
        return {"odds_ratio": None, "odds_ratio_ci_low": None,
                "odds_ratio_ci_high": None, "odds_ratio_continuity_corrected": None}

    corrected = (a == 0 or b == 0)
    if corrected:
        a, b = a + 0.5, b + 0.5

    odds_ratio = b / a
    se_log_or = np.sqrt(1.0 / a + 1.0 / b)
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    log_or = np.log(odds_ratio)
    ci_low = float(np.exp(log_or - z * se_log_or))
    ci_high = float(np.exp(log_or + z * se_log_or))
    return {"odds_ratio": float(odds_ratio), "odds_ratio_ci_low": ci_low,
            "odds_ratio_ci_high": ci_high, "odds_ratio_continuity_corrected": corrected}


def main():
    ensure_dirs()
    df = load_predictions()
    models = sorted(df["model"].unique().tolist())
    if len(models) < 2:
        print("Fewer than 2 models found in predictions file — nothing to compare.")
        return

    rows = []
    for model_a, model_b in itertools.combinations(models, 2):
        rows.append(pairwise_mcnemar(df, model_a, model_b))

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_STATS, "oulad_pairwise_comparison.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote pairwise McNemar comparisons -> {out_path}")
    print(out_df.to_string(index=False))
    print("\nOnly cite a model as 'significantly better' in the manuscript if "
          "significant_at_0.05 is True here — per CLAUDE.md traceability rule.")

    # --- Phase 18 addendum: FDR correction + odds ratio, separate output file ---
    corrected_df = out_df.copy()
    reject, p_bh, _, _ = multipletests(
        corrected_df["p_value"].to_numpy(dtype=float), alpha=0.05, method="fdr_bh"
    )
    corrected_df["p_value_bh_corrected"] = p_bh
    corrected_df["significant_after_correction"] = reject

    or_records = [
        odds_ratio_ci(row["a_only_correct"], row["b_only_correct"])
        for _, row in corrected_df.iterrows()
    ]
    or_df = pd.DataFrame(or_records)
    corrected_df = pd.concat([corrected_df.reset_index(drop=True), or_df], axis=1)

    corrected_out_path = os.path.join(RESULTS_STATS, "oulad_pairwise_comparison_corrected.csv")
    corrected_df.to_csv(corrected_out_path, index=False)
    print(f"\nWrote FDR-corrected + odds-ratio comparisons -> {corrected_out_path}")
    print(corrected_df.to_string(index=False))

    d_vs_e = corrected_df[
        ((corrected_df["model_a"] == "D") & (corrected_df["model_b"] == "E"))
        | ((corrected_df["model_a"] == "E") & (corrected_df["model_b"] == "D"))
    ]
    if len(d_vs_e) == 1:
        row = d_vs_e.iloc[0]
        print(f"\n=== D vs E after Benjamini-Hochberg FDR correction ===")
        print(f"raw p_value = {row['p_value']:.6e}")
        print(f"p_value_bh_corrected = {row['p_value_bh_corrected']:.6e}")
        print(f"significant_after_correction = {row['significant_after_correction']}")
    else:
        print(f"\nWARNING: expected exactly 1 D-vs-E row, found {len(d_vs_e)} — "
              f"cannot report the central D-vs-E FDR result.")


if __name__ == "__main__":
    main()

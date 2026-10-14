"""
CI-5 (2026-10-13) — re-derive the manuscript's pairwise statistics, key-aligned.

Input: the ORIGINAL LR stratified-5-fold predictions,
results/statistical_tests/oulad_fold_predictions.csv (no retraining). Each
model's positional row_index is mapped back to (id_student, code_module,
code_presentation) using the exact frame train_evaluate_oulad.py indexed into:
  A_majority_baseline -> oulad_target.parquet row order
  every other model   -> target.merge(features, on=KEY, how="inner") row order
Pairs are then matched on KEY. McNemar (exact if discordant < 25), odds ratio
b_only/a_only with Wald 95% CI (statistics_tests_oulad.odds_ratio_ci), and BH
over the full 28-pair family (same family as oulad_pairwise_comparison_corrected.csv);
a second BH over only the 13 manuscript-cited pairs is also given.
Pair orientation (model_a, model_b) is copied from the old files.

Outputs:
  ci5_keyaligned_all28_pairs.csv          all 28 pairs, old vs new n_common etc.
  ci5_manuscript_cited_old_vs_new.csv     long table: pair x statistic, old, new, flag
  ci5_significance_count_old_vs_new.csv   count significant after BH (28) old vs new
"""
import os
import numpy as np
import pandas as pd

from utils import ensure_dirs, RESULTS_STATS
from train_evaluate_oulad import KEY, load_target
from statistics_tests_oulad import odds_ratio_ci
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from robustness_common import load_model_frame
from ci_common import write_csv

A = "A_majority_baseline"
CITED = {
    ("B", "E"): ("manuscript.md L463-464 (H1); abstract L72-73", ["statistic", "p_value"]),
    ("C", "E"): ("manuscript.md L463-464 (H1); abstract L72-73", ["statistic", "p_value"]),
    ("D", "E"): ("manuscript.md L467-474; abstract L74-75",
                 ["a_only_correct", "b_only_correct", "p_value", "p_value_bh_corrected",
                  "significant_after_correction", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high"]),
    ("B", "Bsens"): ("manuscript.md L486-491", ["significant_at_0.05", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high"]),
    ("D", "Dsens"): ("manuscript.md L486-488", ["significant_at_0.05"]),
    ("E", "Esens"): ("manuscript.md L496-498", ["p_value", "p_value_bh_corrected", "significant_after_correction"]),
}
for m in ["B", "Bsens", "C", "D", "Dsens", "E", "Esens"]:
    CITED[(A, m)] = ("manuscript.md L458-460 (all p < 0.001; Supplementary Table S1)", ["p_value"])
STATS = ["n_common", "a_only_correct", "b_only_correct", "statistic", "p_value", "significant_at_0.05",
         "p_value_bh_corrected", "significant_after_correction", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high"]


def keyed_predictions() -> pd.DataFrame:
    orig = pd.read_csv(os.path.join(RESULTS_STATS, "oulad_fold_predictions.csv"))
    target = load_target()
    parts = []
    for model, g in orig.groupby("model"):
        frame = target if model == A else load_model_frame(model)[0]
        assert g["row_index"].max() < len(frame) and g["row_index"].is_unique
        keys = frame.iloc[g["row_index"].to_numpy()][KEY + ["target"]].reset_index(drop=True)
        g = g.reset_index(drop=True)
        assert (keys["target"].to_numpy() == g["y_true"].to_numpy()).all(), f"y_true mismatch for {model}"
        parts.append(pd.concat([keys[KEY], g[["model", "y_true", "y_pred", "y_prob"]]], axis=1))
    return pd.concat(parts, ignore_index=True)


def pair(preds, a, b) -> dict:
    pa = preds[preds["model"] == a][KEY + ["y_true", "y_pred"]]
    pb = preds[preds["model"] == b][KEY + ["y_true", "y_pred"]]
    m = pa.merge(pb, on=KEY, how="inner", suffixes=("_a", "_b"))
    assert (m["y_true_a"] == m["y_true_b"]).all()
    ac, bc = m["y_pred_a"] == m["y_true_a"], m["y_pred_b"] == m["y_true_a"]
    both, a_only, b_only, neither = int((ac & bc).sum()), int((ac & ~bc).sum()), int((~ac & bc).sum()), int((~ac & ~bc).sum())
    res = mcnemar([[both, a_only], [b_only, neither]], exact=(a_only + b_only) < 25)
    row = {"model_a": a, "model_b": b, "n_common": len(m), "a_only_correct": a_only, "b_only_correct": b_only,
           "statistic": float(res.statistic), "p_value": float(res.pvalue),
           "significant_at_0.05": bool(res.pvalue < 0.05)}
    row.update(odds_ratio_ci(a_only, b_only))
    return row


def same(o, n):
    if isinstance(o, (bool, np.bool_)) or isinstance(n, (bool, np.bool_)):
        return bool(o) == bool(n)
    o, n = float(o), float(n)
    if o == n:
        return True
    return abs(o - n) <= 1e-9 * max(abs(o), abs(n))


def main():
    ensure_dirs()
    preds = keyed_predictions()
    old = pd.read_csv(os.path.join(RESULTS_STATS, "oulad_pairwise_comparison_corrected.csv"))
    old_raw = pd.read_csv(os.path.join(RESULTS_STATS, "oulad_pairwise_comparison.csv"))
    assert len(old) == 28 and len(old_raw) == 28

    new = pd.DataFrame([pair(preds, a, b) for a, b in zip(old["model_a"], old["model_b"])])
    rej, pbh, _, _ = multipletests(new["p_value"].to_numpy(float), alpha=0.05, method="fdr_bh")
    new["p_value_bh_corrected"], new["significant_after_correction"] = pbh, rej
    cited_mask = [(a, b) in CITED for a, b in zip(new["model_a"], new["model_b"])]
    assert sum(cited_mask) == 13
    rej13, pbh13, _, _ = multipletests(new.loc[cited_mask, "p_value"].to_numpy(float), alpha=0.05, method="fdr_bh")
    new["p_value_bh_cited13_only"] = np.nan
    new.loc[cited_mask, "p_value_bh_cited13_only"] = pbh13
    new["manuscript_cited_pair"] = cited_mask

    all28 = new.copy()
    all28.insert(2, "n_common_OLD_row_position", old["n_common"].to_numpy())
    all28.insert(3, "n_common_changed", all28["n_common"] != all28["n_common_OLD_row_position"])
    for c in ["a_only_correct", "b_only_correct", "p_value", "p_value_bh_corrected", "odds_ratio"]:
        all28[f"{c}_OLD"] = old[c].to_numpy()

    long_rows = []
    for i, r in new.iterrows():
        key = (r["model_a"], r["model_b"])
        if key not in CITED:
            continue
        where, cited_stats = CITED[key]
        o = old.iloc[i]
        assert (o["model_a"], o["model_b"]) == key
        for s in STATS:
            long_rows.append({"model_a": key[0], "model_b": key[1], "statistic_name": s,
                              "cited_in_manuscript_text": s in cited_stats, "manuscript_location": where,
                              "OLD_value_row_position": o[s], "NEW_value_key_aligned": r[s],
                              "flag": "unchanged" if same(o[s], r[s]) else "changed"})
    long_df = pd.DataFrame(long_rows)

    sig = pd.DataFrame([{
        "family_size": 28,
        "OLD_n_significant_after_bh": int(old["significant_after_correction"].sum()),
        "NEW_n_significant_after_bh": int(new["significant_after_correction"].sum()),
        "OLD_nonsignificant_pairs": ";".join(f"{a}-{b}" for a, b, s in zip(old["model_a"], old["model_b"], old["significant_after_correction"]) if not s),
        "NEW_nonsignificant_pairs": ";".join(f"{a}-{b}" for a, b, s in zip(new["model_a"], new["model_b"], new["significant_after_correction"]) if not s),
    }])

    hdr = ["Source predictions: oulad_fold_predictions.csv (LR, stratified 5-fold, seed 42), keys reattached; "
           "OLD = oulad_pairwise_comparison(_corrected).csv (positional row_index alignment). "
           "OR = b_only_correct / a_only_correct. BH over all 28 pairs unless column says otherwise."]
    write_csv(all28, "ci5_keyaligned_all28_pairs.csv", hdr)
    write_csv(long_df, "ci5_manuscript_cited_old_vs_new.csv", hdr)
    write_csv(sig, "ci5_significance_count_old_vs_new.csv", hdr)
    pd.set_option("display.width", 250)
    print(all28[["model_a", "model_b", "n_common_OLD_row_position", "n_common", "n_common_changed",
                 "statistic", "p_value", "p_value_bh_corrected", "odds_ratio", "odds_ratio_ci_low",
                 "odds_ratio_ci_high"]].to_string(index=False))
    print(long_df.to_string(index=False))
    print(sig.to_string(index=False))


if __name__ == "__main__":
    main()

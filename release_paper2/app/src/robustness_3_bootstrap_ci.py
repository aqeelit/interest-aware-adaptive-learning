"""
Robustness 3 (2026-10-13) — bootstrap confidence intervals for D vs E.

Student-level cluster bootstrap (resampling id_student with replacement, 1000
resamples, numpy default_rng(42)) of the pooled out-of-fold Accuracy difference
and AUC difference (D - E), for LR and GBM. Run on the stratified 5-fold
predictions (robustness_1) and the leave-one-presentation-out predictions
(robustness_2). Dsens - Esens is included as the circularity-sensitivity pair.
Percentile 95% CIs.

Output: robustness_bootstrap_d_vs_e.csv
"""
import pandas as pd

from utils import ensure_dirs
from robustness_common import align_two, paired_bootstrap_diff, out, guard_new, N_BOOT

PAIRS = [("E", "D"), ("Esens", "Dsens")]  # (a, b): differences are b - a


def main():
    ensure_dirs()
    sources = {
        "stratified_5fold": pd.read_csv(out("robustness_oof_predictions_stratcv.csv")),
        "leave_one_presentation_out": pd.read_csv(out("robustness_oof_predictions_lopo.csv")),
    }
    rows = []
    for scheme, preds in sources.items():
        for family in ["LR", "GBM"]:
            for a, b in PAIRS:
                m = align_two(preds, family, scheme, a, b)
                res = paired_bootstrap_diff(
                    m["id_student"].to_numpy(), m["y_true_a"].to_numpy(),
                    m["y_pred_a"].to_numpy(), m["y_prob_a"].to_numpy(),
                    m["y_pred_b"].to_numpy(), m["y_prob_b"].to_numpy(), n_boot=N_BOOT)
                rows.append({"scheme": scheme, "family": family, "model_a": a, "model_b": b, **res})
                print(f"done {scheme} {family} {b}-{a}")
    df = pd.DataFrame(rows)
    df.to_csv(guard_new(out("robustness_bootstrap_d_vs_e.csv")), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

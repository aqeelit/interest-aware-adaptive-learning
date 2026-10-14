"""
Robustness 5 (2026-10-13) — gender FNR gap for B, E, D (LR and GBM).

FNR uses the exact definition of subgroup_fairness_oulad.py: among students who
actually failed/withdrew (target = 0), the fraction predicted to pass (pred = 1).
Uses the stratified 5-fold out-of-fold predictions from robustness_1.
gender comes from data/raw/oulad/studentInfo.csv, joined on KEY.

Per (family, model): FNR_F, FNR_M, gap = FNR_F - FNR_M, with a student-level
cluster bootstrap (1000, default_rng(42)) percentile 95% CI.
Additionally, difference-in-gaps (D gap - E gap; D gap - B gap) on the rows
common to both models, with the same paired student-level bootstrap.

Outputs (new files):
  robustness_fairness_gender_fnr.csv
  robustness_fairness_gender_gap_differences.csv
"""
import os
import numpy as np
import pandas as pd

from utils import OULAD_RAW_DIR, RANDOM_SEED, ensure_dirs
from train_evaluate_oulad import KEY
from robustness_common import cluster_weights, out, guard_new, N_BOOT

MODELS = ["B", "E", "D"]


def fnr_parts(y, pred, is_f):
    """Per-row indicator arrays: (fail & F), (fail & F & missed), same for M."""
    fail = y == 0
    missed = fail & (pred == 1)
    return (fail & is_f).astype(float), (missed & is_f).astype(float), \
           (fail & ~is_f).astype(float), (missed & ~is_f).astype(float)


def gap(parts, w=None):
    ff, mf, fm, mm = parts
    if w is None:
        w = np.ones_like(ff)
    fnr_f = np.dot(w, mf) / np.dot(w, ff)
    fnr_m = np.dot(w, mm) / np.dot(w, fm)
    return fnr_f, fnr_m, fnr_f - fnr_m


def main():
    ensure_dirs()
    preds = pd.read_csv(out("robustness_oof_predictions_stratcv.csv"))
    info = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentInfo.csv"))[KEY + ["gender"]]
    assert not info.duplicated(KEY).any()
    print("gender values:", sorted(info["gender"].dropna().unique().tolist()))

    rows, gap_rows = [], []
    for fam in ["LR", "GBM"]:
        frames = {}
        for model in MODELS:
            g = preds[(preds["family"] == fam) & (preds["model"] == model)].merge(info, on=KEY, how="left")
            assert g["gender"].isin(["F", "M"]).all(), "Unexpected/missing gender values"
            frames[model] = g
            y, pred, is_f = g["y_true"].to_numpy(), g["y_pred"].to_numpy(), (g["gender"] == "F").to_numpy()
            parts = fnr_parts(y, pred, is_f)
            fnr_f, fnr_m, d = gap(parts)
            _, inv = np.unique(g["id_student"].to_numpy(), return_inverse=True)
            rng = np.random.default_rng(RANDOM_SEED)
            boots = np.array([gap(parts, cluster_weights(rng, inv, inv.max() + 1))[2] for _ in range(N_BOOT)])
            rows.append({"family": fam, "model": model, "n_rows": int(len(g)),
                         "n_actual_fail_F": int(parts[0].sum()), "n_actual_fail_M": int(parts[2].sum()),
                         "fnr_F": fnr_f, "fnr_M": fnr_m, "gap_F_minus_M": d,
                         "gap_ci_low": float(np.percentile(boots, 2.5)),
                         "gap_ci_high": float(np.percentile(boots, 97.5)),
                         "n_boot": N_BOOT, "seed": RANDOM_SEED})

        for other in ["E", "B"]:
            m = frames["D"].merge(frames[other][KEY + ["y_pred"]], on=KEY, suffixes=("_D", "_o"))
            y, is_f = m["y_true"].to_numpy(), (m["gender"] == "F").to_numpy()
            pd_, po = fnr_parts(y, m["y_pred_D"].to_numpy(), is_f), fnr_parts(y, m["y_pred_o"].to_numpy(), is_f)
            point = gap(pd_)[2] - gap(po)[2]
            _, inv = np.unique(m["id_student"].to_numpy(), return_inverse=True)
            rng = np.random.default_rng(RANDOM_SEED)
            boots = []
            for _ in range(N_BOOT):
                w = cluster_weights(rng, inv, inv.max() + 1)
                boots.append(gap(pd_, w)[2] - gap(po, w)[2])
            gap_rows.append({"family": fam, "comparison": f"gap(D) - gap({other})", "n_common_rows": int(len(m)),
                             "gap_D_on_common": gap(pd_)[2], f"gap_other_on_common": gap(po)[2],
                             "diff_in_gaps": point,
                             "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
                             "n_boot": N_BOOT, "seed": RANDOM_SEED})

    df, gdf = pd.DataFrame(rows), pd.DataFrame(gap_rows)
    df.to_csv(guard_new(out("robustness_fairness_gender_fnr.csv")), index=False)
    gdf.to_csv(guard_new(out("robustness_fairness_gender_gap_differences.csv")), index=False)
    print(df.to_string(index=False))
    print()
    print(gdf.to_string(index=False))


if __name__ == "__main__":
    main()

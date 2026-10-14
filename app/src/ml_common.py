"""
ml_ round (2026-10-14; EXPLORATORY / POST HOC) — shared helpers.
Pre-registration: results/statistical_tests/prereg_ml_2026-10-14.md.

Imports read-only: robustness_common, ci_common (models, folds, McNemar, OR, cluster weights,
FULL sample) and wdearly_common.build_sample (EARLY samples, as in the earlier EARLY rounds).
Nothing from the final_ round is read or imported.

Smoke mode (env ML_SMOKE=1): each setting is subsampled to SMOKE_N rows, the bootstrap uses
SMOKE_BOOT resamples, outputs go to /tmp/ml_smoke (inside the container, discarded) and nothing
is appended to experiment_log.csv. Used only to test every combine/concat step before the full run.
"""
import os
import numpy as np
import pandas as pd

import wdearly_common as W
from utils import RESULTS_STATS, RANDOM_SEED
from train_evaluate_oulad import KEY
from robustness_common import mcnemar_pair, align_two, cluster_weights, safe_auc, log_summary, N_BOOT
from ci_common import load_base

SMOKE = os.environ.get("ML_SMOKE") == "1"
SMOKE_N, SMOKE_BOOT = 3000, 20
OUT = "/tmp/ml_smoke" if SMOKE else RESULTS_STATS
NBOOT = SMOKE_BOOT if SMOKE else N_BOOT
FAMILIES = ["LR", "GBM"]
SETTINGS = ["FULL", "EARLY_t28", "EARLY_t56"]
EXPLORATORY = "EXPLORATORY / POST HOC (prereg_ml_2026-10-14.md); hypothesis-generating only."
CENTER_NOTE = ("Course centering and course means use only the students of that presentation in the analysed "
               "sample, features only, no labels. Columns constant within a presentation get centered value 0.")
BOOT_NOTE = f"Bootstrap: student-level (id_student), {NBOOT} resamples, seed {RANDOM_SEED}; differences b - a."


def write_csv(df: pd.DataFrame, name: str, header_lines=()):
    assert name.startswith("ml_"), name
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    if not SMOKE:
        assert not os.path.exists(path), f"Refusing to overwrite {path}"
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in [EXPLORATORY, *header_lines]:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)
    print(f"wrote {path} ({len(df)} rows)", flush=True)


def log(summary: pd.DataFrame, tag: str, extra: dict):
    """Append to experiment_log.csv; call only after the stage's outputs were written."""
    if SMOKE:
        print(f"[smoke] skipped experiment_log for {tag} ({len(summary)} rows)", flush=True)
        return
    log_summary(summary, f"ML-{tag}", extra)


def _subsample(frame, arrays):
    idx = np.sort(np.random.default_rng(0).choice(len(frame), SMOKE_N, replace=False))
    return frame.iloc[idx].reset_index(drop=True), {k: v[idx] for k, v in arrays.items()}


def load_setting(setting, data=None):
    """Returns dict: frame (KEY, target, code_module...), y, Xb, I (21 cols), icols, course, module, rowinfo."""
    if setting == "FULL":
        b = load_base()
        frame = b["frame"]
        arrays = {"Xb": b["X_E"], "I": b["interest"]}
        icols = b["interest_cols"]
        rowinfo = {"n_source_rows": b["rowinfo"]["n_target_rows"], "n_rows_used": b["rowinfo"]["n_rows_used"],
                   "n_dropped_rows": b["rowinfo"]["n_rows_dropped_by_feature_inner_join"],
                   "dropped_reason": "target rows without all feature sets (inner join, D/E row set)"}
    else:
        t = int(setting.split("_t")[1])
        s = W.build_sample("EARLY", cutoff=t, data=data)
        frame = s["frame"]
        arrays = {"Xb": frame[s["C"]].to_numpy(float), "I": frame[s["I"]].to_numpy(float)}
        icols = s["I"]
        ri = s["rowinfo"]
        rowinfo = {"n_source_rows": ri["n_studentinfo_rows"], "n_rows_used": ri["n_rows_used"],
                   "n_dropped_rows": ri["n_studentinfo_rows"] - ri["n_rows_used"],
                   "dropped_reason": f"unregistered by t={t} (risk set); 0 dropped by feature build"}
    assert len(icols) == 21
    if SMOKE:
        frame, arrays = _subsample(frame, arrays)
        rowinfo = {**rowinfo, "n_rows_used": len(frame), "dropped_reason": "SMOKE subsample"}
    course = (frame["code_module"] + "_" + frame["code_presentation"]).to_numpy()
    return {"setting": setting, "frame": frame, "y": frame["target"].to_numpy(), "Xb": arrays["Xb"],
            "I": arrays["I"], "icols": list(icols), "course": course, "module": frame["code_module"].to_numpy(),
            "rowinfo": rowinfo}


def onehot(course):
    oh = pd.get_dummies(pd.Series(course), dtype=float)
    return oh[sorted(oh.columns)].to_numpy()


def center_and_mean(arr, course):
    """Within-presentation centered values and presentation means (features only).
    Columns constant inside a presentation: centered = 0 exactly, mean = that constant."""
    ctr, mean = np.zeros_like(arr), np.zeros_like(arr)
    for g in sorted(set(course)):
        idx = np.flatnonzero(course == g)
        sub = arr[idx]
        mu = sub.mean(axis=0)
        const = (sub == sub[0]).all(axis=0)
        mu = np.where(const, sub[0], mu)
        c = sub - mu
        c[:, const] = 0.0
        ctr[idx], mean[idx] = c, mu
    return ctr, mean


def icc1(x, groups):
    """ICC(1), one-way ANOVA estimator. Returns dict; icc NaN (undefined) if denominator is 0."""
    x = np.asarray(x, float)
    df = pd.DataFrame({"x": x, "g": groups})
    st = df.groupby("g")["x"].agg(["size", "mean"])
    N, k = len(x), len(st)
    m = x.mean()
    ssb = float((st["size"] * (st["mean"] - m) ** 2).sum())
    ssw = float(((df["x"] - df["g"].map(st["mean"])) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (st["size"] ** 2).sum() / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    defined = bool(den > 0)
    return {"N": N, "n_groups": k, "n0": float(n0), "MSB": msb, "MSW": msw,
            "icc1": float((msb - msw) / den) if defined else np.nan, "icc_defined": defined,
            "n_per_group": ";".join(f"{g}:{int(n)}" for g, n in st["size"].items())}


def boot_compare(preds, family, scheme, a, b):
    """McNemar + OR (robustness_common.mcnemar_pair) + student-level bootstrap of b - a
    (same resampling as robustness_common.paired_bootstrap_diff), also returning the bootstrap SE."""
    row = mcnemar_pair(preds, family, scheme, a, b)
    m = align_two(preds, family, scheme, a, b)
    ids, y = m["id_student"].to_numpy(), m["y_true_a"].to_numpy()
    ca = (m["y_pred_a"].to_numpy() == y).astype(float)
    cb = (m["y_pred_b"].to_numpy() == y).astype(float)
    pa, pb = m["y_prob_a"].to_numpy(), m["y_prob_b"].to_numpy()
    _, inv = np.unique(ids, return_inverse=True)
    nc = inv.max() + 1
    rng = np.random.default_rng(RANDOM_SEED)
    da, du = np.empty(NBOOT), np.empty(NBOOT)
    for i in range(NBOOT):
        w = cluster_weights(rng, inv, nc)
        da[i] = np.dot(w, cb) / w.sum() - np.dot(w, ca) / w.sum()
        du[i] = safe_auc(y, pb, w) - safe_auc(y, pa, w)
    row.update({"n_students": int(nc), "n_boot": NBOOT, "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
                "delta_acc": float(cb.mean() - ca.mean()),
                "delta_acc_ci_low": float(np.percentile(da, 2.5)), "delta_acc_ci_high": float(np.percentile(da, 97.5)),
                "delta_acc_boot_se": float(da.std(ddof=1)),
                "auc_a": safe_auc(y, pa), "auc_b": safe_auc(y, pb), "delta_auc": safe_auc(y, pb) - safe_auc(y, pa),
                "delta_auc_ci_low": float(np.nanpercentile(du, 2.5)),
                "delta_auc_ci_high": float(np.nanpercentile(du, 97.5)),
                "delta_auc_boot_nan": int(np.isnan(du).sum())})
    return row


def repro_check(new, old, new_model, old_model, family, labels):
    a = new[(new["family"] == family) & (new["model"] == new_model)]
    b = old[(old["family"] == family) & (old["model"] == old_model)]
    m = a.merge(b, on=KEY, suffixes=("_n", "_o"))
    return {**labels, "family": family, "new_model": new_model, "earlier_model": old_model, "n_new": len(a),
            "n_earlier": len(b), "n_matched": len(m), "y_pred_identical": int((m["y_pred_n"] == m["y_pred_o"]).sum()),
            "max_abs_prob_diff": float((m["y_prob_n"] - m["y_prob_o"]).abs().max()) if len(m) else np.nan}

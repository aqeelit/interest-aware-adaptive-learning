"""
Final round, Step 3 helper (2026-10-14): INDEPENDENT re-derivation of the numbers in
final_paper_numbers.csv from the raw out-of-fold prediction files.

Deliberately imports NOTHING from the project (no robustness_common, ci_common, statsmodels):
plain pandas / numpy / scipy. Re-implemented from their written definitions:
  accuracy          mean(y_pred == y_true)
  paired difference predictions of a and b inner-joined on (id_student, code_module, code_presentation)
  McNemar           exact two-sided binomial (p = 0.5) if discordant < 25, else chi-square with
                    continuity correction (|b - c| - 1)^2 / (b + c), 1 df
  BH                Benjamini-Hochberg step-up adjusted p-values over the stated family
  bootstrap CI      student-level cluster bootstrap, 1000 resamples, numpy default_rng(42): per resample
                    draw n_students indices uniformly with replacement, weight = times drawn;
                    95% percentile interval
  FNR               among target = 0, share predicted 1; gap = FNR_F - FNR_M (gender from studentInfo.csv)
"""
import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

K = ["id_student", "code_module", "code_presentation"]
STATS = "/workspace/results/statistical_tests"
_cache = {}


def preds(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(f"{STATS}/{name}", comment="#")
    return _cache[name]


def select(name, **flt):
    d = preds(name)
    m = np.ones(len(d), bool)
    for k, v in flt.items():
        m &= (d[k] == v).to_numpy()
    return d[m]


def accuracy(sources, model, **flt):
    d = pd.concat([select(s, model=model, **flt) for s in sources])
    return float((d["y_pred"] == d["y_true"]).mean()), len(d)


def pair(sources, a, b, **flt):
    da = pd.concat([select(s, model=a, **flt) for s in sources])
    db = pd.concat([select(s, model=b, **flt) for s in sources])
    m = da[K + ["y_true", "y_pred"]].merge(db[K + ["y_true", "y_pred"]], on=K, suffixes=("_a", "_b"))
    assert (m["y_true_a"] == m["y_true_b"]).all()
    return m


def mcnemar_p(m):
    ca = m["y_pred_a"] == m["y_true_a"]
    cb = m["y_pred_b"] == m["y_true_a"]
    a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = a_only + b_only
    if n < 25:
        return float(min(1.0, 2 * binom.cdf(min(a_only, b_only), n, 0.5)))
    return float(chi2.sf((abs(a_only - b_only) - 1) ** 2 / n, 1))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(adj, 1.0)
    return out


def boot_ci(ids, stat_fn, n_boot=1000, seed=42):
    _, inv = np.unique(ids, return_inverse=True)
    nc = inv.max() + 1
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, nc, nc), minlength=nc)[inv].astype(float)
        vals[i] = stat_fn(w)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def acc_diff(sources, a, b, **flt):
    m = pair(sources, a, b, **flt)
    ca = (m["y_pred_a"] == m["y_true_a"]).to_numpy(float)
    cb = (m["y_pred_b"] == m["y_true_a"]).to_numpy(float)
    lo, hi = boot_ci(m["id_student"].to_numpy(), lambda w: np.dot(w, cb) / w.sum() - np.dot(w, ca) / w.sum())
    return float(cb.mean() - ca.mean()), lo, hi, len(m), m


def family_bh(sources, members, target):
    """members: list of (a, b, filter dict) forming one BH family. Re-derives every raw McNemar p
    from predictions, then BH; returns the adjusted p of members[target]."""
    ps = [mcnemar_p(pair(sources, a, b, **flt)) for a, b, flt in members]
    return float(bh(ps)[target])


def fnr_gap(source, model, family, gender_df, with_ci=True):
    g = select(source, model=model, family=family).merge(gender_df, on=K, how="left")
    assert g["gender"].isin(["F", "M"]).all()
    y, pr, f = g["y_true"].to_numpy(), g["y_pred"].to_numpy(), (g["gender"] == "F").to_numpy()
    fail = (y == 0)
    ff, mf = (fail & f).astype(float), (fail & f & (pr == 1)).astype(float)
    fm, mm = (fail & ~f).astype(float), (fail & ~f & (pr == 1)).astype(float)

    def gap(w):
        return np.dot(w, mf) / np.dot(w, ff) - np.dot(w, mm) / np.dot(w, fm)
    point = float(gap(np.ones(len(g))))
    lo, hi = boot_ci(g["id_student"].to_numpy(), gap) if with_ci else (np.nan, np.nan)
    return point, lo, hi, len(g)


def course_passrates(source, **flt):
    d = select(source, **flt)
    return d.groupby(["code_module", "code_presentation"])["y_true"].mean(), len(d)


def lopo_course_diffs(source, a, b, family):
    m = pair([source], a, b, family=family)
    m["course"] = m["code_module"] + "_" + m["code_presentation"]
    m["ca"] = (m["y_pred_a"] == m["y_true_a"]).astype(float)
    m["cb"] = (m["y_pred_b"] == m["y_true_a"]).astype(float)
    g = m.groupby("course")[["ca", "cb"]].mean()
    return (g["cb"] - g["ca"]), len(m)

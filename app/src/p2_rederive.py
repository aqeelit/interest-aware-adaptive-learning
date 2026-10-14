"""
Paper 2, Step 2 helper (2026-10-14): INDEPENDENT re-derivation from raw out-of-fold prediction files.
Imports NOTHING from the project (no robustness_common / ci_common / ml_common / statsmodels): pandas, numpy,
scipy only. Definitions re-implemented from their written form:
  accuracy        mean(y_pred == y_true)
  pairing         predictions of a and b inner-joined on (id_student, code_module, code_presentation);
                  models with different row sets are compared on their common rows
  McNemar         exact two-sided binomial (p = 0.5) if discordant < 25; else chi-square with continuity
                  correction (|b - c| - 1)^2 / (b + c), 1 df
  BH              Benjamini-Hochberg step-up adjusted p over the stated family
  bootstrap       student-level cluster bootstrap, 1000 resamples, numpy default_rng(42): per resample draw
                  n_students indices with replacement; weight = times drawn; 95% percentile CI; SE = SD (ddof 1)
  DL / HK         DerSimonian-Laird random effects; Hartung-Knapp CI with t(k - 1)
  FNR gap         among target = 0, share predicted 1, female minus male
"""
import numpy as np
import pandas as pd
from scipy.stats import binom, chi2, norm, t as tdist

K = ["id_student", "code_module", "code_presentation"]
STATS = "/workspace/results/statistical_tests"
_cache = {}


def preds(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(f"{STATS}/{name}", comment="#", low_memory=False)
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
    assert not da.duplicated(K).any() and not db.duplicated(K).any(), "duplicate keys"  # inner join = common rows
    assert (m["y_true_a"] == m["y_true_b"]).all()
    return m


def mcnemar_p(m):
    ca = m["y_pred_a"] == m["y_true_a"]
    cb = m["y_pred_b"] == m["y_true_a"]
    a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = a_only + b_only
    if n == 0:
        return 1.0
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


def boot(ids, stat_fn, n_boot=1000, seed=42):
    _, inv = np.unique(ids, return_inverse=True)
    nc = inv.max() + 1
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, nc, nc), minlength=nc)[inv].astype(float)
        vals[i] = stat_fn(w)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.std(ddof=1))


def acc_diff(sources, a, b, **flt):
    m = pair(sources, a, b, **flt)
    ca = (m["y_pred_a"] == m["y_true_a"]).to_numpy(float)
    cb = (m["y_pred_b"] == m["y_true_a"]).to_numpy(float)
    lo, hi, se = boot(m["id_student"].to_numpy(), lambda w: np.dot(w, cb) / w.sum() - np.dot(w, ca) / w.sum())
    return {"value": float(cb.mean() - ca.mean()), "ci_low": lo, "ci_high": hi, "se": se, "N": len(m),
            "p_raw": mcnemar_p(m)}


def family_bh(sources, members, target):
    """members: list of (a, b, filter dict); raw McNemar p re-derived for every member, then BH."""
    return float(bh([mcnemar_p(pair(sources, a, b, **f)) for a, b, f in members])[target])


def dl_hk(y, se):
    y, se = np.asarray(y, float), np.asarray(se, float)
    k = len(y)
    w = 1 / se ** 2
    mu_f = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - mu_f) ** 2))
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (se ** 2 + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    se_dl = float(np.sqrt(1 / np.sum(ws)))
    se_hk = float(np.sqrt(np.sum(ws * (y - mu) ** 2) / (k - 1) / np.sum(ws)))
    z, tc = norm.ppf(0.975), tdist.ppf(0.975, k - 1)
    return {"pooled": mu, "ci_low_dl": mu - z * se_dl, "ci_high_dl": mu + z * se_dl, "tau2": tau2,
            "I2": max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0, "ci_low_hk": mu - tc * se_hk,
            "ci_high_hk": mu + tc * se_hk}


def fnr_gap(source, model, family, gender_df):
    g = select(source, model=model, family=family).merge(gender_df, on=K, how="left")
    assert g["gender"].isin(["F", "M"]).all()
    y, pr, f = g["y_true"].to_numpy(), g["y_pred"].to_numpy(), (g["gender"] == "F").to_numpy()
    fail = y == 0
    ff, mf = (fail & f).astype(float), (fail & f & (pr == 1)).astype(float)
    fm, mm = (fail & ~f).astype(float), (fail & ~f & (pr == 1)).astype(float)

    def gap(w):
        return np.dot(w, mf) / np.dot(w, ff) - np.dot(w, mm) / np.dot(w, fm)
    lo, hi, _ = boot(g["id_student"].to_numpy(), gap)
    return {"value": float(gap(np.ones(len(g)))), "ci_low": lo, "ci_high": hi, "N": len(g)}

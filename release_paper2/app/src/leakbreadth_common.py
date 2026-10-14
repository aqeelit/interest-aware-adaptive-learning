"""
Leak-audit / breadth round (2026-10-13) — shared pipeline-side helpers.

Pre-registration: results/statistical_tests/prereg_leak_breadth_2026-10-13.md.

Pipeline-side EARLY features come from the EXISTING code path
(wdearly_common.build_sample, imported unchanged). For the provenance audit the
three feature builders it calls are wrapped at runtime (module attributes swapped
for the duration of one call, then restored) so that the raw records actually
passed into them can be inspected; the wrappers pass inputs through unchanged.
New breadth columns (n_activity_types_used, n_unique_resources) are computed here
from the same truncated studentVle records.

Every output written by this round carries explicit setting / feature_family /
scheme columns set AFTER any sample-level metadata, so no label is overwritten.
"""
import os
import numpy as np
import pandas as pd

import wdearly_common as W
from utils import RESULTS_STATS
from train_evaluate_oulad import KEY
from robustness_common import run_cv, summarize
from ci_common import guard_absent

FAMILIES = ["LR", "GBM"]
CUTOFFS = [28, 56]
BREADTH2 = ["resource_diversity_entropy", "n_activity_types_used"]
BREADTH3 = BREADTH2 + ["n_unique_resources"]
LOPO_NOTE = ("LOPO: no course one-hot; centering within each presentation uses only that presentation's own "
             "students (for the held-out course: only its own students), features only, no labels.")
STRAT_NOTE = ("Stratified 5-fold: centering within each presentation uses that presentation's risk-set students "
              "(features only, no labels; includes test-fold rows).")


def write_csv(df: pd.DataFrame, name: str, header_lines=()):
    assert name.startswith(("leak_", "breadth_")), f"output name must start with leak_ or breadth_: {name}"
    path = guard_absent(os.path.join(RESULTS_STATS, name))
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)
    print(f"wrote {path} ({len(df)} rows)", flush=True)


def breadth_features(vle_t: pd.DataFrame) -> pd.DataFrame:
    by_type = vle_t.groupby(KEY + ["activity_type"])["sum_click"].sum()
    n_types = (by_type > 0).groupby(level=KEY).sum().rename("n_activity_types_used")
    by_site = vle_t.groupby(KEY + ["id_site"])["sum_click"].sum()
    n_sites = (by_site > 0).groupby(level=KEY).sum().rename("n_unique_resources")
    return pd.concat([n_types, n_sites], axis=1).reset_index()


def _per_student_max(df, col):
    if len(df) == 0:
        return pd.Series(dtype=float)
    return df.groupby(KEY)[col].max()


def build_early(t: int, data, instrument: bool = False):
    """Existing-path EARLY sample at cutoff t + breadth columns. If instrument, also returns
    provenance records: one per builder call with the per-student max raw timestamp."""
    calls = []
    if instrument:
        orig = {n: getattr(W, n) for n in ["build_feature_set_B_performance", "build_feature_set_C_behavior",
                                           "build_feature_set_D_interest"]}
        spec = {"build_feature_set_B_performance": ("B_assessment", "date_submitted"),
                "build_feature_set_C_behavior": ("C_behaviour", "date"),
                "build_feature_set_D_interest": ("I_interest", "date")}

        def wrap(name):
            fn = orig[name]
            group, col = spec[name]

            def inner(df, *a, **k):
                calls.append({"builder": name, "group": group, "time_col": col, "n_input_rows": len(df),
                              "per_student_max": _per_student_max(df, col)})
                return fn(df, *a, **k)
            return inner
        for n in orig:
            setattr(W, n, wrap(n))
    try:
        s = W.build_sample("EARLY", cutoff=t, data=data)
    finally:
        if instrument:
            for n, fn in orig.items():
                setattr(W, n, fn)

    _, vle, _ = data
    vt = vle[vle["date"] <= t]
    br = breadth_features(vt)
    if instrument:
        calls.append({"builder": "leakbreadth_common.breadth_features", "group": "breadth", "time_col": "date",
                      "n_input_rows": len(vt), "per_student_max": _per_student_max(vt, "date")})
    frame = s["frame"].merge(br, on=KEY, how="left")
    assert len(frame) == len(s["frame"])
    frame[["n_activity_types_used", "n_unique_resources"]] = \
        frame[["n_activity_types_used", "n_unique_resources"]].fillna(0.0)
    s["frame"] = frame
    s["breadth"] = ["n_activity_types_used", "n_unique_resources"]
    return s, calls


def course_info(frame):
    course = (frame["code_module"] + "_" + frame["code_presentation"]).to_numpy()
    oh = pd.get_dummies(pd.Series(course), dtype=float)
    return course, oh[sorted(oh.columns)].to_numpy()


def fit(frame, variants, folds, scheme, labels: dict):
    y = frame["target"].to_numpy()
    preds_l, folds_l = [], []
    for name, X in variants.items():
        for fam in FAMILIES:
            p, fd = run_cv(frame, X, y, folds, fam, name, scheme)
            preds_l.append(p)
            folds_l.append(fd)
    preds = pd.concat(preds_l, ignore_index=True)
    fold_df = pd.concat(folds_l, ignore_index=True)
    summ = summarize(preds, fold_df)
    summ["n_features"] = summ["model"].map({k: v.shape[1] for k, v in variants.items()})
    for df in (preds, fold_df, summ):
        for k, v in labels.items():
            df[k] = v
    return preds, fold_df, summ


def criterion(c: pd.DataFrame, keys=("setting", "feature_family", "scheme", "model_a", "model_b")):
    """Pre-registered criterion (identical definition to wdearly_common.criterion), grouped by full keys."""
    c = c.copy()
    c["ci_excludes_0"] = (c["boot_acc_diff_ci_low"] > 0) | (c["boot_acc_diff_ci_high"] < 0)
    c["dir_acc"] = np.sign(c["boot_acc_diff_b_minus_a"]).astype(int)
    c["dir_mcnemar"] = np.sign(c["b_only_correct"] - c["a_only_correct"]).astype(int)
    c["bh_p_lt_0.05"] = c["p_value_bh"] < 0.05
    c["family_condition_met"] = (c["ci_excludes_0"] & c["bh_p_lt_0.05"] & (c["dir_acc"] == c["dir_mcnemar"])
                                 & (c["dir_acc"] != 0))
    met = {}
    for k, g in c.groupby(list(keys), sort=False):
        g = g.set_index("family")
        met[k] = bool(g.loc["LR", "family_condition_met"] and g.loc["GBM", "family_condition_met"]
                      and g.loc["LR", "dir_acc"] == g.loc["GBM", "dir_acc"])
    c["criterion_met"] = [met[tuple(r)] for r in c[list(keys)].itertuples(index=False)]
    return c

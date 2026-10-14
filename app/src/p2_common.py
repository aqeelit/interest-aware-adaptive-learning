"""
Paper 2 round (p2_, 2026-10-14; EXPLORATORY / POST HOC) — shared helpers.
Pre-registration: results/statistical_tests/p2_prereg_2026-10-14.md.

Imports read-only: ci_common.load_base (FULL D/E row set), wdearly_common.build_sample (EARLY risk sets),
ml_common.boot_compare / center_and_mean (key-aligned McNemar + OR + student-level bootstrap).
Smoke mode (env P2_SMOKE=1): subsample, 20 resamples, outputs to /tmp/p2_smoke, no experiment_log.
"""
import os
import warnings
import numpy as np
import pandas as pd

import wdearly_common as W
from utils import RESULTS_STATS
from ci_common import load_base
from robustness_common import log_summary

SMOKE = os.environ.get("P2_SMOKE") == "1"
OUT = "/tmp/p2_smoke" if SMOKE else RESULTS_STATS
EXPLORATORY = "EXPLORATORY / POST HOC (p2_prereg_2026-10-14.md); hypothesis-generating only."
NORM_NOTE = ("Within-presentation z-scoring, course centering and course means use only the students of that "
             "presentation in the analysed sample; features only, no labels.")


def write_csv(df, name, header_lines=(), outdir=None):
    assert name.startswith("p2_"), name
    d = outdir or OUT
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    if not SMOKE:
        assert not os.path.exists(path), f"Refusing to overwrite {path}"
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in [EXPLORATORY, *header_lines]:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)
    print(f"wrote {path} ({len(df)} rows)", flush=True)
    return path


def log(summary, tag, extra):
    if SMOKE:
        print(f"[smoke] skipped experiment_log for {tag}", flush=True)
        return
    log_summary(summary, f"P2-{tag}", extra)


def load(setting, data):
    """setting in FULL, EARLY_t{28,56}_{VLEonly,withAssess}. Returns frame, Xb, base_cols, I, course, rowinfo."""
    if setting == "FULL":
        b = load_base()
        e_cols = [c for c in b["frame"].columns if c not in b["interest_cols"]
                  and c not in ["id_student", "code_module", "code_presentation", "final_result", "target"]]
        assert b["X_E"].shape[1] == len(e_cols)
        frame, Xb, I = b["frame"], b["X_E"], b["interest"]
        ri = {"n_source_rows": b["rowinfo"]["n_target_rows"], "n_rows_used": b["rowinfo"]["n_rows_used"],
              "n_dropped_rows": b["rowinfo"]["n_rows_dropped_by_feature_inner_join"],
              "dropped_reason": "no complete B/C/D feature set (inner join; D/E row set)"}
        cols = e_cols
    else:
        t = int(setting.split("_")[1][1:])
        fam = setting.split("_")[2]
        s = W.build_sample("EARLY", cutoff=t, data=data)
        frame = s["frame"]
        cols = s["C"] if fam == "VLEonly" else s["B"] + s["C"]
        Xb, I = frame[cols].to_numpy(float), frame[s["I"]].to_numpy(float)
        ri = {"n_source_rows": s["rowinfo"]["n_studentinfo_rows"], "n_rows_used": s["rowinfo"]["n_rows_used"],
              "n_dropped_rows": s["rowinfo"]["n_studentinfo_rows"] - s["rowinfo"]["n_rows_used"],
              "dropped_reason": f"unregistered by t={t} (risk set); 0 dropped by feature build"}
    if SMOKE:
        idx = np.sort(np.random.default_rng(0).choice(len(frame), 3000, replace=False))
        frame, Xb, I = frame.iloc[idx].reset_index(drop=True), Xb[idx], I[idx]
        ri = {**ri, "n_rows_used": len(frame), "dropped_reason": "SMOKE subsample"}
    course = (frame["code_module"] + "_" + frame["code_presentation"]).to_numpy()
    return {"frame": frame, "y": frame["target"].to_numpy(), "Xb": Xb, "base_cols": list(cols), "I": I,
            "course": course, "rowinfo": ri}


def zscore_within(arr, course):
    out = np.full_like(arr, np.nan, dtype=float)
    for g in sorted(set(course)):
        idx = np.flatnonzero(course == g)
        sub = arr[idx]
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column inside a presentation
            mu = np.nanmean(sub, axis=0)
            sd = np.nanstd(sub, axis=0)
            z = (sub - mu) / sd
        z[:, ~(sd > 0)] = np.where(np.isnan(sub[:, ~(sd > 0)]), np.nan, 0.0)
        out[idx] = z
    return out

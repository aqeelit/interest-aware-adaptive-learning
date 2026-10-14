"""
WD / EARLY round (2026-10-13) — shared sample/feature builders and comparison helpers.

Pre-registration: results/statistical_tests/prereg_wd_early_2026-10-13.md.

Features are REBUILT here from the cleaned parquets with the original builder
functions of feature_engineering_oulad.py (imported, not modified), so the same
code path serves the full-course settings and the cutoff-truncated settings.
Models, folds, McNemar, OR, bootstrap and BH come from robustness_common.py and
ci_common.py (imported, not modified).
"""
import os
import numpy as np
import pandas as pd

from utils import DATA_PROCESSED, OULAD_RAW_DIR, RESULTS_STATS
from train_evaluate_oulad import KEY
from feature_engineering_oulad import (build_target, build_feature_set_B_performance,
                                       build_feature_set_C_behavior, build_feature_set_D_interest)
from robustness_common import run_cv, summarize, add_bh, stratified_folds, course_folds, log_summary
from ci_common import center_within_course, shuffle_within_course, compare, guard_absent

FAMILIES = ["LR", "GBM"]
STRAT_NOTE = ("Interest centering: within each code_module+code_presentation, subtract the mean over that "
              "presentation's students in the analysed sample (features only, no labels; includes test-fold rows).")
LOPO_NOTE = ("Interest centering: within each code_module+code_presentation using only that presentation's own "
             "students (for the held-out course: only the held-out course's students), features only, no labels.")
ASSESS_NOTE = ("ASSUMPTION: assessment scores are treated as observable at date_submitted; OULAD has no "
               "grade-release time.")
EARLY_FILL_NOTE = ("ASSUMPTION (pre-registered): risk-set students with no clicks by t get click_share = 0, "
                   "entropy = 0 and zero behaviour counts; students with no scored submission by t are kept with "
                   "n_assessments = 0, score_std = 0 and missing mean scores (median-imputed in LR, native NaN in GBM).")


def write_csv(df: pd.DataFrame, name: str, header_lines=()):
    assert name.startswith(("wd_", "early_")), f"output name must start with wd_ or early_: {name}"
    path = guard_absent(os.path.join(RESULTS_STATS, name))
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)
    print(f"wrote {path} ({len(df)} rows)", flush=True)


def load_clean():
    info = pd.read_parquet(os.path.join(DATA_PROCESSED, "oulad_student_info_clean.parquet"))
    vle = pd.read_parquet(os.path.join(DATA_PROCESSED, "oulad_student_vle_clean.parquet"))
    assess = pd.read_parquet(os.path.join(DATA_PROCESSED, "oulad_student_assessment_clean.parquet"))
    return info, vle, assess


def full_interest_columns(vle) -> list:
    return [c for c in build_feature_set_D_interest(vle).columns if c not in KEY]


def build_sample(setting: str, cutoff=None, sens=False, data=None):
    """Returns dict(frame, B, C, I column lists, rowinfo).
    setting: 'FULL' (all final_result), 'WD' (Withdrawn excluded), 'EARLY' (risk set at cutoff)."""
    info, vle, assess = data if data is not None else load_clean()
    target = build_target(info)
    final = info[KEY + ["final_result"]]
    rowinfo = {"setting": setting, "cutoff_days": cutoff, "sens": sens, "n_studentinfo_rows": len(target)}

    if setting == "WD":
        keep = target.merge(final, on=KEY, how="left")["final_result"].isin(["Pass", "Distinction", "Fail"]).to_numpy()
        rowinfo["n_excluded_withdrawn"] = int((~keep).sum())
        target = target[keep].reset_index(drop=True)
    elif setting == "EARLY":
        reg = pd.read_csv(os.path.join(OULAD_RAW_DIR, "studentRegistration.csv"))
        assert not reg.duplicated(KEY).any()
        t = target.merge(reg[KEY + ["date_unregistration"]], on=KEY, how="left")
        assert len(t) == len(target)
        excl = (t["date_unregistration"] <= cutoff).to_numpy()  # NaN -> False
        fr = target.merge(final, on=KEY, how="left")["final_result"]
        rowinfo["n_excluded_unregistered_by_t"] = int(excl.sum())
        rowinfo["n_excluded_unregistered_by_t_final_withdrawn"] = int((excl & (fr == "Withdrawn")).to_numpy().sum())
        target = target[~excl].reset_index(drop=True)
    rowinfo["n_sample_before_features"] = len(target)

    interest_cols = full_interest_columns(vle)
    if cutoff is None:
        perf = build_feature_set_B_performance(assess, exclude_final=sens)
        beh = build_feature_set_C_behavior(vle)
        intr = build_feature_set_D_interest(vle)
        fused = perf.merge(beh, on=KEY, how="inner").merge(intr, on=KEY, how="inner")
        frame = target.merge(fused, on=KEY, how="inner")
        rowinfo["n_dropped_by_feature_inner_join"] = len(target) - len(frame)
    else:
        vt = vle[vle["date"] <= cutoff]
        at = assess[assess["date_submitted"] <= cutoff]
        perf = build_feature_set_B_performance(at, exclude_final=sens)
        beh = build_feature_set_C_behavior(vt)
        intr = build_feature_set_D_interest(vt)
        intr = intr.reindex(columns=KEY + interest_cols)  # fix the 21 columns; absent types filled below
        frame = target.merge(perf, on=KEY, how="left").merge(beh, on=KEY, how="left").merge(intr, on=KEY, how="left")
        rowinfo["n_no_scored_submission_by_t"] = int(frame["n_assessments"].isna().sum())
        rowinfo["n_no_clicks_by_t"] = int(frame["total_clicks"].isna().sum())
        frame[interest_cols] = frame[interest_cols].fillna(0.0)
        frame[["total_clicks", "active_days", "mean_clicks_per_active_day"]] = \
            frame[["total_clicks", "active_days", "mean_clicks_per_active_day"]].fillna(0.0)
        frame[["n_assessments", "score_std"]] = frame[["n_assessments", "score_std"]].fillna(0.0)
        rowinfo["n_dropped_by_feature_inner_join"] = 0
    b_cols = [c for c in perf.columns if c not in KEY]
    c_cols = [c for c in beh.columns if c not in KEY]
    assert len(interest_cols) == 21, interest_cols
    rowinfo["n_rows_used"] = len(frame)
    rowinfo["pass_rate"] = float(frame["target"].mean())
    rowinfo["n_courses"] = int(frame.groupby(["code_module", "code_presentation"]).ngroups)
    return {"frame": frame.reset_index(drop=True), "B": b_cols, "C": c_cols, "I": interest_cols, "rowinfo": rowinfo}


def course_table(sample, label):
    f = sample["frame"]
    g = f.groupby(["code_module", "code_presentation"])["target"].agg(n="size", pass_rate="mean").reset_index()
    g.insert(0, "sample", label)
    return g


def design(sample, base_cols, base_name, plus_name):
    """Feature matrices for the six stratified models of one family."""
    f = sample["frame"]
    course = (f["code_module"] + "_" + f["code_presentation"]).to_numpy()
    oh = pd.get_dummies(pd.Series(course), dtype=float)
    oh = oh[sorted(oh.columns)].to_numpy()
    Xb = f[base_cols].to_numpy(dtype=float)
    I = f[sample["I"]].to_numpy(dtype=float)
    ctr = center_within_course(I, course)
    ctrshuf, n_unch = shuffle_within_course(ctr, course)
    return {
        base_name: Xb,
        plus_name: np.hstack([Xb, I]),
        f"{base_name}_course": np.hstack([Xb, oh]),
        f"{plus_name}_course": np.hstack([Xb, I, oh]),
        f"{base_name}_course_ctr": np.hstack([Xb, oh, ctr]),
        f"{base_name}_course_ctrshuf": np.hstack([Xb, oh, ctrshuf]),
    }, n_unch


def lopo_design(sample, base_cols, base_name):
    f = sample["frame"]
    course = (f["code_module"] + "_" + f["code_presentation"]).to_numpy()
    Xb = f[base_cols].to_numpy(dtype=float)
    ctr = center_within_course(f[sample["I"]].to_numpy(dtype=float), course)
    ctrshuf, n_unch = shuffle_within_course(ctr, course)
    return {base_name: Xb, f"{base_name}_ctr": np.hstack([Xb, ctr]),
            f"{base_name}_ctrshuf": np.hstack([Xb, ctrshuf])}, n_unch


def fit_all(sample, variants, folds, scheme, setting_label):
    f = sample["frame"]
    y = f["target"].to_numpy()
    preds_l, folds_l = [], []
    for name, X in variants.items():
        for fam in FAMILIES:
            p, fd = run_cv(f, X, y, folds, fam, name, scheme)
            p.insert(0, "setting", setting_label)
            fd.insert(0, "setting", setting_label)
            preds_l.append(p)
            folds_l.append(fd)
    preds = pd.concat(preds_l, ignore_index=True)
    fold_df = pd.concat(folds_l, ignore_index=True)
    summ = summarize(preds.drop(columns=["setting"]), fold_df)
    summ.insert(0, "setting", setting_label)
    summ["n_features"] = summ["model"].map({k: v.shape[1] for k, v in variants.items()})
    for k, v in sample["rowinfo"].items():
        summ[k] = v
    return preds, fold_df, summ


def run_comparisons(preds, scheme, comps, setting_label):
    rows = []
    for fam in FAMILIES:
        for a, b in comps:
            r = compare(preds.drop(columns=["setting"]), fam, scheme, a, b)
            r["setting"] = setting_label
            rows.append(r)
    df = pd.DataFrame(rows)
    return df[["setting"] + [c for c in df.columns if c != "setting"]]


def criterion(comps_bh: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered criterion, per (setting, model_a, model_b) across LR and GBM."""
    c = comps_bh.copy()
    lo, hi = c["boot_acc_diff_ci_low"], c["boot_acc_diff_ci_high"]
    c["ci_excludes_0"] = (lo > 0) | (hi < 0)
    c["dir_acc"] = np.sign(c["boot_acc_diff_b_minus_a"]).astype(int)
    c["dir_mcnemar"] = np.sign(c["b_only_correct"] - c["a_only_correct"]).astype(int)
    c["bh_p_lt_0.05"] = c["p_value_bh"] < 0.05
    c["family_condition_met"] = c["ci_excludes_0"] & c["bh_p_lt_0.05"] & (c["dir_acc"] == c["dir_mcnemar"]) & (c["dir_acc"] != 0)
    rows = []
    for (s, a, b), g in c.groupby(["setting", "model_a", "model_b"], sort=False):
        g = g.set_index("family")
        met = bool(g.loc["LR", "family_condition_met"] and g.loc["GBM", "family_condition_met"]
                   and g.loc["LR", "dir_acc"] == g.loc["GBM", "dir_acc"])
        rows.append({"setting": s, "model_a": a, "model_b": b, "criterion_met": met})
    return c, pd.DataFrame(rows)


def per_course(preds, setting_label, base, plus):
    """Per held-out presentation: base vs plus accuracy, and summary with Spearman (n = courses)."""
    from scipy.stats import spearmanr
    p = preds[preds["model"].isin([base, plus])].copy()
    p["correct"] = (p["y_pred"] == p["y_true"]).astype(int)
    rows, summ = [], []
    for fam in FAMILIES:
        w = p[p["family"] == fam].pivot_table(index=KEY + ["y_true"], columns="model", values="correct").reset_index()
        assert w[[base, plus]].notna().all().all()
        w["course"] = w["code_module"] + "_" + w["code_presentation"]
        fr = []
        for c, g in w.groupby("course"):
            fr.append({"setting": setting_label, "family": fam, "course": c, "n_rows": len(g),
                       "n_students": g["id_student"].nunique(), "pass_rate": g["y_true"].mean(),
                       "model_base": base, "model_with_interest": plus,
                       "acc_base": g[base].mean(), "acc_with_interest": g[plus].mean(),
                       "diff_with_interest_minus_base": g[plus].mean() - g[base].mean()})
        r = pd.DataFrame(fr)
        rho, pv = spearmanr(r["diff_with_interest_minus_base"], r["pass_rate"])
        d = r["diff_with_interest_minus_base"]
        summ.append({"setting": setting_label, "family": fam, "n_courses": len(r), "n_rows_total": int(r["n_rows"].sum()),
                     "n_positive_diff": int((d > 0).sum()), "n_zero_diff": int((d == 0).sum()),
                     "n_negative_diff": int((d < 0).sum()), "mean_diff_unweighted": d.mean(),
                     "sd_diff_unweighted_ddof1": d.std(ddof=1),
                     "pooled_diff": w[plus].mean() - w[base].mean(),
                     "spearman_rho_diff_vs_passrate": rho, "spearman_p": pv, "spearman_n": len(r)})
        rows.extend(fr)
    return pd.DataFrame(rows), pd.DataFrame(summ)

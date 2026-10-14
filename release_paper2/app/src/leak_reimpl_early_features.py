"""
Independent from-scratch re-implementation of the EARLY features (2026-10-13).

Deliberately imports NOTHING from the project (no utils, no feature builders):
plain pandas/numpy on the RAW OULAD CSVs. Raw tables are filtered to <= t FIRST,
then cleaned with the documented rules, then aggregated.

Rules re-implemented from their written definitions:
  risk set   studentInfo, de-duplicated on (id_student, code_module, code_presentation),
             rows with missing final_result dropped; exclude date_unregistration <= t
             (missing date_unregistration = not unregistered). target = Pass/Distinction.
  VLE        studentVle with date <= t; joined to vle.csv activity_type on
             (id_site, code_module, code_presentation); rows without activity_type and
             rows with sum_click < 0 dropped.
  scores     studentAssessment with date_submitted <= t; joined to assessments.csv
             (code_module, code_presentation, weight) on id_assessment; missing scores and
             scores outside [0, 100] dropped.
  B          mean_score; weighted_mean_score (weights = assessment weight, 0 -> 1);
             n_assessments; score_std (sample SD, 0 when undefined). No scored submission:
             n_assessments = 0, score_std = 0, means missing.
  C          total_clicks = sum(sum_click); active_days = distinct dates;
             mean_clicks_per_active_day = mean of sum_click over records (as defined in the
             pipeline despite its name). No clicks: all 0.
  interest   click_share_<type> = clicks on type / total clicks (0 if total is 0), over the
             activity types that occur in the full studentVle-vle join (column labels only);
             resource_diversity_entropy = -sum p ln p over p > 0. No clicks: all 0.
  breadth    n_activity_types_used = #types with summed clicks > 0;
             n_unique_resources = #id_site with summed clicks > 0. No clicks: 0.
"""
import os
import numpy as np
import pandas as pd

RAW = "/workspace/data/raw/oulad"
K = ["id_student", "code_module", "code_presentation"]


def load_raw():
    r = {n: pd.read_csv(os.path.join(RAW, f"{n}.csv")) for n in
         ["studentInfo", "studentRegistration", "studentVle", "vle", "studentAssessment", "assessments"]}
    return r


def activity_type_labels(r):
    j = r["studentVle"].merge(r["vle"][["id_site", "code_module", "code_presentation", "activity_type"]],
                              on=["id_site", "code_module", "code_presentation"], how="inner")
    j = j[j["activity_type"].notna() & (j["sum_click"] >= 0)]
    return sorted(j["activity_type"].unique().tolist())


def reimplement(t: int, r=None, types=None) -> pd.DataFrame:
    r = r if r is not None else load_raw()
    types = types if types is not None else activity_type_labels(r)

    info = r["studentInfo"].drop_duplicates(subset=K)
    info = info[info["final_result"].notna()]
    reg = r["studentRegistration"][K + ["date_unregistration"]]
    risk = info[K + ["final_result"]].merge(reg, on=K, how="left")
    risk = risk[~(risk["date_unregistration"].notna() & (risk["date_unregistration"] <= t))]
    out = risk[K].copy().reset_index(drop=True)
    out["target"] = risk["final_result"].isin(["Pass", "Distinction"]).astype(int).to_numpy()

    # VLE: filter to <= t FIRST
    sv = r["studentVle"]
    sv = sv[sv["date"] <= t]
    sv = sv.merge(r["vle"][["id_site", "code_module", "code_presentation", "activity_type"]],
                  on=["id_site", "code_module", "code_presentation"], how="left")
    sv = sv[sv["activity_type"].notna() & (sv["sum_click"] >= 0)]

    # Scores: filter to <= t FIRST
    sa = r["studentAssessment"]
    sa = sa[sa["date_submitted"] <= t]
    sa = sa.merge(r["assessments"][["id_assessment", "code_module", "code_presentation", "weight"]],
                  on="id_assessment", how="left")
    sa = sa[sa["score"].notna() & (sa["score"] >= 0) & (sa["score"] <= 100)]

    rows_b = {}
    for key, g in sa.groupby(K):
        sc = g["score"].to_numpy(float)
        w = g["weight"].replace(0, 1).to_numpy(float)
        rows_b[key] = (sc.mean(), float(np.sum(sc * w) / np.sum(w)), len(sc),
                       float(np.std(sc, ddof=1)) if len(sc) > 1 else 0.0)
    b = pd.DataFrame([(*k, *v) for k, v in rows_b.items()],
                     columns=K + ["mean_score", "weighted_mean_score", "n_assessments", "score_std"])

    c = sv.groupby(K).agg(total_clicks=("sum_click", "sum"), active_days=("date", "nunique"),
                          mean_clicks_per_active_day=("sum_click", "mean")).reset_index()

    by_type = sv.groupby(K + ["activity_type"])["sum_click"].sum().unstack("activity_type")
    by_type = by_type.reindex(columns=types).fillna(0.0)
    tot = by_type.sum(axis=1)
    share = by_type.div(tot.where(tot != 0, 1), axis=0)
    p = share.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0.0).sum(axis=1)
    share.columns = [f"click_share_{x}" for x in share.columns]
    share["resource_diversity_entropy"] = ent
    share["n_activity_types_used"] = (by_type > 0).sum(axis=1)
    by_site = sv.groupby(K + ["id_site"])["sum_click"].sum()
    share["n_unique_resources"] = (by_site > 0).groupby(level=K).sum()
    share = share.reset_index()

    out = out.merge(b, on=K, how="left").merge(c, on=K, how="left").merge(share, on=K, how="left")
    zero_cols = (["n_assessments", "score_std", "total_clicks", "active_days", "mean_clicks_per_active_day",
                  "resource_diversity_entropy", "n_activity_types_used", "n_unique_resources"]
                 + [f"click_share_{x}" for x in types])
    out[zero_cols] = out[zero_cols].fillna(0.0)
    return out

"""
Phase 9 — Feature engineering for OULAD, implementing the feature sets defined in
experiments/configs/experimental_design.md (Study 1: A/B/C/D), PLUS two additions
approved 2026-08-30 after inspecting real results:

  - E = B+C fused WITHOUT the interest signal. Needed to isolate H2's specific claim
    ("adding interest improves over performance+behavior alone") — D vs B and D vs C
    alone cannot show whether the gain comes from fusion in general or from interest
    specifically. Compare D vs E for that.

  - *_sens variants = same feature sets but with the performance signal computed
    EXCLUDING each student's final/exam assessment. This addresses a circularity
    concern: OULAD's final_result is partly computed FROM assessment scores, so
    predicting it from those same scores can be tautological rather than genuinely
    predictive. Excluding the final assessment tests whether the effect survives.

Input: data/processed/oulad_*_clean.parquet (produced by data_cleaning_oulad.py)
Output: data/processed/oulad_features_<name>.parquet + oulad_target.parquet

Each feature set is saved separately so every model run traces to a distinct,
inspectable file — no feature set is generated implicitly inside train_evaluate.py.
"""
import os
import pandas as pd
import numpy as np
from utils import DATA_PROCESSED, ensure_dirs

RESOURCE_TYPES = [
    "resource", "oucontent", "url", "homepage", "subpage", "glossary",
    "forumng", "oucollaborate", "dataplus", "quiz", "ouwiki", "dualpane",
    "questionnaire", "sharedsubpage", "page", "externalquiz", "ouelluminate",
    "folder", "htmlactivity",
]


def load_processed():
    info = pd.read_parquet(os.path.join(DATA_PROCESSED, "oulad_student_info_clean.parquet"))
    vle = pd.read_parquet(os.path.join(DATA_PROCESSED, "oulad_student_vle_clean.parquet"))
    assess = pd.read_parquet(os.path.join(DATA_PROCESSED, "oulad_student_assessment_clean.parquet"))
    return info, vle, assess


def build_target(info: pd.DataFrame) -> pd.DataFrame:
    """Binary target per experimental_design.md: Pass/Distinction=1, Fail/Withdrawn=0."""
    df = info[["id_student", "code_module", "code_presentation", "final_result"]].copy()
    df["target"] = df["final_result"].isin(["Pass", "Distinction"]).astype(int)
    key = ["id_student", "code_module", "code_presentation"]
    return df[key + ["target"]]


def build_feature_set_B_performance(assess: pd.DataFrame, exclude_final: bool = False) -> pd.DataFrame:
    """Performance-only: aggregate assessment-score-derived ability proxy.
    NOTE: this is an assessment-level proxy, not true item-level IRT — see
    experiments/configs/experimental_design.md Study 1 vs Study 2 distinction.

    If exclude_final=True (sensitivity analysis, approved 2026-08-30): drops each
    student's final assessment per module-presentation before aggregating, to test
    whether results survive once the assessment most likely to overlap directly
    with final_result's own computation is removed.
    """
    key = ["id_student", "code_module", "code_presentation"]
    df = assess.copy()

    if exclude_final:
        course_key = ["code_module", "code_presentation"]
        # Prefer excluding assessment_type == 'Exam' where present; otherwise exclude
        # the assessment with the maximum date per course (the last one administered).
        has_exam = df.groupby(course_key)["assessment_type"].transform(
            lambda s: (s == "Exam").any()
        )
        is_exam_row = (df["assessment_type"] == "Exam")
        max_date_per_course = df.groupby(course_key)["date_submitted"].transform("max")
        is_last_by_date = df["date_submitted"] == max_date_per_course
        drop_mask = np.where(has_exam, is_exam_row, is_last_by_date)
        df = df[~drop_mask]

    agg = df.groupby(key).agg(
        mean_score=("score", "mean"),
        weighted_mean_score=("score", lambda s: np.average(s, weights=df.loc[s.index, "weight"].replace(0, 1))),
        n_assessments=("score", "count"),
        score_std=("score", "std"),
    ).reset_index()
    agg["score_std"] = agg["score_std"].fillna(0.0)
    return agg


def build_feature_set_C_behavior(vle: pd.DataFrame) -> pd.DataFrame:
    """Behavior-only: VLE clickstream aggregates (no resource-type breakdown —
    that granularity is reserved for the interest signal in feature set D)."""
    key = ["id_student", "code_module", "code_presentation"]
    agg = vle.groupby(key).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        mean_clicks_per_active_day=("sum_click", "mean"),
    ).reset_index()
    return agg


def build_feature_set_D_interest(vle: pd.DataFrame) -> pd.DataFrame:
    """Implicit interest signal: resource-type click-share distribution
    (what proportion of a student's total clicks go to each resource type),
    per research/contribution.md's owner-approved implicit-only definition.
    """
    key = ["id_student", "code_module", "code_presentation"]
    pivot = vle.pivot_table(
        index=key, columns="activity_type", values="sum_click", aggfunc="sum", fill_value=0
    )
    # normalize to click-share (proportion of total clicks per student-course)
    row_totals = pivot.sum(axis=1)
    share = pivot.div(row_totals.where(row_totals != 0, 1), axis=0)
    share.columns = [f"click_share_{c}" for c in share.columns]
    share = share.reset_index()

    # resource-type diversity (entropy of click distribution) as a second interest proxy
    def entropy(row):
        p = row[row > 0]
        if len(p) == 0:
            return 0.0
        return float(-(p * np.log(p)).sum())

    share_cols = [c for c in share.columns if c.startswith("click_share_")]
    share["resource_diversity_entropy"] = share[share_cols].apply(entropy, axis=1)
    return share


def main():
    ensure_dirs()
    info, vle, assess = load_processed()
    key = ["id_student", "code_module", "code_presentation"]

    target = build_target(info)
    perf = build_feature_set_B_performance(assess, exclude_final=False)
    perf_sens = build_feature_set_B_performance(assess, exclude_final=True)
    behavior = build_feature_set_C_behavior(vle)
    interest = build_feature_set_D_interest(vle)

    # Feature set A (random/majority baseline) needs no features — only the target.
    target.to_parquet(os.path.join(DATA_PROCESSED, "oulad_target.parquet"), index=False)

    perf.to_parquet(os.path.join(DATA_PROCESSED, "oulad_features_B_performance.parquet"), index=False)
    perf_sens.to_parquet(os.path.join(DATA_PROCESSED, "oulad_features_Bsens_performance.parquet"), index=False)
    behavior.to_parquet(os.path.join(DATA_PROCESSED, "oulad_features_C_behavior.parquet"), index=False)
    interest.to_parquet(os.path.join(DATA_PROCESSED, "oulad_features_D_interest.parquet"), index=False)

    # Model D (proposed, fused) = performance + behavior + interest, joined on key
    fused_D = perf.merge(behavior, on=key, how="inner").merge(interest, on=key, how="inner")
    fused_D.to_parquet(os.path.join(DATA_PROCESSED, "oulad_features_D_fused.parquet"), index=False)

    # Model E (ablation, approved 2026-08-30) = performance + behavior, NO interest.
    # Isolates H2: compare D vs E to see if interest itself adds value beyond B+C.
    fused_E = perf.merge(behavior, on=key, how="inner")
    fused_E.to_parquet(os.path.join(DATA_PROCESSED, "oulad_features_E_fused.parquet"), index=False)

    # Sensitivity variants (approved 2026-08-30): same fusions, but performance
    # signal excludes each student's final/exam assessment — tests whether results
    # survive once the assessment most likely to overlap with final_result's own
    # computation is removed (addresses circularity concern).
    fused_Dsens = perf_sens.merge(behavior, on=key, how="inner").merge(interest, on=key, how="inner")
    fused_Dsens.to_parquet(os.path.join(DATA_PROCESSED, "oulad_features_Dsens_fused.parquet"), index=False)

    fused_Esens = perf_sens.merge(behavior, on=key, how="inner")
    fused_Esens.to_parquet(os.path.join(DATA_PROCESSED, "oulad_features_Esens_fused.parquet"), index=False)

    print(f"Target rows: {len(target)}")
    print(f"Feature set B (performance) rows: {len(perf)}")
    print(f"Feature set B_sens (performance, excl. final) rows: {len(perf_sens)}")
    print(f"Feature set C (behavior) rows: {len(behavior)}")
    print(f"Feature set D_interest (raw) rows: {len(interest)}")
    print(f"Feature set D (fused, B+C+interest) rows: {len(fused_D)}")
    print(f"Feature set E (ablation, B+C only) rows: {len(fused_E)}")
    print(f"Feature set D_sens (fused, B_sens+C+interest) rows: {len(fused_Dsens)}")
    print(f"Feature set E_sens (ablation, B_sens+C only) rows: {len(fused_Esens)}")


if __name__ == "__main__":
    main()

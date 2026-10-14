# Pre-registration: WD and EARLY analyses

Written: 2026-10-13T22:13:30+03:00, before any WD/EARLY script was run (no wd_* or early_* output existed at this time).
This file is not to be edited after the runs.

Seed 42 everywhere. Models: LR = train_evaluate_oulad.make_pipeline (median impute + standardize + LogisticRegression);
GBM = HistGradientBoostingClassifier(random_state=42), library defaults. Statistics per comparison a -> b:
McNemar on predictions matched by (id_student, code_module, code_presentation) (exact if discordant < 25);
matched-pairs odds ratio b_only_correct / a_only_correct with Wald 95% CI; student-level (id_student) cluster
bootstrap, 1000 resamples, percentile 95% CI of pooled delta-accuracy and delta-AUC (b - a); Benjamini-Hochberg.

## Pre-specified criterion

"signal survives" = the student-level bootstrap 95% CI of the accuracy difference excludes 0 AND BH-adjusted
McNemar p < 0.05, in the same direction, for BOTH LR and GBM.

Operationalisation: for each model family, direction of the accuracy difference = sign of the bootstrap point
estimate (b - a); direction of McNemar = sign(b_only_correct - a_only_correct). Per family the condition holds
if the CI excludes 0, p_BH < 0.05, and both directions agree (and are non-zero). The criterion is met only if
the condition holds for LR and for GBM with the same direction. It is evaluated literally as stated (direction
not restricted to b > a); the direction is recorded in the comparison files. Summary files report only
criterion_met (true/false) per comparison.

## Notation

base = baseline feature set; +I = base + the 21 interest columns (20 click_share_* + resource_diversity_entropy);
_course = + one-hot of code_module+code_presentation; _ctr = base_course + interest centered within course
presentation (presentation mean over that presentation's students in the analysed sample, features only);
_ctrshuf = as _ctr with the centered block row-permuted across students within the same presentation (rng 42).

## Step 1 (sanity, not a hypothesis test)

Full sample (Withdrawn included), full-course features rebuilt inside the new code: E_course and D_course
predictions (LR and GBM, stratified 5-fold) must equal ci1_stratcv_oof_predictions.csv exactly. If not, stop.

## Step 2 - WD (Withdrawn excluded; Pass/Distinction = 1, Fail = 0), stratified 5-fold

Primary comparisons, full-course features (base = E = B + C):
  P2.1  E_course -> D_course
  P2.2  E_course -> E_course_ctr
  P2.3  E_course_ctrshuf -> E_course_ctr
Same three comparisons for the sens variant (base = Esens = Bsens + C): P2.4-P2.6.
BH family: all 12 rows of Step 2 (6 comparisons x LR, GBM).

## Step 3 - EARLY (t = 28 and t = 56 days), stratified 5-fold

Truncation before aggregation: studentVle date <= t (negative days included); studentAssessment
date_submitted <= t. Risk set: exclude date_unregistration <= t. Target Pass/Distinction = 1, Fail/Withdrawn = 0.
Feature families per cutoff:
  (i)  VLE-only: base = C_early; +I = C_early + interest_early
  (ii) with assessments: base = E_early = B_early + C_early; +I = D_early (scores assumed observable at submission)
Comparisons per cutoff x feature family (same structure as Step 2):
  base_course -> (+I)_course ; base_course -> base_course_ctr ; base_course_ctrshuf -> base_course_ctr
BH family: the 6 rows of each cutoff x feature family (3 comparisons x LR, GBM).

Handling decisions fixed in advance (ASSUMPTION, stated because the request did not specify them):
- Students in the risk set with no clicks by t: click_share_* = 0, entropy = 0, total_clicks = 0,
  active_days = 0, mean_clicks_per_active_day = 0.
- Students in the risk set with no scored assessment submitted by t: kept; n_assessments = 0, score_std = 0,
  mean_score and weighted_mean_score missing (median-imputed inside the LR pipeline; native NaN handling in GBM).
- click_share columns are fixed to the 20 activity types of the full-course feature set (absent types = 0).

## Step 4 - leave-one-presentation-out (22 groups), LR and GBM

Settings: WD (base = E, full-course features); EARLY t = 28 and t = 56, each for VLE-only (base = C_early) and
with-assessments (base = E_early). Models: base, base + interest-centered (_ctr, no course one-hot),
base + interest-centered-shuffled (_ctrshuf). Centering uses only the held-out course's own students,
features only, no labels.
Comparisons: base -> base_ctr ; base_ctrshuf -> base_ctr. BH family: the 4 rows of each setting.
Per-course breakdown of base vs base_ctr with Spearman rho of the difference against pass rate (n = 22).

The criterion above is evaluated for every listed comparison in Steps 2-4.

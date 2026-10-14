# Pre-registration (post hoc, exploratory): final round, 2026-10-14

Written: 2026-10-14T00:35:47+03:00, BEFORE any analysis of this round was run.
Author of this file: Claude (Claude Code, Opus 5.5), on instruction from the owner
(Dr. Aqeel Mohammed Ali Hussein). This file is not edited after it is written; its
SHA-256 is recorded before and after the runs.

## Status of every analysis in this round

All analyses in this round are EXPLORATORY and POST HOC: the OULAD data, including the
EARLY t28/t56 samples used here, were already analysed in earlier rounds
(robustness, course-identity, WD/EARLY, leak/breadth). No result of this round is
confirmatory. All results will be described as hypothesis-generating.

## Fixed protocol (unchanged from the earlier rounds)

- Seed 42 everywhere. Stratified 5-fold (StratifiedKFold, shuffle, random_state 42).
- LR = train_evaluate_oulad.make_pipeline (median impute + standard scaling + LR).
  GBM = HistGradientBoostingClassifier(random_state=42), library defaults.
- EARLY risk set at cutoff t in {28, 56}: exclude date_unregistration <= t (missing =
  not unregistered); target Pass/Distinction = 1, Fail/Withdrawn = 0. Raw tables are
  truncated to <= t before aggregation (studentVle date, studentAssessment
  date_submitted). Students with no clicks by t: shares, entropy and behaviour
  counts = 0 (as pre-registered on 2026-10-13).
- Course one-hot = code_module + code_presentation (22 presentations).
- Course-centering: subtract the presentation mean over the analysed sample
  (features only, no labels; includes test-fold rows), as in breadth_1_control.py.
- Statistics per comparison a -> b: pooled delta-Accuracy and delta-AUC (b - a);
  McNemar on KEY-aligned predictions (id_student, code_module, code_presentation),
  exact if discordant < 25; matched-pairs odds ratio b_only/a_only with Wald 95% CI;
  student-level (id_student) cluster bootstrap, 1000 resamples, default_rng(42),
  95% percentile CI for delta-Accuracy and delta-AUC.

## Step 1: resource-type attribution (EARLY t28 and t56; VLE-only family)

Base features: C_early (total_clicks, active_days, mean_clicks_per_active_day) +
course one-hot ("base_course"). full21_ctr = base_course + the 21 interest columns,
each course-centered.

ASSUMPTION: the owner's "21 activity types" is read as the 21 interest columns of
full21_ctr. OULAD has 20 activity types (20 click_share_<type> columns); the 21st
unit is resource_diversity_entropy. It is analysed as the 21st unit so that every
family has 21 members as specified, and it is labelled unit_kind = "entropy" (not an
activity type) in every output.

1a. Add-one-in: base_course -> base_course + that unit's course-centered column,
    for each of the 21 units.
1b. Leave-one-out: full21_ctr -> full21_ctr without that unit (a = full21_ctr,
    b = without; delta = b - a).
    For 1a and 1b separately, BH is applied across the 21 units within each
    (cutoff x model family): 4 families of 21 rows per analysis.
1c. Stability, in the full21_ctr model, per held-out fold of the stratified 5-fold:
    LR standardized coefficient (coefficient of the fitted pipeline, whose inputs
    are standard-scaled), mean and SD across the 5 folds; permutation importance in
    AUC (sklearn permutation_importance, scoring roc_auc, n_repeats 5,
    random_state 42) on the held-out fold, mean over repeats per fold, then mean and
    SD across the 5 folds. ASSUMPTION: permutation importance is computed for LR
    as well as GBM, because step 1d asks for LR and GBM separately.
1d. Spearman rank correlation over the 21 units, per cutoff and model family,
    between (i) add-one-in delta-AUC, (ii) leave-one-out delta-AUC (signed as
    reported, b - a = without minus with), (iii) permutation importance mean.
    Three pairwise rhos per cutoff x model family; their p-values are reported
    unadjusted and are descriptive, not part of any flag.
1e. Descriptives per unit and cutoff, over the risk set: number of the 22
    presentations in which at least one risk-set student has a click on that type
    by t; percent of risk-set students with any click on that type by t; mean
    (uncentered) share among Pass/Distinction vs Fail/Withdrawn (students with no
    clicks contribute share 0). For entropy: presentations = number with any
    student entropy > 0; percent = students with entropy > 0; means of entropy.

Flag rule (pre-specified, per analysis 1a or 1b, per unit and cutoff): a unit is
"flagged" only if BH p < 0.05 AND the bootstrap 95% CI of delta-Accuracy excludes 0
in the same direction, for BOTH LR and GBM, and the direction is the same for LR and
GBM. Every unit is reported, flagged or not; there is no selection.

## Step 2: no-click sensitivity (t28, t56; VLE-only and with-assessment; LR, GBM)

Comparisons (as in the breadth round): breadth2_ctr -> full21_ctr and
base_course -> full21_ctr (base = C_early for VLE-only, B_early + C_early for
with-assessment; all arms include course one-hot).
(a) Exclude risk-set students with total_clicks = 0 by t; course-centering is
    recomputed within the reduced sample. Report N used and the N and pass rate of
    the excluded group.
(b) Keep everyone; add a binary no-click indicator (total_clicks = 0 by t) to the
    base of every arm.
BH within each setting = cutoff x feature family x variant (2 comparisons x 2
model families = 4 rows). The criterion of prereg_leak_breadth_2026-10-13.md
(same operationalisation) is reported as a descriptive column.

## Step 3: traceable numbers table (final_paper_numbers.csv)

Values are taken only from the key-aligned outputs (robustness_*, ci*_, wd_,
early_, leak_, breadth_, final_), never from oulad_pairwise_comparison.csv or
oulad_pairwise_comparison_corrected.csv. Each value is re-derived independently from
the raw prediction files by a separate script; rederived_ok = true only if
|difference| <= 1e-9 for every re-derived component of that row.

## Step 4: analysis inventory (read-only)

final_analysis_inventory.csv: per output file, the number of rows with a p-value,
the BH family, and whether a pre-registration file with an earlier timestamp
existed for that round.

## Step 5: repository preparation (read-only, nothing pushed)

List of files in results/ and app/src tracked in the GitHub repository that would
change or become misleading if the key-aligned patch were applied; scan of the files
that would be published for raw OULAD/ASSISTments records and original user_id
values. The patch is not applied.

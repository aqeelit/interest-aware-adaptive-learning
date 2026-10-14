# Pre-registration: EARLY leak audit and breadth control

Written: 2026-10-13T22:48:15+03:00, before any leak_/breadth_ script was run (no leak_* or breadth_* output existed).
This file is not to be edited after the runs. Its SHA-256 is recorded before and after the runs.

Seed 42 everywhere. Models: LR = train_evaluate_oulad.make_pipeline (median impute + standardize +
LogisticRegression); GBM = HistGradientBoostingClassifier(random_state=42), library defaults.
Settings: EARLY t = 28 and t = 56 days (cutoff-first truncation, risk set excludes date_unregistration <= t,
target Pass/Distinction = 1, Fail/Withdrawn = 0), exactly as in the previous EARLY round.
Feature families: VLE-only (base = C_early) and with-assessments (base = E_early = B_early + C_early).

## Leak audit (Step 1)

Leak audit PASS = zero provenance violations AND the independent re-implementation matches the pipeline
features exactly (max absolute difference <= 1e-9, same rows, same order). Otherwise FAIL.

- Provenance: for every student and every feature group (assessment B, behaviour C, interest 21 columns,
  breadth columns), the maximum timestamp of the raw records actually passed into the value-producing
  feature builders (studentVle.date, studentAssessment.date_submitted) must be <= t. Violations are counted
  per group, cutoff and family.
- Independent re-implementation: plain pandas from the raw CSVs, filtering to <= t before aggregation, with
  no import of the pipeline's feature code. Compared column by column with the pipeline's feature tables
  (missing values must coincide; a missing value against a non-missing one counts as a mismatch).
- The audit verdict is evaluated per feature family (VLE-only, with-assessments). A family FAILS if any check
  fails at either cutoff for any column that family uses. The breadth columns
  (n_activity_types_used, n_unique_resources) are included in both families.
- Step 1a sanity check: the rebuilt EARLY predictions (existing code path) must match the earlier early_*
  prediction outputs (same rows, folds and predicted classes; maximum probability difference reported). If
  they do not, everything stops.

## Stop rule

If the leak audit FAILS for a feature family, NO further analysis is run for that family, and the failure is
reported first.

## Breadth control (Step 2), per cutoff x feature family

Arms (all with course one-hot + base features):
  base_course; + breadth2 (resource_diversity_entropy, n_activity_types_used; plain values);
  + breadth2_centered; + breadth2_centered_shuffled; + full21_centered; + full21_centered_shuffled;
  SECONDARY: + breadth3_centered (breadth2 + n_unique_resources, centered).
Centering: within course presentation, mean over that presentation's risk-set students (features only).
Shuffling: whole-row permutation of the centered block within presentation (numpy default_rng(42), groups in
sorted order).
Comparisons a -> b (matched on id_student, code_module, code_presentation):
  C1 base_course -> breadth2_centered
  C2 breadth2_centered -> full21_centered            (PRIMARY)
  C3 breadth2_centered_shuffled -> breadth2_centered
  C4 full21_centered_shuffled -> full21_centered
  C5 base_course -> full21_centered
  C6 breadth2_centered -> breadth3_centered           (SECONDARY)
Statistics: McNemar (exact if discordant < 25); odds ratio b_only_correct / a_only_correct with Wald 95% CI;
student-level (id_student) cluster bootstrap, 1000 resamples, percentile 95% CI of pooled delta-Accuracy
and delta-AUC (b - a). BH across all 12 rows (6 comparisons x LR, GBM) of each cutoff x feature family,
the secondary comparison included.

LOPO (22 held-out presentations, no course one-hot; centering uses only the held-out course's own students,
features only, no labels): arms base, base + breadth2_centered, base + full21_centered. Comparisons:
base -> breadth2_centered; breadth2_centered -> full21_centered (primary, LOPO); base -> full21_centered.
(The shuffled and breadth3 comparisons have no LOPO arms and are not run there.) BH across the 6 rows of each
cutoff x feature family. Per-course tables for base -> breadth2_centered and
breadth2_centered -> full21_centered, with Spearman rho of the difference against pass rate (n = 22).

## Criterion

"Type-specific signal beyond breadth" (primary comparison C2) = the student-level bootstrap 95% CI of the
accuracy difference excludes 0 AND BH-adjusted McNemar p < 0.05, in the same direction, for BOTH LR and GBM.
The same criterion definition is used for every other comparison in the summary files.
Operationalisation (as in prereg_wd_early_2026-10-13.md): per model family, direction of accuracy =
sign(bootstrap point estimate b - a), direction of McNemar = sign(b_only_correct - a_only_correct); the family
condition holds if the CI excludes 0, p_BH < 0.05 and both directions agree (non-zero); the criterion is met
only if it holds for LR and GBM in the same direction. Summary files report criterion_met true/false only.

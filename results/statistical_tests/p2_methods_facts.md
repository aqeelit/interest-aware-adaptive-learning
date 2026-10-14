# Paper 2: methods facts (each with its source)

All analyses EXPLORATORY / POST HOC (p2_prereg_2026-10-14.md). Paths: app/src = code, results/statistical_tests = outputs.

## 1. Data cleaning (app/src/data_cleaning_oulad.py)
- studentInfo: de-duplicated on (id_student, code_module, code_presentation); rows with missing final_result dropped.
- studentVle: joined to vle.csv on (id_site, code_module, code_presentation) to get activity_type; rows without
  activity_type dropped; rows with sum_click < 0 dropped.
- studentAssessment: joined to assessments.csv; rows with missing score dropped; scores outside [0, 100] dropped.
- Target (feature_engineering_oulad.build_target): Pass or Distinction = 1; Fail or Withdrawn = 0.

## 2. Feature definitions
Source: app/src/feature_engineering_oulad.py (B, C, I); app/src/leakbreadth_common.py (breadth).
- B (performance), per student-presentation over scored assessments: mean_score (mean of score);
  weighted_mean_score (np.average of score with weights = assessment weight, weight 0 replaced by 1);
  n_assessments (count of scores); score_std (pandas std, ddof 1; filled with 0 when undefined).
- C (behaviour), per student-presentation over studentVle records: total_clicks (sum of sum_click);
  active_days (number of distinct date values); mean_clicks_per_active_day = groupby(...).agg(("sum_click",
  "mean")), i.e. the MEAN OF sum_click OVER studentVle RECORDS (one record per student x site x day), NOT total
  clicks divided by active days. The name does not match the computation.
- I (interest; 21 columns): 20 click_share_<activity_type> = clicks on that activity type / total clicks of the
  student-presentation (0 if the total is 0), via a pivot of summed sum_click by activity_type; plus
  resource_diversity_entropy = -sum p ln p over shares p > 0 (natural log; 0 if no clicks). The entropy column
  has unit_kind = entropy (it is not an activity type).
- breadth2 = resource_diversity_entropy + n_activity_types_used (number of activity types with summed clicks > 0).
- breadth3 = breadth2 + n_unique_resources (number of id_site with summed clicks > 0).
- EARLY versions (app/src/wdearly_common.build_sample): raw tables truncated FIRST (studentVle date <= t;
  studentAssessment date_submitted <= t), then aggregated with the same builders. Risk set: exclude
  date_unregistration <= t (missing = not unregistered). No clicks by t: shares, entropy and behaviour counts = 0.
  No scored submission by t: n_assessments = 0, score_std = 0, mean scores missing (median-imputed by LR, native
  missing-value handling by GBM). ASSUMPTION recorded there: scores are treated as observable at date_submitted.
- Model sets (FULL): E = B + C (7 columns); D = E + I (28 columns); course one-hot = code_module + code_presentation
  (22 columns). EARLY base: VLE-only = C_early; with-assessments = B_early + C_early. Column counts confirmed in
  ml_1b_metrics_summary.csv (n_features: FULL base 7, base_course 29; EARLY base 3, base_course 25).

## 3. Rows and the 6,823 dropped rows (FULL)
Source: results/statistical_tests/p2_5_dropped_rows_breakdown.csv (script app/src/p2_5_facts.py).
- 32,593 target rows; 25,770 in the D/E row set (inner join with B, C and the interest table); 6,823 dropped.
- Dropped: 3,458 have VLE records but no scored assessment (no B); 50 have B but no VLE record; 3,315 have
  neither. C and the interest table contain the same rows.
- By final_result, the dropped rows are 5,503 Withdrawn, 1,315 Fail, 5 Pass (no Distinction row listed).
- EARLY: N = 27,538 (t28) and 26,522 (t56); dropped = 5,055 and 6,071, all because the student unregistered by
  day t; 0 rows dropped by the feature build (p2_T1_data_settings.csv; p2_1_zbase_metrics_summary.csv).

## 4. Models, folds, seeds (app/src/train_evaluate_oulad.py, app/src/robustness_common.py)
- LR: Pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000,
  random_state=42)); other parameters at scikit-learn defaults.
- GBM: HistGradientBoostingClassifier(random_state=42), all other parameters at library defaults; no tuning.
- Stratified 5-fold: StratifiedKFold(n_splits=5, shuffle=True, random_state=42), split on y only.
- Leave-one-presentation-out: LeaveOneGroupOut over the 22 code_module_code_presentation groups.
- Leave-one-module-out (app/src/ml_2_lomo.py): 7 folds; for each held-out module, every training row of a
  student (id_student) who also appears in the held-out module is removed (ml_2_lomo_fold_rows_removed.csv).
- Within-module 5-fold (app/src/ml_3_module_replication.py): stratified 5-fold seed 42 on each module's rows.
- Global seed: utils.RANDOM_SEED = 42.

## 5. Pairwise tests (app/src/robustness_common.py: mcnemar_pair, align_two, add_bh)
- Key alignment: out-of-fold predictions of models a and b are inner-joined on (id_student, code_module,
  code_presentation); y_true must agree. Models with different row sets are compared on their common rows.
- McNemar: statsmodels.stats.contingency_tables.mcnemar on [[both correct, a only], [b only, neither]],
  exact=True if a_only + b_only < 25, otherwise chi-square with statsmodels' default continuity correction.
- Odds ratio (statistics_tests_oulad.odds_ratio_ci): b_only / a_only, Wald 95% CI on the log scale; +0.5 to both
  discordant cells only if one of them is 0.
- BH: statsmodels multipletests(method="fdr_bh", alpha 0.05) over the family stated in each file header.
- Independent re-check: app/src/p2_rederive.py re-implements McNemar and BH without statsmodels and matches all
  171 rows of p2_numbers.csv within 1e-9 (p2_numbers_rederivation.csv).

## 6. Bootstrap (robustness_common.paired_bootstrap_diff, cluster_weights; ml_common.boot_compare)
- Student-level cluster bootstrap over id_student, 1000 resamples, numpy default_rng(42). Each resample draws
  n_students indices uniformly with replacement; each row gets weight = number of times its student was drawn.
- Delta = weighted pooled metric of b minus a on the same resample; 95% percentile CI (2.5th and 97.5th
  percentiles); bootstrap SE = SD (ddof 1) of the 1000 delta-accuracy values (ml_common.boot_compare).
- AUC under weights: sklearn roc_auc_score with sample_weight.

## 7. Random-effects pooling as coded (app/src/ml_3_module_replication.py: pool)
- Inputs: the 7 module delta-accuracy values y_i and their bootstrap SEs s_i.
- w_i = 1 / s_i^2; fixed estimate mu_F = sum(w y) / sum(w); Q = sum w (y - mu_F)^2; df = k - 1 = 6.
- DerSimonian-Laird: tau^2 = max(0, (Q - df) / (sum w - sum w^2 / sum w)); w*_i = 1 / (s_i^2 + tau^2);
  mu = sum(w* y) / sum(w*); SE = sqrt(1 / sum w*); 95% CI = mu +/- z_0.975 SE.
- I^2 = max(0, (Q - df) / Q) x 100 (0 if Q = 0).
- Hartung-Knapp: q = sum w* (y - mu)^2 / df; SE_HK = sqrt(q / sum w*); CI = mu +/- t_{0.975, df} SE_HK
  (critical value 2.446912 in ml_3_random_effects.csv).

## 8. Normalisations (features only, no labels)
- Course centering (ml_common.center_and_mean): subtract the presentation mean over that presentation's students
  in the analysed sample; columns constant within a presentation are set to 0. Course mean = that mean.
- Course-normalised base (p2_common.zscore_within): z = (x - presentation mean) / presentation SD (ddof 0);
  SD 0 -> 0; missing stays missing.

## 9. Software (results/statistical_tests/p2_5_software_versions.csv)
Python 3.11.17; numpy 1.26.4; pandas 2.2.2; scipy 1.13.1; scikit-learn 1.5.1; statsmodels 0.14.2;
matplotlib 3.9.0; pyarrow 16.1.0. Docker image built from app/Dockerfile (python:3.11-slim) and
app/requirements.txt. The image could not be rebuilt for this round (registry timeout), so the current
app/src was mounted read-only over /workspace/src in the existing image (same dependencies).
